from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, cast
from uuid import UUID

from ai_service.application.deterministic_answer_generator import (
    DeterministicAnswerGenerator,
)
from ai_service.application.ports.answer_generator import AnswerGenerator
from ai_service.config.settings import Settings, get_settings
from ai_service.context.manager import ConversationManager
from ai_service.graphs.comparison_graph import (
    ComparisonInput,
    ComparisonOutput,
    ComparisonState,
    comparison_graph,
)
from ai_service.graphs.shopping_graph import (
    ShoppingInput,
    ShoppingIntent,
    ShoppingOutput,
    ShoppingState,
    shopping_graph,
)
from ai_service.schemas.conversation import (
    ChatData,
    ChatRequest,
    ChatStreamEvent,
    ChatStreamEventType,
    CompareData,
    CompareRequest,
    ConsultData,
    ConsultRequest,
    ConversationMessage,
    EvaluateData,
    EvaluateRequest,
    ProductCard,
    ProductComparison,
    SearchData,
    SearchRequest,
)
from ai_service.schemas.response import ApiResponse, ErrorDetail, ResponseMessage
from ai_service.services.backend_client import BackendClient, BackendUnavailableError
from ai_service.services.retriever import CatalogRetriever
from ai_service.services.semantic_retriever import build_catalog_retriever


@dataclass(frozen=True)
class _PreparedChat:
    conversation_id: UUID
    intent: str
    query: str
    products: list[ProductCard]
    errors: list[ErrorDetail]
    fallback: str
    prompt: str


class AssistantService:
    def __init__(
        self,
        backend_client: BackendClient | None = None,
        context_manager: ConversationManager | None = None,
        answer_generator: AnswerGenerator | None = None,
        retriever: CatalogRetriever | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.backend_client = backend_client or BackendClient(self.settings)
        self.context_manager = context_manager or ConversationManager()
        self.answer_generator = answer_generator or DeterministicAnswerGenerator()
        self.retriever = retriever or build_catalog_retriever(self.backend_client, self.settings)

    async def chat(self, request: ChatRequest) -> ApiResponse[ChatData]:
        prepared = await self._prepare_chat(request)
        answer = await self.answer_generator.generate(prepared.prompt, prepared.fallback)
        self.context_manager.append(
            prepared.conversation_id,
            ConversationMessage(role="assistant", content=answer),
        )
        return ApiResponse(
            data=ChatData(
                conversation_id=prepared.conversation_id,
                intent=prepared.intent,
                answer=answer,
                products=prepared.products,
            ),
            message=ResponseMessage.AI_CHAT_COMPLETED,
            errors=prepared.errors,
        )

    async def stream_chat(
        self,
        request: ChatRequest,
    ) -> AsyncIterator[ApiResponse[ChatStreamEvent]]:
        """Stream a planned, catalog-grounded chat answer as envelope events."""
        prepared = await self._prepare_chat(request)
        yield ApiResponse(
            data=ChatStreamEvent(
                event=ChatStreamEventType.START,
                conversation_id=prepared.conversation_id,
            ),
            message=ResponseMessage.AI_CHAT_STREAM_STARTED,
        )

        chunks: list[str] = []
        try:
            stream_method = getattr(self.answer_generator, "stream", None)
            if callable(stream_method):
                async for chunk in stream_method(prepared.prompt, prepared.fallback):
                    if not chunk:
                        continue
                    chunks.append(chunk)
                    yield ApiResponse(
                        data=ChatStreamEvent(
                            event=ChatStreamEventType.DELTA,
                            conversation_id=prepared.conversation_id,
                            delta=chunk,
                        ),
                        message=ResponseMessage.AI_CHAT_STREAM_DELTA,
                    )
            else:
                # Existing test/deterministic adapters only need the complete
                # application port.  They remain compatible with the stream
                # endpoint by emitting one complete answer delta.
                answer = await self.answer_generator.generate(
                    prepared.prompt,
                    prepared.fallback,
                )
                if answer:
                    chunks.append(answer)
                    yield ApiResponse(
                        data=ChatStreamEvent(
                            event=ChatStreamEventType.DELTA,
                            conversation_id=prepared.conversation_id,
                            delta=answer,
                        ),
                        message=ResponseMessage.AI_CHAT_STREAM_DELTA,
                    )
        except asyncio.CancelledError:
            # Do not append a partial assistant message when the browser
            # cancels a request or navigates away.
            raise
        except Exception:  # noqa: BLE001 - provider implementations vary
            # An unexpected adapter error is terminal for this SSE stream. The
            # provider adapter normally converts its own failures to fallback
            # text, so this branch is reserved for broken custom adapters.
            answer = "".join(chunks).strip() or prepared.fallback
            self.context_manager.append(
                prepared.conversation_id,
                ConversationMessage(role="assistant", content=answer),
            )
            yield ApiResponse(
                data=ChatStreamEvent(
                    event=ChatStreamEventType.ERROR,
                    conversation_id=prepared.conversation_id,
                ),
                message=ResponseMessage.AI_CHAT_STREAM_FAILED,
                errors=[
                    ErrorDetail(
                        code=ResponseMessage.AI_CHAT_STREAM_FAILED.value,
                        message="Streaming answer generation failed",
                    )
                ],
            )
            return

        answer = "".join(chunks).strip() or prepared.fallback
        self.context_manager.append(
            prepared.conversation_id,
            ConversationMessage(role="assistant", content=answer),
        )
        yield ApiResponse(
            data=ChatStreamEvent(
                event=ChatStreamEventType.COMPLETED,
                conversation_id=prepared.conversation_id,
                result=ChatData(
                    conversation_id=prepared.conversation_id,
                    intent=prepared.intent,
                    answer=answer,
                    products=prepared.products,
                ),
            ),
            message=ResponseMessage.AI_CHAT_STREAM_COMPLETED,
            errors=prepared.errors,
        )

    async def _prepare_chat(self, request: ChatRequest) -> _PreparedChat:
        conversation_id = self.context_manager.get_or_create(request.conversation_id)
        self.context_manager.append(
            conversation_id,
            ConversationMessage(role="user", content=request.message),
        )
        intent = self.classify_intent(request.message)
        plan = await self._shopping_plan(request.message, intent)
        products: list[ProductCard] = []
        errors: list[ErrorDetail] = []
        if plan.intent in {"SEARCH", "CONSULT"}:
            try:
                products = self._cards(await self.retriever.search(plan.query, 5))
            except BackendUnavailableError:
                errors.append(
                    ErrorDetail(
                        code=ResponseMessage.AI_BACKEND_UNAVAILABLE.value,
                        message="Catalog service is unavailable",
                    )
                )
        fallback = self._chat_answer(plan.intent, plan.query, products)
        return _PreparedChat(
            conversation_id=conversation_id,
            intent=plan.intent,
            query=plan.query,
            products=products,
            errors=errors,
            fallback=fallback,
            prompt=self._chat_prompt(plan.query, plan.intent, products, conversation_id),
        )

    async def search(self, request: SearchRequest) -> ApiResponse[SearchData]:
        try:
            plan = await self._shopping_plan(request.query, "SEARCH")
            products = self._cards(await self.retriever.search(plan.query, request.limit))
            return ApiResponse(
                data=SearchData(query=plan.query, products=products),
                message=(
                    ResponseMessage.AI_SEARCH_COMPLETED
                    if products
                    else ResponseMessage.AI_SEARCH_NO_RESULTS
                ),
            )
        except BackendUnavailableError:
            return ApiResponse(
                data=SearchData(query=request.query, products=[]),
                message=ResponseMessage.AI_BACKEND_UNAVAILABLE,
                errors=[
                    ErrorDetail(
                        code=ResponseMessage.AI_BACKEND_UNAVAILABLE.value,
                        message="Catalog service is unavailable",
                    )
                ],
            )

    async def consult(self, request: ConsultRequest) -> ApiResponse[ConsultData]:
        try:
            plan = await self._shopping_plan(request.query, "CONSULT")
            products = self._cards(await self.retriever.search(plan.query, request.limit))
            fallback = self._consult_answer(plan.query, products)
            answer = await self.answer_generator.generate(
                self._consult_prompt(plan.query, products),
                fallback,
            )
            return ApiResponse(
                data=ConsultData(query=plan.query, answer=answer, products=products),
                message=ResponseMessage.AI_CONSULT_COMPLETED,
            )
        except BackendUnavailableError:
            return ApiResponse(
                data=ConsultData(query=request.query, answer="Chưa thể truy cập dữ liệu sản phẩm lúc này.", products=[]),
                message=ResponseMessage.AI_BACKEND_UNAVAILABLE,
                errors=[
                    ErrorDetail(
                        code=ResponseMessage.AI_BACKEND_UNAVAILABLE.value,
                        message="Catalog service is unavailable",
                    )
                ],
            )

    async def compare(self, request: CompareRequest) -> ApiResponse[CompareData | None]:
        graph_result = cast(
            ComparisonOutput,
            await comparison_graph.run(
                state=ComparisonState(),
                inputs=ComparisonInput(
                    product_ids=request.product_ids,
                    question=request.question,
                )
            ),
        )
        try:
            raw_products = await asyncio.gather(
                *(self.backend_client.get_product(product_id) for product_id in graph_result.product_ids)
            )
        except BackendUnavailableError:
            return ApiResponse(
                data=None,
                message=ResponseMessage.AI_BACKEND_UNAVAILABLE,
                errors=[
                    ErrorDetail(
                        code=ResponseMessage.AI_BACKEND_UNAVAILABLE.value,
                        message="Catalog service is unavailable",
                    )
                ],
            )
        products = [self._comparison(product) for product in raw_products if product is not None]
        missing = len(products) != len(graph_result.product_ids)
        errors = (
            [
                ErrorDetail(
                    code=ResponseMessage.AI_PRODUCT_NOT_FOUND.value,
                    message="One or more products were not found",
                )
            ]
            if missing
            else []
        )
        fallback = self._compare_answer(products, request.question)
        answer = await self.answer_generator.generate(
            self._compare_prompt(products, request.question),
            fallback,
        )
        return ApiResponse(
            data=CompareData(products=products, answer=answer),
            message=(
                ResponseMessage.AI_COMPARE_PARTIAL
                if missing
                else ResponseMessage.AI_COMPARE_COMPLETED
            ),
            errors=errors,
        )

    async def evaluate(self, request: EvaluateRequest) -> ApiResponse[EvaluateData | None]:
        try:
            raw_product = await self.backend_client.get_product(request.product_id)
        except BackendUnavailableError:
            return ApiResponse(
                data=None,
                message=ResponseMessage.AI_BACKEND_UNAVAILABLE,
                errors=[
                    ErrorDetail(
                        code=ResponseMessage.AI_BACKEND_UNAVAILABLE.value,
                        message="Catalog service is unavailable",
                    )
                ],
            )
        if raw_product is None:
            return ApiResponse(
                data=None,
                message=ResponseMessage.AI_PRODUCT_NOT_FOUND,
                errors=[
                    ErrorDetail(
                        code=ResponseMessage.AI_PRODUCT_NOT_FOUND.value,
                        message="Product was not found",
                    )
                ],
            )
        product = self._card(raw_product)
        fallback = self._evaluate_answer(raw_product, request.question)
        answer = await self.answer_generator.generate(
            self._evaluate_prompt(raw_product, request.question),
            fallback,
        )
        return ApiResponse(
            data=EvaluateData(product=product, answer=answer),
            message=ResponseMessage.AI_EVALUATION_COMPLETED,
        )

    @staticmethod
    def classify_intent(message: str) -> str:
        lowered = message.lower()
        if "so sánh" in lowered or "compare" in lowered:
            return "COMPARE"
        if any(token in lowered for token in ("tư vấn", "nên mua", "phù hợp", "recommend")):
            return "CONSULT"
        explicit_evaluation = any(
            token in lowered for token in ("đánh giá", "giải thích", "evaluate")
        )
        # "review" is also a valid product/name token (for example, a product
        # named "Review Product"). Only treat it as an evaluation request when
        # the message is not clearly asking the catalog to find/show something.
        review_request = "review" in lowered and not any(
            token in lowered
            for token in ("tìm", "search", "find", "gợi ý", "xem", "show")
        )
        if explicit_evaluation or review_request:
            return "EVALUATE"
        return "SEARCH"

    @staticmethod
    async def _shopping_plan(query: str, intent: str) -> ShoppingOutput:
        """Run the deterministic shopping graph before catalog retrieval.

        The graph is intentionally limited to request planning. Retrieval and
        answer generation remain replaceable ports, so the configured vector
        store or model provider can change without changing the HTTP contract.
        """
        plan_intent = cast(
            ShoppingIntent,
            intent if intent in {"SEARCH", "CONSULT", "COMPARE", "EVALUATE"} else "SEARCH",
        )
        return cast(
            ShoppingOutput,
            await shopping_graph.run(
                state=ShoppingState(),
                inputs=ShoppingInput(query=query, intent=plan_intent),
            ),
        )

    @staticmethod
    def _cards(rows: list[dict[str, Any]]) -> list[ProductCard]:
        return [AssistantService._card(row) for row in rows if row.get("name")]

    @staticmethod
    def _card(row: dict[str, Any]) -> ProductCard:
        raw_id = row.get("id")
        try:
            product_id = UUID(str(raw_id)) if raw_id else None
        except ValueError:
            product_id = None
        raw_specifications = row.get("specifications")
        specifications = (
            raw_specifications if isinstance(raw_specifications, dict) else {}
        )
        list_price = AssistantService._list_price(row)
        return ProductCard(
            id=product_id,
            name=str(row.get("name", "Sản phẩm")),
            seo_name=row.get("seoName", row.get("seo_name")),
            list_price=list_price,
            image_url=row.get("imageUrl", row.get("image_url")),
            status=row.get("status"),
            specifications=specifications,
            description=row.get("description"),
        )

    @staticmethod
    def _list_price(row: dict[str, Any]) -> int | None:
        """Read a canonical price from either a summary or detail payload.

        Product summaries expose ``minPrice`` while product details expose
        prices on their variants.  The AI contract keeps one ``list_price``
        field, so detail mapping uses the lowest available variant list price
        without trusting any sale-price field.
        """
        for key in ("listPrice", "list_price", "minPrice", "min_price"):
            value = row.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return int(value)

        variants = row.get("variants")
        if not isinstance(variants, list):
            return None
        variant_prices = [
            int(value)
            for variant in variants
            if isinstance(variant, dict)
            for value in (variant.get("listPrice", variant.get("list_price")),)
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        ]
        return min(variant_prices) if variant_prices else None

    @staticmethod
    def _comparison(row: dict[str, Any]) -> ProductComparison:
        card = AssistantService._card(row)
        return ProductComparison(
            product_id=card.id,
            name=card.name,
            list_price=card.list_price,
            image_url=card.image_url,
            specifications=card.specifications,
            description=card.description,
            status=card.status,
        )

    @staticmethod
    def _chat_answer(intent: str, message: str, products: list[ProductCard]) -> str:
        if not products:
            return f"Tôi đã nhận yêu cầu '{message}'. Hiện chưa tìm thấy sản phẩm phù hợp."
        names = ", ".join(product.name for product in products[:3])
        return f"Tôi tìm thấy các sản phẩm có thể phù hợp: {names}."

    @staticmethod
    def _consult_answer(query: str, products: list[ProductCard]) -> str:
        if not products:
            return f"Chưa có sản phẩm đủ dữ liệu cho nhu cầu: {query}."
        return f"Dựa trên nhu cầu '{query}', bạn có thể bắt đầu xem {products[0].name}."

    @staticmethod
    def _compare_answer(products: list[ProductComparison], question: str | None) -> str:
        if not products:
            return "Không có đủ dữ liệu để so sánh."
        names = " và ".join(product.name for product in products)
        suffix = f" Câu hỏi của bạn: {question}" if question else ""
        return f"So sánh cơ bản giữa {names}.{suffix}"

    @staticmethod
    def _evaluate_answer(row: dict[str, Any], question: str | None) -> str:
        description = row.get("description") or "chưa có mô tả chi tiết"
        suffix = f" Câu hỏi: {question}" if question else ""
        return f"{row.get('name', 'Sản phẩm')} hiện có mô tả: {description}.{suffix}"

    def _chat_prompt(
        self,
        message: str,
        intent: str,
        products: list[ProductCard],
        conversation_id: UUID,
    ) -> str:
        history = self.context_manager.history(conversation_id)
        history_text = "\n".join(f"{item.role}: {item.content}" for item in history[-6:])
        return (
            f"Intent: {intent}\nUser message: {message}\n"
            f"Conversation context:\n{history_text}\n"
            f"Catalog context:\n{self._json_context(products)}\n"
            "Respond concisely and mention only products present in the catalog context."
        )

    def _consult_prompt(self, query: str, products: list[ProductCard]) -> str:
        return (
            f"User need: {query}\nCatalog context:\n{self._json_context(products)}\n"
            "Explain which available products best match the need and why, without inventing specifications."
        )

    def _compare_prompt(
        self,
        products: list[ProductComparison],
        question: str | None,
    ) -> str:
        return (
            f"Comparison question: {question or 'Compare these products.'}\n"
            f"Catalog context:\n{self._json_context(products)}\n"
            "Compare only the fields present in the context and call out missing information."
        )

    def _evaluate_prompt(self, row: dict[str, Any], question: str | None) -> str:
        return (
            f"Product question: {question or 'Explain this product.'}\n"
            f"Catalog context:\n{self._json_context([row])}\n"
            "Give a balanced explanation grounded only in the supplied product data."
        )

    @staticmethod
    def _json_context(rows: list[Any]) -> str:
        serialized = [
            row.model_dump(mode="json") if hasattr(row, "model_dump") else row
            for row in rows
        ]
        return json.dumps(serialized, ensure_ascii=False, separators=(",", ":"))
