from uuid import uuid4

import httpx
import pytest

from ai_service.application.errors import BackendUnavailableError
from ai_service.capabilities.assistant.react.contracts import SearchCatalogArgs
from ai_service.infrastructure.commerce.product_catalog import (
    BackendProductCatalog,
    rrf_fuse_by_product_id,
)


class Budget:
    def __init__(self):
        self.calls = []

    async def invoke(self, kind, key, digest, dispatch):
        self.calls.append(kind)
        return await dispatch()


@pytest.mark.asyncio
async def test_task_lookup_exhausts_cursor_pages_instead_of_assuming_first_unique():
    ids = [uuid4(), uuid4()]
    budget = Budget()
    calls = []

    def respond(request):
        calls.append(dict(request.url.params))
        index = 1 if request.url.params.get("cursor") else 0
        return httpx.Response(
            200,
            json={
                "data": {
                    "items": [
                        {
                            "id": str(ids[index]),
                            "name": f"Vendor {index} RTX 4060",
                            "status": "ACTIVE",
                        }
                    ],
                    "hasNext": index == 0,
                    "nextCursor": "second" if index == 0 else None,
                },
                "message": "SUCCESS",
                "errors": [],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        result = await BackendProductCatalog(
            client, "https://catalog.test", budget
        ).resolve("RTX 4060")
    assert [p.product_id for p in result] == ids
    assert len(budget.calls) == 2
    assert calls[1]["cursor"] == "second"


@pytest.mark.asyncio
async def test_task_lookup_rejects_missing_cursor_instead_of_returning_partial_matches():
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={
                    "data": {"items": [], "hasNext": True, "nextCursor": None},
                    "message": "SUCCESS",
                    "errors": [],
                },
            )
        )
    ) as client:
        with pytest.raises(BackendUnavailableError):
            await BackendProductCatalog(
                client, "https://catalog.test", Budget()
            ).resolve("RTX 4060")


@pytest.mark.asyncio
async def test_microservice_summary_range_is_not_reported_as_exact_price():
    identity = uuid4()
    budget = Budget()

    def respond(request):
        assert request.url.params["keyword"] == "GPU"
        return httpx.Response(
            200,
            json={
                "data": {
                    "items": [
                        {
                            "id": str(identity),
                            "name": "GPU",
                            "status": "ACTIVE",
                            "minPrice": 8_000_000,
                            "maxPrice": 9_000_000,
                        },
                        {"id": str(uuid4()), "name": "Hidden", "status": "INACTIVE"},
                    ],
                    "hasNext": False,
                },
                "message": "SUCCESS",
                "errors": [],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        result = await BackendProductCatalog(
            client, "https://catalog.test/api/v1", budget
        ).search("GPU", 5)
    assert len(result) == 1 and result[0].product_id == identity
    assert result[0].price_vnd is None
    assert budget.calls == ["TOOL_IO"]


@pytest.mark.asyncio
@pytest.mark.parametrize("stale_brand", [False, True])
async def test_hybrid_counts_each_http_attempt_and_revalidates_brand(stale_brand):
    identity = uuid4()
    budget, requests = Budget(), []

    def respond(request):
        requests.append(request)
        if request.url.host == "search.test":
            data = {
                "content": [
                    {"id": str(identity), "name": "Headset", "brandName": "Razer"}
                ]
            }
        elif request.url.path == "/products":
            data = {"items": []}
        else:
            data = {
                "id": str(identity),
                "name": "Headset",
                "status": "ACTIVE",
                "brandName": "Other" if stale_brand else "Razer",
                "categoryName": "Headsets",
                "variants": [
                    {
                        "id": str(uuid4()),
                        "productId": str(identity),
                        "price": 100,
                        "quantity": 1,
                        "status": "ACTIVE",
                    }
                ],
            }
        return httpx.Response(
            200, json={"data": data, "message": "SUCCESS", "errors": []}
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        results = await BackendProductCatalog(
            client, "https://catalog.test", budget, "https://search.test"
        ).search(
            "Headset",
            5,
            semantic_query="Headset",
            brand="Razer",
        )
    assert any("semanticQuery" in request.url.params for request in requests)
    assert len(budget.calls) == len(requests)
    assert bool(results) is not stale_brand


@pytest.mark.asyncio
@pytest.mark.parametrize("search_enabled", [True, False])
async def test_semantic_empty_index_fallback_keeps_search_intent(search_enabled):
    queries = []

    def respond(request):
        if request.url.host == "search.test":
            data = {"content": []}
        else:
            queries.append(request.url.params.get("keyword"))
            data = {"items": []}
        return httpx.Response(
            200, json={"data": data, "message": "SUCCESS", "errors": []}
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        await BackendProductCatalog(
            client,
            "https://catalog.test",
            Budget(),
            "https://search.test" if search_enabled else None,
        ).search(
            "",
            5,
            semantic_query="wireless headset",
        )
    assert queries == ["wireless headset"]


@pytest.mark.asyncio
async def test_filter_only_search_uses_search_service_not_an_unfiltered_first_page():
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "data": {"content": []}
                if request.url.host == "search.test"
                else {"items": []},
                "message": "SUCCESS",
                "errors": [],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        await BackendProductCatalog(
            client, "https://catalog.test", Budget(), "https://search.test"
        ).search("", 5, brand="Sony", max_price=2_000_000, in_stock_only=True)
    first = requests[0]
    assert first.url.host == "search.test"
    assert first.url.params["brand"] == "Sony"
    assert first.url.params["maxPrice"] == "2000000"
    assert first.url.params["inStockOnly"] == "true"


@pytest.mark.asyncio
async def test_catalog_fallback_matches_price_and_stock_on_one_active_variant():
    identity = uuid4()

    def respond(request):
        if request.url.path == "/products":
            data = {
                "items": [
                    {
                        "id": str(identity),
                        "name": "Headset",
                        "status": "ACTIVE",
                        "minPrice": 100,
                        "maxPrice": 1000,
                        "totalQuantity": 1,
                    }
                ]
            }
        else:
            data = {
                "id": str(identity),
                "name": "Headset",
                "status": "ACTIVE",
                "variants": [
                    {
                        "id": str(uuid4()),
                        "productId": str(identity),
                        "status": "ACTIVE",
                        "price": 100,
                        "quantity": 0,
                    },
                    {
                        "id": str(uuid4()),
                        "productId": str(identity),
                        "status": "ACTIVE",
                        "price": 1000,
                        "quantity": 1,
                    },
                ],
            }
        return httpx.Response(
            200, json={"data": data, "message": "SUCCESS", "errors": []}
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        result = await BackendProductCatalog(
            client, "https://catalog.test", Budget()
        ).search("headset", 5, max_price=200, in_stock_only=True)
    assert result == []


@pytest.mark.asyncio
async def test_detail_uses_active_variant_price_and_rechecks_product_identity():
    product_id = uuid4()
    response = {
        "id": str(product_id),
        "name": "RAM",
        "status": "ACTIVE",
        "specifications": {"capacity_gb": 32},
        "variants": [
            {
                "id": str(uuid4()),
                "productId": str(product_id),
                "price": 2_000_000,
                "quantity": 5,
                "status": "ACTIVE",
            },
            {
                "id": str(uuid4()),
                "productId": str(product_id),
                "price": 100,
                "quantity": 5,
                "status": "DELETED",
            },
        ],
    }
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200, json={"data": response, "message": "SUCCESS", "errors": []}
            )
        )
    ) as client:
        catalog = BackendProductCatalog(client, "https://catalog.test/api/v1", Budget())
        detail = await catalog.get(product_id)
        assert detail.price_vnd == 2_000_000
        assert detail.specifications["capacity_gb"] == 32
        with pytest.raises(BackendUnavailableError):
            await catalog.get(uuid4())


@pytest.mark.asyncio
async def test_catalog_error_envelope_never_becomes_successful_empty_search():
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={
                    "data": None,
                    "message": "CATALOG_FAILURE",
                    "errors": [{"message": "failed"}],
                },
            )
        )
    ) as client:
        with pytest.raises(BackendUnavailableError):
            await BackendProductCatalog(
                client, "https://catalog.test/api/v1", Budget()
            ).search("GPU", 5)


def test_rrf_fuse_by_product_id_boosts_items_matching_both_bm25_and_knn():
    id_only_bm25 = uuid4()
    id_both = uuid4()
    id_only_knn = uuid4()

    bm25_ranked = [id_only_bm25, id_both]
    knn_ranked = [id_only_knn, id_both]

    fused = rrf_fuse_by_product_id(bm25_ranked, knn_ranked)
    assert fused[0] == id_both
    assert set(fused) == {id_only_bm25, id_both, id_only_knn}


@pytest.mark.asyncio
async def test_hybrid_search_runs_bm25_and_knn_fuses_via_rrf_and_grounds_against_catalog():
    id_bm25_top = uuid4()
    id_hybrid_winner = uuid4()
    id_inactive = uuid4()
    budget = Budget()
    search_requests = []

    details = {
        str(id_bm25_top): {
            "id": str(id_bm25_top),
            "name": "Tai nghe Razer BlackShark V2 X",
            "categoryName": "Tai nghe",
            "status": "ACTIVE",
            "specifications": {"connection": "3.5mm"},
            "variants": [
                {
                    "id": str(uuid4()),
                    "productId": str(id_bm25_top),
                    "price": 1_290_000,
                    "quantity": 4,
                    "status": "ACTIVE",
                }
            ],
        },
        str(id_hybrid_winner): {
            "id": str(id_hybrid_winner),
            "name": "Tai nghe Sony WH-CH520",
            "categoryName": "Tai nghe",
            "status": "ACTIVE",
            "specifications": {"battery": "50h"},
            "variants": [
                {
                    "id": str(uuid4()),
                    "productId": str(id_hybrid_winner),
                    "price": 1_490_000,
                    "quantity": 8,
                    "status": "ACTIVE",
                }
            ],
        },
        str(id_inactive): {
            "id": str(id_inactive),
            "name": "Tai nghe Cũ Ngừng Bán",
            "status": "INACTIVE",
            "specifications": {},
            "variants": [],
        },
    }

    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.host == "search.test":
            params = dict(request.url.params)
            search_requests.append(params)
            if "q" in params:
                content = [
                    {"id": str(id_bm25_top), "name": "Razer", "inStock": True},
                    {"id": str(id_hybrid_winner), "name": "Sony", "inStock": True},
                ]
            else:
                content = [
                    {"id": str(id_inactive), "name": "Inactive", "inStock": True},
                    {"id": str(id_hybrid_winner), "name": "Sony", "inStock": True},
                ]
            return httpx.Response(
                200,
                json={
                    "data": {
                        "content": content,
                        "page": 0,
                        "size": 20,
                        "totalElements": 2,
                        "totalPages": 1,
                    },
                    "message": "SUCCESS",
                    "errors": [],
                },
            )
        pid = request.url.path.rsplit("/", 1)[-1]
        return httpx.Response(
            200,
            json={"data": details.get(pid), "message": "SUCCESS", "errors": []},
        )

    args = SearchCatalogArgs(
        query="tai nghe",
        semantic_query="tai nghe không dây pin trâu làm việc dưới 2 triệu",
        category="Tai nghe",
        max_price=2_000_000,
        in_stock_only=True,
        limit=5,
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        catalog = BackendProductCatalog(
            client,
            "https://catalog.test/api/v1",
            budget,
            "https://search.test",
        )
        result = await catalog.search(
            args.query or "",
            args.limit,
            semantic_query=args.semantic_query,
            brand=args.brand,
            category=args.category,
            min_price=args.min_price,
            max_price=args.max_price,
            in_stock_only=args.in_stock_only,
        )

    # id_hybrid_winner ranked in both BM25 and kNN -> RRF #1; id_inactive filtered out by catalog grounding
    assert [p.product_id for p in result] == [id_hybrid_winner, id_bm25_top]
    assert result[0].price_vnd == 1_490_000
    assert result[0].specifications == {"battery": "50h"}
    assert len(search_requests) == 2
    assert search_requests[0]["q"] == "tai nghe"
    assert search_requests[0]["maxPrice"] == "2000000"
    assert search_requests[0]["inStockOnly"] == "true"
    assert (
        search_requests[1]["semanticQuery"]
        == "tai nghe không dây pin trâu làm việc dưới 2 triệu"
    )
    assert budget.calls == ["TOOL_IO"] * 5
