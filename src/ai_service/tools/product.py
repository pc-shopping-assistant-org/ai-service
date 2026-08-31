from ai_service.services.backend_client import BackendClient


async def search_products(query: str, limit: int = 10) -> list[dict]:
    """Catalog tool used by search/chat/consult flows."""
    return await BackendClient().search_products(query, limit)
