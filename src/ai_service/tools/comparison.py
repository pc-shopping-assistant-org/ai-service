from ai_service.services.backend_client import BackendClient


async def get_product(product_id):
    return await BackendClient().get_product(product_id)
