from ai_service.application.use_cases.assistant import AssistantService


def test_catalog_summary_preserves_microservice_image_and_price() -> None:
    card = AssistantService._card(
        {
            "name": "Gaming laptop",
            "mainImageUrl": "https://example.com/laptop.png",
            "imageUrl": "https://example.com/legacy.png",
            "minPrice": 20_000_000,
            "status": "ACTIVE",
        }
    )
    assert card.image_url == "https://example.com/laptop.png"
    assert card.list_price == 20_000_000


def test_catalog_image_mapping_keeps_legacy_fallbacks_when_main_image_is_empty() -> (
    None
):
    card = AssistantService._card(
        {
            "name": "GPU",
            "mainImageUrl": None,
            "imageUrl": "https://example.com/gpu.png",
        }
    )
    assert card.image_url == "https://example.com/gpu.png"
    snake_case = AssistantService._card(
        {
            "name": "CPU",
            "mainImageUrl": "",
            "imageUrl": None,
            "image_url": "https://example.com/cpu.png",
        }
    )
    assert snake_case.image_url == "https://example.com/cpu.png"
