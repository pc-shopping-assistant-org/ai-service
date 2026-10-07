"""Pytest fixtures shared across unit and integration tests."""

from __future__ import annotations

import pytest

from ai_service.application.ports.hardware import ComponentCategory, ComponentSpec


@pytest.fixture
def realistic_catalog() -> list[ComponentSpec]:
    """Rich, realistic catalog representing modern hardware for testing."""
    return [
        # CPUs
        ComponentSpec(
            name="AMD Ryzen 5 7600",
            category=ComponentCategory.CPU,
            price=5_200_000,
            socket="AM5",
            tdp_watts=65,
            performance_score=85,
            brand="AMD",
            has_integrated_graphics=True,
            integrated_graphics_score=15,
            integrated_graphics_power_watts=15,
            includes_stock_cooler=True,
            stock_cooler_height_mm=55,
            stock_cooler_score=50,
            stock_cooler_tdp_watts=65,
        ),
        ComponentSpec(
            name="AMD Ryzen 7 7800X3D",
            category=ComponentCategory.CPU,
            price=10_500_000,
            socket="AM5",
            tdp_watts=120,
            performance_score=98,
            brand="AMD",
            has_integrated_graphics=True,
            integrated_graphics_score=15,
            integrated_graphics_power_watts=15,
            includes_stock_cooler=False,
        ),
        ComponentSpec(
            name="Intel Core i5-12400F",
            category=ComponentCategory.CPU,
            price=2_800_000,
            socket="LGA1700",
            tdp_watts=65,
            performance_score=72,
            brand="Intel",
            has_integrated_graphics=False,
            includes_stock_cooler=True,
            stock_cooler_height_mm=47,
            stock_cooler_score=48,
            stock_cooler_tdp_watts=65,
        ),
        ComponentSpec(
            name="Intel Core i5-13600K",
            category=ComponentCategory.CPU,
            price=6_800_000,
            socket="LGA1700",
            tdp_watts=125,
            performance_score=88,
            brand="Intel",
            has_integrated_graphics=True,
            integrated_graphics_score=20,
            integrated_graphics_power_watts=15,
            includes_stock_cooler=False,
        ),

        # Mainboards
        ComponentSpec(name="MSI B650M GAMING PLUS", category=ComponentCategory.MAINBOARD, price=3_400_000, socket="AM5", ram_type="DDR5", form_factor="Micro-ATX", ram_slots=4),
        ComponentSpec(name="ASUS ROG STRIX B650-A", category=ComponentCategory.MAINBOARD, price=5_800_000, socket="AM5", ram_type="DDR5", form_factor="ATX", ram_slots=4),
        ComponentSpec(name="ASRock B760M-HDV DDR4", category=ComponentCategory.MAINBOARD, price=2_100_000, socket="LGA1700", ram_type="DDR4", form_factor="Micro-ATX", ram_slots=2),
        ComponentSpec(name="MSI PRO B760M-A DDR5", category=ComponentCategory.MAINBOARD, price=3_600_000, socket="LGA1700", ram_type="DDR5", form_factor="Micro-ATX", ram_slots=4),

        # RAMs
        ComponentSpec(name="Kingston Fury Beast 16GB DDR4", category=ComponentCategory.RAM, price=950_000, ram_type="DDR4", capacity_gb=16, performance_score=70),
        ComponentSpec(name="Corsair Vengeance 32GB DDR5", category=ComponentCategory.RAM, price=2_800_000, ram_type="DDR5", capacity_gb=32, performance_score=88),
        ComponentSpec(name="Crucial Pro 16GB DDR5", category=ComponentCategory.RAM, price=1_400_000, ram_type="DDR5", capacity_gb=16, performance_score=78),

        # GPUs
        ComponentSpec(name="GeForce RTX 4060 8GB", category=ComponentCategory.GPU, price=7_800_000, tdp_watts=115, gpu_length_mm=240, vram_gb=8, performance_score=76, brand="NVIDIA"),
        ComponentSpec(name="GeForce RTX 4070 SUPER 12GB", category=ComponentCategory.GPU, price=16_500_000, tdp_watts=220, gpu_length_mm=280, vram_gb=12, performance_score=94, brand="NVIDIA"),
        ComponentSpec(name="AMD Radeon RX 6700 XT 12GB", category=ComponentCategory.GPU, price=8_500_000, tdp_watts=230, gpu_length_mm=270, vram_gb=12, performance_score=80, brand="AMD"),
        ComponentSpec(name="AMD Radeon RX 7600 8GB", category=ComponentCategory.GPU, price=6_900_000, tdp_watts=165, gpu_length_mm=210, vram_gb=8, performance_score=72, brand="AMD"),
        ComponentSpec(name="GeForce RTX 3060 12GB", category=ComponentCategory.GPU, price=7_200_000, tdp_watts=170, gpu_length_mm=235, vram_gb=12, performance_score=70, brand="NVIDIA"),

        # Cases
        ComponentSpec(name="Montech Air 100 mATX", category=ComponentCategory.CASE, price=1_100_000, form_factor="Micro-ATX", supported_form_factors=["Micro-ATX", "Mini-ITX"], max_gpu_length_mm=330, max_cooler_height_mm=161),
        ComponentSpec(name="NZXT H5 Flow ATX", category=ComponentCategory.CASE, price=2_100_000, form_factor="ATX", supported_form_factors=["ATX", "Micro-ATX", "Mini-ITX"], max_gpu_length_mm=365, max_cooler_height_mm=165),
        ComponentSpec(name="Compact Cube Case", category=ComponentCategory.CASE, price=900_000, form_factor="Micro-ATX", supported_form_factors=["Micro-ATX"], max_gpu_length_mm=245, max_cooler_height_mm=150),

        # Coolers
        ComponentSpec(name="Thermalright Assassin X 120", category=ComponentCategory.COOLER, price=450_000, cooler_height_mm=148, performance_score=75, supported_sockets=["AM5", "AM4", "LGA1700"]),
        ComponentSpec(name="Deepcool AK620 Digital", category=ComponentCategory.COOLER, price=1_500_000, cooler_height_mm=160, performance_score=90, supported_sockets=["AM5", "AM4", "LGA1700", "LGA1851"]),

        # Storages
        ComponentSpec(name="Kingston NV2 1TB NVMe", category=ComponentCategory.STORAGE, price=1_600_000, capacity_gb=1000, performance_score=75),
        ComponentSpec(name="Samsung 980 PRO 1TB NVMe", category=ComponentCategory.STORAGE, price=2_500_000, capacity_gb=1000, performance_score=92),
        ComponentSpec(name="Kingston NV2 500GB NVMe", category=ComponentCategory.STORAGE, price=950_000, capacity_gb=500, performance_score=65),

        # PSUs
        ComponentSpec(name="MSI MAG A550BN 550W Bronze", category=ComponentCategory.PSU, price=1_100_000, wattage=550, psu_tier="C", performance_score=72),
        ComponentSpec(name="Corsair CV650 650W Bronze", category=ComponentCategory.PSU, price=1_450_000, wattage=650, psu_tier="B", performance_score=80),
        ComponentSpec(name="Super Flower Leadex III 750W Gold", category=ComponentCategory.PSU, price=2_400_000, wattage=750, psu_tier="A", performance_score=94),
        ComponentSpec(name="Corsair RM850e 850W Gold", category=ComponentCategory.PSU, price=3_200_000, wattage=850, psu_tier="A", performance_score=95),
    ]
