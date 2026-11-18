"""
Script sinh 100 cấu hình PC không xung đột (Conflict-Free PC Builds)
theo tài liệu 100_PC_BUILD_GENERATION_SOLUTION.md.

Thực hiện:
1. Kết nối TRỰC TIẾP tới PostgreSQL catalog_db (và media_db nếu có)
   - Đọc bảng products, product_variants, categories, hardware_types
   - Phân loại 8 nhóm linh kiện chính (CPU, MAINBOARD, RAM, GPU, PSU, CASE, COOLER, STORAGE)
2. Thuật toán Anchor-based CSP:
   - 4 Tiers:
     - Tier 1: Entry / Office (20 bộ, 7-14tr)
     - Tier 2: Mainstream Gaming (35 bộ, 15-25tr)
     - Tier 3: Mid-High Creator (30 bộ, 26-45tr)
     - Tier 4: Enthusiast / AI (15 bộ, 50-120+tr)
3. Thẩm định qua LocalHardwareRuleEngine & check_build() của ai-service
4. Xuất file:
   - ai-service/data/prebuilt_pc_builds.json
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

# Setup Python path to import ai_service
SCRIPT_DIR = Path(__file__).resolve().parent
AI_SERVICE_DIR = SCRIPT_DIR.parent
ROOT_DIR = AI_SERVICE_DIR.parent

src_dir = AI_SERVICE_DIR / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

try:
    import psycopg
except ImportError:
    print("[!] Cần thư viện psycopg để kết nối PostgreSQL. Hãy chạy qua Python venv của ai-service:")
    print(r"    .\ai-service\.venv\Scripts\python.exe ai-service\scripts\generate_100_pc_builds.py")
    sys.exit(1)

from ai_service.application.ports.hardware import ComponentCategory, ComponentSpec
from ai_service.infrastructure.hardware.rule_engine import LocalHardwareRuleEngine
from ai_service.capabilities.pc_builder.application.build_check import check_build, CheckStatus

HARDWARE_CODE_TO_CATEGORY = {
    "CPU": ComponentCategory.CPU,
    "MAINBOARD": ComponentCategory.MAINBOARD,
    "RAM": ComponentCategory.RAM,
    "GPU": ComponentCategory.GPU,
    "PSU": ComponentCategory.PSU,
    "CASE": ComponentCategory.CASE,
    "COOLER": ComponentCategory.COOLER,
    "STORAGE": ComponentCategory.STORAGE,
    "SSD": ComponentCategory.STORAGE,
    "HDD": ComponentCategory.STORAGE,
}

def get_db_connection_params():
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    user = os.getenv("POSTGRES_USER", "postgres")
    password = os.getenv("POSTGRES_PASSWORD", "postgres")
    catalog_db = os.getenv("CATALOG_DB", "catalog_db")
    media_db = os.getenv("MEDIA_DB", "media_db")
    return {
        "host": host,
        "port": port,
        "user": user,
        "password": password,
        "catalog_db": catalog_db,
        "media_db": media_db,
    }

def load_catalog_from_database():
    params = get_db_connection_params()

    print(f"[*] Đang kết nối tới PostgreSQL catalog_db ({params['host']}:{params['port']}/{params['catalog_db']})...")

    # 1. Đọc public_url của ảnh từ media_db
    file_urls: dict[str, str] = {}
    try:
        with psycopg.connect(
            host=params["host"],
            port=int(params["port"]),
            user=params["user"],
            password=params["password"],
            dbname=params["media_db"],
        ) as media_conn:
            with media_conn.cursor() as cur:
                cur.execute("SELECT id::text, public_url FROM files WHERE public_url IS NOT NULL;")
                for fid, url in cur.fetchall():
                    file_urls[fid] = url
        print(f"    [+] media_db: Đã nạp {len(file_urls)} liên kết ảnh media.")
    except Exception as e:
        print(f"    [!] Không thể kết nối media_db (bỏ qua ảnh): {e}")

    # 2. Đọc linh kiện từ catalog_db
    components: dict[ComponentCategory, list[dict]] = {cat: [] for cat in ComponentCategory}

    query = """
    SELECT 
        p.id::text AS product_id,
        v.id::text AS variant_id,
        p.name AS product_name,
        h.code AS hardware_code,
        c.seo_name AS category_seo,
        v.price AS price,
        p.specifications AS specs,
        COALESCE(v.image_file_id, img.image_file_id)::text AS image_file_id
    FROM products p
    JOIN categories c ON p.category_id = c.id
    JOIN hardware_types h ON c.hardware_type_id = h.id
    JOIN product_variants v ON v.product_id = p.id
    LEFT JOIN product_images img ON img.product_id = p.id AND img.is_main = true
    WHERE p.status = 'ACTIVE' 
      AND v.status = 'ACTIVE'
      AND h.code IN ('CPU', 'MAINBOARD', 'RAM', 'GPU', 'PSU', 'CASE', 'COOLER', 'STORAGE', 'SSD', 'HDD')
    ORDER BY p.created_at ASC;
    """

    with psycopg.connect(
        host=params["host"],
        port=int(params["port"]),
        user=params["user"],
        password=params["password"],
        dbname=params["catalog_db"],
    ) as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()

            for row in rows:
                prod_id, var_id, name, hw_code, cat_seo, price, specs, img_file_id = row
                cat_enum = HARDWARE_CODE_TO_CATEGORY.get(hw_code)
                if not cat_enum:
                    continue

                img_url = file_urls.get(img_file_id) if img_file_id else None
                specs_dict = specs if isinstance(specs, dict) else {}

                raw_part = {
                    "product_id": prod_id,
                    "variant_id": var_id,
                    "name": name,
                    "category": cat_enum,
                    "category_seo": cat_seo,
                    "price": price,
                    "image_url": img_url,
                    "specs": specs_dict,
                }
                components[cat_enum].append(raw_part)

    return components

def to_component_spec(raw: dict) -> ComponentSpec:
    specs = raw["specs"]
    return ComponentSpec(
        id=raw["variant_id"],
        name=raw["name"],
        category=raw["category"],
        price=raw["price"],
        socket=specs.get("socket"),
        ram_type=specs.get("ram_type"),
        ram_slots=specs.get("ram_slots"),
        form_factor=specs.get("form_factor"),
        supported_form_factors=specs.get("supported_form_factors") or [],
        tdp_watts=specs.get("tdp_watts"),
        wattage=specs.get("wattage"),
        gpu_length_mm=specs.get("gpu_length_mm"),
        max_gpu_length_mm=specs.get("max_gpu_length_mm"),
        cooler_height_mm=specs.get("cooler_height_mm"),
        max_cooler_height_mm=specs.get("max_cooler_height_mm"),
        radiator_mm=specs.get("radiator_mm"),
        max_radiator_mm=specs.get("max_radiator_mm"),
        m2_slots=specs.get("m2_slots"),
        interface=specs.get("interface"),
        supported_sockets=specs.get("supported_sockets") or [],
        extra_specs=specs,
    )

def is_valid_build(engine: LocalHardwareRuleEngine, parts_map: dict[str, dict]) -> bool:
    specs_map = {slot: to_component_spec(raw) for slot, raw in parts_map.items()}
    check = check_build(engine, specs_map)
    if check.overall != CheckStatus.OK:
        return False
    # Extra safety check for PSU headroom:
    p_report = check.power
    if p_report and p_report.is_sufficient is False:
        return False
    return True

def generate_builds():
    catalog = load_catalog_from_database()
    engine = LocalHardwareRuleEngine()

    print("[*] Thống kê linh kiện từ catalog_db:")
    for cat in [ComponentCategory.CPU, ComponentCategory.MAINBOARD, ComponentCategory.RAM,
                ComponentCategory.GPU, ComponentCategory.PSU, ComponentCategory.CASE,
                ComponentCategory.COOLER, ComponentCategory.STORAGE]:
        print(f"    - {cat.value}: {len(catalog[cat])} linh kiện")

    # Group CPU by tier
    cpus = catalog[ComponentCategory.CPU]
    tier_cpus: dict[str, list[dict]] = {
        "tier1": [],
        "tier2": [],
        "tier3": [],
        "tier4": [],
    }
    for cpu in cpus:
        price = cpu["price"]
        name = cpu["name"].lower()
        if price < 4_000_000 or "i3" in name or "4600g" in name or "5600g" in name or "12100" in name:
            tier_cpus["tier1"].append(cpu)
        elif price < 7_000_000 or "i5" in name or "7500" in name or "7600" in name or "5600" in name:
            tier_cpus["tier2"].append(cpu)
        elif price < 12_000_000 or "i7" in name or "13600" in name or "14600" in name or "7700" in name or "7800x3d" in name:
            tier_cpus["tier3"].append(cpu)
        else:
            tier_cpus["tier4"].append(cpu)

    for t in ["tier1", "tier2", "tier3", "tier4"]:
        if not tier_cpus[t]:
            tier_cpus[t] = cpus

    # Group GPU by tier
    gpus = catalog[ComponentCategory.GPU]
    tier_gpus: dict[str, list[dict]] = {
        "tier1": [g for g in gpus if g["price"] <= 5_500_000],
        "tier2": [g for g in gpus if 6_000_000 <= g["price"] <= 12_000_000],
        "tier3": [g for g in gpus if 12_000_000 <= g["price"] <= 26_000_000],
        "tier4": [g for g in gpus if g["price"] >= 26_000_000],
    }

    tier_plans = [
        ("tier1", "office", "Văn phòng & Đồ họa nhẹ", 20, 7_000_000, 14_000_000),
        ("tier2", "gaming", "Chiến mượt Game AAA 1080p", 35, 14_000_000, 25_000_000),
        ("tier3", "creator", "Đồ họa 4K & Gaming 2K", 30, 26_000_000, 48_000_000),
        ("tier4", "gaming", "Enthusiast / AI Workstation", 15, 50_000_000, 130_000_000),
    ]

    all_builds = []
    build_count = 0

    print("\n[*] Bắt đầu sinh 100 cấu hình bằng Anchor-based CSP...")
    random.seed(42)

    for tier_key, goal, subtitle, target_count, min_b, max_b in tier_plans:
        tier_builds = []
        attempts = 0
        cpu_candidates = tier_cpus[tier_key].copy()
        gpu_candidates = tier_gpus[tier_key].copy() or gpus.copy()

        while len(tier_builds) < target_count and attempts < 1500:
            attempts += 1
            cpu = random.choice(cpu_candidates)
            cpu_spec = to_component_spec(cpu)

            # 1. MB matching socket
            mbs = [mb for mb in catalog[ComponentCategory.MAINBOARD] if to_component_spec(mb).socket == cpu_spec.socket]
            if not mbs:
                continue
            mb = random.choice(mbs)
            mb_spec = to_component_spec(mb)

            # 2. RAM matching ram_type
            rams = [r for r in catalog[ComponentCategory.RAM] if to_component_spec(r).ram_type == mb_spec.ram_type]
            if not rams:
                continue
            if tier_key in ("tier3", "tier4"):
                rams_32 = [r for r in rams if (r["specs"].get("capacity_gb") or 0) >= 32]
                ram = random.choice(rams_32 or rams)
            else:
                ram = random.choice(rams)

            # 3. Cooler matching CPU socket
            coolers = [c for c in catalog[ComponentCategory.COOLER] if cpu_spec.socket in (to_component_spec(c).supported_sockets or [])]
            if not coolers:
                continue
            if tier_key == "tier4":
                aio_coolers = [c for c in coolers if c["category_seo"] == "cooler-aio"]
                cooler = random.choice(aio_coolers or coolers)
            elif tier_key == "tier1":
                air_coolers = [c for c in coolers if c["category_seo"] == "cooler-air"]
                cooler = random.choice(air_coolers or coolers)
            else:
                cooler = random.choice(coolers)
            cooler_spec = to_component_spec(cooler)

            # 4. GPU
            gpu = random.choice(gpu_candidates)
            gpu_spec = to_component_spec(gpu)

            # 5. Case matching MB form factor, GPU length, and Cooler clearance
            cases = []
            for c in catalog[ComponentCategory.CASE]:
                cs = to_component_spec(c)
                if mb_spec.form_factor and mb_spec.form_factor not in (cs.supported_form_factors or []):
                    continue
                if cs.max_gpu_length_mm and gpu_spec.gpu_length_mm and gpu_spec.gpu_length_mm > cs.max_gpu_length_mm - 5:
                    continue
                if cooler_spec.radiator_mm:
                    if not cs.max_radiator_mm or cooler_spec.radiator_mm > cs.max_radiator_mm:
                        continue
                elif cooler_spec.cooler_height_mm:
                    if not cs.max_cooler_height_mm or cooler_spec.cooler_height_mm > cs.max_cooler_height_mm - 2:
                        continue
                cases.append(c)

            if not cases:
                continue
            case = random.choice(cases)

            # 6. PSU - Power calculation
            est_tdp = (cpu_spec.tdp_watts or 65) + (gpu_spec.tdp_watts or 150) + 70
            min_psu = int(est_tdp * 1.35)
            psus = [p for p in catalog[ComponentCategory.PSU] if (to_component_spec(p).wattage or 0) >= min_psu]
            if not psus:
                continue
            psu = random.choice(psus)

            # 7. SSD
            ssds = catalog[ComponentCategory.STORAGE]
            if mb_spec.m2_slots and mb_spec.m2_slots > 0:
                ssd = random.choice(ssds)
            else:
                sata_ssds = [s for s in ssds if "SATA" in str(s["specs"].get("interface", "")).upper()]
                ssd = random.choice(sata_ssds or ssds)

            build_parts = {
                "CPU": cpu,
                "MAINBOARD": mb,
                "RAM": ram,
                "GPU": gpu,
                "PSU": psu,
                "CASE": case,
                "COOLER": cooler,
                "SSD": ssd,
            }

            # 8. Thẩm định qua Rule Engine
            if is_valid_build(engine, build_parts):
                total_price = sum(p["price"] for p in build_parts.values())
                fingerprint = "-".join(sorted(p["variant_id"] for p in build_parts.values()))
                if any(b["fingerprint"] == fingerprint for b in tier_builds):
                    continue

                build_count += 1
                build_name = f"PC {cpu['name'].split('(')[0].strip()} · {gpu['specs'].get('gpu_chip', gpu['name']).split('(')[0].strip()}"
                
                tier_builds.append({
                    "id": f"build-{build_count:03d}",
                    "name": build_name,
                    "tier": tier_key,
                    "goal": goal,
                    "subtitle": subtitle,
                    "total_price": total_price,
                    "fingerprint": fingerprint,
                    "selection": {slot: p["variant_id"] for slot, p in build_parts.items()},
                    "components": {
                        slot: {
                            "id": p["variant_id"],
                            "product_id": p["product_id"],
                            "name": p["name"],
                            "price": p["price"],
                            "image_url": p["image_url"],
                        }
                        for slot, p in build_parts.items()
                    }
                })

        print(f"    [+] {tier_key} ({goal}): Sinh thành công {len(tier_builds)}/{target_count} bộ PC.")
        all_builds.extend(tier_builds)

    print(f"\n[V] Đã hoàn thành sinh {len(all_builds)} bộ PC từ PostgreSQL catalog_db!")

    # Sort builds by price
    all_builds.sort(key=lambda b: b["total_price"])

    def write_json_clean(target_path: Path, data: list):
        target_path.parent.mkdir(parents=True, exist_ok=True)
        # Clear content completely before writing (truncate)
        with open(target_path, "w", encoding="utf-8") as f:
            f.truncate(0)
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())

    # Xuất ai-service/data/prebuilt_pc_builds.json
    ai_out_file = AI_SERVICE_DIR / "data" / "prebuilt_pc_builds.json"
    write_json_clean(ai_out_file, all_builds)
    print(f"    -> Đã ghi mới sạch sẽ (clean/truncate): {ai_out_file}")

if __name__ == "__main__":
    generate_builds()
