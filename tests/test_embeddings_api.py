from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest

from ai_service.api.embeddings import get_embedding_service
from ai_service.main import app
from ai_service.services.bge_m3 import (
    BGE_M3_DIMENSION,
    BgeM3EmbeddingService,
    detect_runtime_device,
    resolve_cached_or_download_model,
)


class _FakeBgeM3Model:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def encode(
        self,
        sentences: list[str] | str,
        batch_size: int | None = None,
        max_length: int | None = None,
        return_dense: bool | None = None,
        return_sparse: bool | None = None,
        return_colbert_vecs: bool | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        items = [sentences] if isinstance(sentences, str) else list(sentences)
        self.calls.append(
            {
                "sentences": items,
                "batch_size": batch_size,
                "max_length": max_length,
                "return_dense": return_dense,
                "return_sparse": return_sparse,
                "return_colbert_vecs": return_colbert_vecs,
            }
        )
        dense_vecs = [
            [round((idx + 1) * 0.001, 6)] * BGE_M3_DIMENSION
            for idx in range(len(items))
        ]
        return {"dense_vecs": dense_vecs}


def test_detect_runtime_device_chooses_lightweight_cpu_when_no_gpu() -> None:
    device, use_fp16 = detect_runtime_device(lambda: False)
    assert device == "cpu"
    assert use_fp16 is False


def test_detect_runtime_device_enables_cuda_fp16_when_gpu_present() -> None:
    device, use_fp16 = detect_runtime_device(lambda: True)
    assert device == "cuda:0"
    assert use_fp16 is True


def test_resolve_cached_or_download_model_downloads_on_first_run_and_reuses_mount_afterward(
    tmp_path: Path,
) -> None:
    mount_dir = tmp_path / "hf-mount"
    calls: list[bool] = []
    cached_snapshot = mount_dir / "models--BAAI--bge-m3" / "snapshots" / "rev1"

    def fake_downloader(*, repo_id: str, cache_dir: str, local_files_only: bool) -> str:
        assert repo_id == "BAAI/bge-m3"
        assert cache_dir == str(mount_dir)
        calls.append(local_files_only)
        if local_files_only and not cached_snapshot.exists():
            raise FileNotFoundError("Snapshot not yet in mounted volume")
        cached_snapshot.mkdir(parents=True, exist_ok=True)
        return str(cached_snapshot)

    first_path, first_source = resolve_cached_or_download_model(
        "BAAI/bge-m3",
        str(mount_dir),
        snapshot_download_fn=fake_downloader,
    )
    assert first_source == "downloaded_to_mount"
    assert first_path == str(cached_snapshot)
    assert calls == [True, False]

    calls.clear()
    second_path, second_source = resolve_cached_or_download_model(
        "BAAI/bge-m3",
        str(mount_dir),
        snapshot_download_fn=fake_downloader,
    )
    assert second_source == "mounted_volume_cache"
    assert second_path == str(cached_snapshot)
    assert calls == [True]


@pytest.mark.asyncio
async def test_bge_m3_service_warmup_and_encode_record_benchmark_metrics(
    tmp_path: Path,
) -> None:
    fake_model = _FakeBgeM3Model()
    factory_kwargs: list[dict[str, Any]] = []

    def fake_factory(resolved_path: str, **kwargs: Any) -> _FakeBgeM3Model:
        factory_kwargs.append({"resolved_path": resolved_path, **kwargs})
        return fake_model

    service = BgeM3EmbeddingService(
        model_name="BAAI/bge-m3",
        cache_dir=str(tmp_path),
        batch_size=16,
        max_length=512,
        cuda_available_fn=lambda: False,
        snapshot_download_fn=lambda **_: str(tmp_path / "snapshot"),
        model_factory=fake_factory,
    )

    status = await service.warmup()
    assert status.loaded is True
    assert status.device == "cpu"
    assert status.use_fp16 is False
    assert status.load_source == "mounted_volume_cache"
    assert status.load_ms is not None and status.load_ms >= 0.0
    assert status.rss_before_mb is not None
    assert status.rss_after_mb is not None
    assert len(factory_kwargs) == 1
    assert factory_kwargs[0]["device"] == "cpu"
    assert factory_kwargs[0]["use_fp16"] is False

    vec = await service.embed("Card màn hình RTX 4070 Super")
    assert len(vec) == BGE_M3_DIMENSION
    assert fake_model.calls[0]["return_dense"] is True
    assert fake_model.calls[0]["return_sparse"] is False
    assert fake_model.calls[0]["return_colbert_vecs"] is False
    assert service.status().last_latency_ms is not None


@pytest.mark.asyncio
async def test_embeddings_endpoint_returns_canonical_envelope(tmp_path: Path) -> None:
    fake_model = _FakeBgeM3Model()
    service = BgeM3EmbeddingService(
        model_name="BAAI/bge-m3",
        cache_dir=str(tmp_path),
        cuda_available_fn=lambda: False,
        snapshot_download_fn=lambda **_: str(tmp_path / "snapshot"),
        model_factory=lambda *_, **__: fake_model,
    )

    previous = app.dependency_overrides.get(get_embedding_service)
    app.dependency_overrides[get_embedding_service] = lambda: service
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            single = await client.post(
                "/api/v1/embeddings",
                json={"model": "BAAI/bge-m3", "input": "Mainboard B650 AM5 DDR5"},
            )
            assert single.status_code == 200
            body = single.json()
            assert list(body) == ["data", "message", "errors"]
            assert body["message"] == "SUCCESS"
            assert body["errors"] == []
            assert body["data"]["model"] == "BAAI/bge-m3"
            assert body["data"]["device"] == "cpu"
            assert len(body["data"]["embedding"]) == BGE_M3_DIMENSION
            assert len(body["data"]["data"]) == 1
            assert body["data"]["data"][0]["index"] == 0
            assert len(body["data"]["data"][0]["embedding"]) == BGE_M3_DIMENSION

            batch = await client.post(
                "/api/v1/embeddings",
                json={"model": "BAAI/bge-m3", "input": ["CPU Ryzen 7", "RAM 32GB DDR5"]},
            )
            assert batch.status_code == 200
            batch_data = batch.json()["data"]
            assert len(batch_data["data"]) == 2
            assert batch_data["data"][1]["index"] == 1

            invalid = await client.post(
                "/api/v1/embeddings",
                json={"model": "BAAI/bge-m3", "input": "   "},
            )
            assert invalid.status_code == 422
            invalid_body = invalid.json()
            assert invalid_body["message"] == "VALIDATION_ERROR"
            assert invalid_body["data"] is None
            assert len(invalid_body["errors"]) >= 1
    finally:
        if previous is None:
            app.dependency_overrides.pop(get_embedding_service, None)
        else:
            app.dependency_overrides[get_embedding_service] = previous
