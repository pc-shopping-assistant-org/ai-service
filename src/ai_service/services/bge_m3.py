from __future__ import annotations

import asyncio
import logging
import math
import resource
import sys
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from ai_service.application.cpu_work import run_cpu
from ai_service.config.settings import Settings, get_settings

LOGGER = logging.getLogger(__name__)

BGE_M3_DIMENSION = 1024


class DenseEmbedderModel(Protocol):
    """Minimal protocol implemented by FlagEmbedding's BGEM3FlagModel."""

    def encode(
        self,
        sentences: list[str] | str,
        batch_size: int | None = None,
        max_length: int | None = None,
        return_dense: bool | None = None,
        return_sparse: bool | None = None,
        return_colbert_vecs: bool | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class BgeM3RuntimeStatus:
    model_name: str
    cache_dir: str
    loaded: bool
    device: str
    use_fp16: bool
    load_source: str | None
    load_ms: float | None
    rss_before_mb: float | None
    rss_after_mb: float | None
    rss_delta_mb: float | None
    last_latency_ms: float | None
    current_rss_mb: float


def current_rss_mb() -> float:
    """Read resident set size (RSS) in MiB on Linux, with getrusage fallback."""
    status_path = Path("/proc/self/status")
    if status_path.is_file():
        try:
            for line in status_path.read_text(encoding="utf-8").splitlines():
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return round(int(parts[1]) / 1024.0, 2)
        except (OSError, ValueError):
            pass
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return round(usage / (1024.0 * 1024.0), 2)
    return round(usage / 1024.0, 2)


def detect_runtime_device(cuda_available_fn: Callable[[], bool] | None = None) -> tuple[str, bool]:
    """Check whether the container has a usable GPU; otherwise select lightweight CPU config."""
    if cuda_available_fn is not None:
        has_gpu = bool(cuda_available_fn())
    else:
        try:
            import torch

            has_gpu = bool(torch.cuda.is_available())
        except Exception:  # noqa: BLE001
            has_gpu = False
    if has_gpu:
        return ("cuda:0", True)
    return ("cpu", False)


def resolve_cached_or_download_model(
    model_name: str,
    cache_dir: str,
    *,
    snapshot_download_fn: Callable[..., str] | None = None,
) -> tuple[str, str]:
    """Load from mounted cache volume when present; download to mount only on first run."""
    candidate_dir = Path(model_name)
    if candidate_dir.is_dir():
        return (str(candidate_dir), "local_directory")

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    downloader = snapshot_download_fn
    if downloader is None:
        from huggingface_hub import snapshot_download

        downloader = snapshot_download

    try:
        cached_path = downloader(
            repo_id=model_name,
            cache_dir=str(cache_path),
            local_files_only=True,
        )
        return (str(cached_path), "mounted_volume_cache")
    except Exception:  # noqa: BLE001
        LOGGER.info(
            "Model %s not found in mounted cache %s; downloading into volume on first run...",
            model_name,
            cache_path,
        )
        downloaded_path = downloader(
            repo_id=model_name,
            cache_dir=str(cache_path),
            local_files_only=False,
        )
        return (str(downloaded_path), "downloaded_to_mount")


def _default_model_factory(
    resolved_model_path: str,
    *,
    device: str,
    use_fp16: bool,
    cache_dir: str,
    batch_size: int,
    max_length: int,
) -> DenseEmbedderModel:
    from FlagEmbedding import BGEM3FlagModel

    model: DenseEmbedderModel = BGEM3FlagModel(
        model_name_or_path=resolved_model_path,
        normalize_embeddings=True,
        use_fp16=use_fp16,
        devices=device,
        cache_dir=cache_dir,
        batch_size=batch_size,
        query_max_length=max_length,
        passage_max_length=max_length,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )
    return model


class BgeM3EmbeddingService:
    """In-process BAAI/bge-m3 embedding service backed by FlagEmbedding + PyTorch."""

    def __init__(
        self,
        *,
        model_name: str = "BAAI/bge-m3",
        cache_dir: str = "/var/cache/ai-service/huggingface",
        batch_size: int = 32,
        max_length: int = 512,
        cuda_available_fn: Callable[[], bool] | None = None,
        snapshot_download_fn: Callable[..., str] | None = None,
        model_factory: Callable[..., DenseEmbedderModel] | None = None,
    ) -> None:
        self.model_name = model_name.strip() or "BAAI/bge-m3"
        self.cache_dir = cache_dir.strip() or "/var/cache/ai-service/huggingface"
        self.batch_size = max(1, batch_size)
        self.max_length = max(32, max_length)
        self._cuda_available_fn = cuda_available_fn
        self._snapshot_download_fn = snapshot_download_fn
        self._model_factory = model_factory or _default_model_factory

        self._lock = threading.Lock()
        self._model: DenseEmbedderModel | None = None
        self._device: str = "cpu"
        self._use_fp16: bool = False
        self._load_source: str | None = None
        self._load_ms: float | None = None
        self._rss_before_mb: float | None = None
        self._rss_after_mb: float | None = None
        self._last_latency_ms: float | None = None

    def _ensure_loaded(self) -> DenseEmbedderModel:
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model

            device, use_fp16 = detect_runtime_device(self._cuda_available_fn)
            rss_before = current_rss_mb()
            start = time.perf_counter()

            resolved_path, source = resolve_cached_or_download_model(
                self.model_name,
                self.cache_dir,
                snapshot_download_fn=self._snapshot_download_fn,
            )
            model = self._model_factory(
                resolved_path,
                device=device,
                use_fp16=use_fp16,
                cache_dir=self.cache_dir,
                batch_size=self.batch_size,
                max_length=self.max_length,
            )

            elapsed_ms = round((time.perf_counter() - start) * 1000.0, 2)
            rss_after = current_rss_mb()

            self._model = model
            self._device = device
            self._use_fp16 = use_fp16
            self._load_source = source
            self._load_ms = elapsed_ms
            self._rss_before_mb = rss_before
            self._rss_after_mb = rss_after

            LOGGER.info(
                "Loaded BGE-M3 model=%s source=%s device=%s use_fp16=%s load_ms=%.2f rss_before_mb=%.2f rss_after_mb=%.2f rss_delta_mb=%.2f",
                self.model_name,
                source,
                device,
                use_fp16,
                elapsed_ms,
                rss_before,
                rss_after,
                round(rss_after - rss_before, 2),
            )
            return model

    async def warmup(self) -> BgeM3RuntimeStatus:
        """Ensure model is present in the mounted volume and loaded into memory at startup."""
        await run_cpu(self._ensure_loaded)
        return self.status()

    def _encode_sync(self, texts: list[str]) -> list[list[float]]:
        model = self._ensure_loaded()
        start = time.perf_counter()
        output = model.encode(
            texts,
            batch_size=min(self.batch_size, max(1, len(texts))),
            max_length=self.max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        elapsed_ms = round((time.perf_counter() - start) * 1000.0, 2)
        self._last_latency_ms = elapsed_ms

        raw_dense = output.get("dense_vecs") if isinstance(output, dict) else output
        if raw_dense is None:
            raise RuntimeError("BGEM3FlagModel returned no dense_vecs")

        rows = raw_dense.tolist() if hasattr(raw_dense, "tolist") else list(raw_dense)
        vectors: list[list[float]] = []
        for row in rows:
            vec = [float(val) for val in row]
            if len(vec) != BGE_M3_DIMENSION or not all(math.isfinite(v) for v in vec):
                raise RuntimeError(
                    f"Expected {BGE_M3_DIMENSION} finite floats from BGE-M3, got length={len(vec)}"
                )
            vectors.append(vec)

        LOGGER.debug(
            "BGE-M3 embedded batch_size=%d latency_ms=%.2f rss_mb=%.2f",
            len(texts),
            elapsed_ms,
            current_rss_mb(),
        )
        return vectors

    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        cleaned = [text.strip() for text in texts if isinstance(text, str) and text.strip()]
        if not cleaned:
            return []
        return await run_cpu(lambda: self._encode_sync(cleaned))

    async def embed(self, text: str) -> list[float]:
        vectors = await self.embed_texts([text])
        if not vectors:
            raise ValueError("Cannot embed empty text")
        return vectors[0]

    def status(self) -> BgeM3RuntimeStatus:
        rss_delta = (
            round(self._rss_after_mb - self._rss_before_mb, 2)
            if self._rss_before_mb is not None and self._rss_after_mb is not None
            else None
        )
        return BgeM3RuntimeStatus(
            model_name=self.model_name,
            cache_dir=self.cache_dir,
            loaded=self._model is not None,
            device=self._device,
            use_fp16=self._use_fp16,
            load_source=self._load_source,
            load_ms=self._load_ms,
            rss_before_mb=self._rss_before_mb,
            rss_after_mb=self._rss_after_mb,
            rss_delta_mb=rss_delta,
            last_latency_ms=self._last_latency_ms,
            current_rss_mb=current_rss_mb(),
        )


_SERVICE_LOCK = threading.Lock()
_DEFAULT_SERVICE: BgeM3EmbeddingService | None = None


def get_bge_m3_service(settings: Settings | None = None) -> BgeM3EmbeddingService:
    global _DEFAULT_SERVICE
    if _DEFAULT_SERVICE is not None:
        return _DEFAULT_SERVICE
    with _SERVICE_LOCK:
        if _DEFAULT_SERVICE is None:
            runtime = settings or get_settings()
            _DEFAULT_SERVICE = BgeM3EmbeddingService(
                model_name=runtime.embedding_model_name,
                cache_dir=runtime.embedding_cache_dir,
                batch_size=runtime.embedding_batch_size,
                max_length=runtime.embedding_max_length,
            )
        return _DEFAULT_SERVICE


async def _run_cli_benchmark() -> None:
    service = get_bge_m3_service()
    status = await service.warmup()
    sample_query = "Tư vấn cấu hình PC gaming 25 triệu chiến mượt Black Myth Wukong 2K"
    vec = await service.embed(sample_query)
    after = service.status()
    print(
        f"BGE-M3 Benchmark: source={status.load_source} device={after.device} "
        f"use_fp16={after.use_fp16} load_ms={after.load_ms} "
        f"rss_before_mb={after.rss_before_mb} rss_after_mb={after.rss_after_mb} "
        f"rss_delta_mb={after.rss_delta_mb} query_latency_ms={after.last_latency_ms} "
        f"dims={len(vec)}"
    )


if __name__ == "__main__":
    asyncio.run(_run_cli_benchmark())
