from types import SimpleNamespace
from typing import ClassVar

import onnxruntime
import pytest

from embeddings import embedder


@pytest.fixture
def cuda_available(monkeypatch):
    monkeypatch.setattr(
        onnxruntime, "get_available_providers", lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    monkeypatch.setattr(onnxruntime, "preload_dlls", lambda *args, **kwargs: None, raising=False)


@pytest.fixture
def cpu_only(monkeypatch):
    monkeypatch.setattr(onnxruntime, "get_available_providers", lambda: ["CPUExecutionProvider"])


def test_auto_prefers_cuda_with_cpu_fallback(cuda_available):
    assert embedder.onnx_providers("auto") == ["CUDAExecutionProvider", "CPUExecutionProvider"]


def test_cpu_can_be_forced(cuda_available):
    assert embedder.onnx_providers("cpu") == ["CPUExecutionProvider"]


def test_falls_back_to_cpu_without_cuda(cpu_only):
    assert embedder.onnx_providers("auto") == ["CPUExecutionProvider"]


class FakeTextEmbedding:
    """Stands in for fastembed.TextEmbedding; `session_providers` is what ONNX Runtime really loaded."""

    session_providers: ClassVar[list[str]] = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    def __init__(self, model_name, providers):
        self.kwargs = {"model_name": model_name, "providers": providers}
        session = SimpleNamespace(get_providers=lambda: self.session_providers)
        self.model = SimpleNamespace(model=session)


def test_embeddings_report_the_device_onnx_runtime_actually_uses(cuda_available, monkeypatch):
    import fastembed

    monkeypatch.setattr(fastembed, "TextEmbedding", FakeTextEmbedding)

    on_gpu = embedder.FastEmbedEmbeddings("some/model", device="auto")
    assert on_gpu._model.kwargs["providers"] == ["CUDAExecutionProvider", "CPUExecutionProvider"]
    assert on_gpu.device == "cuda"

    # CUDA was offered but failed to start (missing DLLs): ONNX Runtime quietly uses the CPU.
    monkeypatch.setattr(FakeTextEmbedding, "session_providers", ["CPUExecutionProvider"])
    assert embedder.FastEmbedEmbeddings("some/model", device="auto").device == "cpu"


def test_gpu_uses_larger_batches(cuda_available, monkeypatch):
    import fastembed

    monkeypatch.setattr(fastembed, "TextEmbedding", FakeTextEmbedding)

    assert embedder.FastEmbedEmbeddings("some/model", device="auto").batch_size > embedder.CPU_BATCH_SIZE


def test_concurrent_callers_share_one_model(monkeypatch):
    import threading
    import time

    created = []

    class SlowModel:
        def __init__(self, model_name):
            time.sleep(0.05)  # long enough for the other thread to arrive while this one loads
            created.append(model_name)

    monkeypatch.setattr(embedder, "FastEmbedEmbeddings", SlowModel)
    embedder._load_embedding_model.cache_clear()
    results = []
    threads = [threading.Thread(target=lambda: results.append(embedder.get_embedding_model("m"))) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    embedder._load_embedding_model.cache_clear()

    assert created == ["m"]
    assert all(r is results[0] for r in results)
