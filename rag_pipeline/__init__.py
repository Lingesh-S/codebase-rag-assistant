# rag_pipeline/__init__.py

__all__ = ["Answer", "RAGPipeline", "RAGSettings"]


def __getattr__(name):
    # Imported lazily so `python -m rag_pipeline.main` can load .env before any module reads its settings.
    if name in __all__:
        from rag_pipeline import pipeline

        return getattr(pipeline, name)
    raise AttributeError(f"module 'rag_pipeline' has no attribute {name!r}")
