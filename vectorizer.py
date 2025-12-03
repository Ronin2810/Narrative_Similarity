from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np

# ================= Base interface =================
class BaseVectorizer(ABC):
    """Interface so the rest of the pipeline is decoupled from implementation."""
    @abstractmethod
    def fit(self, texts: List[str]) -> None: ...
    @abstractmethod
    def transform(self, texts: List[str]) -> np.ndarray: ...
    @property
    @abstractmethod
    def dim(self) -> int: ...

# ================= TF–IDF fallback (no native deps) =================
from sklearn.feature_extraction.text import TfidfVectorizer

class TfidfVectorizerWrapper(BaseVectorizer):
    """Fixed-dimension TF–IDF; used as a safe fallback."""
    def __init__(self, max_features: int = 512):
        self._max_features = max_features
        self._vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
        self._fitted = False
    def fit(self, texts: List[str]) -> None:
        self._vectorizer.fit(texts); self._fitted = True
    def transform(self, texts: List[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Vectorizer must be fitted before transform()")
        return self._vectorizer.transform(texts).toarray().astype("float32")
    @property
    def dim(self) -> int: return self._max_features

# ================= Finance-domain embeddings (lazy import) =================
class FinanceEmbeddingVectorizer(BaseVectorizer):
    """
    Finance sentence embeddings via SentenceTransformers (default ~768-d).
    Lazy-imports to avoid Windows DLL errors on import. If unavailable, raise cleanly.
    """
    def __init__(
        self,
        model_name: str = "FinLang/finance-embeddings-investopedia",
        device: Optional[str] = None,    # "cpu" | "cuda" | None
        normalize: bool = True,
        batch_size: int = 32,
    ):
        try:
            from sentence_transformers import SentenceTransformer  # lazy import
        except Exception as e:
            raise ImportError(
                "sentence-transformers (or torch) not available. "
                "Install CPU wheels if needed:\n"
                "  pip install --force-reinstall --no-cache-dir torch "
                "--index-url https://download.pytorch.org/whl/cpu\n"
                "  pip install sentence-transformers"
            ) from e

        self.model = SentenceTransformer(model_name)
        if device:
            try: self.model = self.model.to(device)
            except Exception: pass
        self._dim = self.model.get_sentence_embedding_dimension()
        self._normalize = normalize
        self._batch_size = batch_size
        self._fitted = True  # pretrained

    def fit(self, texts: List[str]) -> None:
        print("FinanceEmbeddingVectorizer: fit() is a no-op for pretrained models.")
        self._fitted = True  # no-op

    def transform(self, texts: List[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Vectorizer must be fitted before transform()")
        if not texts:
            return np.empty((0, self._dim), dtype="float32")
        emb = self.model.encode(
            texts,
            batch_size=self._batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self._normalize,
            show_progress_bar=False,
        )
        return emb.astype("float32")

    @property
    def dim(self) -> int: return self._dim

# ================= Factory (robust selection) =================
def get_vectorizer(
    prefer_finance: bool = True,
    tfidf_dim: int = 512,
    finance_model: str = "FinLang/finance-embeddings-investopedia",
    device: Optional[str] = None,
    normalize: bool = True,
    batch_size: int = 32,
) -> BaseVectorizer:
    """
    Try finance embeddings first; if that import/runtime fails, fall back to TF–IDF.
    """
    if prefer_finance:
        try:
            return FinanceEmbeddingVectorizer(
                model_name=finance_model, device=device,
                normalize=normalize, batch_size=batch_size
            )
        except Exception as e:
            raise e
            # return TfidfVectorizerWrapper(max_features=tfidf_dim)
    # return TfidfVectorizerWrapper(max_features=tfidf_dim)
