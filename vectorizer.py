from abc import ABC, abstractmethod
from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class BaseVectorizer(ABC):
    """Interface so the rest of the pipeline is decoupled from implementation."""

    @abstractmethod
    def fit(self, texts: List[str]) -> None:
        ...

    @abstractmethod
    def transform(self, texts: List[str]) -> np.ndarray:
        ...

    @property
    @abstractmethod
    def dim(self) -> int:
        ...


class TfidfVectorizerWrapper(BaseVectorizer):
    """
    Simple TF–IDF vectorizer with fixed max_features
    so the embedding dimension is known in advance.
    """

    def __init__(self, max_features: int = 512):
        self._max_features = max_features
        self._vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words="english"  # optional, tweak as needed
        )
        self._fitted = False

    def fit(self, texts: List[str]) -> None:
        self._vectorizer.fit(texts)
        self._fitted = True

    def transform(self, texts: List[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Vectorizer must be fitted before transform()")
        # Returns a sparse matrix – convert to dense for pgvector
        sparse = self._vectorizer.transform(texts)
        return sparse.toarray().astype("float32")

    @property
    def dim(self) -> int:
        return self._max_features


# Example of how you'd later swap in a different embedder:
#
# class SentenceTransformerVectorizer(BaseVectorizer):
#     def __init__(self, model_name="all-MiniLM-L6-v2"):
#         from sentence_transformers import SentenceTransformer
#         self.model = SentenceTransformer(model_name)
#         self._dim = self.model.get_sentence_embedding_dimension()
#     def fit(self, texts: List[str]) -> None:
#         pass  # not needed for pretrained models
#     def transform(self, texts: List[str]) -> np.ndarray:
#         return self.model.encode(texts, convert_to_numpy=True)
#     @property
#     def dim(self) -> int:
#         return self._dim
