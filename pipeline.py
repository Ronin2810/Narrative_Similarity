from collections import defaultdict
from typing import Dict, List

import numpy as np

from config import DATA_DIR, EMBED_DIM
from pdf_reader import iter_documents, Document
from vectorizer import TfidfVectorizerWrapper, BaseVectorizer
from db import VectorStore


def build_corpus() -> List[Document]:
    """
    Read all PDFs under data/{TICKER} and return a flat list of Documents.
    """
    return list(iter_documents(DATA_DIR))


def fit_vectorizer(docs: List[Document]) -> BaseVectorizer:
    """
    Fit the chosen vectorizer on the full corpus.
    """
    texts = [d.text for d in docs]
    vec = TfidfVectorizerWrapper(max_features=EMBED_DIM)
    vec.fit(texts)
    return vec


def embed_documents(vec: BaseVectorizer, docs: List[Document]) -> np.ndarray:
    """
    Transform all documents into embeddings.
    """
    texts = [d.text for d in docs]
    embeddings = vec.transform(texts)
    return embeddings


def store_embeddings(
    store: VectorStore,
    docs: List[Document],
    embeddings: np.ndarray,
) -> None:
    """
    Persist all embeddings to the vector DB, grouped by ticker for convenience.
    """
    # Group by ticker so insertion calls can be batched per ticker
    ticker_to_indices: Dict[str, List[int]] = defaultdict(list)
    for idx, doc in enumerate(docs):
        ticker_to_indices[doc.ticker].append(idx)

    for ticker, indices in ticker_to_indices.items():
        sub_docs = [docs[i] for i in indices]
        doc_ids = [d.doc_id for d in sub_docs]
        contents = [d.text for d in sub_docs]
        sub_embeddings = embeddings[indices, :]
        store.insert_documents(
            ticker=ticker,
            doc_ids=doc_ids,
            contents=contents,
            embeddings=sub_embeddings,
        )
