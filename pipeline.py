from collections import defaultdict
from typing import Dict, List
import numpy as np

from config import DATA_DIR, TFIDF_FALLBACK_DIM
from pdf_reader import iter_documents, Document
from vectorizer import get_vectorizer, BaseVectorizer   # <-- factory with fallback
from db import VectorStore

def build_corpus() -> List[Document]:
    """Read all PDFs under data/{TICKER} and return a flat list of Documents."""
    return list(iter_documents(DATA_DIR))

def fit_vectorizer(docs: List[Document]) -> BaseVectorizer:
    """Instantiate and (if needed) fit the chosen vectorizer."""
    texts = [d.text for d in docs]
    vec = get_vectorizer(prefer_finance=True, tfidf_dim=TFIDF_FALLBACK_DIM)
    # vec = get_vectorizer(prefer_finance=True, tfidf_dim=TFIDF_FALLBACK_DIM, finance_model='philschmid/bge-base-financial-matryoshka')
    vec.fit(texts)  # no-op for finance; trains vocab for TF–IDF
    return vec

def embed_documents(vec: BaseVectorizer, docs: List[Document]) -> np.ndarray:
    """Transform all documents into embeddings."""
    texts = [d.text for d in docs]
    return vec.transform(texts)

def store_embeddings(store: VectorStore, docs: List[Document], embeddings: np.ndarray) -> None:
    """Persist all embeddings to the vector DB, grouped by ticker for convenience."""
    ticker_to_indices: Dict[str, List[int]] = defaultdict(list)
    for idx, doc in enumerate(docs):
        ticker_to_indices[doc.ticker].append(idx)

    for ticker, indices in ticker_to_indices.items():
        sub_docs = [docs[i] for i in indices]
        doc_ids = [d.doc_id for d in sub_docs]
        contents = [d.text for d in sub_docs]
        periods = [d.period for d in sub_docs]  # assuming Document has 'period' attribute
        
        sub_embeddings = embeddings[indices, :]
        store.insert_documents(
            ticker=ticker,
            doc_ids=doc_ids,
            contents=contents,
            period=periods,
            embeddings=sub_embeddings,
        )
    
    # #remove all duplicate doc_ids
    # store.remove_duplicate_doc_ids()
