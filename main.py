from config import EMBED_DIM
from pipeline import build_corpus, fit_vectorizer, embed_documents, store_embeddings
from db import VectorStore


def main():
    # 1. Load PDFs
    print("Building corpus from data/{TICKER} directories...")
    docs = build_corpus()
    print(f"Found {len(docs)} documents.")

    if not docs:
        print("No documents found. Exiting.")
        return

    # 2. Fit vectorizer
    print("Fitting vectorizer...")
    vec = fit_vectorizer(docs)
    print(f"Vectorizer dimension: {vec.dim}")

    # 3. Create embeddings
    print("Embedding documents...")
    embeddings = embed_documents(vec, docs)
    print(f"Embeddings shape: {embeddings.shape}")

    # 4. Initialize vector DB
    store = VectorStore(embed_dim=vec.dim)
    try:
        print("Initializing DB schema...")
        store.init_schema()

        # 5. Store embeddings
        print("Storing embeddings in Postgres...")
        store_embeddings(store, docs, embeddings)
    finally:
        store.close()

    print("Done.")


if __name__ == "__main__":
    main()
