import numpy as np
import psycopg2
from pgvector.psycopg2 import register_vector

# If you already have config.py with DB_DSN, you can do:
# from config import DB_DSN
# and delete the hardcoded DSN below.

DB_DSN = "postgresql://postgres:postgres@localhost:5432/narrative_vectors"


def main():
    # 1. Connect to DB
    conn = psycopg2.connect(DB_DSN)
    register_vector(conn)  # enable pgvector type support

    try:
        with conn, conn.cursor() as cur:
            # 2. Fetch one row from document_embeddings
            cur.execute("""
                SELECT id, ticker, doc_id, embedding
                FROM document_embeddings
                ORDER BY id
                LIMIT 1;
            """)
            row = cur.fetchone()

            if row is None:
                print("No rows found in document_embeddings.")
                return

            row_id, ticker, doc_id, embedding = row

            # 3. Convert embedding (pgvector) to numpy array
            emb_array = np.array(embedding, dtype="float32")

            # 4. Print results
            print(f"Row ID:     {row_id}")
            print(f"Ticker:     {ticker}")
            print(f"Doc ID:     {doc_id}")
            print(f"Vector dim: {emb_array.shape[0]}")
            print("Vector (first 20 values):")
            print(emb_array[:20])

    finally:
        conn.close()


if __name__ == "__main__":
    main()
