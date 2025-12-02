import numpy as np
import pandas as pd
import psycopg2
from pgvector.psycopg2 import register_vector

DB_DSN = "postgresql://postgres:postgres@localhost:5432/narrative_vectors"

EMB_NPY = "embeddings.npy"
META_CSV = "embeddings_meta.csv"


def main():
    print("Loading embeddings and metadata")
    embeddings = np.load(EMB_NPY)
    meta = pd.read_csv(META_CSV)

    if embeddings.shape[0] != len(meta):
        raise ValueError("Number of embeddings and meta rows do not match")

    print("Connecting to Postgres")
    conn = psycopg2.connect(DB_DSN)

    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                print("pgvector extension enabled.")

        register_vector(conn)
        print("Vector type registered with psycopg2.")

        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS document_embeddings (
                        id          serial PRIMARY KEY,
                        ticker      text,
                        year        int,
                        quarter     text,
                        period      text,
                        source_file text,
                        embedding   vector(768)
                    );
                    """
                )
                
                print("Inserting rows")
                for i, row in meta.iterrows():
                    emb = embeddings[i, :]
                    cur.execute(
                        """
                        INSERT INTO document_embeddings
                        (ticker, year, quarter, period, source_file, embedding)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        """,
                        (
                            row["ticker"],
                            int(row["year"]),
                            row["quarter"],
                            row["period"],
                            row["source_file"],
                            emb.tolist(),
                        ),
                    )
                print("Done inserting rows.")

    finally:
        conn.close()
        print("Connection closed.")

if __name__ == "__main__":
    main()