from typing import Iterable

import numpy as np
import psycopg2
from pgvector.psycopg2 import register_vector

from config import DB_DSN


class VectorStore:
    """
    Thin wrapper around Postgres with pgvector.

    Assumes a single embeddings table:
      document_embeddings(
        id        serial primary key,
        ticker    text not null,
        doc_id    text not null,
        content   text not null,
        embedding vector(dim) not null
      )
    """

    def __init__(self, embed_dim: int):
        self.embed_dim = embed_dim
        self._conn = psycopg2.connect(DB_DSN)
        register_vector(self._conn)

    def close(self) -> None:
        self._conn.close()

    def init_schema(self) -> None:
        """Create extension + table if not present."""
        with self._conn, self._conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            # embed_dim must be interpolated into SQL (not a parameter)
            create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS document_embeddings (
                id        SERIAL PRIMARY KEY,
                ticker    TEXT NOT NULL,
                doc_id    TEXT NOT NULL,
                content   TEXT NOT NULL,
                embedding VECTOR({self.embed_dim}) NOT NULL
            );
            """
            cur.execute(create_table_sql)

            # Optional: index for similarity search later
            # (you can uncomment when ready)
            # cur.execute("""
            #   CREATE INDEX IF NOT EXISTS idx_document_embeddings_embedding
            #   ON document_embeddings
            #   USING ivfflat (embedding vector_cosine_ops)
            #   WITH (lists = 100);
            # """)

    def insert_documents(
        self,
        ticker: str,
        doc_ids: Iterable[str],
        contents: Iterable[str],
        embeddings: np.ndarray,
    ) -> None:
        """
        Bulk insert a batch of document embeddings.

        embeddings: shape (n_docs, embed_dim)
        """
        if embeddings.ndim != 2 or embeddings.shape[1] != self.embed_dim:
            raise ValueError(
                f"Expected embeddings shape (N, {self.embed_dim}), "
                f"got {embeddings.shape}"
            )

        with self._conn, self._conn.cursor() as cur:
            for doc_id, text, emb in zip(doc_ids, contents, embeddings):
                # emb is a 1D numpy array; pgvector adapter accepts list-like
                cur.execute(
                    """
                    INSERT INTO document_embeddings (ticker, doc_id, content, embedding)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (ticker, doc_id, text, emb.tolist()),
                )
