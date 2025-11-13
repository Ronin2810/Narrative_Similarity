import psycopg2
from pgvector.psycopg2 import register_vector
from config import DB_DSN

def main():
    conn = psycopg2.connect(DB_DSN)
    register_vector(conn)
    with conn, conn.cursor() as cur:
        cur.execute("SELECT version();")
        print("Postgres version:", cur.fetchone()[0])

        cur.execute("SELECT current_database();")
        print("Database:", cur.fetchone()[0])

        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute("SELECT 'ok'::text;")
        print("pgvector extension ready.")

    conn.close()

if __name__ == "__main__":
    main()
