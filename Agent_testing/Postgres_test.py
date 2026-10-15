import psycopg

conn = psycopg.connect(
    host="localhost",
    port=5432,
    dbname="Test_db",
    user="postgres",
    password="Postgres@11"
)

try:
    with conn.cursor() as cur:
        cur.execute("SELECT version(), current_database();")
        print(cur.fetchone())
    print("PostgreSQL connected successfully!")
finally:
    conn.close()