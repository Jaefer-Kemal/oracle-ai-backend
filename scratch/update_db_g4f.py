from sqlalchemy import create_engine, text
import json

DATABASE_URL = "postgresql://neondb_owner:npg_SW5jb6wlURAo@ep-sweet-flower-alolg6z2-pooler.c-3.eu-central-1.aws.neon.tech/neondb?sslmode=require"

def update_db():
    engine = create_engine(DATABASE_URL)
    with engine.connect() as conn:
        chain = json.dumps(["grok", "gemini", "g4f"])
        query = text("UPDATE app_settings SET value = :val WHERE key = 'fallback_chain'")
        conn.execute(query, {"val": chain})
        conn.commit()
        print("Updated fallback_chain successfully.")

if __name__ == "__main__":
    update_db()
