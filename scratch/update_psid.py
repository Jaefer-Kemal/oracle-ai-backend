import os
from sqlalchemy import create_engine, text

DATABASE_URL = "postgresql://neondb_owner:npg_SW5jb6wlURAo@ep-sweet-flower-alolg6z2-pooler.c-3.eu-central-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"
PSID = "g.a0009Aj_Ay-EEGJbohsYvRH9Sy0f57jpWZnR6_871mRe23BsNmdoTMsju6St4yWJCbscz1FgOQACgYKARkSARISFQHGX2Mit4sZW4HSHBpsn7cBtd5a8xoVAUF8yKrzboI9cWJ-TBkBY8SA9Ytz0076"

def update():
    try:
        engine = create_engine(DATABASE_URL)
        with engine.connect() as conn:
            conn.execute(
                text("UPDATE app_settings SET value = :psid WHERE key = 'gemini_1psid'"),
                {"psid": PSID}
            )
            conn.commit()
            print("Successfully updated __Secure-1PSID in the database.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    update()
