import sys
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.providers.factory import ProviderFactory

DATABASE_URL = "postgresql://neondb_owner:npg_SW5jb6wlURAo@ep-sweet-flower-alolg6z2-pooler.c-3.eu-central-1.aws.neon.tech/neondb?sslmode=require"

def final_verify():
    engine = create_engine(DATABASE_URL)
    with Session(engine) as db:
        print("Verifying Active Provider from Factory...")
        try:
            import sys
            # This should try the active provider and fallback automatically
            response = ProviderFactory.generate_answer(db, "Hi")
            print(f"Factory Response: {response}".encode(sys.stdout.encoding, errors='replace').decode(sys.stdout.encoding))
        except Exception as e:
            print(f"Factory Failed: {e}")

if __name__ == "__main__":
    final_verify()
