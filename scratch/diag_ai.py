import os
import sys

# Add the parent directory to sys.path so we can import 'core'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session
from core.providers.factory import ProviderFactory

DATABASE_URL = "postgresql://neondb_owner:npg_SW5jb6wlURAo@ep-sweet-flower-alolg6z2-pooler.c-3.eu-central-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"

def test_providers():
    engine = create_engine(DATABASE_URL)
    prompt = "Hi"
    
    with Session(engine) as db:
        print("--- Testing Grok ---")
        try:
            res = ProviderFactory._call_provider(db, "grok", prompt)
            print(f"Grok Result: {res}")
        except Exception as e:
            print(f"Grok Crash: {e}")
            
        print("\n--- Testing Gemini ---")
        try:
            res = ProviderFactory._call_provider(db, "gemini", prompt)
            print(f"Gemini Result: {res}")
        except Exception as e:
            print(f"Gemini Crash: {e}")

if __name__ == "__main__":
    test_providers()
