import sys
import os
import logging

# Add backend to path
sys.path.append(os.path.join(os.getcwd(), ".."))

from core.providers.grok import Grok
from core.providers.gemini import GeminiProvider
from core.providers.g4f_provider import G4FProvider
from sqlalchemy.orm import Session
from models import SessionLocal

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test-providers")

def test_pipeline():
    db = SessionLocal()
    prompt = "Reply with exactly one word: 'Success'"

    print("\n" + "="*50)
    print("AI PROVIDER PIPELINE VERIFICATION")
    print("="*50)

    # 1. Test Gemini 3 Flash
    print("\n[1/3] Testing Gemini 3 Flash...")
    try:
        gemini = GeminiProvider()
        res = gemini.generate_answer(prompt)
        if "error" in res:
            print(f"[FAIL] Gemini Error: {res['error']}")
        else:
            print(f"[OK] Gemini Response: {res['response'][:50]}... (Model: {res['metadata']['model']})")
    except Exception as e:
        print(f"[CRITICAL] Gemini Failure: {e}")

    # 2. Test G4F (OpenAI)
    print("\n[2/3] Testing G4F (OpenAI)...")
    try:
        g4f_p = G4FProvider()
        res = g4f_p.generate_answer(prompt)
        if "error" in res:
            print(f"[FAIL] G4F Error: {res['error']}")
        else:
            print(f"[OK] G4F Response: {res['response'][:50]}... (Provider: {res.get('provider')})")
    except Exception as e:
        print(f"[CRITICAL] G4F Failure: {e}")

    # 3. Test Grok
    print("\n[3/3] Testing Grok (Primary)...")
    try:
        grok = Grok("grok-3-auto")
        res = grok.start_convo(prompt)
        if "error" in res:
            print(f"[FAIL] Grok Error: {res.get('error')}")
        else:
            print(f"[OK] Grok Response: {res.get('response', '')[:50]}...")
    except Exception as e:
        print(f"[CRITICAL] Grok Failure: {e}")

    print("\n" + "="*50)
    db.close()

if __name__ == "__main__":
    test_pipeline()
