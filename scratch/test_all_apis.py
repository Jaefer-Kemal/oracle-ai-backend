import sys
import os
from fastapi.testclient import TestClient

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app
from core.providers.factory import ProviderFactory

import httpx

def test_api_config():
    print("--- Testing API: GET /api/config ---")
    transport = httpx.ASGITransport(app=app)
    with httpx.Client(transport=transport, base_url="http://test") as client:
        response = client.get("/api/config")
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Active Provider: {data.get('active_provider')}")
            return True
    return False

def test_api_stats():
    print("\n--- Testing API: GET /api/stats ---")
    with httpx.Client(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = client.get("/api/stats")
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Total Docs: {data.get('total_docs')}")
        return True
    return False

def test_api_chat():
    print("\n--- Testing API: POST /api/chat ---")
    # Testing with a simple query
    payload = {
        "query": "Hi",
        "session_id": "test-verify-123",
        "stream": False
    }
    response = client.post("/api/chat", json=payload)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        print("Response received successfully")
        return True
    else:
        print(f"Error: {response.text}")
    return False

def run_all():
    results = {
        "config": test_api_config(),
        "stats": test_api_stats(),
        "chat": test_api_chat()
    }
    
    print("\n" + "="*30)
    print("FINAL API TEST REPORT")
    print("="*30)
    for k, v in results.items():
        print(f"{k.upper():10}: {'PASS' if v else 'FAIL'}")

if __name__ == "__main__":
    run_all()
