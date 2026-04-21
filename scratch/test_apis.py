import requests
import json

BASE_URL = "http://localhost:8000"  # Assuming dev server is running or we test locally

def test_config():
    print("Testing GET /api/config...")
    try:
        # We need a token if it's protected, but let's assume we can check if it responds
        r = requests.get(f"{BASE_URL}/api/config")
        print(f"Status: {r.status_code}")
        if r.status_code == 200:
            print(json.dumps(r.json(), indent=2))
        else:
            print(r.text)
    except Exception as e:
        print(f"Failed: {e}")

def test_chat():
    print("\nTesting POST /api/chat (Hi)...")
    payload = {
        "query": "Hi",
        "session_id": "test-session-123",
        "stream": False
    }
    try:
        r = requests.post(f"{BASE_URL}/api/chat", json=payload)
        print(f"Status: {r.status_code}")
        if r.status_code == 200:
            print(r.text[:200] + "...")
        else:
            print(r.text)
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    # Note: These tests require the server to be running.
    # Alternatively, we can use TestClient if we import the FastAPI app.
    print("--- API Health Check ---")
    test_config()
    test_chat()
