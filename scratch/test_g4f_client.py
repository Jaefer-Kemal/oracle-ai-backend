import g4f
from g4f.client import Client

client = Client()

print("--- Testing G4F Client with Models from Docs ---")
models = ["gpt-4o", "gpt-4.1", "deepseek-v3"]

for m in models:
    try:
        print(f"Testing model: {m}")
        response = client.chat.completions.create(
            model=m,
            messages=[{"role": "user", "content": "Say 'G4F Success'"}],
            web_search=False
        )
        content = response.choices[0].message.content
        print(f"  [+] Success: {content}")
    except Exception as e:
        print(f"  [-] Failed: {e}")
