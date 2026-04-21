import g4f
from g4f.Provider import Blackbox, DuckDuckGo, FreeGpt, Liaobots, You

providers = [Blackbox, DuckDuckGo, FreeGpt, Liaobots, You]
working = []

print("--- G4F Provider Benchmark ---")
for p in providers:
    try:
        print(f"Testing {p.__name__}...")
        response = g4f.ChatCompletion.create(
            model='gpt-4o', 
            provider=p, 
            messages=[{'role': 'user', 'content': 'hi'}],
            timeout=10
        )
        if response:
            print(f"  [+] {p.__name__} works!")
            working.append(p.__name__)
    except Exception as e:
        print(f"  [-] {p.__name__} failed: {e}")

print(f"\nFinal Working List: {working}")
