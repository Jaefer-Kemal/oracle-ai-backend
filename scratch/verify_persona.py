import os
import sys
import logging
from dotenv import load_dotenv

load_dotenv()

# Add backend to path
sys.path.append(os.getcwd())

import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from models import SessionLocal, VectorEntry, Document
from services import GrokService, CohereService

logging.basicConfig(level=logging.INFO)

def test_enhanced_persona():
    db = SessionLocal()
    grok = GrokService()
    cohere = CohereService()
    
    def run_query(q, history=None):
        print(f"\n[USER]: {q}")
        # Search docs
        q_emb = cohere.get_embeddings([q], input_type="search_query")[0]
        results = db.query(VectorEntry).order_by(VectorEntry.embedding.cosine_distance(q_emb)).limit(3).all()
        context = [r.content for r in results]
        
        # Generate answer
        answer = grok.generate_answer(q, context, db, history=history)
        print(f"[AI]: {answer}")
        return answer

    print("--- TEST 1: Factual Retrieval (Oracle Docs) ---")
    ans1 = run_query("What services does Oracle AI offer?")
    
    print("\n--- TEST 2: Vague Follow-up (Context Continuity) ---")
    run_query("Tell me more", history=[{"q": "What services does Oracle AI offer?", "a": ans1}])
    
    print("\n--- TEST 3: Stylish Fallback (Unknown Topic) ---")
    run_query("What is your policy for office pets?")
    
    db.close()

if __name__ == "__main__":
    test_enhanced_persona()
