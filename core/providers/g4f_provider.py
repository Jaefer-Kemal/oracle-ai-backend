import logging
import asyncio
from typing import Optional
import g4f
from g4f.client import Client

logger = logging.getLogger("rag-backend.g4f")

class G4FProvider:
    """
    A robust wrapper for the g4f (GPT4Free) library, providing a 
    third-tier fallback engine for the RAG pipeline.
    """
    
    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        self.client = Client()

    def generate_answer(self, prompt: str) -> dict:
        """
        Generates a response using G4F with a hardcoded high-reliability fallback chain.
        We prioritize zero-auth providers verified in the current environment.
        """
        from g4f.Provider import PollinationsAI, DDGS
        
        # verified stable (provider, model) pairs
        stable_pairs = [
            (PollinationsAI, "openai"),
            (DDGS, "gpt-4o-mini"),
            (None, "gpt-4o-mini"), # Let g4f auto-select if specific ones fail
            (None, "gpt-3.5-turbo")
        ]
        
        last_error = None
        for provider, model_name in stable_pairs:
            try:
                p_name = provider.__name__ if provider else "Auto"
                logger.info(f"G4F: Attempting generation with provider {p_name} and model {model_name}")
                
                # Using the direct Completion API for maximum control over provider selection
                response = g4f.ChatCompletion.create(
                    model=model_name,
                    provider=provider,
                    messages=[{"role": "user", "content": prompt}],
                )
                
                if response and isinstance(response, str):
                    return {
                        "text": response,
                        "provider": f"g4f:{p_name}",
                        "model": model_name
                    }
                
            except Exception as e:
                last_error = str(e)
                logger.warning(f"G4F: Provider {p_name} failed: {e}")
                continue

        return {"error": f"G4F Provider failed all models. Last error: {last_error}"}

    async def generate_answer_async(self, prompt: str) -> dict:
        """
        Async version of the generator.
        """
        return await asyncio.to_thread(self.generate_answer, prompt)
