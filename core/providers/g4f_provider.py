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
        Generates a response using G4F. Implements internal model fallback
        to increase reliability.
        """
        # We try gpt-4.1 as recommended in the documentation plus other staples
        models_to_try = [self.model, "gpt-4.1", "gpt-4o", "gpt-3.5-turbo"]
        
        last_error = None
        for model_name in models_to_try:
            try:
                logger.info(f"G4F: Attempting generation with model {model_name}")
                # We use a simple completions call. 
                # Note: g4f's Client doesn't have a built-in timeout in all versions, 
                # but we can try to pass it or rely on the thread-level timeout if we used one.
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                )
                
                content = response.choices[0].message.content
                if content:
                    return {
                        "text": content,
                        "provider": "g4f",
                        "model": model_name
                    }
                
            except Exception as e:
                last_error = str(e)
                logger.warning(f"G4F: Model {model_name} failed: {e}")
                continue

        return {"error": f"G4F Provider failed all models. Last error: {last_error}"}

    async def generate_answer_async(self, prompt: str) -> dict:
        """
        Async version of the generator.
        """
        return await asyncio.to_thread(self.generate_answer, prompt)
