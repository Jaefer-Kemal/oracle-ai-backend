import os
import logging
from google import genai
from typing import Optional, List, Dict, Any

logger = logging.getLogger("rag-backend.gemini")

class GeminiProvider:
    """
    Official Google Gemini API Provider (v2).
    Uses the modern 'google-genai' SDK with Singleton optimization.
    """
    _shared_client: Optional[genai.Client] = None

    def __init__(self, api_key: Optional[str] = None) -> None:
        # Prioritize passed key, then .env
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        
        if not GeminiProvider._shared_client:
            if not self.api_key:
                logger.warning("Gemini API Key missing. Service will be unavailable.")
            else:
                try:
                    # Initialize the client once globally to save latency
                    GeminiProvider._shared_client = genai.Client(api_key=self.api_key)
                    logger.info("Gemini SDK Singleton instance created successfully.")
                except Exception as e:
                    logger.error(f"Failed to initialize Gemini SDK: {e}")
                    GeminiProvider._shared_client = None

    def generate_answer(self, prompt: str, model_override: Optional[str] = None) -> Dict[str, Any]:
        """
        Generates a response using the modern google-genai SDK.
        Default model: gemini-3-flash
        """
        if not GeminiProvider._shared_client:
            return {"error": "Gemini API Client not initialized (check API Key)."}

        # Use the faster flash-lite or 2.0/2.5 flash for sub-tasks (2026 Standard)
        model = model_override or "gemini-flash-lite-latest"

        try:
            # Using generate_content from the shared client with AFC disabled to reduce overhead
            from google.genai import types
            
            response = GeminiProvider._shared_client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
                )
            )
            
            # Extract text (handling potential candidates)
            text = response.text if response.text else ""
            
            return {
                "response": text,
                "stream_response": [],
                "images": [], 
                "metadata": {
                    "provider": "gemini",
                    "model": model,
                    "usage": {
                        "prompt_token_count": getattr(response.usage_metadata, 'prompt_token_count', 0),
                        "candidates_token_count": getattr(response.usage_metadata, 'candidates_token_count', 0),
                        "total_token_count": getattr(response.usage_metadata, 'total_token_count', 0)
                    }
                }
            }
        except Exception as e:
            logger.error(f"Gemini SDK Error ({model}): {e}")
            return {"error": str(e)}

    # Compatibility method for async calling
    async def generate_answer_async(self, prompt: str, model_override: Optional[str] = None) -> Dict[str, Any]:
        return self.generate_answer(prompt, model_override)
