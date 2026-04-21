import os
import logging
from google import genai
from typing import Optional, List, Dict, Any

logger = logging.getLogger("rag-backend.gemini")

class GeminiProvider:
    """
    Official Google Gemini API Provider (v2).
    Uses the modern 'google-genai' SDK.
    """
    def __init__(self, api_key: Optional[str] = None) -> None:
        # Prioritize passed key, then .env
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self._client = None
        
        if not self.api_key:
            logger.warning("Gemini API Key missing. Service will be unavailable.")
        else:
            try:
                # Initialize the new SDK client
                self._client = genai.Client(api_key=self.api_key)
                logger.info("Gemini SDK initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini SDK: {e}")
                self._client = None

    def generate_answer(self, prompt: str, model: str = "gemini-2.0-flash") -> Dict[str, Any]:
        """
        Generates a response using the modern google-genai SDK.
        """
        if not self._client:
            return {"error": "Gemini API Client not initialized (check API Key)."}

        try:
            # Using generate_content from the new SDK
            response = self._client.models.generate_content(
                model=model,
                contents=prompt
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
            logger.error(f"Gemini SDK Error: {e}")
            return {"error": str(e)}

    # Compatibility method for async calling
    async def generate_answer_async(self, prompt: str, model: str = "gemini-2.0-flash") -> Dict[str, Any]:
        # The new SDK is sync by default but supports direct calling 
        return self.generate_answer(prompt, model)
