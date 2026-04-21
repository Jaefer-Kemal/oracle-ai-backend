import asyncio
import logging
from typing import Optional, List, Union
from pathlib import Path
from gemini_webapi import GeminiClient as WebGeminiClient

logger = logging.getLogger("rag-backend")

class GeminiProvider:
    """
    Modular Gemini Provider for Oracle AI.
    Handles session initialization and content generation.
    """
    def __init__(self, secure_1psid: str, secure_1psidts: str, proxy: str | None = None) -> None:
        self.secure_1psid = secure_1psid
        self.secure_1psidts = secure_1psidts
        self.proxy = proxy
        self._client = None

    async def _get_client(self):
        if not self._client:
            self._client = WebGeminiClient(self.secure_1psid, self.secure_1psidts, self.proxy)
            await self._client.init()
        return self._client

    async def generate_answer_async(self, prompt: str, model: str = "gemini-2.5-flash") -> dict:
        """
        Async implementation of the answer generation.
        """
        try:
            client = await self._get_client()
            response = await client.generate_content(prompt, model=model)
            
            return {
                "response": response.text,
                "stream_response": [], # gemini-webapi handles stream internally or returns full text
                "images": getattr(response, 'images', []),
                "metadata": {
                    "provider": "gemini",
                    "model": model
                }
            }
        except Exception as e:
            logger.error(f"Gemini Provider Error: {e}")
            return {"error": str(e)}

    def generate_answer(self, prompt: str, model: str = "gemini-2.5-flash") -> dict:
        """
        Synchronous wrapper for integration with existing services.
        """
        try:
            return asyncio.run(self.generate_answer_async(prompt, model))
        except Exception as e:
            return {"error": f"Async Bridge Error: {e}"}

    async def close(self) -> None:
        if self._client:
            await self._client.close()
            self._client = None
