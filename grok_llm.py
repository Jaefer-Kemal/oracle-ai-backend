import os
import sys
from typing import Any, List, Optional
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun

# Since we moved the 'core' folder to the project root, we can import directly.
# This is the cleanest way for both runtime and IDE static analysis.
from core import Grok

class GrokLLM(LLM):
    """Custom LangChain LLM wrapper around the Grok-Api (grok-3-auto by default)."""

    model_name: str = "grok-3-auto"

    @property
    def _llm_type(self) -> str:
        return "grok"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Send prompt to Grok and return the plain-text response."""
        grok_client = Grok(self.model_name)
        print(f"   [GrokLLM] Calling Grok ({self.model_name})...")

        response_data = grok_client.start_convo(prompt)
        text: str = response_data.get("response", "")

        # Honour stop sequences when provided by LangChain
        if stop:
            for stop_word in stop:
                if stop_word in text:
                    text = text[: text.index(stop_word)]

        return text
