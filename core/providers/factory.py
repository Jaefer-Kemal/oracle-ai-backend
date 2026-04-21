import logging
from typing import List, Optional
from sqlalchemy.orm import Session
from .grok import Grok
from .gemini import GeminiProvider
from .g4f_provider import G4FProvider

logger = logging.getLogger("rag-backend")

class ProviderFactory:
    """
    Orchestrates multiple AI providers with dynamic fallback logic.
    """
    @staticmethod
    def generate_answer(db: Session, prompt: str, history: List[dict] = None) -> dict:
        from models import AppSettings
        
        # 1. Fetch Configuration
        active_provider_setting = db.query(AppSettings).filter(AppSettings.key == "active_provider").first()
        fallback_chain_setting = db.query(AppSettings).filter(AppSettings.key == "fallback_chain").first()
        
        active_provider = active_provider_setting.value if active_provider_setting else "grok"
        
        # Fallback chain is stored as a JSON string or comma-separated list
        try:
            import json
            fallback_chain = json.loads(fallback_chain_setting.value) if fallback_chain_setting else ["grok", "gemini"]
        except:
            fallback_chain = ["grok", "gemini"]

        # Ensure active_provider is first in the execution list if not already
        if active_provider in fallback_chain:
            fallback_chain.remove(active_provider)
        fallback_chain.insert(0, active_provider)

        print(f"DEBUG: Active Provider: {active_provider}")
        print(f"DEBUG: Fallback Chain: {fallback_chain}")
        
        # 2. Iterate through providers
        last_error = None
        for provider_name in fallback_chain:
            try:
                print(f"DEBUG: Attempting provider: {provider_name}")
                logger.info(f"Attempting generation with provider: {provider_name}")
                result = ProviderFactory._call_provider(db, provider_name, prompt)
                
                if result and "error" not in result:
                    print(f"DEBUG: Provider {provider_name} SUCCEEDED")
                    return result
                
                last_error = result.get("error", "Unknown error")
                logger.warning(f"Provider {provider_name} failed: {last_error}")
            except Exception as e:
                last_error = str(e)
                logger.error(f"Critical failure in provider {provider_name}: {e}")

        return {"error": f"All providers failed. Last error: {last_error}"}

    @staticmethod
    def _call_provider(db: Session, name: str, prompt: str) -> dict:
        from models import AppSettings
        
        if name == "grok":
            model_setting = db.query(AppSettings).filter(AppSettings.key == "grok_model").first()
            model = model_setting.value if model_setting else "grok-3-auto"
            client = Grok(model)
            return client.start_convo(prompt)
            
        elif name == "gemini":
            # Gemini now uses the official API via GOOGLE_API_KEY in .env
            client = GeminiProvider()
            return client.generate_answer(prompt)
            
        elif name == "g4f":
            client = G4FProvider()
            return client.generate_answer(prompt)
            
        # Add others here...
        
        return {"error": f"Provider {name} not implemented."}
