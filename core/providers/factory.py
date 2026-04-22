import logging
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from .grok import Grok
from .gemini import GeminiProvider
from .g4f_provider import G4FProvider

logger = logging.getLogger("rag-backend")

class ProviderFactory:
    """
    Orchestrates multiple AI providers with dynamic fallback logic and instance caching.
    Includes smart re-initialization and memory-cached configuration (Zero-DB per request).
    """
    _instances: Dict[str, Any] = {}
    _config_signatures: Dict[str, str] = {}
    
    # In-memory config cache to avoid DB hits on every AI call
    _config_cache: Dict[str, Any] = {
        "active_provider": "gemini",
        "fallback_chain": ["gemini", "grok", "g4f"],
        "grok_model": "grok-3-auto"
    }

    @staticmethod
    def reload_config(db: Session):
        """Pre-fetches all relevant AI settings and reloads instances if necessary."""
        from models import AppSettings
        logger.info("Syncing AI Configuration Cache with Database...")
        
        settings = {s.key: s.value for s in db.query(AppSettings).all()}
        
        # Update Cache
        ProviderFactory._config_cache["active_provider"] = settings.get("active_provider", "gemini")
        ProviderFactory._config_cache["grok_model"] = settings.get("grok_model", "grok-3-auto")
        
        try:
            import json
            chain = json.loads(settings.get("fallback_chain", '["gemini", "grok", "g4f"]'))
            ProviderFactory._config_cache["fallback_chain"] = chain
        except:
            ProviderFactory._config_cache["fallback_chain"] = ["gemini", "grok", "g4f"]
        
        # Logic for immediate re-instantiation if critical settings changed
        # This occurs during the Admin Save event or Startup
        current_grok_model = ProviderFactory._config_cache["grok_model"]
        if "grok" in ProviderFactory._instances and ProviderFactory._config_signatures.get("grok") != current_grok_model:
            logger.info(f"Re-initializing Grok client for new model setting: {current_grok_model}")
            ProviderFactory._instances["grok"] = Grok(current_grok_model)
            ProviderFactory._config_signatures["grok"] = current_grok_model

    @staticmethod
    def generate_answer(db: Session, prompt: str, history: List[dict] = None, model_override: Optional[str] = None) -> dict:
        """
        Main entry point for AI generation. 
        Uses the in-memory CONFIG CACHE to determine provider and chain.
        """
        # Note: 'db' is kept in signature for compatibility and session logging, 
        # but we no longer query AppSettings here.
        
        active_provider = ProviderFactory._config_cache.get("active_provider", "gemini")
        fallback_chain = list(ProviderFactory._config_cache.get("fallback_chain", ["gemini", "grok", "g4f"]))

        # Ensure active_provider is first in the execution list
        if active_provider in fallback_chain:
            fallback_chain.remove(active_provider)
        fallback_chain.insert(0, active_provider)

        print(f"DEBUG: Active Provider: {active_provider}")
        print(f"DEBUG: Fallback Chain: {fallback_chain}")
        
        # Iterate through providers
        last_error = None
        for provider_name in fallback_chain:
            try:
                print(f"DEBUG: Attempting provider: {provider_name}")
                logger.info(f"Attempting generation with provider: {provider_name}")
                result = ProviderFactory._call_provider(provider_name, prompt, model_override)
                
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
    def _call_provider(name: str, prompt: str, model_override: Optional[str] = None) -> dict:
        """Internal router using cached instances."""
        
        if name == "grok":
            current_model = ProviderFactory._config_cache.get("grok_model", "grok-3-auto")
            
            # Initialization (if not already done by reload_config)
            if "grok" not in ProviderFactory._instances:
                logger.info(f"Lazily initializing Grok for: {current_model}")
                ProviderFactory._instances["grok"] = Grok(current_model)
                ProviderFactory._config_signatures["grok"] = current_model
            
            client = ProviderFactory._instances["grok"]
            return client.start_convo(prompt, model_override=model_override)
            
        elif name == "gemini":
            if "gemini" not in ProviderFactory._instances:
                ProviderFactory._instances["gemini"] = GeminiProvider()
            
            client = ProviderFactory._instances["gemini"]
            return client.generate_answer(prompt, model_override)
            
        elif name == "g4f":
            if "g4f" not in ProviderFactory._instances:
                ProviderFactory._instances["g4f"] = G4FProvider()
                
            client = ProviderFactory._instances["g4f"]
            return client.generate_answer(prompt)
            
        return {"error": f"Provider {name} not implemented."}
