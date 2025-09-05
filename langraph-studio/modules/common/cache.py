"""
Module de gestion du cache pour optimiser les performances
"""
import json
import hashlib
from datetime import datetime, timedelta
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class SimpleCache:
    """Cache en mémoire simple avec expiration"""
    
    def __init__(self, ttl_minutes: int = 60):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.ttl = timedelta(minutes=ttl_minutes)
    
    def _get_key(self, key_data: Any) -> str:
        """Génère une clé unique basée sur les données"""
        if isinstance(key_data, dict):
            key_str = json.dumps(key_data, sort_keys=True)
        else:
            key_str = str(key_data)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, key: Any) -> Optional[Any]:
        """Récupère une valeur du cache si elle existe et n'est pas expirée"""
        cache_key = self._get_key(key)
        
        if cache_key in self.cache:
            entry = self.cache[cache_key]
            if datetime.now() < entry['expires_at']:
                logger.debug(f"🎯 Cache hit for key: {cache_key[:8]}...")
                return entry['value']
            else:
                # Nettoyer l'entrée expirée
                del self.cache[cache_key]
                logger.debug(f"🕒 Cache expired for key: {cache_key[:8]}...")
        
        logger.debug(f"❌ Cache miss for key: {cache_key[:8]}...")
        return None
    
    def set(self, key: Any, value: Any) -> None:
        """Stocke une valeur dans le cache avec expiration"""
        cache_key = self._get_key(key)
        self.cache[cache_key] = {
            'value': value,
            'expires_at': datetime.now() + self.ttl,
            'created_at': datetime.now()
        }
        logger.debug(f"💾 Cached value for key: {cache_key[:8]}...")
    
    def clear(self) -> None:
        """Vide complètement le cache"""
        self.cache.clear()
        logger.info("🗑️ Cache cleared")
    
    def cleanup_expired(self) -> None:
        """Nettoie les entrées expirées"""
        now = datetime.now()
        expired_keys = [
            key for key, entry in self.cache.items() 
            if now >= entry['expires_at']
        ]
        for key in expired_keys:
            del self.cache[key]
        if expired_keys:
            logger.debug(f"🧹 Cleaned {len(expired_keys)} expired cache entries")

# Instances globales de cache pour différents usages
profile_cache = SimpleCache(ttl_minutes=120)  # Cache des profils utilisateurs (2h)
vector_cache = SimpleCache(ttl_minutes=60)    # Cache des recherches vectorielles (1h)
llm_cache = SimpleCache(ttl_minutes=30)       # Cache des réponses LLM (30min)