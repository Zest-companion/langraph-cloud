"""
Module de logging centralisé avec intégration Supabase
"""
import os
import json
import logging
import traceback
from datetime import datetime
from typing import Dict, Any, Optional
from functools import wraps
import time

from supabase import create_client, Client

# Configuration Supabase
supabase_client: Optional[Client] = None
try:
    supabase_client = create_client(
        os.getenv("SUPABASE_URL"),
        os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY")
    )
except:
    pass  # Fallback sur logging local si Supabase non disponible

# Configuration du logger Python standard
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class ZestLogger:
    """Logger personnalisé avec envoi vers Supabase"""
    
    def __init__(self, module_name: str):
        self.module_name = module_name
        self.logger = logging.getLogger(module_name)
        self._context = {}
    
    def set_context(self, **kwargs):
        """Définir le contexte de session (thread_id, user_id, etc.)"""
        self._context.update(kwargs)
    
    def _log_to_supabase(self, level: str, message: str, **kwargs):
        """Envoyer le log vers Supabase"""
        if not supabase_client:
            return
        
        try:
            log_entry = {
                "level": level,
                "module": self.module_name,
                "message": message[:1000],  # Limiter la taille
                "thread_id": self._context.get("thread_id"),
                "user_id": self._context.get("user_id"),
                "user_name": self._context.get("user_name"),
                "cohort": self._context.get("cohort"),
                "workflow_type": kwargs.get("workflow_type"),
                "node_name": kwargs.get("node_name"),
                "intent_type": kwargs.get("intent_type"),
                "execution_time_ms": kwargs.get("execution_time_ms"),
                "tokens_used": kwargs.get("tokens_used"),
                "metadata": kwargs.get("metadata", {}),
                "error_details": kwargs.get("error_details")
            }
            
            # Nettoyer les valeurs None
            log_entry = {k: v for k, v in log_entry.items() if v is not None}
            
            # Envoi asynchrone pour ne pas bloquer
            supabase_client.table("application_logs").insert(log_entry).execute()
        except Exception as e:
            # Fallback silencieux si l'envoi échoue
            self.logger.error(f"Failed to log to Supabase: {e}")
    
    def debug(self, message: str, **kwargs):
        self.logger.debug(message)
        if os.getenv("LOG_LEVEL", "INFO") == "DEBUG":
            self._log_to_supabase("DEBUG", message, **kwargs)
    
    def info(self, message: str, **kwargs):
        self.logger.info(message)
        self._log_to_supabase("INFO", message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        self.logger.warning(message)
        self._log_to_supabase("WARNING", message, **kwargs)
    
    def error(self, message: str, exception: Optional[Exception] = None, **kwargs):
        self.logger.error(message)
        error_details = None
        if exception:
            error_details = {
                "type": type(exception).__name__,
                "message": str(exception),
                "traceback": traceback.format_exc()
            }
        self._log_to_supabase("ERROR", message, error_details=error_details, **kwargs)
    
    def critical(self, message: str, **kwargs):
        self.logger.critical(message)
        self._log_to_supabase("CRITICAL", message, **kwargs)
    
    def log_performance(self, operation: str, execution_time_ms: int, tokens: int = None):
        """Logger spécifique pour les métriques de performance"""
        message = f"Performance: {operation} took {execution_time_ms}ms"
        if tokens:
            message += f" ({tokens} tokens)"
        self.info(message, execution_time_ms=execution_time_ms, tokens_used=tokens)

def track_usage(workflow_type: str):
    """Décorateur pour tracker l'usage des workflows"""
    def decorator(func):
        @wraps(func)
        def wrapper(state: Dict[str, Any]):
            start_time = time.time()
            logger = ZestLogger(f"workflow.{workflow_type}")
            
            # Contexte de session
            logger.set_context(
                thread_id=state.get("thread_id"),
                user_id=state.get("user_id"),
                user_name=state.get("user_name"),
                cohort=state.get("cohort")
            )
            
            try:
                # Exécution
                result = func(state)
                
                # Calcul des métriques
                execution_time = int((time.time() - start_time) * 1000)
                
                # Log de succès
                logger.info(
                    f"Workflow {workflow_type} completed",
                    workflow_type=workflow_type,
                    execution_time_ms=execution_time,
                    node_name=func.__name__
                )
                
                # Analytics d'usage (si disponible)
                if supabase_client and state.get("user_id"):
                    try:
                        analytics = {
                            "thread_id": state.get("thread_id"),
                            "user_id": state.get("user_id"),
                            "workflow_type": workflow_type,
                            "sub_theme": state.get("sub_theme"),
                            "response_time_ms": execution_time,
                            "success": True,
                            "tools_used": state.get("tools_used", []),
                            "intents_detected": state.get("intents_detected", [])
                        }
                        supabase_client.table("usage_analytics").insert(analytics).execute()
                    except:
                        pass
                
                return result
                
            except Exception as e:
                # Log d'erreur
                logger.error(
                    f"Workflow {workflow_type} failed",
                    exception=e,
                    workflow_type=workflow_type,
                    node_name=func.__name__
                )
                raise
        
        return wrapper
    return decorator

# Instance globale pour les logs système
system_logger = ZestLogger("system")