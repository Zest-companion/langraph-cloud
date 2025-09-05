"""
Utilitaires de logging pour les workflows
"""
from app.modules.common.config import log_to_supabase
import time
from functools import wraps

def log_workflow_step(state, message, level="INFO"):
    """
    Logger une étape de workflow avec contexte
    """
    log_to_supabase(
        message=message,
        level=level,
        thread_id=state.get("thread_id"),
        user_id=state.get("user_id"),
        workflow_type=state.get("main_theme"),
        user_name=state.get("user_name"),
        cohort=state.get("cohort")
    )

def log_node_execution(node_name):
    """
    Décorateur pour logger l'exécution des nœuds
    """
    def decorator(func):
        @wraps(func)
        def wrapper(state):
            start_time = time.time()
            
            # Log début
            log_to_supabase(
                message=f"Started {node_name}",
                level="INFO",
                thread_id=state.get("thread_id"),
                user_id=state.get("user_id"),
                workflow_type=state.get("main_theme"),
                node_name=node_name
            )
            
            try:
                # Exécution
                result = func(state)
                
                # Log succès avec timing
                execution_time = int((time.time() - start_time) * 1000)
                log_to_supabase(
                    message=f"Completed {node_name}",
                    level="INFO",
                    thread_id=state.get("thread_id"),
                    user_id=state.get("user_id"),
                    workflow_type=state.get("main_theme"),
                    node_name=node_name,
                    execution_time_ms=execution_time
                )
                
                return result
                
            except Exception as e:
                # Log erreur
                log_to_supabase(
                    message=f"Failed {node_name}: {str(e)}",
                    level="ERROR",
                    thread_id=state.get("thread_id"),
                    user_id=state.get("user_id"),
                    workflow_type=state.get("main_theme"),
                    node_name=node_name,
                    metadata={"error": str(e), "error_type": type(e).__name__}
                )
                raise
        
        return wrapper
    return decorator

def log_vector_search(query, results_count, table_name, state):
    """
    Logger les recherches vectorielles
    """
    log_to_supabase(
        message=f"Vector search: {results_count} results from {table_name}",
        level="INFO",
        thread_id=state.get("thread_id"),
        user_id=state.get("user_id"),
        workflow_type=state.get("main_theme"),
        node_name="vector_search",
        metadata={
            "query": query[:100],
            "results_count": results_count,
            "table_name": table_name
        }
    )

def log_intent_detection(intent_type, confidence, state):
    """
    Logger les détections d'intent
    """
    log_to_supabase(
        message=f"Intent detected: {intent_type}",
        level="INFO",
        thread_id=state.get("thread_id"),
        user_id=state.get("user_id"),
        workflow_type=state.get("main_theme"),
        node_name="intent_detection",
        metadata={
            "intent_type": intent_type,
            "confidence": confidence
        }
    )

def log_llm_usage(model, prompt_tokens, completion_tokens, state):
    """
    Logger l'usage des LLM pour suivi des coûts
    """
    total_tokens = prompt_tokens + completion_tokens
    
    log_to_supabase(
        message=f"LLM call: {model} ({total_tokens} tokens)",
        level="INFO",
        thread_id=state.get("thread_id"),
        user_id=state.get("user_id"),
        workflow_type=state.get("main_theme"),
        node_name="llm_call",
        tokens_used=total_tokens,
        metadata={
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens
        }
    )