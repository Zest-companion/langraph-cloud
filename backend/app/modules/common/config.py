"""
Configuration commune pour tous les modules
"""
import os
import logging
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from supabase import create_client, Client
from langsmith import Client as LangSmithClient

# Configuration LangSmith pour le logging
os.environ["LANGCHAIN_TRACING_V2"] = "true"
if os.getenv("LANGSMITH_API_KEY"):
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "zest-companion")
    langsmith_client = LangSmithClient()
else:
    langsmith_client = None

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def log_to_supabase(message: str, level: str = "INFO", **metadata):
    """
    Logger vers Supabase pour analytics et monitoring
    """
    try:
        # Log local pour debug
        print(f"[{level}] {message}")
        
        # Préparer les données pour Supabase
        log_entry = {
            "level": level,
            "message": message[:1000],  # Limiter la taille
            "created_at": "NOW()",
            "metadata": metadata if metadata else {}
        }
        
        # Ajouter les métadonnées communes si disponibles
        if "thread_id" in metadata:
            log_entry["thread_id"] = metadata["thread_id"]
        if "user_id" in metadata:
            log_entry["user_id"] = metadata["user_id"]
        if "workflow_type" in metadata:
            log_entry["workflow_type"] = metadata["workflow_type"]
        if "node_name" in metadata:
            log_entry["node_name"] = metadata["node_name"]
        if "execution_time_ms" in metadata:
            log_entry["execution_time_ms"] = metadata["execution_time_ms"]
        if "tokens_used" in metadata:
            log_entry["tokens_used"] = metadata["tokens_used"]
            
        # Envoyer vers Supabase (nom de table à créer)
        supabase.table("application_logs").insert(log_entry).execute()
        
    except Exception as e:
        # Fallback - ne pas faire échouer le workflow pour un problème de log
        print(f"❌ Failed to log to Supabase: {e}")
        logger.error(f"Supabase logging failed: {e}")

# Configuration Supabase
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY")
)

# LLM principal
llm = ChatOpenAI(
    model="gpt-4o-mini", 
    temperature=0.7, 
    streaming=True,  # Streaming activé pour la réponse finale
    callbacks=[],
    max_tokens=1500  # Augmenté pour les réponses Lencioni complètes avec insights équipe
)

# LLM d'analyse COMPLÈTEMENT ISOLÉ
analysis_llm = ChatOpenAI(
    model="gpt-3.5-turbo", 
    temperature=0.7,  # Augmenté de 0.3 à 0.7 pour plus de flexibilité
    streaming=False,
    callbacks=[],  # Callbacks vides
    tags=["internal_analysis"],
    # CLÉS D'ISOLATION :
    openai_api_key=os.environ.get("OPENAI_API_KEY"),  # Forcer la clé explicite
    model_kwargs={
        "stream": False,  # Forcer pas de streaming au niveau OpenAI
    },
    logprobs=False,  # Désactiver les logprobs
    # Isolation supplémentaire
    request_timeout=30,
    max_retries=1,
    verbose=False  # Pas de verbose
)

# LLM de fallback
fallback_llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.2, 
    streaming=False,
    callbacks=[],
    tags=["fallback_analysis"],
    openai_api_key=os.environ.get("OPENAI_API_KEY"),
    max_tokens=500,
    request_timeout=15,
    max_retries=2,
    verbose=False
)