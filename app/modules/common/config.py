"""
Configuration commune pour tous les modules
"""
import os
import logging
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from supabase import create_client, Client

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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