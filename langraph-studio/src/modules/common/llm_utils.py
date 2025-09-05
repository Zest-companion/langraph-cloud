"""
Utilitaires pour les appels LLM isolés
"""
import logging
import unicodedata
from typing import Dict, List
from .config import analysis_llm

logger = logging.getLogger(__name__)

def isolated_analysis_call_with_messages(system_content: str, user_content: str, model: str = "gpt-3.5-turbo") -> str:
    """
    Appel OpenAI direct avec messages système et utilisateur séparés
    Garantit l'isolation complète du streaming LangGraph
    """
    try:
        import openai
        import os
        
        # Créer un client complètement isolé
        isolated_client = openai.OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            timeout=30.0
        )
        
        logger.info("🔒 Using completely isolated OpenAI client with separate messages...")
        
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]
        
        response = isolated_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,  # Augmenté de 0.3 à 0.7 pour plus de flexibilité
            max_tokens=500,
            stream=False,  # Absolument pas de streaming
            logprobs=False,  # Désactiver les logprobs
        )
        
        result = response.choices[0].message.content.strip()
        logger.info(f"🔒 Isolated call completed: {len(result)} chars")
        return result
        
    except ImportError:
        logger.warning("⚠️ OpenAI module not available, using fallback")
        return fallback_analysis_call_with_messages(system_content, user_content)
    except Exception as e:
        logger.error(f"❌ Isolated analysis call failed: {e}")
        return fallback_analysis_call_with_messages(system_content, user_content)

def fallback_analysis_call_with_messages(system_content: str, user_content: str) -> str:
    """Fallback using LangChain analysis_llm with separate messages"""
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        logger.info("🔄 Using LangChain fallback with separate messages...")
        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=user_content)
        ]
        response = analysis_llm.invoke(messages)
        return response.content
    except Exception as e:
        logger.error(f"❌ Fallback analysis also failed: {e}")
        return '{"question_type": "PERSONAL_DEVELOPMENT", "instructions": "CALL_AB: Tool A + Tool B", "other_mbti_profiles": null, "continuity_detected": false}'

def isolated_analysis_call(prompt_content: str) -> str:
    """
    Appel OpenAI direct sans passer par LangChain
    Garantit l'isolation complète du streaming LangGraph
    """
    try:
        import openai
        import os
        
        # Créer un client complètement isolé
        isolated_client = openai.OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            timeout=30.0
        )
        
        logger.info("🔒 Using completely isolated OpenAI client...")
        
        # Séparer le prompt système du contenu utilisateur si possible
        if "\n\n" in prompt_content and prompt_content.startswith("You are"):
            # Détection d'un prompt composé (system + user content)
            parts = prompt_content.split("\n\nOriginal User Input:", 1)
            if len(parts) == 2:
                system_part = parts[0]
                user_part = "Original User Input:" + parts[1]
                messages = [
                    {"role": "system", "content": system_part},
                    {"role": "user", "content": user_part}
                ]
            else:
                messages = [{"role": "user", "content": prompt_content}]
        else:
            messages = [{"role": "user", "content": prompt_content}]
        
        response = isolated_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7,  # Augmenté de 0.3 à 0.7 pour plus de flexibilité
            max_tokens=500,
            stream=False,  # Absolument pas de streaming
            logprobs=False,  # Désactiver les logprobs
        )
        
        result = response.choices[0].message.content.strip()
        logger.info(f"🔒 Isolated call completed: {len(result)} chars")
        return result
        
    except ImportError:
        logger.warning("⚠️ OpenAI module not available, using fallback")
        return fallback_analysis_call(prompt_content)
    except Exception as e:
        logger.error(f"❌ Isolated analysis call failed: {e}")
        return fallback_analysis_call(prompt_content)

def fallback_analysis_call(prompt_content: str) -> str:
    """Fallback using LangChain analysis_llm"""
    try:
        from langchain_core.messages import HumanMessage
        logger.info("🔄 Using LangChain fallback for analysis...")
        response = analysis_llm.invoke([HumanMessage(content=prompt_content)])
        return response.content
    except Exception as e:
        logger.error(f"❌ Fallback analysis also failed: {e}")
        return '{"question_type": "PERSONAL_DEVELOPMENT", "instructions": "CALL_AB: Tool A + Tool B", "other_mbti_profiles": null, "continuity_detected": false}'

# Fonction pour gérer les messages manuellement sans auto-ajout
def normalize_name_for_metadata(name: str) -> str:
    """
    Normalise un nom pour matcher le format utilisé dans les métadonnées de vectorisation.
    jean-pierre_aerts -> jean_pierre_aerts (remplace tirets par underscores)
    """
    if not name:
        return name
    # Remplacer les tirets par des underscores pour matcher les métadonnées
    normalized = name.replace('-', '_')
    return normalized

def manage_messages(existing_messages, new_messages):
    """Gérer manuellement les messages sans ajouter automatiquement toutes les réponses LLM"""
    if not existing_messages:
        existing_messages = []
    
    # Seulement ajouter les messages utilisateur et les réponses finales explicites
    result = existing_messages[:]
    for msg in new_messages:
        if msg.get('type') == 'user' or msg.get('role') == 'user':
            result.append(msg)
        elif msg.get('type') == 'assistant' or msg.get('role') == 'assistant':
            # Seulement ajouter si c'est explicitement marqué comme réponse finale
            if msg.get('is_final_response', False):
                result.append(msg)
    
    return result