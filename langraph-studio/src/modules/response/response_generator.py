"""
Générateur de réponses finales pour les différents sous-thèmes
"""
import logging
from typing import Dict, List, Optional
from langchain_core.messages import SystemMessage, HumanMessage
from ..common.types import WorkflowState
from ..common.config import llm
from ..common.llm_utils import manage_messages
from ..prompts.prompt_builder import build_lencioni_prompt, build_mbti_prompt, create_prompt_by_subtheme

logger = logging.getLogger(__name__)

def generate_final_response(state: WorkflowState) -> WorkflowState:
    """Générateur unifié pour tous les sous-thèmes avec streaming garanti"""
    
    
    # Vérifier si une question conversationnelle est en attente (sous-graphe PCM)
    if state.get('conversational_question_pending') and state.get('pcm_analysis_result'):
        logger.info("🔄 Returning conversational question from PCM subgraph")
        return {
            **state,
            "ai_response": state.get('pcm_analysis_result'),
            "response_complete": True
        }
    
    # Déterminer le sous-thème pour choisir le bon template
    sub_theme = state.get('sub_theme', '')
    theme = state.get('theme', '')
    
    logger.info(f"🤖 NODE 6: Generating response for theme={theme}, sub_theme={sub_theme}")
    
    try:
        # Récupérer la question utilisateur
        user_question = state.get('user_message', '') or (state.get('messages', [{}])[-1].get('content', '') if state.get('messages') else '')
        
        # ============= SYSTÈME DE TEMPLATES UNIFIÉ =============
        # Utiliser le factory central pour tous les sous-thèmes
        system_prompt = create_prompt_by_subtheme(sub_theme, state)
        
        # Debug: Show what context is being sent to the final agent
        logger.info(f"🔍 CONTEXT SENT TO FINAL AGENT:")
        logger.info(f"  - Tool A results: {len(state.get('personalized_content', []))} items")
        logger.info(f"  - Tool B results: {len(state.get('generic_content', []))} items") 
        logger.info(f"  - Tool C results: {len(state.get('others_content', []))} items")
        logger.info(f"  - Tool D results: {len(state.get('general_content', []))} items")
        logger.info(f"  - General results: {len(state.get('general_vector_results', []))} items")
        logger.info(f"  - Lencioni results: {len(state.get('lencioni_vector_results', []))} items")
        logger.info(f"  - System prompt length: {len(system_prompt)} characters")
        
        # Debug: Show complete system prompt for LangGraph Studio  
        logger.info(f"📄 COMPLETE SYSTEM PROMPT SENT TO LLM:")
        logger.info(f"{'='*60}")
        logger.info(system_prompt)
        logger.info(f"{'='*60}")
        logger.info(f"📝 USER MESSAGE:")
        logger.info(f"{state.get('user_message', 'No user message')}")
        
        # Messages pour le LLM
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_question)
        ]
        
        # ============= STREAMING UNIFIÉ POUR TOUS LES TEMPLATES =============
        logger.info(f"🔄 Starting {sub_theme} response streaming...")
        final_response = ""
        
        try:
            # Utiliser streaming explicite pour que LangGraph puisse capturer les tokens
            final_response = ""
            for chunk in llm.stream(messages):
                if chunk.content:
                    final_response += chunk.content
                    # Les tokens sont automatiquement capturés par LangGraph Studio
            
            logger.info(f"✅ {sub_theme} response generated via streaming ({len(final_response)} chars)")
            
            # Préparer le message final
            final_assistant_message = {
                "role": "assistant",
                "content": final_response,
                "type": "assistant",
                "is_final_response": True
            }
            
            # Gérer les messages
            updated_messages = manage_messages(state.get('messages', []), [final_assistant_message])
            
            return {
                "final_response": final_response,
                "messages": updated_messages,
                "user_id": state.get("user_id"),
                "user_name": state.get("user_name"),
                "user_email": state.get("user_email"),
                "user_mbti": state.get("user_mbti"),
                "lencioni_data": state.get("lencioni_data"),
                # system_prompt_debug gardé pour LangGraph Studio seulement
                "system_prompt_debug": system_prompt
            }
            
        except Exception as e:
            logger.error(f"❌ Error generating {sub_theme} response: {e}")
            return {
                "final_response": f"Désolé, une erreur s'est produite lors de l'analyse. Pouvez-vous reformuler votre question ?",
                "messages": state.get('messages', []),
                "user_id": state.get("user_id"),
                "user_name": state.get("user_name"),
                "user_email": state.get("user_email"),
                "user_mbti": state.get("user_mbti")
            }
            
    except Exception as e:
        logger.error(f"❌ Error in generate_final_response: {e}")
        return {
            "final_response": "Désolé, une erreur s'est produite. Pouvez-vous reformuler votre question ?",
            "messages": state.get('messages', []),
            "user_id": state.get("user_id"),
            "user_name": state.get("user_name"),
            "user_email": state.get("user_email"),
            "user_mbti": state.get("user_mbti")
        }


def generate_final_response_original(state: WorkflowState) -> WorkflowState:
    """Version originale conservée pour référence - reproduit l'étape 6 du workflow n8n"""
    logger.info("🤖 NODE 6: Generating final response (ORIGINAL)...")
    
    try:
        # Récupérer la question utilisateur
        user_question = state.get('user_message', '') or (state.get('messages', [{}])[-1].get('content', '') if state.get('messages') else '')
        
        # Ajouter l'historique de conversation pour éviter les répétitions
        conversation_history = []
        for msg in state.get('messages', [])[-3:]:  # 3 derniers messages pour contexte
            if hasattr(msg, 'type') and hasattr(msg, 'content'):
                role = "User" if msg.type == 'human' else "Assistant"
                conversation_history.append(f"{role}: {msg.content}")
            elif isinstance(msg, dict):
                role = "User" if msg.get('role') == 'user' else "Assistant"
                conversation_history.append(f"{role}: {msg.get('content', '')}")
        
        history_text = "\n".join(conversation_history) if conversation_history else "No previous conversation"
        
        # Construire le prompt système avec tous les contextes
        system_prompt = f""" 

ROLE: 
You are ZEST COMPANION, an expert leadership mentor & MBTI coach. You coach with concise yet comprehensive answers, prioritizing specificity and actionable guidance to help the user become a more impactful leader. 

GUARDRAILS: 
- Use ONLY the temperament description and ZEST database search results provided. 
- Do not use MBTI nicknames or external MBTI knowledge unless in context. 
- SCOPE: Focus on MBTI leadership coaching, team dynamics, and professional development only
- REDIRECT: For mental health, family issues, clinical conditions, or therapy needs - recommend consulting appropriate professional resources
- If no relevant info, recommend contacting jean-pierre.aerts@zestforleaders.com.


USER QUESTION: "{user_question}"

RECENT CONVERSATION HISTORY:
{history_text}

User MBTI Profile: {state.get('user_mbti', 'Unknown')}
User Temperament: {state.get('user_temperament', 'Unknown')}
Temperament Description: {state.get('temperament_description', 'Unknown')}
"""
        
        # Ajouter le contenu personnalisé, générique, autres profils et général
        personalized_content = state.get('personalized_content', [])
        generic_content = state.get('generic_content', [])
        others_content = state.get('others_content', [])
        general_content = state.get('general_content', [])
        temperament_content = state.get('temperament_content', [])
        
        if personalized_content:
            system_prompt += "\n\nPERSONALIZED CONTENT (user's specific profile):\n"
            for item in personalized_content[:3]:  # Limiter à 3 items pour éviter trop de contexte
                system_prompt += f"- {item.get('content', '')[:500]}...\n"
                
        if generic_content:
            system_prompt += "\n\nGENERIC USER TYPE CONTENT:\n"
            for item in generic_content[:3]:
                system_prompt += f"- {item.get('content', '')[:500]}...\n"
                
        if others_content:
            system_prompt += "\n\nOTHER MBTI TYPES CONTENT:\n"
            for item in others_content[:3]:
                target_mbti = item.get('metadata', {}).get('target_mbti', 'Unknown')
                system_prompt += f"- {target_mbti}: {item.get('content', '')[:500]}...\n"
                
        if general_content:
            system_prompt += "\n\nGENERAL MBTI CONTENT:\n"
            for item in general_content[:3]:
                system_prompt += f"- {item.get('content', '')[:500]}...\n"
                
        if temperament_content:
            system_prompt += "\n\nTEMPERAMENT CONTENT:\n"
            for item in temperament_content[:2]:  # Limiter davantage pour les tempéraments
                temperament_info = item.get('temperament_info', 'Unknown')
                target = item.get('target', 'Unknown')
                system_prompt += f"- {temperament_info} (for {target}): {item.get('content', '')[:400]}...\n"
        
        system_prompt += """

Your task is to provide personalized, actionable guidance based on the user's MBTI profile and the search results provided above. 
Be conversational, supportive, and specific in your advice. Use the content to inform your response but make it engaging and tailored to their question.
"""
        
        # Debug: Show what context is being sent to the final agent
        logger.info(f"🔍 ORIGINAL CONTEXT SENT TO FINAL AGENT:")
        logger.info(f"  - Personalized: {len(personalized_content)} items")
        logger.info(f"  - Generic: {len(generic_content)} items") 
        logger.info(f"  - Others: {len(others_content)} items")
        logger.info(f"  - General: {len(general_content)} items")
        logger.info(f"  - Temperament: {len(temperament_content)} items")
        logger.info(f"  - System prompt length: {len(system_prompt)} characters")
        
        # Messages pour le LLM
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_question)
        ]
        
        # Streaming pour la réponse finale
        logger.info("🔄 Starting original response streaming...")
        final_response = ""
        
        try:
            for chunk in llm.stream(messages):
                if chunk.content:
                    final_response += chunk.content
            
            logger.info(f"✅ Original response generated via streaming ({len(final_response)} chars)")
            
            # Préparer le message final
            final_assistant_message = {
                "role": "assistant", 
                "content": final_response,
                "type": "assistant",
                "is_final_response": True
            }
            
            # Gérer les messages
            updated_messages = manage_messages(state.get('messages', []), [final_assistant_message])
            
            return {
                "final_response": final_response,
                "messages": updated_messages,
                "user_id": state.get("user_id"),
                "user_name": state.get("user_name"),
                "user_email": state.get("user_email"),
                "user_mbti": state.get("user_mbti"),
                "system_prompt_debug": system_prompt
            }
            
        except Exception as e:
            logger.error(f"❌ Error in original response streaming: {e}")
            return {
                "final_response": "Désolé, une erreur s'est produite lors de la génération de la réponse.",
                "messages": state.get('messages', []),
                "user_id": state.get("user_id"),
                "user_name": state.get("user_name"),
                "user_email": state.get("user_email"),
                "user_mbti": state.get("user_mbti")
            }
            
    except Exception as e:
        logger.error(f"❌ Error in generate_final_response_original: {e}")
        return {
            "final_response": "Désolé, une erreur s'est produite.",
            "messages": state.get('messages', []),
            "user_id": state.get("user_id"),
            "user_name": state.get("user_name"),
            "user_email": state.get("user_email"),
            "user_mbti": state.get("user_mbti")
        }


def create_error_response(error_message: str, state: WorkflowState) -> WorkflowState:
    """Crée une réponse d'erreur standardisée"""
    return {
        "final_response": f"Désolé, {error_message}. Pouvez-vous reformuler votre question ?",
        "messages": state.get('messages', []),
        "user_id": state.get("user_id"),
        "user_name": state.get("user_name"),  
        "user_email": state.get("user_email"),
        "user_mbti": state.get("user_mbti")
    }


def format_response_context(state: WorkflowState) -> Dict:
    """Formate le contexte pour la génération de réponse"""
    return {
        "personalized_count": len(state.get('personalized_content', [])),
        "generic_count": len(state.get('generic_content', [])),
        "others_count": len(state.get('others_content', [])),
        "general_count": len(state.get('general_content', [])),
        "temperament_count": len(state.get('temperament_content', [])),
        "lencioni_count": len(state.get('lencioni_data', [])),
        "has_user_data": bool(state.get('user_mbti') and state.get('user_mbti') != 'Unknown')
    }