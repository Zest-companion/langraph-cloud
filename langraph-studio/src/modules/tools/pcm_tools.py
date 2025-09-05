"""
Outils spécialisés PCM - Équivalent des outils MBTI pour PCM
Chaque outil gère un type de flux PCM spécifique
"""
import logging
from typing import Dict, Any, List
from ..common.types import WorkflowState

logger = logging.getLogger(__name__)

# ============= ROUTING FUNCTION =============

def pcm_tools_router(state: WorkflowState) -> str:
    """
    Route vers le bon outil PCM selon le flux détecté
    Équivalent du router MBTI mais pour PCM
    """
    flow_type = state.get('flow_type', '')
    search_focus = state.get('search_focus', '')
    
    logger.info(f"🎯 PCM Tools Router: flow_type='{flow_type}', search_focus='{search_focus}'")
    
    # Routing selon le flow_type du PCMFlowManager
    # PRIORITÉ ABSOLUE: Traiter les SAFETY REFUSAL EN PREMIER
    if flow_type == 'safety_refusal':
        logger.warning("🚫 SAFETY REFUSAL detected - routing to safety response")
        return "execute_pcm_no_search"  # Réponse directe sans recherche
    # IMPORTANT: Traiter les greetings EN SECOND
    elif flow_type == 'greeting' or state.get('skip_search') or state.get('greeting_detected'):
        logger.info("➡️  Routing to: execute_pcm_no_search (greeting/no search)")
        return "execute_pcm_no_search"
    elif flow_type == 'comparison' or search_focus == 'comparison':
        logger.info("➡️  Routing to: execute_pcm_comparison_tool")
        return "execute_pcm_comparison_tool"
    elif flow_type in ['self_base', 'self_phase'] or search_focus in ['user_base', 'user_phase']:
        logger.info("➡️  Routing to: execute_pcm_self_tool")
        return "execute_pcm_self_tool"
    elif flow_type == 'self_action_plan' or search_focus == 'action_plan':
        logger.info("➡️  Routing to: execute_pcm_action_plan_tool")
        return "execute_pcm_action_plan_tool"
    elif flow_type == 'coworker_focused' or search_focus == 'coworker':
        logger.info("➡️  Routing to: execute_pcm_coworker_tool")
        return "execute_pcm_coworker_tool"
    elif flow_type == 'exploration' or search_focus == 'all_bases':
        logger.info("➡️  Routing to: execute_pcm_exploration_tool")
        return "execute_pcm_exploration_tool"
    elif flow_type == 'general_pcm' or search_focus == 'theory':
        logger.info("➡️  Routing to: execute_pcm_general_tool")
        return "execute_pcm_general_tool"
    else:
        # Fallback vers self si on a une base PCM
        if state.get('pcm_base'):
            logger.info(f"⚠️  No clear match, defaulting to self-focused (execute_pcm_self_tool)")
            return "execute_pcm_self_tool"
        else:
            logger.info(f"⚠️  No clear match, defaulting to general (execute_pcm_general_tool)")
            return "execute_pcm_general_tool"

# ============= OUTIL SELF-FOCUSED =============

def execute_pcm_self_tool(state: WorkflowState) -> WorkflowState:
    """
    Outil PCM pour questions sur soi-même (BASE ou PHASE)
    Équivalent de execute_tools_ab pour MBTI
    Utilise le système conversationnel intelligent
    """
    logger.info("🔧 PCM SELF TOOL: Executing self-focused PCM search...")
    
    try:
        # Vérifier si c'est la première interaction PCM
        messages = state.get('messages', [])
        is_first_pcm = len(messages) <= 2
        
        # PRIORITÉ 1: Vérifier d'abord si l'analyse conversationnelle a détecté une transition PHASE
        pcm_conversational_context = state.get('pcm_conversational_context', {})
        conversational_suggests_phase = pcm_conversational_context.get('suggested_context') == 'phase'
        
        flow_type = state.get('flow_type', 'self_base')
        pcm_base_or_phase = state.get('pcm_base_or_phase')
        
        if conversational_suggests_phase:
            # L'analyse conversationnelle suggère une transition PHASE - priorité absolue
            pcm_base_or_phase = 'phase'
            logger.info(f"🔄 PCM Self Tool: Conversational analysis detected PHASE transition → pcm_base_or_phase='phase'")
        elif pcm_base_or_phase:
            # PCMFlowManager a déjà configuré base/phase - l'utiliser
            logger.info(f"📊 PCM Self Tool: Using PCMFlowManager config → pcm_base_or_phase='{pcm_base_or_phase}'")
        else:
            # Fallback: déterminer par flow_type si PCMFlowManager n'a pas configuré
            if 'phase' in flow_type.lower():
                pcm_base_or_phase = 'phase'
            else:
                pcm_base_or_phase = 'base'
            logger.info(f"📊 PCM Self Tool: Fallback detection from flow_type='{flow_type}' → pcm_base_or_phase='{pcm_base_or_phase}'")
        
        # S'assurer qu'on est en mode self avec les bons flags
        updated_state = {
            **state,
            'flow_type': flow_type,
            'search_focus': state.get('search_focus', 'user_base'),
            'pcm_base_or_phase': pcm_base_or_phase
        }
        
        # Si c'est la première interaction, ajouter les flags nécessaires
        if is_first_pcm and not state.get('pcm_specific_dimensions'):
            logger.info("🆕 First PCM interaction detected - setting up conversational flags")
            updated_state['use_first_interaction_prompt'] = True
            updated_state['pcm_context_stage'] = 'dimension_selection'
            updated_state['pcm_available_dimensions'] = [
                "perception", "strengths", "interaction_style", 
                "personality_part", "channel_communication", "environmental_preferences"
            ]
        
        # Déléguer à l'analyse PCM existante
        from ..pcm.pcm_vector_search import pcm_vector_search
        result_state = pcm_vector_search(updated_state)
        
        # Marquer que l'outil PCM self a été utilisé
        result_state['pcm_tool_used'] = 'self'
        result_state['pcm_search_complete'] = True
        result_state['conversational_analysis_complete'] = True  # Pour activer les prompts conversationnels
        
        logger.info("✅ PCM Self Tool completed successfully")
        return result_state
        
    except Exception as e:
        logger.error(f"❌ Error in PCM Self Tool: {e}")
        return {**state, 'pcm_tool_error': str(e)}

# ============= OUTIL ACTION PLAN =============

def execute_pcm_action_plan_tool(state: WorkflowState) -> WorkflowState:
    """
    Outil PCM pour demandes de conseils et plans d'action
    Recherche focalisée sur les stratégies et recommandations
    """
    logger.info("🔧 PCM ACTION PLAN TOOL: Executing action plan search...")
    
    try:
        # Utiliser la recherche PCM avec focus sur les plans d'action
        from ..pcm.pcm_vector_search import pcm_vector_search
        
        # Configurer l'état pour la recherche de plans d'action
        # Focus sur les sections action_plan de la phase de stress
        logger.info(f"🔍 DEBUG action_plan_tool INPUT - pcm_classification: {state.get('pcm_classification')}")
        updated_state = {
            **state,
            'flow_type': 'self_action_plan',
            'search_focus': 'action_plan',
            'pcm_base_or_phase': 'phase',  # Chercher dans les données de phase
            'section_type': 'action_plan',  # Spécifiquement les sections action_plan de la phase
        }
        logger.info(f"🔍 DEBUG action_plan_tool PASSING - pcm_classification: {updated_state.get('pcm_classification')}")
        
        result_state = pcm_vector_search(updated_state)
        
        # Marquer que l'outil PCM action plan a été utilisé
        result_state['pcm_tool_used'] = 'action_plan'
        result_state['pcm_search_complete'] = True
        result_state['conversational_analysis_complete'] = True  # Pour activer les prompts conversationnels
        
        logger.info("✅ PCM Action Plan Tool completed successfully")
        return result_state
        
    except Exception as e:
        logger.error(f"❌ Error in PCM Action Plan Tool: {e}")
        return {**state, 'pcm_tool_error': str(e)}

# ============= OUTIL COMPARISON =============

def execute_pcm_comparison_tool(state: WorkflowState) -> WorkflowState:
    """
    Outil PCM pour comparaisons entre types
    Équivalent de execute_tools_abc pour MBTI (recherche multi-types)
    """
    logger.info("🔧 PCM COMPARISON TOOL: Executing PCM type comparison...")
    
    try:
        # Utiliser la logique de comparaison déjà implémentée
        from ..pcm.pcm_vector_search import _handle_comparison_search
        
        result_state = _handle_comparison_search(state)
        
        # Marquer que l'outil PCM comparison a été utilisé
        result_state['pcm_tool_used'] = 'comparison'
        result_state['pcm_search_complete'] = True
        
        logger.info("✅ PCM Comparison Tool completed successfully")
        return result_state
        
    except Exception as e:
        logger.error(f"❌ Error in PCM Comparison Tool: {e}")
        return {**state, 'pcm_tool_error': str(e)}

# ============= OUTIL COWORKER =============

def execute_pcm_coworker_tool(state: WorkflowState) -> WorkflowState:
    """
    Outil PCM pour relations avec collègues
    Recherche focalisée sur les dynamiques interpersonnelles
    """
    logger.info("🔧 PCM COWORKER TOOL: Executing coworker-focused search...")
    
    try:
        # Utiliser la recherche PCM avec focus coworker
        from ..pcm.pcm_vector_search import pcm_vector_search
        
        # Configurer l'état pour la recherche coworker
        updated_state = {
            **state,
            'flow_type': 'coworker_focused',
            'search_focus': 'coworker',
            'pcm_base_or_phase': 'base'  # Focus sur les interactions de base
        }
        
        result_state = pcm_vector_search(updated_state)
        
        # Marquer que l'outil PCM coworker a été utilisé
        result_state['pcm_tool_used'] = 'coworker'
        result_state['pcm_search_complete'] = True
        
        logger.info("✅ PCM Coworker Tool completed successfully")
        return result_state
        
    except Exception as e:
        logger.error(f"❌ Error in PCM Coworker Tool: {e}")
        return {**state, 'pcm_tool_error': str(e)}

# ============= OUTIL EXPLORATION =============

def execute_pcm_exploration_tool(state: WorkflowState) -> WorkflowState:
    """
    Outil PCM pour exploration de toutes les bases
    Recherche étendue sur tous les types PCM
    """
    logger.info("🔧 PCM EXPLORATION TOOL: Executing multi-base exploration...")
    
    try:
        # Recherche étendue sur tous les types PCM
        from ..pcm.pcm_vector_search import pcm_vector_search
        
        # Configurer pour exploration générale
        updated_state = {
            **state,
            'flow_type': 'exploration',
            'search_focus': 'all_bases',
            'pcm_specific_dimensions': None  # Toutes les dimensions
        }
        
        result_state = pcm_vector_search(updated_state)
        
        # Marquer que l'outil PCM exploration a été utilisé
        result_state['pcm_tool_used'] = 'exploration'
        result_state['pcm_search_complete'] = True
        
        logger.info("✅ PCM Exploration Tool completed successfully")
        return result_state
        
    except Exception as e:
        logger.error(f"❌ Error in PCM Exploration Tool: {e}")
        return {**state, 'pcm_tool_error': str(e)}

# ============= OUTIL GENERAL =============

def execute_pcm_general_tool(state: WorkflowState) -> WorkflowState:
    """
    Outil PCM pour théorie générale et concepts
    Utilise la recherche spécialisée avec filtres corrects
    """
    logger.info("🔧 PCM GENERAL TOOL: Executing general PCM theory search...")
    
    try:
        # Utiliser la recherche spécialisée pour théorie générale
        from ..pcm.pcm_vector_search import _handle_general_pcm_search
        
        result_state = _handle_general_pcm_search(state)
        
        # Marquer que l'outil PCM general a été utilisé
        result_state['pcm_tool_used'] = 'general'
        result_state['pcm_search_complete'] = True
        
        logger.info("✅ PCM General Tool completed successfully")
        return result_state
        
    except Exception as e:
        logger.error(f"❌ Error in PCM General Tool: {e}")
        return {**state, 'pcm_tool_error': str(e)}

# ============= OUTIL NO_SEARCH (pour greetings) =============

def execute_pcm_no_search(state: WorkflowState) -> WorkflowState:
    """
    Outil PCM pour cas sans recherche (salutations, remerciements, safety refusal)
    Équivalent de no_tools pour MBTI - gère les greetings et safety refusal
    """
    flow_type = state.get('flow_type', '')
    
    if flow_type == 'safety_refusal':
        logger.warning("🚫 PCM NO SEARCH: Handling safety refusal - blocking request")
        return {
            **state,
            'pcm_tool_used': 'safety_refusal',
            'pcm_search_complete': True,
            'skip_search': True,
            'no_search_needed': True,
            'pcm_resources': '',  # Pas de ressources pour safety refusal
            'pcm_base_results': [],
            'pcm_phase_results': [],
            'pcm_comparison_results': {},
            'safety_refusal_handled': True  # Flag pour generate_final_response
        }
    else:
        logger.info("🔧 PCM NO SEARCH: Handling greeting/thanks - no search needed...")
        
        # Marquer explicitement qu'aucune recherche n'est nécessaire
        # Exactement comme no_tools pour MBTI
        return {
            **state,
            'pcm_tool_used': 'no_search',
            'pcm_search_complete': True,
            'skip_search': True,
            'no_search_needed': True,  # Flag identique à MBTI pour generate_final_response
            'pcm_resources': '',  # Pas de ressources pour greeting
            'pcm_base_results': [],  # Vider les résultats
            'pcm_phase_results': [],
            'pcm_comparison_results': {},
            'greeting_detected': True  # Flag spécifique pour indiquer un greeting
        }