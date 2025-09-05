"""
PCM Analysis New - Version avec système de flux unifié (inspiré MBTI)
Point d'entrée moderne qui utilise le PCMFlowManager pour tous les flux
"""
import logging
from typing import Dict, Any, List
from ..common.types import WorkflowState
from .pcm_flow_manager import PCMFlowManager

logger = logging.getLogger(__name__)

def pcm_analysis_with_flow_manager(state: WorkflowState) -> Dict[str, Any]:
    """
    Point d'entrée PCM unifié - Système unique pour tous les flux PCM
    
    SYSTÈME UNIFIÉ COMPLET:
    - Classification globale + analyse fine en une seule passe
    - Transitions dynamiques naturelles (PHASE → COWORKER, etc.)
    - Gestion complète BASE/PHASE/ACTION_PLAN + COWORKER/COMPARISON
    - Performance optimisée (un seul appel LLM au lieu de plusieurs)
    """
    logger.info("🧠 PCM Analysis (Unified) - Single Source of Truth")
    
    # Utiliser notre nouveau système unifié
    try:
        from .pcm_unified_analysis import pcm_unified_intent_analysis
        result = pcm_unified_intent_analysis(state)
        logger.info(f"✅ Unified system completed: {result.get('flow_type', 'N/A')}")
        return result
    except Exception as e:
        logger.error(f"❌ Unified system failed, using fallback: {e}")
        return _fallback_to_legacy_system(state)


def _fallback_to_legacy_system(state: WorkflowState) -> Dict[str, Any]:
    """Fallback vers l'ancien système en cas d'erreur"""
    logger.warning("🔄 Using legacy PCM system as fallback")
    
    try:
        # Essayer PCMFlowManager + PCMConversationalAnalysis
        flow_type = state.get('flow_type', '')
        
        if not flow_type:
            classification = PCMFlowManager.classify_pcm_intent(state)
            flow_type = classification.get('flow_type', 'SELF_BASE')
    pcm_context = state.get('pcm_base_or_phase', '')
    
    # Map PCMFlowManager types to context if needed
    if flow_type == 'SELF_BASE' and not pcm_context:
        pcm_context = 'base'
    elif flow_type == 'SELF_PHASE' and not pcm_context:
        pcm_context = 'phase'
    
    specific_dimensions = state.get('pcm_specific_dimensions')
    explored_dimensions = state.get('pcm_explored_dimensions', [])
    is_first_pcm_interaction = len(explored_dimensions) == 0 and len(state.get('messages', [])) <= 2
    
    logger.info(f"📊 Flow detected: {flow_type}, context: {pcm_context}")
    logger.info(f"📋 Specific dimensions: {specific_dimensions}")
    logger.info(f"🗺️ Explored dimensions: {explored_dimensions}")
    logger.info(f"🆕 First PCM interaction: {is_first_pcm_interaction}")
    
    # Si c'est self_focused → utiliser le système conversationnel complet
    if flow_type in ['self_focused', 'self_base', 'SELF_BASE', 'SELF_PHASE']:
        logger.info("🔄 Using complete conversational analysis system for self_focused")
        
        # Cas spécial: première interaction BASE sans dimensions spécifiques
        if is_first_pcm_interaction and pcm_context == 'base' and not specific_dimensions:
            logger.info("🆕 First BASE interaction - presenting overview of 6 dimensions")
            return _handle_first_base_interaction(state)
        
        # Cas spécial: première interaction PHASE 
        elif is_first_pcm_interaction and pcm_context == 'phase':
            logger.info("🆕 First PHASE interaction - explaining stress concept")
            return _handle_first_phase_interaction(state)
            
        # Pour toutes les autres interactions (y compris non-first), d'abord vérifier avec PCMFlowManager
        # pour détecter COMPARISON, COWORKER, etc.
        logger.info("🤖 Checking with PCMFlowManager for intent detection (COMPARISON, COWORKER, etc.)")
        try:
            classification = PCMFlowManager.classify_pcm_intent(state)
            detected_flow = classification.get('flow_type')
            
            # Si c'est une COMPARISON, COWORKER, ou autre flux spécial → utiliser le système unifié
            if detected_flow in ['COMPARISON', 'COWORKER', 'TEAM', 'EXPLORATION', 'GENERAL_PCM', 'GREETING']:
                logger.info(f"🎯 PCMFlowManager detected special flow: {detected_flow} - using unified system")
                updated_state = PCMFlowManager.execute_flow_action(classification, state)
                return {
                    **updated_state,
                    'pcm_classification': classification,
                    'pcm_flow_manager_used': True
                }
            
            # Si c'est SELF_BASE ou SELF_PHASE → continuer avec l'analyse conversationnelle
            logger.info(f"🔄 PCMFlowManager detected self flow: {detected_flow} - using conversational analysis")
            
        except Exception as e:
            logger.warning(f"⚠️ PCMFlowManager classification failed: {e} - proceeding with conversational analysis")
        
        # Cas normal: utiliser l'analyse conversationnelle
        try:
            from .pcm_conversational_analysis import analyze_pcm_conversational_intent
            return analyze_pcm_conversational_intent(state)
        except ImportError:
            logger.warning("⚠️ Conversational analysis not available")
            # Fallback vers analyse standard
            from .pcm_analysis import pcm_analysis
            return pcm_analysis(state)
    
    # Sinon utiliser le nouveau système unifié pour les autres flux
    logger.info("🎯 Using unified flow manager for non-self flows")
    try:
        # Étape 1: Classifier l'intention utilisateur
        classification = PCMFlowManager.classify_pcm_intent(state)
        logger.info(f"📊 PCM Intent classified: {classification.get('flow_type')} (confidence: {classification.get('confidence', 0)})")
        
        # Étape 2: Exécuter l'action appropriée
        updated_state = PCMFlowManager.execute_flow_action(classification, state)
        
        # Étape 3: Ajouter les informations de classification à l'état
        final_state = {
            **updated_state,
            'pcm_classification': classification,
            'pcm_flow_manager_used': True
        }
        
        logger.info(f"✅ PCM Flow executed: {classification.get('flow_type')} → {classification.get('action')}")
        return final_state
        
    except Exception as e:
        logger.error(f"❌ Error in PCM Flow Manager: {e}")
        logger.info("🔄 Falling back to legacy PCM analysis")
        
        # Fallback vers l'ancien système en cas d'erreur
        from .pcm_analysis import pcm_analysis
        return pcm_analysis(state)

def pcm_vector_search_with_flow_manager(state: WorkflowState) -> Dict[str, Any]:
    """
    Recherche vectorielle PCM adaptée au nouveau système de flux avec outils spécialisés
    """
    logger.info("🔍 PCM Vector Search (New) - Using PCM Tools Router")
    
    # Vérifier si on doit poser une question conversationnelle
    if state.get('needs_user_response') and state.get('conversational_question'):
        logger.info("🔄 Conversational question pending - skipping vector search")
        return state
    
    # Vérifier si on doit skiper la recherche (greeting, etc.)
    if state.get('skip_search'):
        logger.info("🔄 Search skipped per flow classification")
        return state
    
    # Vérifier si c'est un flux conversationnel en attente de réponse finale
    if state.get('skip_final_generation'):
        logger.info("🔄 Waiting for user response - skipping search")
        return state
    
    # Utiliser le router des outils PCM
    try:
        from ..tools.pcm_tools import pcm_tools_router
        
        # Déterminer l'outil à utiliser
        tool_name = pcm_tools_router(state)
        logger.info(f"🎯 PCM Tools Router selected: {tool_name}")
        
        # Exécuter l'outil approprié
        if tool_name == "execute_pcm_self_tool":
            from ..tools.pcm_tools import execute_pcm_self_tool
            return execute_pcm_self_tool(state)
        elif tool_name == "execute_pcm_comparison_tool":
            from ..tools.pcm_tools import execute_pcm_comparison_tool
            return execute_pcm_comparison_tool(state)
        elif tool_name == "execute_pcm_coworker_tool":
            from ..tools.pcm_tools import execute_pcm_coworker_tool
            return execute_pcm_coworker_tool(state)
        elif tool_name == "execute_pcm_exploration_tool":
            from ..tools.pcm_tools import execute_pcm_exploration_tool
            return execute_pcm_exploration_tool(state)
        elif tool_name == "execute_pcm_general_tool":
            from ..tools.pcm_tools import execute_pcm_general_tool
            return execute_pcm_general_tool(state)
        elif tool_name == "execute_pcm_no_search":
            from ..tools.pcm_tools import execute_pcm_no_search
            return execute_pcm_no_search(state)
        else:
            logger.warning(f"⚠️ Unknown PCM tool: {tool_name}, using fallback")
            # Fallback vers la recherche standard
            from .pcm_analysis import pcm_vector_search
            return pcm_vector_search(state)
            
    except Exception as e:
        logger.error(f"❌ Error in PCM Tools Router: {e}")
        # Fallback vers la recherche standard
        logger.info("🔄 Using fallback standard PCM search")
        from .pcm_analysis import pcm_vector_search
        return pcm_vector_search(state)

def _handle_coworker_search(state: WorkflowState) -> Dict[str, Any]:
    """Gère la recherche pour le flux coworker selon la stratégie"""
    search_strategy = state.get('search_strategy', 'basic')
    logger.info(f"👥 Coworker search strategy: {search_strategy}")
    
    if search_strategy == 'my_base_other_base':
        # Les deux vont bien : chercher stratégies collaboration
        return _search_collaboration_strategies(state)
    elif search_strategy == 'comprehensive_coworker_support':
        # Collègue en stress : chercher gestion de son stress + collaboration
        return _search_stress_support_strategies(state)
    elif search_strategy == 'mutual_stress_management':
        # Les deux en difficulté : chercher plans d'action pour tous les deux
        return _search_mutual_support_strategies(state)
    else:
        # Stratégie de base
        return _search_basic_coworker_guidance(state)

def _handle_user_base_search(state: WorkflowState) -> Dict[str, Any]:
    """Recherche focalisée sur la base de l'utilisateur"""
    logger.info("🔍 Searching user's PCM base information")
    
    # Utiliser la recherche existante avec focus sur la base
    updated_state = {**state, 'pcm_base_or_phase': 'base'}
    from .pcm_analysis import pcm_vector_search
    return pcm_vector_search(updated_state)

def _handle_user_phase_search(state: WorkflowState) -> Dict[str, Any]:
    """Recherche focalisée sur la phase de l'utilisateur"""
    logger.info("🔍 Searching user's PCM phase/stress information")
    
    # Utiliser la recherche existante avec focus sur la phase
    updated_state = {**state, 'pcm_base_or_phase': 'phase'}
    from .pcm_analysis import pcm_vector_search
    return pcm_vector_search(updated_state)

def _handle_comparison_search(state: WorkflowState) -> Dict[str, Any]:
    """Recherche comparative entre types PCM - recherche pour chaque type à comparer"""
    logger.info("🔍 Searching PCM type comparisons")
    
    # Récupérer les types à comparer
    types_to_compare = state.get('pcm_types_to_compare', [])
    user_base = state.get('pcm_base', '').lower()
    
    logger.info(f"🔍 Comparison: User base '{user_base}' vs extracted types: {types_to_compare}")
    
    # Préparer la liste finale des types à rechercher
    comparison_types = []
    
    # Ajouter la base de l'utilisateur
    if user_base:
        comparison_types.append(user_base)
    
    # Ajouter les types extraits de la question
    for pcm_type in types_to_compare:
        if pcm_type and pcm_type.lower() not in [t.lower() for t in comparison_types]:
            comparison_types.append(pcm_type.lower())
    
    # Si pas de types extraits, essayer d'en déduire depuis la question
    if not types_to_compare:
        comparison_types.extend(_extract_pcm_types_from_message(state))
    
    logger.info(f"🎯 Final comparison types: {comparison_types}")
    
    # Faire une recherche pour chaque type et organiser par type
    from .pcm_analysis import pcm_vector_search
    comparison_results = {}
    
    for pcm_type in comparison_types:
        logger.info(f"🔍 Searching information for PCM type: {pcm_type}")
        
        # Créer un état modifié pour chaque type
        type_state = {
            **state,
            'pcm_base': pcm_type,
            'pcm_base_or_phase': 'base',  # Focus sur les informations de base
            'pcm_specific_dimensions': None,  # Pas de filtre - récupérer TOUTES les dimensions
            'comparison_target_type': pcm_type  # Pour tracking
        }
        
        # Effectuer la recherche pour ce type
        type_results = pcm_vector_search(type_state)
        
        # Stocker les résultats par type
        if type_results.get('pcm_base_results'):
            comparison_results[pcm_type] = type_results['pcm_base_results']
        else:
            comparison_results[pcm_type] = []
    
    # Créer les ressources formatées pour comparaison
    pcm_resources = _format_comparison_resources(comparison_results, comparison_types, state.get('language', 'en'))
    
    # Combiner tous les résultats dans l'état final avec structure spécialisée
    final_state = {
        **state,
        'pcm_comparison_results': comparison_results,
        'pcm_comparison_types': comparison_types,
        'pcm_resources': pcm_resources,
        'vector_search_complete': True
    }
    
    total_results = sum(len(results) for results in comparison_results.values())
    logger.info(f"✅ Comparison search completed: {total_results} results for {len(comparison_types)} types")
    logger.info(f"📊 Results per type: {[(t, len(comparison_results.get(t, []))) for t in comparison_types]}")
    return final_state

def _format_comparison_resources(comparison_results: Dict[str, List], comparison_types: List[str], language: str) -> str:
    """Formate les résultats de comparaison pour le prompt"""
    
    total_results = sum(len(results) for results in comparison_results.values())
    
    if total_results == 0:
        return "No PCM comparison information found for the requested types."
    
    formatted_content = f"# PCM COMPARISON SEARCH RESULTS\n"
    formatted_content += f"Flow: comparison | Language: {language} | Total: {total_results} results\n"
    formatted_content += f"Comparing: {' vs '.join(comparison_types)}\n\n"
    
    # Section pour chaque type PCM
    for pcm_type in comparison_types:
        results = comparison_results.get(pcm_type, [])
        if results:
            type_name = pcm_type.title()
            formatted_content += f"## 🔍 {type_name.upper()} TYPE ({len(results)} items)\n"
            
            for i, result in enumerate(results, 1):
                content = result.get('content', '')
                metadata = result.get('metadata', {})
                similarity = result.get('similarity', 0)
                section_type = metadata.get('section_type', 'Content')
                
                formatted_content += f"### {type_name} Item {i}\n"
                formatted_content += f"**PCM Type**: {pcm_type} | **Section**: {section_type} | **Score**: {similarity:.3f}\n"
                formatted_content += f"{content}\n\n"
            
            formatted_content += "---\n\n"
    
    return formatted_content

def _extract_pcm_types_from_message(state: WorkflowState) -> List[str]:
    """Extrait les types PCM mentionnés dans le message utilisateur"""
    user_message = state.get('user_message', '').lower()
    
    # Types PCM possibles
    pcm_types = ['thinker', 'persister', 'harmonizer', 'imaginer', 'rebel', 'promoter']
    
    found_types = []
    for pcm_type in pcm_types:
        if pcm_type in user_message:
            found_types.append(pcm_type)
    
    logger.info(f"🔍 Extracted PCM types from message: {found_types}")
    return found_types

def _handle_exploration_search(state: WorkflowState) -> Dict[str, Any]:
    """Recherche d'exploration de toutes les bases"""
    logger.info("🔍 Searching all PCM bases exploration")
    
    # TODO: Implémenter recherche multi-bases
    from .pcm_analysis import pcm_vector_search
    return pcm_vector_search(state)

def _handle_theory_search(state: WorkflowState) -> Dict[str, Any]:
    """Recherche de théorie PCM générale"""
    logger.info("🔍 Searching general PCM theory")
    
    # TODO: Implémenter recherche théorique
    from .pcm_analysis import pcm_vector_search
    return pcm_vector_search(state)

def _search_collaboration_strategies(state: WorkflowState) -> Dict[str, Any]:
    """Cherche des stratégies de collaboration (les deux vont bien)"""
    logger.info("👥 Searching collaboration strategies")
    
    # Utiliser la recherche coworker existante
    from .pcm_analysis import pcm_vector_search
    return pcm_vector_search(state)

def _search_stress_support_strategies(state: WorkflowState) -> Dict[str, Any]:
    """Cherche des stratégies de support pour collègue en stress"""
    logger.info("👥 Searching stress support strategies")
    
    # Utiliser la recherche coworker existante
    from .pcm_analysis import pcm_vector_search
    return pcm_vector_search(state)

def _search_mutual_support_strategies(state: WorkflowState) -> Dict[str, Any]:
    """Cherche des stratégies quand les deux sont en difficulté"""
    logger.info("👥 Searching mutual support strategies")
    
    # Utiliser la recherche coworker existante
    from .pcm_analysis import pcm_vector_search
    return pcm_vector_search(state)

def _search_basic_coworker_guidance(state: WorkflowState) -> Dict[str, Any]:
    """Recherche de base pour conseils collègue"""
    logger.info("👥 Searching basic coworker guidance")
    
    # Utiliser la recherche coworker existante
    from .pcm_analysis import pcm_vector_search
    return pcm_vector_search(state)

# Fonctions de première interaction (système base/phase complet restauré)

def _handle_first_base_interaction(state: WorkflowState) -> Dict[str, Any]:
    """
    Gère la première interaction BASE avec le système de prompts complet
    Utilise les prompts spécialisés pour reproduire le comportement original
    """
    logger.info("🆕 First BASE interaction - using specialized first interaction prompts")
    
    # Importer les prompts de première interaction
    try:
        from ..prompts.pcm_first_interaction_prompts import build_pcm_first_interaction_general_prompt
        
        # Construire le prompt spécialisé
        prompt = build_pcm_first_interaction_general_prompt(state)
        
        # Utiliser le prompt pour générer la réponse via LLM
        from ..common.llm_utils import isolated_analysis_call_with_messages
        
        # Obtenir les ressources PCM pour le contexte
        pcm_resources = state.get('pcm_resources', 'No specific PCM resources available for first interaction')
        
        # Préparer le contenu utilisateur
        user_query = state.get('user_message', '') or (
            state.get('messages', [{}])[-1].get('content', '') if state.get('messages') else 'Exploring my PCM BASE'
        )
        
        # Générer la réponse d'introduction
        response = isolated_analysis_call_with_messages(
            system_content=prompt,
            user_content=f"First PCM BASE interaction for: {user_query}"
        )
        
        return {
            **state,
            'analysis_result': response,
            'needs_user_response': False,
            'conversational_question': None,
            'skip_final_generation': False,  # On veut la génération finale avec le contenu
            'skip_search': False,  # On VEUT la recherche vectorielle pour le contenu personnalisé
            'pcm_first_base_interaction_complete': True,
            'use_first_interaction_prompt': True,  # Flag pour utiliser le prompt spécialisé
            'pcm_available_dimensions': [
                "perception", "strengths", "interaction_style", 
                "personality_part", "channel_communication", "environmental_preferences"
            ],
            'pcm_context_stage': 'dimension_selection',
            'flow_type': 'self_focused',  # Assure qu'on reste dans le bon flux
            'pcm_base_or_phase': 'base'   # Assure la recherche BASE
        }
        
    except Exception as e:
        logger.error(f"❌ Error in first BASE interaction: {e}")
        # Fallback simple
        return _simple_first_base_fallback(state)

def _handle_first_phase_interaction(state: WorkflowState) -> Dict[str, Any]:
    """
    Gère la première interaction PHASE avec redirection intelligente
    """
    logger.info("🆕 First PHASE interaction - using phase redirect prompt")
    
    try:
        from ..prompts.pcm_first_interaction_prompts import build_pcm_first_interaction_phase_redirect_prompt
        
        # Construire le prompt de redirection PHASE
        prompt = build_pcm_first_interaction_phase_redirect_prompt(state)
        
        # Utiliser le prompt pour générer la réponse
        from ..common.llm_utils import isolated_analysis_call_with_messages
        
        user_query = state.get('user_message', '') or (
            state.get('messages', [{}])[-1].get('content', '') if state.get('messages') else 'Exploring my PCM PHASE'
        )
        
        response = isolated_analysis_call_with_messages(
            system_content=prompt,
            user_content=f"First PCM PHASE interaction for: {user_query}"
        )
        
        return {
            **state,
            'analysis_result': response,
            'needs_user_response': True,
            'conversational_question': "Would you like to explore your PHASE or start with your BASE?",
            'skip_final_generation': True,  # Skip general response generation for first interaction
            'skip_search': True,
            'pcm_first_phase_interaction_complete': True,
            'pcm_context_stage': 'base_or_phase_choice',
            'final_response': response  # Set the final response directly
        }
        
    except Exception as e:
        logger.error(f"❌ Error in first PHASE interaction: {e}")
        return _simple_first_phase_fallback(state)

def _handle_multi_dimension_exploration(state: WorkflowState, dimensions: List[str]) -> Dict[str, Any]:
    """
    Gère l'exploration de plusieurs dimensions simultanément
    Fonctionnalité importante du système original
    """
    logger.info(f"🔀 Multi-dimension exploration: {dimensions}")
    
    try:
        from ..prompts.pcm_first_interaction_prompts import build_pcm_first_interaction_multi_dimension_prompt
        
        # Construire le prompt multi-dimensions
        prompt = build_pcm_first_interaction_multi_dimension_prompt(state, dimensions)
        
        from ..common.llm_utils import isolated_analysis_call_with_messages
        
        user_query = state.get('user_message', '') or (
            state.get('messages', [{}])[-1].get('content', '') if state.get('messages') else 'Exploring multiple PCM dimensions'
        )
        
        response = isolated_analysis_call_with_messages(
            system_content=prompt,
            user_content=f"Multi-dimension exploration for: {user_query}"
        )
        
        return {
            **state,
            'analysis_result': response,
            'pcm_multi_dimension_exploration': True,
            'pcm_dimensions_being_explored': dimensions,
            'skip_search': False,  # Multi-dimensions nécessitent une recherche
            'needs_user_response': False
        }
        
    except Exception as e:
        logger.error(f"❌ Error in multi-dimension exploration: {e}")
        return state

def _simple_first_base_fallback(state: WorkflowState) -> Dict[str, Any]:
    """Fallback simple pour première interaction BASE"""
    fallback_message = """Welcome to exploring your PCM BASE! 🌟

Your BASE represents your core personality - how you naturally perceive the world and communicate.

The 6 key dimensions we can explore are:
• **Perception** - How you filter and interpret the world
• **Strengths** - Your natural talents and abilities  
• **Interaction Style** - How you connect with others
• **Personality Parts** - Your behavioral patterns
• **Channels of Communication** - Your communication preferences
• **Environmental Preferences** - Settings where you thrive

Which dimension interests you most?"""

    return {
        **state,
        'analysis_result': fallback_message,
        'skip_search': True,
        'pcm_first_base_interaction_complete': True
    }

def _simple_first_phase_fallback(state: WorkflowState) -> Dict[str, Any]:
    """Fallback simple pour première interaction PHASE"""
    fallback_message = """Let's explore your PCM PHASE! 

Your PHASE represents your current motivational needs and stress responses.

Would you like to:
1. Explore your current PHASE and motivational needs
2. Start with your BASE personality foundation first

What feels more relevant to you right now?"""

    return {
        **state,
        'analysis_result': fallback_message,
        'needs_user_response': True,
        'skip_search': True,
        'pcm_first_phase_interaction_complete': True
    }

# Fonction de migration facile
def migrate_to_new_system():
    """
    Instructions pour migrer vers le nouveau système PCM
    
    Dans votre workflow principal, vous pouvez choisir:
    
    OPTION 1 - Remplacement complet:
    ```python
    from modules.pcm.pcm_analysis_new import (
        pcm_analysis_with_flow_manager as pcm_analysis,
        pcm_vector_search_with_flow_manager as pcm_vector_search
    )
    ```
    
    OPTION 2 - Coexistence (recommandé pour tests):
    Garder l'ancien système et tester le nouveau en parallèle
    """
    pass