"""
PCM Analysis V2 - Version améliorée avec détection transitions
Remplace pcm_analysis_new.py avec de meilleures transitions PHASE → COWORKER
"""
import logging
from typing import Dict, Any, List
from ..common.types import WorkflowState
from .pcm_flow_manager import PCMFlowManager

logger = logging.getLogger(__name__)

def pcm_analysis_with_flow_manager(state: WorkflowState) -> Dict[str, Any]:
    """
    Point d'entrée PCM V2 - Même interface que pcm_analysis_new mais avec transitions améliorées
    
    AMÉLIORATIONS V2:
    - ✅ Détection transitions PHASE → COWORKER quand on mentionne un collègue  
    - ✅ Garde toute la logique existante de pcm_analysis_new
    - ✅ Compatible avec l'architecture actuelle
    """
    logger.info("🎯 PCM Analysis V2 - Enhanced Transitions")
    
    # ÉTAPE 0: PRIORITÉ ABSOLUE - Vérification guardrails multicouches AVANT TOUT
    try:
        from .pcm_safety_guardrail import check_workplace_safety, AdvancedSafetyGuard, PCMSafetyGuard
        
        # 🔬 NIVEAU 1: Check de sécurité avancée (contenu illégal/dangereux + contexte)
        logger.info("🔬 Advanced safety check: Detecting illegal/harmful content + contextual continuations...")
        advanced_guard = AdvancedSafetyGuard()
        user_message = state.get('user_message', '')
        
        # 🧠 Extraire le contexte de conversation pour analyse contextuelle
        conversation_context = []
        for msg in state.get('messages', [])[-5:]:  # 5 derniers messages pour contexte
            if hasattr(msg, 'type') and hasattr(msg, 'content'):
                conversation_context.append({
                    "role": "user" if msg.type == 'human' else "assistant",
                    "content": msg.content
                })
            elif isinstance(msg, dict):
                conversation_context.append({
                    "role": msg.get('role', 'user'),
                    "content": msg.get('content', '')
                })
        
        if user_message:
            safety_result = advanced_guard.comprehensive_safety_check(user_message, conversation_context)
            
            if not safety_result.get('is_safe', True):
                logger.error(f"🚫 ADVANCED SAFETY BLOCKED: {safety_result}")
                safety_issues = safety_result.get('safety_issues', [])
                risk_level = safety_result.get('risk_level', 'UNKNOWN')
                
                # Créer un résultat de guardrail pour handle_special_flows
                advanced_guardrail_result = {
                    'is_safe': False,
                    'flow_type': 'safety_refusal',
                    'action': 'REFUSE_NON_WORKPLACE',  # ✅ AJOUT de l'action attendue
                    'safety_message': f"I cannot assist with this request due to safety concerns. Risk level: {risk_level}. Please ensure your question is appropriate and constructive.",
                    'safety_details': safety_issues,
                    'advanced_safety_blocked': True
                }
                
                return _handle_special_flows(state, advanced_guardrail_result)
            
            logger.info(f"✅ Advanced safety check passed - Risk: {safety_result.get('risk_level', 'LOW')}")
        
        # 🎯 NIVEAU 2: Check du scope PCM (workspace + personnel vs spécialiste) AVEC CONTEXTE
        logger.info("🎯 PCM scope check: Workplace + personal development vs specialist (with context)...")
        pcm_guard = PCMSafetyGuard(use_llm_validation=True)
        pcm_result = pcm_guard.validate(user_message, conversation_context) if user_message else {"validation_passed": True}
        
        if not pcm_result.get('validation_passed', True):
            logger.warning("🚫 PCM Scope Guardrail blocked request - requires specialist")
            pcm_guardrail_result = {
                'is_safe': False,
                'flow_type': 'safety_refusal',
                'action': 'REFUSE_NON_WORKPLACE',  # ✅ AJOUT de l'action attendue
                'safety_message': "I'm sorry, but this topic requires specialist professional help beyond my PCM coaching scope. Please consult an appropriate specialist for your specific need.",
                'pcm_scope_blocked': True
            }
            return _handle_special_flows(state, pcm_guardrail_result)
            
        logger.info("✅ PCM Scope check passed - within PCM development scope")
        logger.info("✅ All guardrail checks passed - request approved")
        
    except Exception as e:
        logger.warning(f"⚠️ Guardrail check failed: {e}")
        # En cas d'erreur des guardrails, continuer avec une approche conservatrice
        
    # ÉTAPE 0.5: Classification d'intent (PCM flow uniquement, sécurité déjà gérée)
    try:
        classification = PCMFlowManager.classify_pcm_intent(state)
        logger.info(f"✅ Intent classification completed: {classification.get('flow_type', 'unknown')}")
        
    except Exception as e:
        logger.warning(f"⚠️ Intent classification failed: {e}")
        # Fallback: continuer avec un flow général
        classification = {
            'flow_type': 'SEARCH_GENERAL_THEORY',
            'action': 'SEARCH_GENERAL_THEORY',
            'confidence': 0.5,
            'reasoning': 'Fallback due to classification error'
        }
    
    # Étape 1: Classification avec PCMFlowManager (comme avant)
    flow_type = state.get('flow_type', '')
    logger.info(f"🔍 DEBUG: Initial flow_type from state: '{flow_type}'")
    
    if not flow_type:
        try:
            classification = PCMFlowManager.classify_pcm_intent(state)
            flow_type = classification.get('flow_type', 'SELF_BASE')
            language = classification.get('language', 'en')
            logger.info(f"🤖 PCMFlowManager classification: {flow_type}, language: {language}")
            state['language'] = language
            
            # IMPORTANT: Si c'est un GREETING, le traiter immédiatement
            if flow_type == 'GREETING':
                logger.info("👋 Greeting detected - handling immediately")
                return _handle_special_flows(state, classification)
                
        except Exception as e:
            logger.warning(f"⚠️ PCMFlowManager classification failed: {e}")
            flow_type = 'SELF_BASE'  # Default fallback
    
    # Étape 2: AMÉLIORATION V2 - Détecter transitions dynamiques
    detected_transition = _detect_dynamic_transitions(state)
    
    if detected_transition:
        logger.info(f"🔄 Dynamic transition detected: {detected_transition['from']} → {detected_transition['to']}")
        # Basculer vers le nouveau flow
        flow_type = detected_transition['to']
        state = {
            **state,
            'flow_type': flow_type_mapping.get(flow_type, flow_type.lower()),
            'transition_detected': detected_transition,
            'transition_message': detected_transition.get('message', '')
        }
    
    # Étape 3: Router selon le flow_type (logique existante améliorée)
    logger.info(f"🔍 DEBUG: Checking flow_type routing for: '{flow_type}'")
    
    if flow_type in ['COMPARISON', 'COWORKER', 'TEAM', 'GENERAL_PCM', 'GREETING']:
        logger.info(f"🎯 Special flow detected: {flow_type} - using flow manager")
        # Pour les special flows, on doit d'abord faire la classification pour avoir l'action
        try:
            classification = PCMFlowManager.classify_pcm_intent(state)
            return _handle_special_flows(state, classification)
        except Exception as e:
            logger.error(f"❌ Error classifying special flow {flow_type}: {e}")
            return _fallback_analysis(state)
    
    elif flow_type in ['SELF_ACTION_PLAN', 'self_action_plan']:
        logger.info(f"🎯 Action Plan flow: {flow_type} - checking for coworker mentions first")
        
        # AMÉLIORATION V2: Si on est déjà en mode gathering, continuer sans re-détecter
        if state.get('coworker_gathering_info', False):
            logger.info("🔄 ALREADY in coworker gathering mode - incrementing counter")
            new_attempts = state.get('coworker_info_attempts', 0) + 1
            state['coworker_info_attempts'] = new_attempts
            
            # AMÉLIORATION V2: Après 2 attempts, marquer comme prêt pour l'analyse
            if new_attempts >= 2:
                state['ready_for_coworker_analysis'] = True
                logger.info("🎯 READY FOR COWORKER ANALYSIS after 2 attempts")
            
            # Continuer en mode gathering avec conversational analysis
            from .pcm_conversational_analysis import analyze_pcm_conversational_intent
            conversational_result = analyze_pcm_conversational_intent(state)
            return {
                **conversational_result,
                'coworker_context_detected': True,
                'coworker_gathering_info': True,
                'coworker_info_attempts': new_attempts,
                'ready_for_coworker_analysis': new_attempts >= 2,
                'flow_type': 'self_action_plan',
                'pcm_base_or_phase': 'action_plan'
            }
        
        # AMÉLIORATION V3: Analyser avec LLM intelligence pure - PAS de mots-clés !  
        from .pcm_conversational_analysis import analyze_pcm_conversational_intent
        conversational_result = analyze_pcm_conversational_intent(state)
        primary_suggestion = conversational_result.get('pcm_transition_suggestions', {}).get('primary_suggestion', {})
        
        if primary_suggestion.get('action') == 'suggest_coworker_transition':
            logger.info("🎯🎯🎯 COWORKER MENTION DETECTED IN ACTION_PLAN - SWITCHING TO INFO GATHERING 🎯🎯🎯")
            logger.info(f"🎯 ACTION_PLAN with coworker transition: {primary_suggestion.get('message', '')}")
            # Marquer qu'on va gatherer des infos sur le collègue
            state['coworker_context_detected'] = True
            state['coworker_gathering_info'] = True
            new_attempts = state.get('coworker_info_attempts', 0) + 1
            state['coworker_info_attempts'] = new_attempts
            
            # AMÉLIORATION V2: Après 2 attempts, marquer comme prêt pour l'analyse
            if new_attempts >= 2:
                state['ready_for_coworker_analysis'] = True
                logger.info("🎯 READY FOR COWORKER ANALYSIS after 2 attempts")
            
            # Rester en action_plan mais avec le mode gathering
            return {
                **conversational_result,
                'coworker_context_detected': True,
                'coworker_gathering_info': True,
                'coworker_info_attempts': new_attempts,
                'ready_for_coworker_analysis': new_attempts >= 2,
                'flow_type': 'self_action_plan',  # S'assurer que le flow_type reste correct
                'pcm_base_or_phase': 'action_plan'
            }
        else:
            # Pas de mention de collègue, continuer avec ACTION_PLAN normal
            return _handle_with_flow_manager(state)
    
    elif flow_type == 'self_focused' and state.get('pcm_conversational_context', {}).get('current_context') == 'action_plan':
        logger.info(f"🎯 Self-focused Action Plan detected - using flow manager")
        return _handle_with_flow_manager(state)
    
    elif flow_type in ['SELF_BASE', 'SELF_PHASE', 'self_focused']:
        logger.info(f"🔄 Self-focused flow: {flow_type} - using conversational analysis") 
        return _handle_self_focused_flows(state, flow_type)
    
    else:
        logger.info("📊 Using standard PCM flow manager")
        return _handle_with_flow_manager(state)


def _detect_dynamic_transitions(state: WorkflowState) -> Dict[str, Any]:
    """
    AMÉLIORATION V2: Détecte les transitions dynamiques
    """
    messages = state.get('messages', [])
    if not messages:
        return None
        
    current_message = messages[-1]
    user_message = ""
    if hasattr(current_message, 'content'):
        user_message = current_message.content.lower()
    elif isinstance(current_message, dict):
        user_message = current_message.get('content', '').lower()
    
    previous_flow = state.get('flow_type', '')
    
    # Note: COWORKER transitions supprimées - on fait confiance au PCMFlowManager
    
    # RÈGLE: PHASE + demande conseils → ACTION_PLAN  
    if previous_flow in ['self_phase', 'SELF_PHASE'] or state.get('pcm_base_or_phase') == 'phase':
        action_keywords = ['what should i do', 'recommendations', 'conseils', 'aide-moi', 'help me']
        if any(keyword in user_message for keyword in action_keywords):
            return {
                'from': 'SELF_PHASE',
                'to': 'SELF_ACTION_PLAN',
                'action': 'SEARCH_MY_ACTION_PLAN', 
                'reason': 'User asking for advice while exploring phase',
                'message': 'Transitioning to action planning'
            }
    
    return None


def _handle_special_flows(state: WorkflowState, classification: Dict[str, Any]) -> Dict[str, Any]:
    """Gère les flux spéciaux (COWORKER, COMPARISON, SAFETY_REFUSAL, etc.)"""
    try:
        # PRIORITÉ 0: Gestion des refus de sécurité
        if classification.get('flow_type') == 'SAFETY_REFUSAL':
            logger.warning("🚫 SAFETY REFUSAL - Blocking request")
            reasoning = classification.get('reasoning', '')
            if 'familial' in reasoning.lower() or 'personnel' in reasoning.lower() or 'family' in reasoning.lower():
                safety_message = "I'm sorry but I am not able to answer this. As a Zest Companion specialized in PCM for workplace communication, I cannot provide advice about family or personal relationships. Please consult an appropriate specialist for family relationship guidance."
            else:
                safety_message = "I'm sorry but I am not able to answer this. As a Zest Companion specialized in PCM for workplace communication, this topic is outside my scope. Please consult an appropriate specialist."
            
            return {
                **state,
                'flow_type': 'safety_refusal',
                'skip_search': True,
                'safety_message': safety_message,
                'pcm_analysis_done': True
            }
        # AMÉLIORATION V2: Logique intelligente pour COWORKER
        if classification.get('flow_type') == 'COWORKER':
            # Détecter si c'est une TRANSITION depuis self_focused ou un COWORKER direct
            message_count = len(state.get('messages', []))
            previous_flow = state.get('flow_type', '')
            
            # Si transition depuis self_focused → Questions intermédiaires
            if (message_count > 1 and 
                previous_flow in ['self_action_plan', 'self_phase', 'self_base', 'self_focused']):
                logger.info("🎯🎯🎯 COWORKER TRANSITION FROM SELF_FOCUSED → INFO GATHERING MODE 🎯🎯🎯")
                
                # Retourner DIRECTEMENT en mode gathering SANS appeler analyze_pcm_conversational_intent
                return {
                    **state,
                    'flow_type': previous_flow,  # Garder le contexte self
                    'coworker_context_detected': True,
                    'coworker_gathering_info': True,
                    'coworker_info_attempts': 1,
                    'pcm_classification': classification,
                    'pcm_analysis_done': True
                }
            else:
                # COWORKER direct → Flow normal avec step 3
                logger.info("🎯 COWORKER direct → Normal flow with step 3")
                state['coworker_step'] = 1
                state['coworker_self_ok'] = True
        
        # Pour tous les autres cas, utiliser le flow manager normal
        updated_state = PCMFlowManager.execute_flow_action(classification, state)
        return {
            **updated_state,
            'pcm_classification': classification,
            'pcm_flow_manager_used': True
        }
    except Exception as e:
        logger.error(f"❌ Error in special flow: {e}")
        return _fallback_analysis(state)


def _handle_self_focused_flows(state: WorkflowState, flow_type: str) -> Dict[str, Any]:
    """Gère les flux self-focused avec analyse conversationnelle"""
    try:
        # Utiliser l'analyse conversationnelle existante
        from .pcm_conversational_analysis import analyze_pcm_conversational_intent
        
        # S'assurer que le contexte est bien configuré
        if flow_type == 'SELF_BASE':
            state['pcm_base_or_phase'] = 'base'
            state['flow_type'] = 'self_base'
        elif flow_type == 'SELF_PHASE':
            state['pcm_base_or_phase'] = 'phase'  
            state['flow_type'] = 'self_phase'
        
        result = analyze_pcm_conversational_intent(state)
        
        # AMÉLIORATION V2: Gérer les suggestions de transition de manière intelligente
        primary_suggestion = result.get('pcm_transition_suggestions', {}).get('primary_suggestion', {})
        
        if primary_suggestion.get('action') == 'suggest_coworker_transition':
            logger.info("🔄 Conversational analysis suggests COWORKER transition - but staying in self_focused for context gathering")
            logger.info(f"🔄 Transition reason: {primary_suggestion.get('message', '')}")
            
            # NOUVEAU: Rester en self_focused mais marquer qu'on explore le contexte coworker
            state['coworker_context_detected'] = True
            state['coworker_gathering_info'] = True
            state['coworker_info_attempts'] = state.get('coworker_info_attempts', 0) + 1
            result['coworker_context_detected'] = True
            result['coworker_gathering_info'] = True
            
            # Garder le flow type actuel mais ajouter le contexte
            logger.info("🎯 Staying in self_focused flow but gathering coworker context details")
            
            # Après 2-3 échanges, proposer de passer à l'analyse du collègue
            if state.get('coworker_info_attempts', 0) >= 2:
                logger.info("🎯 Sufficient context gathered - ready to transition to coworker analysis")
                state['ready_for_coworker_analysis'] = True
                result['ready_for_coworker_analysis'] = True
            
        elif primary_suggestion.get('action') == 'suggest_action_plan':
            logger.info("🔄 Conversational analysis suggests ACTION_PLAN transition - executing immediately")
            logger.info(f"🔄 Transition reason: {primary_suggestion.get('message', '')}")
            # Configurer le state pour ACTION_PLAN mais RESTER dans l'analyse conversationnelle
            # pour préserver le contexte de phase spécifique
            state['flow_type'] = 'self_action_plan'
            state['pcm_base_or_phase'] = 'action_plan'
            # Relancer l'analyse conversationnelle avec le nouveau contexte ACTION_PLAN
            result = analyze_pcm_conversational_intent(state)
            
        elif primary_suggestion.get('action') == 'suggest_phase_transition':
            logger.info("🔄 Conversational analysis suggests PHASE transition - executing immediately")
            logger.info(f"🔄 Transition reason: {primary_suggestion.get('message', '')}")
            # Configurer le state pour PHASE et continuer avec l'analyse conversationnelle
            state['flow_type'] = 'self_phase'
            state['pcm_base_or_phase'] = 'phase'
            # Pas besoin de rediriger vers PCMFlowManager, on reste dans l'analyse conversationnelle
            # Juste mettre à jour le contexte et continuer
        
        # AMÉLIORATION V2: Enrichir avec suggestions de transition si détectées
        if state.get('transition_detected'):
            transition = state['transition_detected']
            if not result.get('pcm_transition_suggestions'):
                result['pcm_transition_suggestions'] = {}
            
            result['pcm_transition_suggestions']['primary_suggestion'] = {
                'action': f"suggest_{transition['to'].lower()}_transition",
                'message': transition.get('message', ''),
                'context_switch': flow_type_mapping.get(transition['to'], transition['to'].lower())
            }
        
        return result
        
    except ImportError:
        logger.warning("⚠️ Conversational analysis not available - using fallback")
        return _fallback_analysis(state)


def _handle_with_flow_manager(state: WorkflowState) -> Dict[str, Any]:
    """Fallback vers PCMFlowManager standard"""
    try:
        classification = PCMFlowManager.classify_pcm_intent(state)
        updated_state = PCMFlowManager.execute_flow_action(classification, state)
        
        # IMPORTANT: Toujours préserver la classification pour les tools LangGraph
        logger.info(f"🎯 Classification créée: {classification.get('flow_type')} → {classification.get('action')}")
        
        return {
            **updated_state,
            'pcm_classification': classification,
            'pcm_flow_manager_used': True
        }
    except Exception as e:
        logger.error(f"❌ Error in flow manager: {e}")
        return _fallback_analysis(state)


def _fallback_analysis(state: WorkflowState) -> Dict[str, Any]:
    """Fallback ultime en cas d'erreur"""
    logger.warning("🔄 Using ultimate fallback analysis")
    return {
        **state,
        'flow_type': 'self_base',
        'pcm_base_or_phase': 'base',
        'error': 'PCM analysis failed - using fallback',
        'fallback_used': True
    }


# Mapping des types de flux
flow_type_mapping = {
    'SELF_BASE': 'self_base',
    'SELF_PHASE': 'self_phase', 
    'SELF_ACTION_PLAN': 'self_action_plan',
    'COWORKER': 'coworker_focused',
    'COMPARISON': 'comparison',
    'GENERAL_PCM': 'general_pcm',
    'GREETING': 'greeting'
}


# Fonction de compatibilité pour la recherche vectorielle  
def pcm_vector_search_with_flow_manager(state: WorkflowState) -> Dict[str, Any]:
    """Recherche vectorielle - compatible avec V2"""
    logger.info("🔍 PCM Vector Search V2")
    
    # Utiliser la recherche du module renommé
    from .pcm_vector_search import pcm_vector_search
    return pcm_vector_search(state)