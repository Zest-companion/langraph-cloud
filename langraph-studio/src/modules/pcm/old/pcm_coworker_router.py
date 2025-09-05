"""
PCM Coworker Flow Router - Simplified MBTI-style routing
"""
import logging
from typing import Dict, Any, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)

class CoworkerState(Enum):
    """États possibles dans le flux coworker"""
    INITIAL = "initial"
    CHECK_SELF = "check_self"  # Step 2.1: Comment je vais?
    SELF_OK = "self_ok"  # Je vais bien
    SELF_NOT_OK = "self_not_ok"  # Je ne vais pas bien - Step 2.2
    ASK_OTHER_BASE = "ask_other_base"  # Step 3.1: Demander la base de l'autre
    CHECK_OTHER_STATE = "check_other_state"  # Step 3.1.5: Est-ce qu'il va bien?
    OTHER_OK = "other_ok"  # Il va bien
    OTHER_NOT_OK = "other_not_ok"  # Il ne va pas bien - Step 3.2
    OTHER_UNKNOWN = "other_unknown"  # Je ne sais pas
    ANALYZE_OTHER_PHASE = "analyze_other_phase"  # Step 3.2: Analyser sa phase
    FINAL_RESPONSE = "final_response"  # Step 4: Réponse finale avec action plans

class CoworkerRouter:
    """Router simplifié pour le flux coworker PCM"""
    
    @staticmethod
    def get_current_state(state: Dict[str, Any]) -> CoworkerState:
        """Détermine l'état actuel basé sur le state"""
        coworker_step = state.get('coworker_step', 1)
        coworker_substep = state.get('coworker_step_2_substep', 1)
        self_ok = state.get('coworker_self_ok')
        other_profile = state.get('coworker_other_profile', {})
        
        # Mapping des steps vers les états
        if coworker_step == 1:
            return CoworkerState.INITIAL
        elif coworker_step == 2:
            if coworker_substep == 1:
                return CoworkerState.CHECK_SELF
            elif coworker_substep == 2:
                return CoworkerState.SELF_NOT_OK if not self_ok else CoworkerState.SELF_OK
        elif coworker_step == 3:
            if not other_profile.get('base_type'):
                return CoworkerState.ASK_OTHER_BASE
            elif not other_profile.get('state_assessment'):
                return CoworkerState.CHECK_OTHER_STATE
            elif other_profile.get('is_ok'):
                return CoworkerState.OTHER_OK
            else:
                return CoworkerState.OTHER_NOT_OK
        elif coworker_step == 4:
            return CoworkerState.FINAL_RESPONSE
        
        return CoworkerState.INITIAL
    
    @staticmethod
    def get_next_action(
        current_state: CoworkerState,
        user_response: str,
        state: Dict[str, Any]
    ) -> Tuple[CoworkerState, Dict[str, str]]:
        """
        Détermine la prochaine action basée sur l'état actuel et la réponse
        Retourne: (next_state, routing_instructions)
        """
        routing = {}
        
        if current_state == CoworkerState.INITIAL:
            # Début du flux - demander comment je vais
            next_state = CoworkerState.CHECK_SELF
            routing = {
                "action": "ASK_SELF_STATE",
                "message": "Analysons d'abord comment vous allez",
                "search": "NONE"
            }
            
        elif current_state == CoworkerState.CHECK_SELF:
            # Analyser la réponse sur comment je vais
            self_ok = CoworkerRouter._assess_self_state(user_response)
            if self_ok:
                next_state = CoworkerState.ASK_OTHER_BASE
                routing = {
                    "action": "PROCEED_TO_OTHER",
                    "message": "Bien! Maintenant parlons de votre collègue",
                    "search": "NONE"
                }
            else:
                next_state = CoworkerState.SELF_NOT_OK
                routing = {
                    "action": "ANALYZE_SELF",
                    "message": "Je vois. Analysons votre situation",
                    "search": "USER_BASE_PHASE_ACTION"
                }
                
        elif current_state == CoworkerState.SELF_NOT_OK:
            # Après analyse de ma situation, passer à l'autre
            next_state = CoworkerState.ASK_OTHER_BASE
            routing = {
                "action": "TRANSITION_TO_OTHER",
                "message": "Maintenant, parlons de votre collègue",
                "search": "NONE"
            }
            
        elif current_state == CoworkerState.ASK_OTHER_BASE:
            # Analyser la base du collègue
            other_base = CoworkerRouter._detect_other_base(user_response, state)
            if other_base:
                next_state = CoworkerState.CHECK_OTHER_STATE
                routing = {
                    "action": "ASK_OTHER_STATE",
                    "message": f"Collègue identifié comme {other_base}. Comment va-t-il?",
                    "search": "NONE"
                }
            else:
                # Redemander si pas clair
                routing = {
                    "action": "CLARIFY_OTHER_BASE",
                    "message": "Pouvez-vous préciser le type de personnalité?",
                    "search": "GENERAL_PCM"
                }
                
        elif current_state == CoworkerState.CHECK_OTHER_STATE:
            # Analyser comment va le collègue
            other_state = CoworkerRouter._assess_other_state(user_response)
            
            if other_state == "ok":
                next_state = CoworkerState.OTHER_OK
                routing = {
                    "action": "ANALYZE_BOTH_OK",
                    "message": "Parfait! Analysons la collaboration",
                    "search": "USER_BASE_OTHER_BASE"
                }
            elif other_state == "not_ok":
                next_state = CoworkerState.ANALYZE_OTHER_PHASE
                routing = {
                    "action": "ASK_OTHER_PHASE",
                    "message": "Identifions sa phase de stress",
                    "search": "OTHER_PHASE_DETECTION"
                }
            else:  # unknown
                next_state = CoworkerState.OTHER_UNKNOWN
                routing = {
                    "action": "SUGGEST_PHASE_ANALYSIS",
                    "message": "Analysons quand même sa possible phase",
                    "search": "OTHER_PHASE_DETECTION"
                }
                
        elif current_state == CoworkerState.ANALYZE_OTHER_PHASE:
            # Après identification de la phase du collègue
            next_state = CoworkerState.FINAL_RESPONSE
            routing = {
                "action": "GENERATE_ACTION_PLANS",
                "message": "Voici les plans d'action adaptés",
                "search": "USER_BASE_OTHER_PHASE_ACTION"
            }
            
        elif current_state in [CoworkerState.OTHER_OK, CoworkerState.OTHER_NOT_OK, CoworkerState.OTHER_UNKNOWN]:
            # États terminaux - générer réponse finale
            next_state = CoworkerState.FINAL_RESPONSE
            other_profile = state.get('coworker_other_profile', {})
            
            if other_profile.get('phase_state'):
                search_type = "USER_BASE_OTHER_PHASE_ACTION"
            else:
                search_type = "USER_BASE_OTHER_BASE"
                
            routing = {
                "action": "FINAL_RECOMMENDATIONS",
                "message": "Voici mes recommandations finales",
                "search": search_type
            }
        
        return next_state, routing
    
    @staticmethod
    def _assess_self_state(response: str) -> bool:
        """Évalue si l'utilisateur va bien"""
        positive_indicators = ['bien', 'good', 'ok', 'ça va', 'fine', 'great', 'parfait', 'super']
        negative_indicators = ['pas bien', 'mal', 'stress', 'difficile', 'problème', 'tension', 'conflit']
        
        response_lower = response.lower()
        
        # Check for negative first (they're more specific)
        for indicator in negative_indicators:
            if indicator in response_lower:
                return False
        
        # Then check for positive
        for indicator in positive_indicators:
            if indicator in response_lower:
                return True
        
        # Default to asking for clarification
        return None
    
    @staticmethod
    def _detect_other_base(response: str, state: Dict[str, Any]) -> Optional[str]:
        """Détecte la base PCM du collègue"""
        pcm_bases = {
            'thinker': ['logique', 'analytique', 'facts', 'données', 'thinker', 'penseur'],
            'persister': ['valeur', 'opinion', 'conviction', 'persister', 'persévérant'],
            'harmonizer': ['empathie', 'sentiment', 'harmonizer', 'empathique'],
            'imaginer': ['calme', 'réfléchi', 'imaginer', 'imagineur'],
            'rebel': ['créatif', 'fun', 'spontané', 'rebel', 'rebelle'],
            'promoter': ['action', 'résultat', 'défi', 'promoter', 'promoteur']
        }
        
        response_lower = response.lower()
        
        for base, keywords in pcm_bases.items():
            for keyword in keywords:
                if keyword in response_lower:
                    return base
        
        return None
    
    @staticmethod
    def _assess_other_state(response: str) -> str:
        """Évalue l'état du collègue: ok, not_ok, unknown"""
        response_lower = response.lower()
        
        # Patterns for "ok"
        if any(word in response_lower for word in ['bien', 'good', 'ok', 'fine', 'ça va', 'parfait']):
            return "ok"
        
        # Patterns for "not ok"
        if any(word in response_lower for word in ['pas bien', 'mal', 'stress', 'difficile', 'tension']):
            return "not_ok"
        
        # Patterns for "unknown"
        if any(word in response_lower for word in ['sais pas', "don't know", 'incertain', 'peut-être']):
            return "unknown"
        
        return "unknown"

def route_coworker_flow(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fonction principale de routing pour le flux coworker
    Retourne les instructions de routing et met à jour le state
    """
    router = CoworkerRouter()
    
    # Get current state
    current_state = router.get_current_state(state)
    logger.info(f"🚦 Current coworker state: {current_state.value}")
    
    # Get user response
    messages = state.get('messages', [])
    user_response = ""
    if messages:
        last_message = messages[-1]
        if hasattr(last_message, 'content'):
            user_response = last_message.content
        else:
            user_response = str(last_message)
    
    # Get next action
    next_state, routing = router.get_next_action(current_state, user_response, state)
    logger.info(f"🎯 Next state: {next_state.value}, Action: {routing.get('action')}")
    
    # Update state with routing info
    return {
        **state,
        'coworker_current_state': current_state.value,
        'coworker_next_state': next_state.value,
        'coworker_routing': routing,
        'pcm_search_type': routing.get('search', 'NONE')
    }