"""
PCM Conversation Handler - Gère les flux conversationnels multi-étapes
Inspiré du système MBTI mais adapté aux spécificités PCM
"""
import logging
from typing import Dict, Any, Optional, Tuple
from ..common.types import WorkflowState

logger = logging.getLogger(__name__)

class PCMConversationHandler:
    """Gestionnaire des conversations PCM multi-étapes"""
    
    # Étapes conversationnelles pour coworker
    COWORKER_STAGES = {
        'INITIAL': 'Question initiale sur collègue',
        'ASK_MY_STATE': 'Demander comment je vais',
        'ANALYZE_MY_STATE': 'Analyser ma réponse sur mon état',
        'ASK_OTHER_BASE': 'Demander la base du collègue',
        'ASK_OTHER_STATE': 'Demander l\'état du collègue',
        'ASK_OTHER_PHASE': 'Demander la phase du collègue',
        'FINAL_RESPONSE': 'Générer la réponse finale'
    }
    
    @staticmethod
    def handle_coworker_conversation(state: WorkflowState, classification: Dict[str, Any]) -> Dict[str, Any]:
        """
        Gère la conversation coworker multi-étapes
        Retourne soit une question à poser, soit prépare la recherche finale
        """
        logger.info("👥 Handling coworker conversation")
        
        # Déterminer l'étape actuelle de la conversation
        current_stage = state.get('coworker_conversation_stage', 'INITIAL')
        logger.info(f"📍 Current coworker stage: {current_stage}")
        
        # Router selon l'étape
        if current_stage == 'INITIAL':
            return PCMConversationHandler._handle_initial_coworker(state)
        elif current_stage == 'ASK_MY_STATE':
            return PCMConversationHandler._handle_my_state_response(state)
        elif current_stage == 'ASK_OTHER_BASE':
            return PCMConversationHandler._handle_other_base_response(state)
        elif current_stage == 'ASK_OTHER_STATE':
            return PCMConversationHandler._handle_other_state_response(state)
        elif current_stage == 'ASK_OTHER_PHASE':
            return PCMConversationHandler._handle_other_phase_response(state)
        else:
            # Étape inconnue, revenir au début
            return PCMConversationHandler._handle_initial_coworker(state)
    
    @staticmethod
    def _handle_initial_coworker(state: WorkflowState) -> Dict[str, Any]:
        """Première étape : demander comment je vais"""
        logger.info("👥 Initial coworker stage - asking about user's state")
        
        question = "Avant de parler de votre collègue, comment allez-vous aujourd'hui ? Y a-t-il des tensions ou défis particuliers que vous ressentez ?"
        
        return {
            **state,
            'flow_type': 'coworker_focused',
            'coworker_conversation_stage': 'ASK_MY_STATE',
            'conversational_question': question,
            'needs_user_response': True,
            'skip_final_generation': True  # Ne pas générer de réponse finale maintenant
        }
    
    @staticmethod
    def _handle_my_state_response(state: WorkflowState) -> Dict[str, Any]:
        """Analyser ma réponse sur mon état et poser la question suivante"""
        logger.info("👥 Analyzing user's state response")
        
        user_response = PCMConversationHandler._get_last_user_message(state)
        my_state = PCMConversationHandler._analyze_user_wellbeing(user_response)
        
        logger.info(f"👤 User state analyzed as: {my_state}")
        
        if my_state == 'struggling':
            # Si je vais mal, noter qu'il faut chercher mon plan d'action plus tard
            question = "Je comprends que vous traversez une période difficile. Nous reviendrons sur des stratégies pour vous plus tard. Maintenant, pouvez-vous me décrire votre collègue ? Est-il plutôt logique et analytique, empathique et chaleureux, créatif et spontané, ou orienté action et résultats ?"
            need_my_action_plan = True
        else:
            # Si je vais bien, passer directement au collègue
            question = "Parfait ! Maintenant, pouvez-vous me décrire votre collègue ? Est-il plutôt logique et analytique, empathique et chaleureux, créatif et spontané, ou orienté action et résultats ?"
            need_my_action_plan = False
        
        return {
            **state,
            'coworker_conversation_stage': 'ASK_OTHER_BASE',
            'coworker_my_state': my_state,
            'coworker_need_my_action_plan': need_my_action_plan,
            'conversational_question': question,
            'needs_user_response': True,
            'skip_final_generation': True
        }
    
    @staticmethod
    def _handle_other_base_response(state: WorkflowState) -> Dict[str, Any]:
        """Analyser la base du collègue et demander son état"""
        logger.info("👥 Analyzing colleague's base")
        
        user_response = PCMConversationHandler._get_last_user_message(state)
        other_base = PCMConversationHandler._detect_pcm_base(user_response)
        
        if not other_base:
            # Base pas claire, redemander
            question = "Pouvez-vous être plus précis ? Par exemple : préfère-t-il analyser les faits et données, ou se concentre-t-il sur les relations et émotions ? Prend-il des décisions rapidement ou réfléchit-il longuement ?"
            return {
                **state,
                'conversational_question': question,
                'needs_user_response': True,
                'skip_final_generation': True
                # Reste sur la même étape
            }
        
        logger.info(f"👤 Colleague base identified: {other_base}")
        
        # Base identifiée, demander l'état du collègue
        base_description = PCMConversationHandler._get_base_friendly_name(other_base)
        question = f"D'accord, votre collègue semble être de type {base_description}. Comment va-t-il en ce moment ? Percevez-vous du stress ou des tensions chez lui, ou semble-t-il détendu et dans son élément ?"
        
        return {
            **state,
            'coworker_conversation_stage': 'ASK_OTHER_STATE',
            'coworker_other_base': other_base,
            'conversational_question': question,
            'needs_user_response': True,
            'skip_final_generation': True
        }
    
    @staticmethod
    def _handle_other_state_response(state: WorkflowState) -> Dict[str, Any]:
        """Analyser l'état du collègue et décider de la suite"""
        logger.info("👥 Analyzing colleague's state")
        
        user_response = PCMConversationHandler._get_last_user_message(state)
        other_state = PCMConversationHandler._analyze_other_wellbeing(user_response)
        
        logger.info(f"👤 Colleague state: {other_state}")
        
        if other_state == 'good':
            # Il va bien → Réponse finale avec ma base + sa base
            return {
                **state,
                'coworker_conversation_stage': 'FINAL_RESPONSE',
                'coworker_other_state': other_state,
                'search_strategy': 'my_base_other_base',
                'needs_user_response': False,
                'skip_final_generation': False  # Prêt pour la génération finale
            }
        
        elif other_state in ['struggling', 'unknown']:
            # Il va mal ou je ne sais pas → Demander sa phase
            if other_state == 'struggling':
                question = "Je vois qu'il traverse une période difficile. Pouvez-vous me décrire ses comportements récents ? Est-il plus irritable, renfermé, perfectionniste, émotionnel, ou au contraire très autoritaire ?"
            else:  # unknown
                question = "Pas de problème, analysons quand même ses comportements pour mieux le comprendre. Comment se comporte-t-il généralement ? Plutôt calme, expressif, méticuleux, créatif, ou direct ?"
            
            return {
                **state,
                'coworker_conversation_stage': 'ASK_OTHER_PHASE',
                'coworker_other_state': other_state,
                'conversational_question': question,
                'needs_user_response': True,
                'skip_final_generation': True
            }
    
    @staticmethod
    def _handle_other_phase_response(state: WorkflowState) -> Dict[str, Any]:
        """Analyser la phase du collègue et préparer la réponse finale"""
        logger.info("👥 Analyzing colleague's phase")
        
        user_response = PCMConversationHandler._get_last_user_message(state)
        other_phase = PCMConversationHandler._detect_stress_phase(user_response)
        
        logger.info(f"👤 Colleague phase: {other_phase}")
        
        # Préparer la réponse finale avec ma base/phase + sa base + sa phase
        search_strategy = 'comprehensive_coworker_support'
        if state.get('coworker_need_my_action_plan'):
            search_strategy = 'mutual_stress_management'
        
        return {
            **state,
            'coworker_conversation_stage': 'FINAL_RESPONSE',
            'coworker_other_phase': other_phase,
            'search_strategy': search_strategy,
            'needs_user_response': False,
            'skip_final_generation': False  # Prêt pour la génération finale
        }
    
    @staticmethod
    def _get_last_user_message(state: WorkflowState) -> str:
        """Récupère le dernier message utilisateur"""
        messages = state.get('messages', [])
        if messages:
            last_msg = messages[-1]
            if hasattr(last_msg, 'content'):
                return last_msg.content
            elif isinstance(last_msg, dict):
                return last_msg.get('content', '')
            return str(last_msg)
        return state.get('user_message', '')
    
    @staticmethod
    def _analyze_user_wellbeing(response: str) -> str:
        """Analyse si l'utilisateur va bien ou mal"""
        response_lower = response.lower()
        
        # Indicateurs de difficulté
        struggling_indicators = [
            'pas bien', 'mal', 'stress', 'stressé', 'difficile', 'problème', 
            'tension', 'conflit', 'fatigué', 'épuisé', 'débordé', 'anxieux',
            'frustrated', 'stressed', 'overwhelmed', 'tired', 'difficult', 
            'struggling', 'hard', 'tough', 'problem', 'issue'
        ]
        
        # Indicateurs positifs
        positive_indicators = [
            'bien', 'ça va', 'good', 'fine', 'ok', 'okay', 'great', 'parfait',
            'super', 'excellent', 'motivé', 'énergique', 'positive', 'motivated'
        ]
        
        # Vérifier d'abord les indicateurs négatifs (plus spécifiques)
        for indicator in struggling_indicators:
            if indicator in response_lower:
                return 'struggling'
        
        # Puis les positifs
        for indicator in positive_indicators:
            if indicator in response_lower:
                return 'good'
        
        # Par défaut, considérer comme "ça va"
        return 'neutral'
    
    @staticmethod
    def _analyze_other_wellbeing(response: str) -> str:
        """Analyse l'état du collègue selon la description"""
        response_lower = response.lower()
        
        # Indicateurs qu'il va mal
        struggling_indicators = [
            'stress', 'tendu', 'irritable', 'difficile', 'problème', 'mal',
            'agressif', 'renfermé', 'fatigué', 'débordé', 'frustrated', 
            'stressed', 'tense', 'difficult', 'struggling'
        ]
        
        # Indicateurs qu'il va bien
        good_indicators = [
            'bien', 'détendu', 'serein', 'motivé', 'positif', 'énergique',
            'collaboratif', 'ouvert', 'good', 'relaxed', 'positive', 
            'motivated', 'energetic', 'collaborative'
        ]
        
        # Indicateurs d'incertitude
        unknown_indicators = [
            'sais pas', 'ne sais pas', 'incertain', 'difficile à dire',
            'don\'t know', 'not sure', 'uncertain', 'hard to say'
        ]
        
        # Vérifier dans l'ordre de priorité
        for indicator in unknown_indicators:
            if indicator in response_lower:
                return 'unknown'
        
        for indicator in struggling_indicators:
            if indicator in response_lower:
                return 'struggling'
        
        for indicator in good_indicators:
            if indicator in response_lower:
                return 'good'
        
        return 'unknown'
    
    @staticmethod
    def _detect_pcm_base(response: str) -> Optional[str]:
        """Détecte la base PCM du collègue"""
        response_lower = response.lower()
        
        base_indicators = {
            'thinker': [
                'logique', 'analytique', 'faits', 'données', 'rationnel',
                'méthodique', 'systématique', 'objectif', 'analyse',
                'logical', 'analytical', 'facts', 'data', 'rational', 'methodical'
            ],
            'harmonizer': [
                'empathique', 'chaleureux', 'émotionnel', 'relations', 'humain',
                'bienveillant', 'attentionné', 'sensible', 'caring', 'empathy',
                'warm', 'emotional', 'relationships', 'people-focused'
            ],
            'persister': [
                'valeurs', 'convictions', 'opinions', 'principes', 'déterminé',
                'persévérant', 'moral', 'éthique', 'beliefs', 'values',
                'determined', 'persistent', 'principled'
            ],
            'rebel': [
                'créatif', 'spontané', 'fun', 'original', 'inventif', 'libre',
                'indépendant', 'artistique', 'creative', 'spontaneous',
                'innovative', 'artistic', 'independent'
            ],
            'promoter': [
                'action', 'résultats', 'efficace', 'direct', 'pragmatique',
                'fonceur', 'décisif', 'results', 'action-oriented',
                'efficient', 'decisive', 'practical'
            ],
            'imaginer': [
                'calme', 'réservé', 'discret', 'patient', 'réfléchi',
                'tranquille', 'contemplatif', 'calm', 'quiet', 'reserved',
                'patient', 'thoughtful', 'contemplative'
            ]
        }
        
        # Compter les matches
        scores = {}
        for base, keywords in base_indicators.items():
            scores[base] = sum(1 for keyword in keywords if keyword in response_lower)
        
        # Retourner la base avec le plus de matches (minimum 1)
        if max(scores.values()) > 0:
            return max(scores, key=scores.get)
        
        return None
    
    @staticmethod
    def _detect_stress_phase(response: str) -> str:
        """Détecte la phase de stress du collègue (simplifié)"""
        response_lower = response.lower()
        
        phase_indicators = {
            'thinker': ['perfectionniste', 'critique', 'détails', 'méticuleux', 'obsessionnel'],
            'harmonizer': ['émotionnel', 'pleure', 'sensible', 'blessé', 'triste'],
            'persister': ['rigide', 'obstiné', 'moralisateur', 'intransigeant'],
            'rebel': ['isolé', 'boudeur', 'passif', 'démotivé', 'désengagé'],
            'promoter': ['impatient', 'autoritaire', 'brutal', 'agressif', 'dominant'],
            'imaginer': ['confus', 'indécis', 'paralysé', 'submergé', 'perdu']
        }
        
        for phase, keywords in phase_indicators.items():
            for keyword in keywords:
                if keyword in response_lower:
                    return phase
        
        return 'general_stress'
    
    @staticmethod
    def _get_base_friendly_name(base: str) -> str:
        """Retourne le nom convivial de la base PCM"""
        friendly_names = {
            'thinker': 'Thinker (analytique et logique)',
            'harmonizer': 'Harmonizer (empathique et chaleureux)', 
            'persister': 'Persister (déterminé et orienté valeurs)',
            'rebel': 'Rebel (créatif et spontané)',
            'promoter': 'Promoter (orienté action et résultats)',
            'imaginer': 'Imaginer (calme et réfléchi)'
        }
        return friendly_names.get(base, base.title())