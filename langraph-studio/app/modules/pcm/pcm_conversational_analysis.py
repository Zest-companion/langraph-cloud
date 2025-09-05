"""
PCM Conversational Analysis - Système intelligent 3-contextes
Gère les transitions dynamiques entre BASE/PHASE/ACTION_PLAN dans self_focused
"""

import logging
import json
from typing import Dict, Any, List, Optional, Tuple
from langchain_core.messages import AIMessage

from ..common.types import WorkflowState
from ..common.llm_utils import isolated_analysis_call_with_messages

logger = logging.getLogger(__name__)

class PCMConversationalContext:
    """Gère le contexte conversationnel PCM avec 3 sous-types pour self_focused"""
    
    BASE = "base"           # Comprendre sa base et les 6 dimensions
    PHASE = "phase"         # Comprendre comment on réagit sous stress actuellement
    ACTION_PLAN = "action_plan"  # Conseils situationnels et plan d'action
    
    # Ordre naturel de progression
    PROGRESSION_ORDER = [BASE, PHASE, ACTION_PLAN]
    
    # Dimensions BASE à tracker
    BASE_DIMENSIONS = [
        "perception", "strengths", "interaction_style", 
        "personality_part", "channel_communication", "environmental_preferences"
    ]

def analyze_pcm_conversational_intent(state: WorkflowState) -> Dict[str, Any]:
    """
    Système conversationnel intelligent pour self_focused
    Détecte BASE/PHASE/ACTION_PLAN et gère les transitions
    """
    logger.info("🎯 Starting PCM Conversational Analysis")
    
    messages = state.get('messages', [])
    if not messages:
        return state
        
    # Obtenir le message utilisateur actuel
    current_message = messages[-1]
    if hasattr(current_message, 'content'):
        user_query = current_message.content
    elif isinstance(current_message, dict):
        user_query = current_message.get('content', '')
    else:
        user_query = str(current_message)
    
    # Construire le contexte conversationnel
    conversation_context = _build_conversation_context(messages)
    previous_context = state.get('pcm_conversational_context', {})
    
    logger.info(f"📝 User query: '{user_query}'")
    logger.info(f"🔄 Previous context: {previous_context}")
    
    # Debug: Log conversation context pour identifier le problème
    logger.info(f"🔍 DEBUG: Conversation context being sent to LLM:\n{conversation_context}")
    
    # Analyser avec Chain of Thought
    analysis_result = _analyze_with_chain_of_thought(
        user_query=user_query,
        conversation_context=conversation_context,
        previous_context=previous_context,
        user_profile={
            'pcm_base': state.get('pcm_base', ''),
            'pcm_phase': state.get('pcm_phase', ''),
            'explored_dimensions': state.get('pcm_explored_dimensions', [])
        }
    )
    
    if not analysis_result.get('success', False):
        logger.warning("⚠️ Analysis failed, using fallback")
        return _fallback_context(state, user_query)
    
    conversational_context = analysis_result['conversational_context']
    reasoning = analysis_result['reasoning_process']
    
    logger.info(f"✅ Detected context: {conversational_context['current_context']}")
    logger.info(f"📊 Dimensions covered: {conversational_context.get('dimensions_covered', [])}")
    
    # Mettre à jour les dimensions explorées
    updated_state = _update_explored_dimensions(state, conversational_context)
    
    # Créer les suggestions de transition
    transition_suggestions = _create_transition_suggestions(conversational_context, updated_state)
    
    return {
        **updated_state,
        'pcm_conversational_context': conversational_context,
        'pcm_context_reasoning': reasoning,
        'pcm_transition_suggestions': transition_suggestions,
        'flow_type': 'self_focused',  # Force self_focused context
        'pcm_base_or_phase': conversational_context['current_context'],  # Compatibility
        'pcm_specific_dimensions': conversational_context.get('dimensions_covered', []) if conversational_context.get('dimensions_covered') else None,
        'conversational_analysis_complete': True
    }

def _build_conversation_context(messages: List) -> str:
    """Construit le contexte conversationnel pour l'analyse"""
    if len(messages) <= 1:
        return "Première interaction PCM."
    
    # Prendre les 3 derniers messages pour le contexte
    recent_messages = messages[-3:-1]  # Exclure le message actuel
    
    context_lines = []
    for msg in recent_messages:
        if hasattr(msg, 'type'):
            role = "User" if msg.type == "human" else "Assistant"
            content = msg.content
        else:
            role = "User" if msg.get('type') == "human" else "Assistant"
            content = msg.get('content', '')
        
        # Truncate mais garder les mots-clés PCM
        if len(content) > 300:
            content = content[:300] + "..."
        
        context_lines.append(f"{role}: {content}")
    
    return "\n".join(context_lines)

def _analyze_with_chain_of_thought(
    user_query: str,
    conversation_context: str,
    previous_context: Dict[str, Any],
    user_profile: Dict[str, Any]
) -> Dict[str, Any]:
    """Analyse avec Chain of Thought pour déterminer le contexte conversationnel"""
    
    # Construire le prompt de raisonnement
    system_prompt = f"""Tu es un expert PCM spécialisé dans l'analyse conversationnelle. Analyse cette interaction dans le contexte self_focused.

CONTEXTE UTILISATEUR:
PCM BASE: {user_profile.get('pcm_base', 'Non spécifié')}
PCM PHASE: {user_profile.get('pcm_phase', 'Non spécifié')}
Dimensions explorées: {user_profile.get('explored_dimensions', [])}

CONTEXTE PRÉCÉDENT:
{json.dumps(previous_context, indent=2) if previous_context else "Aucun contexte précédent"}

HISTORIQUE CONVERSATIONNEL:
{conversation_context}

QUESTION ACTUELLE: "{user_query}"

RAISONNEMENT ÉTAPE PAR ÉTAPE:

1. ANALYSE DU CONTEXTE CONVERSATIONNEL:
   - Que disait l'utilisateur dans les messages précédents ?
   - Y a-t-il une continuité avec la conversation ?
   - Si c'est "oui", "continue", "raconte-moi", à quoi ça se réfère ?

2. DÉTECTION DU SOUS-CONTEXTE SELF_FOCUSED:
   - **BASE**: L'utilisateur veut comprendre sa personnalité de base, ses dimensions naturelles
   - **PHASE**: L'utilisateur veut comprendre son état actuel, stress, besoins motivationnels
     * **MOTS-CLÉS PHASE**: "my phase", "and my phase", "et ma phase", "ma phase", "understand my phase", "current phase", "explain my phase", "what is my phase", "phase", "phase?", "stress", "current state", "how am I doing", "feeling", "motivational needs"
   - **ACTION_PLAN**: L'utilisateur demande des conseils, plans d'action pour une situation
   
   **⚠️ RÈGLES CRITIQUES POUR DÉTECTION PHASE:**
   - Si l'utilisateur dit "my phase", "and my phase", "et ma phase", "understand my phase", "explain my phase", "current phase", "what is my phase", "phase?" = TOUJOURS PHASE
   - Ces expressions ont PRIORITÉ ABSOLUE sur BASE
   - MÊME SANS CONTEXTE COMPLET (ex: juste "and my phase" ou "my phase?") = TOUJOURS PHASE
   
   **🚨 RÈGLE CRITIQUE - RÉPONSES POSITIVES AUX SUGGESTIONS PHASE:**
   - Si le message précédent de l'assistant contenait une suggestion d'explorer la PHASE (mots comme "ready to explore your PHASE", "Prêt à explorer votre PHASE", "besoins motivationnels actuels", "current motivational needs")
   - ET que l'utilisateur répond positivement ("oui", "yes", "oui je suis prete", "I'm ready", "ready", "sure", "d'accord", "ok")
   - ALORS: current_context = "phase"
   - Cette règle a PRIORITÉ ABSOLUE sur les autres détections
   
   **⚠️ RÈGLES CRITIQUES POUR ACTION_PLAN:**
   - Si l'utilisateur dit "recommendations", "advice", "actions to take", "what should I do", "help me with", "how can I", "conseils", "que faire", "aide-moi" = ACTION_PLAN
   - MÊME SI le contexte précédent était PHASE, ces mots indiquent une TRANSITION vers ACTION_PLAN
   - ACTION_PLAN a PRIORITÉ sur PHASE quand l'utilisateur demande explicitement des conseils/actions
   
   **🚨 RÈGLES CRITIQUES POUR TRANSITION COWORKER:**
   - **UNIQUEMENT** si l'utilisateur mentionne UNE PERSONNE SPÉCIFIQUE (manager, un collègue précis, chef, boss identifié)
   - **OBLIGATOIRE**: Écrire EXACTEMENT "TRANSITION vers COWORKER" dans suggested_next_steps
   - **EXEMPLE COWORKER**: "Can you help me with a situation I have with my manager?" → suggested_next_steps: ["TRANSITION vers COWORKER"]
   - **EXEMPLE COWORKER**: "My boss micromanages me" → suggested_next_steps: ["TRANSITION vers COWORKER"]
   
   **EXCLUSIONS - RESTENT EN SELF:**
   - **GROUPES VAGUES**: "My colleagues stress me" → RESTER EN SELF (pas de personne spécifique)
   - **SITUATIONS PUBLIQUES**: "Je n'ose pas parler en public" → RESTER EN SELF (pas de relation 1-à-1)
   - **SENTIMENTS GÉNÉRAUX**: "Je ne suis pas à l'aise avec mes collègues" → RESTER EN SELF
   
   **EXEMPLES CRITIQUES:**
   - "I'm stressed" = PHASE
   - "I would like to understand my phase" = PHASE
   - "What is my current phase?" = PHASE
   - "Explain my phase" = PHASE
   - "and my phase" = PHASE (MÊME SANS CONTEXTE)
   - "et ma phase" = PHASE (MÊME SANS CONTEXTE)
   - "my phase?" = PHASE
   - "and my phase?" = PHASE
   - "I'm stressed because of my manager" = PHASE + suggestion TRANSITION COWORKER (personne spécifique)
   - "I'm stressed because of my colleagues" = PHASE (groupe vague, RESTER EN SELF)
   - "Je n'ose pas parler en public" = PHASE ou ACTION_PLAN (situation publique, RESTER EN SELF)
   - "J'ai peur de dire des bêtises devant plein de gens" = PHASE (stress social général, RESTER EN SELF)
   - "I'm stressed, what should I do?" = ACTION_PLAN (transition)
   - "But I would like recommendations" = ACTION_PLAN (transition claire)
   - "He keeps sending me messages during the weekend" = ACTION_PLAN + suggestion TRANSITION COWORKER (personne spécifique)
   - "My colleagues keep interrupting me" = PHASE (groupe vague, RESTER EN SELF)
   - "Can you help me with a situation I have with my manager?" = ACTION_PLAN + suggestion TRANSITION COWORKER (personne spécifique)

3. DÉTECTION DES DIMENSIONS (si BASE):
   
   **⚠️ RÈGLE ABSOLUE: dimensions_covered = UNIQUEMENT la/les dimension(s) demandée(s) dans LE MESSAGE ACTUEL**
   
   **PROCESSUS DE DÉTECTION:**
   1. Regarder "Dimensions explorées" dans le contexte utilisateur = dimensions DÉJÀ explorées avant
   2. Analyser LE MESSAGE ACTUEL de l'utilisateur pour identifier quelle(s) dimension(s) il demande MAINTENANT
   3. dimensions_covered = SEULEMENT la/les nouvelle(s) dimension(s) demandée(s) MAINTENANT
   4. NE JAMAIS inclure les dimensions déjà explorées dans dimensions_covered
   
   **EXEMPLES CRITIQUES:**
   - Si "Dimensions explorées: ['Perception']" et l'utilisateur dit "parlons du channel of communication"
     → dimensions_covered = ["channel_communication"] (PAS "perception"!)
   
   - Si "Dimensions explorées: []" et l'utilisateur dit "je veux explorer ma perception"
     → dimensions_covered = ["perception"]
   
   - Si l'utilisateur dit juste "oui" ou "continue"
     → dimensions_covered = [] (pas de nouvelle dimension demandée)
   
   **LES 6 DIMENSIONS BASE (avec mots-clés pour détection):**
   
   • **perception** = Comment filtrer/interpréter le monde
     Mots-clés: perception, percevoir, voir le monde, filtrer, interpréter, thoughts, feelings, actions
   
   • **strengths** = Talents naturels, forces
     Mots-clés: forces, talents, capacités, strengths, points forts, atouts
   
   • **interaction_style** = Style relationnel (comment vous collaborez)
     Mots-clés: interaction style, relationnel, collaborer, travailler avec, style social, teamwork, collaboration
   
   • **personality_part** = Patterns comportementaux
     Mots-clés: personality parts, comportements, patterns, façons d'être, énergie
   
   • **channel_communication** = Canaux de communication
     Mots-clés: communication channel, channels of communication, canal de communication, façon de parler, s'exprimer, communiquer, tone of voice, non-verbal, gestures
   
   • **environmental_preferences** = Préférences environnementales
     Mots-clés: environnement, préférences, settings, contexte, ambiance, lieu de travail

4. ÉVALUATION DES TRANSITIONS:
   - L'utilisateur est-il prêt à passer de BASE → PHASE ?
   - Ou de PHASE → ACTION_PLAN ?
   - Veut-il approfondir le contexte actuel ?
   
   **⚠️ DÉTECTION PRIORITAIRE PHASE → ACTION_PLAN:**
   - Si PREVIOUS context était PHASE ET current message contient des mots ACTION_PLAN (recommendations, advice, actions, what should I, conseils, aide-moi) → current_context = "action_plan"
   - Cette transition est PLUS IMPORTANTE que la continuité contextuelle

5. GESTION CRITIQUE DES RÉPONSES COURTES:
   - Si réponse très courte (<15 caractères): OBLIGATOIREMENT analyser le message précédent de l'assistant
   
   **RÈGLES STRICTES POUR "YES/OUI" :**
   - Si assistant a posé une QUESTION DE VALIDATION ("Does this resonate?", "Does this feel right?", "Do you recognize yourself?"):
     * Réponse "yes/oui" = CONFIRMATION que l'explication est correcte
     * current_context = RESTE LE MÊME (base ou phase)
     * context_change = false
     * Suggérer d'explorer une autre dimension BASE ou approfondir
     * ⚠️ NE PAS PASSER À PHASE automatiquement !
   
   - Si assistant a posé une QUESTION DE CONTINUATION ("Would you like to continue?", "Want to explore more?"):
     * Réponse "yes/oui" = DEMANDE d'approfondissement  
     * Continuer dans le même contexte
   
   **RÈGLE ABSOLUE**: Une confirmation "yes" après validation BASE ≠ désir de passer à PHASE !

DÉCISION FINALE:

Tu DOIS répondre UNIQUEMENT avec un objet JSON valide, sans texte avant ou après.

Exemple de réponse attendue:
{{
    "current_context": "base",
    "confidence": 0.85,
    "dimensions_covered": ["perception", "strengths"],
    "context_change": false,
    "transition_readiness": {{
        "to_phase": false,
        "to_action_plan": false,
        "reasoning": "Utilisateur veut encore explorer sa BASE"
    }},
    "suggested_next_steps": [
        "Continue exploring BASE dimensions",
        "Ask about interaction style"
    ],
    "reasoning": "L'utilisateur pose des questions sur sa personnalité de base"
}}

**EXEMPLE TRANSITION PHASE → ACTION_PLAN:**
{{
    "current_context": "action_plan",
    "confidence": 0.90,
    "dimensions_covered": [],
    "context_change": true,
    "transition_readiness": {{
        "to_phase": false,
        "to_action_plan": true,
        "reasoning": "Utilisateur demande conseils/actions"
    }},
    "suggested_next_steps": ["Provide action plan"],
    "reasoning": "Message contient 'recommendations' + 'actions to take' → ACTION_PLAN prioritaire"
}}

Maintenant, analyse cette conversation et réponds UNIQUEMENT avec le JSON structuré:"""
    
    try:
        logger.info("🤔 Executing conversational Chain of Thought...")
        reasoning_result = isolated_analysis_call_with_messages(
            system_content=system_prompt,
            user_content="Analyse cette interaction conversationnelle PCM."
        )
        
        logger.info(f"✅ Chain of Thought completed: {len(reasoning_result)} chars")
        
        # Extraire le JSON
        conversational_context = _extract_json_from_reasoning(reasoning_result)
        
        return {
            "conversational_context": conversational_context,
            "reasoning_process": reasoning_result,
            "success": True
        }
        
    except Exception as e:
        logger.error(f"❌ Conversational analysis failed: {e}")
        return {
            "conversational_context": _get_fallback_conversational_context(),
            "reasoning_process": f"Error: {str(e)}",
            "success": False
        }

def _extract_json_from_reasoning(reasoning_text: str) -> Dict[str, Any]:
    """Extrait le JSON de conclusion du raisonnement"""
    try:
        # Nettoyer le texte
        reasoning_text = reasoning_text.strip()
        
        # Tentative 1: JSON direct (le LLM répond uniquement avec JSON)
        if reasoning_text.startswith("{") and reasoning_text.endswith("}"):
            logger.info("🎯 Direct JSON detected")
            return json.loads(reasoning_text)
        
        # Tentative 2: Chercher JSON avec marqueurs
        start_markers = ["```json", "```JSON", "```", "{"]
        
        for start_marker in start_markers:
            start_idx = reasoning_text.find(start_marker)
            if start_idx != -1:
                if start_marker == "{":
                    # Chercher l'objet JSON complet
                    brace_count = 0
                    for i, char in enumerate(reasoning_text[start_idx:]):
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                json_str = reasoning_text[start_idx:start_idx + i + 1]
                                logger.info("📋 Extracted JSON from braces")
                                return json.loads(json_str)
                else:
                    # Avec marqueurs markdown
                    start_idx += len(start_marker)
                    end_idx = reasoning_text.find("```", start_idx)
                    if end_idx != -1:
                        json_str = reasoning_text[start_idx:end_idx].strip()
                        logger.info("📋 Extracted JSON from markdown")
                        return json.loads(json_str)
        
        # Tentative 3: Regex pour trouver un objet JSON valide
        import re
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_pattern, reasoning_text, re.DOTALL)
        for match in matches:
            try:
                result = json.loads(match)
                logger.info("📋 Extracted JSON via regex")
                return result
            except:
                continue
        
        raise ValueError("No valid JSON found in reasoning")
            
    except Exception as e:
        logger.error(f"❌ JSON extraction error: {e}")
        logger.info(f"🔍 Reasoning text (first 500 chars): {reasoning_text[:500]}")
        return _get_fallback_conversational_context()

def _get_fallback_conversational_context() -> Dict[str, Any]:
    """Contexte de fallback en cas d'erreur"""
    return {
        "current_context": "base",
        "confidence": 0.5,
        "dimensions_covered": [],
        "context_change": False,
        "transition_readiness": {
            "to_phase": False,
            "to_action_plan": False,
            "reasoning": "Fallback context used"
        },
        "suggested_next_steps": ["Continue exploring BASE dimensions"],
        "reasoning": "Fallback used due to analysis error"
    }

def _update_explored_dimensions(state: WorkflowState, conversational_context: Dict[str, Any]) -> Dict[str, Any]:
    """Met à jour les dimensions explorées selon le contexte
    
    IMPORTANT: Une dimension n'est marquée comme explorée que si elle a été 
    vraiment discutée en détail, pas juste mentionnée ou présentée comme option.
    """
    current_explored = state.get('pcm_explored_dimensions', [])
    dimensions_covered = conversational_context.get('dimensions_covered', [])
    confidence = conversational_context.get('confidence', 0)
    
    # Si confidence < 0.7, on ne marque pas les dimensions comme explorées
    # (probablement juste une présentation ou mention)
    if confidence < 0.7:
        logger.info(f"⚠️ Confidence too low ({confidence}), not marking dimensions as explored")
        return state
    
    # Si c'est la première interaction (liste vide et on détecte plusieurs dimensions d'un coup)
    # Ne pas marquer comme explorées (c'est probablement juste la présentation)
    if not current_explored and len(dimensions_covered) >= 4:
        logger.info("⚠️ First interaction with many dimensions detected - likely just presentation, not marking as explored")
        return state
    
    if not dimensions_covered:
        return state
    
    # Mapping des noms techniques vers noms d'affichage
    dimension_mapping = {
        "perception": "Perception",
        "strengths": "Strengths", 
        "interaction_style": "Interaction Style",
        "personality_part": "Personality Parts",
        "channel_communication": "Channels of Communication",
        "environmental_preferences": "Environmental Preferences"
    }
    
    updated_explored = current_explored.copy()
    
    # Ne marquer comme explorée que si c'est 1-2 dimensions max à la fois
    # (indique une vraie exploration, pas une présentation)
    if len(dimensions_covered) <= 2:
        for dimension in dimensions_covered:
            if dimension in dimension_mapping:
                display_name = dimension_mapping[dimension]
                if display_name not in updated_explored:
                    updated_explored.append(display_name)
                    logger.info(f"✅ Added {display_name} to explored dimensions (detailed exploration)")
    else:
        logger.info(f"⚠️ {len(dimensions_covered)} dimensions mentioned - likely overview, not marking as explored")
    
    return {
        **state,
        'pcm_explored_dimensions': updated_explored
    }

def _create_transition_suggestions(conversational_context: Dict[str, Any], state: WorkflowState) -> Dict[str, Any]:
    """Crée les suggestions de transition intelligentes"""
    current_context = conversational_context.get('current_context', 'base')
    transition_readiness = conversational_context.get('transition_readiness', {})
    explored_dimensions = state.get('pcm_explored_dimensions', [])
    
    # Utiliser les dimensions du contexte conversationnel si disponibles
    dimensions_just_covered = conversational_context.get('dimensions_covered', [])
    
    # Mapping local pour les suggestions
    dimension_mapping = {
        "perception": "Perception",
        "strengths": "Strengths", 
        "interaction_style": "Interaction Style",
        "personality_part": "Personality Parts",
        "channel_communication": "Channels of Communication",
        "environmental_preferences": "Environmental Preferences"
    }
    
    suggestions = {
        "primary_suggestion": None,
        "alternative_suggestions": [],
        "reasoning": ""
    }
    
    if current_context == "base":
        # Priorité 1: Si le système détecte une readiness pour PHASE transition
        if transition_readiness.get('to_phase', False):
            suggestions["primary_suggestion"] = {
                "action": "suggest_phase_transition",
                "message": "I sense you're getting a good grasp of your BASE. Would you like to explore your PHASE now - understanding your current motivational needs and how you react under stress?",
                "context_switch": "phase"
            }
            # Ajouter alternative pour continuer BASE si l'utilisateur préfère
            remaining_dimensions = [d for d in PCMConversationalContext.BASE_DIMENSIONS 
                                 if dimension_mapping.get(d, d) not in explored_dimensions]
            if remaining_dimensions:
                suggestions["alternative_suggestions"].append({
                    "action": "continue_base_dimensions",
                    "message": f"Or continue exploring BASE dimensions: {', '.join(remaining_dimensions[:3])}",
                    "context_switch": None
                })
        # Priorité 2: Si on a exploré 4+ dimensions
        elif len(explored_dimensions) >= 4:
            suggestions["primary_suggestion"] = {
                "action": "suggest_phase_transition",
                "message": "You've explored several BASE dimensions. Ready to discover your PHASE - your current motivational needs?",
                "context_switch": "phase"
            }
            remaining_dimensions = [d for d in PCMConversationalContext.BASE_DIMENSIONS 
                                 if dimension_mapping.get(d, d) not in explored_dimensions]
            if remaining_dimensions:
                suggestions["alternative_suggestions"].append({
                    "action": "continue_base_dimensions",
                    "message": f"Or explore remaining BASE dimensions: {', '.join(remaining_dimensions[:3])}",
                    "context_switch": None
                })
        else:
            # Moins de 4 dimensions explorées - continuer BASE normalement
            remaining_dimensions = [d for d in PCMConversationalContext.BASE_DIMENSIONS 
                                 if dimension_mapping.get(d, d) not in explored_dimensions]
            if remaining_dimensions:
                # Adapter le message selon si c'est la première exploration ou non
                if len(explored_dimensions) == 0:
                    # Première exploration - ne pas dire "another"
                    message = f"Would you like to explore a BASE dimension? Available: {', '.join(remaining_dimensions[:3])}"
                else:
                    # Déjà exploré au moins une dimension
                    message = f"Would you like to explore another BASE dimension? Available: {', '.join(remaining_dimensions[:3])}"
                
                suggestions["primary_suggestion"] = {
                    "action": "continue_base_dimensions",
                    "message": message,
                    "context_switch": None
                }
    
    elif current_context == "phase":
        # Détecter si suggestion COWORKER est présente
        coworker_suggestion = any("TRANSITION vers COWORKER" in step or 
                                "coworker" in step.lower() or 
                                "colleague" in step.lower() 
                                for step in conversational_context.get('suggested_next_steps', []))
        
        if coworker_suggestion:
            # Priorité 1: Suggérer transition vers COWORKER si détectée
            suggestions["primary_suggestion"] = {
                "action": "suggest_coworker_transition",
                "message": "It seems like the stress you're experiencing might be related to workplace relationships. Would you like to explore how to better understand and work with your colleague? This could help you develop strategies for both managing your own stress and improving the working relationship.",
                "context_switch": "coworker_focused"
            }
            suggestions["alternative_suggestions"].append({
                "action": "continue_phase_exploration", 
                "message": "Or would you prefer to continue exploring your personal stress patterns first?",
                "context_switch": None
            })
        elif transition_readiness.get('to_action_plan', False):
            suggestions["primary_suggestion"] = {
                "action": "suggest_action_plan",
                "message": "Maintenant que vous comprenez mieux votre PHASE, voulez-vous que je vous aide avec un plan d'action pour une situation spécifique ?",
                "context_switch": "action_plan"
            }
        else:
            suggestions["primary_suggestion"] = {
                "action": "continue_phase_exploration",
                "message": "Souhaitez-vous approfondir votre compréhension de votre PHASE actuelle ?",
                "context_switch": None
            }
    
    elif current_context == "action_plan":
        # AMÉLIORATION V3: DEBUG - voir ce que le LLM écrit vraiment !
        reasoning = conversational_context.get('reasoning', '')
        suggested_steps = conversational_context.get('suggested_next_steps', [])
        logger.info(f"🔍 DEBUG ACTION_PLAN Chain of Thought reasoning: {reasoning}")
        logger.info(f"🔍 DEBUG ACTION_PLAN suggested_next_steps: {suggested_steps}")
        
        # AMÉLIORATION V5: Laisser le LLM écrire EXACTEMENT ce qu'on lui demande !
        # Si le LLM écrit "TRANSITION vers COWORKER" dans suggested_next_steps, on l'utilise !
        coworker_suggestion = any("TRANSITION vers COWORKER" in step for step in suggested_steps)
        
        logger.info(f"🔍 DEBUG: Checking for 'TRANSITION vers COWORKER' in steps: {coworker_suggestion}")
        logger.info(f"🔍 DEBUG: Steps content: {suggested_steps}")
        
        if coworker_suggestion:
            # Priorité 1: Suggérer transition vers COWORKER si détectée
            suggestions["primary_suggestion"] = {
                "action": "suggest_coworker_transition",
                "message": "It seems like the stress you're experiencing might be related to workplace relationships. Would you like to explore how to better understand and work with your colleague/manager? This could help you develop strategies for both managing your own stress and improving the working relationship.",
                "context_switch": "coworker_focused"
            }
            suggestions["alternative_suggestions"].append({
                "action": "continue_action_plan", 
                "message": "Or would you prefer to continue with your current action plan first?",
                "context_switch": None
            })
        else:
            suggestions["primary_suggestion"] = {
                "action": "refine_action_plan",
                "message": "Voulez-vous affiner ce plan d'action ou explorer une autre situation ?",
                "context_switch": None
            }
            suggestions["alternative_suggestions"].append({
                "action": "return_to_base_or_phase",
                "message": "Ou préférez-vous revenir à l'exploration de votre BASE ou PHASE ?",
                "context_switch": "base"
            })
    
    # Mapping pour compatibilité
    dimension_mapping = {
        "perception": "Perception",
        "strengths": "Strengths", 
        "interaction_style": "Interaction Style",
        "personality_part": "Personality Parts",
        "channel_communication": "Channels of Communication",
        "environmental_preferences": "Environmental Preferences"
    }
    
    return suggestions

def _fallback_context(state: WorkflowState, user_query: str) -> Dict[str, Any]:
    """Contexte de fallback simple"""
    return {
        **state,
        'pcm_conversational_context': _get_fallback_conversational_context(),
        'pcm_context_reasoning': "Fallback context used due to analysis failure",
        'pcm_transition_suggestions': {
            "primary_suggestion": {
                "action": "continue_base_exploration",
                "message": "Continuons à explorer votre personnalité BASE.",
                "context_switch": None
            },
            "alternative_suggestions": [],
            "reasoning": "Fallback suggestion"
        },
        'flow_type': 'self_focused',
        'pcm_base_or_phase': 'base',
        'conversational_analysis_complete': True
    }