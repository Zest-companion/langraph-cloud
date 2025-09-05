"""
PCM Unified Analysis - Système unifié d'analyse d'intent PCM
Remplace PCMFlowManager + PCMConversationalAnalysis par un seul système cohérent
"""
import logging
import json
from typing import Dict, Any, List, Optional, Tuple
from ..common.types import WorkflowState
from ..common.llm_utils import isolated_analysis_call_with_messages

logger = logging.getLogger(__name__)

class PCMUnifiedAnalysis:
    """Système unifié d'analyse PCM - Remplace les systèmes séparés"""
    
    # Contextes PCM possibles
    CONTEXT_TYPES = {
        'SELF_BASE': 'Exploration de sa base PCM personnelle',
        'SELF_PHASE': 'Exploration de sa phase/stress actuel', 
        'SELF_ACTION_PLAN': 'Plan d\'action basé sur PCM',
        'COWORKER': 'Relations avec collègues/coworkers',
        'COMPARISON': 'Comparaisons entre types PCM',
        'EXPLORATION': 'Exploration générale des 6 bases',
        'GENERAL_PCM': 'Théorie et concepts PCM',
        'GREETING': 'Salutations et remerciements'
    }
    
    # Dimensions BASE pour l'exploration fine
    BASE_DIMENSIONS = [
        "perception", "strengths", "interaction_style", 
        "personality_part", "channel_communication", "environmental_preferences"
    ]

    @staticmethod
    def analyze_pcm_unified_intent(state: WorkflowState) -> Dict[str, Any]:
        """
        Analyse unifiée complète - Classification globale + analyse fine + transitions
        """
        logger.info("🧠 STARTING PCM Unified Analysis - Single Source of Truth")
        
        messages = state.get('messages', [])
        if not messages:
            return state
            
        # Construire le contexte pour l'analyse
        analysis_context = PCMUnifiedAnalysis._build_analysis_context(state)
        
        # Analyse complète avec Chain of Thought
        try:
            analysis_result = PCMUnifiedAnalysis._unified_chain_of_thought_analysis(
                state=state,
                context=analysis_context
            )
            
            if not analysis_result.get('success', False):
                logger.warning("⚠️ Unified analysis failed, using fallback")
                return PCMUnifiedAnalysis._fallback_analysis(state)
            
            # Traiter les résultats
            return PCMUnifiedAnalysis._process_unified_results(state, analysis_result)
            
        except Exception as e:
            logger.error(f"❌ Error in unified analysis: {e}")
            return PCMUnifiedAnalysis._fallback_analysis(state)

    @staticmethod
    def _build_analysis_context(state: WorkflowState) -> Dict[str, Any]:
        """Construit le contexte complet pour l'analyse"""
        messages = state.get('messages', [])
        
        return {
            'user_message': state.get('user_message', '') or (
                messages[-1].get('content', '') if messages else ''
            ),
            'conversation_history': PCMUnifiedAnalysis._format_conversation_history(messages[-5:]),
            'user_profile': {
                'pcm_base': state.get('pcm_base', ''),
                'pcm_phase': state.get('pcm_phase', ''),
                'explored_dimensions': state.get('pcm_explored_dimensions', [])
            },
            'previous_context': {
                'flow_type': state.get('flow_type'),
                'pcm_base_or_phase': state.get('pcm_base_or_phase'),
                'conversational_context': state.get('pcm_conversational_context', {})
            },
            'conversation_length': len(messages)
        }

    @staticmethod
    def _unified_chain_of_thought_analysis(state: WorkflowState, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyse unifiée avec Chain of Thought - Remplace les deux systèmes"""
        
        system_prompt = f"""Tu es l'expert PCM unifié qui analyse TOUT en une seule passe.

## CONTEXTE UTILISATEUR
PCM BASE: {context['user_profile'].get('pcm_base', 'Non spécifié')}
PCM PHASE: {context['user_profile'].get('pcm_phase', 'Non spécifié')}
Dimensions explorées: {context['user_profile'].get('explored_dimensions', [])}

## CONTEXTE PRÉCÉDENT
Flow précédent: {context['previous_context'].get('flow_type', 'Aucun')}
Context PCM: {context['previous_context'].get('pcm_base_or_phase', 'Aucun')}

## HISTORIQUE CONVERSATIONNEL
{context['conversation_history']}

## MESSAGE ACTUEL
"{context['user_message']}"

## RAISONNEMENT UNIFIÉ (ANALYSE COMPLÈTE EN UNE PASSE)

### 1. CLASSIFICATION GLOBALE (remplace PCMFlowManager)
**Analyse le message pour déterminer le CONTEXTE PRINCIPAL:**

**SELF_BASE** = Questions sur ma base personnelle, mes dimensions naturelles
- "Ma base PCM", "Comment je fonctionne", "Mes forces", "Ma personnalité"
- Focus sur SOI sans mention stress/phase

**SELF_PHASE** = Questions sur mon stress/phase/état actuel
- "Mon stress", "Comment je gère", "Ma phase actuelle", "Mes besoins actuels"
- Indicateurs: stress, feeling, current, lately, motivated, energy

**SELF_ACTION_PLAN** = Demandes de conseils/actions
- "What should I do", "conseils", "recommendations", "aide-moi", "how can I"
- Transition depuis PHASE/BASE vers action concrète

**COWORKER** = Relations collègues (DÉTECTION CRITIQUE)
- "Mon collègue", "colleague", "coworker", "manager", "boss", "chef", "équipe"
- Relations au travail, conflits, collaboration

**COMPARISON** = Comparaisons entre types
- "Différence entre", "compare", "versus", "vs", "different than/from"
- Questions avec comparaisons explicites entre types PCM

**EXPLORATION** = Exploration générale
- "Les 6 bases", "All PCM types", "Show me everything", "Explore"

**GENERAL_PCM** = Théorie PCM
- "Qu'est-ce que PCM", "What is PCM", "PCM theory", "How does PCM work"

**GREETING** = Salutations
- "Bonjour", "Hello", "Thanks", "Merci", "Hi", "Good morning"

### 2. ANALYSE FINE (remplace PCMConversationalAnalysis)
**Si SELF_BASE/SELF_PHASE/SELF_ACTION_PLAN, analyser en détail:**

**Contexte spécifique:**
- base = Exploration des dimensions personnelles
- phase = Exploration stress/besoins actuels  
- action_plan = Conseils pratiques

**Dimensions BASE couvertes (si applicable):**
- Quelles dimensions sont demandées dans CE MESSAGE?
- NE PAS inclure les dimensions déjà explorées, SEULEMENT les nouvelles

**Détection de CONTINUITÉ:**
- "oui", "continue", "all", "tous", "the ones you mentioned" = continuer contexte précédent

### 3. DÉTECTION DE TRANSITIONS DYNAMIQUES (CRUCIAL)
**Règles de transition critique:**

**PHASE → COWORKER** (priorité absolue):
- Si contexte = PHASE ET mention "colleague"/"coworker"/"manager"/"boss"/"collègue"/"chef"
- → Suggérer transition vers COWORKER dans suggested_transitions

**PHASE → ACTION_PLAN**:
- Si contexte = PHASE ET "recommendations"/"conseils"/"what should I do"/"aide-moi"
- → Transition vers ACTION_PLAN

**BASE → PHASE**:
- Si 4+ dimensions BASE explorées ET utilisateur montre intérêt pour phase
- → Suggérer transition vers PHASE

### 4. DÉCISION FINALE
Tu DOIS répondre UNIQUEMENT avec un JSON structuré:

```json
{{
    "primary_context": "SELF_BASE|SELF_PHASE|SELF_ACTION_PLAN|COWORKER|COMPARISON|EXPLORATION|GENERAL_PCM|GREETING",
    "sub_context": "base|phase|action_plan" (si applicable),
    "confidence": 0.0-1.0,
    "dimensions_covered": ["perception", "strengths"] (si SELF_BASE),
    "context_change": true/false,
    "transition_detected": {{
        "from": "current_context",
        "to": "target_context",
        "reason": "explanation"
    }} (si transition détectée),
    "suggested_transitions": [
        {{
            "target": "COWORKER",
            "message": "It seems your stress is related to workplace relationships...",
            "priority": "high|medium|low"
        }}
    ],
    "continuity_detected": true/false,
    "reasoning": "Explication complète du raisonnement"
}}
```

**EXEMPLE CRITIQUE PHASE → COWORKER:**
Message: "I'm stressed because of my colleague"
→ primary_context: "SELF_PHASE", transition_detected: {{"from": "SELF_PHASE", "to": "COWORKER", "reason": "stress caused by colleague"}}, suggested_transitions: [...] 

Analyse maintenant cette interaction:"""

        try:
            response = isolated_analysis_call_with_messages(
                system_content=system_prompt,
                user_content="Analyse cette interaction PCM avec le système unifié."
            )
            
            # Extraire le JSON
            unified_result = PCMUnifiedAnalysis._extract_json_from_response(response)
            
            return {
                "unified_analysis": unified_result,
                "reasoning_process": response,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"❌ Unified analysis failed: {e}")
            return {
                "unified_analysis": {},
                "reasoning_process": f"Error: {str(e)}",
                "success": False
            }

    @staticmethod
    def _extract_json_from_response(response_text: str) -> Dict[str, Any]:
        """Extrait le JSON de la réponse unifiée"""
        try:
            # Même logique que dans l'ancien système
            response_text = response_text.strip()
            
            if response_text.startswith("{") and response_text.endswith("}"):
                return json.loads(response_text)
            
            # Chercher JSON avec marqueurs
            start_markers = ["```json", "```JSON", "```", "{"]
            
            for start_marker in start_markers:
                start_idx = response_text.find(start_marker)
                if start_idx != -1:
                    if start_marker == "{":
                        # Extraire objet JSON complet
                        brace_count = 0
                        for i, char in enumerate(response_text[start_idx:]):
                            if char == '{':
                                brace_count += 1
                            elif char == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    json_str = response_text[start_idx:start_idx + i + 1]
                                    return json.loads(json_str)
                    else:
                        start_idx += len(start_marker)
                        end_idx = response_text.find("```", start_idx)
                        if end_idx != -1:
                            json_str = response_text[start_idx:end_idx].strip()
                            return json.loads(json_str)
            
            raise ValueError("No valid JSON found")
            
        except Exception as e:
            logger.error(f"❌ JSON extraction failed: {e}")
            return PCMUnifiedAnalysis._get_fallback_result()

    @staticmethod
    def _process_unified_results(state: WorkflowState, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Traite les résultats de l'analyse unifiée en état LangGraph"""
        
        unified = analysis_result.get('unified_analysis', {})
        
        # Extraire les informations principales
        primary_context = unified.get('primary_context', 'SELF_BASE')
        sub_context = unified.get('sub_context', 'base')
        confidence = unified.get('confidence', 0.8)
        dimensions_covered = unified.get('dimensions_covered', [])
        
        # Détecter transitions
        transition_detected = unified.get('transition_detected')
        suggested_transitions = unified.get('suggested_transitions', [])
        
        # Mapper vers l'ancien format pour compatibilité
        flow_type_mapping = {
            'SELF_BASE': 'self_base',
            'SELF_PHASE': 'self_phase', 
            'SELF_ACTION_PLAN': 'self_action_plan',
            'COWORKER': 'coworker_focused',
            'COMPARISON': 'comparison',
            'EXPLORATION': 'exploration',
            'GENERAL_PCM': 'general_pcm',
            'GREETING': 'greeting'
        }
        
        flow_type = flow_type_mapping.get(primary_context, 'self_base')
        
        # Si transition détectée, utiliser le contexte cible
        if transition_detected and suggested_transitions:
            target_context = suggested_transitions[0].get('target', primary_context)
            flow_type = flow_type_mapping.get(target_context, flow_type)
            logger.info(f"🔄 Transition detected: {primary_context} → {target_context}")
        
        # Construire l'état mis à jour
        updated_state = {
            **state,
            'flow_type': flow_type,
            'pcm_base_or_phase': sub_context,
            'unified_analysis_complete': True,
            'pcm_unified_context': unified,
            'pcm_unified_reasoning': analysis_result.get('reasoning_process', ''),
            'pcm_confidence': confidence
        }
        
        # Ajouter les dimensions si applicables
        if dimensions_covered and primary_context in ['SELF_BASE']:
            updated_state['pcm_specific_dimensions'] = dimensions_covered
            
            # Mettre à jour les dimensions explorées
            current_explored = state.get('pcm_explored_dimensions', [])
            dimension_mapping = {
                "perception": "Perception",
                "strengths": "Strengths", 
                "interaction_style": "Interaction Style",
                "personality_part": "Personality Parts",
                "channel_communication": "Channels of Communication",
                "environmental_preferences": "Environmental Preferences"
            }
            
            for dim in dimensions_covered:
                display_name = dimension_mapping.get(dim, dim)
                if display_name not in current_explored:
                    current_explored.append(display_name)
            
            updated_state['pcm_explored_dimensions'] = current_explored
        
        # Ajouter les suggestions de transition
        if suggested_transitions:
            updated_state['pcm_transition_suggestions'] = {
                "primary_suggestion": {
                    "action": f"suggest_{suggested_transitions[0].get('target', '').lower()}_transition",
                    "message": suggested_transitions[0].get('message', ''),
                    "context_switch": flow_type_mapping.get(suggested_transitions[0].get('target'), flow_type)
                },
                "alternative_suggestions": suggested_transitions[1:] if len(suggested_transitions) > 1 else []
            }
        
        logger.info(f"✅ Unified analysis complete: {primary_context} → flow_type: {flow_type}")
        return updated_state

    @staticmethod
    def _format_conversation_history(messages: List) -> str:
        """Formate l'historique conversationnel"""
        if not messages:
            return "Première interaction PCM."
        
        formatted = []
        for msg in messages[-3:]:  # 3 derniers messages
            if hasattr(msg, 'content'):
                content = msg.content[:200]
                role = "User" if getattr(msg, 'type', '') == "human" else "Assistant"
            elif isinstance(msg, dict):
                content = msg.get('content', str(msg))[:200]
                role = "User" if msg.get('type') == "human" else "Assistant"
            else:
                continue
            formatted.append(f"{role}: {content}")
        
        return "\\n".join(formatted)

    @staticmethod
    def _get_fallback_result() -> Dict[str, Any]:
        """Résultat de fallback"""
        return {
            "primary_context": "SELF_BASE",
            "sub_context": "base", 
            "confidence": 0.5,
            "dimensions_covered": [],
            "context_change": False,
            "continuity_detected": False,
            "reasoning": "Fallback result used due to analysis error"
        }

    @staticmethod
    def _fallback_analysis(state: WorkflowState) -> Dict[str, Any]:
        """Analyse de fallback simple"""
        return {
            **state,
            'flow_type': 'self_base',
            'pcm_base_or_phase': 'base',
            'unified_analysis_complete': True,
            'pcm_unified_context': PCMUnifiedAnalysis._get_fallback_result()
        }


# Fonction d'entrée principale - remplace les anciens systèmes
def pcm_unified_intent_analysis(state: WorkflowState) -> Dict[str, Any]:
    """
    Point d'entrée unifié - remplace PCMFlowManager + PCMConversationalAnalysis
    """
    return PCMUnifiedAnalysis.analyze_pcm_unified_intent(state)