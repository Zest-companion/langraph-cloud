"""
Module d'analyse pour le Process Communication Model (PCM)
Analyse spécialisée pour les questions PCM
"""

import logging
import re
from typing import Dict, Any, List
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

from ..common.types import WorkflowState
from ..common.llm_utils import isolated_analysis_call_with_messages
from ..common.config import llm, analysis_llm, supabase
from ..lencioni.lencioni_analysis import perform_supabase_vector_search, sanitize_vector_results
from ..prompts.pcm_prompts_builder import (
    build_pcm_self_focused_base_prompt,
    build_pcm_self_focused_phase_prompt,
    build_pcm_general_knowledge_prompt,
    build_pcm_coworker_focused_prompt,
    build_pcm_coworker_focused_action_plan_prompt
)

logger = logging.getLogger(__name__)

# Mapping from intent analysis dimension names to Supabase section_type values
DIMENSION_MAPPING = {
    'perception': 'perception',  # Fixed: was incorrectly mapped to 'characteristics'
    'strengths': 'strengths',  # lowercase in database
    'interaction_style': 'interaction_style', 
    'personality_part': 'personality_part',
    'channel_communication': 'channel_communication',
    'environmental_preferences': 'environmental_preferences'
}

# Reverse mapping from display names back to technical names
DISPLAY_TO_TECHNICAL = {
    "Perception": "perception",
    "Strengths": "strengths", 
    "Interaction Style": "interaction_style",
    "Personality Parts": "personality_part",
    "Channels of Communication": "channel_communication",
    "Environmental Preferences": "environmental_preferences"
}


def pcm_intent_analysis(state: WorkflowState) -> Dict[str, Any]:
    """
    Analyzes user's intent using intelligent PCMFlowManager + conversational context
    Combines LLM classification with conversational flow tracking for the best of both worlds
    """
    logger.info("🎯 STARTING PCM Intent Analysis - INTELLIGENT HYBRID SYSTEM")
    logger.info(f"🔍 Current state keys: {list(state.keys())}")
    
    # ÉTAPE 1: Classification intelligente avec PCMFlowManager
    try:
        from .pcm_flow_manager import PCMFlowManager
        
        logger.info("🤖 Using PCMFlowManager for intelligent classification")
        classification = PCMFlowManager.classify_pcm_intent(state)
        flow_type = classification.get('flow_type', 'SELF_BASE')
        confidence = classification.get('confidence', 0.8)
        
        logger.info(f"📊 Intelligent classification: {flow_type} (confidence: {confidence})")
        logger.info(f"🧠 Reasoning: {classification.get('reasoning', 'N/A')}")
        
        # ÉTAPE 2: Router intelligemment selon le type + contexte conversationnel
        previous_flow = state.get('flow_type')
        explored_dimensions = state.get('pcm_explored_dimensions', [])
        
        # Cas spécial: COWORKER flow en cours
        if previous_flow == 'coworker_focused':
            logger.info("🔒 Coworker flow in progress - continuing with legacy analysis")
            return _continue_coworker_legacy_analysis(state)
        
        # Cas spécial: COMPARISON dans un contexte self_focused conversationnel
        elif flow_type == 'COMPARISON' and (previous_flow == 'self_focused' or explored_dimensions):
            logger.info("🔄 COMPARISON in conversational context - enriching flow")
            return _handle_comparison_in_conversational_flow(state, classification)
        
        # Cas: SELF_BASE ou SELF_PHASE → flux conversationnel classique
        elif flow_type in ['SELF_BASE', 'SELF_PHASE']:
            logger.info("🎯 Self-focused flow - using conversational analysis")
            return _handle_self_focused_with_context(state, classification)
        
        # Cas: COWORKER nouveau → démarrer flux coworker
        elif flow_type == 'COWORKER':
            logger.info("👥 New coworker conversation starting")
            return _handle_new_coworker_flow(state, classification)
        
        # Cas: Flux GENERAL_PCM - théorie générale
        elif flow_type == 'GENERAL_PCM':
            logger.info("📚 GENERAL_PCM flow - PCM theory and concepts")
            return _handle_general_pcm_search(state)
        
        # Cas: Flux directs (TEAM, EXPLORATION, etc.)
        elif flow_type in ['TEAM', 'EXPLORATION']:
            logger.info(f"🎯 {flow_type} flow - using direct processing")
            return _handle_direct_flow_processing(state, classification)
        
        # Cas: GREETING
        elif flow_type == 'GREETING':
            logger.info("👋 Greeting detected")
            return {**state, 'flow_type': 'greeting', 'skip_search': True, 'pcm_analysis_done': True}
        
        # Fallback: traiter comme self_focused
        else:
            logger.warning(f"⚠️ Unknown flow {flow_type} - defaulting to self_focused")
            return _handle_self_focused_with_context(state, classification)
            
    except ImportError:
        logger.warning("⚠️ PCMFlowManager not available, using legacy analysis")
        return _fallback_to_legacy_analysis(state)
    
    # Legacy analysis for other flows
    logger.info("🔄 Using legacy PCM intent analysis")
    
    messages = state['messages']
    if messages:
        last_message = messages[-1]
        # Gérer à la fois les objets Message et les dictionnaires
        if hasattr(last_message, 'content'):
            user_query = last_message.content
        elif isinstance(last_message, dict):
            user_query = last_message.get('content', '')
        else:
            user_query = str(last_message)
    else:
        user_query = ""
    
    # Build conversation context from recent messages
    conversation_context = []
    recent_messages = messages[-3:] if len(messages) >= 3 else messages  # Last 2-3 messages
    
    for msg in recent_messages:
        if hasattr(msg, 'content') and hasattr(msg, 'type'):
            role = "User" if msg.type == 'human' else "Assistant"
            content = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
            conversation_context.append(f"{role}: {content}")
        elif isinstance(msg, dict):
            role = "User" if msg.get('role') == 'user' else "Assistant"
            content = msg.get('content', '')
            content = content[:200] + "..." if len(content) > 200 else content
            conversation_context.append(f"{role}: {content}")
    
    context_text = "\n".join(conversation_context) if conversation_context else "No previous conversation"
    
    # Get previous intent context if available
    previous_flow_type = state.get('flow_type', None)
    previous_pcm_context = state.get('pcm_base_or_phase', None)
    
    # Check if we're in BASE exploration mode (after first PCM interaction)
    is_exploring_base = False
    if len(messages) >= 2:  # At least one exchange
        # Check if previous assistant message introduced BASE dimensions
        for msg in messages[-3:]:  # Check last few messages
            content = ""
            if hasattr(msg, 'content'):
                content = msg.content.lower()
            elif isinstance(msg, dict):
                content = msg.get('content', '').lower()
            
            # Look for signs we introduced BASE dimensions
            if any(phrase in content for phrase in [
                "6 key dimensions", "6 base dimensions", 
                "perception", "strengths", "interaction style",
                "which dimension would you like to explore"
            ]):
                is_exploring_base = True
                break
    
    logger.info(f"📝 Current user query: '{user_query}'")
    logger.info(f"📚 Conversation context: {len(conversation_context)} messages")
    logger.info(f"🎯 Previous flow_type: {previous_flow_type}, pcm_context: {previous_pcm_context}")
    logger.info(f"🎯 BASE exploration mode: {is_exploring_base}")
    
    # Create prompt for PCM intent analysis with context
    previous_context_info = ""
    if previous_flow_type:
        previous_context_info = f"""
PREVIOUS CONVERSATION CONTEXT:
- Previous flow type: {previous_flow_type}
- Previous PCM context: {previous_pcm_context or 'None'}
- BASE exploration in progress: {is_exploring_base}
- This helps maintain consistency in ongoing conversations.
"""
    
    # If we're exploring BASE dimensions, strongly bias toward self_focused
    base_exploration_hint = ""
    if is_exploring_base:
        base_exploration_hint = """
⚠️ IMPORTANT: We are currently exploring the user's BASE dimensions. 
Unless the user explicitly asks about general PCM theory or other people, classify as **self_focused**.
"""

    system_prompt = f"""You are a PCM expert. Analyze the user's current question in the context of the ongoing conversation.

COMPLETE PCM CLASSIFICATION REQUIRED:

1. FLOW TYPES:
   - **general_knowledge**: User wants to learn PCM concepts, theory, or how the model works
   - **self_focused**: User explores THEIR OWN PCM profile, personality, growth
   - **coworker_focused**: ANY situation involving OTHER PEOPLE - relationships, interactions, adaptation to someone else (keywords: "my colleague", "my boss", "my manager", "my team", "coworker", "working with", "dealing with", "person I work with", "relationship with", "someone", "this person", "he/she", "they", "how to adapt to", "situation with")
     
     **EXAMPLES:**
     - "I'm stressed when working with my manager" → coworker_focused (involves manager)
     - "How do I adapt to this difficult colleague?" → coworker_focused (involves colleague) 
     - "I'm having trouble in meetings with my team" → coworker_focused (involves team)
     - "I'm stressed" → self_focused (only about self)
   - **non_pcm**: User's message is NOT related to PCM (greetings, small talk, unrelated topics)

2. PCM CONTEXT (for self_focused queries):
   - **base**: User wants to UNDERSTAND their personality foundation - their natural perception, strengths, communication style (keywords: "my perception", "my strengths", "how I am", "my personality", "tell me about my...")
   - **phase**: User wants to UNDERSTAND their current motivational state - what drives them now, stress patterns (keywords: "what I need now", "my current motivation", "lately I", "recently", "how am I feeling")
   - **situational**: User wants PRACTICAL HELP - concrete steps, advice, solutions for specific situations (keywords: "give me steps", "recommendations", "advice", "what should I do", "how can I", "help me with", "conseils", "que faire")
   
   **KEY DIFFERENCE:**
   - BASE/PHASE = "Help me UNDERSTAND myself" 
   - SITUATIONAL = "Help me DO something / take action"

3. SPECIFIC DIMENSION (if user asks about a BASE dimension):
   - perception: Perception is the filter through which we gather information, experience the outside world,
and interpret others, situations, and our environment.
   - strengths: Core talents, natural abilities, and what someone naturally excels at
   - interaction_style: How someone naturally engages and collaborates with others in work/social settings  
   - personality_part: Observable behavioral patterns, use of energy, observable characteristics
   - channel_communication: Preferred communication style, channels, and how someone naturally expresses themselves. Use of non-verbal language through body language, tone of voice, etc.
   - environmental_preferences: Natural tendencies for different social/work settings (alone vs group, structured vs flexible)

4. EXPLORATION MODE (detect user's exploration intent):
   - **systematic**: User wants to explore ALL dimensions systematically (keywords: "all", "toutes", "every", "complete", "everything", "tous")
   - **flexible**: User wants flexible exploration without commitment to explore all dimensions

{previous_context_info}
{base_exploration_hint}

RECENT CONVERSATION HISTORY:
{context_text}

CURRENT USER QUESTION: {user_query}

CRITICAL INSTRUCTIONS:
- **NON-PCM Detection**: If user's message is greeting ("hello", "hi", "bonjour"), goodbye ("bye", "au revoir"), small talk ("how are you?", "thanks"), or completely unrelated to PCM → **non_pcm**
- If we just introduced PCM and the 6 BASE dimensions, and user agrees to explore them → **self_focused**
- If user mentions "my perception", "let's start with", "yes let's explore" after PCM intro → **self_focused**
- If user says "we can start with perception" or mentions any BASE dimension after intro → **self_focused**
- If previous message was about exploring user's BASE dimensions → continue with **self_focused**
- Keywords like "start with", "explore", "my", "I", "me" in PCM context → **self_focused**
- Consider the conversation flow and context - maintain continuity is KEY
- If the current query is very short/ambiguous (like "yes", "no", "tell me more"), strongly favor continuing the previous flow_type and context UNLESS it's clearly a greeting/goodbye
- For SHORT RESPONSES (<10 chars), default to previous flow_type unless clearly changing topics OR it's non-PCM
- If continuing a previous topic, maintain consistency unless clearly shifting topics
- Pay attention to pronouns ("How does this apply to me?" likely means self_focused if discussing PCM)  
- LANGUAGE DETECTION - MANDATORY: Detect the language of the user's message
  * If the message is in French → return 'fr'
  * If the message is in English → return 'en'  
  * If the message is in any other language → return 'en' (English as default)

Respond in JSON format:
{{
    "flow_type": "general_knowledge|self_focused|coworker_focused|non_pcm",
    "pcm_context": "base|phase|situational|null",
    "specific_dimensions": ["perception", "strengths", "interaction_style", "personality_part", "channel_communication", "environmental_preferences"] or null,
    "exploration_mode": "systematic|flexible",
    "reasoning": "Complete reasoning for all classifications",
    "language": "fr|en"
}}

IMPORTANT: 
- If user mentions multiple dimensions (e.g. "explore my perception and strengths"), return them as an array: ["perception", "strengths"]
- If user mentions one dimension, return as single-item array: ["perception"] 
- If no specific dimension mentioned, return null
"""
    
    # Use isolated call to analyze intent
    logger.info("🚀 About to call LLM for PCM intent analysis with conversation context")
    try:
        pcm_intent_raw = isolated_analysis_call_with_messages(
            system_content=system_prompt,
            user_content="Please analyze the conversation and classify the current intent."
        )
        logger.info(f"✅ PCM intent analysis LLM call successful: {len(pcm_intent_raw)} chars")
        logger.info(f"🔤 First 200 chars of result: {pcm_intent_raw[:200]}...")
    except Exception as e:
        logger.error(f"❌ PCM intent analysis LLM call failed: {e}")
        pcm_intent_raw = f"LLM_ERROR: {{str(e)}}"
    
    # Ensure we always have content for LangGraph Studio visibility
    if not pcm_intent_raw or pcm_intent_raw.strip() == "":
        logger.warning("⚠️ Empty result from LLM, using fallback")
        pcm_intent_raw = f"EMPTY_LLM_RESULT for query: '{user_query}'"
    
    # Parse JSON to extract all fields from unified LLM response
    flow_type = 'non_pcm'  # Default fallback (plus sûr que general_knowledge)
    language = 'en'  # Default fallback
    pcm_base_or_phase = None  # Default null for non-PCM
    specific_dimensions = None  # Default no dimensions
    reasoning = ''  # Default empty reasoning
    
    try:
        import json
        parsed_intent = json.loads(pcm_intent_raw)
        flow_type = parsed_intent.get('flow_type', 'general_knowledge')
        language = parsed_intent.get('language', 'en')
        pcm_base_or_phase = parsed_intent.get('pcm_context', None)
        if pcm_base_or_phase == 'null':
            pcm_base_or_phase = None
        elif pcm_base_or_phase == 'situational':
            # Map situational to action_plan for consistency
            pcm_base_or_phase = 'action_plan'
        # Si flow_type est non_pcm, forcer pcm_context à None
        if flow_type == 'non_pcm':
            pcm_base_or_phase = None
            
        specific_dimensions = parsed_intent.get('specific_dimensions')
        if specific_dimensions == 'null' or specific_dimensions == [] or specific_dimensions is None:
            specific_dimensions = None
        # Ensure we have a list even if LLM returns a single string (backward compatibility)
        elif isinstance(specific_dimensions, str):
            specific_dimensions = [specific_dimensions] if specific_dimensions != 'null' else None
        
        exploration_mode = parsed_intent.get('exploration_mode', 'flexible')
        if exploration_mode == 'null':
            exploration_mode = 'flexible'
            
        reasoning = parsed_intent.get('reasoning', '')
        
        logger.info(f"✅ Parsed unified intent: flow_type={flow_type}, pcm_context={pcm_base_or_phase}, dimensions={specific_dimensions}, language={language}")
    except json.JSONDecodeError as e:
        logger.warning(f"⚠️ Failed to parse JSON intent: {e} - using fallbacks")
    except Exception as e:
        logger.warning(f"⚠️ Error extracting intent fields: {e} - using fallbacks")
    
    # PROTECTION FINALE: Forcer coworker_focused si on y était déjà
    if previous_flow == 'coworker_focused':
        flow_type = 'coworker_focused'
        pcm_base_or_phase = 'coworker'  # Force le contexte coworker pour éviter la branche base/phase
        logger.info("🔒 FORCED: Preserving coworker_focused flow_type AND context")
    
    # Build base_phase_reasoning structure for visibility in LangGraph Studio
    base_phase_reasoning = {
        'classification': pcm_base_or_phase,
        'reasoning': reasoning,
        'specific_dimensions': specific_dimensions,
        'user_query': user_query
    }
    
    logger.info("📦 Creating return state with simplified PCM intent")
    
    # Get additional PHASE transition info
    phase_request = _detect_phase_request(user_query)
    should_suggest_phase = _should_suggest_phase_transition(messages, user_query)
    
    # Create a structured result for LangGraph Studio visualization
    structured_result = {
        'flow_type': flow_type,
        'language': language,
        'pcm_base_or_phase': pcm_base_or_phase,  # Visible in LangGraph Studio
        'specific_dimensions': specific_dimensions,  # NEW: Specific BASE dimensions if detected (array)
        'base_phase_reasoning': base_phase_reasoning,  # Reasoning for BASE vs PHASE classification
        'phase_request_detected': phase_request,  # Explicit PHASE request
        'should_suggest_phase': should_suggest_phase,  # Proactive PHASE suggestion
        'raw_analysis': pcm_intent_raw,
        'user_query': user_query,
        'conversation_context_used': len(conversation_context),
        'previous_context': f"flow: {previous_flow_type}, pcm: {previous_pcm_context}" if previous_flow_type else "none"
    }
    
    return {
        **state,
        'pcm_intent_analysis': structured_result,  # Structured result for LangGraph Studio
        'flow_type': flow_type,  # 3 options: general_knowledge, self_focused, coworker_focused
        'language': language,  # fr ou en
        'pcm_base_or_phase': pcm_base_or_phase,  # BASE/PHASE context for self_focused flow
        'pcm_specific_dimensions': specific_dimensions,  # Specific BASE dimensions if detected (array)
        'exploration_mode': exploration_mode,  # systematic|flexible - persists exploration intent
        'pcm_analysis_done': True,
        'debug_pcm_intent': f"EXECUTED: {len(pcm_intent_raw)} chars, flow={flow_type}, lang={language}, context={pcm_base_or_phase}, dims={specific_dimensions}, mode={exploration_mode}, history={len(conversation_context)} msgs, prev={previous_flow_type or 'None'}",
        # Explicitly preserve PCM profile from fetch_user_profile
        'pcm_base': state.get('pcm_base'),
        'pcm_phase': state.get('pcm_phase')
    }

def pcm_vector_search(state: WorkflowState) -> Dict[str, Any]:
    """
    Recherche vectorielle intelligente dans les documents PCM
    Organise les résultats par catégorie (Base, Phase, General)
    """
    logger.info("🔎 Starting PCM Vector Search")
    logger.info(f"🔍 DEBUG state type: {type(state)}")
    logger.info(f"🔍 DEBUG state is None: {state is None}")
    logger.info(f"🔍 DEBUG state keys: {list(state.keys()) if state else 'N/A'}")
    
    # PRIORITÉ 1: Utiliser directement le résultat du routing/classification
    logger.info("🔍 DEBUG: About to access pcm_classification")
    pcm_classification = state.get('pcm_classification', {})
    logger.info(f"🔍 DEBUG pcm_classification: {pcm_classification}")
    flow_type_classified = pcm_classification.get('flow_type') if pcm_classification else None
    
    if flow_type_classified:
        logger.info(f"🎯 Using PCM classification result: {flow_type_classified}")
        
        # Gérer le refus de sécurité en premier
        if flow_type_classified == 'SAFETY_REFUSAL':
            logger.warning("🚫 SAFETY REFUSAL detected in classification - Skipping all searches")
            return {
                **state,
                "pcm_resources": "",
                "skip_search": True,
                "vector_search_complete": True
            }
        elif flow_type_classified == 'SELF_ACTION_PLAN':
            return _handle_self_action_plan_search(state)
        elif flow_type_classified == 'SELF_BASE':
            return _handle_self_base_search(state)
        elif flow_type_classified == 'SELF_PHASE':
            return _handle_self_phase_search(state)
        elif flow_type_classified == 'COWORKER':
            return _handle_coworker_search(state)
        elif flow_type_classified == 'COMPARISON':
            return _handle_comparison_search(state)
        elif flow_type_classified == 'GREETING':
            return _handle_greeting_search(state)
        # Si classification inconnue, continuer avec la logique legacy
        else:
            logger.info(f"⚠️ Unknown classification {flow_type_classified}, using legacy logic")
    
    # PRIORITÉ 2: Vérifier les flags de greeting/skip avant le contexte conversationnel
    if state.get('skip_search') or state.get('greeting_detected') or state.get('flow_type') == 'greeting':
        logger.info("🎯 Skip search flag detected (greeting or safety refusal)")
        # Si c'est un safety refusal, on ne fait aucune recherche
        if state.get('flow_type') == 'safety_refusal':
            logger.warning("🚫 SAFETY REFUSAL - Skipping all searches")
            return {
                **state,
                "pcm_resources": "",
                "skip_search": True,
                "vector_search_complete": True
            }
        # Sinon c'est un greeting
        logger.info("🎯 Greeting detected → using _handle_greeting_search")
        return _handle_greeting_search(state)
    
    # PRIORITÉ 3: Vérifier le contexte conversationnel pour flux self_focused
    flow_type = state.get('flow_type', 'general_knowledge')
    if flow_type == 'self_focused':
        pcm_conversational_context = state.get('pcm_conversational_context', {})
        current_context = pcm_conversational_context.get('current_context')
        
        # Vérifier si c'est une première interaction
        explored_dimensions = state.get('pcm_explored_dimensions', [])
        messages = state.get('messages', [])
        is_first_pcm_interaction = len(explored_dimensions) == 0 and len(messages) <= 2
        specific_dimensions = state.get('pcm_specific_dimensions')
        
        logger.info(f"🔍 First interaction check: explored={len(explored_dimensions)}, messages={len(messages)}, is_first={is_first_pcm_interaction}, context={current_context}")
        
        if current_context == 'phase':
            if is_first_pcm_interaction:
                logger.info("🆕 Self-focused flow: FIRST PHASE interaction → using first phase prompt")
                try:
                    from .pcm_analysis_new import _handle_first_phase_interaction
                    return _handle_first_phase_interaction(state)
                except ImportError:
                    logger.warning("⚠️ First PHASE interaction handler not available, using normal phase search")
                    return _handle_self_phase_search(state)
            else:
                logger.info("🎯 Self-focused flow with PHASE context → using _handle_self_phase_search")
                return _handle_self_phase_search(state)
        elif current_context == 'base':
            if is_first_pcm_interaction and not specific_dimensions:
                logger.info("🆕 Self-focused flow: FIRST BASE interaction → using first base prompt")
                try:
                    from .pcm_analysis_new import _handle_first_base_interaction
                    return _handle_first_base_interaction(state)
                except ImportError:
                    logger.warning("⚠️ First BASE interaction handler not available, using normal base search")
                    return _handle_self_base_search(state)
            else:
                logger.info("🎯 Self-focused flow with BASE context → using _handle_self_base_search")
                return _handle_self_base_search(state)
    
    # FALLBACK: Logique legacy si pas de classification
    logger.info("🔄 No PCM classification found, using legacy logic")
    
    # Protection contre state None
    if not state:
        logger.error("❌ State is None in pcm_vector_search")
        return {'pcm_resources': 'Error: No state provided', 'vector_search_complete': True}
    
    messages = state.get('messages', [])
    if messages:
        last_message = messages[-1]
        # Gérer à la fois les objets Message et les dictionnaires
        if hasattr(last_message, 'content'):
            user_query = last_message.content
        elif isinstance(last_message, dict):
            user_query = last_message.get('content', '')
        else:
            user_query = str(last_message)
    else:
        user_query = ""
        
    # 1. Get classification from intent analysis
    flow_type = state.get('flow_type', 'general_knowledge')
    language = state.get('language', 'en')
    
    logger.info(f"🎯 Using flow type: {flow_type}, Language: {language}")
    
    # 2. Recherche vectorielle adaptée
    pcm_base_results = []
    pcm_phase_results = []
    pcm_general_results = []
    pcm_base_or_phase = None  # Track BASE/PHASE classification for visibility
    
    # Initialize coworker_focused state variables (preserved across calls)
    coworker_step = state.get('coworker_step', 1)
    coworker_self_ok = state.get('coworker_self_ok', False)
    coworker_other_profile = state.get('coworker_other_profile', {})
    coworker_step_2_substep = state.get('coworker_step_2_substep', 1)  # Sub-steps for step 2
    
    try:
        pcm_results, pcm_base_or_phase, updated_coworker_state = _execute_pcm_search(user_query, flow_type, language, state)
        
        # Update coworker state variables if they were updated
        if updated_coworker_state:
            print(f"🔍 DEBUG: About to update local variables from updated_coworker_state: {updated_coworker_state}")
            coworker_step = updated_coworker_state.get('coworker_step', coworker_step)
            coworker_self_ok = updated_coworker_state.get('coworker_self_ok', coworker_self_ok)
            coworker_other_profile = updated_coworker_state.get('coworker_other_profile', coworker_other_profile)
            coworker_step_2_substep = updated_coworker_state.get('coworker_step_2_substep', coworker_step_2_substep)
            print(f"🔍 DEBUG: After update - coworker_step = {coworker_step}, coworker_self_ok = {coworker_self_ok}")
        else:
            print(f"🔍 DEBUG: No updated_coworker_state to apply (empty dict)")
        
        logger.info(f"🎯 Retrieved {len(pcm_results)} PCM results from vector search")
        
        # 3. Organiser les résultats par catégorie selon les métadonnées
        # Since filtering is now done at query level, this is simplified
        for result in pcm_results:
            metadata = result.get('metadata', {})
            pcm_phase_type = metadata.get('pcm_phase_type', '').lower() if metadata.get('pcm_phase_type') else None
            pcm_base_type = metadata.get('pcm_base_type', '').lower() if metadata.get('pcm_base_type') else None
            pcm_type = metadata.get('pcm_type', '').lower() if metadata.get('pcm_type') else None
            
            # Classifier selon les métadonnées PCM
            if pcm_phase_type is not None:
                # Document Phase
                pcm_phase_results.append({
                    'content': result.get('content', ''),
                    'metadata': metadata,
                    'similarity': result.get('similarity', 0),
                    'category': 'PHASE',
                    'user_match': True  # Already filtered at query level
                })
                    
            elif pcm_base_type is not None:
                # Document Base
                pcm_base_results.append({
                    'content': result.get('content', ''),
                    'metadata': metadata,
                    'similarity': result.get('similarity', 0),
                    'category': 'BASE',
                    'user_match': True  # Already filtered at query level
                })
                    
            elif pcm_type == 'general':
                # Document général (accessible à tous)
                pcm_general_results.append({
                    'content': result.get('content', ''),
                    'metadata': metadata,
                    'similarity': result.get('similarity', 0),
                    'category': 'GENERAL',
                    'user_match': True
                })
            else:
                # Fallback : mettre dans général
                pcm_general_results.append({
                    'content': result.get('content', ''),
                    'metadata': metadata,
                    'similarity': result.get('similarity', 0),
                    'category': 'OTHER_TYPE',
                    'user_match': True  # Query-level filtered results are considered matches
                })
        
        # 5. Continue to search_results assignment (standard flow)
        
        logger.info(f"✅ PCM vector search completed:")
        logger.info(f"   📊 Base results: {len(pcm_base_results)}")
        logger.info(f"   🔄 Phase results: {len(pcm_phase_results)}")
        logger.info(f"   📚 General results: {len(pcm_general_results)}")
        
        # 6. Format resources for the prompt
        logger.info(f"🔍 FORMATTING DEBUG: flow_type={flow_type}, coworker_step={coworker_step}")
        logger.info(f"🔍 FORMATTING DEBUG: pcm_results length = {len(pcm_results)}")
        logger.info(f"🔍 FORMATTING DEBUG: pcm_results content = {pcm_results[:2] if pcm_results else 'EMPTY'}")
        if flow_type == 'coworker_focused' and coworker_step == 4:
            # Step 4: Special formatting for 4-profile analysis (USER + COLLEAGUE)
            # Use raw pcm_results instead of re-classified results
            logger.info(f"🎯 STEP 4 FORMATTING: Calling special formatter with {len(pcm_results)} results")
            pcm_resources = _format_coworker_step4_results(
                pcm_results,  # All raw results from Step 4 searches
                state,
                language
            )
        elif flow_type == 'coworker_focused' and coworker_step == 2 and coworker_step_2_substep == 1:
            # Step 2.1: Special educational formatting (BASE + 3 PHASE sections)
            logger.info(f"🎯 Step 2.1 FORMATTING: Using educational format for BASE + PHASE sections")
            logger.info(f"🔍 DEBUG FORMATTING: flow_type={flow_type}, coworker_step={coworker_step}, substep={coworker_step_2_substep}")
            pcm_resources = _format_coworker_step2_base_phase_results(
                pcm_base_results + pcm_phase_results,
                state,
                language
            )
            logger.info(f"🔍 DEBUG FORMATTING: pcm_resources formatted, length={len(pcm_resources)} chars, starts with: {pcm_resources[:100]}...")
        elif pcm_base_or_phase == 'action_plan' or (flow_type == 'coworker_focused' and coworker_step == 2 and coworker_step_2_substep == 2):
            # ACTION_PLAN uses special 3-section formatting (only for coworker_focused step 2.2, not 2.1)
            logger.info(f"🔍 DEBUG FORMATTING: Using ACTION_PLAN formatting - pcm_base_or_phase={pcm_base_or_phase}, flow_type={flow_type}, coworker_step={coworker_step}, substep={coworker_step_2_substep}")
            all_results = pcm_base_results + pcm_phase_results + pcm_general_results
            pcm_resources = _format_action_plan_results_by_sections(all_results, flow_type, language)
        else:
            # Regular formatting for BASE/PHASE
            logger.info(f"🔍 DEBUG FORMATTING: Using REGULAR formatting - flow_type={flow_type}, coworker_step={coworker_step}, substep={coworker_step_2_substep}")
            pcm_resources = _format_pcm_results_by_category(
                pcm_base_results, 
                pcm_phase_results, 
                pcm_general_results,
                flow_type,
                language
            )
            logger.info(f"🔍 DEBUG FORMATTING: REGULAR pcm_resources formatted, length={len(pcm_resources)} chars")
        
    except Exception as e:
        logger.warning(f"⚠️ PCM vector search failed: {str(e)}")
        logger.info("📚 Using PCM fallback")
        
        pcm_resources = _get_pcm_fallback(flow_type, language)
        pcm_base_or_phase = None  # No classification available in fallback case
    
    # Update explored dimensions if we have a specific dimension
    updated_state = update_explored_dimensions(state)
    
    # DEBUG: Check if conversational keys exist in input state
    logger.info(f"🔍 DEBUG pcm_vector_search INPUT - state keys: {list(state.keys())}")
    logger.info(f"🔍 DEBUG pcm_vector_search INPUT - pcm_classification: {state.get('pcm_classification')}")
    logger.info(f"🔍 DEBUG pcm_vector_search INPUT - conversational_context: {state.get('pcm_conversational_context')}")
    logger.info(f"🔍 DEBUG pcm_vector_search INPUT - conversational_complete: {state.get('conversational_analysis_complete')}")
    logger.info(f"🔍 DEBUG pcm_vector_search INPUT - flow_type: {state.get('flow_type')}")
    
    # DEBUG: Log the final coworker state values being returned
    print(f"🔍 DEBUG pcm_vector_search RETURN: coworker_step = {coworker_step}, coworker_self_ok = {coworker_self_ok}")
    print(f"🔍 DEBUG pcm_vector_search RETURN: coworker_step_2_substep = {coworker_step_2_substep}")
    print(f"🔍 DEBUG pcm_vector_search RETURN: coworker_other_profile = {coworker_other_profile}")
    print(f"🔍 DEBUG pcm_vector_search RETURN: coworker_other_profile.base_type = {coworker_other_profile.get('base_type') if coworker_other_profile else 'N/A'}")
    
    final_state = {
        **updated_state,
        'pcm_resources': pcm_resources,
        'pcm_base_results': pcm_base_results,  # Visible dans LangGraph Studio
        'pcm_phase_results': pcm_phase_results,  # Visible dans LangGraph Studio
        'pcm_general_results': pcm_general_results,  # Visible dans LangGraph Studio
        'pcm_base_or_phase': pcm_base_or_phase,  # BASE/PHASE classification visible in LangGraph Studio
        'flow_type': flow_type,
        'language': language,
        # Coworker_focused state tracking
        'coworker_step': coworker_step,
        'coworker_self_ok': coworker_self_ok, 
        'coworker_other_profile': coworker_other_profile,
        'coworker_step_2_substep': coworker_step_2_substep,
        'coworker_step_1_attempts': state.get('coworker_step_1_attempts', 0),  # Preserve step 1 attempts counter
        'pcm_search_debug': f"Base: {len(pcm_base_results)}, Phase: {len(pcm_phase_results)}, General: {len(pcm_general_results)}, Context: {pcm_base_or_phase or 'None'}, CoworkerStep: {coworker_step}",
        'vector_search_complete': True,
        # Explicitly preserve PCM profile from fetch_user_profile
        'pcm_base': state.get('pcm_base'),
        'pcm_phase': state.get('pcm_phase'),
        # CRUCIAL: Preserve classification from flow manager
        'pcm_classification': state.get('pcm_classification'),
        # CRUCIAL: Preserve conversational context keys from pcm_intent_analysis
        'pcm_conversational_context': state.get('pcm_conversational_context'),
        'pcm_context_reasoning': state.get('pcm_context_reasoning'),
        'pcm_transition_suggestions': state.get('pcm_transition_suggestions'),
        'conversational_analysis_complete': state.get('conversational_analysis_complete', False)
    }
    
    print(f"🔍 DEBUG: Final state contains coworker_step = {final_state.get('coworker_step', 'NOT_FOUND')}")
    return final_state

def _detect_base_dimension(user_query: str) -> str:
    """
    Detect which specific BASE dimension is being requested
    Returns the content_type format (e.g., 'base_perception') or None if not clear
    """
    query_lower = user_query.lower()
    
    # Map keywords to dimension content_type values
    dimension_keywords = {
        'perception': ['perception', 'perceive', 'see the world', 'filter', 'interpret', 'view'],
        'strengths': ['strength', 'talent', 'good at', 'abilities', 'skills', 'capabilities'],
        'interaction_style': ['interaction', 'interact', 'work with others', 'team style', 'collaborate', 'approach people'],
        'personality_part': ['personality parts', 'behaviors', 'observable', 'parts', 'behavioral patterns'],
        'channel_communication': ['communication', 'communicate', 'channels', 'talk', 'express', 'speak'],
        'environmental_preferences': ['environment', 'prefer', 'group', 'alone', 'social setting', 'work setting']
    }
    
    # Check for dimension keywords in user query
    for content_type, keywords in dimension_keywords.items():
        for keyword in keywords:
            if keyword in query_lower:
                return content_type
    
    return None

def _detect_phase_request(user_query: str) -> bool:
    """
    Detect if user is explicitly requesting PHASE exploration
    Returns True if PHASE request detected
    """
    query_lower = user_query.lower()
    
    # Keywords that indicate PHASE interest
    phase_keywords = [
        'phase', 'current needs', 'motivation', 'what motivates me', 
        'current state', 'what I need now', 'recently changed',
        'different lately', 'evolved', 'growth', 'development',
        'where I am now', 'current focus', 'priorities now'
    ]
    
    # Check for phase keywords
    for keyword in phase_keywords:
        if keyword in query_lower:
            return True
    
    return False

# Fonction de comptage supprimée car elle ne fonctionnait pas correctement

def _should_suggest_phase_transition(messages: list, user_query: str) -> bool:
    """
    Determine if we should proactively suggest transitioning to PHASE
    Based on conversation length and engagement indicators
    """
    # If conversation is getting long (suggesting substantial exploration)
    if len(messages) >= 8:
        return True
    
    # If user mentions stress, emotions, or current situation (PHASE indicators)
    phase_indicators = ['stress', 'stressed', 'feeling', 'current', 'right now', 'lately', 'these days']
    query_lower = user_query.lower()
    if any(indicator in query_lower for indicator in phase_indicators):
        return True
    
    return False

def _execute_pcm_search(user_query: str, flow_type: str, language: str, state: WorkflowState) -> tuple[List[Dict], str, Dict[str, Any]]:
    """
    Execute PCM-specific vector search based on flow type (simplified to 3 options)
    Filters results based on user's PCM profile when appropriate
    """
    logger.info(f"🔍 Executing PCM search for flow_type: {flow_type}, language: {language}")
    
    # Initialize variables used in return statement (for all flow types)
    coworker_step_2_substep = 1  # Default value
    coworker_step = 1  # Default value
    self_ok = False  # Default value 
    other_profile = {}  # Default value
    
    # Base filters for PCM documents
    base_filters = {
        'theme': 'A_UnderstandingMyselfAndOthers',
        'sub_theme': 'A2_PersonalityPCM', 
        'language': state.get('language', 'en')
    }
    
    # Get user's PCM profile for filtering
    user_pcm_base = state.get('pcm_base', '').lower() if state.get('pcm_base') else None
    user_pcm_phase = state.get('pcm_phase', '').lower() if state.get('pcm_phase') else None
    
    # Get the BASE/PHASE context determined during intent analysis
    pcm_base_or_phase = state.get('pcm_base_or_phase')
    if pcm_base_or_phase:
        logger.info(f"🎯 Using PCM context from intent analysis: {pcm_base_or_phase}")
    else:
        logger.info("🎯 No specific PCM context determined (general search)")
    
    # Get specific dimensions from unified LLM analysis (no need to detect again!)
    specific_dimensions_list = state.get('pcm_specific_dimensions')
    if specific_dimensions_list:
        logger.info(f"🎯 Using specific dimensions from unified analysis: {specific_dimensions_list}")
    else:
        logger.info("🎯 No specific dimensions detected - will retrieve all relevant content")
    
    # Detect if user is requesting PHASE exploration
    phase_request = _detect_phase_request(user_query)
    if phase_request:
        logger.info(f"🔄 Detected PHASE request: {phase_request}")
    
    # Check if we should suggest PHASE transition
    # Extract messages from state first
    state_messages = state.get('messages', [])
    should_suggest_phase = _should_suggest_phase_transition(state_messages, user_query)
    if should_suggest_phase:
        logger.info(f"💡 Should suggest PHASE transition")
    
    # Add specific filtering based on context and user profile
    if pcm_base_or_phase == 'action_plan':
        # ACTION_PLAN context: recherche dans les documents d'action planning
        logger.info(f"🎯 Filtering for ACTION_PLAN documents for stress/workplace situations")
        action_plan_filters = base_filters.copy()
        action_plan_filters['section_type'] = 'action_plan'
        
        # Recherche pour des stratégies et plans d'action
        search_results = perform_supabase_vector_search(
            query=user_query,
            match_function='match_documents',
            metadata_filters=action_plan_filters,
            limit=8
        )
        logger.info(f"🔍 ACTION_PLAN search completed: {len(search_results)} results")
        
    # IMPORTANT: If user asks for specific BASE dimensions, always use BASE filtering
    elif specific_dimensions_list and user_pcm_base and pcm_base_or_phase != 'phase':
        # User is asking for specific BASE dimensions - need to handle multiple dimensions
        # We'll do multiple searches and merge results
        logger.info(f"🔍 Filtering for BASE documents (multi-dimension): {user_pcm_base} -> {specific_dimensions_list}")
        
        all_results = []
        for dimension in specific_dimensions_list:
            # Convert display names to technical names first, then map to database values
            technical_dimension = DISPLAY_TO_TECHNICAL.get(dimension, dimension.lower())
            mapped_dimension = DIMENSION_MAPPING.get(technical_dimension, technical_dimension)
            
            dimension_filters = base_filters.copy()  # Copy base filters
            dimension_filters['pcm_base_type'] = user_pcm_base
            dimension_filters['section_type'] = mapped_dimension
            
            logger.info(f"🔍 Searching for dimension: {dimension}")
            dimension_results = perform_supabase_vector_search(
                query=user_query,
                match_function='match_documents', 
                metadata_filters=dimension_filters,
                limit=4  # Limit per dimension to avoid too many results
            )
            all_results.extend(dimension_results)
        
        # Use merged results instead of single search
        search_results = all_results
        logger.info(f"🔍 Multi-dimension search completed: {len(search_results)} total results")
        
    elif pcm_base_or_phase == 'base' and user_pcm_base:
        # Filter for BASE documents matching user's base type
        base_filters['pcm_base_type'] = user_pcm_base
        logger.info(f"🔍 Filtering for BASE documents matching user's base: {user_pcm_base}")
        
        # Execute single search with base filters
        search_results = perform_supabase_vector_search(
            query=user_query,
            match_function='match_documents',
            metadata_filters=base_filters,
            limit=8
        )
            
    elif flow_type == 'coworker_focused':
        # COWORKER_FOCUSED: 4-step process (Self-Assessment → Other-Assessment → Adaptation Strategy)
        logger.info(f"👥 Starting COWORKER_FOCUSED search: 4-step relationship analysis")
        
        # Initialize or get current coworker step
        coworker_step = state.get('coworker_step', 1)  # Default to step 1
        self_ok = state.get('coworker_self_ok', False)
        other_profile = state.get('coworker_other_profile', {})
        coworker_step_2_substep = state.get('coworker_step_2_substep', 1)  # Sub-steps for step 2
        
        logger.info(f"👥 COWORKER_FOCUSED - Current step: {coworker_step}, Self OK: {self_ok}")
        logger.info(f"🔥 DEBUG: About to call _analyze_coworker_progression")
        
        # Auto-progression logic based on user response analysis
        try:
            coworker_step, self_ok, other_profile, coworker_step_2_substep = _analyze_coworker_progression(
                user_query=user_query,
                current_step=coworker_step,
                current_self_ok=self_ok,
                current_other_profile=other_profile,
                conversation_context=_build_conversation_context_coworker(state.get('messages', [])),
                state=state,
                current_step_2_substep=coworker_step_2_substep
            )
            logger.info(f"🔥 DEBUG: _analyze_coworker_progression completed successfully")
        except Exception as e:
            logger.error(f"🔥 ERROR in _analyze_coworker_progression: {e}")
            import traceback
            logger.error(f"🔥 TRACEBACK: {traceback.format_exc()}")
            # Use defaults on error
            coworker_step = coworker_step  # Keep current
            self_ok = self_ok  # Keep current
            other_profile = other_profile  # Keep current
            coworker_step_2_substep = coworker_step_2_substep  # Keep current
        
        logger.info(f"👥 After progression analysis - Step: {coworker_step}, Self OK: {self_ok}, Step 2 substep: {coworker_step_2_substep if coworker_step == 2 else 'N/A'}")
        
        # Step-specific search logic selon nouveau flux
        if coworker_step == 1:
            # Step 1: Matrice +/+ ou -/- assessment - PAS DE RECHERCHE VECTORIELLE
            # On fait juste l'évaluation émotionnelle, pas besoin de contenu PCM
            logger.info("👥 Step 1: No vector search needed, just emotional assessment")
            all_results = []  # Aucune recherche pour l'étape 1
        elif coworker_step == 2:
            # Step 2: Two substeps with different search logic
            logger.info(f"👥 Step 2.{coworker_step_2_substep}: Coworker focused ACTION_PLAN phase")
            all_results = []
            
            if coworker_step_2_substep == 1:
                # Step 2.1: Get ALL BASE dimensions + ALL PHASE sections for explanation
                logger.info("👥 Step 2.1: Getting comprehensive BASE and PHASE for education")
                
                # Get ALL 6 BASE dimensions if user has a BASE
                if user_pcm_base:
                    base_filters_action = base_filters.copy()
                    base_filters_action['pcm_base_type'] = user_pcm_base.lower()
                    
                    base_results = perform_supabase_vector_search(
                        query=user_query,
                        match_function='match_documents',
                        metadata_filters=base_filters_action,
                        limit=12  # All dimensions for complete understanding
                    )
                    all_results.extend(base_results)
                    logger.info(f"👥 Step 2.1 BASE: {len(base_results)} comprehensive BASE results")
                
                # Get PHASE if user has a PHASE (3 sections: needs, negative_satisfaction, distress_sequence)
                if user_pcm_phase:
                    phase_section_types = ['psychological_needs', 'negative_satisfaction', 'distress_sequence']
                    
                    for section_type in phase_section_types:
                        phase_filters = base_filters.copy()
                        phase_filters['pcm_phase_type'] = user_pcm_phase
                        phase_filters['section_type'] = section_type
                        
                        phase_results = perform_supabase_vector_search(
                            query=user_query,
                            match_function='match_documents',
                            metadata_filters=phase_filters,
                            limit=3  # Per section for complete understanding
                        )
                        all_results.extend(phase_results)
                    
                    logger.info(f"👥 Step 2.1 PHASE: Complete psychological needs, negative satisfaction, distress sequence")
                # NO ACTION_PLAN search for step 2.1 - just education
                
            else:  # coworker_step_2_substep == 2
                # Step 2.2: Get ACTION_PLAN + BASE + PHASE for comprehensive guidance
                logger.info("👥 Step 2.2: Getting comprehensive ACTION_PLAN with BASE and PHASE context")
                
                # 1. Get ACTION_PLAN guidance for concrete steps
                action_plan_filters = base_filters.copy()
                action_plan_filters['pcm_phase_type'] = user_pcm_phase
                action_plan_filters['section_type'] = 'action_plan'
                
                action_plan_results = perform_supabase_vector_search(
                    query=user_query + " workplace stress management action plan",
                    match_function='match_documents',
                    metadata_filters=action_plan_filters,
                    limit=6  # Reduced to make room for BASE/PHASE
                )
                all_results.extend(action_plan_results)
                logger.info(f"👥 Step 2.2: {len(action_plan_results)} ACTION_PLAN results")
                
                # 2. Get user BASE info to explain natural strengths 
                if user_pcm_base:
                    base_filters_user = base_filters.copy()
                    base_filters_user['pcm_base_type'] = user_pcm_base.lower()
                    base_filters_user['section_type'] = 'strengths'  # Focus on strengths for action plan
                    
                    base_results = perform_supabase_vector_search(
                        query="strengths personality foundation natural traits",
                        match_function='match_documents',
                        metadata_filters=base_filters_user,
                        limit=2
                    )
                    all_results.extend(base_results)
                    logger.info(f"👥 Step 2.2: {len(base_results)} BASE strengths results")
                
                # 3. Get user PHASE psychological needs info
                if user_pcm_phase:
                    phase_filters_user = base_filters.copy()
                    phase_filters_user['pcm_phase_type'] = user_pcm_phase.lower()
                    phase_filters_user['section_type'] = 'psychological_needs'
                    
                    phase_results = perform_supabase_vector_search(
                        query="psychological needs motivational requirements current phase",
                        match_function='match_documents',
                        metadata_filters=phase_filters_user,
                        limit=2
                    )
                    all_results.extend(phase_results)
                    logger.info(f"👥 Step 2.2: {len(phase_results)} PHASE psychological needs results")
        
        elif coworker_step == 3:
            # Step 3: Explore colleague (BASE then PHASE)
            logger.info("👥 Step 3: Exploring colleague profile")
            logger.info("👥 Step 3 DEBUG: ENTERING Step 3 block successfully")
            all_results = []
            logger.info("👥 Step 3 DEBUG: Step 3 should trigger research or progress to Step 4")
            # TODO: Add Step 3 logic or ensure progression to Step 4

        elif coworker_step == 4:
            # Step 4: Final recommendations with ACTION PLANs for both user and colleague
            logger.info("👥 Step 4: Generating final recommendations with action plans")
            logger.info("👥 Step 4 DEBUG: ENTERING Step 4 block successfully")
            
            # NOUVEAU: Détecter si l'utilisateur veut changer d'état émotionnel
            emotional_change = _detect_emotional_state_change(user_query)
            if emotional_change:
                logger.info(f"🔄 EMOTIONAL STATE CHANGE detected: {emotional_change}")
                # Appliquer le changement et régénérer Step 4
                coworker_self_ok, other_profile = _apply_emotional_state_change(
                    emotional_change, coworker_self_ok, other_profile
                )
                logger.info(f"🔄 Emotional states updated - regenerating Step 4 recommendations")
            
            all_results = []
            logger.info("👥 Step 4 DEBUG: all_results initialized")
            
            # Get colleague profile information from updated progression (not state!)
            colleague_profile = other_profile if other_profile else state.get('coworker_other_profile', {})
            colleague_base = colleague_profile.get('base_type')
            colleague_phase = colleague_profile.get('phase_state')
            
            # DEBUG: Step 4 - Check what values we received
            logger.info(f"🔍 STEP 4 INPUT DEBUG: other_profile = {other_profile}")
            logger.info(f"🔍 STEP 4 INPUT DEBUG: colleague_profile = {colleague_profile}")
            logger.info(f"🔍 STEP 4 INPUT DEBUG: colleague_base from profile = '{colleague_base}'")
            logger.info(f"🔍 STEP 4 INPUT DEBUG: colleague_phase from profile = '{colleague_phase}'")
            
            # If base_type is wrong, let's see what's in state
            if colleague_base and colleague_base.lower() == 'thinker' and colleague_phase and colleague_phase.lower() == 'thinker':
                logger.error(f"🚨 FOUND THE BUG! BASE and PHASE are both 'thinker' - likely confusion!")
                logger.error(f"🚨 The BASE should be 'Promoter' but it's '{colleague_base}'")
                logger.error(f"🚨 The PHASE 'thinker' is correct (user selected A)")
                logger.error(f"🚨 DEBUG: state.get('coworker_other_profile') = {state.get('coworker_other_profile', {})}")
                logger.error(f"🚨 DEBUG: other_profile = {other_profile}")
            # Get user profile information from updated progression (not state!)
            user_phase = user_pcm_phase if user_pcm_phase and user_pcm_phase != 'Non spécifié' else None
            user_base = user_pcm_base if user_pcm_base and user_pcm_base != 'Non spécifié' else None
            # Get all phase section types
            phase_section_types = ['psychological_needs', 'negative_satisfaction', 'distress_sequence', 'action_plan']

            logger.info(f"👥 Step 4: User: BASE={user_base}, PHASE={user_phase}, Colleague: BASE={colleague_base}, PHASE={colleague_phase}")
            
            logger.info(f"👥 Step 4 DEBUG: colleague_base condition: {colleague_base} != 'Unknown' = {colleague_base != 'Unknown'}")
            logger.info(f"👥 Step 4 DEBUG: colleague_phase condition: {colleague_phase} != 'Unknown' = {colleague_phase != 'Unknown'}")
            logger.info(f"👥 Step 4 DEBUG: user_phase condition: {user_phase} != 'Unknown' = {user_phase != 'Unknown'}")
            logger.info(f"👥 Step 4 DEBUG: user_base condition: {user_base} != 'Unknown' = {user_base != 'Unknown'}")

            # If we have colleague's BASE, get BASE adaptation strategies
            if colleague_base and colleague_base != 'Unknown':
                # Get adaptation strategies for working with this colleague (base)
                colleague_base_filters = base_filters.copy()
                colleague_base_filters['pcm_base_type'] = colleague_base.lower()

                colleague_base_results = perform_supabase_vector_search(
                    query=user_query + f" communicate with {colleague_base} workplace",
                    match_function='match_documents',
                    metadata_filters=colleague_base_filters,
                    limit=12
                )
                all_results.extend(colleague_base_results)
                logger.info(f"👥 Step 4: {len(colleague_base_results)} colleague base results")
            
            # If we have colleague's PHASE, get PHASE adaptation strategies
            if colleague_phase and colleague_phase != 'Unknown':     
                for section_type in phase_section_types:
                    colleague_phase_filters = base_filters.copy()
                    colleague_phase_filters['pcm_phase_type'] = colleague_phase.lower()
                    colleague_phase_filters['section_type'] = section_type
                    
                    colleague_phase_results = perform_supabase_vector_search(
                        query=user_query,
                        match_function='match_documents',
                        metadata_filters=colleague_phase_filters,
                        limit=3  # Per section for complete understanding
                    )
                    all_results.extend(colleague_phase_results)
                    logger.info(f"👥 Step 4: {len(colleague_phase_results)} colleague phase results")

            # Action plan for user (based on their phase if identified)
            if user_phase and user_phase != 'Unknown':
                for section_type in phase_section_types:
                    user_phase_filters = base_filters.copy()
                    user_phase_filters['pcm_phase_type'] = user_phase.lower()
                    user_phase_filters['section_type'] = section_type
                    
                    user_phase_results = perform_supabase_vector_search(
                        query=user_query,
                        match_function='match_documents',
                        metadata_filters=user_phase_filters,
                        limit=3  # Per section for complete understanding
                    )
                    all_results.extend(user_phase_results)
                    logger.info(f"👥 Step 4: {len(user_phase_results)} user phase results")

            # User BASE strategies
            if user_base and user_base != 'Unknown':
                user_base_filters = base_filters.copy()
                user_base_filters['pcm_base_type'] = user_base.lower()
                user_base_results = perform_supabase_vector_search(
                    query=user_query,
                    match_function='match_documents',
                    metadata_filters=user_base_filters,
                    limit=3  # Per section for complete understanding
                )
                all_results.extend(user_base_results)
                logger.info(f"👥 Step 4: {len(user_base_results)} user base results")
            
            logger.info(f"👥 Step 4: Total action plans and recommendations: {len(all_results)} results")
        
        # Use the all_results that were populated during Step 2.1 or 2.2
        search_results = all_results
        logger.info(f"👥 COWORKER_FOCUSED step {coworker_step} search completed: {len(search_results)} results")
        
        # These variables were updated by _analyze_coworker_progression
        logger.info(f"🔄 BEFORE creating updated_coworker_state: other_profile = {other_profile}")
        # IMPORTANT: La valeur a pu être modifiée dans _analyze_coworker_progression
        # On doit récupérer la valeur ACTUELLE du state, pas l'ancienne
        updated_coworker_state = {
            'coworker_step': coworker_step,
            'coworker_self_ok': self_ok,
            'coworker_other_profile': other_profile,
            'coworker_step_2_substep': coworker_step_2_substep,
            'coworker_step_1_attempts': state.get('coworker_step_1_attempts', 0)  # Récupère la valeur modifiée
        }
        logger.info(f"🔄 AFTER creating updated_coworker_state: other_profile in dict = {updated_coworker_state.get('coworker_other_profile')}")
    elif pcm_base_or_phase == 'phase' and user_pcm_phase:
        # Filter for PHASE documents matching user's phase type
        # PHASE requires 3 separate searches: psychological_needs, negative_satisfaction, distress_sequence
        all_results = []
        phase_section_types = ['psychological_needs', 'negative_satisfaction', 'distress_sequence']
        
        for section_type in phase_section_types:
            phase_filters = base_filters.copy()
            phase_filters['pcm_phase_type'] = user_pcm_phase
            phase_filters['section_type'] = section_type
            
            logger.info(f"🔍 PHASE search {section_type} for {user_pcm_phase}")
            section_results = perform_supabase_vector_search(
                query=user_query,
                match_function='match_documents',
                metadata_filters=phase_filters,
                limit=3  # Limit per section_type
            )
            all_results.extend(section_results)
        
        search_results = all_results
        logger.info(f"🔍 PHASE 3-search completed: {len(search_results)} total results")
    elif flow_type == 'self_focused' and pcm_base_or_phase == 'action_plan':
        # ACTION_PLAN requires comprehensive search: BASE (6 dimensions) + PHASE (3 sections) + ACTION_PLAN (specific to phase)
        logger.info(f"🎯 Starting comprehensive ACTION_PLAN search: BASE + PHASE + ACTION_PLAN")
        all_results = []
        
        # SECTION 1: Get 2 most relevant BASE results for foundation
        if user_pcm_base:
            logger.info(f"🔍 ACTION_PLAN Section 1: BASE foundation ({user_pcm_base}) - top 2 most relevant")
            
            base_dim_filters = base_filters.copy()
            base_dim_filters['pcm_base_type'] = user_pcm_base
            
            base_results = perform_supabase_vector_search(
                query=user_query,
                match_function='match_documents',
                metadata_filters=base_dim_filters,
                limit=2  # Only 2 most relevant BASE results
            )
            all_results.extend(base_results)
            
            logger.info(f"🔍 ACTION_PLAN Section 1 completed: {len(base_results)} BASE results")
        
        # SECTION 2: Get PHASE context (3 sections)
        if user_pcm_phase:
            phase_section_types = ['psychological_needs', 'negative_satisfaction', 'distress_sequence']
            logger.info(f"🔍 ACTION_PLAN Section 2: PHASE context ({user_pcm_phase}) - 3 sections")
            
            for section_type in phase_section_types:
                phase_filters = base_filters.copy()
                phase_filters['section_type'] = section_type
                phase_filters['pcm_phase_type'] = user_pcm_phase
                
                phase_results = perform_supabase_vector_search(
                    query=user_query,
                    match_function='match_documents',
                    metadata_filters=phase_filters,
                    limit=2  # Limit per PHASE section
                )
                all_results.extend(phase_results)
            
            logger.info(f"🔍 ACTION_PLAN Section 2 completed: {len([r for r in all_results if r.get('metadata', {}).get('pcm_phase_type')])} PHASE results")
        
        # SECTION 3: Get ACTION_PLAN specific guidance for the user's phase
        if user_pcm_phase:
            logger.info(f"🔍 ACTION_PLAN Section 3: Practical guidance for {user_pcm_phase}")
            action_plan_filters = base_filters.copy()
            action_plan_filters['section_type'] = 'action_plan'
            action_plan_filters['pcm_phase_type'] = user_pcm_phase
            
            action_plan_results = perform_supabase_vector_search(
                query=user_query,
                match_function='match_documents',
                metadata_filters=action_plan_filters,
                limit=5  # More results for ACTION_PLAN as it's the key
            )
            all_results.extend(action_plan_results)
            
            logger.info(f"🔍 ACTION_PLAN Section 3 completed: {len(action_plan_results)} ACTION_PLAN results")
        
        search_results = all_results
        logger.info(f"🎯 ACTION_PLAN comprehensive search completed: {len(search_results)} total results (BASE + PHASE + ACTION_PLAN)")
    elif flow_type == 'self_action_plan':
        # SELF_ACTION_PLAN: Focus spécifique sur les sections action_plan de la phase
        logger.info(f"🎯 Starting SELF_ACTION_PLAN search: Focus on action_plan sections")
        all_results = []
        
        if user_pcm_phase:
            logger.info(f"🔍 ACTION_PLAN Section: Practical guidance for {user_pcm_phase}")
            action_plan_filters = base_filters.copy()
            action_plan_filters['section_type'] = 'action_plan'
            action_plan_filters['pcm_phase_type'] = user_pcm_phase
            
            action_plan_results = perform_supabase_vector_search(
                query=user_query,
                match_function='match_documents',
                metadata_filters=action_plan_filters,
                limit=5
            )
            all_results.extend(action_plan_results)
            logger.info(f"🔍 ACTION_PLAN results: {len(action_plan_results)} results")
        else:
            logger.warning("⚠️ No user PCM phase found - cannot search action_plan sections")
        
        search_results = all_results
        logger.info(f"🎯 SELF_ACTION_PLAN search completed: {len(search_results)} total results")
    
    else:
        # For other cases: general_knowledge, or when no specific BASE/PHASE context
        # Initialize all_results for all paths
        all_results = []
        
        # IMPORTANT: Check specific contexts first
        if flow_type == 'self_focused' and pcm_base_or_phase == 'phase':
            # User asking about PHASE specifically - use 3 separate searches
            all_results = []
            phase_section_types = ['psychological_needs', 'negative_satisfaction', 'distress_sequence']
            
            for section_type in phase_section_types:
                phase_filters = base_filters.copy()
                phase_filters['section_type'] = section_type
                if user_pcm_phase:
                    phase_filters['pcm_phase_type'] = user_pcm_phase
                    logger.info(f"🔍 PHASE search {section_type} for specific user phase: {user_pcm_phase}")
                else:
                    logger.info(f"🔍 PHASE search {section_type} (general exploration)")
                
                section_results = perform_supabase_vector_search(
                    query=user_query,
                    match_function='match_documents',
                    metadata_filters=phase_filters,
                    limit=3  # Limit per section_type
                )
                all_results.extend(section_results)
            
            search_results = all_results
            logger.info(f"🔍 PHASE 3-search (general) completed: {len(search_results)} total results")
        elif specific_dimensions_list and pcm_base_or_phase == 'base':
            # Specific dimensions detected but no user profile - get all BASE types for these dimensions
            logger.info(f"🔍 Filtering for specific dimensions without user profile: {specific_dimensions_list}")
            
            all_results = []
            for dimension in specific_dimensions_list:
                # Convert display names to technical names first, then map to database values
                technical_dimension = DISPLAY_TO_TECHNICAL.get(dimension, dimension.lower())
                mapped_dimension = DIMENSION_MAPPING.get(technical_dimension, technical_dimension)
                
                dimension_filters = base_filters.copy()
                dimension_filters['section_type'] = mapped_dimension
                
                dimension_results = perform_supabase_vector_search(
                    query=user_query,
                    match_function='match_documents',
                    metadata_filters=dimension_filters,
                    limit=4
                )
                all_results.extend(dimension_results)
            
            search_results = all_results
        else:
            # True general search - no user profile or not self_focused
            logger.info("🔍 No additional PCM filtering applied (general search)")
        
        # For cases that didn't set search_results already, execute default search
        if 'search_results' not in locals():
            search_results = perform_supabase_vector_search(
                query=user_query,
                match_function='match_documents',
                metadata_filters=base_filters,
                limit=8
            )
    
    # Sanitize results
    # For PCM ACTION_PLAN searches OR coworker_focused step 2.2, remove similarity threshold and character limits
    if pcm_base_or_phase == 'action_plan':
        # No filtering for ACTION_PLAN - return all results from all 3 sections
        sanitized_results = sanitize_vector_results(
            results=search_results,
            required_filters=None,
            top_k=50,  # Allow many results for comprehensive ACTION_PLAN (BASE + PHASE + ACTION_PLAN)
            min_similarity=None,  # No similarity threshold
            max_chars_per_item=999999,  # No character limit per item
            max_total_chars=999999  # No total character limit
        )
    elif flow_type == 'coworker_focused' and coworker_step == 2 and coworker_step_2_substep == 1:
        # Step 2.1: Educational content (BASE + PHASE) - NO SANITIZATION to keep all sections
        logger.info(f"🎯 Step 2.1: SKIPPING sanitization to preserve all {len(search_results)} educational results")
        sanitized_results = search_results  # Use raw results without filtering
    elif pcm_base_or_phase == 'phase':
        # No filtering for PHASE - return all results
        sanitized_results = sanitize_vector_results(
            results=search_results,
            required_filters=None,
            top_k=20,  # Allow many more results for PHASE
            min_similarity=None,  # No similarity threshold for PHASE
            max_chars_per_item=999999,  # No character limit per item
            max_total_chars=999999  # No total character limit
        )
    elif pcm_base_or_phase == 'base' and user_pcm_base:
        # BASE keeps its original logic unchanged
        min_sim = 0.15  # Lower threshold to capture all BASE dimensions
        sanitized_results = sanitize_vector_results(
            results=search_results,
            required_filters=None,
            top_k=8,  # Allow more results for BASE (6 dimensions + buffer)
            min_similarity=min_sim,
            max_chars_per_item=1500,
            max_total_chars=10000  # More space for all dimensions
        )
    else:
        # Standard filtering for other cases
        min_sim = 0.30  # Standard threshold for other searches
        sanitized_results = sanitize_vector_results(
            results=search_results,
            required_filters=None,
            top_k=8,
            min_similarity=min_sim,
            max_chars_per_item=1500,
            max_total_chars=10000
        )
    
    # Prepare updated coworker state (if applicable)
    updated_coworker_state = {}
    if flow_type == 'coworker_focused':
        # These variables were updated by _analyze_coworker_progression
        # CRITICAL: Include the attempts counter that was modified in _analyze_coworker_progression
        updated_coworker_state = {
            'coworker_step': coworker_step,
            'coworker_self_ok': self_ok,
            'coworker_other_profile': other_profile,
            'coworker_step_2_substep': coworker_step_2_substep,
            'coworker_step_1_attempts': state.get('coworker_step_1_attempts', 0)  # This was modified in the state!
        }
    
    logger.info(f"📊 PCM search completed: {len(sanitized_results)} sanitized results")
    return sanitized_results, pcm_base_or_phase, updated_coworker_state

def _format_pcm_results_by_category(base_results: List[Dict], phase_results: List[Dict], general_results: List[Dict], flow_type: str, language: str) -> str:
    """Format PCM search results organized by category (Base, Phase, General)"""
    total_results = len(base_results) + len(phase_results) + len(general_results)
    
    if total_results == 0:
        return "No PCM-specific information found for this query."
    
    formatted_content = f"# PCM VECTOR SEARCH RESULTS\n"
    formatted_content += f"Flow: {flow_type} | Language: {language} | Total: {total_results} results\n\n"
    
    # Section BASE
    if base_results:
        formatted_content += f"## 📊 BASE RESULTS ({len(base_results)} items)\n"
        for i, result in enumerate(base_results, 1):
            content = result.get('content', '')
            metadata = result.get('metadata', {})
            similarity = result.get('similarity', 0)
            pcm_type = metadata.get('pcm_type', 'Unknown')
            section_type = metadata.get('section_type', 'Content')
            
            formatted_content += f"### Base Item {i}\n"
            formatted_content += f"**PCM Type**: {pcm_type} | **Section**: {section_type} | **Score**: {similarity:.3f}\n"
            formatted_content += f"{content}\n\n"
        formatted_content += "---\n\n"
    
    # Section PHASE
    if phase_results:
        formatted_content += f"## 🔄 PHASE RESULTS ({len(phase_results)} items)\n"
        for i, result in enumerate(phase_results, 1):
            content = result.get('content', '')
            metadata = result.get('metadata', {})
            similarity = result.get('similarity', 0)
            pcm_type = metadata.get('pcm_type', 'Unknown')
            section_type = metadata.get('section_type', 'Content')
            
            formatted_content += f"### Phase Item {i}\n"
            formatted_content += f"**PCM Type**: {pcm_type} | **Section**: {section_type} | **Score**: {similarity:.3f}\n"
            formatted_content += f"{content}\n\n"
        formatted_content += "---\n\n"
    
    # Section GENERAL
    if general_results:
        formatted_content += f"## 📚 GENERAL RESULTS ({len(general_results)} items)\n"
        for i, result in enumerate(general_results, 1):
            content = result.get('content', '')
            metadata = result.get('metadata', {})
            similarity = result.get('similarity', 0)
            pcm_type = metadata.get('pcm_type', 'Unknown')
            section_type = metadata.get('section_type', 'Content')
            
            formatted_content += f"### General Item {i}\n"
            formatted_content += f"**PCM Type**: {pcm_type} | **Section**: {section_type} | **Score**: {similarity:.3f}\n"
            formatted_content += f"{content}\n\n"
        formatted_content += "---\n\n"
    
    return formatted_content

def _format_action_plan_results_by_sections(all_results: List[Dict], flow_type: str, language: str) -> str:
    """
    Format ACTION_PLAN results into 3 distinct sections:
    SECTION 1: BASE Foundation (6 dimensions)
    SECTION 2: PHASE Current State (psychological needs, negative satisfaction, distress sequence)
    SECTION 3: ACTION_PLAN Practical Guidance (specific action plans)
    """
    # Separate results by their source
    base_foundation_results = []
    phase_current_state_results = []
    action_plan_guidance_results = []
    
    base_dimensions = ['perception', 'strengths', 'interaction_style', 'personality_part', 'channel_communication', 'environmental_preferences']
    phase_sections = ['psychological_needs', 'negative_satisfaction', 'distress_sequence']
    
    for result in all_results:
        metadata = result.get('metadata', {})
        section_type = metadata.get('section_type', '')
        pcm_base_type = metadata.get('pcm_base_type', '')
        pcm_phase_type = metadata.get('pcm_phase_type', '')
        
        if section_type in base_dimensions and pcm_base_type:
            base_foundation_results.append(result)
        elif section_type in phase_sections and pcm_phase_type:
            phase_current_state_results.append(result)
        elif section_type == 'action_plan' and pcm_phase_type:
            action_plan_guidance_results.append(result)
    
    formatted_content = f"# PCM ACTION_PLAN COMPREHENSIVE RESULTS\n"
    formatted_content += f"Flow: {flow_type} | Language: {language} | Total: {len(all_results)} results\n\n"
    
    # SECTION 1: BASE Foundation
    if base_foundation_results:
        formatted_content += f"## SECTION 1: BASE FOUNDATION ({len(base_foundation_results)} items)\n"
        formatted_content += f"Your foundational personality traits across all 6 dimensions:\n\n"
        
        for i, result in enumerate(base_foundation_results, 1):
            content = result.get('content', '')
            metadata = result.get('metadata', {})
            similarity = result.get('similarity', 0)
            pcm_type = metadata.get('pcm_base_type', 'Unknown')
            section_type = metadata.get('section_type', 'Content')
            
            formatted_content += f"### Base Foundation Item {i}\n"
            formatted_content += f"**PCM Base**: {pcm_type} | **Dimension**: {section_type} | **Score**: {similarity:.3f}\n"
            formatted_content += f"{content}\n\n"
        formatted_content += "---\n\n"
    
    # SECTION 2: PHASE Current State
    if phase_current_state_results:
        formatted_content += f"## SECTION 2: PHASE CURRENT STATE ({len(phase_current_state_results)} items)\n"
        formatted_content += f"Your current psychological needs, negative satisfaction patterns, and distress sequence:\n\n"
        
        for i, result in enumerate(phase_current_state_results, 1):
            content = result.get('content', '')
            metadata = result.get('metadata', {})
            similarity = result.get('similarity', 0)
            pcm_type = metadata.get('pcm_phase_type', 'Unknown')
            section_type = metadata.get('section_type', 'Content')
            
            formatted_content += f"### Phase State Item {i}\n"
            formatted_content += f"**PCM Phase**: {pcm_type} | **Section**: {section_type} | **Score**: {similarity:.3f}\n"
            formatted_content += f"{content}\n\n"
        formatted_content += "---\n\n"
    
    # SECTION 3: ACTION_PLAN Practical Guidance
    if action_plan_guidance_results:
        formatted_content += f"## SECTION 3: ACTION_PLAN PRACTICAL GUIDANCE ({len(action_plan_guidance_results)} items)\n"
        formatted_content += f"Specific strategies and recommendations for your current phase:\n\n"
        
        for i, result in enumerate(action_plan_guidance_results, 1):
            content = result.get('content', '')
            metadata = result.get('metadata', {})
            similarity = result.get('similarity', 0)
            pcm_type = metadata.get('pcm_phase_type', 'Unknown')
            section_type = metadata.get('section_type', 'Content')
            
            formatted_content += f"### Action Plan Item {i}\n"
            formatted_content += f"**PCM Phase**: {pcm_type} | **Section**: {section_type} | **Score**: {similarity:.3f}\n"
            formatted_content += f"{content}\n\n"
        formatted_content += "---\n\n"
    
    return formatted_content

def _format_step21_educational_results(all_results: List[Dict], state: Dict, language: str) -> str:
    """
    Format Step 2.1 educational results: BASE (6 dimensions) + PHASE (3 sections)
    Clear separation for user education about their stress patterns
    """
    user_pcm_base = state.get('pcm_base', 'Unknown')
    user_pcm_phase = state.get('pcm_phase', 'Unknown')
    
    formatted_content = f"# 📚 STEP 2.1: YOUR PCM PROFILE EDUCATION\n\n"
    
    # Séparer les résultats BASE vs PHASE
    base_results = []
    phase_results = []
    
    for result in all_results:
        metadata = result.get('metadata', {})
        pcm_base_type = metadata.get('pcm_base_type')
        pcm_phase_type = metadata.get('pcm_phase_type')
        
        if pcm_base_type:
            base_results.append(result)
        elif pcm_phase_type:
            phase_results.append(result)
    
    # 1. BASE SECTION (6 dimensions)
    if base_results:
        formatted_content += f"## 🏗️ YOUR BASE: {user_pcm_base.upper()}\n\n"
        
        # Organiser par section_type
        base_sections = {}
        for result in base_results:
            section_type = result.get('metadata', {}).get('section_type', 'general')
            if section_type not in base_sections:
                base_sections[section_type] = []
            base_sections[section_type].append(result)
        
        for section_type, section_results in base_sections.items():
            formatted_content += f"### {section_type.replace('_', ' ').title()}\n"
            for result in section_results[:2]:  # Limite par section
                content = result.get('content', '')[:500]
                formatted_content += f"{content}\n\n"
    
    # 2. PHASE SECTION (3 sections)
    if phase_results:
        formatted_content += f"## ⚡ YOUR PHASE: {user_pcm_phase.upper()}\n\n"
        
        # Organiser par section_type pour les 3 sections spécifiques
        phase_sections = {
            'psychological_needs': [],
            'negative_satisfaction': [], 
            'distress_sequence': []
        }
        
        for result in phase_results:
            section_type = result.get('metadata', {}).get('section_type', 'unknown')
            if section_type in phase_sections:
                phase_sections[section_type].append(result)
        
        # Afficher chaque section avec son titre descriptif
        section_titles = {
            'psychological_needs': '🎯 Besoins Psychologiques',
            'negative_satisfaction': '⚠️ Satisfaction Négative',
            'distress_sequence': '🌪️ Séquence de Détresse'
        }
        
        for section_type, section_results in phase_sections.items():
            if section_results:
                title = section_titles.get(section_type, section_type.title())
                formatted_content += f"### {title}\n"
                for result in section_results[:2]:  # Limite par section
                    content = result.get('content', '')[:400]
                    formatted_content += f"{content}\n\n"
    
    if not base_results and not phase_results:
        formatted_content += "No specific profile information found for your education.\n"
    
    return formatted_content

def _format_coworker_step2_base_phase_results(all_results: List[Dict], state: Dict, language: str) -> str:
    """
    Format Step 2 coworker results into BASE and PHASE sections (no ACTION_PLAN)
    Used for educating user about their own profile in coworker context
    """
    # Separate results by their source
    base_results = []
    phase_results = []
    
    base_dimensions = ['perception', 'strengths', 'interaction_style', 'personality_part', 'channel_communication', 'environmental_preferences']
    phase_sections = ['psychological_needs', 'negative_satisfaction', 'distress_sequence']
    
    for result in all_results:
        metadata = result.get('metadata', {})
        section_type = metadata.get('section_type', '')
        pcm_base_type = metadata.get('pcm_base_type', '')
        pcm_phase_type = metadata.get('pcm_phase_type', '')
        
        if section_type in base_dimensions and pcm_base_type:
            base_results.append(result)
        elif section_type in phase_sections and pcm_phase_type:
            phase_results.append(result)
    
    flow_type = state.get('flow_type', 'coworker_focused')
    formatted_content = f"# PCM COWORKER STEP 2 - USER PROFILE RESULTS\n"
    formatted_content += f"Flow: {flow_type} | Language: {language} | Total: {len(all_results)} results\n\n"
    
    # SECTION BASE  
    if base_results:
        formatted_content += f"## SECTION BASE\n"
        formatted_content += f"Your core personality traits that shape how you approach workplace relationships:\n\n"
        
        for i, result in enumerate(base_results, 1):
            content = result.get('content', '')
            metadata = result.get('metadata', {})
            similarity = result.get('similarity', 0)
            pcm_type = metadata.get('pcm_base_type', 'Unknown')
            section_type = metadata.get('section_type', 'Content')
            
            formatted_content += f"### Base Trait {i}\n"
            formatted_content += f"**PCM Base**: {pcm_type} | **Dimension**: {section_type} | **Score**: {similarity:.3f}\n"
            formatted_content += f"{content}\n\n"
        formatted_content += "---\n\n"
    
    # SECTION PHASE
    if phase_results:
        formatted_content += f"## SECTION PHASE\n"
        formatted_content += f"Your current psychological needs and stress patterns:\n\n"
        
        for i, result in enumerate(phase_results, 1):
            content = result.get('content', '')
            metadata = result.get('metadata', {})
            similarity = result.get('similarity', 0)
            pcm_type = metadata.get('pcm_phase_type', 'Unknown')
            section_type = metadata.get('section_type', 'Content')
            
            formatted_content += f"### Phase Element {i}\n"
            formatted_content += f"**PCM Phase**: {pcm_type} | **Section**: {section_type} | **Score**: {similarity:.3f}\n"
            formatted_content += f"{content}\n\n"
        formatted_content += "---\n\n"
    
    # Summary
    total_sections = len([s for s in [base_results, phase_results] if s])
    formatted_content += f"**Summary**: {len(all_results)} total results across {total_sections} sections\n\n"
    
    logger.info(f"🔍 DEBUG: _format_coworker_step2_base_phase_results returning: {len(formatted_content)} chars")
    logger.info(f"🔍 DEBUG: Content starts with: {formatted_content[:200]}...")
    
    return formatted_content


def _format_coworker_step4_results(all_results: List[Dict], state: Dict, language: str) -> str:
    """
    Format Step 4 results with clear USER vs COLLEAGUE separation
    Uses raw results from Step 4 searches and classifies them internally
    """
    user_pcm_base = state.get('pcm_base', 'Unknown')
    user_pcm_phase = state.get('pcm_phase', 'Unknown')
    colleague_profile = state.get('coworker_other_profile', {})
    colleague_base = colleague_profile.get('base_type', 'Unknown')
    colleague_phase = colleague_profile.get('phase_state', 'Unknown')
    
    formatted_content = f"# 🎯 STEP 4: COMPLETE 4-PROFILE ANALYSIS\n\n"
    
    # 1. USER PROFILE SECTION
    formatted_content += f"## 👤 USER PROFILE\n\n"
    formatted_content += f"### **BASE: {user_pcm_base}**\n"
    
    user_base_count = 0
    for result in all_results:
        metadata = result.get('metadata', {})
        result_base = metadata.get('pcm_base_type', '').lower()
        section_type = metadata.get('section_type', '')
        
        # USER BASE results (strengths, interaction_style, communication_channels)
        if (result_base == user_pcm_base.lower() and 
            section_type in ['strengths', 'interaction_style', 'communication_channels']):
            formatted_content += f"**{section_type.replace('_', ' ').title()}**: {result.get('content', '')}\n\n"
            user_base_count += 1
    
    formatted_content += f"### **PHASE: {user_pcm_phase} (Action Plan)**\n"
    user_phase_count = 0
    for result in all_results:
        metadata = result.get('metadata', {})
        result_phase = metadata.get('pcm_phase_type', '').lower()
        section_type = metadata.get('section_type', '')
        
        # USER PHASE results (action_plan)
        if (result_phase == user_pcm_phase.lower() and section_type == 'action_plan'):
            formatted_content += f"**Self Management**: {result.get('content', '')}\n\n"
            user_phase_count += 1
    
    # 2. COLLEAGUE PROFILE SECTION
    formatted_content += f"---\n\n## 👥 COLLEAGUE PROFILE\n\n"
    formatted_content += f"### **BASE: {colleague_base}**\n"
    
    colleague_base_count = 0
    for result in all_results:
        metadata = result.get('metadata', {})
        result_base = metadata.get('pcm_base_type', '').lower()
        section_type = metadata.get('section_type', '')
        
        # COLLEAGUE BASE results (communication_style, strengths, personality_parts)
        if (result_base == colleague_base.lower() and 
            section_type in ['communication_style', 'strengths', 'personality_parts']):
            formatted_content += f"**{section_type.replace('_', ' ').title()}**: {result.get('content', '')}\n\n"
            colleague_base_count += 1
    
    formatted_content += f"### **PHASE: {colleague_phase} (Action Plan)**\n"
    colleague_phase_count = 0
    for result in all_results:
        metadata = result.get('metadata', {})
        result_phase = metadata.get('pcm_phase_type', '').lower()
        section_type = metadata.get('section_type', '')
        
        # COLLEAGUE PHASE results (action_plan)
        if (result_phase == colleague_phase.lower() and section_type == 'action_plan'):
            formatted_content += f"**Support Strategy**: {result.get('content', '')}\n\n"
            colleague_phase_count += 1
    
    # Summary
    total_results = user_base_count + user_phase_count + colleague_base_count + colleague_phase_count
    formatted_content += f"---\n**Summary**: {total_results} total results | User: {user_base_count + user_phase_count} | Colleague: {colleague_base_count + colleague_phase_count}\n\n"
    
    return formatted_content

def _format_phase_results(results: List[Dict], language: str) -> str:
    """
    Format SELF_PHASE results from 3 sections: psychological_needs, negative_satisfaction, distress_sequence
    """
    if not results:
        return "# PCM PHASE SEARCH RESULTS\n\nNo phase-specific information found for your current state.\n"
    
    formatted_content = f"# PCM PHASE CURRENT STATE RESULTS\n"
    formatted_content += f"Language: {language} | Total: {len(results)} results\n\n"
    
    # Group results by section type
    psychological_needs = []
    negative_satisfaction = []
    distress_sequence = []
    
    for result in results:
        metadata = result.get('metadata', {})
        section_type = metadata.get('section_type', '')
        
        if section_type == 'psychological_needs':
            psychological_needs.append(result)
        elif section_type == 'negative_satisfaction':
            negative_satisfaction.append(result)
        elif section_type == 'distress_sequence':
            distress_sequence.append(result)
    
    # Format each section
    if psychological_needs:
        formatted_content += f"## PSYCHOLOGICAL NEEDS ({len(psychological_needs)} items)\n"
        formatted_content += f"Your current motivational needs and what energizes you:\n\n"
        
        for i, result in enumerate(psychological_needs, 1):
            content = result.get('content', '')
            metadata = result.get('metadata', {})
            similarity = result.get('similarity', 0)
            pcm_type = metadata.get('pcm_phase_type', 'Unknown')
            
            formatted_content += f"### Psychological Need {i}\n"
            formatted_content += f"**PCM Phase**: {pcm_type} | **Score**: {similarity:.3f}\n"
            formatted_content += f"{content}\n\n"
        formatted_content += "---\n\n"
    
    if negative_satisfaction:
        formatted_content += f"## NEGATIVE SATISFACTION ({len(negative_satisfaction)} items)\n"
        formatted_content += f"How you might be getting needs met in negative ways:\n\n"
        
        for i, result in enumerate(negative_satisfaction, 1):
            content = result.get('content', '')
            metadata = result.get('metadata', {})
            similarity = result.get('similarity', 0)
            pcm_type = metadata.get('pcm_phase_type', 'Unknown')
            
            formatted_content += f"### Negative Satisfaction {i}\n"
            formatted_content += f"**PCM Phase**: {pcm_type} | **Score**: {similarity:.3f}\n"
            formatted_content += f"{content}\n\n"
        formatted_content += "---\n\n"
    
    if distress_sequence:
        formatted_content += f"## DISTRESS SEQUENCE ({len(distress_sequence)} items)\n"
        formatted_content += f"Your patterns when under severe stress:\n\n"
        
        for i, result in enumerate(distress_sequence, 1):
            content = result.get('content', '')
            metadata = result.get('metadata', {})
            similarity = result.get('similarity', 0)
            pcm_type = metadata.get('pcm_phase_type', 'Unknown')
            
            formatted_content += f"### Distress Pattern {i}\n"
            formatted_content += f"**PCM Phase**: {pcm_type} | **Score**: {similarity:.3f}\n"
            formatted_content += f"{content}\n\n"
        formatted_content += "---\n\n"
    
    return formatted_content

def _format_base_results(results: List[Dict], language: str) -> str:
    """
    Format SELF_BASE results by grouping by dimension type
    """
    if not results:
        return "# PCM BASE SEARCH RESULTS\n\nNo base-specific information found for your personality.\n"
    
    formatted_content = f"# PCM BASE PERSONALITY RESULTS\n"
    formatted_content += f"Language: {language} | Total: {len(results)} results\n\n"
    
    # Group results by dimension (section_type in metadata)
    dimensions = {}
    for result in results:
        metadata = result.get('metadata', {})
        section_type = metadata.get('section_type', 'general')
        
        if section_type not in dimensions:
            dimensions[section_type] = []
        dimensions[section_type].append(result)
    
    # Order dimensions logically
    dimension_order = [
        'perception', 'strengths', 'interaction_style', 
        'personality_part', 'channel_communication', 'environmental_preferences'
    ]
    
    # Format each dimension section
    for dimension in dimension_order:
        if dimension in dimensions:
            dimension_results = dimensions[dimension]
            formatted_content += f"## {dimension.upper().replace('_', ' ')} ({len(dimension_results)} items)\n"
            formatted_content += f"Your {dimension.replace('_', ' ')} characteristics:\n\n"
            
            for i, result in enumerate(dimension_results, 1):
                content = result.get('content', '')
                metadata = result.get('metadata', {})
                similarity = result.get('similarity', 0)
                pcm_type = metadata.get('pcm_base_type', 'Unknown')
                
                formatted_content += f"### {dimension.title().replace('_', ' ')} {i}\n"
                formatted_content += f"**PCM Base**: {pcm_type} | **Score**: {similarity:.3f}\n"
                formatted_content += f"{content}\n\n"
            formatted_content += "---\n\n"
    
    # Add any remaining dimensions not in the standard order
    for dimension, dimension_results in dimensions.items():
        if dimension not in dimension_order:
            formatted_content += f"## {dimension.upper().replace('_', ' ')} ({len(dimension_results)} items)\n"
            
            for i, result in enumerate(dimension_results, 1):
                content = result.get('content', '')
                metadata = result.get('metadata', {})
                similarity = result.get('similarity', 0)
                pcm_type = metadata.get('pcm_base_type', 'Unknown')
                
                formatted_content += f"### {dimension.title().replace('_', ' ')} {i}\n"
                formatted_content += f"**PCM Base**: {pcm_type} | **Score**: {similarity:.3f}\n"
                formatted_content += f"{content}\n\n"
            formatted_content += "---\n\n"
    
    return formatted_content

def _get_pcm_fallback(flow_type: str, language: str) -> str:
    """Provide fallback PCM information when vector search fails"""
    pcm_overview = f"""
# Process Communication Model (PCM) Overview (Flow: {flow_type})

PCM identifies 6 personality types, each with unique characteristics:

## The 6 PCM Types:
1. **Thinker** - Logical, organized, responsible
2. **Persister** - Dedicated, conscientious, observant  
3. **Harmonizer** - Compassionate, sensitive, warm
4. **Imaginer** - Reflective, imaginative, calm
5. **Rebel** - Spontaneous, creative, playful
6. **Promoter** - Adaptable, charming, persuasive

Each type has specific psychological needs, communication preferences, and stress patterns.
"""
    
    return pcm_overview

def _has_previous_step21_education(conversation_context: str) -> bool:
    """Check if we've already provided PCM education in Step 2.1"""
    if not conversation_context or conversation_context == "First coworker interaction.":
        return False
    
    # Look for signs of PCM education already provided
    pcm_education_indicators = [
        "harmonizer", "base", "phase", "psychological needs", "distress sequence", 
        "negative satisfaction", "interaction style", "strengths", "perception",
        "your personality", "pcm profile", "communication style", "compassionate",
        "sensitive", "warm", "benevolent", "recognition of person"
    ]
    
    context_lower = conversation_context.lower()
    education_mentions = sum(1 for indicator in pcm_education_indicators if indicator in context_lower)
    
    # If we have 3+ education indicators, we've likely done Step 2.1 education before
    return education_mentions >= 3

def _count_step21_turns(messages: List, current_step: int = None, current_substep: int = None) -> int:
    """Count how many assistant messages we've had in Step 2.1"""
    if current_step != 2 or current_substep != 1:
        return 0
    
    # Simple approach: count assistant messages in recent conversation
    assistant_messages_in_step21 = 0
    
    # Count assistant messages (we're in Step 2.1, so count recent assistant messages)
    for msg in messages[-4:]:  # Look at last 4 messages
        if hasattr(msg, 'type'):
            msg_type = msg.type
        else:
            msg_type = msg.get('type', 'unknown')
        
        # Count assistant messages
        if msg_type != "human":
            assistant_messages_in_step21 += 1
    
    logger.info(f"👥 Step 2.1 turn counter: {assistant_messages_in_step21} assistant messages in Step 2.1")
    return assistant_messages_in_step21

def _build_conversation_context_coworker(messages: List) -> str:
    """Build conversation context specifically for coworker_focused analysis"""
    if len(messages) <= 1:
        return "First coworker interaction."
    
    # Take the last 4 messages for coworker context
    recent_messages = messages[-4:-1]  # Exclude current message
    
    context_lines = []
    for msg in recent_messages:
        if hasattr(msg, 'type'):
            role = "User" if msg.type == "human" else "Assistant"
            content = msg.content
        else:
            role = "User" if msg.get('type') == "human" else "Assistant"
            content = msg.get('content', '')
        
        # Truncate but keep coworker-relevant keywords
        if len(content) > 200:
            content = content[:200] + "..."
        
        context_lines.append(f"{role}: {content}")
    
    return "\n".join(context_lines)

def _assess_emotional_state_with_llm(user_query: str, conversation_context: str) -> str:
    """
    Utilise un LLM pour évaluer l'état émotionnel +/+ ou -/- de l'utilisateur
    Returns: 'positive', 'negative', or 'unclear'
    """
    assessment_prompt = f"""You are a workplace psychology expert. Analyze the user's emotional state regarding their workplace relationship situation.

**CONVERSATION CONTEXT:**
{conversation_context}

**CURRENT USER MESSAGE:**
"{user_query}"

**YOUR TASK:**
Determine if the user is in a positive (+/+) or negative (-/-) emotional state about this workplace relationship.

**POSITIVE STATE (+/+) indicators:**
- Feels comfortable, confident, calm about the relationship
- No significant stress or tension
- Manageable situation, feeling in control
- Curious about improving but not urgent/distressed

**NEGATIVE STATE (-/-) indicators:**
- Experiencing stress, frustration, tension, conflict
- Feeling overwhelmed, anxious, or upset
- Relationship is causing emotional distress
- Urgent need for help or relief

**UNCLEAR STATE:**
- Not enough information to determine emotional state
- Neutral or mixed signals
- Need more information

Respond with EXACTLY ONE WORD: "positive", "negative", or "unclear"

Examples:
- "I'm struggling with my boss" → negative
- "I feel anxious about my manager" → negative
- "He is always contacting me and it's stressful" → negative
- "Things are going okay but I want to improve" → positive  
- "My colleague and I work well together" → positive
- "This situation is really stressing me out" → negative
- "Tell me about PCM" → unclear

Your response:"""

    try:
        logger.info("🤖 Calling LLM for emotional state assessment")
        result = isolated_analysis_call_with_messages(
            system_content=assessment_prompt,
            user_content="Analyze the emotional state based on the context above."
        )
        
        # Clean and validate result
        result = result.strip().lower()
        if result in ['positive', 'negative', 'unclear']:
            logger.info(f"✅ LLM assessment: {result}")
            return result
        else:
            logger.warning(f"⚠️ Unexpected LLM response: {result}, defaulting to unclear")
            return 'unclear'
            
    except Exception as e:
        logger.error(f"❌ LLM assessment failed: {e}, defaulting to unclear")
        return 'unclear'

def _assess_understanding_with_llm(user_query: str, topic: str) -> str:
    """
    Utilise un LLM pour évaluer si l'utilisateur a compris l'explication
    Returns: 'understood' or 'needs_more'
    """
    understanding_prompt = f"""You are analyzing if a user has understood an explanation about {topic}.

**USER RESPONSE:**
"{user_query}"

**YOUR TASK:**
Determine if the user shows understanding and is ready to move to the next step, or if they need more explanation.

**SIGNS OF UNDERSTANDING:**
- Acknowledges the explanation ("I understand", "makes sense", "interesting", "I see")
- Asks follow-up questions showing engagement
- Relates it to their situation
- Shows insight or "aha" moments
- Engaged response (more than just "ok" or "yes")

**NEEDS MORE EXPLANATION:**
- Very short, non-engaged responses ("ok", "yes", "sure")
- Shows confusion or asks for clarification
- Doesn't acknowledge the information
- Seems distracted or off-topic

Respond with EXACTLY ONE WORD: "understood" or "needs_more"

Examples:
- "That really makes sense for my situation" → understood
- "Interesting, I can see how that applies to me" → understood
- "I understand, that explains a lot" → understood
- "Ok" → needs_more
- "Yes" → needs_more
- "Can you clarify what you mean?" → needs_more

Your response:"""

    try:
        logger.info(f"🤖 Calling LLM to assess understanding of {topic}")
        result = isolated_analysis_call_with_messages(
            system_content=understanding_prompt,
            user_content="Analyze if the user understood based on their response above."
        )
        
        # Clean and validate result
        result = result.strip().lower()
        if result in ['understood', 'needs_more']:
            logger.info(f"✅ LLM understanding assessment: {result}")
            return result
        else:
            logger.warning(f"⚠️ Unexpected LLM response: {result}, defaulting to needs_more")
            return 'needs_more'
            
    except Exception as e:
        logger.error(f"❌ LLM understanding assessment failed: {e}, defaulting to needs_more")
        return 'needs_more'

def _assess_coworker_readiness_with_llm(user_query: str, conversation_context: str, is_in_coworker_flow: bool = False) -> str:
    """
    Fonction spécialisée pour évaluer si l'utilisateur est prêt à passer à l'exploration du collègue (Step 2.2→3)
    Le LLM doit d'abord vérifier s'il y a une personne spécifique mentionnée avant d'évaluer la disponibilité
    """
    assessment_prompt = f"""You evaluate if a user is ready to explore their colleague after receiving stress management advice.

**CONVERSATION CONTEXT:**
{conversation_context}

**CURRENT USER MESSAGE:**
"{user_query}"

**STEP 1 - VERIFY SPECIFIC PERSON (if not in confirmed coworker flow):**
{"Skip this - we're already in confirmed coworker flow with specific person identified." if is_in_coworker_flow else "Look at User: messages above. Is there a SPECIFIC individual mentioned?"}

**SPECIFIC PERSON = YES:**
- "My manager", "my boss", "my colleague [name/description]", "this person", "he/she"  
- "Mon manager", "mon chef", "ma collègue", "il/elle", specific names
- Any identifiable individual (not groups)

**SPECIFIC PERSON = NO:**
- "My colleagues" (plural), "people at work", "everyone", "the team"
- Public speaking, general social anxiety, meetings with groups

**STEP 2 - ASSESS READINESS:**
{"Since specific person confirmed, assess if ready:" if is_in_coworker_flow else "If NO specific person → respond 'needs_more'. If YES specific person → assess readiness:"}

**READY INDICATORS:**
- Positive responses: "ok", "yes", "oui", "sure", "d'accord", "sounds good", "perfect"
- Readiness: "I'm ready", "je suis prête", "je suis prêt", "let's go", "allons-y" 
- Simple agreement: single word confirmations

**NOT READY INDICATORS:**
- Still stressed: "I'm still anxious", "encore stressée"
- Wants more help: "need more strategies", "besoin d'aide"
- Clear rejection: "not ready", "pas encore", "not yet"

**EXAMPLES:**
- "oui je suis prête" → ready_for_action
- "ok" → ready_for_action
- "sounds good" → ready_for_action  
- "I'm still stressed" → needs_more

Respond EXACTLY: "ready_for_action" or "needs_more"

Your response:"""

    try:
        logger.info(f"🤖 COWORKER READINESS: Assessing if user ready for colleague exploration")
        logger.info(f"🔍 DEBUG: User query: '{user_query}'")
        logger.info(f"🔍 DEBUG: Conversation context: '{conversation_context[:200]}...'")
        logger.info(f"🔍 DEBUG: is_in_coworker_flow: {is_in_coworker_flow}")
        
        # NOUVELLE LOGIQUE: D'abord vérifier s'il y a une personne spécifique dans l'historique
        # Le LLM va faire cette vérification et l'évaluation en une seule fois
        
        result = isolated_analysis_call_with_messages(
            system_content=assessment_prompt,
            user_content="",
            model="gpt-3.5-turbo"
        )
        
        cleaned_result = result.strip().lower()
        logger.info(f"🤖 COWORKER READINESS LLM result: '{cleaned_result}'")
        
        if 'ready_for_action' in cleaned_result:
            logger.info("✅ COWORKER READINESS: ready_for_action")
            return 'ready_for_action'
        elif 'needs_more' in cleaned_result:
            logger.info("✅ COWORKER READINESS: needs_more")
            return 'needs_more'
        else:
            logger.warning(f"⚠️ Unexpected COWORKER READINESS response: {result}, defaulting to needs_more")
            return 'needs_more'
            
    except Exception as e:
        logger.error(f"❌ COWORKER READINESS assessment failed: {e}, defaulting to needs_more")
        return 'needs_more'

def _assess_understanding_and_readiness_with_llm(user_query: str, conversation_context: str, current_topic: str, next_action: str) -> str:
    """
    Utilise un LLM pour évaluer si l'utilisateur a compris le topic actuel ET est prêt pour la prochaine action
    Returns: 'ready_for_action' or 'needs_more'
    """
    assessment_prompt = f"""You are a coaching psychology expert. Analyze if the user has understood the current topic and is ready to move to the next action.

**CONVERSATION CONTEXT:**
{conversation_context}

**CURRENT TOPIC BEING EXPLAINED:**
{current_topic}

**NEXT ACTION BEING PROPOSED:**
{next_action}

**CURRENT USER MESSAGE:**
"{user_query}"

**YOUR TASK:**
Determine if the user:
1. Shows understanding of the current topic
2. Is ready/wants to move to the next action

**READY FOR ACTION indicators:**
- Shows clear understanding: "makes sense", "I understand", "that explains it", "helpful"
- Explicitly asks for next step: "what should I do?", "action plan", "help me", "next step"
- Confirms readiness: "yes", "ready", "let's do it", "I'd like that"
- Acknowledges and wants to progress: "yes, how can we fix this?", "that helps, what now?"
- Expresses comfort/willingness to proceed: "I feel ok to", "I'm ready to", "I think I can", "feel comfortable"
- Shows implementation readiness: "start implementing", "try implementing", "ready to implement", "feel ok to start"
- Positive attitude about moving forward: "I feel confident", "I'm willing to try", "sounds good to me"

**NEEDS MORE indicators:**
- Very short responses without engagement: just "ok", "sure"
- Shows confusion: "I don't understand", "can you clarify?"
- Asks for more explanation: "tell me more", "can you explain?"
- Changes topic or avoids the question
- Neutral acknowledgment without readiness: "interesting", "I see"

Respond with EXACTLY ONE WORD: "ready_for_action" or "needs_more"

Examples:
- "Yes, that makes sense. What should I do about it?" → ready_for_action
- "That really helps me understand. I'd like an action plan" → ready_for_action
- "Exactly! How can I improve this situation?" → ready_for_action
- "I understand. What's next?" → ready_for_action
- "Yes it helps" → ready_for_action
- "Yes I would like some concrete strategies" → ready_for_action
- "yes" (after asking for strategies) → ready_for_action
- "Yes I think I feel ok to start implementing this" → ready_for_action
- "I feel ready to try this" → ready_for_action
- "I'm comfortable moving forward" → ready_for_action
- "Ok" (without context) → needs_more
- "Interesting" → needs_more
- "Can you tell me more about that?" → needs_more

Your response:"""

    try:
        logger.info(f"🤖 Calling LLM to assess understanding of '{current_topic}' and readiness for '{next_action}'")
        result = isolated_analysis_call_with_messages(
            system_content=assessment_prompt,
            user_content="",
            model="gpt-3.5-turbo"  # GPT-3.5 is sufficient for simple transitions
        )
        
        cleaned_result = result.strip().lower()
        
        if 'ready_for_action' in cleaned_result:
            logger.info("✅ LLM assessment: ready_for_action")
            return 'ready_for_action'
        elif 'needs_more' in cleaned_result:
            logger.info("✅ LLM assessment: needs_more")
            return 'needs_more'
        else:
            logger.warning(f"⚠️ Unexpected LLM response: {result}, defaulting to needs_more")
            return 'needs_more'
            
    except Exception as e:
        logger.error(f"❌ LLM understanding and readiness assessment failed: {e}, defaulting to needs_more")
        return 'needs_more'

def _normalize_text(text: str) -> str:
    """Normalise le texte : supprime accents, convertit en majuscules"""
    import unicodedata
    # Supprimer les accents
    text = unicodedata.normalize('NFD', text)
    text = ''.join(char for char in text if unicodedata.category(char) != 'Mn')
    return text.upper().strip()

def _detect_base_choice(user_query: str) -> str:
    """
    Détecte si l'utilisateur a choisi une BASE (A, B, C, D, E, F) par lettre ou nom
    Gère majuscules/minuscules et accents/sans accents
    Retourne la lettre choisie ou None
    """
    user_query_normalized = _normalize_text(user_query)
    
    # Recherche directe de lettre isolée
    if user_query_normalized in ['A', 'B', 'C', 'D', 'E', 'F']:
        return user_query_normalized
    
    # Recherche par nom de type (toutes variantes normalisées)
    name_mapping = {
        # Anglais (normalisé)
        'THINKER': 'A',
        'PERSISTER': 'B', 
        'HARMONIZER': 'C',
        'REBEL': 'D',
        'IMAGINER': 'E',
        'PROMOTER': 'F',
        # Français (normalisé - sans accents)
        'ANALYSEUR': 'A',
        'PERSEVERANT': 'B',  # persévérant devient perseverant
        'EMPATHIQUE': 'C', 
        'IMAGINEUR': 'E',
        'PROMOTEUR': 'F'
    }
    
    for name, letter in name_mapping.items():
        if name in user_query_normalized:
            return letter
    
    # Recherche de patterns plus complexes
    import re
    
    # Patterns comme "I choose A", "A)", "Option A", "A - Thinker", etc.
    patterns = [
        r'\b([ABCDEF])\b',  # Lettre isolée
        r'(?:choose|select|pick)\s*([ABCDEF])',  # "I choose A"
        r'([ABCDEF])\s*[\)\-\:]',  # "A)" ou "A-" ou "A:"
        r'option\s*([ABCDEF])',  # "Option A"
        r'([ABCDEF])\s*(?:thinker|persister|harmonizer|rebel|imaginer|promoter)',  # "A Thinker"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, user_query_clean, re.IGNORECASE)
        if match:
            letter = match.group(1).upper()
            if letter in ['A', 'B', 'C', 'D', 'E', 'F']:
                return letter
    
    # Recherche de noms de BASE directement
    base_names = {
        'THINKER': 'A',
        'PERSISTER': 'B', 
        'HARMONIZER': 'C',
        'REBEL': 'D',
        'IMAGINER': 'E',
        'PROMOTER': 'F'
    }
    
    for base_name, letter in base_names.items():
        if base_name in user_query_clean:
            return letter
    
    return None

def _detect_phase_choice(user_query: str) -> str:
    """
    Détecte si l'utilisateur a choisi une PHASE (A pour OK, B pour Distress)
    Retourne 'ok' ou 'distress' ou None
    """
    user_query_clean = user_query.strip().upper()
    
    # Recherche directe de lettre
    if user_query_clean == 'A':
        return 'ok'
    elif user_query_clean == 'B':
        return 'distress'
    
    # Recherche de patterns plus complexes
    import re
    
    # Patterns pour A (OK/Positive)
    positive_patterns = [
        r'(A|POSITIVE|OK|GOOD|FINE|COMFORTABLE|CONFIDENT)',
        r'(?:choose|select|pick)\s*A',
        r'A\s*[\)\-\:]',
        r'positive\s*phase',
        r'ok\s*phase'
    ]
    
    # Patterns pour B (Distress/Negative)
    negative_patterns = [
        r'(B|DISTRESS|NEGATIVE|STRESSED|OVERWHELMED)',
        r'(?:choose|select|pick)\s*B',
        r'B\s*[\)\-\:]',
        r'distress\s*phase',
        r'negative\s*phase'
    ]
    
    for pattern in positive_patterns:
        if re.search(pattern, user_query_clean, re.IGNORECASE):
            return 'ok'
    
    for pattern in negative_patterns:
        if re.search(pattern, user_query_clean, re.IGNORECASE):
            return 'distress'
    
    return None

def _detect_emotional_state_choice(user_query: str) -> str:
    """
    Utilise un LLM pour détecter si l'utilisateur a choisi un état émotionnel (A pour Positif, B pour Négatif/Stress)
    Retourne 'positive' ou 'negative' ou None
    """
    choice_prompt = f"""You are analyzing if a user is making a choice about their coworker's emotional state.

CONTEXT: The user has been presented with two options:
- A) Positive/OK emotional state 
- B) Negative/Stressed emotional state

USER MESSAGE: "{user_query}"

TASK: Determine if the user is clearly choosing option A (positive) or B (negative) for their coworker's emotional state.

CLEAR A/POSITIVE CHOICES:
- "A" (standalone)
- "Option A" / "Choice A"
- "A) positive" / "A positive"
- "I choose A"
- "Positive" / "OK" / "Good" (when referring to emotional state)

CLEAR B/NEGATIVE CHOICES:
- "B" (standalone)
- "Option B" / "Choice B" 
- "B) negative" / "B negative"
- "I choose B" / "I would choose B" / "I would say B"
- "Negative" / "Stressed" / "Not OK" (when referring to emotional state)

NOT CHOICES (return null):
- "Yes maybe in B" (this is agreeing to something, not choosing B)
- "I think B might work" (discussing an option, not choosing)
- "We could try B" (suggesting, not choosing)
- General conversation not about the A/B choice

Respond with EXACTLY ONE WORD: "positive", "negative", or "null"

Examples:
- "A" → positive
- "I choose B" → negative  
- "I would say B" → negative
- "Option A looks right" → positive
- "Yes maybe in B" → null
- "B seems like the better choice" → negative
- "Sounds good" → null
- "I think my coworker is stressed" → null (description, not A/B choice)

Your response:"""

    try:
        logger.info(f"🤖 Calling LLM for emotional state choice detection: '{user_query}'")
        result = isolated_analysis_call_with_messages(
            system_content=choice_prompt,
            user_content=user_query,
            model="gpt-3.5-turbo"
        )
        
        # Clean and validate result
        result = result.strip().lower()
        if result == 'positive':
            logger.info(f"✅ LLM detected A/positive choice")
            return 'positive'
        elif result == 'negative':
            logger.info(f"✅ LLM detected B/negative choice")
            return 'negative'
        elif result == 'null':
            logger.info(f"✅ LLM detected no clear A/B choice")
            return None
        else:
            logger.warning(f"⚠️ Unexpected LLM response: {result}, treating as no choice")
            return None
            
    except Exception as e:
        logger.error(f"❌ LLM choice detection failed: {e}, falling back to regex")
        
        # Fallback simple pour les cas évidents
        user_query_clean = user_query.strip().upper()
        if user_query_clean == 'A':
            return 'positive'
        elif user_query_clean == 'B':
            return 'negative'
        
        return None

def _detect_emotional_state_change_with_llm(user_query: str, current_user_state: bool, current_colleague_state: str) -> Dict[str, Any]:
    """
    Utilise un LLM pour détecter si l'utilisateur veut changer l'état émotionnel (multilingue)
    """
    # Convertir current states en format lisible
    user_state_text = "POSITIVE (OK)" if current_user_state else "NEGATIVE (NOT OK)"
    colleague_state_text = current_colleague_state.upper() if current_colleague_state else "UNKNOWN"
    
    detection_prompt = f"""You are analyzing if a user wants to change emotional states in a workplace relationship context.

CURRENT EMOTIONAL STATES:
- User: {user_state_text}  
- Colleague: {colleague_state_text}

TASK: Detect if the user wants to change either emotional state based on their message.

DETECTION PATTERNS:
1. **User state changes**:
   - "Actually I feel better/positive/ok now"
   - "I changed my mind, I'm ok"
   - "Non je me sens bien en fait"
   - "En fait je vais mieux"

2. **Colleague state changes**:
   - "Actually my colleague is stressed/not ok"
   - "I think he/she is struggling"
   - "Mon collègue est stressé en fait"
   - "Elle ne va pas bien"

USER MESSAGE: "{user_query}"

RESPOND WITH JSON:
{{
  "change_detected": true/false,
  "target": "user"/"colleague"/null,
  "new_state": "positive"/"negative"/null,
  "confidence": 0.0-1.0,
  "reasoning": "explanation"
}}

If no emotional state change is detected, return {{"change_detected": false}}."""

    try:
        response = isolated_analysis_call_with_messages(
            system_content=detection_prompt,
            user_content="",
            model="gpt-3.5-turbo"
        )
        
        # Parse JSON response
        import json
        result = json.loads(response.strip())
        
        if result.get('change_detected'):
            return {
                'type': result.get('target'),
                'new_state': result.get('new_state'),
                'reason': result.get('reasoning', 'Emotional state change detected'),
                'confidence': result.get('confidence', 0.8)
            }
        
        return None
        
    except Exception as e:
        logger.error(f"❌ LLM emotional state change detection failed: {e}")
        return None


def _apply_emotional_state_change(
    emotional_change: Dict[str, Any], 
    current_self_ok: bool, 
    current_other_profile: Dict[str, Any]
) -> tuple:
    """
    Applique le changement d'état émotionnel et retourne les nouvelles valeurs
    """
    logger.info(f"🔄 Applying emotional change: {emotional_change}")
    
    if emotional_change['type'] == 'user':
        # Changer l'état de l'utilisateur
        new_self_ok = emotional_change['new_state'] == 'positive'
        logger.info(f"🔄 User emotional state: {current_self_ok} → {new_self_ok}")
        return new_self_ok, current_other_profile
        
    elif emotional_change['type'] == 'colleague':
        # Changer l'état du collègue
        updated_profile = current_other_profile.copy()
        updated_profile['emotional_state'] = emotional_change['new_state']
        
        # Si changement vers positif, supprimer phase_state (pas besoin de stress phase)
        if emotional_change['new_state'] == 'positive':
            updated_profile.pop('phase_state', None)
            updated_profile['recommendation_type'] = 'adaptation'
        # Si changement vers négatif, garder ou demander la phase de stress
        else:
            updated_profile['recommendation_type'] = 'stress_management'
        
        logger.info(f"🔄 Colleague emotional state changed to: {emotional_change['new_state']}")
        return current_self_ok, updated_profile
    
    return current_self_ok, current_other_profile


def _detect_stress_phase_choice(user_query: str) -> str:
    """
    Détecte si l'utilisateur a choisi une PHASE (A-F) par lettre ou nom
    Gère majuscules/minuscules et accents/sans accents
    Important: Les PHASES sont indépendantes des BASE types
    Retourne la lettre choisie (A-F) ou None
    """
    user_query_normalized = _normalize_text(user_query)
    
    # Recherche directe de lettre A-F
    valid_choices = ['A', 'B', 'C', 'D', 'E', 'F']
    if user_query_normalized in valid_choices:
        return user_query_normalized
    
    # Recherche par nom de PHASE (toutes variantes normalisées)
    phase_name_mapping = {
        # Anglais (normalisé)
        'THINKER': 'A',
        'PERSISTER': 'B', 
        'HARMONIZER': 'C',
        'REBEL': 'D',
        'IMAGINER': 'E',
        'PROMOTER': 'F',
        # Français (normalisé - sans accents)
        'ANALYSEUR': 'A',
        'PERSEVERANT': 'B',  # persévérant devient perseverant
        'EMPATHIQUE': 'C', 
        'IMAGINEUR': 'E',
        'PROMOTEUR': 'F'
    }
    
    for name, letter in phase_name_mapping.items():
        if name in user_query_normalized:
            return letter
    
    # Recherche de patterns plus complexes
    import re
    
    # Patterns pour chaque lettre
    for letter in valid_choices:
        patterns = [
            f'(?:choose|select|pick)\\s*{letter}',
            f'{letter}\\s*[\\)\\-\\:]',
            f'option\\s*{letter}',
            f'choice\\s*{letter}'
        ]
        
        for pattern in patterns:
            if re.search(pattern, user_query_clean, re.IGNORECASE):
                return letter
    
    # Recherche de noms de PHASES (avec mots-clés phase)
    phase_names = {
        'THINKER': 'A',
        'PERSISTER': 'B', 
        'HARMONIZER': 'C',
        'REBEL': 'D',
        'IMAGINER': 'E',
        'PROMOTER': 'F'
    }
    
    # Chercher "THINKER PHASE", "PHASE THINKER", etc.
    for phase_name, letter in phase_names.items():
        phase_patterns = [
            f'{phase_name}\\s*PHASE',
            f'PHASE\\s*{phase_name}',
            f'{phase_name}'  # Fallback pour nom seul
        ]
        
        for pattern in phase_patterns:
            if re.search(pattern, user_query_clean, re.IGNORECASE):
                return letter
    
    return None

def _analyze_coworker_progression(
    user_query: str,
    current_step: int,
    current_self_ok: bool, 
    current_other_profile: dict,
    conversation_context: str,
    state: dict,
    current_step_2_substep: int = 1,
    coworker_context_type: str = 'direct'
) -> tuple[int, bool, dict, int]:
    """
    Nouveau flux coworker_focused basé sur matrice +/+ et -/-:
    1: Identifier +/+ ou -/- → si +/+ proposer BASE, si -/- aller à ACTION_PLAN
    2: Si -/-, utiliser logique ACTION_PLAN self_focused → demander accord
       2.1: Expliquer BASE/PHASE et psychological needs
       2.2: Donner ACTION_PLAN concret
    3: Explorer collègue (BASE puis PHASE)
    4: Recommandations d'adaptation
    Returns: (new_step, self_ok, other_profile, step_2_substep)
    """
    
    # Sauvegarder la question originale pour Step 4 si c'est le premier message
    if not state.get('original_user_query') and current_step == 1:
        state['original_user_query'] = user_query
        logger.info(f"💾 Saved original question: '{user_query}'")
    
    logger.info(f"👥 Analyzing coworker progression from step {current_step}")
    logger.info(f"👥 DEBUG INPUT STATE: base_type={current_other_profile.get('base_type')}, base_confirmed={current_other_profile.get('base_confirmed')}, emotional_state={current_other_profile.get('emotional_state')}")
    
    # DEBUG: Track base_type changes throughout the function
    logger.info(f"🔍 PROGRESSION DEBUG START: base_type = '{current_other_profile.get('base_type')}'")
    original_base_type = current_other_profile.get('base_type')
    
    # Step 1: Matrice +/+ ou -/- assessment via LLM
    if current_step == 1:
        logger.info(f"👥 Step 1 assessment - User query: '{user_query}'")
        logger.info(f"👥 Step 1 assessment - Conversation context: '{conversation_context}'")
        logger.info(f"👥 Step 1 assessment - Context type: '{coworker_context_type}'")
        
        # Count total messages in this coworker conversation
        coworker_messages = len(state.get('messages', []))
        
        # Pas de règle spéciale pour contextual_base_phase - on laisse la progression normale se faire
        
        # RÈGLE SIMPLIFIÉE: Step 1 = TOUJOURS poser la question émotionnelle
        # On évalue SEULEMENT après avoir reçu une réponse à notre question Step 1
        
        coworker_step_1_attempts = state.get('coworker_step_1_attempts', 0)
        
        if coworker_step_1_attempts == 0:
            # Première fois en Step 1 → poser la question, pas d'analyse
            logger.info("👥 Step 1: First time - posing emotional assessment question")
            # Incrémenter pour la prochaine fois
            state['coworker_step_1_attempts'] = 1
            logger.info(f"👥 DEBUG: Updated coworker_step_1_attempts to {state['coworker_step_1_attempts']}")
            return 1, False, current_other_profile, current_step_2_substep
        else:
            # À partir du 3ème message → évaluer la réponse à notre question Step 1
            logger.info(f"👥 Step 1 assessment for {coworker_messages} messages - Assessing RESPONSE: '{user_query}'")
            assessment_result = _assess_emotional_state_with_llm(user_query, conversation_context)
            logger.info(f"👥 Step 1 assessment result: {assessment_result}")
            
            if assessment_result == 'negative':
                # État négatif détecté → passer à Step 2.1 pour éducation BASE/PHASE
                logger.info("👥 Step 1→2.1: Negative state detected, moving to BASE/PHASE education")
                return 2, False, current_other_profile, 1
            elif assessment_result == 'positive':
                # État positif détecté → passer à Step 3 pour explorer collègue  
                logger.info("👥 Step 1→3: Positive state detected, moving to colleague exploration")
                return 3, True, current_other_profile, current_step_2_substep
            else:  # 'unclear'
                # Si toujours unclear après question Step 1, forcer progression
                logger.info("👥 Step 1→3: Still unclear after Step 1 question, defaulting to colleague exploration")
                return 3, True, current_other_profile, current_step_2_substep
    
    # Step 2: ACTION_PLAN phase (quand -/-) avec sous-étapes
    elif current_step == 2:
        user_query_lower = user_query.lower()
        
        # Sub-step 2.1: BASE/PHASE explanation
        if current_step_2_substep == 1:
            user_query_lower = user_query.lower().strip()
            
            # Check if we've already had Step 2.1 conversations
            has_previous_21_education = _has_previous_step21_education(conversation_context)
            
            # Count turns in Step 2.1 to prevent infinite loops
            step21_turns = _count_step21_turns(state.get('messages', []), current_step, current_step_2_substep)
            
            logger.info(f"👥 Step 2.1 DEBUG: user_query='{user_query}', lower='{user_query_lower}'")
            logger.info(f"👥 Step 2.1 EDUCATION CHECK: has_previous_21_education={has_previous_21_education}")
            logger.info(f"👥 Step 2.1 TURN COUNTER: {step21_turns} turns in Step 2.1")
            
            # FORCED PROGRESSION: After 3 turns in Step 2.1, offer choices
            if step21_turns >= 3:
                logger.info("👥 Step 2.1→FORCED: 3+ turns detected, offering progression choices")
                
                # Check if user seems satisfied/understanding without wanting action plan
                satisfaction_indicators = [
                    "thanks", "merci", "that helps", "i understand", "makes sense", "i see", 
                    "helpful", "clear", "good to know", "interesting", "appreciate"
                ]
                
                shows_satisfaction = any(indicator in user_query_lower for indicator in satisfaction_indicators)
                
                if shows_satisfaction:
                    logger.info("👥 Step 2.1→3: FORCED progression - user shows satisfaction, moving to colleague exploration")
                    return 3, current_self_ok, current_other_profile, 1  # Skip to Step 3
                else:
                    logger.info("👥 Step 2.1→2.2: FORCED progression - moving to action plan after 3 turns")
                    return 2, current_self_ok, current_other_profile, 2  # Force to Step 2.2
            
            # SAFETY CHECK: Obvious action plan requests (fallback keywords)
            obvious_action_requests = [
                "concrete strategies", "action plan", "what should i do", "specific strategies",
                "yes i would like", "ready for strategies", "let's do it", "i'm ready"
            ]
            
            # CONTEXTUAL SAFETY CHECK: If recent conversation mentioned action plans and user says yes
            contextual_yes_pattern = (
                user_query_lower in ["yes", "oui", "sure", "ok", "d'accord"] and
                any(keyword in conversation_context.lower() for keyword in ["action plan", "strategies", "concrete", "help", "ready"])
            )
            
            has_obvious_request = any(phrase in user_query_lower for phrase in obvious_action_requests) or contextual_yes_pattern
            logger.info(f"👥 Step 2.1 CONTEXTUAL CHECK: contextual_yes_pattern={contextual_yes_pattern}")
            logger.info(f"👥 Step 2.1 SAFETY CHECK: has_obvious_request={has_obvious_request}")
            
            if has_obvious_request:
                logger.info(f"👥 Step 2.1→2.2: OBVIOUS action plan request detected: '{user_query}'")
                return 2, current_self_ok, current_other_profile, 2  # Move to substep 2.2
            
            # Use LLM to assess if user understands PCM explanation and wants action plan
            logger.info(f"👥 Step 2.1: Calling LLM assessment with query='{user_query}' and context='{conversation_context[:100]}...'")
            understanding_result = _assess_understanding_and_readiness_with_llm(
                user_query, 
                conversation_context,
                "PCM BASE/PHASE explanation",
                "action plan to get back to positive state"
            )
            logger.info(f"👥 Step 2.1 LLM assessment result: '{understanding_result}'")
            
            if understanding_result == 'ready_for_action':
                logger.info("👥 Step 2.1→2.2: LLM detected user understands and wants action plan")
                return 2, current_self_ok, current_other_profile, 2  # Move to substep 2.2
            else:
                logger.info(f"👥 Staying in Step 2.1: LLM result '{understanding_result}' suggests continuing BASE/PHASE explanation")
                return 2, current_self_ok, current_other_profile, 1
        
        # Sub-step 2.2: ACTION_PLAN concrete
        else:  # substep 2
            user_query_lower = user_query.lower().strip()
            
            # DIRECT DETECTION: Common French/English readiness phrases
            explicit_ready_phrases = [
                "oui je suis prête", "oui je suis pret", "yes i'm ready", "yes i am ready",
                "ok", "oui", "yes", "d'accord", "sure", "allons-y", "let's go"
            ]
            
            # Check for explicit readiness first (bypass LLM for obvious cases)
            if any(phrase in user_query_lower for phrase in explicit_ready_phrases):
                logger.info(f"👥 Step 2.2→3: EXPLICIT readiness detected: '{user_query}' - bypassing LLM")
                return 3, current_self_ok, current_other_profile, 1
            
            # Fallback to LLM assessment for complex cases
            readiness_result = _assess_coworker_readiness_with_llm(
                user_query,
                conversation_context,
                is_in_coworker_flow=True
            )
            logger.info(f"👥 Step 2.2 LLM assessment result: {readiness_result}")
            
            if readiness_result == 'ready_for_action':
                logger.info("👥 Step 2.2→3: LLM detected user ready for colleague exploration")
                return 3, current_self_ok, current_other_profile, 1  # Reset substep for next time
            else:
                logger.info("👥 Staying in Step 2.2: LLM suggests continuing ACTION_PLAN discussion")
                return 2, current_self_ok, current_other_profile, 2
    
    # Step 3: Explorer le collègue (BASE → PHASE → ÉTAT ÉMOTIONNEL)
    elif current_step == 3:
        # Step 3.1: Identifier la PHASE du collègue (après BASE confirmée)
        if current_other_profile.get('base_confirmed') and not current_other_profile.get('phase_state'):
            logger.info("👥 Step 3.1: Identifying colleague's PHASE")
            logger.info(f"👥 Step 3.1 DEBUG: base_confirmed={current_other_profile.get('base_confirmed')}, phase_state={current_other_profile.get('phase_state')}")
            
            # Détecter si l'utilisateur a choisi une phase (A-F)
            phase_choice = _detect_stress_phase_choice(user_query)
            
            if phase_choice:
                # Mapper le choix utilisateur vers le nom de la phase
                phase_mapping = {
                    'A': 'thinker',
                    'B': 'persister', 
                    'C': 'harmonizer',
                    'D': 'rebel',
                    'E': 'imaginer',
                    'F': 'promoter'
                }
                
                if phase_choice in phase_mapping:
                    selected_phase = phase_mapping[phase_choice]
                    current_other_profile['phase_state'] = selected_phase
                    logger.info(f"👥 Step 3.1: User selected PHASE: {selected_phase}")
                    
                    # Passer à Step 3.2 pour évaluer l'état émotionnel
                    return 3, current_self_ok, current_other_profile, current_step_2_substep
                else:
                    logger.info(f"👥 Step 3.1: Invalid phase choice '{phase_choice}' - staying in selection")
                    return 3, current_self_ok, current_other_profile, current_step_2_substep
            else:
                # Pas de choix clair - continuer à présenter les options de phase
                logger.info("👥 Step 3.1: No clear phase choice detected - presenting phase options")
                return 3, current_self_ok, current_other_profile, current_step_2_substep
                
        # Step 3.2: Évaluer l'état émotionnel (après PHASE confirmée)
        elif current_other_profile.get('phase_state') and not current_other_profile.get('emotional_state'):
            logger.info("👥 Step 3.2: Assessing colleague's emotional state")
            logger.info(f"👥 Step 3.2 DEBUG: phase_state={current_other_profile.get('phase_state')}, emotional_state={current_other_profile.get('emotional_state')}")
            
            # Détecter si l'utilisateur a choisi un état émotionnel (A pour OK, B pour pas OK)
            emotional_choice = _detect_emotional_state_choice(user_query)
            
            if emotional_choice:
                # Utilisateur a fait un choix d'état émotionnel - l'enregistrer
                current_other_profile['emotional_state'] = emotional_choice
                # Reset le flag de clarification puisqu'on a maintenant un choix clair
                current_other_profile.pop('needs_emotional_clarification', None)
                logger.info(f"👥 Step 3.2: User selected emotional state: {emotional_choice}")
                
                # Déterminer le type de recommandation selon l'état
                if emotional_choice == 'positive':
                    current_other_profile['recommendation_type'] = 'adaptation'
                    logger.info("👥 Colleague in positive state - will provide adaptation recommendations")
                else:
                    current_other_profile['recommendation_type'] = 'stress_management'
                    logger.info("👥 Colleague in negative state - will provide stress management recommendations")
                
                # Passer à Step 4 avec toutes les infos
                return 4, current_self_ok, current_other_profile, current_step_2_substep
            else:
                # Vérifier le nombre de tentatives de clarification
                clarification_attempts = current_other_profile.get('clarification_attempts', 0) + 1
                current_other_profile['clarification_attempts'] = clarification_attempts
                
                # Après 2 tentatives, forcer une progression avec un état par défaut
                if clarification_attempts >= 2:
                    logger.info("👥 Step 3.2: Max clarification attempts reached - forcing progression with negative state")
                    current_other_profile['emotional_state'] = 'negative'
                    current_other_profile['recommendation_type'] = 'stress_management'
                    current_other_profile.pop('needs_emotional_clarification', None)
                    current_other_profile.pop('clarification_attempts', None)
                    return 4, current_self_ok, current_other_profile, current_step_2_substep
                else:
                    # Pas de choix clair - demander clarification avec le prompt spécialisé
                    logger.info(f"👥 Step 3.2: No clear emotional state choice detected - requesting clarification (attempt {clarification_attempts}/2)")
                    current_other_profile['needs_emotional_clarification'] = True
                    return 3, current_self_ok, current_other_profile, current_step_2_substep
        # Vérifier si on a assez d'informations sur la BASE du collègue
        colleague_profile = current_other_profile.copy()
        
        # Si pas encore de BASE identifiée, détecter le choix utilisateur
        if not colleague_profile.get('base_type'):
            logger.info(f"👥 Step 3.1: No BASE yet, detecting user choice from: '{user_query}'")
            # Détecter si l'utilisateur a choisi une BASE (A, B, C, D, E, F)
            user_choice = _detect_base_choice(user_query)
            logger.info(f"👥 Step 3.1: Detection result: '{user_choice}'")
            
            if user_choice:
                # Utilisateur a fait un choix - enregistrer la BASE
                base_mapping = {
                    'A': 'Thinker',
                    'B': 'Persister', 
                    'C': 'Harmonizer',
                    'D': 'Rebel',
                    'E': 'Imaginer',
                    'F': 'Promoter'
                }
                
                selected_base = base_mapping.get(user_choice.upper())
                if selected_base:
                    colleague_profile['base_type'] = selected_base
                    colleague_profile['base_confidence'] = 'user_selected'
                    colleague_profile['base_reasoning'] = f"User selected {selected_base} from the 6 PCM BASE options"
                    logger.info(f"👥 Step 3.1: User selected BASE: {selected_base}")
                    
                    # DEBUG: Confirm base_type is correctly set
                    logger.info(f"🔍 BASE SELECTION DEBUG: colleague_profile['base_type'] = '{colleague_profile.get('base_type')}'")
                    logger.info(f"🔍 BASE SELECTION DEBUG: current_other_profile['base_type'] = '{current_other_profile.get('base_type')}'")
                    
                    # Update current_other_profile to match colleague_profile
                    current_other_profile.update(colleague_profile)
                    
                    # Passer à Step 3.1 (sélection PHASE)
                    colleague_profile['base_confirmed'] = True
                    logger.info(f"👥 Step 3.0→3.1: BASE confirmed as {selected_base}, moving to PHASE selection")
                    return 3, current_self_ok, colleague_profile, current_step_2_substep
                else:
                    # Choix invalide - continuer en Step 3.1
                    logger.info(f"👥 Step 3.1: Invalid choice '{user_choice}' - staying in selection")
                    return 3, current_self_ok, colleague_profile, current_step_2_substep
            else:
                # Pas de choix clair - continuer à présenter les options
                logger.info("👥 Step 3.1: No clear BASE choice detected - presenting options")
                return 3, current_self_ok, colleague_profile, current_step_2_substep



    
    # Step 4: Recommandations finales
    else:
        logger.info("👥 Step 4: Providing final adaptation recommendations")
        return 4, current_self_ok, current_other_profile, current_step_2_substep

def pcm_analysis(state: WorkflowState) -> Dict[str, Any]:
    """
    Analyse approfondie PCM basée sur le profil utilisateur
    """
    logger.info("🔍 Starting PCM Deep Analysis")
    
    # Récupération des données du state
    pcm_base = state.get('pcm_base', '')
    pcm_phase = state.get('pcm_phase', '')
    flow_type = state.get('flow_type', 'general_knowledge')
    language = state.get('language', 'en')
    messages = state['messages']
    
    if messages:
        last_message = messages[-1]
        if hasattr(last_message, 'content'):
            user_query = last_message.content
        elif isinstance(last_message, dict):
            user_query = last_message.get('content', '')
        else:
            user_query = str(last_message)
    else:
        user_query = ""
    
    logger.info(f"📊 User PCM Base: {pcm_base}, Phase: {pcm_phase}")
    logger.info(f"📝 Flow type: {flow_type}, Language: {language}")
    logger.info(f"📝 Query: {user_query}")
    
    pcm_resources = state.get('pcm_resources', 'No specific PCM resources found')
    
    # Sélection du prompt selon le flow type
    if flow_type == 'self_focused':
        # Pour self_focused, on peut utiliser soit base soit phase selon la question
        # On peut analyser la question pour déterminer si elle concerne plus la base ou la phase
        if 'phase' in user_query.lower() or 'evolution' in user_query.lower() or 'change' in user_query.lower():
            analysis_prompt = build_pcm_self_focused_phase_prompt(state)
        elif 'base' in user_query.lower() or 'foundation' in user_query.lower() or 'core' in user_query.lower():
            analysis_prompt = build_pcm_self_focused_base_prompt(state)
        else:
            # Par défaut, on utilise le prompt de base
            analysis_prompt = build_pcm_self_focused_base_prompt(state)
    elif flow_type == 'coworker_focused':
        # Check current coworker step to determine prompt
        coworker_step = state.get('coworker_step', 1)
        coworker_step_2_substep = state.get('coworker_step_2_substep', 1)
        
        if coworker_step == 2:  # Step 2 uses ACTION_PLAN prompt
            # Check if we've already had Step 2.1 conversations
            conversation_context = _build_conversation_context_coworker(state.get('messages', []))
            has_previous_education = _has_previous_step21_education(conversation_context)
            
            logger.info(f"👥 PROMPT SELECTION: Step 2.{coworker_step_2_substep}, has_previous_education={has_previous_education}")
            
            # Pass the education info to the prompt builder
            analysis_prompt = build_pcm_coworker_focused_action_plan_prompt(
                state, 
                coworker_step_2_substep, 
                has_previous_education=has_previous_education
            )
        else:  # Step 1, 3, 4 use regular coworker prompt
            analysis_prompt = build_pcm_coworker_focused_prompt(state)
    else:  # general_knowledge
        analysis_prompt = build_pcm_general_knowledge_prompt(state)
    
    try:
        pcm_analysis_result = isolated_analysis_call_with_messages(
            system_content=analysis_prompt,
            user_content=f"Please analyze this PCM question: {user_query}"
        )
        
        logger.info(f"✅ PCM analysis completed: {len(pcm_analysis_result)} chars")
        
    except Exception as e:
        logger.error(f"❌ PCM analysis failed: {e}")
        pcm_analysis_result = f"I understand you're asking about PCM. While I encountered an issue with detailed analysis, I can tell you that PCM focuses on understanding personality types and communication patterns. Please contact jean-pierre.aerts@zestforleaders.com for expert PCM guidance."
    
    return {
        **state,
        'pcm_analysis_result': pcm_analysis_result,
        'analysis_complete': True
    }

def fetch_pcm_profile(state: WorkflowState) -> WorkflowState:
    """
    Fetch user's PCM profile from database
    """
    logger.info("🔍 Fetching user PCM profile...")
    
    user_email = state.get("user_email", "").strip()
    user_name = state.get("user_name", "").strip()
    
    if not user_email and not user_name:
        logger.info("❌ No user_email or user_name provided")
        return {**state, "user_pcm": None, "user_pcm_base": None, "user_pcm_phase": None}
    
    try:
        # Similar to MBTI profile fetching but for PCM
        # This would need to be implemented based on your PCM profile table structure
        # For now, placeholder implementation
        
        logger.info("⚠️ PCM profile fetching not yet implemented")
        return {**state, "user_pcm": None, "user_pcm_base": None, "user_pcm_phase": None}
        
    except Exception as e:
        logger.error(f"❌ Error fetching PCM profile: {e}")
        return {**state, "user_pcm": None, "user_pcm_base": None, "user_pcm_phase": None}


def _assess_colleague_base_with_llm(conversation_context: str, enhanced_context: str = "") -> Dict[str, Any]:
    """
    Use LLM to assess colleague's PCM BASE type with systematic funnel approach.
    Requires evidence across multiple dimensions before identification.
    
    Returns:
    {'base_type': str, 'confidence': str, 'reasoning': str, 'contradictions': str}
    or {'base_type': None} if insufficient information
    """
    logger.info("🤖 Starting systematic LLM colleague BASE assessment")
    
    try:
        prompt = f"""
You are a PCM (Process Communication Model) expert analyzing behavioral observations to form HYPOTHESES about someone's BASE type.

CONVERSATION CONTEXT:
{conversation_context}

ENHANCED CONTEXT:
{enhanced_context}

**CRITICAL: SYSTEMATIC FUNNEL APPROACH REQUIRED**

**PHASE 1: COMPREHENSIVE EVIDENCE GATHERING**
You must have evidence across AT LEAST 4 of these 6 dimensions before making ANY hypothesis:
1. **PERCEPTION** (How they process information)
2. **STRENGTHS** (What they excel at)  
3. **INTERACTION STYLE** (How they work with others)
4. **PERSONALITY** (Core characteristics)
5. **COMMUNICATION CHANNELS** (How they express themselves)
6. **ENVIRONMENT PREFERENCES** (Where they thrive)

**PHASE 2: PATTERN RECOGNITION ACROSS ALL 6 BASE TYPES**
Consider ALL 6 BASE types systematically - DO NOT jump to conclusions:

**THINKER:** Facts/data perception, problem-solving strengths, democratic interaction, logical personality, analytical communication, independent environment
**PERSISTER:** Values/principles perception, principled strengths, autocratic interaction, conviction-driven personality, values-based communication, structured environment  
**HARMONIZER:** Feelings/harmony perception, caring strengths, benevolent interaction, nurturing personality, warm communication, collaborative environment
**REBEL:** Fun/creativity perception, innovative strengths, laissez-faire interaction, spontaneous personality, playful communication, dynamic environment
**IMAGINER:** Possibilities/reflection perception, visionary strengths, quiet interaction, contemplative personality, reflective communication, private environment
**PROMOTER:** Action/results perception, execution strengths, autocratic interaction, directive personality, commanding communication, leadership environment

**PHASE 3: CONFIDENCE ASSESSMENT**
- **HIGH:** Strong consistent evidence across 5+ dimensions, clear dominant pattern
- **MEDIUM:** Good evidence across 4 dimensions, some pattern but needs validation  
- **LOW:** Insufficient evidence (<4 dimensions) or contradictory signals

**HANDLING CONTRADICTIONS:**
- Look for dominant patterns across multiple dimensions
- Consider natural vs. learned behaviors  
- Note if behaviors seem situational or stress-related
- Lower confidence if truly mixed signals

**IMPORTANT REMINDERS:**
- These are BEHAVIORAL OBSERVATIONS forming HYPOTHESES
- Only official PCM assessment can determine actual BASE type
- Use careful language: "appears to", "suggests", "indicates"

RESPONSE FORMAT (JSON only):
{{
  "base_type": "THINKER/PERSISTER/HARMONIZER/REBEL/IMAGINER/PROMOTER or null",
  "confidence": "high/medium/low", 
  "reasoning": "Systematic analysis across dimensions with specific evidence",
  "contradictions": "Any conflicting behaviors noted",
  "dimensions_analyzed": ["dimension1", "dimension2", ...],
  "evidence_strength": "Strength of evidence for each dimension"
}}

RESPOND ONLY WITH JSON - NO OTHER TEXT."""

        response = isolated_analysis_call_with_messages(
            system_content="You are a PCM expert. Analyze behavioral observations systematically using the funnel approach. Respond only with JSON.",
            user_content=prompt,
            model="gpt-4"
        )
        
        logger.info(f"🤖 LLM BASE assessment response: {response}")
        
        # Parse JSON response
        import json
        try:
            result = json.loads(response.strip())
            
            # Validate required fields
            if 'base_type' not in result:
                logger.error("❌ LLM response missing base_type field")
                return {'base_type': None}
            
            # Ensure base_type is properly formatted or None
            base_type = result.get('base_type')
            if base_type and base_type.lower() == 'null':
                base_type = None
            
            return {
                'base_type': base_type,
                'confidence': result.get('confidence', 'low'),
                'reasoning': result.get('reasoning', 'LLM analysis of behavioral patterns'),
                'contradictions': result.get('contradictions', ''),
                'dimensions_analyzed': result.get('dimensions_analyzed', []),
                'evidence_strength': result.get('evidence_strength', '')
            }
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ Failed to parse LLM response as JSON: {e}")
            logger.error(f"Raw response: {response}")
            return {'base_type': None}
            
    except Exception as e:
        logger.error(f"❌ Error in LLM colleague BASE assessment: {e}")
        return {'base_type': None}


def update_explored_dimensions(state: WorkflowState) -> Dict[str, Any]:
    """
    Update the list of explored BASE dimensions based on current interaction
    """
    current_dimensions = state.get('pcm_specific_dimensions')
    explored_dimensions = state.get('pcm_explored_dimensions', [])
    
    # Map technical names to display names for tracking
    dimension_mapping = {
        "perception": "Perception",
        "strengths": "Strengths", 
        "interaction_style": "Interaction Style",
        "personality": "Personality",
        "communication_channels": "Communication Channels",
        "environment_preferences": "Environment Preferences"
    }
    
    if current_dimensions:
        for dimension in current_dimensions:
            if dimension in dimension_mapping:
                dimension_name = dimension_mapping[dimension]
                if dimension_name not in explored_dimensions:
                    explored_dimensions = explored_dimensions + [dimension_name]
                    logger.info(f"📋 Added {dimension_name} to explored dimensions: {explored_dimensions}")
    
    return {
        **state,
        'pcm_explored_dimensions': explored_dimensions
    }

# ============================================================================
# HELPER FUNCTIONS for Intelligent Hybrid PCM System
# ============================================================================

def _handle_comparison_in_conversational_flow(state: WorkflowState, classification: Dict[str, Any]) -> Dict[str, Any]:
    """
    Gère les comparaisons dans un contexte conversationnel
    Le meilleur des deux mondes : comparaison intelligente + suivi conversationnel
    """
    logger.info("🔄 Handling COMPARISON in conversational context")
    
    # Marquer que c'est une comparaison mais garder le contexte conversationnel
    return {
        **state,
        'flow_type': 'self_focused',  # Garder le contexte conversationnel
        'pcm_base_or_phase': 'base',  # On reste en BASE exploration
        'pcm_comparison_request': True,  # Flag pour indiquer qu'il faut faire une comparaison
        'pcm_comparison_target': 'promoter',  # Extracté de la requête
        'pcm_flow_classification': classification,
        'pcm_specific_dimensions': None  # Laisser l'analyse conversationnelle décider
    }

def _handle_self_focused_with_context(state: WorkflowState, classification: Dict[str, Any]) -> Dict[str, Any]:
    """
    Gère les flux self_focused avec le contexte de classification
    Inclut la logique de première interaction avec présentation des 6 dimensions
    """
    logger.info("🎯 Handling self_focused with intelligent context")
    
    # ÉTAPE 1: Vérifier s'il faut faire une première interaction  
    explored_dimensions = state.get('pcm_explored_dimensions', [])
    pcm_context = state.get('pcm_base_or_phase', 'base')
    specific_dimensions = state.get('pcm_specific_dimensions')
    is_first_pcm_interaction = len(explored_dimensions) == 0 and len(state.get('messages', [])) <= 2
    
    logger.info(f"🔍 First interaction check: explored={len(explored_dimensions)}, messages={len(state.get('messages', []))}, is_first={is_first_pcm_interaction}")
    
    # Cas spécial: première interaction BASE sans dimensions spécifiques
    if is_first_pcm_interaction and pcm_context == 'base' and not specific_dimensions:
        logger.info("🆕 First BASE interaction - presenting overview of 6 dimensions")
        try:
            from .pcm_analysis_new import _handle_first_base_interaction
            return _handle_first_base_interaction(state)
        except ImportError:
            logger.warning("⚠️ First interaction handler not available, using conversational")
    
    # Cas spécial: première interaction PHASE
    elif is_first_pcm_interaction and pcm_context == 'phase':
        logger.info("🆕 First PHASE interaction - explaining stress concept")
        try:
            from .old.pcm_analysis_new import _handle_first_phase_interaction
            return _handle_first_phase_interaction(state)
        except ImportError:
            logger.warning("⚠️ First interaction handler not available, using conversational")
    
    # ÉTAPE 2: Cas normal - utiliser l'analyse conversationnelle
    try:
        from .pcm_conversational_analysis import analyze_pcm_conversational_intent
        result_state = analyze_pcm_conversational_intent(state)
        
        # Enrichir avec la classification intelligente
        final_state = {
            **state,
            **result_state,
            'pcm_flow_classification': classification,
            'conversational_analysis_complete': True
        }
        
        logger.info("✅ Self-focused conversational analysis complete")
        return final_state
        
    except ImportError:
        logger.warning("⚠️ Conversational analysis not available")
        return _fallback_to_legacy_analysis(state)

def _continue_coworker_legacy_analysis(state: WorkflowState) -> Dict[str, Any]:
    """
    Continue le flux coworker existant (garde l'ancien système)
    """
    logger.info("🔒 Continuing coworker flow with legacy analysis")
    return _fallback_to_legacy_analysis(state)

def _handle_new_coworker_flow(state: WorkflowState, classification: Dict[str, Any]) -> Dict[str, Any]:
    """
    Démarre un nouveau flux coworker
    """
    logger.info("👥 Starting new coworker flow")
    return {
        **state,
        'flow_type': 'coworker_focused',
        'coworker_step': 1,
        'pcm_flow_classification': classification,
        'pcm_analysis_done': True
    }

def _handle_direct_flow_processing(state: WorkflowState, classification: Dict[str, Any]) -> Dict[str, Any]:
    """
    Gère les flux directs (TEAM, EXPLORATION, etc.)
    """
    flow_type = classification.get('flow_type')
    logger.info(f"🎯 Processing direct flow: {flow_type}")
    
    return {
        **state,
        'flow_type': flow_type.lower(),
        'pcm_flow_classification': classification,
        'pcm_analysis_done': True
    }

def _fallback_to_legacy_analysis(state: WorkflowState) -> Dict[str, Any]:
    """
    Fallback vers l'ancien système d'analyse
    """
    logger.info("🔄 Using legacy analysis fallback")
    
    # Détection simple pour continuer à fonctionner
    messages = state.get('messages', [])
    if messages:
        last_message = messages[-1]
        if hasattr(last_message, 'content'):
            user_query = last_message.content
        elif isinstance(last_message, dict):
            user_query = last_message.get('content', '')
        else:
            user_query = str(last_message)
    else:
        user_query = ""
    
    # Simple fallback logic
    query_lower = user_query.lower()
    if any(word in query_lower for word in ['my', 'me', 'i', 'je', 'mon', 'ma']):
        return {
            **state,
            'flow_type': 'self_focused',
            'pcm_base_or_phase': 'base',
            'pcm_analysis_done': True
        }
    else:
        return {
            **state,
            'flow_type': 'general_pcm',
            'pcm_analysis_done': True
        }

def _handle_comparison_search(state: WorkflowState) -> Dict[str, Any]:
    """Recherche comparative entre types PCM - recherche pour chaque type à comparer"""
    logger.info("🔍 Searching PCM type comparisons")
    
    # Base filters for PCM documents (comme dans coworker)
    base_filters = {
        'theme': 'A_UnderstandingMyselfAndOthers',
        'sub_theme': 'A2_PersonalityPCM', 
        'language': state.get('language', 'en')
    }
    
    # Récupérer les types à comparer
    types_to_compare = state.get('pcm_types_to_compare', [])
    user_base = state.get('pcm_base', '').lower()
    
    logger.info(f"🔍 Comparison: User base '{user_base}' vs extracted types: {types_to_compare}")
    logger.info(f"🔍 DEBUG: State keys related to PCM: {[k for k in state.keys() if 'pcm' in k.lower()]}")
    
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
    
    # NOUVEAU: Garder les types de la conversation précédente si on est dans un contexte de comparaison
    previous_comparison_types = state.get('pcm_comparison_types', [])
    if not comparison_types and previous_comparison_types:
        # Si on n'a trouvé aucun type mais qu'on était déjà en comparaison, continuer avec les types précédents
        logger.info(f"🔄 No new types found, continuing with previous comparison types: {previous_comparison_types}")
        comparison_types = previous_comparison_types
    
    # 🔍 DEBUG: Voir l'état avant et après continuité
    logger.info(f"🔍 DEBUG: comparison_types before continuity: {comparison_types}")
    logger.info(f"🔍 DEBUG: previous_comparison_types from state: {previous_comparison_types}")
    logger.info(f"🔍 DEBUG: comparison_types after continuity: {comparison_types}")
    
    logger.info(f"🎯 Final comparison types: {comparison_types}")
    
    # Faire une recherche BASE + PHASE pour chaque type (inspiré de coworker step 2.1)
    from ..lencioni.lencioni_analysis import perform_supabase_vector_search, sanitize_vector_results
    comparison_results = {}
    
    # Get the user query for search (used in the loop)
    user_query = state.get('user_message', '')
    
    for pcm_type in comparison_types:
        logger.info(f"🔍 Searching BASE + PHASE for PCM type: {pcm_type}")
        
        type_results = []
        
        # 1. Recherche BASE (toutes les 6 dimensions)
        base_filters_type = base_filters.copy()
        base_filters_type['pcm_base_type'] = pcm_type.lower()
        
        base_results = perform_supabase_vector_search(
            query=user_query,
            match_function='match_documents',
            metadata_filters=base_filters_type,
            limit=12  # Toutes les dimensions BASE
        )
        type_results.extend(base_results)
        logger.info(f"📊 Type {pcm_type} BASE: {len(base_results)} results")
        
        # 2. Recherche PHASE (3 sections: needs, negative_satisfaction, distress_sequence)
        phase_filters_type = base_filters.copy()
        phase_filters_type['pcm_phase_type'] = pcm_type.lower()
        phase_section_types = ['psychological_needs', 'negative_satisfaction', 'distress_sequence', 'action_plan']
        
        for section_type in phase_section_types:
            phase_filters_section = phase_filters_type.copy()
            phase_filters_section['section_type'] = section_type
            
            phase_section_results = perform_supabase_vector_search(
                query=user_query,
                match_function='match_documents',
                metadata_filters=phase_filters_section,
                limit=3  # Par section
            )
            type_results.extend(phase_section_results)
        
        logger.info(f"📊 Type {pcm_type} TOTAL: {len(type_results)} results (BASE + PHASE)")
        comparison_results[pcm_type] = type_results
    
    # Créer les ressources formatées pour comparaison
    pcm_resources = _format_comparison_resources(comparison_results, comparison_types, state.get('language', 'en'), state.get('pcm_base', ''), state.get('pcm_phase', ''))
    
    # NOTE: L'historique conversationnel est géré dans le prompt, pas besoin de le dupliquer ici
    
    # Combiner tous les résultats dans l'état final avec structure spécialisée
    final_state = {
        **state,
        'pcm_comparison_results': comparison_results,
        'pcm_comparison_types': comparison_types,
        'comparison_types': comparison_types,  # Pour compatibilité avec prompt_builder
        'pcm_resources': pcm_resources,
        'vector_search_complete': True
    }
    
    total_results = sum(len(results) for results in comparison_results.values())
    logger.info(f"✅ Comparison search completed: {total_results} results for {len(comparison_types)} types")
    logger.info(f"📊 Results per type: {[(t, len(comparison_results.get(t, []))) for t in comparison_types]}")
    return final_state




# ============= FONCTIONS DE ROUTING BASÉES SUR LA CLASSIFICATION =============

def _handle_self_action_plan_search(state: WorkflowState) -> Dict[str, Any]:
    """Recherche spécifique pour SELF_ACTION_PLAN - 3 sections: BASE + PHASE (4 types) + ACTION_PLAN"""
    logger.info("🎯 PCM: SELF_ACTION_PLAN search - BASE foundation + PHASE context + ACTION_PLAN guidance")
    
    from ..lencioni.lencioni_analysis import perform_supabase_vector_search, sanitize_vector_results
    
    # Récupérer les infos utilisateur
    user_pcm_base = state.get("pcm_base", "").lower()
    user_pcm_phase = state.get("pcm_phase", "").lower()
    language = state.get("language", "en")
    messages = state.get("messages", [])
    
    if not messages:
        return {**state, "pcm_resources": "No messages provided"}
        
    # Extraire la requête utilisateur
    last_message = messages[-1]
    if hasattr(last_message, "content"):
        user_query = last_message.content
    else:
        user_query = last_message.get("content", "")
    
    # Filtres de base
    base_filters = {
        "theme": "A_UnderstandingMyselfAndOthers",
        "sub_theme": "A2_PersonalityPCM", 
        "language": state.get("language", "en")
    }
    
    all_results = []
    
    # 1. BASE Foundation (6 dimensions)
    if user_pcm_base:
        logger.info(f"🔍 1/3 BASE foundation search for: {user_pcm_base}")
        base_foundation_filters = base_filters.copy()
        base_foundation_filters["pcm_base_type"] = user_pcm_base
        
        base_results = perform_supabase_vector_search(
            query=user_query,
            match_function="match_documents",
            metadata_filters=base_foundation_filters,
            limit=3
        )
        all_results.extend(base_results)
        logger.info(f"✅ BASE results: {len(base_results)} results")
    
    # 2. PHASE Current State (4 section types)
    if user_pcm_phase:
        phase_section_types = ['psychological_needs', 'negative_satisfaction', 'distress_sequence', 'action_plan']
        logger.info(f"🔍 2/3 PHASE context search for {user_pcm_phase} - 4 sections")
        
        for section_type in phase_section_types:
            phase_filters = base_filters.copy()
            phase_filters['section_type'] = section_type
            phase_filters['pcm_phase_type'] = user_pcm_phase
            
            phase_results = perform_supabase_vector_search(
                query=user_query,
                match_function="match_documents",
                metadata_filters=phase_filters,
                limit=2
            )
            all_results.extend(phase_results)
            logger.info(f"✅ PHASE {section_type}: {len(phase_results)} results")
    
    # Formater directement sans sanitize (on veut garder tous les résultats des 3 sections)
    if all_results:
        pcm_resources = _format_action_plan_results_by_sections(all_results, "SELF_ACTION_PLAN", language)
        logger.info(f"✅ ACTION_PLAN formatting complete: 3 sections from {len(all_results)} results")
    else:
        pcm_resources = "No specific action plan guidance found for your profile."
    
    return {
        **state,
        "pcm_resources": pcm_resources,
        "pcm_phase_results": all_results,
        "vector_search_complete": True,
        "pcm_search_debug": f"ACTION_PLAN 3-section search: {len(all_results)} total results",
        "has_explored_action_plan": True,  # 🎯 Flag: utilisateur a exploré son ACTION_PLAN
        "previous_context": "action_plan"  # 🔄 Contexte précédent
    }

def _format_action_plan_results_simple(results: list) -> str:
    """Formatage simple pour les résultats action_plan"""
    if not results:
        return "No action plan guidance available."
    
    formatted_content = f"## PCM ACTION PLAN GUIDANCE ({len(results)} items)\n"
    
    for i, result in enumerate(results, 1):
        content = result.get("content", "No content available")[:800]
        formatted_content += f"### Action Plan Item {i}\n{content}\n\n"
    
    return formatted_content

def _handle_self_base_search(state: WorkflowState) -> Dict[str, Any]:
    """Recherche spécifique pour SELF_BASE - informations sur la base PCM de l'utilisateur"""
    logger.info("🎯 PCM: SELF_BASE search")
    
    from ..lencioni.lencioni_analysis import perform_supabase_vector_search, sanitize_vector_results
    
    # Récupérer les infos utilisateur
    user_pcm_base = state.get('pcm_base', '').lower()
    messages = state.get('messages', [])
    
    if not messages:
        return {**state, 'pcm_resources': 'No messages provided'}
        
    # Extraire la requête utilisateur
    last_message = messages[-1]
    if hasattr(last_message, 'content'):
        user_query = last_message.content
    else:
        user_query = last_message.get('content', '')
    
    # Filtres de base
    base_filters = {
        'theme': 'A_UnderstandingMyselfAndOthers',
        'sub_theme': 'A2_PersonalityPCM', 
        'language': state.get('language', 'en')
    }
    
    search_results = []
    
    if user_pcm_base:
        # Filter for BASE documents matching user's base type
        base_filters['pcm_base_type'] = user_pcm_base
        logger.info(f"🔍 Filtering for BASE documents matching user's base: {user_pcm_base}")
        
        # Execute single search with base filters
        search_results = perform_supabase_vector_search(
            query=user_query,
            match_function='match_documents',
            metadata_filters=base_filters,
            limit=8
        )
    else:
        logger.warning("⚠️ No user PCM base found")
    
    # Sanitize results with BASE-specific parameters
    min_sim = 0.05  # Lower threshold to capture all BASE dimensions
    sanitized_results = sanitize_vector_results(
        results=search_results,
        required_filters=None,
        top_k=8,  # Allow more results for BASE (6 dimensions + buffer)
        min_similarity=min_sim,
        max_chars_per_item=1500,
        max_total_chars=10000  # More space for all dimensions
    )
    
    # Format results
    pcm_resources = _format_base_results(sanitized_results, state.get('language', 'en'))
    
    # Tracking des dimensions explorées
    explored_dimensions = state.get('pcm_explored_dimensions', [])
    current_dimensions = state.get('pcm_specific_dimensions', [])
    
    # Ajouter les dimensions actuelles aux dimensions explorées avec mapping vers noms d'affichage
    dimension_mapping = {
        "perception": "Perception",
        "strengths": "Strengths", 
        "interaction_style": "Interaction Style",
        "personality_part": "Personality Parts",
        "channel_communication": "Channels of Communication",
        "environmental_preferences": "Environmental Preferences"
    }
    
    if current_dimensions:
        for dim in current_dimensions:
            if dim and dim in dimension_mapping:
                dimension_name = dimension_mapping[dim]
                if dimension_name not in explored_dimensions:
                    explored_dimensions.append(dimension_name)
                    logger.info(f"✅ Added dimension '{dim}' → '{dimension_name}' to explored list")
            elif dim and dim not in explored_dimensions:
                # Fallback pour dimensions non mappées
                explored_dimensions.append(dim)
                logger.info(f"✅ Added unmapped dimension '{dim}' to explored list")
    
    return {
        **state,
        'pcm_resources': pcm_resources,
        'pcm_base_results': sanitized_results,
        'pcm_explored_dimensions': explored_dimensions,
        'vector_search_complete': True,
        'pcm_search_debug': f"BASE search: {len(sanitized_results)} results, explored dimensions: {explored_dimensions}",
        'has_explored_base': True,  # 🏗️ Flag: utilisateur a exploré sa BASE
        'base_exploration_level': len(sanitized_results) if sanitized_results else 0,  # 📈 Niveau d'exploration
        'previous_context': 'base'  # 🔄 Contexte précédent
    }

def _handle_self_phase_search(state: WorkflowState) -> Dict[str, Any]:
    """Recherche spécifique pour SELF_PHASE - informations sur la phase PCM de l'utilisateur"""
    logger.info("🎯 PCM: SELF_PHASE search")
    
    from ..lencioni.lencioni_analysis import perform_supabase_vector_search, sanitize_vector_results
    
    # Récupérer les infos utilisateur
    user_pcm_phase = state.get('pcm_phase', '').lower().strip()
    messages = state.get('messages', [])
    
    if not messages:
        return {**state, 'pcm_resources': 'No messages provided'}
        
    # Extraire la requête utilisateur
    last_message = messages[-1]
    if hasattr(last_message, 'content'):
        user_query = last_message.content
    else:
        user_query = last_message.get('content', '')
    
    # Filtres de base
    base_filters = {
        'theme': 'A_UnderstandingMyselfAndOthers',
        'sub_theme': 'A2_PersonalityPCM', 
        'language': state.get('language', 'en')
    }
    
    all_results = []
    
    # PHASE recherche avec 3 sections: psychological_needs, negative_satisfaction, distress_sequence
    phase_section_types = ['psychological_needs', 'negative_satisfaction', 'distress_sequence']
    
    for section_type in phase_section_types:
        phase_filters = base_filters.copy()
        phase_filters['section_type'] = section_type
        
        if user_pcm_phase and user_pcm_phase != 'Non spécifié':
            phase_filters['pcm_phase_type'] = user_pcm_phase
            logger.info(f"🔍 PHASE search {section_type} for specific user phase: {user_pcm_phase}")
        else:
            logger.info(f"🔍 PHASE search {section_type} (general exploration)")
        
        section_results = perform_supabase_vector_search(
            query=user_query,
            match_function='match_documents',
            metadata_filters=phase_filters,
            limit=3  # Limit per section_type
        )
        all_results.extend(section_results)
    
    logger.info(f"🔍 SELF_PHASE 3-section search completed: {len(all_results)} total results")
    
    # Sanitize results with PHASE-specific parameters
    sanitized_results = sanitize_vector_results(
        results=all_results,
        required_filters=None,
        top_k=9,  # Allow 3 results per section (3 sections × 3)
        min_similarity=None,  # No similarity threshold for PHASE
        max_chars_per_item=1500,
        max_total_chars=12000  # More space for all 3 sections
    )
    
    # Format results
    pcm_resources = _format_phase_results(sanitized_results, state.get('language', 'en'))
    
    return {
        **state,
        'pcm_resources': pcm_resources,
        'pcm_phase_results': sanitized_results,
        'vector_search_complete': True,
        'pcm_search_debug': f"PHASE search: {len(sanitized_results)} results",
        'has_explored_phase': True,
        'previous_context': 'phase'
    }

def _handle_coworker_search(state: WorkflowState) -> Dict[str, Any]:
    """Recherche spécifique pour COWORKER - processus 3-step d'analyse relationnelle"""
    logger.info("👥 PCM: COWORKER search - 3-step relationship analysis")
    
    # 🎯 NOUVEAUTÉ: Déterminer le type de contexte COWORKER
    has_explored_base = state.get('has_explored_base', False)
    has_explored_phase = state.get('has_explored_phase', False)
    has_explored_action_plan = state.get('has_explored_action_plan', False)
    previous_context = state.get('previous_context')
    
    # Déterminer le type de contexte COWORKER
    if (has_explored_base or has_explored_phase or has_explored_action_plan or 
        previous_context in ['base', 'phase', 'action_plan']):
        coworker_context_type = 'contextual_base_phase'
        logger.info(f"🎯 COWORKER CONTEXTUAL detected - User has explored: BASE={has_explored_base}, PHASE={has_explored_phase}, ACTION_PLAN={has_explored_action_plan}, previous={previous_context}")
    elif (has_explored_action_plan  or previous_context in ['action_plan']):
        coworker_context_type = 'contextual_action_plan'
        logger.info(f"🎯 COWORKER CONTEXTUAL ACTION_PLAN detected - User has explored: ACTION_PLAN={has_explored_action_plan}, previous={previous_context}")
    else:
        coworker_context_type = 'direct'
        logger.info("🎯 COWORKER DIRECT detected - No prior self_focused exploration")
    
    from ..lencioni.lencioni_analysis import perform_supabase_vector_search, sanitize_vector_results
    
    # Get language and flow_type for formatting
    language = state.get('language', 'en')
    flow_type = state.get('flow_type', 'coworker_focused')
    
    # Get user query
    messages = state['messages']
    if messages:
        last_message = messages[-1]
        if hasattr(last_message, 'content'):
            user_query = last_message.content
        elif isinstance(last_message, dict):
            user_query = last_message.get('content', '')
        else:
            user_query = str(last_message)
    else:
        user_query = ""
    
    # Base filters for PCM documents
    base_filters = {
        'theme': 'A_UnderstandingMyselfAndOthers',
        'sub_theme': 'A2_PersonalityPCM', 
        'language': state.get('language', 'en')
    }
    
    # Initialize or get current coworker step - adapt based on context
    if coworker_context_type == 'contextual_action_plan':
        # User has explored BASE/PHASE/ACTION_PLAN - start at step 3 if new
        coworker_step = state.get('coworker_step', 3)  # Default to step 3 for contextual action plan
        coworker_step_2_substep = state.get('coworker_step_2_substep', 1)
        logger.info(f"🚀 CONTEXTUAL ACTION_PLAN - Step {coworker_step} (user already did self work)")
    elif coworker_context_type == 'contextual_base_phase':
        # User has explored BASE/PHASE - start at step 1 if new
        coworker_step = state.get('coworker_step', 1)  # Default to step 1 for contextual base/phase
        coworker_step_2_substep = state.get('coworker_step_2_substep', 1)
        logger.info(f"🚀 CONTEXTUAL BASE_PHASE - Step {coworker_step} (start with emotional assessment)")
    else:
        # Direct coworker - start at step 1 if new
        coworker_step = state.get('coworker_step', 1)  # Default to step 1
        coworker_step_2_substep = state.get('coworker_step_2_substep', 1)
        logger.info(f"🚀 DIRECT COWORKER - Step {coworker_step}")
    self_ok = state.get('coworker_self_ok', False)
    other_profile = state.get('coworker_other_profile', {})
    
    logger.info(f"👥 COWORKER_FOCUSED - Current step: {coworker_step}, Self OK: {self_ok}")
    logger.info(f"👥 DEBUG: Initial other_profile from state = {other_profile}")
    
    # Auto-progression logic based on user response analysis
    try:
        conversation_context = "\n".join([msg.get('content', '') if isinstance(msg, dict) else getattr(msg, 'content', str(msg)) for msg in messages[-3:]])
        coworker_step, self_ok, other_profile, coworker_step_2_substep = _analyze_coworker_progression(
            user_query, coworker_step, self_ok, other_profile, 
            conversation_context, state, coworker_step_2_substep,
            coworker_context_type=coworker_context_type
        )
        logger.info(f"👥 After progression analysis: Step={coworker_step}, Self OK={self_ok}, Substep={coworker_step_2_substep}")
        logger.info(f"👥 DEBUG: After progression other_profile = {other_profile}")
    except Exception as e:
        logger.warning(f"⚠️ Coworker progression analysis failed: {e}")
    
    # Execute search based on current step
    search_results = []
    
    if coworker_step == 1:
        # Step 1: Évaluation émotionnelle- pas besoin de recherche vectorielle ()
        user_pcm_base = state.get('pcm_base', '').lower() if state.get('pcm_base') else None
        search_results = []
        logger.info("👥 Step 1: No vector search needed, just emotional assessment")

    elif coworker_step == 2:
        # Step 2: Two substeps with different search logics 
        logger.info(f"👥 Step 2: Gestion du stress personnel - appel car emotionnel state identifié comme négatif ")
        
        # Get user profile for Step 2
        user_pcm_base = state.get('pcm_base', '').lower() if state.get('pcm_base') else None
        user_pcm_phase = state.get('pcm_phase', '').lower() if state.get('pcm_phase') else None
        
        search_results = []
        # Step 2.1: Education Base / phase de l'utilisateur pour comprendre ses réactions au stress (et sa base)
        if coworker_step_2_substep == 1:
            # Get ALL 6 BASE dimensions if user has a BASE
            logger.info(f"👥 Step 2.1: Get ALL 6 BASE dimensions if user has a BASE - user base: {user_pcm_base} ")
            
            if user_pcm_base and user_pcm_base != 'Non spécifié':
                base_filters_user_base = base_filters.copy()
                base_filters_user_base['pcm_base_type'] = user_pcm_base.lower()
                    
                user_base_search_results = perform_supabase_vector_search(
                    query=user_query,
                    match_function='match_documents',
                    metadata_filters=base_filters_user_base,
                    limit=12  # All dimensions for complete understanding
                    )
                search_results.extend(user_base_search_results)
                logger.info(f"👥 Step 2.1 BASE: {len(search_results)} comprehensive BASE results")
            
            # Get PHASE if user has a PHASE (3 sections: needs, negative_satisfaction, distress_sequence)
            logger.info(f"👥 Step 2.1: Get ALL 3 PHASE dimensions if user has a PHASE - user phase: {user_pcm_phase} ")
            
            if user_pcm_phase and user_pcm_phase != 'Non spécifié':
                base_filters_user_phase = base_filters.copy()
                base_filters_user_phase['pcm_phase_type'] = user_pcm_phase.lower()
                phase_section_types = ['psychological_needs', 'negative_satisfaction', 'distress_sequence']
                user_phase_results = []
                for section_type in phase_section_types:
                    user_phase_filters = base_filters_user_phase.copy()
                    user_phase_filters['section_type'] = section_type
                        
                    user_phase_section_results = perform_supabase_vector_search(
                        query=user_query,
                        match_function='match_documents',
                        metadata_filters=user_phase_filters,
                        limit=3  # Per section for complete understanding
                        )
                    user_phase_results.extend(user_phase_section_results)
                search_results.extend(user_phase_results)
                logger.info(f"👥 Step 2.1 PHASE: Complete psychological needs, negative satisfaction, distress sequence - {len(search_results)} comprehensive PHASE results")
                
                # IMPORTANT: Assigner les résultats aux variables globales pour le formatage
                pcm_base_results = user_base_search_results if 'user_base_search_results' in locals() else []
                pcm_phase_results = user_phase_results                    
                    
        else:
            # Step 2.2: Get ACTION_PLAN + BASE + PHASE for comprehensive guidance
            logger.info("👥 Step 2.2: Getting comprehensive ACTION_PLAN with BASE and PHASE context")
            
            # 1. Get ACTION_PLAN guidance for concrete steps
            action_plan_filters = base_filters.copy()
            action_plan_filters['pcm_phase_type'] = user_pcm_phase.lower()
            action_plan_filters['section_type'] = 'action_plan'
            
            action_plan_results = perform_supabase_vector_search(
                query=user_query + " workplace stress management action plan",
                match_function='match_documents',
                metadata_filters=action_plan_filters,
                limit=6  # Reduced to make room for BASE/PHASE
            )
            search_results.extend(action_plan_results)
            
            # 2. Get user BASE info to explain natural strengths 
            if user_pcm_base:
                base_filters_user = base_filters.copy()
                base_filters_user['pcm_base_type'] = user_pcm_base.lower()
                base_filters_user['section_type'] = 'strengths'
                
                base_results = perform_supabase_vector_search(
                    query="strengths personality foundation natural traits",
                    match_function='match_documents',
                    metadata_filters=base_filters_user,
                    limit=2
                )
                search_results.extend(base_results)
            
            # 3. Get user PHASE psychological needs info
            if user_pcm_phase:
                phase_filters_user = base_filters.copy()
                phase_filters_user['pcm_phase_type'] = user_pcm_phase.lower()
                phase_filters_user['section_type'] = 'psychological_needs'
                
                phase_results = perform_supabase_vector_search(
                    query="psychological needs motivational requirements current phase",
                    match_function='match_documents',
                    metadata_filters=phase_filters_user,
                    limit=2
                )
                search_results.extend(phase_results)
            
            logger.info(f"👥 Step 2.2: {len(action_plan_results)} ACTION_PLAN + {len(base_results) if user_pcm_base else 0} BASE + {len(phase_results) if user_pcm_phase else 0} PHASE = {len(search_results)} total results")
    
    elif coworker_step == 3:
        # Step 3: Explorer Base et puis phase du collègue
        user_pcm_base = state.get('pcm_base', '').lower() if state.get('pcm_base') else None
        other_pcm_base = other_profile.get('pcm_base', '').lower() if other_profile.get('pcm_base') else None
        
        if user_pcm_base and other_pcm_base:
            step3_filters = {
                **base_filters,
                'pcm_base': user_pcm_base,
                'section_type': 'adaptation'
            }
            search_results = perform_supabase_vector_search(
                query=f"{user_query} adaptation {other_pcm_base}",
                match_function='match_documents', 
                metadata_filters=step3_filters,
                limit=8
            )
    
    elif coworker_step == 4:
        # Step 4: Explorer Base et puis phase du collègue
        logger.info(f"👥 Step 4: Recherche base / phase / action plan du collègue et de l'utilisateur")
        
        # NOUVEAU: Détecter si l'utilisateur veut changer d'état émotionnel (LLM multilingue)
        emotional_change = _detect_emotional_state_change_with_llm(
            user_query, self_ok, other_profile.get('emotional_state', '')
        )
        if emotional_change:
            logger.info(f"🔄 EMOTIONAL STATE CHANGE detected: {emotional_change}")
            # Appliquer le changement et régénérer Step 4
            self_ok, other_profile = _apply_emotional_state_change(
                emotional_change, self_ok, other_profile
            )
            logger.info(f"🔄 Emotional states updated - regenerating Step 4 recommendations")
        
        all_results = []
 
        # Step 4: Final recommendations with ACTION PLANs for both user and colleague
        logger.info("👥 Step 4: Generating final recommendations with action plans")
            
        # Get colleague profile information from updated progression (not state!)
        colleague_profile = other_profile if other_profile else state.get('coworker_other_profile', {})
        colleague_base = colleague_profile.get('base_type')
        colleague_phase = colleague_profile.get('phase_state')
        # Get user profile information from updated progression (not state!)
        user_pcm_base = state.get('pcm_base', '').lower() if state.get('pcm_base') else None
        user_pcm_phase = state.get('pcm_phase', '').lower() if state.get('pcm_phase') else None
        user_phase = user_pcm_phase if user_pcm_phase and user_pcm_phase != 'Non spécifié' else None
        user_base = user_pcm_base if user_pcm_base and user_pcm_base != 'Non spécifié' else None
        # Get all phase section types
        phase_section_types = ['psychological_needs', 'negative_satisfaction', 'distress_sequence', 'action_plan']

        logger.info(f"👥 Step 4: User: BASE={user_base}, PHASE={user_phase}, Colleague: BASE={colleague_base}, PHASE={colleague_phase}")
            
        logger.info(f"👥 Step 4 DEBUG: colleague_base condition: {colleague_base} != 'Unknown' = {colleague_base != 'Unknown'}")
        logger.info(f"👥 Step 4 DEBUG: colleague_phase condition: {colleague_phase} != 'Unknown' = {colleague_phase != 'Unknown'}")
        logger.info(f"👥 Step 4 DEBUG: user_phase condition: {user_phase} != 'Unknown' = {user_phase != 'Unknown'}")
        logger.info(f"👥 Step 4 DEBUG: user_base condition: {user_base} != 'Unknown' = {user_base != 'Unknown'}")


        # If we have colleague's BASE, get BASE adaptation strategies
        if user_base and user_base != 'Unknown':
            # Get adaptation strategies for working with this colleague (base)
            user_base_filters = base_filters.copy()
            user_base_filters['pcm_base_type'] = user_base.lower()

            user_base_results = perform_supabase_vector_search(
                query=user_query + f" communicate with {user_base} workplace",
                match_function='match_documents',
                metadata_filters=user_base_filters,
                limit=12
                )
            all_results.extend(user_base_results)
            logger.info(f"👥 Step 4: {len(user_base_results)} user base results")
            
        # If we have colleague's PHASE, get PHASE adaptation strategies
        if user_phase and user_phase != 'Unknown':     
            user_phase_results = []
            for section_type in phase_section_types:
                user_phase_filters = base_filters.copy()
                user_phase_filters['pcm_phase_type'] = user_phase.lower()
                user_phase_filters['section_type'] = section_type
                
                user_phase_section_results = perform_supabase_vector_search(
                    query=user_query,
                    match_function='match_documents',
                    metadata_filters=user_phase_filters,
                    limit=3  # Per section for complete understanding
                )
                user_phase_results.extend(user_phase_section_results)

            all_results.extend(user_phase_results)
            logger.info(f"👥 Step 4: {len(user_phase_results)} user phase results")

        # If we have colleague's BASE, get BASE adaptation strategies
        if colleague_base and colleague_base != 'Unknown':
            # Get adaptation strategies for working with this colleague (base)
            colleague_base_filters = base_filters.copy()
            colleague_base_filters['pcm_base_type'] = colleague_base.lower()

            colleague_base_results = perform_supabase_vector_search(
                query=user_query + f" communicate with {colleague_base} workplace",
                match_function='match_documents',
                metadata_filters=colleague_base_filters,
                limit=12
                )
            all_results.extend(colleague_base_results)
            logger.info(f"👥 Step 4: {len(colleague_base_results)} colleague base results")
            
        # If we have colleague's PHASE, get PHASE adaptation strategies
        if colleague_phase and colleague_phase != 'Unknown':     
            colleague_phase_results = []
            for section_type in phase_section_types:
                colleague_phase_filters = base_filters.copy()
                colleague_phase_filters['pcm_phase_type'] = colleague_phase.lower()
                colleague_phase_filters['section_type'] = section_type
                
                colleague_phase_section_results = perform_supabase_vector_search(
                    query=user_query,
                    match_function='match_documents',
                    metadata_filters=colleague_phase_filters,
                    limit=3  # Per section for complete understanding
                )
                colleague_phase_results.extend(colleague_phase_section_results)

            all_results.extend(colleague_phase_results)
            logger.info(f"👥 Step 4: {len(colleague_phase_results)} colleague phase results")

        search_results = all_results
        #Créer un formattage structuré pour l'étape 4
        if search_results:
            pcm_resources_step4 = "# STEP 4: COMPREHENSIVE RELATIONSHIP ANALYSIS\n\n"
            user_base_section = [r for r in all_results if r.get('metadata', {}).get('pcm_base_type') == (user_base.lower() if user_base else None)]
            user_phase_section = [r for r in all_results if r.get('metadata', {}).get('pcm_phase_type') == (user_phase.lower() if user_phase else None)]
            colleague_base_section = [r for r in all_results if r.get('metadata', {}).get('pcm_base_type') == (colleague_base.lower() if colleague_base else None)]
            colleague_phase_section = [r for r in all_results if r.get('metadata', {}).get('pcm_phase_type') == (colleague_phase.lower() if colleague_phase else None)]

            if user_base_section:
                pcm_resources_step4 += f"## USER BASE: {user_base}\n\n"
                for i, result in enumerate(user_base_section[:6], 1):
                    content = result.get('content', 'No content available')
                    pcm_resources_step4 += f"### {i}. User Base Characteristic\n{content}\n\n"  
            if colleague_base_section:
                pcm_resources_step4 += f"## COLLEAGUE BASE: {colleague_base}\n\n"
                for i, result in enumerate(colleague_base_section[:6], 1):
                    content = result.get('content', 'No content available')
                    pcm_resources_step4 += f"### {i}. Colleague Base Characteristic\n{content}\n\n"
            
            if user_phase_section:
                pcm_resources_step4 += f"## USER PHASE: {user_phase}\n\n"
                pcm_resources_step4 += f"## ⚠️ USER Pshycological needs, distress sequence, action plan:  ({user_phase.title()} Phase)\n"
                for result in user_phase_section:
                    content = result.get("content", "")
                    section_type = result.get("metadata",{}).get("section_type", "")
                    pcm_resources_step4 += f"### {section_type.replace('_', ' ').title()}\n{content}\n\n"

            if colleague_phase_section:
                pcm_resources_step4 += f"## ⚠️ COLLEAGUE Pshycological needs, distress sequence, action plan:  ({colleague_phase.title()} Phase)\n"
                for result in colleague_phase_section:
                    content = result.get("content", "")
                    section_type = result.get("metadata",{}).get("section_type", "")
                    pcm_resources_step4 += f"### {section_type.replace('_', ' ').title()}\n{content}\n\n"
        else:
            pcm_resources_step4 = "No results found"
        logger.info(f"👥 Step 4:  Formatted pcm_resources with {len(pcm_resources_step4)} characters")


    # Sanitize results based on step
    if coworker_step == 2:
        # Step 2 (both 2.1 and 2.2): No sanitization to preserve all educational and action plan content
        sanitized_results = search_results
        logger.info(f"👥 Step 2 (substep {coworker_step_2_substep}): Using {len(sanitized_results)} raw results without sanitization")
    else:
        # Other steps: Apply standard sanitization
        sanitized_results = sanitize_vector_results(search_results)
    
    # Format as PCM resources with special formatting for step 2.1
    pcm_resources = ""
    if sanitized_results:
        if coworker_step == 2 and coworker_step_2_substep == 1:
            # Step 2.1: Special educational formatting (BASE + PHASE sections)
            logger.info(f"🎯 Step 2.1 FORMATTING: Using educational format for BASE + PHASE sections")
            logger.info(f"🔍 DEBUG FORMATTING: flow_type={flow_type}, coworker_step={coworker_step}, substep={coworker_step_2_substep}")
            pcm_resources = _format_coworker_step2_base_phase_results(
                sanitized_results,  # All results (base + phase)
                state,
                language
            )
            logger.info(f"🔍 DEBUG FORMATTING: pcm_resources formatted, length={len(pcm_resources)} chars, starts with: {pcm_resources[:100]}...")
        elif coworker_step == 2 and coworker_step_2_substep == 2:
            # Step 2.2: Special ACTION_PLAN formatting (3 sections)
            logger.info(f"🎯 Step 2.2 FORMATTING: Using ACTION_PLAN format for action plan content")
            logger.info(f"🔍 DEBUG FORMATTING: flow_type={flow_type}, coworker_step={coworker_step}, substep={coworker_step_2_substep}")
            pcm_resources = _format_action_plan_results_by_sections(
                sanitized_results,  # All results (action plan)
                flow_type,
                language
            )
            logger.info(f"🔍 DEBUG FORMATTING: ACTION_PLAN pcm_resources formatted, length={len(pcm_resources)} chars")
        else:
            # Regular formatting for other steps
            pcm_resources = "# Coworker Relationship Analysis\n\n"
            for i, result in enumerate(sanitized_results, 1):
                content = result.get("content", "No content available")[:800]
                pcm_resources += f"### Step {coworker_step} Resource {i}\n{content}\n\n"
    
    # DEBUG: Log what we're returning
    logger.info(f"🔄 RETURNING from _handle_coworker_search:")
    logger.info(f"  - coworker_step: {coworker_step}")
    logger.info(f"  - coworker_other_profile: {other_profile}")
    logger.info(f"  - coworker_other_profile.base_type: {other_profile.get('base_type')}")
    logger.info(f"🔍 DEBUG: coworker_step_1_attempts in state = {state.get('coworker_step_1_attempts', 'NOT FOUND')}")
    
    return {
        **state,
        'pcm_resources': pcm_resources_step4 if coworker_step == 4 else pcm_resources,
        'pcm_base_results': sanitized_results if coworker_step == 1 else [],
        'pcm_phase_results': sanitized_results if coworker_step > 1 else [],
        'vector_search_complete': True,
        'pcm_search_debug': f"COWORKER search step {coworker_step}: {len(sanitized_results)} results",
        # Update coworker state - PRESERVE the profile!
        'coworker_step': coworker_step,
        'coworker_self_ok': self_ok,
        'coworker_other_profile': other_profile,  # This contains the BASE selection!
        'coworker_step_2_substep': coworker_step_2_substep,
        'coworker_step_1_attempts': state.get('coworker_step_1_attempts', 0),  # CRITICAL: Get the UPDATED value after _analyze_coworker_progression
        'coworker_context_type': coworker_context_type,
        'flow_type': 'coworker_focused'  # 🎯 CORRECTION: S'assurer que le flow_type reste COWORKER
    }

def _extract_pcm_types_from_message(state: WorkflowState) -> List[str]:
    """Extrait intelligemment les types PCM mentionnés dans le message utilisateur"""
    messages = state.get('messages', [])
    if not messages:
        return []
    
    # Obtenir le dernier message
    last_message = messages[-1]
    if hasattr(last_message, 'content'):
        user_query = last_message.content
    else:
        user_query = last_message.get('content', '')
    
    # Contexte de comparaison précédente
    previous_comparison_types = state.get('pcm_comparison_types', [])
    
    # 🔍 DEBUG: Vérifier le contenu exact des types précédents
    logger.info(f"🔍 DEBUG EXTRACTION - previous_comparison_types: {previous_comparison_types}")
    logger.info(f"🔍 DEBUG EXTRACTION - user_query: '{user_query}'")
    
    # Construire l'historique conversationnel
    conversation_history = ""
    if len(messages) > 1:
        recent_messages = messages[-4:-1]  # Les 3 derniers messages avant le message actuel
        for msg in recent_messages:
            if hasattr(msg, 'content'):
                role = "User" if hasattr(msg, 'type') and msg.type == "human" else "Assistant"
                content = msg.content[:200]  # Limiter la taille
            elif isinstance(msg, dict):
                role = "User" if msg.get('type') == "human" else "Assistant"
                content = msg.get('content', '')[:200]
            else:
                continue
            conversation_history += f"\n{role}: {content}"
    
    # Prompt pour extraction intelligente
    system_prompt = f"""Tu es un expert PCM qui extrait les types PCM mentionnés dans un message utilisateur.

🚨 **PRIORITÉ ABSOLUE - CONTINUITÉ CONVERSATIONNELLE:**
- **RÈGLE #1**: Si l'utilisateur répond à une question de l'assistant (visible dans l'historique), c'est une CONTINUATION
- **RÈGLE #2**: Toute réponse courte ("leur base", "la phase", "oui", "non", "base", "phase") = CONTINUATION avec les types précédents
- **RÈGLE #3**: Si types dans previous_comparison_types ET le message est une réponse/continuation → utiliser ces types précédents
- **RÈGLE #4**: Phrases comme "plus de détails", "approfondis", "précise" = CONTINUATION évidente
- **RÈGLE #5**: Pronoms de références ("elles", "ils", "ces types", "them", "they", "those types") = CONTINUATION ÉVIDENTE avec types précédents
- **RÈGLE #6**: Questions sur stress/phase après discussion de types = CONTINUATION ÉVIDENTE ("comment réagissent elles", "how do they react", "leur stress", "their stress")
- **PRINCIPE**: En cas de doute sur la continuité → TOUJOURS utiliser les types précédents

HISTORIQUE CONVERSATIONNEL:{conversation_history}

CONTEXTE CRUCIAL:
Types de la conversation précédente: {previous_comparison_types}

TYPES PCM POSSIBLES:
- THINKER (analyseur)
- HARMONIZER (empathique) 
- PERSISTER (persévérant)
- REBEL (énergiseur)
- IMAGINER (imagineur)
- PROMOTER (promoteur)

RÈGLES D'EXTRACTION (TRÈS STRICTES):
1. **SCAN OBLIGATOIRE**: Chercher CHAQUE synonyme dans CHAQUE type
2. **CONTEXTE**: Utiliser l'historique pour comprendre les références ("plus de détails", "approfondis")
3. **PRÉCISION**: Si message contient "empathique ET analyseur" → extraire LES DEUX types
4. **ROBUSTESSE**: Gérer les typos et variantes
5. **EXCLUSIONS**: Ignorer "base", "phase" seuls (catégories, pas types)

SYNONYMES COMPLETS (SCAN TOUS CES MOTS):
• THINKER = "thinker", "analyseur", "analizer", "analyzeur", "analyze", "analyser", "logique", "logical", "analytique"
• HARMONIZER = "harmonizer", "empathique", "emphatique", "harmoniez", "harmoniser", "empathy", "empathic", "relationnel"  
• PERSISTER = "persister", "persévérant", "perseverant", "valeurs", "values", "convictions", "opinions"
• REBEL = "rebel", "énergiseur", "energiseur", "energizer", "créatif", "creative", "fun", "ludique", "spontané"
• IMAGINER = "imaginer", "imagineur", "imaginair", "calme", "calm", "réflexion", "imagination"
• PROMOTER = "promoteur", "promoter", "promotor", "action", "résultats", "results", "efficace"

EXEMPLES OBLIGATOIRES À SUIVRE:
- "empathique et analyseur" → ["harmonizer", "thinker"] (LES DEUX!)
- "analyseur vs promoteur" → ["thinker", "promoter"] (LES DEUX!)
- "différence entre harmonizer et rebel" → ["harmonizer", "rebel"]
- "plus de détails sur le promoteur" → ["promoter"]

MESSAGE UTILISATEUR: "{user_query}"

Réponds UNIQUEMENT avec un JSON contenant la liste des types (noms anglais):
{{"extracted_types": ["type1", "type2"]}}"""

    try:
        from ..common.llm_utils import isolated_analysis_call_with_messages
        
        response = isolated_analysis_call_with_messages(
            system_content=system_prompt,
            user_content="Extrais les types PCM de ce message."
        )
        
        # Nettoyer et parser la réponse
        response = response.strip()
        if '```json' in response:
            start = response.find('```json') + 7
            end = response.find('```', start)
            if end != -1:
                response = response[start:end].strip()
        
        import json
        result = json.loads(response)
        extracted_types = result.get('extracted_types', [])
        
        logger.info(f"🧠 Smart extraction from '{user_query}': {extracted_types}")
        return extracted_types
        
    except Exception as e:
        logger.error(f"❌ Smart extraction failed: {e}, using fallback")
        # Fallback simple
        pcm_types = ['thinker', 'harmonizer', 'persister', 'rebel', 'imaginer', 'promoter', 'analyseur', 'empathique', 'énergiseur', 'promoteur']
        extracted = [t for t in pcm_types if t in user_query.lower()]
        return extracted

def _format_comparison_resources(comparison_results: Dict[str, List], comparison_types: List[str], language: str, user_base: str = "", user_phase: str = "") -> str:
    """Formate les résultats de comparaison entre types PCM avec séparation BASE + PHASE"""
    if not comparison_results:
        return "No comparison results available."
    
    # Dimensions et sections pour classification (comme dans coworker step 2.1)
    base_dimensions = ['perception', 'strengths', 'interaction_style', 'personality_part', 'channel_communication', 'environmental_preferences']
    phase_sections = ['psychological_needs', 'negative_satisfaction', 'distress_sequence']
    
    formatted_sections = []
    
    # Titre selon la langue - sans instructions (gérées dans le prompt)
    if language == 'fr':
        title = "📊 DONNÉES COMPARAISON PCM"
    else:
        title = "📊 PCM COMPARISON DATA"
    
    formatted_sections.append(f"{title}\n{'='*60}")
    
    # Ajouter les types comparés pour que le prompt final les connaisse
    if language == 'fr':
        formatted_sections.append(f"🎯 TYPES À COMPARER: {', '.join([t.upper() for t in comparison_types])}")
    else:
        formatted_sections.append(f"🎯 TYPES TO COMPARE: {', '.join([t.upper() for t in comparison_types])}")
    formatted_sections.append("")
    
    # SECTION 1: BASES (types mentionnés + utilisateur si différent)
    if language == 'fr':
        formatted_sections.append("## 1️⃣ SECTION BASES (Personnalités naturelles)")
    else:
        formatted_sections.append("## 1️⃣ BASES SECTION (Natural personalities)")
    
    # Créer liste unique des types pour BASES (types mentionnés + user si différent)
    all_base_types = set(comparison_types)
    if user_base and user_base.lower() not in [t.lower() for t in comparison_types]:
        all_base_types.add(user_base.lower())
    
    base_types_found = 0
    for pcm_type in sorted(all_base_types):
        results = comparison_results.get(pcm_type, [])
        base_results = [r for r in results if r.get('metadata', {}).get('section_type', '') in base_dimensions]
        
        if base_results:
            base_types_found += 1
            type_label = f"{pcm_type.upper()}"
            if user_base and pcm_type.lower() == user_base.lower():
                type_label += " (votre base)" if language == 'fr' else " (your base)"
            
            formatted_sections.append(f"\n🔹 **{type_label}**")
            for result in base_results:
                content = result.get('content', '')
                section = result.get('metadata', {}).get('section_type', 'General')
                formatted_sections.append(f"  • **{section}**: {content}")
    
    # Si aucune donnée BASE trouvée
    if base_types_found == 0:
        if language == 'fr':
            formatted_sections.append("\n⚠️ Aucune donnée BASE disponible pour ces types")
        else:
            formatted_sections.append("\n⚠️ No BASE data available for these types")
    
    # SECTION 2: PHASES (types mentionnés + utilisateur si différent)
    if language == 'fr':
        formatted_sections.append(f"\n\n## 2️⃣ SECTION PHASES (Sous stress)")
    else:
        formatted_sections.append(f"\n\n## 2️⃣ PHASES SECTION (Under stress)")
    
    # Créer liste unique des types pour PHASES (types mentionnés + user phase si différent)
    all_phase_types = set(comparison_types)
    if user_phase and user_phase.lower() not in [t.lower() for t in comparison_types]:
        all_phase_types.add(user_phase.lower())
    
    phase_types_found = 0
    for pcm_type in sorted(all_phase_types):
        results = comparison_results.get(pcm_type, [])
        phase_results = [r for r in results if r.get('metadata', {}).get('section_type', '') in phase_sections]
        
        if phase_results:
            phase_types_found += 1
            type_label = f"{pcm_type.upper()} stress"
            if user_phase and pcm_type.lower() == user_phase.lower():
                type_label += " (votre phase actuelle)" if language == 'fr' else " (your current phase)"
            
            formatted_sections.append(f"\n🔸 **{type_label}**")
            for result in phase_results:
                content = result.get('content', '')
                section = result.get('metadata', {}).get('section_type', 'General')
                formatted_sections.append(f"  • **{section}**: {content}")
    
    # Si aucune donnée PHASE trouvée
    if phase_types_found == 0:
        if language == 'fr':
            formatted_sections.append("\n⚠️ Aucune donnée PHASE disponible pour ces types")
        else:
            formatted_sections.append("\n⚠️ No PHASE data available for these types")
    
    # Ajouter une section de synthèse comparative
    if len(comparison_types) >= 2:
        if language == 'fr':
            formatted_sections.append("\n\n💡 POINTS CLÉS DE COMPARAISON\n" + "="*30)
            formatted_sections.append("• Chaque type a des forces et des modes de fonctionnement uniques")
            formatted_sections.append("• La compréhension des différences aide à mieux collaborer")
            formatted_sections.append("• L'adaptation de votre communication selon le type améliore les relations")
        else:
            formatted_sections.append("\n\n💡 KEY COMPARISON POINTS\n" + "="*30)
            formatted_sections.append("• Each type has unique strengths and operating modes")
            formatted_sections.append("• Understanding differences helps better collaboration")
            formatted_sections.append("• Adapting your communication to each type improves relationships")
    
    return "\n".join(formatted_sections)

def _handle_general_pcm_search(state: WorkflowState) -> Dict[str, Any]:
    """Recherche théorique PCM générale - seulement base filters et section_type=simple_content"""
    logger.info("📚 PCM: GENERAL_PCM search - theory and concepts")
    
    user_query = state.get('user_message', '')
    language = state.get('language', 'en')
    
    # Base filters + section_type=simple_content uniquement
    base_filters = {
        'theme': 'A_UnderstandingMyselfAndOthers',
        'sub_theme': 'A2_PersonalityPCM', 
        'section_type': 'simple_content' }
    
    try:
        # Recherche théorique générale PCM
        results = perform_supabase_vector_search(
            query=user_query,
            match_function='match_documents',
            metadata_filters=base_filters,
            limit=10        )
        
        pcm_resources = ""
        if results:
            logger.info(f"📚 Found {len(results)} general PCM theory results")
            for result in results:
                content = result.get('content', '')
                document_name = result.get('document_name', 'PCM Theory')
                similarity = result.get('similarity', 0.0)
                
                pcm_resources += f"\n**{document_name}** (relevance: {similarity:.2f}):\n{content}\n"
        else:
            logger.warning("📚 No general PCM theory results found")
            pcm_resources = "No specific PCM theory content found in knowledge base."
        
        return {
            **state,
            "pcm_resources": pcm_resources,
            "flow_type": "general_pcm",
            "vector_search_complete": True
        }
        
    except Exception as e:
        logger.error(f"❌ Error in general PCM search: {e}")
        return {
            **state,
            "pcm_resources": "Error retrieving PCM theory information.",
            "flow_type": "general_pcm",
            "vector_search_complete": True
        }

def _handle_greeting_search(state: WorkflowState) -> Dict[str, Any]:
    """Gestion des salutations - pas de recherche vectorielle"""
    logger.info("👋 PCM: GREETING - no search needed")
    return {
        **state,
        "pcm_resources": "",
        "skip_search": True,
        "vector_search_complete": True
    }

