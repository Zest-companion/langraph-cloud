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
from ..prompts.prompt_builder import (
    build_pcm_self_focused_base_prompt,
    build_pcm_self_focused_phase_prompt,
    build_pcm_general_knowledge_prompt,
    build_pcm_coworker_focused_prompt
)

logger = logging.getLogger(__name__)

# Mapping from intent analysis dimension names to Supabase section_type values
DIMENSION_MAPPING = {
    'perception': 'characteristics',
    'strengths': 'strengths',
    'interaction_style': 'interaction_style', 
    'personality_part': 'personality_part',
    'channel_communication': 'channel_communication',
    'environmental_preferences': 'environmental_preferences'
}


def pcm_intent_analysis(state: WorkflowState) -> Dict[str, Any]:
    """
    Analyzes user's intent for PCM questions using LLM
    Based on Process Communication Model framework
    """
    logger.info("🎯 STARTING PCM Intent Analysis - ENTRY POINT")
    logger.info(f"🔍 Current state keys: {list(state.keys())}")
    
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
   - **coworker_focused**: User wants to understand/improve interactions with OTHERS
   - **non_pcm**: User's message is NOT related to PCM (greetings, small talk, unrelated topics)

2. PCM CONTEXT (for self_focused queries):
   - **base**: Natural way of perceiving the world, stable patterns, fundamental identity (keywords: "my perception", "my strengths", "how I am", "my personality")
   - **phase**: Current motivational needs and stress responses, what drives them now (keywords: "what I need now", "my current motivation", "lately I", "recently")
   - **situational**: Contextual/situational questions about applying PCM in specific scenarios (keywords: "in this situation", "when dealing with", "how to handle", "adapt my approach")

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
- Detect language: 'fr' for French, 'en' for English (default to 'en' if unclear)

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
        
    # 1. Get classification from intent analysis
    flow_type = state.get('flow_type', 'general_knowledge')
    language = state.get('language', 'en')
    
    logger.info(f"🎯 Using flow type: {flow_type}, Language: {language}")
    
    # 2. Recherche vectorielle adaptée
    pcm_base_results = []
    pcm_phase_results = []
    pcm_general_results = []
    pcm_base_or_phase = None  # Track BASE/PHASE classification for visibility
    
    try:
        pcm_results, pcm_base_or_phase = _execute_pcm_search(user_query, flow_type, language, state)
        
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
        
        # 5. Formatter les résultats pour le prompt
        pcm_resources = _format_pcm_results_by_category(
            pcm_base_results, 
            pcm_phase_results, 
            pcm_general_results,
            flow_type, 
            language
        )
        
        logger.info(f"✅ PCM vector search completed:")
        logger.info(f"   📊 Base results: {len(pcm_base_results)}")
        logger.info(f"   🔄 Phase results: {len(pcm_phase_results)}")
        logger.info(f"   📚 General results: {len(pcm_general_results)}")
        
    except Exception as e:
        logger.warning(f"⚠️ PCM vector search failed: {str(e)}")
        logger.info("📚 Using PCM fallback")
        
        pcm_resources = _get_pcm_fallback(flow_type, language)
        pcm_base_or_phase = None  # No classification available in fallback case
    
    # Update explored dimensions if we have a specific dimension
    updated_state = update_explored_dimensions(state)
    
    return {
        **updated_state,
        'pcm_resources': pcm_resources,
        'pcm_base_results': pcm_base_results,  # Visible dans LangGraph Studio
        'pcm_phase_results': pcm_phase_results,  # Visible dans LangGraph Studio
        'pcm_general_results': pcm_general_results,  # Visible dans LangGraph Studio
        'pcm_base_or_phase': pcm_base_or_phase,  # BASE/PHASE classification visible in LangGraph Studio
        'flow_type': flow_type,
        'language': language,
        'pcm_search_debug': f"Base: {len(pcm_base_results)}, Phase: {len(pcm_phase_results)}, General: {len(pcm_general_results)}, Context: {pcm_base_or_phase or 'None'}",
        'vector_search_complete': True,
        # Explicitly preserve PCM profile from fetch_user_profile
        'pcm_base': state.get('pcm_base'),
        'pcm_phase': state.get('pcm_phase')
    }

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

def _execute_pcm_search(user_query: str, flow_type: str, language: str, state: WorkflowState) -> tuple[List[Dict], str]:
    """
    Execute PCM-specific vector search based on flow type (simplified to 3 options)
    Filters results based on user's PCM profile when appropriate
    """
    logger.info(f"🔍 Executing PCM search for flow_type: {flow_type}, language: {language}")
    
    # Base filters for PCM documents
    base_filters = {
        'theme': 'A_UnderstandingMyselfAndOthers',
        'sub_theme': 'A2_PersonalityPCM'
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
    # IMPORTANT: If user asks for specific BASE dimensions, always use BASE filtering
    if specific_dimensions_list and user_pcm_base:
        # User is asking for specific BASE dimensions - need to handle multiple dimensions
        # We'll do multiple searches and merge results
        logger.info(f"🔍 Filtering for BASE documents (multi-dimension): {user_pcm_base} -> {specific_dimensions_list}")
        
        all_results = []
        for dimension in specific_dimensions_list:
            # Map dimension names to database section_type values
            mapped_dimension = DIMENSION_MAPPING.get(dimension, dimension)
            
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
            
    elif pcm_base_or_phase == 'phase' and user_pcm_phase:
        # Filter for PHASE documents matching user's phase type  
        base_filters['pcm_phase_type'] = user_pcm_phase
        logger.info(f"🔍 Filtering for PHASE documents matching user's phase: {user_pcm_phase}")
        
        # Execute single search with phase filters
        search_results = perform_supabase_vector_search(
            query=user_query,
            match_function='match_documents',
            metadata_filters=base_filters,
            limit=8
        )
    else:
        # For other cases: general_knowledge, coworker_focused, or when no specific BASE/PHASE context
        # IMPORTANT: Still filter by user's PCM type if available to get relevant content
        if user_pcm_base and flow_type == 'self_focused':
            # User has a profile and is asking about themselves - filter by their BASE type
            base_filters['pcm_base_type'] = user_pcm_base
            logger.info(f"🔍 Filtering for user's BASE type (general self_focused): {user_pcm_base}")
        elif user_pcm_phase and flow_type == 'self_focused' and pcm_base_or_phase == 'phase':
            # User asking about PHASE specifically  
            base_filters['pcm_phase_type'] = user_pcm_phase
            logger.info(f"🔍 Filtering for user's PHASE type: {user_pcm_phase}")
        elif specific_dimensions_list and pcm_base_or_phase == 'base':
            # Specific dimensions detected but no user profile - get all BASE types for these dimensions
            logger.info(f"🔍 Filtering for specific dimensions without user profile: {specific_dimensions_list}")
            
            all_results = []
            for dimension in specific_dimensions_list:
                # Map dimension names to database section_type values
                mapped_dimension = DIMENSION_MAPPING.get(dimension, dimension)
                
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
    # For PCM BASE searches, use lower similarity threshold to get all 6 dimensions
    if pcm_base_or_phase == 'base' and user_pcm_base:
        min_sim = 0.15  # Lower threshold to capture all BASE dimensions
    else:
        min_sim = 0.30  # Standard threshold for other searches
    
    sanitized_results = sanitize_vector_results(
        results=search_results,
        required_filters=None,
        top_k=8,  # Allow more results for BASE (6 dimensions + buffer)
        min_similarity=min_sim,
        max_chars_per_item=1500,
        max_total_chars=10000  # More space for all dimensions
    )
    
    logger.info(f"📊 PCM search completed: {len(sanitized_results)} sanitized results")
    return sanitized_results, pcm_base_or_phase

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
        "personality_part": "Personality Parts",
        "channel_communication": "Channels of Communication",
        "environmental_preferences": "Environmental Preferences"
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

