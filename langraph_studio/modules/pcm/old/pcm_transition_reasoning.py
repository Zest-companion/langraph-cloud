"""
PCM Transition Chain of Thought Reasoning
Intelligent decision making for BASE dimension tracking and PHASE transition
"""

from typing import Dict, Any, List, Optional
import json
import logging

logger = logging.getLogger(__name__)

def create_pcm_transition_prompt(
    conversation_history: str, 
    current_question: str,
    user_profile: str,
    explored_dimensions: List[str],
    vector_results: List[Dict]
) -> str:
    """Creates Chain of Thought prompt for PCM transition logic"""
    
    all_dimensions = ["perception", "strengths", "interaction_style", 
                     "personality_part", "channel_communication", "environmental_preferences"]
    remaining_dimensions = [dim for dim in all_dimensions if dim not in explored_dimensions]
    
    return f"""You are a PCM expert managing a user's exploration journey. Use step-by-step reasoning to determine the best next steps.

USER PROFILE:
{user_profile}

RECENT CONVERSATION HISTORY:
{conversation_history}

CURRENT QUESTION: "{current_question}"

EXPLORATION STATUS:
- Explored dimensions: {explored_dimensions if explored_dimensions else "None yet"}
- Remaining BASE dimensions: {remaining_dimensions if remaining_dimensions else "All 6 dimensions completed!"}
- Total explored: {len(explored_dimensions)}/6 BASE dimensions

AVAILABLE CONTENT FROM SEARCH:
{format_vector_results_for_reasoning(vector_results)}

STEP-BY-STEP REASONING:

1. CONVERSATION CONTEXT ANALYSIS:
   - What was the user's previous interaction about?
   - Are they continuing from a previous dimension exploration?
   - If they said "yes", "continue", etc., what are they agreeing to?

2. DIMENSION TRACKING ASSESSMENT:
   - Based on the conversation and available content, which dimension(s) does this interaction cover?
   - Should any new dimensions be marked as "explored"?
   - Are there any dimensions the user seems particularly interested in?

3. TRANSITION READINESS EVALUATION:
   - Have all 6 BASE dimensions been sufficiently explored?
   - Is the user showing signs of wanting deeper exploration vs. breadth?
   - Would this be a good moment to suggest PHASE exploration?

4. NEXT STEP RECOMMENDATION:
   - Should we continue with BASE dimensions?
   - Should we suggest transitioning to PHASE?
   - Should we go deeper into a specific dimension?
   - What's the most natural next step based on the conversation flow?

5. FINAL DECISION:
Based on this analysis, here's my recommendation:

```json
{{
    "dimensions_covered_in_this_interaction": ["list", "of", "dimensions"],
    "updated_explored_dimensions": ["complete", "list", "after", "this", "interaction"],
    "transition_recommendation": "continue_base|suggest_phase|go_deeper|clarify_needs",
    "reasoning": "Detailed explanation of the decision",
    "suggested_next_action": "Specific suggestion for what to do next",
    "phase_transition_ready": true/false
}}
```"""

def format_vector_results_for_reasoning(vector_results: List[Dict]) -> str:
    """Format vector search results for the reasoning prompt"""
    if not vector_results:
        return "No specific content found from search."
    
    formatted = []
    for i, result in enumerate(vector_results[:6], 1):  # Top 6 results
        content = result.get('content', '')[:200] + "..."
        similarity = result.get('similarity', 0)
        formatted.append(f"[{i}] {content} (similarity: {similarity:.3f})")
    
    return "\n".join(formatted)

def analyze_pcm_transition(
    current_question: str,
    conversation_history: List[Dict],
    user_profile: Dict[str, Any],
    explored_dimensions: List[str],
    vector_results: List[Dict],
    llm
) -> Dict[str, Any]:
    """
    Analyzes PCM transition needs with Chain of Thought reasoning
    """
    logger.info("🔄 Starting PCM Transition Chain of Thought analysis")
    
    # Format conversation history
    history_text = format_conversation_history_for_transition(conversation_history)
    
    # Format user profile
    profile_text = f"""
PCM BASE: {user_profile.get('pcm_base', 'Not specified')}
PCM PHASE: {user_profile.get('pcm_phase', 'Not specified')}
Exploration Mode: {user_profile.get('exploration_mode', 'flexible')}
"""
    
    # Create transition reasoning prompt
    prompt = create_pcm_transition_prompt(
        conversation_history=history_text,
        current_question=current_question,
        user_profile=profile_text,
        explored_dimensions=explored_dimensions,
        vector_results=vector_results
    )
    
    try:
        logger.info("🔄 Executing PCM Transition reasoning...")
        reasoning_result = llm.invoke(prompt).content
        logger.info(f"✅ Transition reasoning completed: {len(reasoning_result)} chars")
        
        # Extract JSON from conclusion
        transition_data = extract_transition_json(reasoning_result)
        
        # Validation and cleanup
        validated_transition = validate_transition_data(transition_data)
        
        return {
            "transition_analysis": validated_transition,
            "reasoning_process": reasoning_result,
            "success": True
        }
        
    except Exception as e:
        logger.error(f"❌ PCM Transition reasoning failed: {e}")
        return {
            "transition_analysis": get_fallback_transition(explored_dimensions),
            "reasoning_process": f"Transition reasoning error: {str(e)}",
            "success": False
        }

def format_conversation_history_for_transition(messages: List[Dict]) -> str:
    """Format conversation history focusing on dimension exploration"""
    if not messages:
        return "No conversation history."
    
    # Take recent messages but focus on PCM-related content
    recent_messages = messages[-8:] if len(messages) > 8 else messages
    
    formatted_history = []
    for msg in recent_messages[:-1]:  # Exclude current message
        # Handle both dict and object formats
        if hasattr(msg, 'type'):
            role = "User" if msg.type == "human" else "Assistant"
            content = msg.content
        else:
            role = "User" if msg.get('type') == "human" else "Assistant"
            content = msg.get('content', '')
        
        # Truncate but keep PCM-relevant keywords
        if len(content) > 200:
            content = content[:200] + "..."
        
        formatted_history.append(f"{role}: {content}")
    
    return "\\n".join(formatted_history) if formatted_history else "First PCM interaction."

def extract_transition_json(reasoning_text: str) -> Dict[str, Any]:
    """Extract JSON conclusion from transition reasoning"""
    try:
        # Look for JSON between ```json and ```
        start_marker = "```json"
        end_marker = "```"
        
        start_idx = reasoning_text.find(start_marker)
        if start_idx == -1:
            raise ValueError("No JSON found in transition reasoning")
        
        start_idx += len(start_marker)
        end_idx = reasoning_text.find(end_marker, start_idx)
        
        if end_idx == -1:
            raise ValueError("Unclosed JSON in transition reasoning")
        
        json_str = reasoning_text[start_idx:end_idx].strip()
        return json.loads(json_str)
        
    except Exception as e:
        logger.error(f"❌ Transition JSON extraction error: {e}")
        raise

def validate_transition_data(transition_data: Dict[str, Any]) -> Dict[str, Any]:
    """Validates and cleans transition data"""
    
    valid_dimensions = ["perception", "strengths", "interaction_style", 
                       "personality_part", "channel_communication", "environmental_preferences"]
    
    valid_transitions = ["continue_base", "suggest_phase", "go_deeper", "clarify_needs"]
    
    # Clean dimensions covered
    covered = transition_data.get("dimensions_covered_in_this_interaction", [])
    if isinstance(covered, list):
        covered = [dim for dim in covered if dim in valid_dimensions]
    else:
        covered = []
    
    # Clean updated explored dimensions
    explored = transition_data.get("updated_explored_dimensions", [])
    if isinstance(explored, list):
        explored = [dim for dim in explored if dim in valid_dimensions]
    else:
        explored = []
    
    cleaned = {
        "dimensions_covered_in_this_interaction": covered,
        "updated_explored_dimensions": explored,
        "transition_recommendation": transition_data.get("transition_recommendation", "continue_base"),
        "reasoning": transition_data.get("reasoning", "No reasoning provided"),
        "suggested_next_action": transition_data.get("suggested_next_action", "Continue exploring BASE dimensions"),
        "phase_transition_ready": bool(transition_data.get("phase_transition_ready", False))
    }
    
    # Validate transition recommendation
    if cleaned["transition_recommendation"] not in valid_transitions:
        cleaned["transition_recommendation"] = "continue_base"
    
    # Auto-detect phase readiness based on explored dimensions
    if len(cleaned["updated_explored_dimensions"]) >= 6:
        cleaned["phase_transition_ready"] = True
        if cleaned["transition_recommendation"] == "continue_base":
            cleaned["transition_recommendation"] = "suggest_phase"
    
    return cleaned

def get_fallback_transition(explored_dimensions: List[str]) -> Dict[str, Any]:
    """Fallback transition data in case of error"""
    return {
        "dimensions_covered_in_this_interaction": [],
        "updated_explored_dimensions": explored_dimensions,  # Keep existing
        "transition_recommendation": "continue_base",
        "reasoning": "Fallback used due to transition reasoning error",
        "suggested_next_action": "Continue exploring your BASE dimensions",
        "phase_transition_ready": len(explored_dimensions) >= 6
    }