"""
PCM Chain of Thought Reasoning (Pure Python - No LangChain)
Improves intent analysis with explicit contextual reasoning
"""

from typing import Dict, Any, List, Optional
import json
import logging

logger = logging.getLogger(__name__)

def create_pcm_reasoning_prompt(conversation_history: str, current_question: str, user_profile: str) -> str:
    """Creates Chain of Thought prompt for PCM"""
    return f"""You are a Process Communication Model (PCM) expert. Analyze this question using step-by-step reasoning.

USER PROFILE:
{user_profile}

RECENT CONVERSATION HISTORY:
{conversation_history}

CURRENT QUESTION: "{current_question}"

STEP-BY-STEP REASONING:

1. CONVERSATIONAL CONTEXT:
   - What was the user saying in previous messages?
   - Is there continuity with the conversation?
   - If it's "yes", "oui", "continue", "tell me about...", what does it refer to?

2. QUESTION ANALYSIS:
   - Does this question concern the user themselves (self) or someone else (coworker)?
   - Does it talk about their PCM BASE (stable personality) or PHASE (temporary evolution)?
   - Does it mention specific dimensions (perception, strengths, interaction style, etc.)?

3. DETECTED INTENTION:
   - What does the user really want to know?
   - What type of PCM knowledge base search will be necessary?
   - Do they want systematic exploration of all dimensions or focus on something specific?

4. FINAL DECISION:
Based on this reasoning, here is my conclusion:

```json
{{
    "flow_type": "self_focused|coworker_focused|general_knowledge",
    "language": "fr|en",
    "pcm_context": "base|phase|null",
    "dimensions": ["perception", "strengths", "etc..."],
    "exploration_mode": "systematic|flexible",
    "reasoning": "Clear explanation of the reasoning that led to this conclusion"
}}
```"""

def analyze_pcm_intent_with_reasoning(
    current_question: str,
    conversation_history: List[Dict],
    user_profile: Dict[str, Any],
    llm
) -> Dict[str, Any]:
    """
    Analyzes PCM intent with Chain of Thought reasoning
    """
    logger.info("🧠 Starting PCM Chain of Thought analysis")
    
    # Format conversation history
    history_text = format_conversation_history(conversation_history)
    
    # Format user profile
    profile_text = f"""
PCM BASE: {user_profile.get('pcm_base', 'Not specified')}
PCM PHASE: {user_profile.get('pcm_phase', 'Not specified')}
Email: {user_profile.get('email', 'Not specified')}
"""
    
    # Create Chain of Thought prompt
    prompt = create_pcm_reasoning_prompt(
        conversation_history=history_text,
        current_question=current_question,
        user_profile=profile_text
    )
    
    try:
        logger.info("🤔 Executing Chain of Thought reasoning...")
        reasoning_result = llm.invoke(prompt).content
        logger.info(f"✅ Chain of Thought completed: {len(reasoning_result)} chars")
        
        # Extract JSON from conclusion
        intent_data = extract_json_from_reasoning(reasoning_result)
        
        # Validation and cleanup
        validated_intent = validate_and_clean_intent(intent_data)
        
        return {
            "intent_analysis": validated_intent,
            "reasoning_process": reasoning_result,
            "success": True
        }
        
    except Exception as e:
        logger.error(f"❌ Chain of Thought reasoning failed: {e}")
        return {
            "intent_analysis": get_fallback_intent(),
            "reasoning_process": f"Reasoning error: {str(e)}",
            "success": False
        }

def format_conversation_history(messages: List[Dict]) -> str:
    """Formats conversation history for the prompt"""
    if not messages:
        return "No conversation history."
    
    # Take last 6 messages for context
    recent_messages = messages[-6:] if len(messages) > 6 else messages
    
    formatted_history = []
    for i, msg in enumerate(recent_messages[:-1]):  # Exclude current message
        # Handle both dict and object formats
        if hasattr(msg, 'type'):
            role = "User" if msg.type == "human" else "Assistant"
            content = msg.content
        else:
            role = "User" if msg.get('type') == "human" else "Assistant"
            content = msg.get('content', '')
        
        content = content[:300] + "..." if len(content) > 300 else content
        formatted_history.append(f"{role}: {content}")
    
    return "\\n".join(formatted_history) if formatted_history else "First interaction."

def extract_json_from_reasoning(reasoning_text: str) -> Dict[str, Any]:
    """Extracts JSON conclusion from reasoning"""
    try:
        # Look for JSON between ```json and ```
        start_marker = "```json"
        end_marker = "```"
        
        start_idx = reasoning_text.find(start_marker)
        if start_idx == -1:
            raise ValueError("No JSON found in reasoning")
        
        start_idx += len(start_marker)
        end_idx = reasoning_text.find(end_marker, start_idx)
        
        if end_idx == -1:
            raise ValueError("Unclosed JSON in reasoning")
        
        json_str = reasoning_text[start_idx:end_idx].strip()
        return json.loads(json_str)
        
    except Exception as e:
        logger.error(f"❌ JSON extraction error: {e}")
        raise

def validate_and_clean_intent(intent_data: Dict[str, Any]) -> Dict[str, Any]:
    """Validates and cleans intent data"""
    
    # Default values
    valid_flow_types = ["self_focused", "coworker_focused", "general_knowledge"]
    valid_languages = ["fr", "en"] 
    valid_contexts = ["base", "phase", None]
    valid_modes = ["systematic", "flexible"]
    
    cleaned = {
        "flow_type": intent_data.get("flow_type", "general_knowledge"),
        "language": intent_data.get("language", "fr"),
        "pcm_context": intent_data.get("pcm_context"),
        "dimensions": intent_data.get("dimensions", []),
        "exploration_mode": intent_data.get("exploration_mode", "flexible"),
        "reasoning": intent_data.get("reasoning", "Reasoning not available")
    }
    
    # Validate values
    if cleaned["flow_type"] not in valid_flow_types:
        cleaned["flow_type"] = "general_knowledge"
        
    if cleaned["language"] not in valid_languages:
        cleaned["language"] = "fr"
        
    if cleaned["pcm_context"] == "null":
        cleaned["pcm_context"] = None
        
    if cleaned["exploration_mode"] not in valid_modes:
        cleaned["exploration_mode"] = "flexible"
    
    # Clean dimensions - using same mapping as pcm_analysis.py
    if isinstance(cleaned["dimensions"], list):
        # Same 6 BASE dimensions as in DIMENSION_MAPPING
        valid_dimensions = ["perception", "strengths", "interaction_style", 
                          "personality_part", "channel_communication", "environmental_preferences"]
        cleaned["dimensions"] = [dim for dim in cleaned["dimensions"] if dim in valid_dimensions]
    else:
        cleaned["dimensions"] = []
    
    return cleaned

def get_fallback_intent() -> Dict[str, Any]:
    """Fallback intent in case of error"""
    return {
        "flow_type": "general_knowledge",
        "language": "fr", 
        "pcm_context": None,
        "dimensions": [],
        "exploration_mode": "flexible",
        "reasoning": "Fallback used due to reasoning error"
    }