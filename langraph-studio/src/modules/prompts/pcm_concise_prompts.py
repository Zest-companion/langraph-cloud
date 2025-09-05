"""
Concise PCM Prompts - Simplified templates using Chain of Thought results
"""

def create_concise_pcm_prompt(
    intent_analysis: dict,
    transition_analysis: dict,
    user_profile: dict,
    vector_context: str
) -> str:
    """
    Creates a concise PCM prompt using Chain of Thought conclusions
    """
    
    # Extract key information
    flow_type = intent_analysis.get('flow_type', 'general_knowledge')
    reasoning = intent_analysis.get('reasoning', 'User asking about PCM')
    transition_rec = transition_analysis.get('transition_recommendation', 'continue_base')
    suggested_action = transition_analysis.get('suggested_next_action', 'Continue exploring')
    dimensions_covered = transition_analysis.get('dimensions_covered_in_this_interaction', [])
    phase_ready = transition_analysis.get('phase_transition_ready', False)
    
    pcm_base = user_profile.get('pcm_base', 'Unknown')
    pcm_phase = user_profile.get('pcm_phase', 'Unknown')
    
    # Base template
    prompt = f"""You are a PCM expert coaching {pcm_base.title()} BASE, {pcm_phase.title()} PHASE.

CONTEXT ANALYSIS: {reasoning}

RECOMMENDATION: {suggested_action}

RELEVANT CONTENT:
{vector_context}

RESPONSE GUIDELINES:
"""
    
    # Add specific guidelines based on analysis
    if flow_type == "self_focused":
        if transition_rec == "suggest_phase":
            prompt += """- All 6 BASE dimensions explored → Suggest PHASE transition
- Say: "We've covered your BASE foundation. Ready to explore your current PHASE?\""""
        elif transition_rec == "continue_base":
            remaining_dims = 6 - len(transition_analysis.get('updated_explored_dimensions', []))
            if remaining_dims > 0:
                prompt += f"""- Continue BASE exploration ({remaining_dims} dimensions remaining)
- Focus on the content provided, ask for examples, offer next dimension"""
            else:
                prompt += """- BASE complete → Suggest PHASE exploration"""
        elif transition_rec == "go_deeper":
            prompt += """- User wants deeper exploration of current dimension
- Provide more detailed insights, ask follow-up questions"""
    
    elif flow_type == "coworker_focused":
        prompt += """- User asking about someone else's PCM
- Focus on general PCM principles, avoid personal assumptions"""
    
    else:  # general_knowledge
        prompt += """- General PCM education
- Explain concepts clearly, offer to explore user's profile"""
    
    # Add validation and next steps
    prompt += f"""

ALWAYS END WITH:
1. Validation question about the content
2. Next step based on: {transition_rec}

Keep response focused and conversational."""
    
    return prompt

def create_simple_pcm_fallback_prompt(user_query: str, user_profile: dict) -> str:
    """Simple fallback when Chain of Thought fails"""
    pcm_base = user_profile.get('pcm_base', 'Unknown')
    
    return f"""You are a PCM expert helping a {pcm_base.title()} BASE.

USER QUESTION: {user_query}

RESPONSE:
- Answer their PCM question naturally
- Use your PCM knowledge 
- Ask for validation
- Suggest next exploration step

Keep it conversational and helpful."""

def create_phase_transition_prompt(
    user_profile: dict,
    explored_dimensions: list,
    vector_context: str
) -> str:
    """Specialized prompt for BASE → PHASE transition"""
    
    pcm_base = user_profile.get('pcm_base', 'Unknown')
    pcm_phase = user_profile.get('pcm_phase', 'Unknown')
    
    return f"""You are a PCM expert. User has completed exploring all 6 BASE dimensions: {', '.join(explored_dimensions)}.

TIME FOR PHASE TRANSITION:

USER PROFILE:
- BASE: {pcm_base.title()} (stable personality - explored)  
- PHASE: {pcm_phase.title()} (current motivational state)

PHASE CONTENT:
{vector_context}

TRANSITION RESPONSE:
"Excellent! We've explored all 6 dimensions of your {pcm_base.title()} BASE - your stable personality foundation. 

Now let's explore your current PHASE: {pcm_phase.title()}. This represents your evolving motivational needs and how you handle stress.

[Present PHASE content from context]

Does this {pcm_phase.title()} PHASE description reflect your current motivational state?"

Keep the transition smooth and natural."""

def create_dimension_specific_prompt(
    dimension: str,
    user_profile: dict, 
    vector_context: str,
    is_systematic: bool = False
) -> str:
    """Focused prompt for specific dimension exploration"""
    
    pcm_base = user_profile.get('pcm_base', 'Unknown')
    
    dimension_names = {
        'perception': 'Perception',
        'strengths': 'Strengths', 
        'interaction_style': 'Interaction Style',
        'personality_part': 'Personality Parts',
        'channel_communication': 'Communication Channels',
        'environmental_preferences': 'Environmental Preferences'
    }
    
    dim_name = dimension_names.get(dimension, dimension.title())
    
    prompt = f"""You are a PCM expert focusing on {dim_name} for a {pcm_base.title()} BASE.

DIMENSION CONTENT:
{vector_context}

RESPONSE STRUCTURE:
1. "Let's explore your {dim_name} as a {pcm_base.title()}."
2. [Present the dimension content naturally]
3. "Does this resonate with your experience?"
4. "Can you share a recent example where this showed up?"
5. """
    
    if is_systematic:
        prompt += "Suggest continuing systematically to the next dimension"
    else:
        prompt += "Offer flexible next steps (deeper dive, another dimension, or PHASE)"
    
    return prompt