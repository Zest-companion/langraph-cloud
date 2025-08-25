"""
Constructeurs de prompts pour différents sous-thèmes
"""
import logging
from typing import Dict, List, Optional
from ..common.types import WorkflowState
from ..common.config import supabase
from ..common.types import WorkflowState
from .pcm_first_interaction_prompts import build_pcm_first_interaction_general_prompt, build_pcm_first_interaction_dimension_prompt, build_pcm_first_interaction_multi_dimension_prompt, build_pcm_first_interaction_phase_redirect_prompt

logger = logging.getLogger(__name__)

def get_lencioni_overview():
    """Récupère l'overview Lencioni directement depuis la base de données documents_content_test"""
    try:
        logger.info("🔍 Attempting to fetch Lencioni overview from 'documents_content_test' table...")
        response = supabase.table('documents_content_test').select('content').eq(
            'metadata->>lencioni_dysfunction', 'general'
        ).eq(
            'metadata->>lencioni_content_type', 'overview'
        ).limit(10).execute()
        
        if response.data:
            logger.info(f"✅ Found {len(response.data)} overview items from documents_content_test")
            overview_content = ""
            for item in response.data:
                overview_content += item.get('content', '') + "\n\n"
            return overview_content
        else:
            logger.warning("⚠️ No Lencioni overview content found in documents_content_test")
            return "Lencioni overview not available from knowledge base."
                
    except Exception as e:
        logger.error(f"❌ Error fetching Lencioni overview: {e}")
        return f"Error retrieving Lencioni overview: {str(e)}"

def generate_contextual_coaching_instructions(lencioni_data, intent_type, dysfunctions_mentioned=None):
    """Génère des instructions de coaching contextuelles basées sur les scores et la pyramide"""
    if not lencioni_data:
        return """
**COACHING APPROACH**: Generate contextual coaching questions that:
• Prompt reflection about team dynamics and the model concepts
• Suggest practical next steps for team assessment
• Guide toward actionable improvements based on their context"""

    # Analyser les scores
    dysfunction_scores = {}
    for item in lencioni_data:
        dysfunction = item.get('dysfunction', '').lower()
        score = item.get('score', 0)
        dysfunction_scores[dysfunction] = score

    # Identifier la prochaine priorité selon la pyramide avec distinction Low/Medium
    next_priority = None
    priority_score = 0
    reasoning = ""
    urgency = ""

    trust_score = dysfunction_scores.get('trust', 0)
    conflict_score = dysfunction_scores.get('conflict', 0)
    commitment_score = dysfunction_scores.get('commitment', 0)
    accountability_score = dysfunction_scores.get('accountability', 0)
    results_score = dysfunction_scores.get('results', 0)

    # Priorité 1: Scores Low (≤ 3.0) selon la pyramide
    if trust_score <= 3.0:
        next_priority = "Trust"
        priority_score = trust_score
        reasoning = "Foundation issue - critical priority"
        urgency = "URGENT"
    elif conflict_score <= 3.0:
        next_priority = "Conflict"
        priority_score = conflict_score
        reasoning = "Trust adequate, but Conflict needs urgent attention"
        urgency = "URGENT"
    elif commitment_score <= 3.0:
        next_priority = "Commitment"
        priority_score = commitment_score
        reasoning = "Trust & Conflict adequate, but Commitment critically low"
        urgency = "URGENT"
    elif accountability_score <= 3.0:
        next_priority = "Accountability"
        priority_score = accountability_score
        reasoning = "Foundation solid, but Accountability critically low"
        urgency = "URGENT"
    elif results_score <= 3.0:
        next_priority = "Results"
        priority_score = results_score
        reasoning = "All foundations adequate, but Results critically low"
        urgency = "URGENT"
    # Priorité 2: Scores Medium (3.1-4.0) selon la pyramide
    elif trust_score <= 4.0:
        next_priority = "Trust"
        priority_score = trust_score
        reasoning = "Foundation medium - still room for improvement before moving up"
        urgency = "IMPORTANT"
    elif conflict_score <= 4.0:
        next_priority = "Conflict"
        priority_score = conflict_score
        reasoning = "Trust solid, Conflict medium - good opportunity to strengthen"
        urgency = "IMPORTANT"
    elif commitment_score <= 4.0:
        next_priority = "Commitment"
        priority_score = commitment_score
        reasoning = "Strong foundation, Commitment medium - ready for improvement"
        urgency = "IMPORTANT"
    elif accountability_score <= 4.0:
        next_priority = "Accountability"
        priority_score = accountability_score
        reasoning = "Solid foundation, Accountability medium - good focus area"
        urgency = "IMPORTANT"
    else:
        next_priority = "Results"
        priority_score = results_score
        reasoning = "Strong team foundation - focus on optimizing collective results"
        urgency = "OPTIMIZATION"

    def get_status_label(score):
        if score > 4.0:
            return "Strong"
        elif score > 3.0:
            return "Medium - room for improvement"
        else:
            return "Low - needs urgent attention"

    # Construire les instructions contextuelles
    coaching_instructions = f"""
**CONTEXTUAL COACHING APPROACH**:

**TEAM'S CURRENT SITUATION**:
• Trust: {trust_score:.1f}/5.0 ({get_status_label(trust_score)})
• Conflict: {conflict_score:.1f}/5.0 ({get_status_label(conflict_score)})
• Commitment: {commitment_score:.1f}/5.0 ({get_status_label(commitment_score)})
• Accountability: {accountability_score:.1f}/5.0 ({get_status_label(accountability_score)})
• Results: {results_score:.1f}/5.0 ({get_status_label(results_score)})

**PYRAMID LOGIC PRIORITY**: 
→ **{next_priority.upper()}** (Score: {priority_score:.1f}/5.0) - {urgency} - {reasoning}

**COACHING APPROACH BY CONTEXT**:"""

    # Déterminer le niveau de détail selon l'intent_type et les dysfonctions mentionnées
    if intent_type in ["GENERAL_KNOWLEDGE", "LENCIONI_GENERAL_KNOWLEDGE"]:
        coaching_instructions += f"""
• **THEORETICAL FOCUS**: Explain the model and acknowledge their scores
• **REDIRECTION STRATEGY**: Guide them toward exploring their priority dysfunction ({next_priority})
• **FOLLOW-UP QUESTIONS**: Generate your own redirection questions that:
  - Invite them to explore strategies for their priority dysfunction ({next_priority})
  - Reference their specific score ({priority_score:.1f}) to justify the focus
  - Use inviting language like "Would you like to...", "Should we explore...", "Are you curious about..."
• **REDIRECTION STYLE**: Questions should be invitations to go deeper, not diagnostic questions
• **STRICT AVOID**: 
  - Questions about their current behaviors ("can you think of instances when...")
  - Questions about specific team situations or examples
  - Detailed analysis questions about team dynamics
  - Coaching questions that try to solve the problem directly
• **GOAL**: Make them WANT to ask follow-up questions like "How can we improve conflict?" or "What strategies work for this?"""

    elif intent_type == "REPORT_LOOKUP":
        if dysfunctions_mentioned and len(dysfunctions_mentioned) > 0:
            coaching_instructions += f"""
• **DETAILED REPORT ANALYSIS**: Focus on the mentioned dysfunction(s): {', '.join(dysfunctions_mentioned)}
• **SCORE BREAKDOWN**: Analyze scores by level for insights:
  - High scores (>4.0): What's working well
  - Medium scores (3.1-4.0): Areas with room for improvement  
  - Low scores (≤3.0): Priority areas needing urgent attention
• **DETAILED QUESTIONS CONTEXT**: Use lencioni_details to explain specific behaviors behind the scores
• **INSIGHTS GENERATION**: Create insights based on:
  - Pattern analysis of high/medium/low scores
  - Specific question responses that explain the scores
  - Connection between detailed behaviors and overall dysfunction score
• **INTERPRETIVE FOCUS**: Help them understand WHY they got these specific scores
• **AVOID**: Action recommendations - focus on interpretation and understanding"""
        else:
            coaching_instructions += f"""
• **INTERPRETIVE GUIDANCE**: Help them understand what all their scores mean
• **HIGH-LEVEL RECOMMENDATIONS**: Provide overview-level guidance for all dysfunctions based on their scores:
  - Trust: {trust_score:.1f}/5.0 ({get_status_label(trust_score)})
  - Conflict: {conflict_score:.1f}/5.0 ({get_status_label(conflict_score)})
  - Commitment: {commitment_score:.1f}/5.0 ({get_status_label(commitment_score)})
  - Accountability: {accountability_score:.1f}/5.0 ({get_status_label(accountability_score)})
  - Results: {results_score:.1f}/5.0 ({get_status_label(results_score)})
• **PRIORITY IDENTIFICATION**: Explain why {next_priority} is their logical next step based on pyramid logic
• **COACHING QUESTIONS**: Generate redirection questions for deeper exploration:
  - Help them interpret their overall pattern of scores
  - Connect their results to pyramid logic and next steps
  - Invite them to explore specific dysfunctions in more detail
• **AVOID**: Detailed tactics or specific exercises - focus on understanding results + high-level direction"""

    elif intent_type == "INSIGHT_BLEND":
        if dysfunctions_mentioned and len(dysfunctions_mentioned) > 0:
            coaching_instructions += f"""
• **DETAILED RECOMMENDATIONS**: Provide specific, actionable strategies
• **TARGETED COACHING**: Focus on the mentioned dysfunction(s): {', '.join(dysfunctions_mentioned)}
• **SPECIFIC ACTIONS**: Generate detailed coaching questions about:
  - Concrete behaviors they can change
  - Specific team exercises or practices
  - Detailed next steps and implementation strategies
  - How to measure progress on this specific dysfunction
• **DEEP DIVE**: Provide comprehensive guidance for their specific focus area"""
        else:
            coaching_instructions += f"""
• **HIGH-LEVEL RECOMMENDATIONS**: Provide overview-level guidance for all dysfunctions based on their scores:
  - Trust: {trust_score:.1f}/5.0 ({get_status_label(trust_score)})
  - Conflict: {conflict_score:.1f}/5.0 ({get_status_label(conflict_score)})
  - Commitment: {commitment_score:.1f}/5.0 ({get_status_label(commitment_score)})
  - Accountability: {accountability_score:.1f}/5.0 ({get_status_label(accountability_score)})
  - Results: {results_score:.1f}/5.0 ({get_status_label(results_score)})
• **COACHING APPROACH**: Address their overall team pattern with all scores:
  - Comment on each dysfunction's status with specific scores
  - Give general direction for each area without detailed strategies
  - Emphasize their pyramid priority ({next_priority}) as the logical starting point
• **FOLLOW-UP QUESTIONS**: Generate redirection questions for deeper exploration:
  - Invite them to explore specific dysfunctions in more detail
  - "Would you like specific strategies for improving [priority dysfunction]?"
• **AVOID**: Detailed tactics, specific exercises, or deep behavioral analysis
• **GOAL**: Overview-level coaching with all scores + redirection to specific dysfunction focus"""

    else:
        coaching_instructions += f"""
• **CONTEXTUAL COACHING**: Adapt coaching to their specific context
• **PRIORITY FOCUS**: Guide toward {next_priority} as their logical next step
• **BALANCED APPROACH**: Provide appropriate level of detail for the context"""

    coaching_instructions += f"""

**ALWAYS REFERENCE**: Their specific scores and pyramid logic in your coaching questions"""

    return coaching_instructions

def select_pcm_prompt(state: WorkflowState) -> str:
    """
    Sélectionne le bon prompt PCM selon le contexte
    Gère les transitions BASE → PHASE automatiquement
    
    EXPERIMENTAL: Test PCM Agent vs Traditional approach
    """
    import logging
    logger = logging.getLogger(__name__)
    
    # EXPERIMENTAL: PCM Agent - Activé pour LangGraph Cloud
    use_pcm_agent = True  # ← ACTIVÉ pour Cloud - Désactivez pour local Studio
    
    if use_pcm_agent:
        try:
            from ..pcm.pcm_agent import pcm_agent_response
            logger.info("🤖 Using EXPERIMENTAL PCM Agent instead of traditional prompts")
            return pcm_agent_response(state)
        except Exception as e:
            logger.error(f"❌ PCM Agent failed, falling back to traditional: {e}")
            # Fall back to traditional approach
    
    # TRADITIONAL APPROACH (current)
    logger.info("📝 Using traditional PCM prompt selection")
    
    # Get intent analysis results
    pcm_intent = state.get('pcm_intent_analysis', {})
    pcm_base_or_phase = state.get('pcm_base_or_phase')
    
    # Check if this is the first PCM interaction
    messages = state.get('messages', [])
    is_first_interaction = len(messages) <= 2  # Very beginning of conversation
    
    # Check PHASE transition signals
    phase_request_detected = pcm_intent.get('phase_request_detected', False)
    should_suggest_phase = pcm_intent.get('should_suggest_phase', False)
    
    # Route to appropriate prompt
    if is_first_interaction:
        # First interaction - use specialized first interaction prompt
        return build_pcm_first_interaction_prompt(state)
    
    elif phase_request_detected or (should_suggest_phase and pcm_base_or_phase != 'phase'):
        # But don't suggest PHASE if user is asking for a specific BASE dimension
        from ..pcm.pcm_analysis import _detect_base_dimension
        specific_base_request = _detect_base_dimension(state.get('user_message', ''))
        
        if specific_base_request:
            # User wants specific BASE dimension - don't override with PHASE
            return build_pcm_self_focused_base_prompt(state)
        else:
            # User wants PHASE or we should suggest it → Use transition prompt
            return build_pcm_phase_transition_prompt(state)
    
    elif pcm_base_or_phase == 'phase':
        # Check if user is now asking for a specific BASE dimension
        from ..pcm.pcm_analysis import _detect_base_dimension
        specific_base_request = _detect_base_dimension(state.get('user_message', ''))
        
        if specific_base_request:
            # User switched from PHASE to asking about BASE dimension
            return build_pcm_self_focused_base_prompt(state)
        else:
            # Continue with PHASE context → Use regular PHASE prompt
            return build_pcm_self_focused_phase_prompt(state)
    
    else:
        # Regular BASE exploration → Use BASE prompt
        return build_pcm_self_focused_base_prompt(state)

def create_prompt_by_subtheme(sub_theme: str, state: WorkflowState) -> str:
    """
    Factory qui retourne le bon prompt selon le sub_theme
    Système extensible pour supporter 15+ sous-thèmes différents
    """
    logger.info(f"🎯 Creating prompt for sub_theme: {sub_theme}")
    
    # Debug: Log PCM values for A2_PersonalityPCM
    if sub_theme == 'A2_PersonalityPCM':
        pcm_base = state.get('pcm_base')
        pcm_phase = state.get('pcm_phase')
        flow_type = state.get('flow_type', 'general_knowledge')
        logger.info(f"🔍 DEBUG create_prompt_by_subtheme A2: pcm_base={pcm_base}, pcm_phase={pcm_phase}, flow_type={flow_type}")
    
    if sub_theme == 'D6_CollectiveSuccess':
        return build_lencioni_prompt(state)
    elif sub_theme == 'A1_PersonalityMBTI':
        return build_mbti_prompt(state)
    elif sub_theme == 'A2_PersonalityPCM':
        # Use intelligent PCM prompt selection with transition management
        flow_type = state.get('flow_type', 'general_knowledge')
        if flow_type == 'self_focused':
            return select_pcm_prompt(state)
        elif flow_type == 'coworker_focused':
            return build_pcm_coworker_focused_prompt(state)
        else:  # general_knowledge
            return build_pcm_general_knowledge_prompt(state)
    elif sub_theme == 'A4_LeadershipStyle':
        return build_leadership_prompt(state)
    elif sub_theme == 'C8_Introspection':
        return build_general_prompt(state)
    # Ici on peut facilement ajouter d'autres sous-thèmes :
    # elif sub_theme == 'B2_Communication':
    #     return build_communication_prompt(state)
    # elif sub_theme == 'C3_Leadership':
    #     return build_leadership_prompt(state)
    else:
        # Fallback vers le prompt MBTI par défaut
        logger.info(f"⚠️ No specific prompt for {sub_theme}, using MBTI default")
        return build_mbti_prompt(state)


def build_lencioni_prompt(state: WorkflowState) -> str:
    """Construit le prompt spécialisé pour Lencioni/D6_CollectiveSuccess"""
    # Récupérer les informations d'intent et dysfonctions mentionnées
    intent_analysis = state.get("lencioni_intent_analysis", {})
    dysfunctions_mentioned = intent_analysis.get("dysfunctions_mentioned", [])
    
    # Vérifier si une clarification est nécessaire
    if state.get("needs_user_clarification", False):
        # Analyser les scores pour suggérer des priorités
        lencioni_data = state.get("lencioni_data", [])
        
        clarification_message = "Your question about team improvement is quite broad. To provide the most relevant guidance, let me suggest where to focus based on "
        
        if lencioni_data:
            # Analyser les scores et suggérer selon la pyramide
            dysfunction_scores = {}
            for item in lencioni_data:
                dysfunction = item.get('dysfunction', '')
                score = item.get('score', 0)
                dysfunction_scores[dysfunction.lower()] = score
            
            # Ordre pyramidal pour analyse
            pyramid_order = ['trust', 'conflict', 'commitment', 'accountability', 'results']
            problematic_areas = []
            
            for dysfunction in pyramid_order:
                score = dysfunction_scores.get(dysfunction, 0)
                # Scores bas/moyens (ajuster selon votre échelle)
                if score <= 2.5:  # Ajuster ce seuil selon votre échelle de notation
                    problematic_areas.append(dysfunction.title())
            
            if problematic_areas:
                clarification_message += f"your team's assessment scores:\n\n"
                clarification_message += f"**Priority areas** (following the pyramid foundation):\n"
                for i, area in enumerate(problematic_areas[:3], 1):  # Top 3 priorités
                    clarification_message += f"{i}. **{area}** - Your team shows room for improvement here\n"
                clarification_message += f"\nI recommend starting with **{problematic_areas[0]}** as it forms the foundation for other improvements.\n\n"
                clarification_message += "Which area would you like to focus on?"
            else:
                clarification_message += "your team's strong assessment scores:\n\nYour team is performing well across all areas! For continuous improvement, which aspect would you like to explore further?\n• **Trust** - Building deeper vulnerability-based trust\n• **Conflict** - Enhancing productive debate\n• **Commitment** - Strengthening decision clarity\n• **Accountability** - Improving peer accountability\n• **Results** - Focusing on collective outcomes"
        else:
            clarification_message += "Lencioni's pyramid model:\n\nTo provide targeted advice, which aspect of team dynamics would you like to focus on?\n• **Trust** - Building vulnerability-based trust (foundation)\n• **Conflict** - Engaging in productive conflict\n• **Commitment** - Achieving buy-in and clarity\n• **Accountability** - Holding each other accountable\n• **Results** - Focusing on collective outcomes\n\nI recommend starting with Trust if you're unsure, as it forms the foundation."
        
        return f"""You are a team development coach specializing in the Lencioni Five Dysfunctions of a Team model.

{clarification_message}"""
    
    lencioni_data = state.get("lencioni_data", [])
    lencioni_details = state.get("lencioni_details", [])
    dysfunction_focus = state.get("dysfunction_focus", [])
    user_question = state.get('user_message', '') or (state.get('messages', [{}])[-1].get('content', '') if state.get('messages') else '')
    intent_analysis = state.get("lencioni_intent_analysis", {})
    intent_type = intent_analysis.get("intent_type", "INSIGHT_BLEND")
    search_results = state.get("lencioni_search_results", {})
    
    # Debug: Log du state Lencioni
    logger.info(f"🏛️ Building Lencioni prompt for intent: {intent_type}")
    logger.info(f"🏛️ User question: '{user_question}'")
    logger.info(f"🏛️ Lencioni data available: {len(lencioni_data) if lencioni_data else 0} items")
    if lencioni_data:
        for item in lencioni_data:
            logger.info(f"  - {item.get('dysfunction', 'Unknown')}: {item.get('score', 'N/A')}")
    
    # Ajouter l'historique de conversation pour éviter les répétitions
    conversation_history = []
    for msg in state.get('messages', [])[-3:]:  # 3 derniers messages pour contexte
        if hasattr(msg, 'type') and hasattr(msg, 'content'):
            role = "User" if msg.type == 'human' else "Assistant"
            conversation_history.append(f"{role}: {msg.content}")
        elif isinstance(msg, dict):
            role = "User" if msg.get('role') == 'user' else "Assistant"
            conversation_history.append(f"{role}: {msg.get('content', '')}")
    
    history_text = "\n".join(conversation_history) if conversation_history else "No previous conversation"
    
    system_prompt = f"""You are ZEST COMPANION, an expert team development and leadership mentor, specialized in Patrick Lencioni's Five Dysfunctions of a Team model.

⚠️ CRITICAL SCOPE RESTRICTION - YOU MUST ENFORCE THIS:
- You ONLY address WORKPLACE TEAM DYNAMICS and PROFESSIONAL TEAM ISSUES
- You MUST REFUSE to answer questions about:
  • Family relationships or personal/romantic relationships
  • Non-work related issues
  • Individual therapy or personal psychological issues
  • Any topic not directly related to workplace team dynamics
- If the question is not about workplace teams, respond: "I specialize exclusively in workplace team dynamics using the Lencioni model. For personal or non-work topics, please consult an appropriate specialist. How can I help with your team challenges at work?"

The 5 dysfunctions are (in pyramid order):
1. Absence of Trust - Foundation of the team
2. Fear of Conflict - Open debates and productive conflict
3. Lack of Commitment - Buy-in to decisions
4. Avoidance of Accountability - Mutual accountability
5. Inattention to Results - Focus on collective objectives

USER QUESTION: "{user_question}"

RECENT CONVERSATION HISTORY:
{history_text}

User Name: {state.get('user_name', 'Unknown')}
"""
    
    # REPORT_LOOKUP SIMPLIFIÉ: Scores + redirection vers conseils
    if intent_type == "REPORT_LOOKUP" and lencioni_data:
        system_prompt += "\n\n📊 **Your Team's Lencioni Assessment Scores:**\n"
        
        # Organiser par ordre pyramidal
        dysfunction_order = ['Trust', 'Conflict', 'Commitment', 'Accountability', 'Results']
        organized_data = {}
        for item in lencioni_data:
            dysfunction = item.get('dysfunction', '').title()
            organized_data[dysfunction] = item
        
        for dysfunction in dysfunction_order:
            if dysfunction in organized_data:
                item = organized_data[dysfunction]
                score = item.get('score', 0)
                level = item.get('level', 'Unknown')
                system_prompt += f"• **{dysfunction}**: {score}/5.0 ({level})\n"
        
        system_prompt += "\n💡 **Want actionable recommendations?** Ask me 'How can we improve our team dynamics?' or 'What should we focus on?' for personalized coaching advice based on these scores."
    elif intent_type == "REPORT_LOOKUP" and not lencioni_data:
        system_prompt += "\n\n📊 **No Assessment Data Available**\nNo Lencioni assessment data found for your profile. Consider taking the team assessment to get personalized insights.\n\n💡 **Want to learn about the model?** Ask me about Lencioni's Five Dysfunctions or specific concepts like trust, conflict, commitment, accountability, or results."
    elif lencioni_data:
        # For other intents, we'll add the detailed scores in the search results section
        system_prompt += "\nTeam assessment data available - see detailed context below."
    else:
        system_prompt += "\nThe user doesn't have a Lencioni profile yet. Provide general team development advice."
    
    # Ajouter les résultats de recherche vectorielle selon l'intent
    # Pour REPORT_LOOKUP avec dysfonctions, toujours montrer la section même si search_results est vide
    if search_results or (intent_type == "REPORT_LOOKUP" and lencioni_details and dysfunctions_mentioned):
        system_prompt += "\n\n📚 RELEVANT KNOWLEDGE BASE CONTENT:\n"
        
        if intent_type == "REPORT_LOOKUP" and ((search_results and search_results.get("report_lookup_content")) or (lencioni_details and dysfunctions_mentioned)):
            system_prompt += "\n**📖 SUPPLEMENTARY CONTEXT - LENCIONI MODEL OVERVIEW:**\n"
            system_prompt += "USE THIS TO COMPLEMENT (NOT REPLACE) THE TEAM SCORES ABOVE.\n"
            system_prompt += "This overview helps explain the theory behind each dysfunction to enrich your interpretation of the team's specific scores.\n\n"
            
            # Ajouter le contenu des search_results s'il existe
            if search_results and search_results.get("report_lookup_content"):
                for item in search_results["report_lookup_content"]:
                    if item.get('type') == 'lencioni_overview':
                        # Pour l'overview, inclure TOUT le contenu (pas de limite)
                        full_content = item.get('content', '')
                        system_prompt += f"{full_content}\n\n"
                        logger.info(f"📖 Added full overview content: {len(full_content)} characters")
                    else:
                        # Pour d'autres contenus, garder une limite raisonnable
                        system_prompt += f"- {item.get('content', '')[:500]}...\n"
            
            # Ajouter les questions détaillées si des dysfonctions spécifiques sont mentionnées
            if lencioni_details and dysfunctions_mentioned:
                system_prompt += "\n**📋 DETAILED ASSESSMENT QUESTIONS**\n"
                system_prompt += "Specific questions and individual scores for analysis:\n"
                
                # Organiser par dysfonction
                details_by_dysfunction = {}
                for detail in lencioni_details:
                    dysfunction = detail.get('dysfunction', '').title()
                    if dysfunction not in details_by_dysfunction:
                        details_by_dysfunction[dysfunction] = []
                    details_by_dysfunction[dysfunction].append(detail)
                
                # Afficher pour chaque dysfonction mentionnée
                for dysfunction in dysfunctions_mentioned:
                    dysfunction_title = dysfunction.title()
                    if dysfunction_title in details_by_dysfunction:
                        system_prompt += f"\n**{dysfunction_title} Questions:**\n"
                        for detail in details_by_dysfunction[dysfunction_title]:
                            question = detail.get('question', '')
                            score = detail.get('score', 0)
                            system_prompt += f"- {question} → {score}/5.0\n"
        
        elif intent_type in ["LENCIONI_GENERAL_KNOWLEDGE", "GENERAL_KNOWLEDGE"]:
            system_prompt += "\n**📚 LENCIONI FIVE DYSFUNCTIONS FRAMEWORK OVERVIEW:**\n"
            # Toujours récupérer et afficher l'overview pour GENERAL_KNOWLEDGE
            overview_from_search = False
            if search_results.get("general_knowledge_content"):
                for item in search_results["general_knowledge_content"]:
                    content = item.get('content', '')
                    if content:
                        system_prompt += f"{content}\n\n"
                        overview_from_search = True
            
            # Si pas d'overview dans les search_results, le récupérer directement
            if not overview_from_search:
                logger.info("🔍 No overview in search_results, fetching directly from database...")
                overview_content = get_lencioni_overview()
                system_prompt += f"{overview_content}\n"
                logger.info(f"📚 Added overview directly from database: {len(overview_content)} characters")
        
        elif intent_type == "INSIGHT_BLEND" and search_results.get("insight_blend_content"):
            # Organiser les données par dysfonction pour éviter les doublons
            dysfunction_order = ['Trust', 'Conflict', 'Commitment', 'Accountability', 'Results']
            organized_data = {}
            if lencioni_data:
                for item in lencioni_data:
                    dysfunction = item.get('dysfunction', '').title()
                    organized_data[dysfunction] = item
            
            # SOURCE 1: SCORES GLOBAUX DE L'ÉQUIPE
            if lencioni_data:
                system_prompt += "\n**📊 SOURCE 1: TEAM'S OVERALL SCORES**\n"
                system_prompt += "High-level assessment results for each dysfunction:\n\n"
                
                for dysfunction in dysfunction_order:
                    if dysfunction in organized_data:
                        item = organized_data[dysfunction]
                        score = item.get('score', 0)
                        level = item.get('level', 'Unknown')
                        summary = item.get('summary', '')
                        system_prompt += f"• **{dysfunction}**: {score}/5.0 ({level})\n"
                        if summary:
                            system_prompt += f"  Summary: {summary}\n"
                system_prompt += "\n"
            
            # SOURCE 2: QUESTIONS DÉTAILLÉES (si dysfonction spécifique)
            if lencioni_details and dysfunction_focus:
                system_prompt += "**📋 SOURCE 2: DETAILED ASSESSMENT QUESTIONS**\n"
                system_prompt += "Specific questions and individual scores for mentioned dysfunctions:\n"
                
                # Organiser les détails par dysfonction
                details_by_dysfunction = {}
                for detail in lencioni_details:
                    dysfunction = detail.get('dysfunction', '').title()
                    if dysfunction not in details_by_dysfunction:
                        details_by_dysfunction[dysfunction] = []
                    details_by_dysfunction[dysfunction].append(detail)
                
                # Afficher les détails pour chaque dysfonction mentionnée
                target_dysfunctions = dysfunction_focus if dysfunction_focus else dysfunctions_mentioned
                for dysfunction in target_dysfunctions:
                    dysfunction_title = dysfunction.title()
                    if dysfunction_title in details_by_dysfunction:
                        system_prompt += f"\n**{dysfunction_title} Questions:**\n"
                        
                        # Afficher chaque question avec son score
                        for detail in details_by_dysfunction[dysfunction_title]:
                            question = detail.get('question', '')
                            score = detail.get('score', 'N/A')
                            level = detail.get('level', 'Unknown')
                            system_prompt += f"• {question}\n"
                            system_prompt += f"  → Score: {score}/5.0 ({level})\n"
                        system_prompt += "\n"
            
            # Séparer les différents types de contenu
            lencioni_overview = [item for item in search_results["insight_blend_content"] if item.get("type") == "lencioni_overview"]
            lencioni_recommendations = [item for item in search_results["insight_blend_content"] if item.get("type") == "lencioni_recommendation"]
            team_recommendations = [item for item in search_results["insight_blend_content"] if item.get("type") == "team_recommendation"]
            
            # Si overview disponible (pas de dysfunction spécifique mentionnée) OU si aucune dysfonction identifiée
            if lencioni_overview or (lencioni_data and not lencioni_recommendations and not team_recommendations):
                system_prompt += "\n**Lencioni Five Dysfunctions Overview:**\n"
                if lencioni_overview:
                    for item in lencioni_overview:
                        system_prompt += f"📊 {item.get('content', '')}\n"
                else:
                    # Ajouter un overview général quand aucune dysfonction spécifique n'est mentionnée
                    system_prompt += "📊 The Five Dysfunctions model provides a framework for understanding team effectiveness through five interconnected levels: Trust (foundation), Conflict (productive debate), Commitment (buy-in), Accountability (peer responsibility), and Results (collective focus).\n"
            
            # SOURCE 3: RECOMMANDATIONS THÉORIQUES LENCIONI
            if lencioni_recommendations:
                system_prompt += "\n**📚 SOURCE 3: LENCIONI FRAMEWORK BEST PRACTICES**\n"
                system_prompt += "Theoretical recommendations from Lencioni's model:\n\n"
                for item in lencioni_recommendations:
                    dysfunction = item.get("dysfunction", "Unknown")
                    system_prompt += f"**{dysfunction}:**\n{item.get('content', '')}\n\n"
            
            # SOURCE 4: INSIGHTS SPÉCIFIQUES DE L'ÉQUIPE (Culture Questions)
            if team_recommendations:
                system_prompt += "**👥 SOURCE 4: YOUR TEAM'S CULTURE ASSESSMENT**\n"
                system_prompt += "Direct feedback from your team members:\n"
                
                # Organiser les insights par dysfonction
                insights_by_dysfunction = {}
                for item in team_recommendations:
                    dysfunction = item.get("dysfunction", "Unknown")
                    if dysfunction not in insights_by_dysfunction:
                        insights_by_dysfunction[dysfunction] = []
                    insights_by_dysfunction[dysfunction].append(item)
                
                # Afficher selon l'ordre pyramidal avec les bonnes questions
                for dysfunction in dysfunction_order:
                    if dysfunction in insights_by_dysfunction:
                        items = insights_by_dysfunction[dysfunction]
                        
                        # Titre spécifique selon la dysfonction avec la bonne question
                        if dysfunction == "Trust":
                            system_prompt += f"\n**{dysfunction}: IDENTIFY SPECIFIC AREAS TO BUILD MORE TRUST**\n"
                        elif dysfunction == "Conflict":
                            system_prompt += f"\n**{dysfunction}: TEAM MEMBERS WERE ASKED WHETHER CERTAIN BEHAVIORS OR ACTIONS ARE ACCEPTABLE WHILE ENGAGING IN CONFLICT AND HOW MANY OF YOU DISPLAY THEM AT WORK**\n"
                        elif dysfunction == "Commitment":
                            system_prompt += f"\n**{dysfunction}: WHAT PREVENTS TEAM MEMBERS FROM COMMITTING TO DECISIONS?**\n"
                        elif dysfunction == "Accountability":
                            system_prompt += f"\n**{dysfunction}: WHAT WOULD IMPROVE YOUR TEAM'S ABILITY TO HOLD ONE ANOTHER ACCOUNTABLE?**\n"
                        elif dysfunction == "Results":
                            system_prompt += f"\n**{dysfunction}: WHAT IS NEEDED TO FOCUS ON RESULTS?**\n"
                        else:
                            system_prompt += f"\n**{dysfunction} - Team Insights:**\n"
                        
                        # Afficher les items
                        for item in items:
                            content = item.get('content', '')
                            system_prompt += f"• {content}\n"
                system_prompt += "\n"
            
            # INSTRUCTIONS DE SYNTHÈSE
            system_prompt += """
**🎯 SYNTHESIS INSTRUCTIONS - HOW TO COMBINE THE 4 SOURCES:**

You have access to 4 distinct sources of information:
1. **SOURCE 1 - Overall Scores**: Shows the big picture of team health
2. **SOURCE 2 - Detailed Questions**: Reveals specific behaviors and patterns 
3. **SOURCE 3 - Lencioni Framework**: Provides theoretical understanding
4. **SOURCE 4 - Team Culture**: Shows what team members themselves identify as priorities

**HOW TO WEAVE THESE SOURCES INTO FLUID ADVICE:**

📊 **DIAGNOSIS - Start with the Big Picture**: 
- Begin with overall dysfunction scores to identify priority areas
- Seamlessly integrate specific question scores to explain WHY the overall score is what it is
- Example: "Your Accountability score of 2.8/5.0 reflects specific challenges, particularly with team members calling out deficiencies (scoring only 2.5/5.0)"

🔍 **ROOT CAUSE CONNECTION - Show the Patterns**:
- Naturally connect low-scoring behaviors with what the team identified as priorities
- Make it clear these aren't separate issues - they're connected
- Example: "This challenge with giving feedback isn't just reflected in your assessment scores - 9 team members specifically identified this as an area needing improvement"

📚 **THEORETICAL INSIGHT - Explain the Why**:
- Weave in Lencioni's framework to explain why these patterns matter
- Don't quote theory in isolation - make it relevant to their specific situation
- Connect the framework to both their scores AND their team priorities

🎯 **ACTIONABLE FOCUS - Highlight Team Priorities**:
- **EMPHASIZE what the team members themselves voted as important** - these are their priorities!
- Show how their recommendations align with (or sometimes contrast with) their scores
- Example: "Importantly, 11 team members identified 'reviewing progress in meetings' as crucial for accountability - this represents your team's collective wisdom about what will work"

**WRITING STYLE - ATTRIBUTE SOURCES CLEARLY**:
✓ NEVER write "SOURCE 1 shows..." but DO make attribution clear
✓ Assessment scores: "Your team scored X/5.0 on..." 
✓ Detailed questions: "Looking at specific behaviors, the assessment shows..."
✓ Team insights: "Your team members identified..." "8 team members voted for..." "Team members specifically mentioned..."
✓ Lencioni theory: "According to Lencioni's framework..." "The research shows..."
✓ ALWAYS make it clear when something comes FROM the team members vs. FROM the assessment vs. FROM theory
✓ Use phrases like: "This assessment finding is reinforced by your team members, who specifically identified..."

**DISTINGUISH DATA SOURCES CLEARLY**:
- **Assessment Scores**: "Your team scored X/5.0 on..."
- **Team Culture Questions**: "11 team members identified..." "Your team voted for..."  
- **Behavior Assessments**: "Team members admit to..." "3 members acknowledged..."
- **Lencioni Content**: Use context-appropriate attribution based on the type of question

**RESPONSE STRUCTURE - HIGHLIGHT EACH ELEMENT**:
1. **Diagnosis**: Weave together scores + specific behaviors + team input with clear attribution
2. **Key Insights**: Call out important patterns where assessment data + team feedback align or differ
3. **Clear Recommendations**: 
   - Headers like "Here are my key recommendations:" 
   - **Number based on the data**: If 15 members identified something vs 3 for something else, prioritize accordingly
   - Focus on high-impact areas: lowest scores + highest team member votes + pyramid logic
   - Each recommendation should reference specific numbers: "Given that 11 members identified X and your score is Y..."
4. **Follow-up Questions**: 
   - Number and focus based on their specific patterns and opportunities
   - When suggesting other dysfunctions, reference their actual scores in pyramid context 

**ZEST LIBRARY & DEEPER SUPPORT**:
- Reference Zest Library tools naturally when giving recommendations
- Mention Jean-Pierre Aerts when appropriate for personalized coaching, complex situations, or when additional expertise would be valuable
- Use your judgment - be helpful in suggesting next steps
"""
    
    # INSTRUCTIONS DE PYRAMIDE ET LOGIQUE STRICTE - APPLIQUÉES À TOUS LES INTENTS
    if lencioni_data:
        system_prompt += f"""

🏗️ **CRITICAL: LENCIONI PYRAMID LOGIC - YOU MUST FOLLOW THIS STRICTLY:**

**Pyramid Foundation Rules:**
1. **Trust is the foundation** - Must be strong before other dysfunctions can be effectively addressed
2. **Sequential dependency** - Higher levels depend on lower levels being solid
3. **Diagnostic order**: Trust → Conflict → Commitment → Accountability → Results

**Recommendation Logic (MANDATORY):**
- **If Trust score ≤ 3.0**: Focus PRIMARILY on Trust-building. Don't recommend higher-level work until Trust improves.
- **If Trust > 3.0 but Conflict ≤ 3.0**: Focus on productive Conflict while maintaining Trust gains.
- **If Trust & Conflict > 3.0 but Commitment ≤ 3.0**: Work on Commitment with foundation of Trust & Conflict.
- **Only recommend Accountability when Trust, Conflict & Commitment are solid (>3.0)**
- **Only focus on Results when all foundational elements are strong**

**Context Instructions:**
- **Always acknowledge the current team scores** - don't ask for information you already have
- **Use the team insights contextually** - reference specific insights that align with the user's question
- **Provide actionable, specific recommendations** based on both the framework and the team's specific situation
- **Follow the pyramid logic** - don't skip steps or recommend advanced concepts when foundations are weak

**Response Style:**
- Conversational and supportive
- Reference the team's actual situation (don't be generic)
- Keep responses comprehensive but not overwhelming (aim for 1500-2000 characters)
- Structure your response with clear sections if appropriate
- **Don't create or name exercises**: Simply say "practical tools and exercises are available in the Zest Library"
- **End your response appropriately**: For more specific questions or personalized coaching, suggest contacting Jean-Pierre Aerts at jean-pierre.aerts@zestforleaders.com"""

    # LIGNE DE FINITION AVEC INSTRUCTIONS SPÉCIFIQUES PAR INTENT TYPE
    if intent_type == "INSIGHT_BLEND":
        # Générer les instructions de coaching contextuelles pour INSIGHT_BLEND
        coaching_instructions = generate_contextual_coaching_instructions(lencioni_data, intent_type, dysfunctions_mentioned)
        
        system_prompt += f"""

**🎯 INSIGHT_BLEND TASK:** Provide actionable coaching by combining:
1. Assessment scores and detailed behavioral data
2. Team member priorities and feedback  
3. Lencioni's practical recommendations for improvement
4. Pyramid hierarchy logic

**PRIORITIZATION LOGIC:**
- **High impact = Low scores + High team votes + Pyramid readiness**
- If 11 members voted for something and 3 for something else, weight accordingly
- If a behavior scored 2.1/5.0 vs another at 4.2/5.0, focus on the 2.1 area
- Respect pyramid: don't recommend Accountability fixes if Trust is 2.5/5.0

{coaching_instructions}

**For Lencioni content in this context**: These are practical recommendations:
- "Lencioni recommends..." "Best practices suggest..." "The framework advises..."
- Always connect to their specific data: "Given your score of X and that Y members identified this..."
- End with contextual coaching questions that reference their priority areas and specific scores"""
    
    elif intent_type == "REPORT_LOOKUP":
        # Générer les instructions de coaching contextuelles pour REPORT_LOOKUP
        coaching_instructions = generate_contextual_coaching_instructions(lencioni_data, intent_type, dysfunctions_mentioned)
        
        system_prompt += f"""

**🎯 REPORT_LOOKUP TASK:** Present comprehensive assessment results with theoretical context.

{coaching_instructions}

**For Lencioni content in this context**: This is explanatory theory, so use phrases like:
- "According to Lencioni's model..." "The theory explains..." "Research shows..."
- Focus on helping understand what the scores mean and the framework behind them
- End with contextual coaching questions that help them interpret their specific results"""
    
    elif intent_type in ["LENCIONI_GENERAL_KNOWLEDGE", "GENERAL_KNOWLEDGE"]:
        # Récupérer l'overview et les instructions de coaching contextuelles
        team_overview = ""
        
        if lencioni_data:
            team_overview = "Your team assessment shows:\n"
            for item in lencioni_data:
                dysfunction = item.get('dysfunction', '').title()
                score = item.get('score', 0)
                level = item.get('level', 'Unknown')
                team_overview += f"• {dysfunction}: {score}/5.0 ({level})\n"
        else:
            team_overview = "No team assessment data available - providing general framework information."
        
        # Générer les instructions de coaching contextuelles
        coaching_instructions = generate_contextual_coaching_instructions(lencioni_data, intent_type, dysfunctions_mentioned)
        
        general_approach = """
**GENERAL_KNOWLEDGE APPROACH:**
1. **FOCUS ON THEORY**: Explain Lencioni's model using the overview content
2. **ACKNOWLEDGE THEIR DATA**: Briefly mention they have assessment scores available
3. **CONTEXTUAL COACHING**: Use the coaching instructions below to generate relevant follow-up questions"""
            
        system_prompt += f"""

**🎯 GENERAL_KNOWLEDGE TASK:** Explain Lencioni's framework and concepts.

**TEAM CONTEXT - OVERVIEW:**
{team_overview}

{general_approach}

{coaching_instructions}

**For Lencioni content in this context**: Theoretical explanation with contextual coaching:
- "Lencioni defines..." "The model states..." "According to the research..."
- Focus on educating about the framework and pyramid logic
- When team data exists, briefly acknowledge it but don't analyze scores in detail
- Use the contextual coaching instructions above to generate relevant follow-up questions
- End with coaching questions that reference their specific situation and priority areas"""

    return system_prompt


def build_mbti_prompt(state: WorkflowState) -> str:
    """Construit le prompt MBTI original avec recherches vectorielles complètes"""
    # Récupérer la question utilisateur
    user_question = state.get('user_message', '') or (state.get('messages', [{}])[-1].get('content', '') if state.get('messages') else '')
    
    # Ajouter l'historique de conversation pour éviter les répétitions
    conversation_history = []
    for msg in state.get('messages', [])[-3:]:  # 3 derniers messages pour contexte
        if hasattr(msg, 'type') and hasattr(msg, 'content'):
            role = "User" if msg.type == 'human' else "Assistant"
            conversation_history.append(f"{role}: {msg.content}")
        elif isinstance(msg, dict):
            role = "User" if msg.get('role') == 'user' else "Assistant"
            conversation_history.append(f"{role}: {msg.get('content', '')}")
    
    history_text = "\n".join(conversation_history) if conversation_history else "No previous conversation"
    
    # Construire le prompt système avec tous les contextes
    system_prompt = f""" 
ROLE: 
You are ZEST COMPANION, an expert leadership mentor & MBTI coach. You coach with concise yet comprehensive answers, prioritizing specificity and actionable guidance to help the user become a more impactful leader. 

GUARDRAILS: 
- Use ONLY the temperament description and ZEST database search results provided. 
- Do not use MBTI nicknames or external MBTI knowledge unless in context. 
- SCOPE: Focus on MBTI leadership coaching, team dynamics, and professional development only
- REDIRECT: For mental health, family issues, clinical conditions, or therapy needs - recommend consulting appropriate professional resources
- For more specific questions or personalized coaching, suggest contacting jean-pierre.aerts@zestforleaders.com.

USER QUESTION: "{user_question}"

RECENT CONVERSATION HISTORY:
{history_text}

User MBTI Profile: {state.get('user_mbti', 'Unknown')}
User Temperament: {state.get('user_temperament', 'Unknown')}
Temperament Description: {state.get('temperament_description', 'Unknown')}

Context from vector search:"""
    
    # Organisation par profils MBTI au lieu d'outils séparés
    user_mbti = state.get('user_mbti', 'Unknown')
    mbti_analysis = state.get('mbti_analysis', {})
    other_profiles_str = mbti_analysis.get('other_mbti_profiles', '')
    other_profiles_list = [p.strip() for p in other_profiles_str.split(',') if p.strip()] if other_profiles_str else []
    
    # 🔄 DÉDUPLICATION GLOBALE ROBUSTE des tempéraments
    def deduplicate_temperaments(temperament_list, target_filter=None):
        """Déduplique les tempéraments par temperament_info de façon robuste"""
        if not temperament_list:
            return []
        
        # Filtrer par target si spécifié
        if target_filter:
            filtered_list = [item for item in temperament_list 
                           if item.get('target') == target_filter or target_filter in item.get('targets', [])]
        else:
            filtered_list = temperament_list
        
        # Déduplication basée sur temperament_info + facet complète
        seen_keys = set()
        deduplicated = []
        
        for item in filtered_list:
            temperament_info = item.get('temperament_info', '')
            content_snippet = item.get('content', '')[:50]  # Premier 50 chars comme clé
            dedup_key = f"{temperament_info}::{content_snippet}"
            
            if dedup_key and dedup_key not in seen_keys:
                seen_keys.add(dedup_key)
                deduplicated.append(item)
        
        return deduplicated
    
    # === SECTION UTILISATEUR ===
    if user_mbti != 'Unknown' and (state.get("personalized_content") or state.get("generic_content") or state.get("temperament_content")):
        system_prompt += f"\n\n=== USER PROFILE INSIGHTS: {user_mbti} ===\n"
        
        # 1. Tempérament de l'utilisateur avec déduplication robuste
        all_temperaments = state.get("temperament_content", [])
        logger.info(f"🔍 DEBUG FINAL: temperament_content = {len(all_temperaments)} items")
        
        # Debug: afficher tous les tempéraments avant filtrage
        for i, item in enumerate(all_temperaments):
            target = item.get('target', 'NO_TARGET')
            temp_info = item.get('temperament_info', 'NO_INFO')
            logger.info(f"   [{i+1}] target='{target}', temperament_info='{temp_info}'")
        
        user_temperament_content = deduplicate_temperaments(all_temperaments, 'user')
        
        logger.info(f"🔍 DEBUG FINAL: user_temperament_content = {len(user_temperament_content)} items (deduplicated)")
        if user_temperament_content:
            temperament_name = user_temperament_content[0].get('temperament_info', '').split('/')[0] if user_temperament_content else 'Unknown'
            system_prompt += f"\n--- 1. Temperament Foundation ({temperament_name}) ---\n"
            for i, item in enumerate(user_temperament_content, 1):
                facet = item.get('temperament_info', 'Unknown').split('/')[-1] if '/' in item.get('temperament_info', '') else 'Unknown'
                system_prompt += f"[{temperament_name} - {facet}]\n{item['content']}\n\n"
        
        # 2. PROFIL MBTI COMPLET - Synthèse Tool A + Tool B
        if state.get("personalized_content") or state.get("generic_content"):
            system_prompt += f"--- 2. Your {user_mbti} Profile Insights ---\n"
            
            # Combiner Tool A (expériences personnelles) et Tool B (documentation générale)
            if state.get("personalized_content"):
                system_prompt += "**Personal Experiences & Individual Patterns:**\n"
                for i, item in enumerate(state["personalized_content"], 1):
                    system_prompt += f"{item['content']}\n\n"
            
            if state.get("generic_content"):
                system_prompt += f"**{user_mbti} Type Characteristics & Development Areas:**\n"
                for i, item in enumerate(state["generic_content"], 1):
                    system_prompt += f"{item['content']}\n\n"
    
    # === SECTIONS AUTRES PROFILS ===
    if other_profiles_list and (state.get("others_content") or state.get("temperament_content")):
        # Grouper others_content par profil MBTI
        others_by_profile = {}
        for item in state.get("others_content", []):
            profile = item.get('metadata', {}).get('target_mbti', 'Unknown')
            if profile not in others_by_profile:
                others_by_profile[profile] = []
            others_by_profile[profile].append(item)
        
        # Organiser par profil MBTI
        for profile in other_profiles_list:
            if profile == user_mbti:  # Skip si même que l'utilisateur
                continue
                
            profile_others_content = others_by_profile.get(profile, [])
            
            # Calculer le tempérament de ce profil MBTI (codes et noms complets)
            MBTI_TO_TEMPERAMENT_CODE = {'INTJ': 'NT', 'INTP': 'NT', 'ENTJ': 'NT', 'ENTP': 'NT',
                                       'INFJ': 'NF', 'INFP': 'NF', 'ENFJ': 'NF', 'ENFP': 'NF',
                                       'ISTJ': 'SJ', 'ISFJ': 'SJ', 'ESTJ': 'SJ', 'ESFJ': 'SJ',
                                       'ISTP': 'SP', 'ISFP': 'SP', 'ESTP': 'SP', 'ESFP': 'SP'}
            TEMPERAMENT_CODE_TO_NAME = {'NT': 'Architect', 'NF': 'Catalyst', 'SJ': 'Guardian', 'SP': 'Commando'}
            
            profile_temperament_code = MBTI_TO_TEMPERAMENT_CODE.get(profile, 'Unknown')
            profile_temperament_name = TEMPERAMENT_CODE_TO_NAME.get(profile_temperament_code, 'Unknown')
            
            logger.info(f"   🔍 {profile}: temperament_code='{profile_temperament_code}', temperament_name='{profile_temperament_name}'")
            
            # Filtrer temperament_content pour ce profil spécifique (recherche par code)
            all_profile_temperament_content = [item for item in state.get("temperament_content", []) 
                                              if item.get('target') == 'others' and 
                                              item.get('temperament_info', '').startswith(profile_temperament_code)]
            
            # Déduplication robuste pour ce profil
            profile_temperament_content = deduplicate_temperaments(all_profile_temperament_content)
            
            logger.info(f"   📊 {profile} temperament content: {len(all_profile_temperament_content)} → {len(profile_temperament_content)} (after dedup)")
            
            if profile_others_content or profile_temperament_content:
                system_prompt += f"\n\n=== OTHER PROFILE INSIGHTS: {profile} ===\n"
                
                # 1. Tempérament de ce profil
                if profile_temperament_content:
                    system_prompt += f"\n--- 1. Temperament Foundation ({profile_temperament_name}) ---\n"
                    for i, item in enumerate(profile_temperament_content, 1):
                        facet = item.get('temperament_info', 'Unknown').split('/')[-1] if '/' in item.get('temperament_info', '') else 'Unknown'
                        system_prompt += f"[{profile_temperament_name} - {facet}]\n{item['content']}\n\n"
                
                # 2. Documentation du type (Tool C)
                if profile_others_content:
                    system_prompt += f"--- 2. {profile} Type Documentation ---\n"
                    for i, item in enumerate(profile_others_content, 1):
                        system_prompt += f"[{profile} Context {i}]\n{item['content']}\n\n"
    
    # === SECTION GÉNÉRALE (si pas de profils spécifiques) ===
    if state.get("general_content"):
        system_prompt += f"\n\n=== GENERAL MBTI KNOWLEDGE ===\n"
        system_prompt += f"The following content provides general MBTI theory and concepts:\n\n"
        for i, item in enumerate(state["general_content"], 1):
            system_prompt += f"[General MBTI Context {i}]\n{item['content']}\n\n"

    # Détecter si c'est une salutation ou pas de recherche nécessaire
    if state.get("no_search_needed"):
        # Cas spécial: salutation ou question simple
        analysis = state.get("mbti_analysis", {})
        question_type = analysis.get("question_type", "")
        
        if question_type == "GREETING_SMALL_TALK":
            system_prompt = f"""You are ZEST COMPANION, a friendly MBTI leadership coach.

User said: "{user_question}"

Respond warmly and briefly. If they're greeting you, greet them back and ask how you can help with their leadership development.
Keep it under 50 words, be personable and engaging.

User's MBTI: {state.get('user_mbti', 'Unknown')}
User's Name: {state.get('user_name', 'Friend')}"""
        else:
            system_prompt += "\n\nNo search results available. Ask the user to provide more specific information about what they'd like to know."
    
    # Instructions dynamiques basées sur les tools exécutés
    available_tools = []
    if state.get("personalized_content"):
        available_tools.append("Tool A (Participants Content)")
    if state.get("generic_content"):
        available_tools.append("Tool B (Documents Content)")
    if state.get("others_content"):
        available_tools.append("Tool C (Others Collection)")
    if state.get("general_content"):
        available_tools.append("Tool D (General MBTI)")
    if state.get("temperament_content"):
        available_tools.append("Tool T (Temperament Insights)")
    
    if available_tools and not state.get("no_search_needed"):
        # Instructions dynamiques basées sur les tools disponibles
        tool_instructions = []
        
        if state.get("personalized_content"):
            tool_instructions.append("- Tool A (Personal): Use for specific insights about the user's individual experiences and challenges")
        
        if state.get("generic_content"):
            tool_instructions.append(f"- Tool B (User Type): Use for general {state.get('user_mbti', '')} characteristics and patterns")
        
        if state.get("others_content"):
            # Identifier les profils spécifiques
            other_profiles = set()
            for item in state["others_content"]:
                target_mbti = item.get("metadata", {}).get("target_mbti", "")
                if target_mbti and target_mbti != "Unknown":
                    other_profiles.add(target_mbti)
            
            if other_profiles:
                profiles_str = ", ".join(sorted(other_profiles))
                tool_instructions.append(f"- Tool C (Other Types): Use for {profiles_str} characteristics and dynamics")
        
        if state.get("general_content"):
            tool_instructions.append("- Tool D (General): Use for MBTI theory and universal concepts")
        
        tools_instructions_text = "\n".join(tool_instructions)
        
        system_prompt += f"""

CRITICAL INSTRUCTIONS:
1. User's temperament is "{state.get('user_temperament', 'Unknown')}" (MBTI: {state.get('user_mbti', 'Unknown')})
2. Use ONLY information from the temperament description and search results provided above
3. NEVER use MBTI nicknames unless they appear in the provided context

HOW TO USE EACH CONTEXT:
{tools_instructions_text}

RESPONSE STRUCTURE:
- If conversation history is empty: Mention the {state.get('user_temperament', '')} temperament once
- If continuing conversation: DO NOT repeat previous advice or temperament mentions
- MUST synthesize insights from ALL available contexts above
- CRITICAL LAYERED APPROACH - Always structure your response in this order:
  
  **1. TEMPERAMENT FOUNDATION FIRST** (if available):
  * Start with broad temperament patterns for quick understanding
  * Clearly label this section (e.g., "Your Commando temperament...")
  * Highlight key temperament characteristics that apply to the situation
  * Use temperament insights to set the context and general approach
  
  **2. THEN DETAILED MBTI PROFILE INSIGHTS** (clearly distinguished):
  * Start a new section with clear transition (e.g., "More specifically, your ISFP profile...")
  * Build upon the temperament foundation with specific 4-letter type patterns
  * Create ONE unified analysis combining all available insights without referencing tools
  * Provide detailed, actionable insights for personal development and relationships
  * Include specific examples and concrete steps based on the full type
  
  **3. OTHER TYPES INFORMATION** (if relevant):
  * Compare/contrast with other personalities using their temperament AND specific types
  
- Always progress from GENERAL (temperament) → SPECIFIC (full MBTI profile)
- CRITICAL: In the full MBTI profile section, create ONE concise unified analysis - never reference or separate tool sources
- If search results lack detail, acknowledge this: "Based on available information..."
- End with 2-3 specific follow-up questions to guide deeper exploration
- Aim for 150-250 words; keep responses concise and focused
- Encourage users to ask for deeper analysis on specific points rather than giving everything at once; focus on insights and refer to Zest Library for implementation

COACHING APPROACH:
- Extract UNIQUE insights from each context section - don't generalize
- VARY your language between responses - avoid repeating the same phrases  
- For comparisons: Clearly contrast the different types using their specific contexts
- For personal development: Seamlessly blend personal experiences with type characteristics
- End with questions inviting deeper exploration:
  * "Souhaitez-vous que je détaille leur approche du [aspect spécifique] ?"
  * "Vous intéresse-t-il d'explorer des exemples concrets de [situation] ?"
  * "Voulez-vous approfondir leur différence dans [domaine précis] ?"
- When relevant, cover: work style, communication, team, decision-making, leadership, conflict, stress, change, and type dynamics

FORBIDDEN: Do not use any MBTI knowledge not in the provided context."""
    else:
        system_prompt += f"\n\nNo vector search results available. Provide a response indicating that you need more specific information to give a personalized answer."
    
    return system_prompt


def build_leadership_prompt(state: WorkflowState) -> str:
    """
    Constructeur de prompt spécialisé pour le leadership (A4_LeadershipStyle)
    Adapte le prompt selon le question_type identifié par l'intent analysis
    """
    logger.info("🎯 Building specialized leadership prompt")
    
    # Get question type from intent analysis
    question_type = state.get('question_type', 'general_leadership')
    detected_styles = state.get('detected_styles', [])
    leadership_resources = state.get('leadership_resources', '')
    
    # Get user information
    user_name = state.get('user_name', 'User')
    user_mbti = state.get('user_mbti', 'Unknown')
    user_temperament = state.get('user_temperament', 'Unknown')
    
    # Get user question
    user_question = state.get('user_message', '') or (
        state.get('messages', [{}])[-1].get('content', '') if state.get('messages') else ''
    )
    
    # Build conversation history
    conversation_history = []
    for msg in state.get('messages', [])[-3:]:  # Last 3 messages for context
        if hasattr(msg, 'type') and hasattr(msg, 'content'):
            role = "User" if msg.type == 'human' else "Assistant"
            conversation_history.append(f"{role}: {msg.content}")
        elif isinstance(msg, dict):
            role = "User" if msg.get('role') == 'user' else "Assistant"
            conversation_history.append(f"{role}: {msg.get('content', '')}")
    
    history_text = "\n".join(conversation_history) if conversation_history else "No previous conversation"
    
    # Base leadership context
    base_context = f"""
ROLE: 
You are ZEST COMPANION, an expert leadership mentor specialized in Daniel Goleman's 6 Leadership Styles framework.

USER CONTEXT:
- Name: {user_name}
- MBTI Type: {user_mbti}
- Temperament: {user_temperament}
- Question Type: {question_type}
- Detected Styles: {detected_styles if detected_styles else 'None specifically mentioned'}

CURRENT QUESTION: {user_question}

RECENT CONVERSATION HISTORY:
{history_text}

LEADERSHIP KNOWLEDGE BASE:
{leadership_resources}

ZEST COMPANION GUIDANCE:
- You are the ZEST COMPANION coach
- For questions specifically about MBTI, redirect to the chatbot in "Better understand and manage myself" section titled "Explore my personality with MBTI"  
- Contact your academic program director at jean-pierre.aerts@zestforleaders.com for any advanced questions

GUARDRAILS: 
- Use ONLY ZEST database search results provided
- SCOPE: Focus EXCLUSIVELY on WORKPLACE leadership coaching, professional team dynamics, and business/organizational development only
- REDIRECT: For family leadership, parenting, personal relationships, mental health, clinical conditions, or therapy needs - redirect to appropriate professional resources
- Leadership questions MUST be about professional/business contexts only - no family or personal relationship leadership advice
- For more specific questions or personalized coaching, suggest contacting jean-pierre.aerts@zestforleaders.com

INSTRUCTIONS:
"""
    
    # Specific instructions based on question type
    if question_type == 'personal_style':
        
        temperament_guidance = ""
        if user_temperament:
            temperament_guidance = f"""
6. **Temperament Insights**: Consider how the {user_temperament} temperament may naturally align with certain leadership approaches - use this as gentle guidance, not rigid prescription
"""
        
        specific_instructions = f"""
Focus on SELF-ASSESSMENT and IDENTIFICATION of natural leadership tendencies:

1. **Foundation Building**: Use the research foundations to explain the conceptual framework
2. **Self-Evaluation Guide**: Apply integration principles for personal style assessment  
3. **Style Identification**: Help user recognize their dominant and secondary styles
4. **Neutral Approach**: Present all styles as valuable, avoid prescriptive recommendations
5. **Assessment Questions**: Guide self-reflection rather than direct diagnosis{temperament_guidance}

DO NOT include MBTI profile or temperament analysis in your response.
Respond based purely on Goleman's leadership framework and self-assessment principles.

ENGAGEMENT & FOLLOW-UP:
- End your response with 2-3 targeted follow-up questions to help the user explore their style further
- Examples: "Would you like to explore how your natural [style] tendency shows up in challenging situations?"
- "Are you curious about developing complementary styles to broaden your leadership range?"
- "Should we examine specific scenarios where you could practice your emerging [style] skills?"
- Focus on practical next steps and deeper self-discovery

Your response should help the user understand how to identify their own leadership style using Goleman's framework.
"""
    
    elif question_type == 'comparative':
        specific_instructions = f"""
Focus on BALANCED COMPARISON between leadership styles:

1. **Equal Treatment**: Give equal weight and attention to each style being compared
2. **Structured Comparison**: Use the comparison matrix provided in the knowledge base
3. **Climate Impact Focus**: Emphasize the different climate impacts (positive/negative) of each style
4. **Situational Appropriateness**: Explain WHEN each style is most effective vs problematic
5. **No Style Bias**: Avoid favoring one style over others - present objective analysis
6. **Integration Opportunities**: Explain how styles can complement each other
7. **Practical Distinctions**: Highlight key behavioral differences that matter in practice

COMPARISON STRUCTURE:
- Start with the comparison matrix overview (mottos, climate impacts)
- Then dive into specific differences in:
  * When to use each style
  * Required emotional intelligence competencies
  * Potential pitfalls and benefits
  * Real-world scenarios where each excels
- End with guidance on style flexibility and integration

CRITICAL: Ensure balanced coverage - if comparing 2 styles, split content 50/50. If comparing 3 styles, 33/33/33, etc.
DO NOT let one style dominate the narrative or examples.

ADDITIONAL CONTEXT: 
- If "Additional Context" section is provided, use it as supplementary information only
- This contextual information (similarity > 0.6) provides highly relevant background
- Integrate these insights naturally into your comparison, but don't let them overshadow the main Goleman comparison
- Treat this as supporting evidence rather than primary content

ENGAGEMENT & FOLLOW-UP:
- End your response with 2-3 targeted follow-up questions to help the user dive deeper
- Examples: "Would you like me to explore how [specific style] performs in crisis situations?"
- "Are you curious about the emotional intelligence competencies needed for [style A] vs [style B]?"
- "Should we examine when to transition between these styles in real scenarios?"
- Tailor questions to the specific styles they're comparing and their apparent interests

Styles being compared: {detected_styles if detected_styles else 'To be determined from question'}
"""
    
    elif question_type == 'implementation':
        temperament_guidance = ""
        if user_temperament:
            temperament_guidance = f"""
- **Temperament Insights**: Consider how the {user_temperament} temperament may naturally approach implementation - tailor your recommendations to leverage their natural strengths while addressing potential blind spots
- Use temperament patterns as gentle guidance for suggesting the most effective implementation strategies for their personality
- If temperament leadership style data is available, include a section "🧬 **Your Natural Leadership Tendencies**" to show how their temperament influences their implementation approach
"""
        
        specific_instructions = f"""
Focus on PRACTICAL IMPLEMENTATION with priority content:

**CONTENT PRIORITIES:**
- MASSIVELY PRIORITIZE: content_type = "practical_template" + goleman_section = "partial_guide"
- COMPLEMENT (if style mentioned): content_type = "style_complete" for underlying emotional intelligence competencies
- Logical flow: practical → conceptual understanding → emotional intelligence foundations

**RESPONSE SEQUENCING:**
Present the "how to do it right now" immediately, then deepen with "why it works" conceptually:

1. **Immediate Implementation** (from practical_template + partial_guide):
   - Step-by-step actionable templates
   - Concrete phrases and behaviors to use
   - Specific situations where to apply

2. **Underlying Mechanisms** (from style_complete, if style mentioned):
   - Why this approach works psychologically
   - Emotional intelligence competencies involved
   - Connection to research foundations

**RESPONSE FORMAT:**
Write in natural, flowing language that seamlessly includes:
- Immediate practical steps (80% of content)
- Conceptual understanding of mechanisms (20% of content)
- Success indicators and common pitfalls
- Progressive difficulty: easy wins → complex applications{temperament_guidance}

Target style for implementation: {detected_styles[0] if detected_styles else 'Not specified - may need clarification'}
"""
    
    elif question_type == 'situational':
        temperament_guidance = ""
        if user_temperament:
            temperament_guidance = f"""
- **Temperament Insights**: Consider how the {user_temperament} temperament may naturally navigate situations - recommend styles and approaches that align with their natural strengths while stretching them appropriately  
- Use temperament patterns to suggest the most effective situational response strategies for their personality type
- If temperament leadership style data is available, include a section "🧬 **Your Natural Leadership Tendencies**" to show how their temperament naturally approaches this type of situation
"""
        
        specific_instructions = f"""
Focus on TARGETED & ACTIONABLE LEADERSHIP:

**PRIMARY APPROACH:** 
- Select the MOST appropriate style first (maximum 2 styles)
- Provide concrete, actionable steps the user can take immediately
- Focus on practical implementation over theoretical combinations

**RESPONSE FORMAT:**
Write a natural, flowing response that includes these elements seamlessly:
- Brief context assessment (2-3 sentences)  
- Primary style recommendation with clear rationale
- 3-4 specific, actionable steps to take immediately
- Optional secondary style mention (only if essential)
- Key risk/pitfall to avoid
- How to measure success

**CONSTRAINTS:**
- Write in fluent, natural language (avoid numbered lists or rigid structure)
- Keep response focused and concise
- Prioritize actionable advice over style theory
- Avoid recommending more than 2 styles total
- Emphasize "what to do" rather than "what styles exist"
- Maintain conversational tone throughout
"""
    
    elif question_type == 'specific_style':
        specific_instructions = f"""
Focus on COMPREHENSIVE SINGLE STYLE ANALYSIS:

1. **Layered Presentation**: Present practical → analytical → theoretical
2. **Complete Picture**: Cover all aspects of this one style thoroughly
3. **Nuanced Understanding**: Include both strengths and limitations
4. **Application Range**: Show the full spectrum of where this style applies
5. **Development Path**: How to build competence in this specific style

For problematic styles (coercive/pacesetting): ALWAYS include limitations and alternatives.

Focus style: {detected_styles[0] if detected_styles else 'To be determined'}
"""
    
    else:
        # Default for other question types (general_leadership)
        specific_instructions = """
Provide comprehensive leadership guidance based on Goleman's 6 leadership styles framework.
Draw from research foundations, integration principles, and contextual insights.
Focus on practical application and evidence-based recommendations.
If the question is broad, provide an overview then suggest specific areas to explore deeper.
"""
    
    return base_context + specific_instructions




def build_general_prompt(state: WorkflowState) -> str:
    """Construit le prompt pour contenu général selon le thème spécifique"""
    theme = state.get('theme', '')
    sub_theme = state.get('sub_theme', '')
    
    logger.info(f"🎯 Building general prompt for theme={theme}, sub_theme={sub_theme}")
    
    # Détecter le type de contenu général
    if sub_theme == 'C8_Introspection':
        return build_introspection_prompt(state)
    # Futurs sujets généraux :
    # elif sub_theme == 'C1_Communication':
    #     return build_communication_prompt(state)
    # elif sub_theme == 'C2_Leadership':
    #     return build_leadership_prompt(state)
    else:
        return build_default_general_prompt(state)


def build_introspection_prompt(state: WorkflowState) -> str:
    """Prompt spécialisé pour l'introspection et développement personnel"""
    # Récupérer les informations utilisateur
    user_name = state.get('user_name', 'User')
    user_mbti = state.get('user_mbti', 'Unknown')
    user_temperament = state.get('user_temperament', 'Unknown')
    
    # Récupérer les résultats de recherche vectorielle
    general_results = state.get('general_vector_results', [])
    
    # DEBUG: Log pour comprendre pourquoi les résultats ne passent pas
    logger.info(f"🔍 DEBUG build_introspection_prompt - general_vector_results: {len(general_results)} items")
    if general_results:
        logger.info(f"   First result: {general_results[0].get('document_name', 'Unknown')[:50] if general_results else 'None'}")
    
    # Récupérer la question et l'historique
    user_message = state.get('user_message', '') or (
        state.get('messages', [{}])[-1].get('content', '') if state.get('messages') else ''
    )
    
    # Construire l'historique de conversation
    history_text = ""
    messages = state.get('messages', [])
    if len(messages) > 1:
        for msg in messages[-5:-1]:  # Derniers 5 messages (exclut le dernier qui est la question actuelle)
            if hasattr(msg, 'type') and hasattr(msg, 'content'):
                role = "User" if msg.type == "human" else "Assistant"
                history_text += f"{role}: {msg.content[:200]}...\n"
    
    # Construire le prompt système spécialisé pour l'introspection
    system_prompt = f"""You are ZEST, an expert personal development coach specializing in introspection, self-awareness, and personal growth.

Your expertise includes:
- Self-reflection and mindfulness practices
- Overcoming perfectionism and limiting beliefs
- Building healthy habits and breaking negative patterns
- Career transition and professional development
- Stress management and work-life balance
- Leadership development and authentic communication
- Personal values clarification and goal setting

CURRENT USER CONTEXT:
- Name: {user_name}
- MBTI Type: {user_mbti}
- Temperament: {user_temperament}

CURRENT QUESTION: {user_message}

RECENT CONVERSATION HISTORY:
{history_text}

RELEVANT KNOWLEDGE BASE:"""
    
    # Ajouter le contenu trouvé par recherche vectorielle
    if general_results:
        system_prompt += "\n\n📚 EXPERT CONTENT ON INTROSPECTION & PERSONAL DEVELOPMENT:\n"
        for i, result in enumerate(general_results, 1):
            content = result.get('content', '')
            document_name = result.get('document_name', 'Unknown Document')
            similarity = result.get('similarity', 0.0)
            
            # Tronquer le contenu si trop long
            if len(content) > 800:
                content = content[:800] + "..."
            
            system_prompt += f"""
{i}. Source: {document_name} (Relevance: {similarity:.2f})
   Insights: {content}
"""
    else:
        system_prompt += "\n\n⚠️ No specific content found in knowledge base for this introspection query."
    
    system_prompt += """

SPECIALIZED GUIDANCE FOR INTROSPECTION:
1. Focus on self-awareness and personal insight development
2. Provide practical exercises for reflection and growth
3. Address perfectionism, limiting beliefs, and growth mindset
4. Offer strategies for authentic leadership and communication
5. Consider the user's MBTI type in understanding their natural preferences and blind spots
6. Encourage deeper self-examination and conscious behavior change
7. Provide actionable steps for habit formation and personal transformation
8. Be empathetic to struggles with authenticity, career uncertainty, and personal development

RESPONSE APPROACH:
- Be deeply empathetic and understanding
- Offer specific, actionable advice and exercises
- Draw connections between insights and practical application
- Encourage self-compassion alongside growth
- Help identify patterns and underlying beliefs
- Provide frameworks for ongoing self-development

Your role is to guide individuals toward greater self-awareness, authentic expression, and meaningful personal growth through thoughtful introspection and practical development strategies."""
    
    return system_prompt


def build_default_general_prompt(state: WorkflowState) -> str:
    """Prompt par défaut pour autres contenus généraux"""
    # Récupérer les informations utilisateur
    user_name = state.get('user_name', 'User')
    user_mbti = state.get('user_mbti', 'Unknown')
    user_temperament = state.get('user_temperament', 'Unknown')
    
    # Récupérer les résultats de recherche vectorielle
    general_results = state.get('general_vector_results', [])
    
    # Récupérer la question
    user_message = state.get('user_message', '') or (
        state.get('messages', [{}])[-1].get('content', '') if state.get('messages') else ''
    )
    
    # Prompt générique pour autres sujets
    system_prompt = f"""You are ZEST, an expert coach providing guidance on professional and personal development.

CURRENT USER CONTEXT:
- Name: {user_name}
- MBTI Type: {user_mbti}
- Temperament: {user_temperament}

CURRENT QUESTION: {user_message}"""
    
    # Ajouter le contenu trouvé par recherche vectorielle
    if general_results:
        system_prompt += "\n\nRELEVANT CONTENT:\n"
        for result in general_results:
            content = result.get('content', '')[:500]
            system_prompt += f"- {content}...\n"
    
    system_prompt += """

Provide thoughtful, practical guidance based on the available knowledge and general coaching principles. 
Be supportive, actionable, and consider the user's personality type in your advice."""
    
    return system_prompt


def build_pcm_prompt(state: WorkflowState) -> str:
    """
    Construit le prompt pour la réponse finale PCM
    Sélectionne le bon prompt selon le flow_type
    """
    # Récupérer les informations de contexte
    flow_type = state.get('flow_type', 'general_knowledge')
    user_query = state.get('user_message', '') or (
        state.get('messages', [{}])[-1].get('content', '') if state.get('messages') else ''
    )
    
    # Sélectionner le bon prompt selon le flow_type et la question
    if flow_type == 'self_focused':
        # Détecter si c'est le premier message PCM
        messages = state.get('messages', [])
        is_first_pcm_message = len(messages) <= 1  # Seulement le message utilisateur actuel
        
        # Utiliser l'information déjà déterminée dans le state si disponible
        base_or_phase = state.get('pcm_base_or_phase')
        if not base_or_phase:
            # Fallback: analyser la query si pas d'info dans le state  
            base_or_phase = _determine_base_or_phase(user_query)
            logger.info(f"🎯 PCM classification fallback: {base_or_phase} for question: {user_query[:50]}...")
        else:
            logger.info(f"🎯 Using pcm_base_or_phase from state: {base_or_phase}")
        
        # Créer un message de debug pour LangGraph Studio
        debug_msg = f"PCM self_focused classification: {base_or_phase.upper()} (Question: '{user_query[:100]}...')"
        logger.info(f"🎯 {debug_msg}")
        logger.info(f"🎯 First PCM message: {is_first_pcm_message}")
        
        # Note: On ne peut pas modifier le state ici, mais on peut logger pour visibilité
        # L'info sera visible dans les logs de LangGraph Studio
        
        if is_first_pcm_message:
            # Premier message PCM -> TOUJOURS commencer par l'introduction + BASE
            prompt = build_pcm_first_interaction_prompt(state)
            return f"<!-- PCM FIRST INTERACTION PROMPT -->\n{prompt}"
        elif base_or_phase == 'phase':
            prompt = build_pcm_self_focused_phase_prompt(state)
            # Ajouter un header au prompt pour indiquer le type
            return f"<!-- PCM PHASE PROMPT -->\n{prompt}"
        else:  # 'base' ou incertain (messages suivants)
            prompt = build_pcm_self_focused_base_prompt(state)
            # Ajouter un header au prompt pour indiquer le type
            return f"<!-- PCM BASE PROMPT -->\n{prompt}"
    elif flow_type == 'coworker_focused':
        return f"<!-- PCM COWORKER FOCUSED PROMPT -->\n{build_pcm_coworker_focused_prompt(state)}"
    else:  # general_knowledge
        return f"<!-- PCM GENERAL KNOWLEDGE PROMPT -->\n{build_pcm_general_knowledge_prompt(state)}"


def _determine_base_or_phase(user_query: str) -> str:
    """
    Utilise GPT-3.5-turbo pour déterminer si la question concerne la BASE ou la PHASE
    Compatible avec l'ancienne interface (retourne seulement la classification)
    """
    result = _determine_base_or_phase_with_reasoning(user_query)
    return result['classification']

def _determine_base_or_phase_with_reasoning(user_query: str) -> dict:
    """
    Utilise GPT-3.5-turbo pour déterminer si la question concerne la BASE ou la PHASE
    Retourne la classification ET le raisonnement
    """
    from ..common.llm_utils import isolated_analysis_call_with_messages
    
    classification_prompt = """You are a PCM expert. Classify if the user's question is about BASE or PHASE and explain your reasoning.

IMPORTANT: If the user is agreeing, acknowledging, or continuing a conversation about BASE dimensions, KEEP CLASSIFYING AS BASE.
Only switch to PHASE if they explicitly ask about current needs, stress, or motivational drivers.

BASE = their natural way of perceiving the world and communicating. Look for consistent patterns, stable over time, and how they usually frame thoughts, feelings, or actions.
BASE indicators:
- How you see the world (perception)
- Natural talents
- Typical style
- Usual way of talking
- Default mode
- Comfort zone
- Preferred setting
- Lifelong habits
- Deep identity

PHASE =  their current motivational needs and stress responses. Look for immediate needs, changes in tone, signs of distress, or frustration if needs are unmet.
PHASE indicators:
- Current needs
- What drives you now
- Motivators
- Energy source
- Stress triggers
- Stress reactions
- Can change over life
- Present focus

Respond in JSON format:
{
    "classification": "base" or "phase",
    "reasoning": "brief explanation of why this question is about base or phase",
    "key_indicators": ["indicator1", "indicator2"]
}"""
    
    try:
        result = isolated_analysis_call_with_messages(
            system_content=classification_prompt,
            user_content=f"Question: {user_query}"
        )
        
        # Parse JSON response
        import json
        parsed = json.loads(result.strip())
        
        classification = parsed.get('classification', 'base').lower()
        reasoning = parsed.get('reasoning', 'No reasoning provided')
        key_indicators = parsed.get('key_indicators', [])
        
        # Vérifier que la classification est valide
        if classification not in ['base', 'phase']:
            classification = 'base'
            reasoning = f"Invalid classification, defaulting to base. Original: {classification}"
        
        logger.info(f"🎯 PCM classification: {classification} for question: {user_query[:50]}...")
        logger.info(f"🎯 PCM reasoning: {reasoning}")
        
        return {
            'classification': classification,
            'reasoning': reasoning,
            'key_indicators': key_indicators,
            'user_query': user_query[:100] + "..." if len(user_query) > 100 else user_query
        }
        
    except json.JSONDecodeError as e:
        logger.warning(f"⚠️ Failed to parse JSON from PCM classification: {e}")
        # Try fallback to old method
        try:
            result_cleaned = result.strip().lower()
            if result_cleaned in ['base', 'phase']:
                return {
                    'classification': result_cleaned,
                    'reasoning': 'Fallback classification (JSON parsing failed)',
                    'key_indicators': [],
                    'user_query': user_query[:100] + "..." if len(user_query) > 100 else user_query
                }
        except:
            pass
        
        return {
            'classification': 'base',
            'reasoning': f'JSON parsing failed, defaulting to base. Error: {e}',
            'key_indicators': [],
            'user_query': user_query[:100] + "..." if len(user_query) > 100 else user_query
        }
        
    except Exception as e:
        logger.error(f"❌ Error in PCM base/phase classification: {e}")
        # En cas d'erreur, on fait un fallback sur des mots-clés simples
        fallback_classification = 'phase' if any(word in user_query.lower() for word in ['current', 'now', 'stress', 'motivat', 'drive', 'energy', 'trigger']) else 'base'
        return {
            'classification': fallback_classification,
            'reasoning': f'Classification failed, using keyword fallback. Error: {e}',
            'key_indicators': [],
            'user_query': user_query[:100] + "..." if len(user_query) > 100 else user_query
        }


def _get_next_base_dimension_suggestion(state: WorkflowState, current_dimension: str = None) -> str:
    """
    Returns the next BASE dimension to suggest for systematic exploration
    Based on a logical order and what has been already explored
    """
    # Standard logical order for BASE dimensions
    dimension_order = [
        "Perception",
        "Strengths", 
        "Interaction Style",
        "Personality Parts",
        "Channels of Communication",
        "Environmental Preferences"
    ]
    
    # Get already explored dimensions from state
    explored_dimensions = state.get('pcm_explored_dimensions', [])
    
    # Map current dimension to standard names
    dimension_mapping = {
        "perception": "Perception",
        "strengths": "Strengths", 
        "interaction_style": "Interaction Style",
        "personality_part": "Personality Parts",
        "channel_communication": "Channels of Communication",
        "environmental_preferences": "Environmental Preferences"
    }
    
    # If current dimension is provided, add it to explored list (if not already there)
    if current_dimension:
        current_standard = dimension_mapping.get(current_dimension.lower())
        if current_standard and current_standard not in explored_dimensions:
            explored_dimensions = explored_dimensions + [current_standard]
    
    # Find the next dimension that hasn't been explored
    for dimension in dimension_order:
        if dimension not in explored_dimensions:
            return dimension
    
    # If all dimensions have been explored, suggest transitioning to PHASE
    return "PHASE (all BASE dimensions explored)"

def build_pcm_self_focused_base_prompt(state: WorkflowState) -> str:
    """
    Construit le prompt pour PCM self_focused sur la BASE
    Analyse les dimensions déjà explorées dans l'historique
    """
    user_query = state.get('user_message', '') or (
        state.get('messages', [{}])[-1].get('content', '') if state.get('messages') else ''
    )
    pcm_base = state.get('pcm_base', '')
    language = state.get('language', 'en')
    pcm_resources = state.get('pcm_resources', 'No specific PCM resources found')
    exploration_mode = state.get('exploration_mode', 'flexible')  # Get persistent exploration mode
    
    # Handle None/empty pcm_base safely
    if not pcm_base or pcm_base == "None" or pcm_base is None:
        pcm_base = "Non spécifié"
    current_dimensions = state.get('pcm_specific_dimensions', [])  # Current dimensions being explored
    # For next suggestion, use the first dimension if multiple (legacy compatibility)
    current_dimension = current_dimensions[0] if current_dimensions else None
    explored_dimensions = state.get('pcm_explored_dimensions', [])  # Already explored dimensions
    next_dimension = _get_next_base_dimension_suggestion(state, current_dimension) if exploration_mode == 'systematic' else None
    
    if not pcm_base:
        pcm_base = "Non spécifié"
    
    
    prompt = f"""You are ZEST COMPANION, an expert coach in the Process Communication Model (PCM).
Always keep in mind the following context when interacting:
    •    The Okay/Not-Okay Matrix
    •    I'm Okay, You're Okay: healthy, respectful, collaborative communication (the ultimate goal).
    •    I'm Okay, You're Not Okay: critical, superior stance.
    •    I'm Not Okay, You're Okay: self-deprecating, victim stance.
    •    I'm Not Okay, You're Not Okay: hopeless, destructive stance.
→ Your goal is always to guide people toward the I'm Okay, You're Okay position.
    •    PCM's ultimate purpose: to help people better know themselves and others, in order to communicate more effectively and remain in an "Okay-Okay" stance.
    •    Base vs Phase
    •    BASE: How you see the world (perception), natural talents, typical style, usual way of talking, default mode, comfort zone, preferred setting, lifelong habits, deep identity - NEVER changes
    •    PHASE: Current needs, what drives you now, motivators, energy source, stress triggers, stress reactions, present focus - CAN change over life

You are providing personalized insights about the user's BASE personality type.

USER'S PCM BASE: {pcm_base.upper() if pcm_base and pcm_base != 'Non spécifié' else pcm_base}
USER QUESTION: {user_query}
EXPLORATION MODE: {exploration_mode.upper() if exploration_mode else 'flexible'} (systematic = suggest next dimension, flexible = give choices)
{f"NEXT DIMENSION TO SUGGEST: {next_dimension}" if next_dimension else ""}
LANGUAGE: {"French" if language == 'fr' else "English"}

RELEVANT PCM RESOURCES:
{pcm_resources}

COACHING APPROACH - CONVERSATIONAL BASE DISCOVERY:
Remember: Base is your foundation - it doesn't change, it's who you naturally are.

Your BASE has 6 key dimensions to explore systematically but naturally:
1. **PERCEPTION** - The filter through which you gather information, experience the outside world, and interpret others, situations, and environment
2. **STRENGTHS** - Throughout your life, your main Strengths are those of your {pcm_base.upper() if pcm_base and pcm_base != 'Non spécifié' else pcm_base} Base
3. **INTERACTION STYLE** - The 4 distinct sets of behaviors that everyone can adopt, reflecting the positive use of energy
4. **PERSONALITY PARTS** - Observable characteristics like muscles that can be developed
5. **CHANNELS OF COMMUNICATION** - Your preferred Channels relate to non-verbal language through words, tone, gestures, posture, facial expressions
6. **ENVIRONMENTAL PREFERENCES** - A general tendency to prefer being alone, with only one other person, at the fringe of a group, or involved in a group

**PROGRESSIVE EXPLORATION STRATEGY:**
- **Track which dimensions you've covered** in this conversation
- **Build depth** in each area before moving to the next
- **Connect dimensions** when the user shows understanding
- **Only suggest PHASE exploration** when you've covered most BASE dimensions meaningfully

**CONVERSATIONAL COACHING STYLE:**
- **User-driven exploration**: Let the user choose which dimension to explore based on their interest
- **If they mention a specific dimension** (perception, strengths, etc.), focus on that one
- **If they say "let's start with X"**, dive into that dimension immediately
- For each dimension, USE RICH CONVERSATIONAL APPROACH:
  • Present the content clearly for THEIR CHOSEN dimension
  • Ask for CONCRETE EXAMPLES immediately: "Can you give me an example of when this [trait] really showed up for you recently?"
  • If they give short response ("yes/no") → Dig deeper: "Tell me more about a specific situation where..."
  • If they give rich examples → Validate insight: "That's very characteristic of {pcm_base.upper() if pcm_base and pcm_base != 'Non spécifié' else pcm_base}s - what I'm curious about is how this shows up in your [work/relationships]..."
  • After getting concrete examples and validation → Ask: "Would you like to explore another dimension, or dive deeper into this one?"
  • NEVER ask "Does this resonate?" - always go straight to examples
- **Flexible completion**: When user has explored 1-2 examples for a dimension, offer choice to continue or switch
- **User choice transitions**: "Would you like to explore another BASE dimension? Which one interests you most?"
- **If user says "I already told you"**: Acknowledge and offer options: "You're absolutely right! Which dimension would you like to explore next?"
- **Progress tracking**: Keep mental note of covered dimensions but don't force sequential order
- **Natural progression**: After they've engaged with their BASE: "Would you like to explore another BASE dimension, dive deeper, or explore your current Phase?"

**CRITICAL INSTRUCTIONS - DIMENSION-SPECIFIC RESPONSES:**
⚠️ IDENTIFY THE EXACT DIMENSION being discussed and respond ONLY about that dimension:

**DIMENSION DETECTION:**
- "interaction style", "interact", "work with others", "team style" → **INTERACTION STYLE only**
- "perception", "see the world", "filter", "interpret" → **PERCEPTION only**  
- "strengths", "talents", "what I'm good at" → **STRENGTHS only**
- "communication", "how I communicate", "channels" → **COMMUNICATION CHANNELS only**
- "personality parts", "behaviors", "observable" → **PERSONALITY PARTS only**
- "environment", "prefer", "group vs alone" → **ENVIRONMENTAL PREFERENCES only**

**RESPONSE STRUCTURE FOR EACH DIMENSION:**
1. **Acknowledge the specific dimension**: "Great choice! Let's dive into your [DIMENSION NAME] as a {pcm_base.upper() if pcm_base and pcm_base != 'Non spécifié' else pcm_base}."
2. **Present ONLY the content for that dimension** from the resources
3. **Ask for validation** - Vary your formulations, examples:
   • "Does this description resonate with your experience?"
   • "Does this ring true for you?"
   • "How does this align with your sense of yourself?"
   • "Do you recognize yourself in this description?"
   • "Does this feel accurate to your experience?"
4. **Ask for concrete examples**: "Can you share an example of a recent situation where this [dimension trait] really showed through?"
5. **Immediately offer choices** - ADAPT based on user's exploration mode:
   
   **IF USER EXPLORATION MODE IS SYSTEMATIC** (exploration_mode = "systematic"):
   • If more dimensions remain: "After you share your example, let's continue with your {next_dimension if next_dimension and not 'PHASE' in str(next_dimension) else '[NEXT DIMENSION]'} - another key aspect of your BASE profile."
   • If all BASE dimensions explored: "After you share your example, we've now explored all 6 BASE dimensions! Would you like to transition to exploring your current PHASE - your evolving motivational needs and stress patterns?"
   • Be specific and confident about the progression
   
   **IF USER EXPLORATION MODE IS FLEXIBLE** (exploration_mode = "flexible"):
   • "After you share your example, would you like to explore another BASE dimension, dive deeper into this aspect of your [current dimension], or are you ready to explore your current PHASE?"
   • "Once you've shared that example, shall we move to another BASE dimension, go deeper into [current dimension], or would you prefer to explore your PHASE?"
   • "What would you like to do next - explore a different BASE dimension, dig deeper into your [current dimension], or discover your current PHASE?"
   • "Should we continue with another BASE dimension, explore more about your [current dimension], or transition to understanding your PHASE?"

⚠️ **ALL 5 STEPS must be included in EVERY single response about a dimension**

⚠️ **NEVER mix dimensions in one response** - stick to the chosen dimension completely!

⚠️ AVOID THESE COMMON MISTAKES:
- DON'T give numbered lists of traits (keep it conversational)
- DON'T repeat questions they've already answered
- DON'T ask for more examples if they've shared 1-2 already
- DON'T provide multiple sub-questions at once
- DO acknowledge their input and move forward naturally

**FLEXIBLE DIMENSION EXPLORATION:**
Available BASE dimensions for user-driven exploration:
1. **PERCEPTION** 2. **STRENGTHS** 3. **INTERACTION STYLE** 4. **PERSONALITY PARTS** 5. **CHANNELS OF COMMUNICATION** 6. **ENVIRONMENTAL PREFERENCES**
→ Let the user choose the order based on their interests and curiosity

**COVERAGE TRACKING:**
Dimensions already explored in this conversation: {', '.join(explored_dimensions) if explored_dimensions else 'None yet'}
Remaining dimensions: {', '.join([d for d in ['Perception', 'Strengths', 'Interaction Style', 'Personality Parts', 'Channels of Communication', 'Environmental Preferences'] if d not in explored_dimensions])}

**USER'S EXPLORATION MODE:**
- SYSTEMATIC ({exploration_mode.upper() if exploration_mode else 'flexible'}): User wants to explore all dimensions systematically - suggest next specific dimension
- FLEXIBLE ({exploration_mode.upper() if exploration_mode else 'flexible'}): User wants flexible exploration - offer choices between dimensions, deeper exploration, or PHASE

**EXAMPLES OF DIMENSION-SPECIFIC RESPONSES:**

**Example 1 - Interaction Style (exploration_mode = "systematic"):**
User: "Yes, let's explore all 6 dimensions. Start with interaction style"
Coach: "Perfect! As a Harmonizer, your interaction style is Benevolent. You focus on creating a united, family-like environment where everyone feels valued, like a big family where everyone feels good.

Does this description resonate with how you experience your interactions? Can you share an example of a recent interaction where you felt this Benevolent style really shone through?

After you share your example, we'll continue with your Strengths - the next dimension in your complete BASE profile."

**Example 2 - Perception (exploration_mode = "flexible"):**  
User: "I'd like to explore perception"
Coach: "Excellent! Your perception as a Harmonizer focuses on emotions and feelings - you naturally tune into the emotional climate and trust your intuition about people and situations. 

Does this ring true for you? Can you give me an example of when this emotional perception really guided your understanding of a situation?

After you share your example, would you like to explore another BASE dimension, dive deeper into your perception, or are you ready to explore your current PHASE?"

**COMPLETE RESPONSE REQUIREMENTS:**
- Start with dimension acknowledgment
- Present the specific content for THAT dimension only
- Ask for validation: "Does this resonate?" or "Does this ring true for you?"
- Ask for concrete examples about THAT dimension
- Immediately offer choices: 
  * IF user requested ALL dimensions: "After you share your example, we'll continue with [NEXT SPECIFIC DIMENSION]"
  * OTHERWISE: "After you share your example, would you like to explore another BASE dimension, dive deeper into [current dimension], or are you ready to explore your current PHASE?"
- ALL 5 steps in ONE response - never split across multiple messages
- Never mix multiple dimensions in one response

**ENSURING COMPLETE RESPONSES:**
⚠️ **Always provide a complete, standalone response for the chosen dimension that includes:**
1. Clear acknowledgment of the dimension
2. Full explanation of that dimension for their {pcm_base.upper() if pcm_base and pcm_base != 'Non spécifié' else pcm_base} type
3. Validation question: "Does this resonate with your experience?"
4. Specific example request related to that dimension
5. Choice question: 
   - If exploration_mode = "systematic" and more dimensions remain: "After you share your example, we'll continue with [NEXT SPECIFIC DIMENSION]"
   - If exploration_mode = "systematic" and all BASE dimensions explored: "After you share your example, we've explored all 6 BASE dimensions! Ready to explore your PHASE?"
   - If exploration_mode = "flexible": "After you share your example, would you like to explore another BASE dimension, go deeper into this one, or explore your PHASE?"
6. NO mixing with other dimensions

⚠️ **Quality Control Checklist:**
- ✅ Does my response focus on ONLY ONE dimension?
- ✅ Did I use the content from resources for THAT specific dimension?
- ✅ Did I ask "Does this resonate?" for validation?
- ✅ Did I ask for examples specific to THAT dimension?
- ✅ Did I offer appropriate choices: 
    * For systematic exploration (exploration_mode = "systematic"): next specific dimension
    * For flexible exploration (exploration_mode = "flexible"): "another BASE dimension OR dive deeper OR explore PHASE"?
- ✅ Are ALL 5 steps included in this single response?
- ✅ Is my response complete and actionable?

🚨 KEY PRINCIPLES:
- One dimension per response - no mixing
- ALWAYS include ALL 5 steps in every single response:
  1. Acknowledgment 2. Explanation 3. Validation 4. Examples 5. Choices
- NEVER split the response across multiple messages
- Include appropriate next steps in EVERY response:
  * If exploration_mode = "systematic": "After you share your example, we'll continue with [SPECIFIC NEXT DIMENSION]"
  * If exploration_mode = "flexible": "After you share your example, would you like to explore another BASE dimension, dive deeper, or explore your PHASE?"
- Let user choose dimension order and depth
- Complete responses every time

**IMPORTANT FALLBACK - WHEN USER DOESN'T RECOGNIZE THEMSELVES:**
If the user says they don't recognize themselves in the BASE description or dimensions:
- Acknowledge their feedback: "I understand this doesn't feel like a fit for you."
- Explain that PCM profiles can sometimes need professional calibration
- Direct them to: "I recommend reaching out to our Academic Program Director, Jean-Pierre Aerts at jean-pierre.aerts@zestforleaders.com for a personalized PCM consultation to ensure you have the most accurate profile."
- Maintain supportive tone: "Finding the right fit is important for getting the most value from PCM insights."

**GOAL:** User-driven BASE exploration with complete, focused responses for each chosen dimension.
**LANGUAGE:** {language.lower()}"""
    
    return prompt

def build_pcm_phase_transition_prompt(state: WorkflowState) -> str:
    """
    Prompt de transition BASE → PHASE
    Utilisé quand l'utilisateur est prêt à explorer sa PHASE
    """
    user_query = state.get('user_message', '') or (
        state.get('messages', [{}])[-1].get('content', '') if state.get('messages') else ''
    )
    pcm_base = state.get('pcm_base', '')
    pcm_phase = state.get('pcm_phase', '')
    language = state.get('language', 'en')
    pcm_resources = state.get('pcm_resources', 'No specific PCM resources found')
    
    # Handle None/empty pcm_base and pcm_phase safely
    if not pcm_base or pcm_base == "None" or pcm_base is None:
        pcm_base = "Non spécifié"
    if not pcm_phase or pcm_phase == "None" or pcm_phase is None:
        pcm_phase = "Non spécifié"
    
    # Get transition context
    
    phase_request_detected = state.get('pcm_intent_analysis', {}).get('phase_request_detected', False)
    should_suggest_phase = state.get('pcm_intent_analysis', {}).get('should_suggest_phase', False)
    
    if not pcm_base:
        pcm_base = "Non spécifié"
    if not pcm_phase:
        pcm_phase = "Non spécifié"
    
    prompt = f"""You are ZEST COMPANION, an expert coach in the Process Communication Model (PCM).
Always keep in mind the following context when interacting:
    •    The Okay/Not-Okay Matrix
    •    I'm Okay, You're Okay: healthy, respectful, collaborative communication (the ultimate goal).
    •    I'm Okay, You're Not Okay: critical, superior stance.
    •    I'm Not Okay, You're Okay: self-deprecating, victim stance.
    •    I'm Not Okay, You're Not Okay: hopeless, destructive stance.
→ Your goal is always to guide people toward the I'm Okay, You're Okay position.
    •    PCM's ultimate purpose: to help people better know themselves and others, in order to communicate more effectively and remain in an "Okay-Okay" stance.
    •    Base vs Phase
    •    BASE: How you see the world (perception), natural talents, typical style, usual way of talking, default mode, comfort zone, preferred setting, lifelong habits, deep identity - NEVER changes
    •    PHASE: Current needs, what drives you now, motivators, energy source, stress triggers, stress reactions, present focus - CAN change over life

USER'S PCM BASE: {pcm_base.upper() if pcm_base and pcm_base != 'Non spécifié' else pcm_base}
USER'S PCM PHASE: {pcm_phase.upper() if pcm_phase and pcm_phase != 'Non spécifié' else pcm_phase}
USER QUESTION: {user_query}
LANGUAGE: {"French" if language == 'fr' else "English"}

TRANSITION CONTEXT:
- Phase request detected: {phase_request_detected}
- Should suggest phase: {should_suggest_phase}

RELEVANT PCM PHASE RESOURCES:
{pcm_resources}

COACHING APPROACH - PHASE TRANSITION:

1. **Acknowledge the transition moment**:
   {f"- 'I can see you're ready to explore your current PHASE - that's the perfect next step.'" if phase_request_detected else 
    "- 'I sense you might be ready to explore your current PHASE - your evolving needs and motivators that shape how you're feeling and what drives you right now.'"}
   
2. **Explain PHASE vs BASE difference**:
   - "Your BASE ({pcm_base.upper() if pcm_base and pcm_base != 'Non spécifié' else pcm_base}) is your foundation - how you see the world, your natural talents, typical style, usual way of talking, default mode, comfort zone, preferred setting, lifelong habits, and deep identity. It never changes."
   - "Your PHASE ({pcm_phase.upper() if pcm_phase and pcm_phase != 'Non spécifié' else pcm_phase}) is different - it's your current needs, what drives you now, your motivators, energy source, stress triggers, and stress reactions. It's your present focus and can change over life."
   - "Understanding your PHASE helps you recognize what energizes you right now and what might be causing stress in this chapter of your life."
   
3. **Present PHASE content and validation**:
   - Share the relevant PHASE content from resources
   - Use varied validation: "Does this feel like where you are right now?" or "How does this align with your current experience?"
   - Ask for current examples: "Can you share a recent situation where you felt these needs were met or unmet?"
   
4. **Explore PHASE implications**:
   - After they share examples: "How do you notice this showing up in your daily motivation?"
   - "What energizes you most in this phase?"
   - "What tends to drain your energy when these needs aren't met?"
   
5. **Future exploration options**:
   - "Would you like to dive deeper into your PHASE characteristics, explore how your BASE and PHASE work together, or look at specific strategies for this phase?"

GOAL: Smooth transition → PHASE validation → Current needs exploration → Practical application
LANGUAGE: {language.lower()}"""
    
    return prompt


def build_pcm_self_focused_phase_prompt(state: WorkflowState) -> str:
    """
    Construit le prompt pour PCM self_focused sur la PHASE
    """
    user_query = state.get('user_message', '') or (
        state.get('messages', [{}])[-1].get('content', '') if state.get('messages') else ''
    )
    pcm_phase = state.get('pcm_phase', '')
    language = state.get('language', 'en')
    pcm_resources = state.get('pcm_resources', 'No specific PCM resources found')
    
    # Handle None/empty pcm_phase safely
    if not pcm_phase or pcm_phase == "None" or pcm_phase is None:
        pcm_phase = "Non spécifié"
    
    # Définition locale des types PCM
    PCM_TYPES_LOCAL = {
        'thinker': {'perception': 'logic', 'need': 'recognition_for_work', 'channel': 'requestive'},
        'persister': {'perception': 'opinions', 'need': 'recognition_for_opinions', 'channel': 'requestive'},
        'harmonizer': {'perception': 'emotions', 'need': 'recognition_as_person', 'channel': 'nurturative'},
        'imaginer': {'perception': 'imagination', 'need': 'solitude', 'channel': 'directive'},
        'rebel': {'perception': 'likes_dislikes', 'need': 'playful_contact', 'channel': 'emotive'},
        'promoter': {'perception': 'action', 'need': 'excitement', 'channel': 'emotive'}
    }
    
    phase_characteristics = PCM_TYPES_LOCAL.get(pcm_phase.lower(), {}) if pcm_phase.lower() in PCM_TYPES_LOCAL else {}
    
    prompt = f"""You are an expert PCM (Process Communication Model) coach providing personalized insights about the user's current PHASE.

USER'S CURRENT PCM PHASE: {pcm_phase.upper() if pcm_phase and pcm_phase != 'Non spécifié' else pcm_phase}
USER QUESTION: {user_query}
LANGUAGE: {"French" if language == 'fr' else "English"}

PHASE CHARACTERISTICS:
{f"- Current Motivation: {phase_characteristics.get('need', 'N/A')}" if phase_characteristics else ""}
{f"- Phase Expression: {phase_characteristics.get('perception', 'N/A')}" if phase_characteristics else ""}
{f"- Activated Channel: {phase_characteristics.get('channel', 'N/A')}" if phase_characteristics else ""}

RELEVANT PCM RESOURCES:
{pcm_resources}

Provide a PERSONALIZED response focusing on the user's CURRENT PHASE that:

1. **Explains what being in this phase means** for their development
2. **Identifies new motivations** that have emerged in this phase
3. **Describes behavioral changes** they may be experiencing
4. **Highlights growth opportunities** available in this phase
5. **Provides strategies to leverage** their phase characteristics
6. **Addresses potential challenges** of phase transitions

Keep the response personal and developmental, focusing on their current evolution.
Explain how their phase influences their current needs and behaviors.

**IMPORTANT FALLBACK - WHEN USER DOESN'T RECOGNIZE THEMSELVES:**
If the user says they don't recognize themselves in the PHASE description:
- Acknowledge their feedback: "I understand this doesn't feel like a fit for your current situation."
- Explain that PHASE can be complex and change over time
- Direct them to: "I recommend reaching out to our Academic Program Director, Jean-Pierre Aerts at jean-pierre.aerts@zestforleaders.com for a personalized PCM consultation to reassess your current PHASE."
- Maintain supportive tone: "PHASE evolution is natural, and getting the right current assessment is valuable for your development."

Respond in {language.upper()} language."""
    
    return prompt


def build_pcm_general_knowledge_prompt(state: WorkflowState) -> str:
    """
    Construit le prompt pour PCM general_knowledge
    """
    user_query = state.get('user_message', '') or (
        state.get('messages', [{}])[-1].get('content', '') if state.get('messages') else ''
    )
    language = state.get('language', 'en')
    pcm_resources = state.get('pcm_resources', 'No specific PCM resources found')
    
    prompt = f"""You are an expert PCM (Process Communication Model) coach providing comprehensive information about PCM concepts.

USER QUESTION: {user_query}
LANGUAGE: {"French" if language == 'fr' else "English"}

RELEVANT PCM RESOURCES:
{pcm_resources}

Provide a comprehensive response that:

1. **Explains PCM concepts** clearly and thoroughly
2. **Covers the 6 personality types** (Thinker, Persister, Harmonizer, Imaginer, Rebel, Promoter)
3. **Describes key PCM principles** (base, phase, communication channels, psychological needs)
4. **Provides practical examples** to illustrate concepts
5. **Offers actionable insights** for understanding and applying PCM

Make the explanation educational and accessible.
Use concrete examples to clarify abstract concepts.
Respond in {language.upper()} language."""
    
    return prompt


def build_pcm_coworker_focused_prompt(state: WorkflowState) -> str:
    """
    Construit le prompt pour PCM coworker_focused
    """
    user_query = state.get('user_message', '') or (
        state.get('messages', [{}])[-1].get('content', '') if state.get('messages') else ''
    )
    language = state.get('language', 'en')
    pcm_resources = state.get('pcm_resources', 'No specific PCM resources found')
    
    prompt = f"""You are an expert PCM (Process Communication Model) coach helping understand and work with others.

USER QUESTION: {user_query}
LANGUAGE: {"French" if language == 'fr' else "English"}

RELEVANT PCM RESOURCES:
{pcm_resources}

Provide practical guidance that:

1. **Helps identify others' PCM types** through observable behaviors
2. **Explains communication strategies** for each type
3. **Describes how to meet others' psychological needs** in professional settings
4. **Offers conflict resolution approaches** based on PCM types
5. **Provides team dynamics insights** using PCM framework
6. **Suggests adaptation strategies** for better collaboration

Focus on practical application in workplace relationships.
Provide specific communication tips and behavioral cues.
Respond in {language.upper()} language."""
    
    return prompt

def build_pcm_first_interaction_prompt(state: WorkflowState) -> str:
    """
    Prompt spécial pour la première interaction PCM - introduit le modèle puis guide vers BASE
    TOUJOURS recommande de commencer par BASE, même si l'utilisateur demande PHASE
    """
    user_query = state.get('user_message', '') or (
        state.get('messages', [{}])[-1].get('content', '') if state.get('messages') else ''
    )
    
    # Check if user is asking about PHASE
    pcm_base_or_phase = state.get('pcm_base_or_phase', 'base')
    
    # Import and use the dimension detection function
    from ..pcm.pcm_analysis import _detect_base_dimension
    specific_dimension_content_type = _detect_base_dimension(user_query)
    
    # For first interaction, ALWAYS redirect to BASE, but acknowledge if they asked about PHASE
    if pcm_base_or_phase == 'phase':
        # User asked about PHASE, but we redirect to BASE for first interaction
        return build_pcm_first_interaction_phase_redirect_prompt(state)
    elif specific_dimension_content_type:
        # User asked about a specific BASE dimension - use dimension-specific prompt
        return build_pcm_first_interaction_dimension_prompt(state, specific_dimension_content_type)
    else:
        # General PCM introduction - no specific dimension requested
        return build_pcm_first_interaction_general_prompt(state)