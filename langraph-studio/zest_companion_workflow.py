"""
ZEST MBTI Workflow - LangGraph Studio
Reproduit exactement le workflow n8n avec:
1. Récupération profil MBTI
2. Agent MBTI Expert 
3. Recherche vectorielle (Tools A/B/C/D)
4. Agent principal
5. Mémoire de conversation
"""

import os
import json
import unicodedata
import logging
from typing import Dict, List, Optional, Annotated
from typing_extensions import TypedDict
# from langgraph.graph.message import add_messages

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from supabase import create_client, Client

# Configuration
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY")
)

llm = ChatOpenAI(
    model="gpt-4o-mini", 
    temperature=0.7, 
    streaming=True,  # Streaming activé pour la réponse finale
    callbacks=[],
    max_tokens=1500  # Augmenté pour les réponses Lencioni complètes avec insights équipe
)

# LLM d'analyse COMPLÈTEMENT ISOLÉ
analysis_llm = ChatOpenAI(
    model="gpt-3.5-turbo", 
    temperature=0.7,  # Augmenté de 0.3 à 0.7 pour plus de flexibilité
    streaming=False,
    callbacks=[],  # Callbacks vides
    tags=["internal_analysis"],
    # CLÉS D'ISOLATION :
    openai_api_key=os.environ.get("OPENAI_API_KEY"),  # Forcer la clé explicite
    model_kwargs={
        "stream": False,  # Forcer pas de streaming au niveau OpenAI
    },
    logprobs=False,  # Désactiver les logprobs
    # Isolation supplémentaire
    request_timeout=30,
    max_retries=1,
    verbose=False  # Pas de verbose
)

# Alternative : Utiliser directement OpenAI sans LangChain pour l'analyse
def isolated_analysis_call_with_messages(system_content: str, user_content: str) -> str:
    """
    Appel OpenAI direct avec messages système et utilisateur séparés
    Garantit l'isolation complète du streaming LangGraph
    """
    try:
        import openai
        import os
        
        # Créer un client complètement isolé
        isolated_client = openai.OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            timeout=30.0
        )
        
        logger.info("🔒 Using completely isolated OpenAI client with separate messages...")
        
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]
        
        response = isolated_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7,  # Augmenté de 0.3 à 0.7 pour plus de flexibilité
            max_tokens=500,
            stream=False,  # Absolument pas de streaming
            logprobs=False,  # Désactiver les logprobs
        )
        
        result = response.choices[0].message.content.strip()
        logger.info(f"🔒 Isolated call completed: {len(result)} chars")
        return result
        
    except ImportError:
        logger.warning("⚠️ OpenAI module not available, using fallback")
        return fallback_analysis_call_with_messages(system_content, user_content)
    except Exception as e:
        logger.error(f"❌ Isolated analysis call failed: {e}")
        return fallback_analysis_call_with_messages(system_content, user_content)

def fallback_analysis_call_with_messages(system_content: str, user_content: str) -> str:
    """Fallback using LangChain analysis_llm with separate messages"""
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        logger.info("🔄 Using LangChain fallback with separate messages...")
        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=user_content)
        ]
        response = analysis_llm.invoke(messages)
        return response.content
    except Exception as e:
        logger.error(f"❌ Fallback analysis also failed: {e}")
        return '{"question_type": "PERSONAL_DEVELOPMENT", "instructions": "CALL_AB: Tool A + Tool B", "other_mbti_profiles": null, "continuity_detected": false}'

def isolated_analysis_call(prompt_content: str) -> str:
    """
    Appel OpenAI direct sans passer par LangChain
    Garantit l'isolation complète du streaming LangGraph
    """
    try:
        import openai
        import os
        
        # Créer un client complètement isolé
        isolated_client = openai.OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            timeout=30.0
        )
        
        logger.info("🔒 Using completely isolated OpenAI client...")
        
        # Séparer le prompt système du contenu utilisateur si possible
        if "\n\n" in prompt_content and prompt_content.startswith("You are"):
            # Détection d'un prompt composé (system + user content)
            parts = prompt_content.split("\n\nOriginal User Input:", 1)
            if len(parts) == 2:
                system_part = parts[0]
                user_part = "Original User Input:" + parts[1]
                messages = [
                    {"role": "system", "content": system_part},
                    {"role": "user", "content": user_part}
                ]
            else:
                messages = [{"role": "user", "content": prompt_content}]
        else:
            messages = [{"role": "user", "content": prompt_content}]
        
        response = isolated_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7,  # Augmenté de 0.3 à 0.7 pour plus de flexibilité
            max_tokens=500,
            stream=False,  # Absolument pas de streaming
            logprobs=False,  # Désactiver les logprobs
        )
        
        result = response.choices[0].message.content.strip()
        logger.info(f"🔒 Isolated call completed: {len(result)} chars")
        return result
        
    except ImportError:
        logger.warning("⚠️ OpenAI module not available, using fallback")
        return fallback_analysis_call(prompt_content)
    except Exception as e:
        logger.error(f"❌ Isolated analysis call failed: {e}")
        return fallback_analysis_call(prompt_content)

def fallback_analysis_call(prompt_content: str) -> str:
    """Fallback using LangChain analysis_llm"""
    try:
        from langchain_core.messages import HumanMessage
        logger.info("🔄 Using LangChain fallback for analysis...")
        response = analysis_llm.invoke([HumanMessage(content=prompt_content)])
        return response.content
    except Exception as e:
        logger.error(f"❌ Fallback analysis also failed: {e}")
        return '{"question_type": "PERSONAL_DEVELOPMENT", "instructions": "CALL_AB: Tool A + Tool B", "other_mbti_profiles": null, "continuity_detected": false}'



embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Fonction pour gérer les messages manuellement sans auto-ajout
def normalize_name_for_metadata(name: str) -> str:
    """
    Normalise un nom pour matcher le format utilisé dans les métadonnées de vectorisation.
    jean-pierre_aerts -> jean_pierre_aerts (remplace tirets par underscores)
    """
    if not name:
        return name
    # Remplacer les tirets par des underscores pour matcher les métadonnées
    normalized = name.replace('-', '_')
    return normalized

def manage_messages(existing_messages, new_messages):
    """Gérer manuellement les messages sans ajouter automatiquement toutes les réponses LLM"""
    if not existing_messages:
        existing_messages = []
    
    # Seulement ajouter les messages utilisateur et les réponses finales explicites
    result = existing_messages[:]
    for msg in new_messages:
        if msg.get('type') == 'user' or msg.get('role') == 'user':
            result.append(msg)
        elif msg.get('type') == 'assistant' or msg.get('role') == 'assistant':
            # Seulement ajouter si c'est explicitement marqué comme réponse finale
            if msg.get('is_final_response', False):
                result.append(msg)
    
    return result


# État du graph
class WorkflowState(TypedDict):
    messages: List[dict]  # Pas d'agrégation auto; on contrôle explicitement l'ajout
    user_message: str
    main_theme: str
    sub_theme: str
    user_id: str
    user_name: str
    user_email: Optional[str]  # Nouvel identifiant pour la recherche
    client: str
    cohort: str
    filter: str
    folder_path: str
    
    # Données utilisateur
    user_mbti: Optional[str]
    user_temperament: Optional[str]
    temperament_description: Optional[str]
    
    # Analyse MBTI Expert
    mbti_analysis: Optional[Dict]
    reformulated_query: Optional[str]  # 🔄 AJOUTÉ: Query reformulée par NODE 3
    
    # Analyse Lencioni Intent
    lencioni_intent_analysis: Optional[Dict]  # 🎯 AJOUTÉ: Analyse d'intent Lencioni
    lencioni_data: Optional[List[Dict]]  # 📊 AJOUTÉ: Scores Lencioni de l'équipe
    lencioni_details: Optional[List[Dict]]  # 📋 AJOUTÉ: Questions détaillées par dysfonction
    dysfunction_focus: Optional[List[str]]  # 🎯 AJOUTÉ: Dysfonctions spécifiques mentionnées
    lencioni_search_results: Optional[Dict]  # 🔍 AJOUTÉ: Résultats de recherche Lencioni
    search_executed_for_intent: Optional[str]  # 🎯 AJOUTÉ: Type d'intent pour debug
    
    # Analyse des tempéraments (NODE 3.5)
    temperament_analysis: Optional[Dict]
    temperament_search_results: Optional[List[Dict]]
    
    # Résultats recherche vectorielle
    personalized_content: List[Dict]
    generic_content: List[Dict]
    others_content: List[Dict]
    general_content: List[Dict]
    temperament_content: List[Dict]  # 🏛️ NOUVEAU: Contenu des tempéraments (user et/ou others)
    
    # Réponse finale
    final_response: str
    streaming_active: Optional[bool]
    
    # Debug - Visible dans LangGraph Studio
    system_prompt_debug: Optional[str]
    reformulated_query_debug: Optional[str]

# ============================================================================
# SYSTÈME DE TEMPLATES POUR STREAMING ROBUSTE 
# Support pour tous les sous-thèmes avec streaming garanti
# ============================================================================

def create_prompt_by_subtheme(sub_theme: str, state: WorkflowState) -> str:
    """
    Factory qui retourne le bon prompt selon le sub_theme
    Système extensible pour supporter 15+ sous-thèmes différents
    """
    logger.info(f"🎯 Creating prompt for sub_theme: {sub_theme}")
    
    if sub_theme == 'D6_CollectiveSuccess':
        return build_lencioni_prompt(state)
    elif sub_theme == 'A1_PersonalityMBTI':
        return build_mbti_prompt(state)
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
        # For other intents, use simpler format
        system_prompt += "\nTeam's Lencioni Profile:\n"
        for item in lencioni_data:
            system_prompt += f"- {item['dysfunction']}: {item['score']}/5.0 ({item['level']}) - {item.get('summary', '')}\n"
        
        system_prompt += "\nUse this information to personalize your advice and recommendations."
    else:
        system_prompt += "\nThe user doesn't have a Lencioni profile yet. Provide general team development advice."
    
    # Ajouter les résultats de recherche vectorielle selon l'intent
    if search_results:
        system_prompt += "\n\n📚 RELEVANT KNOWLEDGE BASE CONTENT:\n"
        
        if intent_type == "REPORT_LOOKUP" and search_results.get("report_lookup_content"):
            system_prompt += "\n**📖 SUPPLEMENTARY CONTEXT - LENCIONI MODEL OVERVIEW:**\n"
            system_prompt += "USE THIS TO COMPLEMENT (NOT REPLACE) THE TEAM SCORES ABOVE.\n"
            system_prompt += "This overview helps explain the theory behind each dysfunction to enrich your interpretation of the team's specific scores.\n\n"
            
            for item in search_results["report_lookup_content"]:
                if item.get('type') == 'lencioni_overview':
                    # Pour l'overview, inclure TOUT le contenu (pas de limite)
                    full_content = item.get('content', '')
                    system_prompt += f"{full_content}\n\n"
                    logger.info(f"📖 Added full overview content: {len(full_content)} characters")
                else:
                    # Pour d'autres contenus, garder une limite raisonnable
                    system_prompt += f"- {item.get('content', '')[:500]}...\n"
        
        elif intent_type == "LENCIONI_GENERAL_KNOWLEDGE" and search_results.get("general_knowledge_content"):
            system_prompt += "\n**Lencioni Five Dysfunctions Overview:**\n"
            # Afficher l'overview complet pour donner le contexte général
            for item in search_results["general_knowledge_content"]:
                content = item.get('content', '')
                if content:
                    system_prompt += f"{content}\n\n"
        
        elif intent_type == "INSIGHT_BLEND" and search_results.get("insight_blend_content"):
            # Séparer les différents types de contenu
            lencioni_overview = [item for item in search_results["insight_blend_content"] if item.get("type") == "lencioni_overview"]
            lencioni_recommendations = [item for item in search_results["insight_blend_content"] if item.get("type") == "lencioni_recommendation"]
            team_recommendations = [item for item in search_results["insight_blend_content"] if item.get("type") == "team_recommendation"]
            
            # Si overview disponible (pas de dysfunction spécifique mentionnée)
            if lencioni_overview:
                system_prompt += "\n**Lencioni Five Dysfunctions Overview:**\n"
                for item in lencioni_overview:
                    system_prompt += f"📊 {item.get('content', '')}\n"
            
            if lencioni_recommendations:
                system_prompt += "\n**Lencioni Framework Recommendations:**\n"
                for item in lencioni_recommendations:  # Supprimé [:3] - toutes les recommandations
                    dysfunction = item.get("dysfunction", "Unknown")
                    system_prompt += f"📋 {dysfunction}: {item.get('content', '')}\n"
            
            if team_recommendations:
                system_prompt += "\n**Your Team's Specific Insights (USE AS CONTEXT - DO NOT LIST DIRECTLY):**\n"
                system_prompt += "Below are your team's voted priorities and detailed question scores. Use contextually based on the user's question:\n"
                system_prompt += "• For improvement questions: Focus on LOW-scoring areas and their corresponding team insights\n"
                system_prompt += "• For strengths questions: Highlight HIGH-scoring areas where team is performing well\n"
                system_prompt += "• Connect team votes with question scores to validate patterns\n\n"
                
                # Grouper par dysfonction pour une meilleure organisation
                trust_items = [item for item in team_recommendations if item.get("dysfunction", "").lower() == "trust"]
                conflict_items = [item for item in team_recommendations if item.get("dysfunction", "").lower() == "conflict"]
                commitment_items = [item for item in team_recommendations if item.get("dysfunction", "").lower() == "commitment"]
                accountability_items = [item for item in team_recommendations if item.get("dysfunction", "").lower() == "accountability"]
                results_items = [item for item in team_recommendations if item.get("dysfunction", "").lower() == "results"]
                
                if trust_items:
                    # Récupérer le score pour contextualiser
                    trust_score = None
                    if lencioni_data:
                        for item in lencioni_data:
                            if item.get('dysfunction', '').lower() == 'trust':
                                trust_score = item.get('score', 0)
                                break
                    
                    score_context = f" (Team Score: {trust_score}/5.0)" if trust_score else ""
                    system_prompt += f"**🤝 Trust - Specific areas to build more trust{score_context}:**\n"
                    for item in trust_items:
                        recommendation = item.get("recommendation", "")
                        vote_count = item.get("vote_count", 0)
                        system_prompt += f"• \"{recommendation}\" (identified by {vote_count} team members)\n"
                    system_prompt += "\n"
                
                if conflict_items:
                    system_prompt += "**⚔️ Conflict - Behaviors/actions acceptability during conflict:**\n"
                    for item in conflict_items:
                        behavior = item.get("behavior", "")
                        admitting = item.get("admitting_count", 0)
                        acceptable = item.get("acceptable_count", 0)
                        tolerable = item.get("tolerable_count", 0)
                        unacceptable = item.get("unacceptable_count", 0)
                        system_prompt += f"• \"{behavior}\"\n"
                        system_prompt += f"  - {admitting} members admit to this behavior\n"
                        system_prompt += f"  - Team rating: Acceptable({acceptable}), Tolerable({tolerable}), Unacceptable({unacceptable})\n"
                    system_prompt += "\n"
                
                if commitment_items:
                    # Récupérer le score pour contextualiser
                    commitment_score = None
                    if lencioni_data:
                        for item in lencioni_data:
                            if item.get('dysfunction', '').lower() == 'commitment':
                                commitment_score = item.get('score', 0)
                                break
                    
                    score_context = f" (Team Score: {commitment_score}/5.0)" if commitment_score else ""
                    system_prompt += f"**🎯 Commitment - Reasons contributing to lack of commitment{score_context}:**\n"
                    for item in commitment_items:
                        reason = item.get("reason", "")
                        vote_count = item.get("vote_count", 0)
                        vote_percentage = item.get("vote_percentage", 0)
                        system_prompt += f"• \"{reason}\" (identified by {vote_count} members - {vote_percentage}%)\n"
                    system_prompt += "\n"
                
                if accountability_items:
                    # Récupérer le score pour contextualiser
                    accountability_score = None
                    if lencioni_data:
                        for item in lencioni_data:
                            if item.get('dysfunction', '').lower() == 'accountability':
                                accountability_score = item.get('score', 0)
                                break
                    
                    score_context = f" (Team Score: {accountability_score}/5.0)" if accountability_score else ""
                    system_prompt += f"**📋 Accountability - Areas to hold one another more accountable{score_context}:**\n"
                    for item in accountability_items:
                        recommendation = item.get("recommendation", "")
                        vote_count = item.get("vote_count", 0)
                        system_prompt += f"• \"{recommendation}\" (identified by {vote_count} team members)\n"
                    system_prompt += "\n"
                
                if results_items:
                    system_prompt += "**🎯 Results - Distractions keeping team from focusing on results:**\n"
                    for item in results_items:
                        distraction = item.get("distraction", "")
                        vote_count = item.get("vote_count", 0)
                        vote_percentage = item.get("vote_percentage", 0)
                        system_prompt += f"• \"{distraction}\" (identified by {vote_count} members - {vote_percentage}%)\n"
                    system_prompt += "\n"
            
            # Ajouter les scores détaillés des questions pour plus de finesse
            if intent_type == "INSIGHT_BLEND" and lencioni_details:
                system_prompt += "\n**Detailed Question Scores (for deeper insights):**\n"
                # Grouper par dysfonction mentionnée
                details_by_dysfunction = {}
                for detail in lencioni_details:
                    dysfunction = detail.get('dysfunction', 'Unknown')
                    if dysfunction not in details_by_dysfunction:
                        details_by_dysfunction[dysfunction] = []
                    details_by_dysfunction[dysfunction].append(detail)
                
                # Afficher seulement pour les dysfonctions demandées
                dysfunctions_mentioned = intent_analysis.get("dysfunctions_mentioned", [])
                for dysfunction in dysfunctions_mentioned:
                    if dysfunction in details_by_dysfunction:
                        system_prompt += f"\n🔍 {dysfunction} - Individual Question Scores:\n"
                        # Trier par score (plus bas = plus problématique)
                        sorted_details = sorted(details_by_dysfunction[dysfunction], key=lambda x: x.get('score', 0))
                        for detail in sorted_details:
                            question = detail.get('question', '')[:100] + "..." if len(detail.get('question', '')) > 100 else detail.get('question', '')
                            score = detail.get('score', 0)
                            level = detail.get('level', 'Unknown')
                            system_prompt += f"  • {question} = {score}/5.0 ({level})\n"
        
        # Toujours inclure les outils/exercices s'ils sont pertinents
        if search_results.get("tools_exercises_content"):
            system_prompt += "\n**Relevant Tools & Exercises:**\n"
            for item in search_results["tools_exercises_content"][:2]:
                system_prompt += f"- {item.get('content', '')[:200]}...\n"
    
    # Adjust guidelines based on intent
    if intent_type == "REPORT_LOOKUP":
        system_prompt += """\n
RESPONSE GUIDELINES FOR REPORT LOOKUP (SIMPLIFIED):

📊 TASK: Present the team's assessment scores clearly and concisely

STRUCTURE YOUR RESPONSE:
- Show the 5 dysfunction scores as listed above (Trust, Conflict, Commitment, Accountability, Results)
- Briefly explain what each dysfunction measures using the model overview
- End with: "For actionable improvement strategies and personalized coaching, ask me 'How can we improve our team dynamics?' or focus on a specific area."

Keep it simple and factual - the goal is to show scores and guide toward actionable insights."""
    
    elif intent_type == "OUT_OF_SCOPE":
        system_prompt += """\n
⚠️ OUT OF SCOPE RESPONSE:

The user's question appears to be about personal or family relationships, not workplace team dynamics.

Please respond with:
"I'm specialized in workplace team dynamics using the Lencioni Five Dysfunctions model. This framework is specifically designed for professional teams in organizational settings.

For family or personal relationship matters, I recommend consulting resources specifically designed for those contexts.

If you have questions about your workplace team dynamics, team performance, or organizational collaboration challenges, I'd be happy to help with those!"
"""
    
    elif intent_type == "LENCIONI_GENERAL_KNOWLEDGE":
        system_prompt += """\n
RESPONSE GUIDELINES FOR GENERAL KNOWLEDGE:

⚠️ SCOPE RESTRICTION:
- Explain Lencioni concepts ONLY in the context of WORKPLACE TEAMS
- If asked about applying to family/personal situations, redirect: "The Lencioni model is designed specifically for workplace teams. For personal relationships, other frameworks may be more appropriate. How can I help you apply this to your professional team?"
- Focus examples on professional team scenarios only

STRUCTURE YOUR RESPONSE:
- Explain the requested Lencioni concept clearly
- Use workplace team examples only
- Connect theory to practical team applications
- Offer to explore how this applies to their specific team context"""
    
    elif intent_type == "INSIGHT_BLEND":
        system_prompt += """\n
RESPONSE GUIDELINES FOR INSIGHT BLEND:

⚠️ SCOPE CHECK - ENFORCE STRICTLY:
Before providing any insights, verify the question is about WORKPLACE TEAM DYNAMICS.
If the question mentions family, personal relationships, or non-work topics:
STOP and redirect: "I specialize in workplace team dynamics only. For personal or family matters, please consult appropriate resources. How can I help with your professional team challenges?"

🎯 COACHING FOCUS: Provide actionable advice that combines expert knowledge with your team's specific insights

PRIORITY ORDER - COMBINE ALL THREE ESSENTIAL SOURCES:
1. FIRST: Reference your team's current Lencioni assessment scores to understand where you are
2. SECOND: Use the Lencioni Framework Recommendations provided above for expert guidance
3. THIRD: Leverage your team's specific insights (what your team members voted for) to personalize advice
4. FOURTH: Refer to the Zest Library for concrete tools and structured approaches

CRITICAL: Always synthesize and connect all three sources of information in your response. Show how the team's scores, the framework recommendations, and the specific team insights work together to create a complete picture.

STRUCTURE YOUR RESPONSE (SYNTHESIZED RECOMMENDATIONS):
- Start with acknowledging their current situation based on assessment data
- ALWAYS REMIND: Address dysfunctions in pyramid order (Trust → Conflict → Commitment → Accountability → Results)
- Use general scores to guide priority: focus on foundational levels (Trust/Conflict) before higher levels

CRITICAL - USE BOTH LEVELS OF TEAM INSIGHTS:

**LEVEL 1 - Team Collective Insights** (Most Important):
- These are your team's specific recommendations, barriers, and focus areas voted by team members
- Trust recommendations, conflict behaviors, commitment barriers, accountability areas, results distractions
- MUST be prominently featured in your recommendations
- Example: "Your team specifically identified [trust area] as a priority - this directly informs our focus"

**LEVEL 2 - Detailed Question Scores**:
- Individual question scores reveal specific strengths and weak points
- Use to validate and enrich the Level 1 insights
- Example: "This aligns with your low scores on [specific questions about trust]"

SYNTHESIS REQUIREMENTS:
- START with Level 1 insights as your primary foundation
- REINFORCE with Level 2 detailed scores for validation
- Transform both into actionable coaching advice (don't just list)
- ALWAYS highlight patterns that emerge from BOTH levels
- Connect team-specific insights to expert Lencioni principles
- Explain WHY these specific insights matter for your team's improvement
- Include ALL relevant insights from both levels - don't omit important team feedback

FORMATTING FOR READABILITY:
- Use clear headings and bullet points
- Group similar insights together
- Highlight highest/lowest voted items for emphasis
- Keep paragraphs short and scannable

KEY PRINCIPLES:
- Give complete picture but organize it well
- Use ALL team-specific voting data but present it clearly
- STRICTLY respect Lencioni pyramid hierarchy (NO EXCEPTIONS):
  • Trust (≤3.0) = STOP, focus here first regardless of other scores
  • If Trust OK, Conflict (≤3.0) = STOP, focus here before higher levels  
  • If Trust + Conflict OK, then consider Commitment, Accountability, Results
- NEVER recommend Commitment if Conflict is low - this violates pyramid logic
- NEVER skip foundational levels even if higher dysfunctions seem "more urgent"
- Balance comprehensiveness with readability
- END WITH INTELLIGENT FOLLOW-UP: When asking which area to focus on, STRICTLY follow pyramid logic:
  1. FIRST check Trust - if low (≤3.0), recommend Trust regardless of other scores
  2. IF Trust is solid, check Conflict - if low (≤3.0), recommend Conflict before higher levels
  3. IF Trust + Conflict are solid, then consider Commitment, etc.
  4. NEVER recommend higher dysfunctions if lower ones are problematic
  Example: "Your Trust (3.5/5.0) is solid, but Conflict (2.1/5.0) needs attention before addressing Commitment. The pyramid requires building this foundation first. Shall we focus on Conflict dynamics?"
  Counter-example: DON'T say "focus on Trust and Commitment" if Conflict is low - that skips a foundational level
- Let the user choose which thread to pull on next

CRITICAL - TOOLS & EXERCISES:
- NEVER invent or suggest your own exercises, tools, or step-by-step action plans
- ALWAYS refer users to the Zest Library for exercises, tools, and structured approaches
- Say something like: "The Zest Library provides specific exercises and tools for this situation"
- If asked for specific exercises or action steps, respond: "I recommend checking the Zest Library which has curated tools and step-by-step approaches specifically for [this dysfunction/situation]"
- DO NOT create, describe, or detail any exercises, action plans, or step-by-step processes yourself
- Focus on insights and analysis, not implementation steps"""
    
    else:
        system_prompt += """\n
RESPONSE GUIDELINES:
- Respond in a practical and actionable manner, providing concrete advice to improve team dynamics
- Adapt to the previous conversation context if relevant
- Focus on leadership coaching, team dynamics, and professional development
- Use the Five Dysfunctions framework to structure your insights
- Provide specific examples and concrete steps when possible
- End with 2-3 specific follow-up questions to guide deeper exploration
- Aim for 150-250 words; keep responses concise and focused
- Encourage users to ask for deeper analysis on specific points rather than giving everything at once

COACHING APPROACH:
- Extract insights from the Lencioni model to guide your advice
- For team challenges: Reference the specific dysfunction(s) involved
- For leadership development: Connect behaviors to team trust and results
- When relevant, cover: team trust, healthy conflict, commitment, accountability, results focus

GUARDRAILS:  
- SCOPE: Focus on Lencioni coaching only
- REDIRECT: For mental health, family issues, clinical conditions, or therapy needs - recommend consulting appropriate professional resources and do not answer. 
- If no relevant info, recommend contacting jean-pierre.aerts@zestforleaders.com."""

    logger.info(f"🏛️ Lencioni prompt built: {len(system_prompt)} characters")
    
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
- If no relevant info, recommend contacting jean-pierre.aerts@zestforleaders.com.

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

# NODE 1: Récupérer le profil MBTI de l'utilisateur
def fetch_user_profile(state: WorkflowState) -> WorkflowState:
    """
    Étape 1: Récupération du profil MBTI depuis Supabase
    - Input: user_email (email) ou user_name (nom complet) comme fallback
    - Process: Recherche par email dans la table profiles
    - Output: user_mbti, user_temperament
    """
    logger.info("🔍 NODE 1: Fetching user MBTI profile...")
    
    # Priorité à l'email, fallback sur user_name
    user_email = state.get("user_email", "").strip()
    user_name = state.get("user_name", "").strip()
    
    # 🔴 MODE TEST JEAN-PIERRE - DÉCOMMENTER POUR TESTER
    # user_email = "jean-pierre.aerts@zestforleaders.com"
    # user_name = "Jean-Pierre Aerts"
    # logger.info("🔴 TEST MODE: Forçage Jean-Pierre Aerts")
    
    if not user_email and not user_name:
        logger.info("❌ No user_email or user_name provided")
        return {**state, "user_mbti": None, "user_temperament": None}
    
    try:
        response = None
        
        # PRIORITÉ 1: Recherche par email si disponible
        if user_email:
            logger.info(f"🔍 Searching by email: '{user_email}'")
            
            # Essayer d'abord avec la colonne 'email'
            try:
                
                # Utiliser ilike qui est moins strict que eq
                query = supabase.table("profiles").select("mbti, temperament, first_name, last_name, email")
                query = query.ilike("email", user_email.lower())  # Forcer en minuscules
                response = query.execute()
                logger.info(f"🔍 Email search (ilike) result: {len(response.data) if response.data else 0} records found")
                
                # Si ça ne marche pas, essayer avec eq
                if not response.data:
                    logger.info(f"🔍 Trying eq search for email...")
                    query = supabase.table("profiles").select("mbti, temperament, first_name, last_name, email")
                    query = query.eq("email", user_email)
                    response = query.execute()
                    logger.info(f"🔍 Email search (eq) result: {len(response.data) if response.data else 0} records found")
                    
                # Dernière tentative : recherche avec filter
                if not response.data:
                    logger.info(f"🔍 Trying filter search for email...")
                    query = supabase.table("profiles").select("mbti, temperament, first_name, last_name, email")
                    query = query.filter("email", "eq", user_email)
                    response = query.execute()
                    logger.info(f"🔍 Email filter search result: {len(response.data) if response.data else 0} records found")
            except Exception as email_error:
                logger.info(f"⚠️ Email search failed (column may not exist): {email_error}")
                
                # Essayer avec 'user_email' comme nom de colonne alternative
                try:
                    query = supabase.table("profiles").select("mbti, temperament, first_name, last_name, user_email")
                    query = query.eq("user_email", user_email)
                    response = query.execute()
                    logger.info(f"🔍 User_email search result: {len(response.data) if response.data else 0} records found")
                except Exception as user_email_error:
                    logger.info(f"⚠️ User_email search also failed: {user_email_error}")
                    response = None
        
        # PRIORITÉ 2: Fallback sur le nom si pas d'email ou pas de résultat
        if (not response or not response.data) and user_name:
            logger.info(f"🔍 Falling back to name search: '{user_name}'")
            
            # Parser le nom complet en prénom/nom
            name_parts = [part.strip() for part in user_name.split() if part.strip()]
            
            if name_parts:
                first_name = name_parts[0]
                last_name = " ".join(name_parts[1:]) if len(name_parts) > 1 else ""
                
                logger.info(f"🔍 Searching by name: first_name='{first_name}', last_name='{last_name}'")
                
                # Recherche par nom avec eq
                query = supabase.table("profiles").select("mbti, temperament, first_name, last_name")
                query = query.eq("first_name", first_name)
                if last_name:
                    query = query.eq("last_name", last_name)
                
                response = query.execute()
                logger.info(f"🔍 Name search (eq) result: {len(response.data) if response.data else 0} records found")
                
                # Si pas de résultat, essayer avec ilike et wildcards
                if not response.data:
                    logger.info("🔍 Trying with ilike and wildcards...")
                    query = supabase.table("profiles").select("mbti, temperament, first_name, last_name")
                    query = query.ilike("first_name", f"%{first_name}%")
                    if last_name:
                        query = query.ilike("last_name", f"%{last_name}%")
                    response = query.execute()
                    logger.info(f"🔍 Name search (ilike) result: {len(response.data) if response.data else 0} records found")
        
        # Vérifier si on a trouvé un profil
        if response and response.data and len(response.data) > 0:
            # Prendre le premier résultat si plusieurs matches
            profile = response.data[0]
            user_mbti = profile.get("mbti")
            user_temperament = profile.get("temperament")
            
            logger.info(f"✅ Profile found: {profile.get('first_name')} {profile.get('last_name')} | MBTI: {user_mbti} | Temperament: {user_temperament}")
            
            # Validation des données récupérées
            if user_mbti and user_temperament:
                return {**state, "user_mbti": user_mbti, "user_temperament": user_temperament}
            else:
                logger.info(f"⚠️ Profile found but missing data: mbti={user_mbti}, temperament={user_temperament}")
                return {**state, "user_mbti": user_mbti, "user_temperament": user_temperament}
        else:
            search_info = f"email: {user_email}" if user_email else f"name: {user_name}"
            logger.info(f"❌ No profile found for {search_info}")
            return {**state, "user_mbti": None, "user_temperament": None}
    
    except Exception as e:
        logger.info(f"❌ Error fetching profile: {type(e).__name__}: {str(e)}")
        # Log plus de détails pour debug
        logger.info(f"   - user_name: '{user_name}'")
        logger.info(f"   - parsed: first_name='{first_name}', last_name='{last_name}'")
        return {**state, "user_mbti": None, "user_temperament": None}

# NODE 2: Récupérer la description du tempérament
def fetch_temperament_description(state: WorkflowState) -> WorkflowState:
    """
    Étape 2: Récupération de la description du tempérament depuis Supabase
    - Input: user_temperament (ex: "NF", "ST", etc.)
    - Process: Requête table temperament avec matching sur colonne type
    - Output: temperament_description
    """
    logger.info("🔍 NODE 2: Fetching temperament description...")
    
    user_temperament = state.get("user_temperament")
    if not user_temperament:
        logger.info("❌ No user_temperament provided")
        return {**state, "temperament_description": None}
    
    logger.info(f"🔍 Searching temperament: type='{user_temperament}'")
    
    try:
        # Construire la requête Supabase
        query = supabase.table("temperament").select("description, temperament").eq("temperament", user_temperament)
        
        logger.info(f"🔍 Temperament query: temperament = '{user_temperament}'")
        
        # Exécuter la requête
        response = query.execute()
        
        logger.info(f"🔍 Raw temperament response: {response}")
        logger.info(f"🔍 Temperament data: {response.data}")
        logger.info(f"🔍 Temperament count: {response.count}")
        
        if response.data and len(response.data) > 0:
            # Prendre le premier résultat
            temperament_row = response.data[0]
            description = temperament_row.get("description")
            
            logger.info(f"✅ Temperament found: {temperament_row.get('temperament')} | Description: {description[:100]}..." if description else "No description")
            
            # Validation des données récupérées
            if description:
                return {**state, "temperament_description": description}
            else:
                logger.info(f"⚠️ Temperament found but no description: {temperament_row}")
                return {**state, "temperament_description": None}
        else:
            logger.info(f"❌ No temperament found for type: {user_temperament}")
            
            # Debug: afficher quelques tempéraments disponibles
            try:
                debug_response = supabase.table("temperament").select("temperament, description").limit(5).execute()
                logger.info(f"🔍 Available temperaments: {[t.get('temperament') for t in debug_response.data] if debug_response.data else []}")
            except Exception as debug_e:
                logger.info(f"🔍 Debug temperament query failed: {debug_e}")
            
            return {**state, "temperament_description": None}
    
    except Exception as e:
        logger.info(f"❌ Error fetching temperament: {type(e).__name__}: {str(e)}")
        logger.info(f"   - user_temperament: '{user_temperament}'")
        return {**state, "temperament_description": None}

# Helper function: Reformule TOUJOURS la query avec le contexte MBTI
def reformulate_query_with_context(user_msg: str, messages_history: List[dict], historical_mbti_types: List[str], user_mbti: str = None) -> str:
    """Reformule systématiquement la query utilisateur en l'enrichissant avec le contexte MBTI disponible"""
    
    logger.info(f"🔄 Reformulating query: '{user_msg}' with MBTI context")
    
    # Construire le contexte des derniers messages
    recent_context = []
    for msg in messages_history[-3:]:  # 3 derniers messages pour contexte
        role = msg.get('role', 'user')
        content = msg.get('content', '')
        if content and len(content) > 5:  # Ignorer les très courts messages
            recent_context.append(f"{role}: {content}")
    
    context_text = "\n".join(recent_context) if recent_context else "No previous conversation"
    
    # Types MBTI disponibles
    all_types = []
    if user_mbti:
        all_types.append(user_mbti)
    if historical_mbti_types:
        all_types.extend([t for t in historical_mbti_types if t != user_mbti])
    
    types_info = ", ".join(set(all_types)) if all_types else "No specific types mentioned"
    
    # Détecter les types MBTI mentionnés dans la question originale
    import re
    mbti_pattern = r'\b(INTJ|INTP|ENTJ|ENTP|INFJ|INFP|ENFJ|ENFP|ISTJ|ISFJ|ESTJ|ESFJ|ISTP|ISFP|ESTP|ESFP)\b'
    mentioned_types_in_query = re.findall(mbti_pattern, user_msg.upper())
    
    logger.info(f"🔍 Types detected in query: {mentioned_types_in_query}")
    logger.info(f"🔍 Historical types: {historical_mbti_types}")
    logger.info(f"🔍 User MBTI: {user_mbti}")
    
    # Priorité: types mentionnés dans la question > types historiques > type utilisateur
    primary_types = []
    if mentioned_types_in_query:
        primary_types = mentioned_types_in_query
        focus_context = f"MBTI Types in Question: {', '.join(mentioned_types_in_query)}"
        logger.info(f"🎯 Using types from question: {mentioned_types_in_query}")
    elif historical_mbti_types:
        primary_types = [t for t in historical_mbti_types if t != user_mbti]
        focus_context = f"MBTI Types from Context: {', '.join(primary_types)}"
        logger.info(f"🎯 Using historical types: {primary_types}")
    else:
        primary_types = [user_mbti] if user_mbti else []
        focus_context = f"User MBTI Type: {user_mbti or 'Unknown'}"
        logger.info(f"🎯 Using user type: {user_mbti}")

    reformulation_prompt = f"""You are an expert at reformulating user queries for MBTI personality type vector search. 

ANALYZE the full conversation context to understand what the user is asking about, then reformulate their query to be more specific and searchable.

User Query: "{user_msg}"
{focus_context}

Full Conversation Context:
{context_text}

UNIVERSAL INSTRUCTIONS:
1. If the user query is short (oui, yes, tell me more, continue, etc.), analyze the ENTIRE conversation to understand what specific topic they want to continue exploring
2. Always include relevant MBTI types in the reformulated query
3. Extract the main topic/theme from the conversation (leadership, communication, stress management, decision-making, etc.)
4. Combine the topic with the MBTI types to create a specific, searchable query
5. Keep the core intent but make it more specific for vector search

REFORMULATION APPROACH:
- Short responses (oui, yes, continue) → Look at the previous AI response to understand the topic, then reformulate as "[MBTI_TYPES] [SPECIFIC_TOPIC]"
- Specific questions with MBTI types → Use those exact types, not the user's type
- General questions → Add the user's MBTI type for personalization
- Follow-up questions → Combine the current question with conversation context

Output ONLY the reformulated query, nothing else."""
    
    try:
        logger.info(f"🔄 Calling isolated LLM for reformulation...")
        # Utiliser l'appel isolé pour éviter le streaming indésirable
        raw_response = isolated_analysis_call(reformulation_prompt)
        reformulated = raw_response.strip().strip('"').strip("'")
        logger.info(f"✅ Reformulated query: '{reformulated}'")
        return reformulated
    except Exception as e:
        logger.info(f"❌ Query reformulation failed: {e}")
        import traceback
        logger.info(f"❌ Full error traceback: {traceback.format_exc()}")
        # Fallback: enrichir manuellement si possible
        if historical_mbti_types and len(user_msg) < 20:
            fallback_query = f"{user_msg} - MBTI types: {', '.join(historical_mbti_types)}"
            logger.info(f"🔄 Using fallback query: '{fallback_query}'")
            return fallback_query
        logger.info(f"🔄 Using original query: '{user_msg}'")
        return user_msg

# NODE 3: Agent MBTI Expert
def mbti_expert_analysis(state: WorkflowState) -> WorkflowState:
    """Reproduit l'étape 4 du workflow n8n - Agent MBTI Expert"""
    logger.info("🧠 NODE 3: MBTI Expert Analysis...")
    
    expert_prompt = """You are an MBTI expert analyzing questions to determine which search tools to use.

You will receive:
- the user's current message,
- conversation history and context,
- recently mentioned MBTI types,
- the user's MBTI profile (e.g., "INTP").

Goal: Analyze the question and determine which search strategy to use, considering conversation continuity.

## ANALYSIS STEPS:

### Step 1: Check for Conversation Continuity
CRITICAL: If the user's message shows continuity (either explicit or implicit):

EXPLICIT CONTINUATIONS:
- "oui", "yes", "d'accord", "ok", "s'il te plaît"
- "tell me more", "continue", "explain further"
- "give me examples", "what about...", "how so?"

IMPLICIT CONTINUATIONS (referring to previously mentioned types):
- "comment ils réagissent", "how they react", "leur comportement"
- "je peux gérer cela", "I can manage this", "dealing with them"
- "comprendre comment ils", "understand how they", "leur façon de"
- Any pronoun references ("ils", "them", "their", "eux") that refer to MBTI types

AND there were MBTI types mentioned in recent conversation, treat this as continuing the previous topic about those types.

### Step 2: Detect Question Type and Tool Strategy
🚨 STEP 2A: MANDATORY MBTI EXTRACTION FIRST

Before any classification, SCAN for 4-letter MBTI codes in:
1. Reformulated query (PRIMARY source)
2. Original user message  
3. Historical context for pronouns ("ils", "them", "they")

Valid codes: INTJ, INTP, ENTJ, ENTP, INFJ, INFP, ENFJ, ENFP, ISTJ, ISFJ, ESTJ, ESFJ, ISTP, ISFP, ESTP, ESFP

🎯 STEP 2B: CLASSIFICATION + TOOL SELECTION (priority order)

**1. GREETING/SMALL_TALK** → NO_TOOLS
- "Bonjour", "Merci", simple acknowledgments WITHOUT continuity
- No MBTI context or types mentioned
- → Instructions: "NO_TOOLS"

**2. COMPARISON** → ABC (User + Generic + Others)  
- Multiple MBTI types mentioned: "difference between X and Y"
- Comparative language: "vs", "compared to", "différence"
- Continuation of previous comparison topic
- → Instructions: "CALL_ABC: Tool A (user) + Tool B (generic user) + Tool C (other types)"
- → Extract ALL non-user types for other_mbti_profiles

**3. OTHER_TYPES** → C (Others only)
🔥 PRIORITY: Any specific MBTI types mentioned (except user's own type)
- Direct focus: "tell me about INFJ", "focus on ESTJ", "comment les ENTP..."  
- Pure information seeking about other types
- Pronouns referring to historical types: "comment ils..." + history contains INFJ
- → Instructions: "CALL_C: Tool C only (other types information)"
- → Extract non-user types for other_mbti_profiles

**4. PERSONAL_DEVELOPMENT** → AB or ABC
- Self-improvement: "How can I...", "Comment puis-je...", "I struggle with..."
- Personal identity: "What is my personality?", "What am I like?", "My MBTI type", "Tell me about myself"
- Self-reflection: "Am I...", "Do I...", "My strengths", "My weaknesses", "My style"
- **4A. WITH other MBTI types** → ABC: "How can I manage INFJs?"
  - → Instructions: "CALL_ABC: Tool A (user) + Tool B (generic user) + Tool C (other types)"  
  - → Extract other types for other_mbti_profiles
- **4B. WITHOUT other types** → AB: "How can I improve my leadership?"
  - → Instructions: "CALL_AB: Tool A (personalized) + Tool B (generic user type)"
  - → other_mbti_profiles: null

**5. GENERAL_MBTI** → D (General knowledge)
- Theory questions: "What is MBTI?", "explain temperaments", "How does MBTI work?"
- Universal concepts: "What are the 16 types?", "MBTI dimensions", "temperament theory"
- No specific types mentioned AND no personal pronouns (I, my, me, myself)
- → Instructions: "CALL_D: Tool D (general MBTI knowledge)"

🚨 ABSOLUTE DECISION TREE:
- MBTI types found + comparative language → COMPARISON (ABC)
- MBTI types found + pure info seeking → OTHER_TYPES (C)  
- MBTI types found + personal question → PERSONAL_DEVELOPMENT (ABC)
- No MBTI types + personal → PERSONAL_DEVELOPMENT (AB)
- No MBTI types + theory → GENERAL_MBTI (D)

### Step 3: MBTI Extraction Protocol
🔍 MANDATORY EXTRACTION PROCESS:

**3A. Primary Extraction Sources (in order):**
1. **Reformulated Query** (highest priority - already context-enriched)
2. **Original User Message**  
3. **Historical Context** (for pronouns like "ils", "them", "they")

**3B. Extraction Rules:**
- Extract ALL 4-letter MBTI codes found
- EXCLUDE user's own MBTI type from other_mbti_profiles  
- Format as comma-separated: "INFJ,ENFJ" 
- NEVER return null if any non-user types are found

**3C. Extraction Examples:**
- Reformulated: "Conflict resolution strategies for INFJ and ENFJ types" → Extract: INFJ,ENFJ
- Original: "I would like to focus on INFJ" → Extract: INFJ  
- Historical: "comment ils réagissent?" + history has ESTJ,ISFJ → Extract: ESTJ,ISFJ
- Mixed: Personal question mentioning ENTP → Extract: ENTP

**3D. Context-Aware Extraction:**
- Pronouns ("ils", "them") + no types in current message → Use historical types
- Explicit types mentioned → Prioritize those over historical context
- Comparative language + types → Extract all types mentioned

🚨 CRITICAL: If Step 2A found MBTI types, they MUST appear in other_mbti_profiles unless they're the user's type

### Step 4: Final Validation and Output

🔍 VALIDATION CHECKLIST:
1. ✅ MBTI types extracted correctly from reformulated query?
2. ✅ User's MBTI type excluded from other_mbti_profiles?  
3. ✅ Tool strategy matches the types found?
4. ✅ Instructions format consistent with examples?

🎯 TOOL STRATEGY SUMMARY:
- **GREETING** → NO_TOOLS
- **COMPARISON** → ABC (user + generic + others) 
- **OTHER_TYPES** → C (others only)
- **PERSONAL_DEVELOPMENT** → AB (self only) or ABC (self + others)
- **GENERAL_MBTI** → D (general knowledge)

🚨 FINAL VERIFICATION:
- If you found MBTI types in Step 2A but other_mbti_profiles is null → ERROR, fix extraction
- If classification is OTHER_TYPES but instructions show CALL_AB → ERROR, should be CALL_C
- If MBTI types present but using CALL_AB → ERROR, should be CALL_ABC or CALL_C

Output format must be valid JSON:

EXAMPLE OUTPUTS:
```json
{
  "question_type": "COMPARISON",
  "instructions": "CALL_ABC: Tool A (user) + Tool B (generic user) + Tool C (other types)",
  "other_mbti_profiles": "ESFJ,ESTJ",
  "continuity_detected": false
}
```

```json
{
  "question_type": "OTHER_TYPES", 
  "instructions": "CALL_C: Tool C only (other types information)",
  "other_mbti_profiles": "ENTP",
  "continuity_detected": false
}
```

```json
{
  "question_type": "PERSONAL_DEVELOPMENT",
  "instructions": "CALL_AB: Tool A (personalized) + Tool B (generic user type)",
  "other_mbti_profiles": null,
  "continuity_detected": false
}
```

REQUIRED FORMAT:
```json
{
  "question_type": "PERSONAL_DEVELOPMENT|COMPARISON|OTHER_TYPES|GENERAL_MBTI|GREETING_SMALL_TALK",
  "instructions": "CALL_AB|CALL_ABC|CALL_C|CALL_D|NO_TOOLS: [description]",
  "other_mbti_profiles": "TYPE1,TYPE2" or null,
  "continuity_detected": true/false
}
```"""
    
    try:
        # Construire le contexte avec messages sérialisables
        user_msg = state.get('user_message', '') or (state.get('messages', [{}])[-1].get('content', '') if state.get('messages') else '')
        
        # Convertir les messages en dictionnaires sérialisables
        messages_history = []
        raw_messages = state.get('messages', [])
        for msg in raw_messages[-10:]:  # Derniers 10 messages
            if hasattr(msg, 'type') and hasattr(msg, 'content'):
                # Objet LangChain Message
                messages_history.append({
                    "role": msg.type if msg.type in ['human', 'ai'] else 'human',
                    "content": msg.content
                })
            elif isinstance(msg, dict):
                # Déjà un dictionnaire
                messages_history.append({
                    "role": msg.get('role', 'human'),
                    "content": msg.get('content', '')
                })
            else:
                # Fallback
                messages_history.append({
                    "role": "human",
                    "content": str(msg)
                })
        
        # Analyser l'historique pour détecter les types MBTI mentionnés récemment
        historical_context = ""
        historical_mbti_types = []
        
        if messages_history:
            # Construire un contexte historique avec les 3 derniers messages
            recent_messages = messages_history[-6:]  # Plus de contexte pour détecter les types
            for msg in recent_messages:
                content = msg.get('content', '').lower()
                historical_context += f" {content}"
                
                # Extraire les types MBTI de chaque message
                import re
                mbti_pattern = r'\b(intj|intp|entj|entp|infj|infp|enfj|enfp|istj|isfj|estj|esfj|istp|isfp|estp|esfp)\b'
                found_types = re.findall(mbti_pattern, content, re.IGNORECASE)
                for mbti_type in found_types:
                    if mbti_type.upper() not in historical_mbti_types:
                        historical_mbti_types.append(mbti_type.upper())
        
        # REFORMULATION SYSTEMATIQUE de la query avec le contexte MBTI
        logger.info(f"🔄 Starting query reformulation for: '{user_msg}'")
        logger.info(f"🔄 Historical MBTI types: {historical_mbti_types}")
        reformulated_query = reformulate_query_with_context(
            user_msg=user_msg,
            messages_history=messages_history,
            historical_mbti_types=historical_mbti_types,
            user_mbti=state.get('user_mbti')
        )
        logger.info(f"🔄 Reformulation completed: '{reformulated_query}'")
        
        # Ajouter la query reformulée au state pour LangGraph Studio Debug
        state = {**state, "reformulated_query_debug": f"Original: '{user_msg}' → Reformulated: '{reformulated_query}'"}
        
        # Construire le contexte enrichi pour l'analyse
        context = f"""Original User Input: {user_msg}

Reformulated Query (use this for analysis): {reformulated_query}

User Name: {state.get('user_name', 'Unknown')}

User MBTI profile: {state.get('user_mbti', 'Unknown')}

Recent Conversation Context:
{historical_context.strip() if historical_context.strip() else "No recent conversation"}

MBTI Types mentioned in recent conversation:
{', '.join(historical_mbti_types) if historical_mbti_types else "None"}

Full Conversation History:
{json.dumps(messages_history[-3:], indent=2) if messages_history else "No previous conversation"}

IMPORTANT: Use the reformulated query to understand what the user wants. If the user's original message was a short response like "oui", "yes", "d'accord", the reformulated query clarifies their intent based on conversation context."""
        
        # 🔑 UTILISER L'APPEL ISOLÉ avec messages séparés pour l'analyse MBTI
        # Cela préserve la structure system/user qui est importante pour la détection
        raw_response = isolated_analysis_call_with_messages(expert_prompt, context)
        
        # DEBUG: Afficher la réponse mais ne pas la streamer
        logger.info(f"🔒 Isolated Analysis Response: '{raw_response}'")
        
        # Parser la réponse JSON
        try:
            import re
            json_match = re.search(r'\{[\s\S]*\}', raw_response)
            if json_match:
                analysis = json.loads(json_match.group(0))
                logger.info(f"✅ MBTI Analysis JSON parsed: {analysis}")
                logger.info(f"🔍 Question type: '{analysis.get('question_type', '')}'")
                logger.info(f"🔍 Instructions field: '{analysis.get('instructions', '')}'")
                logger.info(f"🔍 Other profiles: '{analysis.get('other_mbti_profiles', '')}')")
                
                # VALIDATION 1: Vérifier que l'extraction MBTI a fonctionné pour les comparaisons
                if analysis.get('question_type') == 'COMPARISON' and not analysis.get('other_mbti_profiles'):
                    logger.info(f"⚠️  WARNING: COMPARISON detected but no other_mbti_profiles extracted!")
                    logger.info(f"🔍 Reformulated query was: '{reformulated_query}'")
                    logger.info(f"🔍 User message was: '{user_msg}'")
                
                # VALIDATION 2: Auto-correction si MBTI types détectés mais ignorés par l'IA
                import re
                user_mbti = state.get('user_mbti', '').upper()
                mbti_pattern = r'\b(INTJ|INTP|ENTJ|ENTP|INFJ|INFP|ENFJ|ENFP|ISTJ|ISFJ|ESTJ|ESFJ|ISTP|ISFP|ESTP|ESFP)\b'
                detected_types_local = re.findall(mbti_pattern, reformulated_query, re.IGNORECASE)
                other_types_detected = [t.upper() for t in detected_types_local if t.upper() != user_mbti]
                
                if other_types_detected and not analysis.get('other_mbti_profiles'):
                    logger.info(f"🔄 AUTO-CORRECTION: Types détectés {other_types_detected} ignorés par l'IA")
                    logger.info(f"🔄 Forcer extraction: other_mbti_profiles = {other_types_detected}")
                    
                    # Forcer l'extraction des types détectés
                    analysis['other_mbti_profiles'] = ','.join(other_types_detected)
                    
                    # Adapter les instructions selon le type de question
                    if analysis.get('question_type') == 'PERSONAL_DEVELOPMENT':
                        analysis['instructions'] = 'CALL_ABC: Tool A (user) + Tool B (generic user) + Tool C (other types)'
                        logger.info(f"🔄 PERSONAL_DEVELOPMENT avec autres types → CALL_ABC")
                    elif analysis.get('question_type') == 'OTHER_TYPES':
                        analysis['instructions'] = 'CALL_C: Tool C only (other types information)'
                        logger.info(f"🔄 OTHER_TYPES → CALL_C")
                    
                    logger.info(f"✅ Analysis corrected: {analysis}")
                
                
                # Log l'analyse mais ne pas l'exposer dans le state streamé
                analysis_debug = f"Type: {analysis.get('question_type', 'UNKNOWN')} | Instructions: {analysis.get('instructions', 'NONE')} | Other Profiles: {analysis.get('other_mbti_profiles', 'NULL')}"
                logger.info(f"📋 Analysis Debug: {analysis_debug}")
                
                # Retourner seulement les champs essentiels sans exposer le JSON complet
                return {
                    **state, 
                    "mbti_analysis": analysis,  # Nécessaire pour les autres nœuds
                    "reformulated_query": reformulated_query,
                    # Éviter d'exposer les détails JSON dans les champs streamés
                    "question_type": analysis.get('question_type'),
                    "instructions": analysis.get('instructions'),
                    "other_mbti_profiles": analysis.get('other_mbti_profiles')
                }
        except Exception as parse_error:
            logger.error(f"❌ JSON parsing failed: {parse_error}")
            logger.info(f"🔍 Raw response content: '{response.content}'")
        
        # Fallback with simple pattern matching enhanced with historical context
        logger.info(f"⚠️  Using fallback analysis for: '{user_msg}'")
        
        # Detect MBTI types in current message
        import re
        mbti_pattern = r'\b(INTJ|INTP|ENTJ|ENTP|INFJ|INFP|ENFJ|ENFP|ISTJ|ISFJ|ESTJ|ESFJ|ISTP|ISFP|ESTP|ESFP)\b'
        mentioned_types = re.findall(mbti_pattern, user_msg.upper())
        
        # Check for continuation phrases
        continuation_phrases = [
            'oui', 'yes', 'ok', 'd\'accord', 's\'il te plaît', 'please',
            'tell me more', 'continue', 'explain', 'give me examples',
            'what about', 'how so', 'vraiment', 'comment', 'pourquoi'
        ]
        is_continuation = any(phrase in user_msg.lower() for phrase in continuation_phrases) and len(user_msg) < 50
        
        # Simple greeting detection (but not if it's continuation)
        greetings = ['bonjour', 'hello', 'salut', 'bonsoir', 'merci', 'thanks', 'au revoir', 'goodbye']
        is_greeting = any(greeting in user_msg.lower() for greeting in greetings) and len(user_msg) < 50
        
        # If continuation and we have historical types, use them
        if is_continuation and historical_mbti_types:
            logger.info(f"🔄 Detected continuation with historical types: {historical_mbti_types}")
            other_types = [t for t in historical_mbti_types if t != state.get('user_mbti')]
            if other_types:
                analysis = {
                    "question_type": "COMPARISON",
                    "instructions": "CALL_ABC: Tool A + Tool B + Tool C",
                    "other_mbti_profiles": ','.join(other_types),
                    "continuity_detected": True
                }
            else:
                analysis = {
                    "question_type": "PERSONAL_DEVELOPMENT", 
                    "instructions": "CALL_AB: Tool A + Tool B",
                    "other_mbti_profiles": None,
                    "continuity_detected": True
                }
        # Regular greeting without context
        elif is_greeting and not is_continuation:
            analysis = {
                "question_type": "GREETING_SMALL_TALK",
                "instructions": "NO_TOOLS: Provide a friendly greeting",
                "other_mbti_profiles": None,
                "continuity_detected": False
            }
        # Personal development keywords
        elif any(word in user_msg.lower() for word in ['i ', 'me ', 'my ', 'myself', 'je ', 'mon ', 'ma ']):
            all_types = mentioned_types + historical_mbti_types
            other_types = [t for t in all_types if t != state.get('user_mbti')]
            other_types = list(set(other_types))  # Déduplication
            
            if other_types:
                analysis = {
                    "question_type": "COMPARISON",
                    "instructions": "CALL_ABC: Tool A + Tool B + Tool C",
                    "other_mbti_profiles": ','.join(other_types),
                    "continuity_detected": bool(historical_mbti_types)
                }
            else:
                analysis = {
                    "question_type": "PERSONAL_DEVELOPMENT",
                    "instructions": "CALL_AB: Tool A + Tool B",
                    "other_mbti_profiles": None,
                    "continuity_detected": bool(historical_mbti_types)
                }
        # Check for comparison keywords first
        elif mentioned_types or historical_mbti_types:
            all_types = mentioned_types + historical_mbti_types
            other_types = [t for t in all_types if t != state.get('user_mbti')]
            other_types = list(set(other_types))  # Déduplication
            
            # Check for comparison keywords
            comparison_keywords = [
                'différence', 'difference', 'compare', 'comparison', 'vs', 'versus', 
                'entre', 'between', 'and', 'et', 'contrast', 'contraste'
            ]
            is_comparison = any(keyword in user_msg.lower() for keyword in comparison_keywords)
            
            if other_types and is_comparison:
                analysis = {
                    "question_type": "COMPARISON",
                    "instructions": "CALL_ABC: Tool A + Tool B + Tool C",
                    "other_mbti_profiles": ','.join(other_types),
                    "continuity_detected": bool(historical_mbti_types)
                }
            elif other_types:
                analysis = {
                    "question_type": "OTHER_TYPES",
                    "instructions": "CALL_C: Tool C only",
                    "other_mbti_profiles": ','.join(other_types),
                    "continuity_detected": bool(historical_mbti_types)
                }
            else:
                analysis = {
                    "question_type": "PERSONAL_DEVELOPMENT",
                    "instructions": "CALL_AB: Tool A + Tool B", 
                    "other_mbti_profiles": None,
                    "continuity_detected": bool(historical_mbti_types)
                }
        # General MBTI
        else:
            analysis = {
                "question_type": "GENERAL_MBTI",
                "instructions": "CALL_D: Tool D",
                "other_mbti_profiles": None,
                "continuity_detected": False
            }
        
        logger.info(f"🔍 Fallback analysis result: {analysis}")
        
        # Ajouter l'analyse fallback au debug pour LangGraph Studio
        analysis_debug = f"FALLBACK | Type: {analysis.get('question_type', 'UNKNOWN')} | Instructions: {analysis.get('instructions', 'NONE')} | Other Profiles: {analysis.get('other_mbti_profiles', 'NULL')}"
        state = {**state, "reformulated_query_debug": f"{state.get('reformulated_query_debug', '')} | Analysis: {analysis_debug}"}
        
        return {**state, "mbti_analysis": analysis, "reformulated_query": reformulated_query}
    
    except Exception as e:
        logger.info(f"❌ Error in MBTI analysis: {e}")
        # Fallback par défaut
        error_analysis = {
            "instructions": "Call Tool D: get contextual and general info",
            "other_mbti_profiles": None
        }
        
        # Ajouter l'erreur au debug pour LangGraph Studio
        analysis_debug = f"ERROR | Instructions: {error_analysis.get('instructions', 'NONE')} | Error: {str(e)}"
        state = {**state, "reformulated_query_debug": f"{state.get('reformulated_query_debug', '')} | Analysis: {analysis_debug}"}
        
        return {**state, "mbti_analysis": error_analysis, "reformulated_query": reformulated_query or user_msg}

# NODE 3.5: Temperament Facet Analyzer
def analyze_temperament_facets(state: WorkflowState) -> WorkflowState:
    """
    Analyse la query reformulée pour identifier:
    1. Les facettes de tempérament pertinentes (values, strengths, leadership_style, etc.)
    2. Si c'est pour l'utilisateur ou d'autres types MBTI
    3. Les tempéraments à rechercher (SJ, SP, NF, NT)
    """
    # 🔥 DEBUG MODE: Activer pour tester l'analyse sans affecter les recherches existantes
    # True = Mode debug avec logs détaillés, les recherches de tempéraments sont désactivées  
    # False = Mode production, les recherches de tempéraments sont activées
    DEBUG_MODE = False  # ⚠️ Mode production activé
    
    if DEBUG_MODE:
        logger.info("🔥 DEBUG MODE ACTIVÉ - NODE 3.5 en mode log uniquement")
        logger.info("   ⚠️ Les recherches de tempéraments ne seront pas intégrées aux outils A, B, C, D")
        logger.info("   ℹ️ Mettre DEBUG_MODE = False pour activer les recherches")
    
    logger.info("🔍 NODE 3.5: Temperament Facet Analysis...")
    
    # Mapping MBTI types to temperaments
    MBTI_TO_TEMPERAMENT = {
        # SJ - Guardian
        "ISTJ": "SJ", "ISFJ": "SJ", "ESTJ": "SJ", "ESFJ": "SJ",
        # SP - Commando/Artisan
        "ISTP": "SP", "ISFP": "SP", "ESTP": "SP", "ESFP": "SP",
        # NF - Catalyst/Idealist
        "INFJ": "NF", "INFP": "NF", "ENFJ": "NF", "ENFP": "NF",
        # NT - Architect/Rational
        "INTJ": "NT", "INTP": "NT", "ENTJ": "NT", "ENTP": "NT"
    }
    
    # Classification IA des facettes (remplace l'approche par mots-clés)
    def classify_facets_with_ai(query: str, user_mbti: str = None) -> List[str]:
        """
        Utilise l'IA pour identifier intelligemment les facettes de tempérament pertinentes
        Multilingue et adaptable à tout type de question
        """
        try:
            prompt = f"""Vous êtes un expert MBTI. Analysez cette question et identifiez les 3-4 facettes de tempérament les plus pertinentes.

FACETTES DISPONIBLES avec définitions:

FACETTES PRINCIPALES:
• overview – Vue d'ensemble : Intro sur la vision et les priorités du tempérament, ce sur quoi il se focalise
• mottos – Mots d'ordre : Phrases clés ou slogans résumant leur état d'esprit
• values – Valeurs : Principes et croyances fondamentales qu'ils défendent
• desires – Désirs : Ce qu'ils recherchent ou souhaitent accomplir dans leur vie
• needs – Besoins : Conditions essentielles pour donner leur meilleur
• aversions – Aversions : Ce qu'ils évitent ou supportent mal (SP, NF, NT uniquement)
• learning_style – Style d'apprentissage : Manière préférée d'apprendre et d'acquérir des compétences
• leadership_style – Style de leadership : Façon de diriger, de prendre des décisions et de mobiliser les autres
• strengths – Forces : Points forts, talents naturels et atouts principaux
• recognition – Reconnaissance souhaitée : Formes d'appréciation ou de reconnaissance qu'ils valorisent
• general_traits – Traits généraux : Caractéristiques communes à tous les membres du tempérament
• weaknesses – Faiblesses potentielles : Limites ou zones de vulnérabilité typiques
• recommendations – Recommandations : Conseils pratiques pour progresser et mieux interagir avec eux

CONTEXTES SPÉCIFIQUES:
• context_family – Contexte familial : Comportement en famille, avec les enfants, relations parentales
• context_education – Contexte éducatif : Apprentissage, école, formation, développement des compétences
• context_work – Contexte professionnel : Travail, carrière, environnement professionnel, productivité
• context_authority – Exercice de l'autorité : Leadership, hiérarchie, prise de décision, management
• context_sectors – Secteurs d'activité : Domaines professionnels privilégiés, industries, métiers
• context_time – Relation au temps : Gestion du temps, ponctualité, planning, organisation temporelle
• context_money – Relation à l'argent : Gestion financière, attitude envers l'argent, priorités économiques

QUESTION: "{query}"
{f"TYPE MBTI UTILISATEUR: {user_mbti}" if user_mbti else ""}

Analysez l'intention de la question et choisissez les 3-4 facettes les plus pertinentes pour y répondre efficacement.

Répondez uniquement par les noms de facettes séparés par des virgules, sans explication.
Exemple: strengths,recommendations,context_work"""

            from openai import OpenAI
            client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,  # Augmenté de 0.3 à 0.7 pour plus de flexibilité
                max_tokens=50
            )
            
            facets_str = response.choices[0].message.content.strip()
            facets = [f.strip() for f in facets_str.split(',') if f.strip()]
            
            logger.info(f"🤖 IA Classification: {facets}")
            return facets[:3]  # Limiter à 3 facettes max
            
        except Exception as e:
            logger.info(f"⚠️ Erreur classification IA: {e}")
            # Fallback vers facettes par défaut
            return ["overview", "general_traits"]
    
    # Récupérer les données nécessaires
    reformulated_query = state.get('reformulated_query', '')
    user_mbti = state.get('user_mbti')
    mbti_analysis = state.get('mbti_analysis', {})
    other_mbti_profiles = mbti_analysis.get('other_mbti_profiles', [])
    
    # DEBUG: Vérifier pourquoi la query est vide
    logger.info(f"🔍 STATE KEYS: {list(state.keys())}")
    logger.info(f"📝 Reformulated query: '{reformulated_query}'")
    logger.info(f"👤 User MBTI: {user_mbti}")
    logger.info(f"🎯 Other profiles: {other_mbti_profiles}")
    logger.info(f"📊 Full mbti_analysis: {mbti_analysis}")
    
    # Essayer d'utiliser user_message comme fallback si reformulated_query est vide
    if not reformulated_query.strip():
        user_message = state.get('user_message', '')
        logger.info(f"⚠️ Reformulated query vide, utilisation user_message: '{user_message}'")
        reformulated_query = user_message
    
    # Déterminer les tempéraments à rechercher et les cibles
    temperaments_to_search = set()
    search_for_user = False
    search_for_others = False
    question_type = mbti_analysis.get('question_type', '')
    
    logger.info(f"🔍 Analyzing search targets for question type: {question_type}")
    
    # LOGIQUE BASÉE SUR LE TYPE DE QUESTION ET LE CONTENU
    
    # 1. QUESTIONS PERSONNELLES: "How can I...", "Comment puis-je...", etc.
    personal_indicators = ['how can i', 'comment puis-je', 'comment je peux', 'how do i', 'how should i']
    is_personal_question = any(indicator in reformulated_query.lower() for indicator in personal_indicators)
    
    # 2. Si l'utilisateur est mentionné ET d'autres types
    if other_mbti_profiles and is_personal_question:
        # Cas: "How can I manage conflict with ESTP" → Chercher pour USER + OTHERS
        search_for_user = True
        search_for_others = True
        logger.info(f"🎯 Personal question with other types → Search for USER + OTHERS")
        
    elif other_mbti_profiles and question_type == 'COMPARISON':
        # Cas: "Difference between ISFP and ESTP" → Chercher pour USER + OTHERS
        search_for_user = True
        search_for_others = True
        logger.info(f"🎯 Comparison question → Search for USER + OTHERS")
        
    elif other_mbti_profiles and not is_personal_question:
        # Cas: "How do ESTP handle conflict" → Chercher seulement OTHERS
        search_for_user = False
        search_for_others = True
        logger.info(f"🎯 Others-focused question → Search for OTHERS only")
        
    elif not other_mbti_profiles:
        # Cas: "How can I improve my leadership" → Chercher seulement USER
        search_for_user = True
        search_for_others = False
        logger.info(f"🎯 User-only question → Search for USER only")
    
    # 3. AJOUTER LES TEMPÉRAMENTS CORRESPONDANTS
    
    # Ajouter tempérament utilisateur si nécessaire
    if search_for_user and user_mbti and user_mbti in MBTI_TO_TEMPERAMENT:
        user_temperament = MBTI_TO_TEMPERAMENT[user_mbti]
        temperaments_to_search.add(user_temperament)
        logger.info(f"  ➕ Adding user temperament {user_temperament} ({user_mbti})")
    
    # Ajouter tempéraments des autres types si nécessaire  
    if search_for_others and other_mbti_profiles:
        logger.info(f"🔍 Processing other_mbti_profiles: {other_mbti_profiles} (type: {type(other_mbti_profiles)})")
        
        # Gérer différents formats possibles
        if isinstance(other_mbti_profiles, str):
            mbti_types = [t.strip() for t in other_mbti_profiles.split(',')]
        elif isinstance(other_mbti_profiles, list):
            mbti_types = other_mbti_profiles
        else:
            mbti_types = []
            
        logger.info(f"🔍 MBTI types to process: {mbti_types}")
        
        for mbti_type in mbti_types:
            if mbti_type in MBTI_TO_TEMPERAMENT:
                other_temperament = MBTI_TO_TEMPERAMENT[mbti_type]
                temperaments_to_search.add(other_temperament)
                logger.info(f"  ➕ Adding other temperament {other_temperament} ({mbti_type})")
            else:
                logger.info(f"  ⚠️ MBTI type '{mbti_type}' not found in mapping")
    
    # Identifier les facettes pertinentes avec l'IA
    logger.info(f"🔍 Query utilisée pour analyse facettes: '{reformulated_query}'")
    
    # Utiliser la classification IA pour identifier les facettes
    relevant_facets = classify_facets_with_ai(reformulated_query, user_mbti)
    
    # Fallback si la classification IA échoue ou retourne des facettes vides
    if not relevant_facets or relevant_facets == ["overview", "general_traits"]:
        logger.info("⚠️ Classification IA non optimale, utilisation de la logique de fallback")
        query_lower = reformulated_query.lower()
        
        # Logique de fallback simplifiée et intelligente
        if any(word in query_lower for word in ["leader", "manage", "diriger", "authority", "équipe", "team"]):
            relevant_facets = ["leadership_style", "context_authority", "strengths"]
        elif any(word in query_lower for word in ["learn", "apprendre", "étudier", "education", "formation"]):
            relevant_facets = ["learning_style", "context_education"]
        elif any(word in query_lower for word in ["stress", "pressure", "difficile", "challenge", "problème", "struggle"]):
            relevant_facets = ["weaknesses", "recommendations", "aversions"]
        elif any(word in query_lower for word in ["travail", "work", "job", "career", "professional", "productivité", "productivity"]):
            relevant_facets = ["strengths", "recommendations", "context_work"]
        elif any(word in query_lower for word in ["famille", "family", "enfant", "children", "parent", "familial"]):
            relevant_facets = ["context_family", "general_traits"]
        elif any(word in query_lower for word in ["value", "valeur", "belief", "principe", "important", "matter"]):
            relevant_facets = ["values", "needs", "desires"]
        elif any(word in query_lower for word in ["qui suis", "who am", "myself", "me comprendre", "understand me"]):
            relevant_facets = ["overview", "general_traits", "values"]
        else:
            # Défaut basé sur le type de question
            relevant_facets = ["overview", "general_traits"]
    
    # Garder seulement les facettes valides et limiter à 3
    valid_facets = [
        "overview", "mottos", "values", "desires", "needs", "aversions", 
        "learning_style", "leadership_style", "strengths", "recognition", 
        "general_traits", "context_family", "context_education", "context_work",
        "context_authority", "context_sectors", "context_time", "context_money",
        "weaknesses", "recommendations"
    ]
    relevant_facets = [f for f in relevant_facets if f in valid_facets][:3]
    
    logger.info(f"✅ Identified facets: {relevant_facets}")
    logger.info(f"🎯 Temperaments to search: {list(temperaments_to_search)}")
    logger.info(f"👤 Search for user: {search_for_user}")
    logger.info(f"👥 Search for others: {search_for_others}")
    
    # Créer le résultat de l'analyse
    temperament_analysis = {
        "temperaments_to_search": list(temperaments_to_search),
        "relevant_facets": relevant_facets,
        "search_for_user": search_for_user,
        "search_for_others": search_for_others,
        "classification_method": "AI",  # Indiquer que c'est la classification IA
        "debug_mode": DEBUG_MODE
    }
    
    # 🔥 DEBUG LOGGING - Affichage détaillé pour test
    if DEBUG_MODE:
        logger.info("\n" + "="*60)
        logger.info("🔥 DEBUG - ANALYSE DES TEMPÉRAMENTS")
        logger.info("="*60)
        logger.info(f"📝 Query analysée: '{reformulated_query}'")
        logger.info(f"👤 MBTI utilisateur: {user_mbti}")
        logger.info(f"👥 Autres profils MBTI: {other_mbti_profiles}")
        logger.info(f"🎯 Tempéraments identifiés: {list(temperaments_to_search)}")
        logger.info(f"📊 Facettes pertinentes: {relevant_facets}")
        logger.info(f"🔍 Rechercher pour utilisateur: {search_for_user}")
        logger.info(f"🔍 Rechercher pour autres: {search_for_others}")
        logger.info("🤖 Classification IA utilisée pour identifier les facettes pertinentes")
        
        # Simulation de recherche temperament (sans vraiment chercher)
        if list(temperaments_to_search) and relevant_facets:
            logger.info("\n🔍 SIMULATION - Recherches qui seraient effectuées:")
            for temp in list(temperaments_to_search)[:2]:
                for facet in relevant_facets[:2]:
                    logger.info(f"   📄 {temp} + {facet}")
        
        logger.info("="*60)
        logger.info("🔥 FIN DEBUG - NODE 3.5")
        logger.info("="*60 + "\n")
    
    # En mode debug, on peut désactiver les recherches réelles
    if DEBUG_MODE:
        # Désactiver temporairement l'analyse pour les outils
        logger.info("🔥 DEBUG: Analyse des tempéraments enregistrée mais pas utilisée dans les recherches")
        return {
            **state,
            "temperament_analysis": None  # Désactivé pour le debug
        }
    else:
        # Mode normal - ajouter au state
        return {
            **state,
            "temperament_analysis": temperament_analysis
        }

# Helper function: Récupération directe des tempéraments par filtres (pas de recherche vectorielle)
def search_temperaments_documents(temperament_analysis: Dict, limit: int = 5) -> List[Dict]:
    """
    Récupère directement les documents de tempéraments par filtres métadonnées
    (pas de recherche vectorielle, juste récupération par tempérament + facette)
    """
    if not temperament_analysis:
        return []
    
    logger.info("🏛️ Fetching temperaments documents by metadata filters...")
    
    temperaments = temperament_analysis.get("temperaments_to_search", [])
    facets = temperament_analysis.get("relevant_facets", [])
    
    if not temperaments or not facets:
        logger.info("  ⚠️ No temperaments or facets to fetch")
        return []
    
    logger.info(f"  📊 Processing {len(temperaments)} temperaments: {temperaments}")
    logger.info(f"  📊 Processing {len(facets)} facets: {facets}")
    
    all_results = []
    
    try:
        # Récupération directe par filtres sans recherche vectorielle
        for temperament in temperaments:  # Traiter tous les tempéraments demandés
            for facet in facets[:3]:  # Limiter à 3 facettes max
                logger.info(f"  🔍 Fetching {temperament}/{facet}...")
                
                # Requête directe sur la table documents_content_test
                try:
                    response = supabase.table('documents_content_test').select('content,metadata').eq(
                        'metadata->>mbti_family', temperament
                    ).eq(
                        'metadata->>facet', facet
                    ).eq(
                        'metadata->>document_key', 'MBTI_Temperaments_v2'
                    ).limit(2).execute()
                    
                    if response.data:
                        for item in response.data:
                            # Formater le résultat
                            result = {
                                'content': item.get('content', ''),
                                'metadata': item.get('metadata', {}),
                                'temperament': temperament,
                                'facet': facet,
                                'similarity': 1.0  # Score fixe car pas de recherche vectorielle
                            }
                            all_results.append(result)
                            
                        logger.info(f"    ✅ Found {len(response.data)} chunks for {temperament}/{facet}")
                    else:
                        logger.info(f"    ⚠️ No content found for {temperament}/{facet}")
                        
                except Exception as inner_e:
                    logger.info(f"    ❌ Error fetching {temperament}/{facet}: {inner_e}")
                    
    except Exception as e:
        logger.info(f"  ❌ Error fetching temperaments: {e}")
    
    # Retourner directement tous les résultats (plus de déduplication défectueuse)
    logger.info(f"✅ Total temperament documents found: {len(all_results)}")
    
    return all_results[:limit]

# NODE 4: Router conditionnel basé sur l'analyse MBTI
def route_to_tools(state: WorkflowState) -> str:
    """Router qui détermine quels outils exécuter"""
    logger.info("🔀 NODE 4: Routing to appropriate tools...")
    
    analysis = state.get("mbti_analysis", {})
    instructions = analysis.get("instructions", "").upper()
    question_type = analysis.get("question_type", "").upper()
    
    logger.info(f"🔍 Question type: '{question_type}'")
    logger.info(f"🔍 Analysis instructions: '{instructions}'")
    
    # Routing basé sur les nouvelles instructions standardisées
    if "NO_TOOLS" in instructions or question_type == "GREETING_SMALL_TALK":
        logger.info("➡️  Routing to: no_tools")
        return "no_tools"
    elif "CALL_ABC" in instructions or question_type == "COMPARISON":
        logger.info("➡️  Routing to: execute_tools_abc")
        return "execute_tools_abc"
    elif "CALL_AB" in instructions or question_type == "PERSONAL_DEVELOPMENT":
        logger.info("➡️  Routing to: execute_tools_ab")
        return "execute_tools_ab" 
    elif "CALL_C" in instructions or question_type == "OTHER_TYPES":
        logger.info("➡️  Routing to: execute_tools_c")
        return "execute_tools_c"
    elif "CALL_D" in instructions or question_type == "GENERAL_MBTI":
        logger.info("➡️  Routing to: execute_tools_d")
        return "execute_tools_d"
    else:
        # Fallback intelligent basé sur la présence de données
        if state.get('user_mbti'):
            logger.info(f"⚠️  No clear match, defaulting to user-focused search (execute_tools_ab)")
            return "execute_tools_ab"
        else:
            logger.info(f"⚠️  No clear match, defaulting to general search (execute_tools_d)")
            return "execute_tools_d"

# Helper function for Supabase vector search using match functions
def perform_supabase_vector_search(query: str, match_function: str, metadata_filters: dict = None, limit: int = 5) -> List[Dict]:
    """
    Perform vector search in Supabase using match functions
    Args:
        query: User's question/message
        match_function: Supabase function name (e.g., 'match_participants', 'match_documents')
        metadata_filters: Dictionary of metadata filters
        limit: Number of results to return
    Returns:
        List of search results with content, metadata, and similarity scores
    """
    try:
        # Generate embedding for the query
        logger.info(f"🔍 Generating embedding for query: '{query[:50]}...'")
        query_embedding = embeddings.embed_query(query)
        logger.info(f"✅ Embedding generated, length: {len(query_embedding)}")
        
        # Prepare function parameters
        params = {
            'query_embedding': query_embedding,
            'match_count': limit,
            'filter': metadata_filters or {}  # Send filters as single JSON object
        }
        
        logger.info(f"🔍 Calling Supabase function '{match_function}' with params:")
        logger.info(f"   - match_count: {params['match_count']}")
        logger.info(f"   - filter: {params['filter']}")
        logger.info(f"   - query_embedding: [vector of {len(query_embedding)} dimensions]")
        
        # Call the Supabase function
        logger.info(f"🔄 Executing supabase.rpc('{match_function}', params)...")
        # Avoid logging the full embedding vector in console; only show dimension count
        safe_params_log = {
            'match_count': params['match_count'],
            'filter': params['filter'],
            'query_embedding': f"[vector of {len(query_embedding)} dimensions]",
        }
        logger.info(f"🔍 Params summary: {safe_params_log}")
        try:
            response = supabase.rpc(match_function, params).execute()
            logger.info(f"✅ RPC call successful, got response")
        except Exception as rpc_error:
            logger.info(f"❌ RPC function '{match_function}' failed: {rpc_error}")
            logger.info("🔄 Trying direct table query as fallback...")
            
            # Fallback: direct table query without RPC function
            table_name = "participants_content_test" if "participants" in match_function else "documents_content_test"
            
            query = supabase.table(table_name).select("content, metadata")
            
            # Add metadata filters with correct JSON syntax
            if metadata_filters:
                for key, value in metadata_filters.items():
                    query = query.filter(f"metadata->>'{key}'", "eq", value)
            
            response = query.limit(limit).execute()
            logger.info(f"✅ Fallback query executed on table '{table_name}'")
            
            # Debug: Check if table has any data at all
            total_count_response = supabase.table(table_name).select("id", count="exact").execute()
            logger.info(f"🔍 Total rows in table '{table_name}': {total_count_response.count}")
            
            # Debug: Show sample rows without filters
            sample_response = supabase.table(table_name).select("content, metadata").limit(3).execute()
            logger.info(f"🔍 Sample rows in '{table_name}': {len(sample_response.data)} found")
            for i, row in enumerate(sample_response.data[:2]):
                logger.info(f"   [{i+1}] Content: {row.get('content', '')[:50]}...")
                logger.info(f"       Metadata: {row.get('metadata', {})}")
                
            if not sample_response.data:
                logger.info(f"❌ Table '{table_name}' appears to be empty!")
        logger.info(f"✅ Supabase response received")
        logger.info(f"🔍 Response.data type: {type(response.data)}")
        # logger.info(f"🔍 Response.data: {response.data}")  # Commenté car trop verbeux
        logger.info(f"🔍 Response data: {len(response.data) if response.data else 0} results")
        
        # Debug: Always check table contents if no results
        if not response.data:
            logger.info(f"🔍 DEBUG: No results found, testing individual filters...")
            table_name = "participants_content_test" if "participants" in match_function else "documents_content_test"
            
            try:
                # Test each filter individually with correct JSON syntax
                if metadata_filters:
                    for key, value in metadata_filters.items():
                        logger.info(f"🔍 Testing filter: {key} = '{value}'")
                        # Try different approaches for JSON filtering
                        test_query = supabase.table(table_name).select("content, metadata").filter(f"metadata->>'{key}'", "eq", value).limit(2)
                        test_response = test_query.execute()
                        logger.info(f"   → Found {len(test_response.data)} rows with filter() method")
                        if test_response.data:
                            for i, row in enumerate(test_response.data[:1]):
                                logger.info(f"   [{i+1}] {row.get('content', '')[:50]}...")
                                logger.info(f"       {key}: {row.get('metadata', {}).get(key, 'NOT_FOUND')}")
                
                # Test combined filters with correct syntax
                logger.info(f"🔍 Testing COMBINED filters with filter() method...")
                combined_query = supabase.table(table_name).select("content, metadata")
                for key, value in metadata_filters.items() if metadata_filters else []:
                    combined_query = combined_query.filter(f"metadata->>'{key}'", "eq", value)
                combined_response = combined_query.limit(2).execute()
                logger.info(f"   → Combined filters found: {len(combined_response.data)} rows")
                
            except Exception as debug_error:
                logger.info(f"❌ Debug query failed: {debug_error}")
        
        results = []
        if response.data:
            logger.info(f"✅ Found {len(response.data)} results from Supabase")
            for i, item in enumerate(response.data):
                result = {
                    "content": item.get("content", ""),
                    "metadata": item.get("metadata", {}),
                    "similarity": item.get("similarity", 0.0)
                }
                results.append(result)
                logger.info(f"   [{i+1}] Content: {result['content'][:100]}...")
                logger.info(f"       Similarity: {result['similarity']}")
        else:
            logger.info("❌ No results found in response.data")
        
        logger.info(f"🔍 Returning {len(results)} results")
        return results
        
    except Exception as e:
        logger.info(f"❌ Vector search error with function {match_function}: {e}")
        return []

# Helper to sanitize and constrain vector results before injecting into prompts
def sanitize_vector_results(
    results: List[Dict],
    required_filters: Optional[Dict] = None,
    top_k: int = 3,
    min_similarity: Optional[float] = 0.0,
    max_chars_per_item: int = 600,
    max_total_chars: int = 2000,
) -> List[Dict]:
    """Filter, deduplicate, sort and truncate vector search results.

    - Enforces metadata matches to ensure content comes from the intended section(s)
    - Drops low-similarity items (when similarity available)
    - Sorts by similarity desc when available
    - Deduplicates by normalized content prefix
    - Truncates content to keep prompt size under control
    - Applies a global character budget across items
    """
    if not results:
        return []

    # 1) Enforce required metadata filters
    filtered: List[Dict] = []
    for item in results:
        metadata = item.get("metadata", {}) or {}
        if required_filters:
            metadata_ok = True
            for key, expected_value in required_filters.items():
                # Compare as strings to avoid type mismatches (e.g., int vs str)
                if str(metadata.get(key)) != str(expected_value):
                    metadata_ok = False
                    break
            if not metadata_ok:
                logger.info(f"⚠️  Dropping item due to metadata mismatch. Required={required_filters} | Got={metadata}")
                continue

        # 2) Similarity threshold if available
        if min_similarity is not None and "similarity" in item:
            if float(item.get("similarity", 0.0)) < float(min_similarity):
                continue

        filtered.append(item)

    if not filtered:
        logger.info("⚠️  No results left after metadata/similarity filtering")
        return []

    # 3) Sort by similarity when present
    filtered.sort(key=lambda r: r.get("similarity", 0.0), reverse=True)

    # 4) Deduplicate by normalized content prefix
    def _normalize_prefix(text: str) -> str:
        return " ".join((text or "").lower().split())[:400]

    seen_norms = set()
    deduped: List[Dict] = []
    for item in filtered:
        content = (item.get("content") or "").strip()
        norm = _normalize_prefix(content)
        if norm in seen_norms:
            continue
        seen_norms.add(norm)
        deduped.append(item)
        if len(deduped) >= top_k:
            break

    # 5) Enforce total character budget across items without truncation
    total_chars = 0
    budgeted: List[Dict] = []
    for item in deduped:
        content = item.get("content", "")
        if total_chars + len(content) > max_total_chars:
            # If nothing selected yet, include the first item even if it exceeds the budget
            if not budgeted:
                budgeted.append(item)
                total_chars += len(content)
            # Otherwise skip this item and try next ones (they may be shorter)
            else:
                continue
        else:
            budgeted.append(item)
            total_chars += len(content)

    logger.info(
        f"🔎 Sanitized results: input={len(results)} -> kept={len(budgeted)} "
        f"(top_k={top_k}, min_sim={min_similarity}, max_chars/item={max_chars_per_item}, total_budget={max_total_chars})"
    )
    return budgeted

# NODE 5A: Exécuter Tools A + B (User only)
def execute_tools_ab(state: WorkflowState) -> WorkflowState:
    """Execute 2 vector searches: Tool A (participants) + Tool B (documents), then combine results"""
    logger.info("🔧 NODE 5A: Executing 2 vector searches - Tools A + B...")
    
    personalized_content = []  # Tool A: Vector search results from participants collection
    generic_content = []       # Tool B: Vector search results from documents collection
    
    try:
        # Utiliser la query reformulée si disponible, sinon fallback sur le message original
        search_query = state.get('reformulated_query') or state.get('user_message', '') or (state.get('messages', [{}])[-1].get('content', '') if state.get('messages') else '')
        user_msg = search_query  # Pour compatibilité avec le reste du code
        logger.info(f"🔍 Using search query: '{search_query}'")
        
        # Tool A: Vector search in participants_content_test with folder_path filter
        logger.info("🔍 Tool A: Vector search in participants_content_test...")
        folder_path = state.get('folder_path', '')
        
        # NORMALISATION: Convertir jean-pierre_aerts -> jean_pierre_aerts pour matcher les métadonnées
        normalized_folder_path = normalize_name_for_metadata(folder_path)
        logger.info(f"🔍 Tool A - original folder_path: '{folder_path}'")
        logger.info(f"🔍 Tool A - normalized folder_path: '{normalized_folder_path}'")
        logger.info(f"🔍 Tool A - user_msg: '{user_msg}'")
        
        if user_msg and normalized_folder_path:
            base_filters = {'folder_path': normalized_folder_path}
            # 🔄 SIMPLIFIÉ: Une seule recherche sans filtre de langue
            raw_personalized_all = perform_supabase_vector_search(
                query=user_msg,
                match_function='match_participants',
                metadata_filters=base_filters,
                limit=10  # Augmenté car une seule recherche
            )

            # Sanitize with a slightly more permissive threshold to handle multilingual variance
            personalized_content = sanitize_vector_results(
                results=raw_personalized_all,
                required_filters=None,  # Pas de filtrage strict par métadonnées
                top_k=3,
                min_similarity=0.30,
                max_chars_per_item=1800,
                max_total_chars=5000,
            )

            # Add tool identifier to metadata
            for item in personalized_content:
                item.setdefault("metadata", {})
                item["metadata"]["tool"] = "A"
                item["metadata"]["source"] = "participants_content_test"
        else:
            logger.info(f"⚠️ Tool A skipped - missing user_msg or folder_path: msg='{user_msg}', original='{folder_path}', normalized='{normalized_folder_path}'")
        
        # Tool B: Vector search in documents_content_test with sub_theme and mbti_type filters
        logger.info("🔍 Tool B: Vector search in documents_content_test...")
        sub_theme = state.get('sub_theme', '')
        user_mbti = state.get('user_mbti', '')
        logger.info(f"🔍 Tool B - sub_theme: '{sub_theme}'")
        logger.info(f"🔍 Tool B - user_mbti: '{user_mbti}'")
        logger.info(f"🔍 Tool B - user_msg: '{user_msg}'")
        
        if user_msg and sub_theme and user_mbti:
            base_filters = {
                'sub_theme': sub_theme,
                'mbti_type': user_mbti
            }
            # 🔄 SIMPLIFIÉ: Une seule recherche sans filtre de langue
            raw_generic_all = perform_supabase_vector_search(
                query=user_msg,
                match_function='match_documents',
                metadata_filters=base_filters,
                limit=10  # Augmenté car une seule recherche
            )

            # Sanitize with the standard threshold for documents
            generic_content = sanitize_vector_results(
                results=raw_generic_all,
                required_filters=None,  # Pas de filtrage strict par métadonnées
                top_k=3,
                min_similarity=0.30,
                max_chars_per_item=1800,
                max_total_chars=5000,
            )

            # Add tool identifier to metadata
            for item in generic_content:
                item.setdefault("metadata", {})
                item["metadata"]["tool"] = "B"
                item["metadata"]["source"] = "documents_content_test"
        else:
            logger.info(f"⚠️ Tool B skipped - missing required fields: msg='{user_msg}', sub_theme='{sub_theme}', mbti='{user_mbti}'")
        
        logger.info(f"✅ Tool A results: {len(personalized_content)} items")
        logger.info(f"✅ Tool B results: {len(generic_content)} items")
        
        # Debug: show detailed results for LangGraph Studio visibility
        if personalized_content:
            logger.info(f"\n📋 TOOL A RESULTS ({len(personalized_content)} items):")
            for i, item in enumerate(personalized_content):
                logger.info(f"  [{i+1}] Similarity: {item.get('similarity', 'N/A')}")
                logger.info(f"      Content: {item['content'][:200]}...")
                logger.info(f"      Metadata: {item.get('metadata', {})}")
                logger.info("")
        else:
            logger.info("❌ Tool A: No results found")
            
        if generic_content:
            logger.info(f"\n📋 TOOL B RESULTS ({len(generic_content)} items):")
            for i, item in enumerate(generic_content):
                logger.info(f"  [{i+1}] Similarity: {item.get('similarity', 'N/A')}")
                logger.info(f"      Content: {item['content'][:200]}...")
                logger.info(f"      Metadata: {item.get('metadata', {})}")
                logger.info("")
        else:
            logger.info("❌ Tool B: No results found")
    
    except Exception as e:
        logger.info(f"❌ Error in tools A+B: {e}")
        # Fallback to original static content if vector search fails
        if state.get("user_mbti"):
            personalized_content = [{
                "content": f"[Tool A Fallback] En tant que {state.get('user_mbti', '')}, vous avez tendance à... (caractéristiques générales MBTI pour: {user_msg})",
                "metadata": {"source": "participants_content_test_fallback", "mbti": state.get('user_mbti', ''), "tool": "A"},
                "similarity": 0.85
            }]
        
        if state.get("sub_theme"):
            generic_content = [{
                "content": f"[Tool B Fallback] Informations sur le thème {state.get('sub_theme', '')} concernant: {user_msg}",
                "metadata": {"source": "documents_content_test_fallback", "sub_theme": state.get('sub_theme', ''), "tool": "B"},
                "similarity": 0.80
            }]
    
    # 🏛️ NOUVEAU: Recherche des tempéraments si disponible (après Tools A + B)
    temperament_content = []  # Initialiser le contenu des tempéraments
    try:
        temperament_analysis = state.get("temperament_analysis")
        if temperament_analysis:
            logger.info("🏛️ Tool T: Recherche supplémentaire des tempéraments...")
            temperament_results = search_temperaments_documents(temperament_analysis, limit=15)  # 3 temperaments × 3 facettes × max 2 chunks each = 18 max
            if temperament_results:
                logger.info(f"   ✅ Trouvé {len(temperament_results)} résultats de tempéraments")
                
                # Déterminer si c'est pour l'utilisateur ou d'autres profils
                search_for_user = temperament_analysis.get('search_for_user', False)
                search_for_others = temperament_analysis.get('search_for_others', False)
                
                # Récupérer les tempéraments user et others pour assignation intelligente
                user_mbti = state.get('user_mbti', '')
                user_temperament = None
                if user_mbti and user_mbti in {'INTJ': 'NT', 'INTP': 'NT', 'ENTJ': 'NT', 'ENTP': 'NT',
                                              'INFJ': 'NF', 'INFP': 'NF', 'ENFJ': 'NF', 'ENFP': 'NF',
                                              'ISTJ': 'SJ', 'ISFJ': 'SJ', 'ESTJ': 'SJ', 'ESFJ': 'SJ',
                                              'ISTP': 'SP', 'ISFP': 'SP', 'ESTP': 'SP', 'ESFP': 'SP'}:
                    MBTI_TO_TEMPERAMENT = {'INTJ': 'NT', 'INTP': 'NT', 'ENTJ': 'NT', 'ENTP': 'NT',
                                          'INFJ': 'NF', 'INFP': 'NF', 'ENFJ': 'NF', 'ENFP': 'NF',
                                          'ISTJ': 'SJ', 'ISFJ': 'SJ', 'ESTJ': 'SJ', 'ESFJ': 'SJ',
                                          'ISTP': 'SP', 'ISFP': 'SP', 'ESTP': 'SP', 'ESFP': 'SP'}
                    user_temperament = MBTI_TO_TEMPERAMENT.get(user_mbti)
                
                # Ajouter les métadonnées appropriées avec assignation intelligente
                for result in temperament_results:
                    result['source_category'] = 'temperament'
                    result['tool'] = 'T'  # Tool Temperament
                    temperament_name = result.get('temperament', 'Unknown')
                    facet_name = result.get('facet', 'Unknown') 
                    result['temperament_info'] = f"{temperament_name}/{facet_name}"
                    
                    # Assignation intelligente basée sur le tempérament
                    if search_for_user and not search_for_others:
                        result['target'] = 'user'
                        logger.info(f"    → Target: 'user' (user only)")
                    elif search_for_others and not search_for_user:
                        result['target'] = 'others'
                        logger.info(f"    → Target: 'others' (others only)")
                    elif search_for_user and search_for_others:
                        # Si les deux, assigner selon le tempérament du résultat
                        # Mapping des noms de tempéraments vers les codes
                        temperament_to_code = {'Commando': 'SP', 'Guardian': 'SJ', 'Catalyst': 'NF', 'Architect': 'NT'}
                        temperament_code = temperament_to_code.get(temperament_name, temperament_name)
                        
                        logger.info(f"    🔄 Target assignation: '{temperament_name}' (code: {temperament_code}) vs user: {user_temperament}")
                        
                        if temperament_code == user_temperament:
                            result['target'] = 'user'  # Ce tempérament correspond à l'utilisateur
                            logger.info(f"    → Target: 'user' (matches user)")
                        else:
                            result['target'] = 'others'  # Ce tempérament correspond aux autres types
                            logger.info(f"    → Target: 'others' (different from user)")
                    else:
                        result['target'] = 'mixed'
                        logger.info(f"    → Target: 'mixed' (fallback)")
                        
                    logger.info(f"    ✅ Added: {temperament_name}/{facet_name} → target='{result['target']}')")
                    
                    temperament_content.append(result)
                
                logger.info(f"   📈 Total temperament_content: {len(temperament_content)}")
                
                # Debug des résultats de tempéraments
                logger.info(f"\n🏛️ TOOL T RESULTS ({len(temperament_content)} items):")
                for i, item in enumerate(temperament_content):
                    logger.info(f"  [T{i+1}] Temperament: {item.get('temperament_info', 'N/A')}")
                    logger.info(f"       Target: {item.get('target', 'N/A')}")
                    logger.info(f"       Similarity: {item.get('similarity', 'N/A')}")
                    logger.info(f"       Content: {item['content'][:150]}...")
                    logger.info("")
            else:
                logger.info("   ⚠️ Aucun résultat de tempérament trouvé")
        else:
            logger.info("   ℹ️ Pas d'analyse de tempérament disponible (mode debug ou analyse échouée)")
    except Exception as e:
        logger.info(f"⚠️ Erreur recherche tempéraments: {e}")
    
    return {
        **state, 
        "personalized_content": personalized_content, 
        "generic_content": generic_content,
        "temperament_content": temperament_content  # 🏛️ NOUVEAU: Ajouter au state
    }

# NODE 5B: Exécuter Tools A + B + C (User + Others)
def execute_tools_abc(state: WorkflowState) -> WorkflowState:
    """Execute 3 vector searches: Tool A + B + C, then combine all results for synthesis/comparison"""
    logger.info("🔧 NODE 5B: Executing 3 vector searches - Tools A + B + C...")
    
    # Exécuter A + B d'abord (2 recherches vectorielles)
    state = execute_tools_ab(state)
    
    # Ajouter Tool C (3ème recherche vectorielle pour les autres profils MBTI)
    others_content = []
    try:
        logger.info("🔍 Tool C: Vector search for other MBTI profiles...")
        analysis = state.get("mbti_analysis", {})
        other_profiles = analysis.get("other_mbti_profiles")
        # Utiliser la query reformulée pour une meilleure recherche
        search_query = state.get('reformulated_query') or state.get('user_message', '') or (state.get('messages', [{}])[-1].get('content', '') if state.get('messages') else '')
        user_msg = search_query
        sub_theme = state.get('sub_theme', '')
        logger.info(f"🔍 Tool C - Using search query: '{search_query}'")
        
        if other_profiles and user_msg:
            profiles = [p.strip() for p in other_profiles.split(",")]
            # Filtrer les profils pour exclure celui de l'utilisateur
            filtered_profiles = [p for p in profiles if p and p != state.get('user_mbti')]
            
            if filtered_profiles:
                logger.info(f"🔍 Tool C - Searching individually for MBTI types: {filtered_profiles}")
                logger.info(f"🔍 Tool C - sub_theme: '{sub_theme}'")
                
                # Faire une recherche séparée pour chaque profil MBTI
                for profile in filtered_profiles:
                    logger.info(f"🔍 Tool C - Individual search for: {profile}")
                    
                    # Préparer les filtres pour ce profil spécifique
                    filters = {'mbti_type': profile}
                    if sub_theme:
                        filters['sub_theme'] = sub_theme
                    
                    # Recherche individuelle pour ce profil
                    raw_profile_results = perform_supabase_vector_search(
                        query=user_msg,
                        match_function='match_documents',
                        metadata_filters=filters,
                        limit=4  # 4 résultats max par profil pour éviter trop de contenu
                    )
                    
                    if raw_profile_results:
                        # Sanitize pour ce profil spécifique
                        profile_content = sanitize_vector_results(
                            results=raw_profile_results,
                            required_filters=None,
                            top_k=2,  # 2 meilleurs résultats par profil
                            min_similarity=0.30,
                            max_chars_per_item=1200,
                            max_total_chars=2400,
                        )
                        
                        # Ajouter les résultats avec metadata enrichie
                        for item in profile_content:
                            item.setdefault("metadata", {})
                            item["metadata"]["tool"] = "C"
                            item["metadata"]["source"] = "documents_content_test"
                            item["metadata"]["target_mbti"] = profile
                            others_content.append(item)
                        
                        logger.info(f"  ✅ Found {len(profile_content)} results for {profile}")
                    else:
                        logger.info(f"  ❌ No results found for {profile}")
            else:
                logger.info("🔍 Tool C - No valid profiles to search (all filtered out)")
        
        logger.info(f"✅ Tool C results: {len(others_content)} items")
        logger.info(f"📊 Total results for synthesis: A={len(state.get('personalized_content', []))} + B={len(state.get('generic_content', []))} + C={len(others_content)}")
        
        if others_content:
            logger.info(f"\n📋 TOOL C RESULTS ({len(others_content)} items):")
            for i, item in enumerate(others_content):
                logger.info(f"  [{i+1}] MBTI: {item.get('metadata', {}).get('target_mbti', 'N/A')}")
                logger.info(f"      Similarity: {item.get('similarity', 'N/A')}")
                logger.info(f"      Content: {item['content'][:200]}...")
                logger.info("")
    
    except Exception as e:
        logger.info(f"❌ Error in tool C: {e}")
    
    # Récupérer les tempéraments déjà générés par Tool T dans execute_tools_ab
    temperament_content = state.get("temperament_content", [])
    logger.info(f"   ✅ Using temperament content from AB: {len(temperament_content)} items")
    
    return {**state, "others_content": others_content, "temperament_content": temperament_content}

# NODE 5C: Exécuter Tool C uniquement (Others only)
def execute_tools_c(state: WorkflowState) -> WorkflowState:
    """Execute 1 vector search: Tool C only (others collection) for other people's MBTI profiles"""
    logger.info("🔧 NODE 5C: Executing 1 vector search - Tool C only...")
    
    others_content = []
    try:
        logger.info("🔍 Tool C: Vector search for other MBTI profiles (no user profile)...")
        analysis = state.get("mbti_analysis", {})
        other_profiles = analysis.get("other_mbti_profiles")
        # Utiliser la query reformulée pour une meilleure recherche
        search_query = state.get('reformulated_query') or state.get('user_message', '') or (state.get('messages', [{}])[-1].get('content', '') if state.get('messages') else '')
        user_msg = search_query
        sub_theme = state.get('sub_theme', '')
        logger.info(f"🔍 Tool C - Using search query: '{search_query}'")
        
        if other_profiles and user_msg:
            profiles = [p.strip() for p in other_profiles.split(",")]
            # Pas besoin de filtrer par user_mbti car c'est Tool C seul
            filtered_profiles = [p for p in profiles if p]
            
            if filtered_profiles:
                logger.info(f"🔍 Tool C - Searching individually for MBTI types: {filtered_profiles}")
                logger.info(f"🔍 Tool C - sub_theme: '{sub_theme}'")
                
                # Faire une recherche séparée pour chaque profil MBTI
                for profile in filtered_profiles:
                    logger.info(f"🔍 Tool C - Individual search for: {profile}")
                    
                    # Préparer les filtres pour ce profil spécifique
                    filters = {'mbti_type': profile}
                    if sub_theme:
                        filters['sub_theme'] = sub_theme
                    
                    # Recherche individuelle pour ce profil
                    raw_profile_results = perform_supabase_vector_search(
                        query=user_msg,
                        match_function='match_documents',
                        metadata_filters=filters,
                        limit=6  # Plus de résultats par profil car c'est le seul tool
                    )
                    
                    if raw_profile_results:
                        # Sanitize pour ce profil spécifique - plus généreux car Tool C seul
                        profile_content = sanitize_vector_results(
                            results=raw_profile_results,
                            required_filters=None,
                            top_k=3,  # 3 meilleurs résultats par profil
                            min_similarity=0.30,
                            max_chars_per_item=1500,
                            max_total_chars=4000,
                        )
                        
                        # Ajouter avec metadata enrichie
                        for item in profile_content:
                            item.setdefault("metadata", {})
                            item["metadata"]["tool"] = "C"
                            item["metadata"]["source"] = "documents_content_test"
                            item["metadata"]["target_mbti"] = profile
                            others_content.append(item)
                        
                        logger.info(f"  ✅ Found {len(profile_content)} results for {profile}")
                    else:
                        logger.info(f"  ❌ No results found for {profile}")
            else:
                logger.info("🔍 Tool C - No valid profiles to search")
        
        logger.info(f"✅ Tool C results: {len(others_content)} items")
        
        if others_content:
            logger.info(f"\n📋 TOOL C RESULTS ({len(others_content)} items):")
            for i, item in enumerate(others_content):
                logger.info(f"  [{i+1}] MBTI: {item.get('metadata', {}).get('target_mbti', 'N/A')}")
                logger.info(f"      Similarity: {item.get('similarity', 'N/A')}")
                logger.info(f"      Content: {item['content'][:200]}...")
                logger.info("")
    
    except Exception as e:
        logger.info(f"❌ Error in tool C: {e}")
    
    # 🏛️ NOUVEAU: Recherche des tempéraments pour les AUTRES profils uniquement
    temperament_content = []  # Initialiser vide car pas de recherche user
    
    try:
        temperament_analysis = state.get("temperament_analysis")
        if temperament_analysis and temperament_analysis.get('search_for_others'):
            logger.info("🏛️ Tool T (Others only): Recherche des tempéraments pour les autres profils...")
            
            # Cloner l'analyse et forcer la recherche pour others uniquement
            others_temperament_analysis = temperament_analysis.copy()
            others_temperament_analysis['search_for_user'] = False
            others_temperament_analysis['search_for_others'] = True
            
            temperament_results = search_temperaments_documents(others_temperament_analysis, limit=5)
            if temperament_results:
                logger.info(f"   ✅ Trouvé {len(temperament_results)} résultats de tempéraments pour others")
                
                # Ajouter les métadonnées pour others (Tool C context - others only)
                for result in temperament_results:
                    result['source_category'] = 'temperament'
                    result['tool'] = 'T'
                    temperament_name = result.get('temperament', 'Unknown')
                    facet_name = result.get('facet', 'Unknown')
                    result['temperament_info'] = f"{temperament_name}/{facet_name}"
                    result['target'] = 'others'  # Toujours others pour Tool C
                    
                    temperament_content.append(result)
                
                logger.info(f"   📈 Total temperament_content (others only): {len(temperament_content)}")
                
                # Debug des résultats
                logger.info(f"\n🏛️ TOOL T RESULTS (Others only - {len(temperament_content)} items):")
                for i, item in enumerate(temperament_content):
                    logger.info(f"  [T{i+1}] Temperament: {item.get('temperament_info', 'N/A')}")
                    logger.info(f"       Target: {item.get('target', 'N/A')}")
                    logger.info(f"       Content: {item['content'][:150]}...")
                    logger.info("")
            else:
                logger.info("   ⚠️ Aucun résultat de tempérament trouvé pour others")
        else:
            logger.info("   ℹ️ Pas de recherche tempérament nécessaire (pas d'autres profils)")
    except Exception as e:
        logger.info(f"⚠️ Erreur recherche tempéraments others: {e}")
    
    return {**state, "others_content": others_content, "temperament_content": temperament_content}

# NODE 5D: Exécuter Tool D (General)
def execute_tools_d(state: WorkflowState) -> WorkflowState:
    """Execute 1 vector search: Tool D only (general MBTI collection) for general MBTI knowledge"""
    logger.info("🔧 NODE 5D: Executing 1 vector search - Tool D...")
    
    general_content = []
    try:
        logger.info("🔍 Tool D: Vector search for general MBTI knowledge...")
        # Utiliser la query reformulée pour une meilleure recherche
        search_query = state.get('reformulated_query') or state.get('user_message', '') or (state.get('messages', [{}])[-1].get('content', '') if state.get('messages') else '')
        user_msg = search_query
        sub_theme = state.get('sub_theme', '')
        logger.info(f"🔍 Tool D - Using search query: '{search_query}'")
        
        if user_msg:
            # Tool D cherche dans documents_content_test SANS filtre mbti_type
            # pour trouver des informations générales sur le MBTI
            logger.info(f"🔍 Tool D - Searching general MBTI info")
            logger.info(f"🔍 Tool D - sub_theme: '{sub_theme}'")
            
            # Recherche avec seulement sub_theme si disponible
            filters = {}
            if sub_theme:
                filters['sub_theme'] = sub_theme
            
            # 🔄 SIMPLIFIÉ: Une seule recherche sans filtre de langue
            raw_general_all = perform_supabase_vector_search(
                query=user_msg,
                match_function='match_documents',
                metadata_filters=filters if filters else None,
                limit=10  # Augmenté car une seule recherche
            )
            
            # Sanitize avec paramètres généreux car c'est le seul tool
            general_content = sanitize_vector_results(
                results=raw_general_all,
                required_filters=None,  # Pas de filtrage strict par métadonnées
                top_k=4,
                min_similarity=0.25,  # Seuil plus bas pour contenu général
                max_chars_per_item=1800,
                max_total_chars=6000,
            )
            
            # Ajouter metadata
            for item in general_content:
                item.setdefault("metadata", {})
                item["metadata"]["tool"] = "D"
                item["metadata"]["source"] = "documents_content_test"
                item["metadata"]["search_type"] = "general_mbti"
        
        logger.info(f"✅ Tool D results: {len(general_content)} items")
        
        if general_content:
            logger.info(f"\n📋 TOOL D RESULTS ({len(general_content)} items):")
            for i, item in enumerate(general_content):
                logger.info(f"  [{i+1}] Similarity: {item.get('similarity', 'N/A')}")
                logger.info(f"      Content: {item['content'][:200]}...")
                logger.info(f"      Metadata: {item.get('metadata', {})}")
                logger.info("")
    
    except Exception as e:
        logger.info(f"❌ Error in tool D: {e}")
        # Fallback content si erreur
        general_content = [{
            "content": f"Les types MBTI sont basés sur 4 dimensions: Extraversion/Introversion, Sensation/Intuition, Pensée/Sentiment, Jugement/Perception. Chaque combinaison forme un type unique avec ses forces et défis.",
            "metadata": {"source": "fallback", "tool": "D"},
            "similarity": 0.50
        }]
    
    return {**state, "general_content": general_content, "temperament_content": []}

# NODE 5E: Pas d'outils - Réponse directe
def no_tools(state: WorkflowState) -> WorkflowState:
    """Pas d'outils nécessaires - réponse directe basée sur l'analyse"""
    logger.info("🔧 NODE 5E: No tools needed - direct response...")
    
    # Marquer explicitement qu'aucune recherche n'est nécessaire
    # Cela permet au generate_final_response de savoir qu'il doit répondre différemment
    return {**state, 
            "personalized_content": [],
            "generic_content": [],
            "others_content": [],
            "general_content": [],
            "temperament_content": [],  # 🏛️ Initialiser vide
            "no_search_needed": True}

# ============================================================================
# FLUX LENCIONI pour D6_CollectiveSuccess
# ============================================================================

def lencioni_intent_analysis(state: WorkflowState) -> WorkflowState:
    """
    Premier node Lencioni: Analyse l'intent de la question
    Routes vers: 1) report_lookup 2) lencioni_general_knowledge 3) insight_blend 4) meta_out_of_scope
    """
    logger.info("🎯 NODE: Lencioni Intent Analysis...")
    
    try:
        # Récupérer la question utilisateur
        user_msg = state.get('user_message', '') or (state.get('messages', [{}])[-1].get('content', '') if state.get('messages') else '')
        
        # FILTRE PRÉCOCE: Vérifier si la question concerne des équipes de travail
        non_team_keywords = ['family', 'famille', 'marriage', 'mariage', 'spouse', 'époux', 'épouse', 
                           'children', 'enfants', 'personal relationship', 'relation personnelle',
                           'romantic', 'romantique', 'dating', 'couple', 'parenting', 'parentalité']
        
        user_msg_lower = user_msg.lower()
        if any(keyword in user_msg_lower for keyword in non_team_keywords):
            logger.warning(f"⚠️ Non-team topic detected in question: {user_msg[:100]}...")
            return {
                **state, 
                "lencioni_intent_analysis": {
                    "intent_type": "OUT_OF_SCOPE",
                    "dysfunctions_mentioned": [],
                    "needs_clarification": False,
                    "clarification_question": "",
                    "confidence": 1.0,
                    "reasoning": "Question is about personal/family relationships, not workplace teams"
                }
            }
        
        # Construire le contexte historique
        messages_history = []
        raw_messages = state.get('messages', [])
        for msg in raw_messages[-5:]:  # Derniers 5 messages pour contexte
            if hasattr(msg, 'type') and hasattr(msg, 'content'):
                messages_history.append({
                    "role": msg.type if msg.type in ['human', 'ai'] else 'human',
                    "content": msg.content
                })
            elif isinstance(msg, dict):
                messages_history.append({
                    "role": msg.get('role', 'human'),
                    "content": msg.get('content', '')
                })
        
        # Construire le contexte pour l'analyse avec analyse contextuelle
        history_text = ""
        previously_discussed_dysfunctions = []
        previous_intent_types = []
        
        if messages_history:
            for msg in messages_history[-5:]:  # Analyser plus de messages
                content = msg.get('content', '').lower()
                history_text += f" {content}"
                
                # Analyser les dysfonctions mentionnées dans l'historique
                if 'trust' in content or 'confiance' in content:
                    previously_discussed_dysfunctions.append('Trust')
                if 'conflict' in content or 'conflit' in content:
                    previously_discussed_dysfunctions.append('Conflict')
                if 'commitment' in content or 'engagement' in content:
                    previously_discussed_dysfunctions.append('Commitment')
                if 'accountability' in content or 'responsabilité' in content:
                    previously_discussed_dysfunctions.append('Accountability')
                if 'results' in content or 'résultats' in content:
                    previously_discussed_dysfunctions.append('Results')
                
                # Détecter les types d'interaction précédents
                if any(word in content for word in ['score', 'assessment', 'results', 'data']):
                    previous_intent_types.append('REPORT_LOOKUP')
                elif any(word in content for word in ['explain', 'what is', 'tell me about']):
                    previous_intent_types.append('GENERAL_KNOWLEDGE')
                elif any(word in content for word in ['improve', 'help', 'strategies', 'advice']):
                    previous_intent_types.append('INSIGHT_BLEND')
        
        # Supprimer les doublons
        previously_discussed_dysfunctions = list(set(previously_discussed_dysfunctions))
        previous_intent_types = list(set(previous_intent_types))
        
        # NOUVEAU PROMPT SIMPLIFIÉ
        intent_prompt = f"""You are analyzing a user's question about their team's Lencioni Five Dysfunctions assessment.

CONVERSATION HISTORY:
{history_text.strip() if history_text.strip() else "No previous conversation"}

USER'S CURRENT QUESTION: "{user_msg}"

CONTEXT: 
- Previously discussed: {previously_discussed_dysfunctions if previously_discussed_dysfunctions else "None"}
- Previous interactions: {previous_intent_types if previous_intent_types else "None"}

TASK: Determine the user's intent and which dysfunctions they want information about.

INTENT TYPES:
1. **REPORT_LOOKUP** - User wants to see their team's specific assessment data/scores
2. **GENERAL_KNOWLEDGE** - User wants to learn about the Lencioni model theory
3. **INSIGHT_BLEND** - User wants coaching/advice for improving their team

DYSFUNCTION DETECTION (BE STRICT):
- The 5 dysfunctions are: Trust, Conflict, Commitment, Accountability, Results
- EXACT MATCHES ONLY: Only return dysfunctions if they are clearly mentioned or implied
- If user asks about "all", "each", "every", "rest", "others", "complete picture" etc. → Return ALL 5: ["Trust", "Conflict", "Commitment", "Accountability", "Results"]
- If user mentions specific dysfunction names or clear synonyms → Return only those mentioned
- If ambiguous or unclear terms are used → Return empty list [] and set needs_clarification: true
- SYNONYMS ACCEPTED (must be specific):
  • Trust: confidence, trust-building, vulnerability, openness
  • Conflict: disagreement, debate, confrontation, healthy conflict
  • Commitment: engagement, buy-in, decision clarity, alignment
  • Accountability: responsibility, peer accountability, calling out
  • Results: performance metrics, specific outcomes, measurable results
- GENERAL TERMS that need clarification: "success", "team effectiveness", "team performance", "collective success" → Ask for specifics

EXAMPLES:
- "What about the rest?" (after discussing Trust) → ALL dysfunctions
- "Performance for each dysfunction" → ALL dysfunctions  
- "How did we score on trust?" → ["Trust"]
- "Show me our conflict results" → ["Conflict"]
- "How can I foster collective success?" → [] + needs_clarification: true (too general)
- "How to improve team performance?" → [] + needs_clarification: true (too general)
- "Help with accountability issues" → ["Accountability"]

Return ONLY this JSON format:
{{
  "intent_type": "REPORT_LOOKUP|GENERAL_KNOWLEDGE|INSIGHT_BLEND",
  "dysfunctions_mentioned": ["Trust", "Conflict", "Commitment", "Accountability", "Results"] or specific ones or [],
  "needs_clarification": true/false,
  "clarification_question": "Question to ask user if needs_clarification is true",
  "confidence": 0.0-1.0,
  "reasoning": "Brief explanation"
}}"""
        
        # Utiliser l'appel isolé pour éviter le streaming
        logger.info(f"🔒 Calling isolated analysis for Lencioni intent...")
        raw_response = isolated_analysis_call(intent_prompt)
        
        # Parser la réponse JSON
        try:
            import re
            json_match = re.search(r'\{[\s\S]*\}', raw_response)
            if json_match:
                analysis = json.loads(json_match.group(0))
                logger.info(f"✅ Lencioni Intent Analysis: {analysis}")
                
                # Vérifier si une clarification est nécessaire
                if analysis.get("needs_clarification", False):
                    clarification_question = analysis.get("clarification_question", "Could you please specify which dysfunction you'd like to focus on? The 5 dysfunctions are: Trust, Conflict, Commitment, Accountability, or Results.")
                    logger.info(f"❓ Clarification needed: {clarification_question}")
                    
                    # Retourner un state avec une demande de clarification
                    return {
                        **state, 
                        "lencioni_intent_analysis": analysis,
                        "needs_user_clarification": True,
                        "clarification_question": clarification_question
                    }
                
                # Ajouter l'analyse au state
                return {**state, "lencioni_intent_analysis": analysis}
            else:
                logger.warning(f"⚠️ No JSON found in Lencioni intent response")
                # Fallback
                fallback_analysis = {
                    "intent_type": "INSIGHT_BLEND",
                    "instructions": "INSIGHT_BLEND: Combine user data with coaching advice",
                    "dysfunctions_mentioned": [],
                    "needs_clarification": False,
                    "clarification_question": "",
                    "confidence": 0.5,
                    "reasoning": "Fallback - could not parse intent"
                }
                return {**state, "lencioni_intent_analysis": fallback_analysis}
                
        except Exception as parse_error:
            logger.error(f"❌ Error parsing Lencioni intent JSON: {parse_error}")
            # Fallback basé sur des mots-clés
            user_msg_lower = user_msg.lower()
            
            if any(word in user_msg_lower for word in ['my score', 'my results', 'what did i get', 'show me my']):
                intent_type = "REPORT_LOOKUP"
            elif any(word in user_msg_lower for word in ['what is', 'explain', 'tell me about', 'how does']):
                intent_type = "LENCIONI_GENERAL_KNOWLEDGE"
            elif any(word in user_msg_lower for word in ['help me', 'how can i', 'improve', 'strategies']):
                intent_type = "INSIGHT_BLEND"
            else:
                intent_type = "INSIGHT_BLEND"  # Default
            
            # Détecter les dysfonctionnements mentionnés
            dysfunctions = []
            if 'trust' in user_msg_lower:
                dysfunctions.append("Trust")
            if 'conflict' in user_msg_lower:
                dysfunctions.append("Conflict")
            if 'commitment' in user_msg_lower or 'commit' in user_msg_lower:
                dysfunctions.append("Commitment")
            if 'accountab' in user_msg_lower:
                dysfunctions.append("Accountability")
            if 'result' in user_msg_lower or 'outcome' in user_msg_lower:
                dysfunctions.append("Results")
            
            fallback_analysis = {
                "intent_type": intent_type,
                "instructions": f"{intent_type}: Fallback classification",
                "dysfunctions_mentioned": dysfunctions,
                "needs_clarification": False,
                "clarification_question": "",
                "confidence": 0.6,
                "reasoning": "Keyword-based fallback classification"
            }
            logger.info(f"🔄 Lencioni Intent Fallback: {fallback_analysis}")
            return {**state, "lencioni_intent_analysis": fallback_analysis}
            
    except Exception as e:
        logger.error(f"❌ Error in Lencioni intent analysis: {e}")
        # Fallback complet
        fallback_analysis = {
            "intent_type": "INSIGHT_BLEND",
            "instructions": "INSIGHT_BLEND: Error fallback",
            "dysfunctions_mentioned": [],
            "needs_clarification": False,
            "clarification_question": "",
            "confidence": 0.3,
            "reasoning": "Error occurred during analysis"
        }
        return {**state, "lencioni_intent_analysis": fallback_analysis}

def lencioni_analysis(state: WorkflowState) -> WorkflowState:
    """
    Analyse spécialisée pour D6_CollectiveSuccess (Lencioni)
    Récupère les données Lencioni du profil utilisateur avec filtrage optionnel par dysfonction
    """
    logger.info("🏛️ NODE: Lencioni Analysis (D6_CollectiveSuccess)")
    
    try:
        user_id = state.get('user_id')
        if not user_id:
            logger.warning("⚠️ No user_id for Lencioni analysis")
            return {**state, "lencioni_data": None, "lencioni_details": None}
        
        # Vérifier si des dysfonctions spécifiques sont mentionnées
        intent_analysis = state.get("lencioni_intent_analysis", {})
        dysfunctions_mentioned = intent_analysis.get("dysfunctions_mentioned", [])
        intent_type = intent_analysis.get("intent_type", "")
        
        logger.info(f"🔍 Intent: {intent_type}, Dysfunctions mentioned: {dysfunctions_mentioned}")
        
        # Récupérer les scores principaux (toujours nécessaires)
        if dysfunctions_mentioned and intent_type in ["REPORT_LOOKUP", "INSIGHT_BLEND"]:
            # Cas spécifique: filtrer par dysfonctions mentionnées
            logger.info(f"📊 Fetching specific dysfunction data for: {dysfunctions_mentioned}")
            
            # Récupérer les scores filtrés
            query = supabase.table("lensioni_team_assessment_score").select(
                "dysfunction, score, level, summary"
            ).eq("profile_id", user_id)
            
            # Ajouter les filtres de dysfonctions
            for dysfunction in dysfunctions_mentioned:
                # Note: On va faire plusieurs requêtes ou utiliser .in_() selon la DB
                pass  # On va implémenter en bas
            
            # Pour l'instant, récupérer tout et filtrer ensuite
            lencioni_response = query.execute()
            
            # Garder TOUS les scores (pas de filtrage)
            filtered_data = lencioni_response.data if lencioni_response.data else []
            logger.info(f"✅ Found {len(filtered_data)} total scores (no filtering)")
            
            # Récupérer aussi les détails spécifiques depuis lensioni_details
            # Il faut d'abord récupérer les IDs des scores, puis chercher dans lensioni_details
            details_data = []
            for dysfunction in dysfunctions_mentioned:
                logger.info(f"🔍 Searching details for dysfunction: {dysfunction}")
                
                # 1. Récupérer l'ID du score pour cette dysfonction
                score_response = supabase.table("lensioni_team_assessment_score").select(
                    "id, dysfunction"
                ).eq("profile_id", user_id).eq("dysfunction", dysfunction).execute()
                
                logger.info(f"📊 Score query result for {dysfunction}: {len(score_response.data) if score_response.data else 0} records")
                
                if score_response.data:
                    for score_record in score_response.data:
                        team_assessment_score_id = score_record["id"]
                        logger.info(f"🎯 Using score_id {team_assessment_score_id} for dysfunction {dysfunction}")
                        
                        # 2. Chercher dans lensioni_details avec le bon lien
                        details_response = supabase.table("lensioni_details").select(
                            "dysfunction, question, score, level"
                        ).eq("team_assessment_score_id", team_assessment_score_id).execute()
                        
                        logger.info(f"📋 Details query result for score_id {team_assessment_score_id}: {len(details_response.data) if details_response.data else 0} records")
                        
                        if details_response.data:
                            details_data.extend(details_response.data)
                            logger.info(f"✅ Found {len(details_response.data)} detail questions for {dysfunction} (score_id: {team_assessment_score_id})")
                        else:
                            logger.warning(f"⚠️ No details found in lensioni_details for score_id: {team_assessment_score_id}")
                else:
                    logger.warning(f"⚠️ No score found for dysfunction {dysfunction} and user {user_id}")
            
            return {
                **state, 
                "lencioni_data": filtered_data,
                "lencioni_details": details_data,
                "dysfunction_focus": dysfunctions_mentioned
            }
            
        else:
            # Cas général: récupérer tous les scores
            lencioni_response = supabase.table("lensioni_team_assessment_score").select(
                "dysfunction, score, level, summary"
            ).eq("profile_id", user_id).execute()
            
            if lencioni_response.data:
                logger.info(f"✅ Found {len(lencioni_response.data)} Lencioni scores (general)")
                return {**state, "lencioni_data": lencioni_response.data, "lencioni_details": None}
            else:
                logger.info("ℹ️ No Lencioni data found for this user")
                return {**state, "lencioni_data": None, "lencioni_details": None}
            
    except Exception as e:
        logger.error(f"❌ Error in Lencioni analysis: {e}")
        return {**state, "lencioni_data": None}

def lencioni_vector_search(state: WorkflowState) -> WorkflowState:
    """
    Effectue des recherches vectorielles spécifiques selon l'intent Lencioni
    """
    logger.info("🔍 NODE: Lencioni Vector Search")
    
    try:
        intent_analysis = state.get("lencioni_intent_analysis", {})
        intent_type = intent_analysis.get("intent_type", "INSIGHT_BLEND")
        user_question = state.get('user_message', '')
        dysfunctions_mentioned = intent_analysis.get("dysfunctions_mentioned", [])
        
        logger.info(f"🔍 Searching for intent: {intent_type}")
        logger.info(f"🔍 Dysfunctions mentioned: {dysfunctions_mentioned}")
        
        search_results = {
            "report_lookup_content": [],
            "general_knowledge_content": [],
            "insight_blend_content": [],
            "tools_exercises_content": []
        }
        
        # Adapter la recherche selon l'intent
        if intent_type == "REPORT_LOOKUP":
            # Pour REPORT_LOOKUP: chercher des exemples d'interprétation de scores
            logger.info("📊 REPORT_LOOKUP: Fetching Lencioni model overview from database")
            
            # Si aucune dysfunction spécifique mentionnée, récupérer l'overview du modèle
            if not dysfunctions_mentioned:
                try:
                    # Recherche directe dans documents_content_test pour l'overview
                    response = supabase.table('documents_content_test').select('content,metadata').eq(
                        'metadata->>source_type', 'lencioni_document'
                    ).eq(
                        'metadata->>lencioni_dysfunction', 'general'
                    ).eq(
                        'metadata->>lencioni_content_type', 'overview'
                    ).limit(10).execute()  # Augmenté à 10 pour avoir plus de contexte
                    
                    if response.data:
                        logger.info(f"✅ Found {len(response.data)} overview chunks from Lencioni document")
                        for item in response.data:
                            search_results["report_lookup_content"].append({
                                "content": item.get('content', ''),
                                "metadata": item.get('metadata', {}),
                                "type": "lencioni_overview"
                            })
                    else:
                        logger.warning("⚠️ No Lencioni overview content found in database")
                    
                except Exception as e:
                    logger.error(f"❌ Error fetching Lencioni overview: {e}")
            
            # Recherche 2: Si des dysfonctionnements spécifiques sont mentionnés
            # TEMPORAIREMENT DÉSACTIVÉ
            # if dysfunctions_mentioned:
            #     for dysfunction in dysfunctions_mentioned:
            #         query2 = f"Lencioni {dysfunction} assessment score meaning team impact"
            #         results2 = search_documents(query2, 2, sub_theme="D6_CollectiveSuccess")
            #         search_results["report_lookup_content"].extend(results2)
            
            logger.info(f"✅ Found {len(search_results['report_lookup_content'])} report lookup results")
            
        elif intent_type == "LENCIONI_GENERAL_KNOWLEDGE":
            # Pour GENERAL_KNOWLEDGE: récupérer uniquement l'overview général
            logger.info("📚 Fetching Lencioni overview for general knowledge...")
            
            try:
                # Récupérer uniquement l'overview du modèle Lencioni
                response = supabase.table('documents_content_test').select('content,metadata').eq(
                    'metadata->>source_type', 'lencioni_document'
                ).eq(
                    'metadata->>lencioni_dysfunction', 'general'
                ).eq(
                    'metadata->>lencioni_content_type', 'overview'
                ).limit(10).execute()
                
                if response.data:
                    logger.info(f"✅ Found {len(response.data)} overview chunks for general knowledge")
                    for item in response.data:
                        search_results["general_knowledge_content"].append({
                            "content": item.get('content', ''),
                            "metadata": item.get('metadata', {}),
                            "type": "lencioni_overview"
                        })
                else:
                    logger.warning("⚠️ No Lencioni overview content found for general knowledge")
                    
            except Exception as e:
                logger.error(f"❌ Error fetching Lencioni overview for general knowledge: {e}")
            
            logger.info(f"✅ Found {len(search_results['general_knowledge_content'])} general knowledge results")
            
        elif intent_type == "INSIGHT_BLEND":
            # Pour INSIGHT_BLEND: combiner données utilisateur et conseils
            logger.info("🎯 Searching for actionable insights...")
            
            # Si aucune dysfunction spécifique mentionnée, récupérer l'overview général
            if not dysfunctions_mentioned:
                logger.info("📚 No specific dysfunction mentioned - fetching Lencioni model overview")
                try:
                    # Recherche directe dans documents_content_test pour l'overview
                    response = supabase.table('documents_content_test').select('content,metadata').eq(
                        'metadata->>source_type', 'lencioni_document'
                    ).eq(
                        'metadata->>lencioni_dysfunction', 'general'
                    ).eq(
                        'metadata->>lencioni_content_type', 'overview'
                    ).limit(10).execute()
                    
                    if response.data:
                        logger.info(f"✅ Found {len(response.data)} overview chunks from Lencioni document")
                        for item in response.data:
                            search_results["insight_blend_content"].append({
                                "content": item.get('content', ''),
                                "metadata": item.get('metadata', {}),
                                "type": "lencioni_overview"
                            })
                    else:
                        logger.warning("⚠️ No Lencioni overview content found in database")
                    
                except Exception as e:
                    logger.error(f"❌ Error fetching Lencioni overview: {e}")
            
            # ÉTAPE 1: Rechercher les recommandations Lencioni dans documents_content_test
            if dysfunctions_mentioned:
                logger.info(f"📋 Searching Lencioni recommendations for dysfunctions: {dysfunctions_mentioned}")
                
                for dysfunction in dysfunctions_mentioned:
                    try:
                        # Recherche dans documents_content_test pour les recommandations Lencioni
                        response = supabase.table("documents_content_test").select(
                            "content, metadata"
                        ).eq(
                            'metadata->>lencioni_content_type', 'recommendation'
                        ).eq(
                            'metadata->>lencioni_dysfunction', dysfunction.lower()
                        ).eq(
                            'metadata->>source_type', 'lencioni_document'
                        ).execute()
                        
                        if response.data:
                            logger.info(f"✅ Found {len(response.data)} Lencioni recommendations for {dysfunction}")
                            for item in response.data:
                                search_results["insight_blend_content"].append({
                                    "content": item["content"],
                                    "metadata": item["metadata"],
                                    "type": "lencioni_recommendation",
                                    "dysfunction": dysfunction
                                })
                        else:
                            logger.warning(f"⚠️ No Lencioni recommendations found for {dysfunction}")
                            
                    except Exception as e:
                        logger.error(f"❌ Error fetching Lencioni recommendations for {dysfunction}: {e}")
            
            # ÉTAPE 2: Rechercher les recommandations spécifiques à l'équipe dans lensioni_team_culture_questions
            user_id = state.get('user_id')
            if user_id and dysfunctions_mentioned:
                logger.info(f"👥 Searching team-specific recommendations for user {user_id}")
                
                for dysfunction in dysfunctions_mentioned:
                    try:
                        if dysfunction.lower() == "trust":
                            # Pour Trust : récupérer trust_reco et trust_reco_count
                            logger.info(f"🤝 Fetching Trust team recommendations for user {user_id}")
                            
                            # D'abord récupérer l'ID du score assessment pour ce user et cette dysfunction
                            score_response = supabase.table("lensioni_team_assessment_score").select(
                                "id"
                            ).eq("profile_id", user_id).eq("dysfunction", "Trust").execute()
                            
                            if score_response.data:
                                team_assessment_score_id = score_response.data[0]["id"]
                                logger.info(f"✅ Found assessment score ID: {team_assessment_score_id}")
                                
                                # Récupérer les recommandations Trust de l'équipe
                                trust_response = supabase.table("lensioni_team_culture_questions").select(
                                    "trust_reco, trust_reco_count"
                                ).eq("team_assessment_score_id", team_assessment_score_id).neq(
                                    "trust_reco", None
                                ).order("trust_reco_count", desc=True).execute()
                                
                                if trust_response.data:
                                    logger.info(f"✅ Found {len(trust_response.data)} Trust team recommendations")
                                    for item in trust_response.data:
                                        search_results["insight_blend_content"].append({
                                            "content": f"Team recommendation: {item['trust_reco']} (voted by {item['trust_reco_count']} team members)",
                                            "type": "team_recommendation",
                                            "dysfunction": "Trust",
                                            "recommendation": item['trust_reco'],
                                            "vote_count": item['trust_reco_count'],
                                            "category": "trust_building"
                                        })
                                else:
                                    logger.warning(f"⚠️ No Trust team recommendations found for assessment {team_assessment_score_id}")
                            else:
                                logger.warning(f"⚠️ No Trust assessment found for user {user_id}")
                        
                        elif dysfunction.lower() == "conflict":
                            # Pour Conflict : récupérer conflict_resp et les counts (acceptable/tolerable/unacceptable + admitting)
                            logger.info(f"⚔️ Fetching Conflict team recommendations for user {user_id}")
                            
                            # D'abord récupérer l'ID du score assessment pour ce user et cette dysfunction
                            score_response = supabase.table("lensioni_team_assessment_score").select(
                                "id"
                            ).eq("profile_id", user_id).eq("dysfunction", "Conflict").execute()
                            
                            if score_response.data:
                                team_assessment_score_id = score_response.data[0]["id"]
                                logger.info(f"✅ Found assessment score ID: {team_assessment_score_id}")
                                
                                # Récupérer les données Conflict de l'équipe
                                conflict_response = supabase.table("lensioni_team_culture_questions").select(
                                    "conflict_resp, conflict_resp_unacceptable_count, conflict_resp_tolerable_count, conflict_resp_acceptable_count, conflict_resp_count"
                                ).eq("team_assessment_score_id", team_assessment_score_id).neq(
                                    "conflict_resp", None
                                ).execute()
                                
                                if conflict_response.data:
                                    logger.info(f"✅ Found {len(conflict_response.data)} Conflict team behaviors")
                                    for item in conflict_response.data:
                                        behavior = item['conflict_resp']
                                        unacceptable = item['conflict_resp_unacceptable_count']
                                        tolerable = item['conflict_resp_tolerable_count'] 
                                        acceptable = item['conflict_resp_acceptable_count']
                                        admitting = item['conflict_resp_count']  # Nombre qui admettent avoir ce comportement
                                        
                                        search_results["insight_blend_content"].append({
                                            "content": f"Team conflict behavior: \"{behavior}\" - {admitting} members admit to this behavior. Team rating: Acceptable: {acceptable}, Tolerable: {tolerable}, Unacceptable: {unacceptable}",
                                            "type": "team_recommendation",
                                            "dysfunction": "Conflict",
                                            "behavior": behavior,
                                            "acceptable_count": acceptable,
                                            "tolerable_count": tolerable,
                                            "unacceptable_count": unacceptable,
                                            "admitting_count": admitting,
                                            "category": "conflict_behavior"
                                        })
                                else:
                                    logger.warning(f"⚠️ No Conflict team behaviors found for assessment {team_assessment_score_id}")
                            else:
                                logger.warning(f"⚠️ No Conflict assessment found for user {user_id}")
                        
                        elif dysfunction.lower() == "commitment":
                            # Pour Commitment : récupérer commitment related data
                            logger.info(f"🤝 Fetching Commitment team recommendations for user {user_id}")
                            
                            # D'abord récupérer l'ID du score assessment pour ce user et cette dysfunction
                            score_response = supabase.table("lensioni_team_assessment_score").select(
                                "id"
                            ).eq("profile_id", user_id).eq("dysfunction", "Commitment").execute()
                            
                            if score_response.data:
                                team_assessment_score_id = score_response.data[0]["id"]
                                logger.info(f"🎯 Found Commitment assessment ID: {team_assessment_score_id}")
                                
                                # Récupérer les raisons du manque d'engagement de l'équipe
                                commitment_response = supabase.table("lensioni_team_culture_questions").select(
                                    "comm_reason, comm_reason_count, comm_reason_perc"
                                ).eq("team_assessment_score_id", team_assessment_score_id).neq(
                                    "comm_reason", None
                                ).order("comm_reason_count", desc=True).execute()
                                
                                if commitment_response.data:
                                    logger.info(f"✅ Found {len(commitment_response.data)} Commitment team insights")
                                    for item in commitment_response.data:
                                        search_results["insight_blend_content"].append({
                                            "content": f"Reason contributing to lack of commitment: \"{item['comm_reason']}\" - identified by {item['comm_reason_count']} team members ({item['comm_reason_perc']}%)",
                                            "type": "team_recommendation",
                                            "dysfunction": "Commitment",
                                            "reason": item['comm_reason'],
                                            "vote_count": item['comm_reason_count'],
                                            "vote_percentage": item['comm_reason_perc'],
                                            "category": "commitment_barriers"
                                        })
                                else:
                                    logger.warning(f"⚠️ No Commitment team data found for assessment {team_assessment_score_id}")
                            else:
                                logger.warning(f"⚠️ No Commitment assessment found for user {user_id}")
                        
                        elif dysfunction.lower() == "accountability":
                            # Pour Accountability : récupérer acc_reco et acc_reco_count
                            logger.info(f"📋 Fetching Accountability team recommendations for user {user_id}")
                            
                            # D'abord récupérer l'ID du score assessment pour ce user et cette dysfunction
                            score_response = supabase.table("lensioni_team_assessment_score").select(
                                "id"
                            ).eq("profile_id", user_id).eq("dysfunction", "Accountability").execute()
                            
                            if score_response.data:
                                team_assessment_score_id = score_response.data[0]["id"]
                                logger.info(f"🎯 Found Accountability assessment ID: {team_assessment_score_id}")
                                
                                # Récupérer les recommandations Accountability de l'équipe
                                accountability_response = supabase.table("lensioni_team_culture_questions").select(
                                    "acc_reco, acc_reco_count"
                                ).eq("team_assessment_score_id", team_assessment_score_id).neq(
                                    "acc_reco", None
                                ).order("acc_reco_count", desc=True).execute()
                                
                                if accountability_response.data:
                                    logger.info(f"✅ Found {len(accountability_response.data)} Accountability team recommendations")
                                    for item in accountability_response.data:
                                        search_results["insight_blend_content"].append({
                                            "content": f"Accountability area: \"{item['acc_reco']}\" - identified by {item['acc_reco_count']} team members",
                                            "type": "team_recommendation",
                                            "dysfunction": "Accountability",
                                            "recommendation": item['acc_reco'],
                                            "vote_count": item['acc_reco_count'],
                                            "category": "accountability_areas"
                                        })
                                else:
                                    logger.warning(f"⚠️ No Accountability team recommendations found for assessment {team_assessment_score_id}")
                            else:
                                logger.warning(f"⚠️ No Accountability assessment found for user {user_id}")
                        
                        elif dysfunction.lower() == "results":
                            # Pour Results : récupérer res_distraction, res_distraction_count et res_distraction_perc
                            logger.info(f"🎯 Fetching Results team distractions for user {user_id}")
                            
                            # D'abord récupérer l'ID du score assessment pour ce user et cette dysfunction
                            score_response = supabase.table("lensioni_team_assessment_score").select(
                                "id"
                            ).eq("profile_id", user_id).eq("dysfunction", "Results").execute()
                            
                            if score_response.data:
                                team_assessment_score_id = score_response.data[0]["id"]
                                logger.info(f"🎯 Found Results assessment ID: {team_assessment_score_id}")
                                
                                # Récupérer les distractions Results de l'équipe
                                results_response = supabase.table("lensioni_team_culture_questions").select(
                                    "res_distraction, res_distraction_count, res_distraction_perc"
                                ).eq("team_assessment_score_id", team_assessment_score_id).neq(
                                    "res_distraction", None
                                ).order("res_distraction_count", desc=True).execute()
                                
                                if results_response.data:
                                    logger.info(f"✅ Found {len(results_response.data)} Results team distractions")
                                    for item in results_response.data:
                                        search_results["insight_blend_content"].append({
                                            "content": f"Distraction keeping team from focusing on results: \"{item['res_distraction']}\" - identified by {item['res_distraction_count']} team members ({item['res_distraction_perc']}%)",
                                            "type": "team_recommendation",
                                            "dysfunction": "Results",
                                            "distraction": item['res_distraction'],
                                            "vote_count": item['res_distraction_count'],
                                            "vote_percentage": item['res_distraction_perc'],
                                            "category": "results_distractions"
                                        })
                                else:
                                    logger.warning(f"⚠️ No Results team distractions found for assessment {team_assessment_score_id}")
                            else:
                                logger.warning(f"⚠️ No Results assessment found for user {user_id}")
                        
                    except Exception as e:
                        logger.error(f"❌ Error fetching team recommendations for {dysfunction}: {e}")
            else:
                logger.info("ℹ️ No user_id or dysfunctions for team-specific recommendations")
            
            logger.info(f"✅ Found {len(search_results['insight_blend_content'])} insight blend results")
        
        # Toujours chercher des outils et exercices pertinents
        # TEMPORAIREMENT DÉSACTIVÉ
        # if dysfunctions_mentioned:
        #     logger.info("🛠️ Searching for relevant tools and exercises...")
        #     for dysfunction in dysfunctions_mentioned:
        #         query = f"Lencioni {dysfunction} tools exercises activities team building"
        #         results = search_documents(query, 2, sub_theme="D6_CollectiveSuccess", content_type="tools")
        #         search_results["tools_exercises_content"].extend(results)
            
            logger.info(f"✅ Found {len(search_results['tools_exercises_content'])} tools/exercises")
        
        # Ajouter les résultats au state (lencioni_data est automatiquement préservé par LangGraph)
        return {
            **state,
            "lencioni_search_results": search_results,
            "search_executed_for_intent": intent_type
        }
        
    except Exception as e:
        logger.error(f"❌ Error in Lencioni vector search: {e}")
        return {
            **state, 
            "lencioni_search_results": {}
        }

# NODE 6: Agent principal - Génération de la réponse finale
def generate_final_response(state: WorkflowState) -> WorkflowState:
    """Générateur unifié pour tous les sous-thèmes avec streaming garanti"""
    
    # Déterminer le sous-thème pour choisir le bon template
    sub_theme = state.get('sub_theme', '')
    theme = state.get('theme', '')
    
    logger.info(f"🤖 NODE 6: Generating response for theme={theme}, sub_theme={sub_theme}")
    
    try:
        # Récupérer la question utilisateur
        user_question = state.get('user_message', '') or (state.get('messages', [{}])[-1].get('content', '') if state.get('messages') else '')
        
        # ============= SYSTÈME DE TEMPLATES UNIFIÉ =============
        # Utiliser le bon prompt selon le sous-thème avec la logique complète
        if sub_theme == 'D6_CollectiveSuccess':
            system_prompt = build_lencioni_prompt(state)
        else:
            # Utiliser la logique MBTI complète (identique à l'ancien workflow)
            system_prompt = build_mbti_prompt(state)
        
        # Debug: Show what context is being sent to the final agent
        logger.info(f"🔍 CONTEXT SENT TO FINAL AGENT:")
        logger.info(f"  - Tool A results: {len(state.get('personalized_content', []))} items")
        logger.info(f"  - Tool B results: {len(state.get('generic_content', []))} items") 
        logger.info(f"  - Tool C results: {len(state.get('others_content', []))} items")
        logger.info(f"  - Tool D results: {len(state.get('general_content', []))} items")
        logger.info(f"  - System prompt length: {len(system_prompt)} characters")
        
        # Debug: Show complete system prompt for LangGraph Studio  
        logger.info(f"📄 COMPLETE SYSTEM PROMPT SENT TO LLM:")
        logger.info(f"{'='*60}")
        logger.info(system_prompt)
        logger.info(f"{'='*60}")
        logger.info(f"📝 USER MESSAGE:")
        logger.info(f"{state.get('user_message', 'No user message')}")
        
        # Messages pour le LLM
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_question)
        ]
        
        # ============= STREAMING UNIFIÉ POUR TOUS LES TEMPLATES =============
        logger.info(f"🔄 Starting {sub_theme} response streaming...")
        final_response = ""
        
        try:
            # Utiliser streaming explicite pour que LangGraph puisse capturer les tokens
            final_response = ""
            for chunk in llm.stream(messages):
                if chunk.content:
                    final_response += chunk.content
                    # Les tokens sont automatiquement capturés par LangGraph Studio
            
            logger.info(f"✅ {sub_theme} response generated via streaming ({len(final_response)} chars)")
            
            # Préparer le message final
            final_assistant_message = {
                "role": "assistant",
                "content": final_response,
                "type": "assistant",
                "is_final_response": True
            }
            
            # Gérer les messages
            updated_messages = manage_messages(state.get('messages', []), [final_assistant_message])
            
            return {
                "final_response": final_response,
                "messages": updated_messages,
                "user_id": state.get("user_id"),
                "user_name": state.get("user_name"),
                "user_email": state.get("user_email"),
                "user_mbti": state.get("user_mbti"),
                "lencioni_data": state.get("lencioni_data"),
                # system_prompt_debug gardé pour LangGraph Studio seulement
                "system_prompt_debug": system_prompt
            }
            
        except Exception as e:
            logger.error(f"❌ Error generating {sub_theme} response: {e}")
            return {
                "final_response": f"Désolé, une erreur s'est produite lors de l'analyse. Pouvez-vous reformuler votre question ?",
                "messages": state.get('messages', []),
                "user_id": state.get("user_id"),
                "user_name": state.get("user_name"),
                "user_email": state.get("user_email"),
                "user_mbti": state.get("user_mbti")
            }
            
    except Exception as e:
        logger.error(f"❌ Error in generate_final_response: {e}")
        return {
            "final_response": "Désolé, une erreur s'est produite. Pouvez-vous reformuler votre question ?",
            "messages": state.get('messages', []),
            "user_id": state.get("user_id"),
            "user_name": state.get("user_name"),
            "user_email": state.get("user_email"),
            "user_mbti": state.get("user_mbti")
        }

# ============= VERSION ORIGINALE POUR RÉFÉRENCE =============        
def generate_final_response_original(state: WorkflowState) -> WorkflowState:
    """Version originale conservée pour référence - reproduit l'étape 6 du workflow n8n"""
    logger.info("🤖 NODE 6: Generating final response (ORIGINAL)...")
    
    try:
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
- If no relevant info, recommend contacting jean-pierre.aerts@zestforleaders.com.


USER QUESTION: "{user_question}"

RECENT CONVERSATION HISTORY:
{history_text}

User MBTI Profile: {state.get('user_mbti', 'Unknown')}
User Temperament: {state.get('user_temperament', 'Unknown')}
Temperament Description: {state.get('temperament_description', 'Unknown')}

Context from vector search:"""

        # 🔄 NOUVEAU: Organisation par profils MBTI au lieu d'outils séparés
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
        
        # Debug: Show what context is being sent to the final agent
        logger.info(f"🔍 CONTEXT SENT TO FINAL AGENT:")
        logger.info(f"  - Tool A results: {len(state.get('personalized_content', []))} items")
        logger.info(f"  - Tool B results: {len(state.get('generic_content', []))} items") 
        logger.info(f"  - Tool C results: {len(state.get('others_content', []))} items")
        logger.info(f"  - Tool D results: {len(state.get('general_content', []))} items")
        logger.info(f"  - System prompt length: {len(system_prompt)} characters")
        
        # Debug: Show complete system prompt for LangGraph Studio  
        logger.info(f"📄 COMPLETE SYSTEM PROMPT SENT TO LLM:")
        logger.info(f"{'='*60}")
        logger.info(system_prompt)
        logger.info(f"{'='*60}")
        logger.info(f"📝 USER MESSAGE:")
        logger.info(f"{state.get('user_message', 'No user message')}")
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=state.get('user_message', '') or (state.get('messages', [{}])[-1].get('content', '') if state.get('messages') else ''))
        ]
        
        # Génération avec streaming émis vers LangGraph Studio
        logger.info("🔄 Starting streaming response generation...")
        final_response = ""
        
        try:
            # Utiliser streaming explicite pour que LangGraph puisse capturer les tokens
            final_response = ""
            for chunk in llm.stream(messages):
                if chunk.content:
                    final_response += chunk.content
                    # Les tokens sont automatiquement capturés par LangGraph Studio
            
            logger.info(f"✅ Response generated via streaming ({len(final_response)} chars)")
            
            # Ajouter explicitement la réponse finale aux messages
            final_assistant_message = {
                "role": "assistant",
                "content": final_response,
                "type": "assistant",
                "is_final_response": True  # Marquer comme réponse finale
            }
            
            updated_messages = manage_messages(state.get('messages', []), [final_assistant_message])
            
            # Retourner seulement les champs essentiels pour le frontend
            return {
                "final_response": final_response, 
                "messages": updated_messages,
                "user_id": state.get("user_id"),
                "user_name": state.get("user_name"),
                "user_email": state.get("user_email"),
                "user_mbti": state.get("user_mbti"),
                # system_prompt_debug gardé pour LangGraph Studio seulement
                "system_prompt_debug": system_prompt
            }
            
        except Exception as e:
            logger.error(f"❌ Error generating response: {e}")
            return {
                "final_response": "Je m'excuse, une erreur s'est produite. Pouvez-vous reformuler votre question ?",
                "messages": state.get('messages', []),
                "user_id": state.get("user_id"),
                "user_name": state.get("user_name"),
                "user_email": state.get("user_email"),
                "user_mbti": state.get("user_mbti")
            }
            
    except Exception as e:
        logger.info(f"❌ Error in generate_final_response: {e}")
        return {
            "final_response": "Je m'excuse, une erreur s'est produite. Pouvez-vous reformuler votre question ?",
            "messages": state.get('messages', []),
            "user_id": state.get("user_id"),
            "user_name": state.get("user_name"),
            "user_email": state.get("user_email"),
            "user_mbti": state.get("user_mbti")
        }

# Configuration du graph
workflow = StateGraph(WorkflowState)

# Ajouter tous les nodes
workflow.add_node("fetch_user_profile", fetch_user_profile)
workflow.add_node("fetch_temperament_description", fetch_temperament_description)
workflow.add_node("mbti_expert_analysis", mbti_expert_analysis)
workflow.add_node("analyze_temperament_facets", analyze_temperament_facets)  # NODE 3.5
workflow.add_node("execute_tools_ab", execute_tools_ab)
workflow.add_node("execute_tools_abc", execute_tools_abc)
workflow.add_node("execute_tools_c", execute_tools_c)
workflow.add_node("execute_tools_d", execute_tools_d)
workflow.add_node("no_tools", no_tools)
workflow.add_node("generate_final_response", generate_final_response)

# Nodes pour le flux Lencioni
workflow.add_node("lencioni_intent_analysis", lencioni_intent_analysis)
workflow.add_node("lencioni_analysis", lencioni_analysis)
workflow.add_node("lencioni_vector_search", lencioni_vector_search)

# Fonction de routage par sous-thème
def route_by_subtheme(state: WorkflowState) -> str:
    """Route vers différents flux selon le sous-thème"""
    theme = state.get('theme', '')
    sub_theme = state.get('sub_theme', '')
    
    logger.info(f"🔀 Routing: theme={theme}, sub_theme={sub_theme}")
    
    if sub_theme == 'D6_CollectiveSuccess':
        logger.info("📊 → Routing to Lencioni flow")
        return "lencioni_flow"
    else:
        logger.info("🧠 → Routing to MBTI flow")
        return "mbti_flow"

# Définir les connexions
workflow.set_entry_point("fetch_user_profile")
workflow.add_edge("fetch_user_profile", "fetch_temperament_description")

# Routage conditionnel APRÈS fetch_temperament_description (AVANT MBTI analysis)
workflow.add_conditional_edges(
    "fetch_temperament_description",
    route_by_subtheme,
    {
        "lencioni_flow": "lencioni_intent_analysis",
        "mbti_flow": "mbti_expert_analysis"
    }
)

# Pour le flux MBTI: continuer après MBTI analysis vers temperament facets
workflow.add_edge("mbti_expert_analysis", "analyze_temperament_facets")

# Pour le flux Lencioni: continuer après intent analysis vers data retrieval
workflow.add_edge("lencioni_intent_analysis", "lencioni_analysis")
# Après avoir récupéré les données, faire la recherche vectorielle
workflow.add_edge("lencioni_analysis", "lencioni_vector_search")

# Router conditionnel après l'analyse des tempéraments (au lieu d'après l'analyse MBTI)
workflow.add_conditional_edges(
    "analyze_temperament_facets",  # Changé de mbti_expert_analysis
    route_to_tools,
    {
        "execute_tools_ab": "execute_tools_ab",
        "execute_tools_abc": "execute_tools_abc", 
        "execute_tools_c": "execute_tools_c",
        "execute_tools_d": "execute_tools_d",
        "no_tools": "no_tools"
    }
)

# Tous les outils mènent à la génération de réponse
workflow.add_edge("execute_tools_ab", "generate_final_response")
workflow.add_edge("execute_tools_abc", "generate_final_response")
workflow.add_edge("execute_tools_c", "generate_final_response")
workflow.add_edge("execute_tools_d", "generate_final_response")
workflow.add_edge("no_tools", "generate_final_response")

# Flux Lencioni : la recherche vectorielle mène à la génération de réponse
workflow.add_edge("lencioni_vector_search", "generate_final_response")

# Fin du workflow
workflow.add_edge("generate_final_response", END)

# Compiler le graph (la mémoire est gérée automatiquement par LangGraph Studio)
graph = workflow.compile()

# Point d'entrée pour LangGraph Studio - pas besoin de fonction create_graph
# graph est exporté directement