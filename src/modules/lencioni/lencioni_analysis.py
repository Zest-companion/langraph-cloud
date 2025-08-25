"""
Fonctions d'analyse Lencioni pour les dysfonctions d'équipe
"""
import logging
import json
import re
from typing import Dict, List, Optional
from ..common.types import WorkflowState
from ..common.config import supabase
from ..common.llm_utils import isolated_analysis_call

logger = logging.getLogger(__name__)

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


def get_dysfunction_scores(user_id: str, dysfunctions: List[str] = None) -> List[Dict]:
    """Récupère les scores des dysfonctions pour un utilisateur"""
    try:
        query = supabase.table("lensioni_team_assessment_score").select(
            "dysfunction, score, level, summary"
        ).eq("profile_id", user_id)
        
        if dysfunctions:
            # Filtrer par dysfonctions spécifiques si fourni
            query = query.in_("dysfunction", dysfunctions)
        
        response = query.execute()
        return response.data if response.data else []
        
    except Exception as e:
        logger.error(f"❌ Error fetching dysfunction scores: {e}")
        return []


def get_dysfunction_details(user_id: str, dysfunction: str) -> List[Dict]:
    """Récupère les détails d'une dysfonction spécifique"""
    try:
        # 1. Récupérer l'ID du score pour cette dysfonction
        score_response = supabase.table("lensioni_team_assessment_score").select(
            "id"
        ).eq("profile_id", user_id).eq("dysfunction", dysfunction).execute()
        
        if not score_response.data:
            return []
        
        # 2. Récupérer les détails
        details_data = []
        for score_record in score_response.data:
            team_assessment_score_id = score_record["id"]
            
            details_response = supabase.table("lensioni_details").select(
                "dysfunction, question, score, level"
            ).eq("team_assessment_score_id", team_assessment_score_id).execute()
            
            if details_response.data:
                details_data.extend(details_response.data)
        
        return details_data
        
    except Exception as e:
        logger.error(f"❌ Error fetching dysfunction details: {e}")
        return []


def validate_dysfunction_name(dysfunction: str) -> bool:
    """Valide qu'un nom de dysfonction est correct"""
    valid_dysfunctions = ['Trust', 'Conflict', 'Commitment', 'Accountability', 'Results']
    return dysfunction in valid_dysfunctions


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


def perform_supabase_vector_search(query: str, match_function: str = 'match_documents', 
                                 metadata_filters: Dict = None, limit: int = 5) -> List[Dict]:
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
        from langchain_openai import OpenAIEmbeddings
        
        # Generate embedding for the query - MÊME MODÈLE que vectorize_documents_simple.py
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
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


def extract_dysfunctions_from_text(text: str) -> List[str]:
    """Extrait les dysfonctions mentionnées dans un texte"""
    text_lower = text.lower()
    dysfunctions = []
    
    if 'trust' in text_lower or 'confiance' in text_lower:
        dysfunctions.append('Trust')
    if 'conflict' in text_lower or 'conflit' in text_lower:
        dysfunctions.append('Conflict')
    if 'commitment' in text_lower or 'engagement' in text_lower:
        dysfunctions.append('Commitment')
    if 'accountability' in text_lower or 'responsabilité' in text_lower:
        dysfunctions.append('Accountability')
    if 'results' in text_lower or 'résultats' in text_lower:
        dysfunctions.append('Results')
    
    return list(set(dysfunctions))