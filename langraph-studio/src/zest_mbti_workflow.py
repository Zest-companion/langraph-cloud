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
    model="gpt-4", 
    temperature=0.7, 
    streaming=True,  # Streaming activé pour la réponse finale
    callbacks=[],
    max_tokens=700
)

# LLM d'analyse COMPLÈTEMENT ISOLÉ
analysis_llm = ChatOpenAI(
    model="gpt-3.5-turbo", 
    temperature=0.3,
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
            temperature=0.3,
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
            temperature=0.3,
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
        logger.info(f"🔄 Calling analysis_llm for reformulation...")
        response = analysis_llm.invoke([HumanMessage(content=reformulation_prompt)])
        reformulated = response.content.strip().strip('"').strip("'")
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
                temperature=0.3,
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

# NODE 6: Agent principal - Génération de la réponse finale
def generate_final_response(state: WorkflowState) -> WorkflowState:
    """Reproduit l'étape 6 du workflow n8n - Agent principal"""
    logger.info("🤖 NODE 6: Generating final response...")
    
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
- End with 1 question offering to explore the topic further
- Aim for 220–350 words; use short paragraphs and bullet points where helpful; prioritize concrete, actionable steps and include 1–2 concise examples if relevant

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

# Définir les connexions
workflow.set_entry_point("fetch_user_profile")
workflow.add_edge("fetch_user_profile", "fetch_temperament_description")
workflow.add_edge("fetch_temperament_description", "mbti_expert_analysis")

# NOUVEAU: Ajouter NODE 3.5 après NODE 3
workflow.add_edge("mbti_expert_analysis", "analyze_temperament_facets")

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

# Fin du workflow
workflow.add_edge("generate_final_response", END)

# Compiler le graph (la mémoire est gérée automatiquement par LangGraph Studio)
graph = workflow.compile()

# Point d'entrée pour LangGraph Studio - pas besoin de fonction create_graph
# graph est exporté directement