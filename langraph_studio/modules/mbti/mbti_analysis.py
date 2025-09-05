"""
Fonctions d'analyse MBTI
"""
import json
import logging
from typing import Dict, List, Optional
from ..common.types import WorkflowState
from ..common.llm_utils import isolated_analysis_call_with_messages

logger = logging.getLogger(__name__)

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
    
    # Initialiser reformulated_query pour éviter l'erreur UnboundLocalError
    reformulated_query = None
    
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
        
        # TODO: Implémenter reformulate_query_with_context
        # Pour l'instant, utiliser la query originale
        reformulated_query = user_msg
        
        # reformulated_query = reformulate_query_with_context(
        #     user_msg=user_msg,
        #     messages_history=messages_history,
        #     historical_mbti_types=historical_mbti_types,
        #     user_mbti=state.get('user_mbti')
        # )
        logger.info(f"🔄 Using original query (reformulation disabled): '{reformulated_query}'")
        
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


def extract_mbti_profiles_from_text(text: str, exclude_user_type: str = None) -> List[str]:
    """Extrait les profils MBTI du texte (4 lettres)"""
    import re
    
    # Pattern pour les types MBTI valides
    mbti_pattern = r'\b(INTJ|INTP|ENTJ|ENTP|INFJ|INFP|ENFJ|ENFP|ISTJ|ISFJ|ESTJ|ESFJ|ISTP|ISFP|ESTP|ESFP)\b'
    
    # Trouver tous les matches
    matches = re.findall(mbti_pattern, text, re.IGNORECASE)
    
    # Convertir en majuscules et éliminer les doublons
    profiles = list(set([match.upper() for match in matches]))
    
    # Exclure le type de l'utilisateur si spécifié
    if exclude_user_type:
        profiles = [p for p in profiles if p != exclude_user_type.upper()]
    
    return profiles


def validate_mbti_type(mbti_type: str) -> bool:
    """Valide qu'un type MBTI est correct"""
    valid_types = [
        'INTJ', 'INTP', 'ENTJ', 'ENTP', 
        'INFJ', 'INFP', 'ENFJ', 'ENFP',
        'ISTJ', 'ISFJ', 'ESTJ', 'ESFJ',
        'ISTP', 'ISFP', 'ESTP', 'ESFP'
    ]
    return mbti_type.upper() in valid_types