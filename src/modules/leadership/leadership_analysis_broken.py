"""
Module d'analyse pour le leadership (thème A4_LeadershipStyle)
Analyse spécialisée pour les questions de style de leadership
"""

import logging
from typing import Dict, Any
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

from ..common.types import WorkflowState
from ..common.llm_utils import isolated_analysis_call_with_messages
from ..common.config import llm, analysis_llm, supabase

logger = logging.getLogger(__name__)

def leadership_intent_analysis(state: WorkflowState) -> Dict[str, Any]:
    """
    Analyzes user's intent for leadership questions using LLM
    Based on Goleman's 6 leadership styles framework
    """
    logger.info("🎯 STARTING Leadership Intent Analysis - ENTRY POINT")
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
    
    logger.info(f"📝 User query extracted: '{user_query}'")
    
    # Create prompt for leadership intent analysis (no MBTI mapping)
    system_prompt = """You are a leadership expert specialized in Goleman's 6 Leadership Styles framework.

Analyze the user's question and identify:

1. **Question Type** (select one):
   - personal_style: User asking about their own leadership style ("What's my natural style?", "Am I more authoritative?")
   - situational: When/how to use a style in specific situations ("When should I use coercive?", "What situation calls for coaching?")  
   - comparative: Comparing different leadership styles ("Authoritative vs Democratic", "Which is better?", "Strengths of each style", "Compare all 6 styles", "What are the advantages of each approach?")
   - implementation: How to develop or practice a leadership style ("How to develop coaching skills?", "Steps to improve democratic leadership")
   - specific_style: Focus on one particular Goleman style - its characteristics, limits, benefits, definition ("What are the limits of coercive?", "Explain authoritative style", "Benefits of coaching approach")
   - general_leadership: General leadership principles/questions not focused on specific Goleman styles

2. **Goleman Styles Mentioned** (if any):
   - coercive (directive, "Do what I say")
   - authoritative (visionary, "Come with me")
   - affiliative (people-focused, "People come first") 
   - democratic (participative, "What do you think?")
   - pacesetting (high standards, "Go as fast as I go")
   - coaching (developmental, "What could you try?")

3. **Intent Summary**: What does the user really want to understand?

4. **Goleman Focus**: Most relevant Goleman aspect for this question

IMPORTANT: Never make connections to MBTI profiles. Focus only on Goleman framework.

Respond in structured JSON format:
{
    "question_type": "...",
    "detected_styles": ["style1", "style2"], 
    "intent_summary": "Clear description of user's intent",
    "goleman_focus": "Most relevant Goleman aspect"
}
"""
    
    # Use isolated call to analyze intent
    logger.info("🚀 About to call LLM for leadership intent analysis")
    try:
        leadership_intent_raw = isolated_analysis_call_with_messages(
            system_content=system_prompt,
            user_content=user_query
        )
        logger.info(f"✅ Leadership intent analysis LLM call successful: {len(leadership_intent_raw)} chars")
        logger.info(f"🔤 First 200 chars of result: {leadership_intent_raw[:200]}...")
    except Exception as e:
        logger.error(f"❌ Leadership intent analysis LLM call failed: {e}")
        leadership_intent_raw = f"LLM_ERROR: {str(e)}"
    
    # Ensure we always have content for LangGraph Studio visibility
    if not leadership_intent_raw or leadership_intent_raw.strip() == "":
        logger.warning("⚠️ Empty result from LLM, using fallback")
        leadership_intent_raw = f"EMPTY_LLM_RESULT for query: '{user_query}'"
    
    # Parse JSON to extract key fields for routing logic
    question_type = 'general_leadership'  # Default fallback
    detected_styles = []  # Default fallback
    
    try:
        import json
        parsed_intent = json.loads(leadership_intent_raw)
        question_type = parsed_intent.get('question_type', 'general_leadership')
        detected_styles = parsed_intent.get('detected_styles', [])
        logger.info(f"✅ Parsed intent: type={question_type}, styles={detected_styles}")
    except json.JSONDecodeError as e:
        logger.warning(f"⚠️ Failed to parse JSON intent: {e} - using fallbacks")
    except Exception as e:
        logger.warning(f"⚠️ Error extracting intent fields: {e} - using fallbacks")
    
    # Check if leadership_intent_analysis already exists in state
    if 'leadership_intent_analysis' in state:
        logger.warning(f"⚠️ leadership_intent_analysis already exists in state: {str(state['leadership_intent_analysis'])[:100]}...")
    
    logger.info("📦 Creating return state with leadership_intent_analysis + extracted fields")
    
    return {
        **state,
        'leadership_intent_analysis': leadership_intent_raw,  # JSON complet pour LangGraph Studio
        'question_type': question_type,  # Extrait pour routing vector search
        'detected_styles': detected_styles,  # Extrait pour routing vector search  
        'leadership_analysis_done': True,
        'debug_leadership_intent': f"EXECUTED: {len(leadership_intent_raw)} chars, type={question_type}"
    }


def leadership_analysis(state: WorkflowState) -> Dict[str, Any]:
    """
    Analyse approfondie du style de leadership basée sur le profil utilisateur
    """
    logger.info("🔍 Starting Leadership Deep Analysis")
    
    user_mbti = state.get('user_mbti', '')
    leadership_intent = state.get('leadership_intent', '')
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
    
    # Créer une analyse détaillée du leadership
    system_prompt = f"""Tu es un expert en leadership organisationnel et en développement du management.
    
    Profil MBTI: {user_mbti}
    Intention identifiée: {leadership_intent}
    
    Fournis une analyse approfondie incluant:
    
    1. **Style de leadership naturel** basé sur le profil MBTI
    2. **Forces en leadership** spécifiques à ce profil
    3. **Zones de développement** potentielles
    4. **Stratégies d'amélioration** concrètes
    5. **Exemples de leaders** ayant un profil similaire
    6. **Situations optimales** pour ce style de leadership
    7. **Défis courants** et comment les surmonter
    
    Base ton analyse sur les modèles de leadership reconnus (Goleman, Kouzes & Posner, etc.)"""
    
    # Utiliser le LLM d'analyse pour une analyse approfondie
    response = analysis_llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Question: {user_query}\n\nAnalyse le style de leadership pour ce profil."}
    ])
    
    leadership_analysis = response.content
    
    logger.info("✅ Leadership analysis completed")
    
    return {
        **state,
        'leadership_analysis': leadership_analysis,
        'analysis_complete': True
    }


def leadership_vector_search(state: WorkflowState) -> Dict[str, Any]:
    """
    Recherche vectorielle intelligente dans les documents Goleman Leadership
    Utilise une stratégie adaptée selon le type de question
    """
    logger.info("🔎 Starting Leadership Vector Search")
    
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
        
    leadership_intent = state.get('leadership_intent', '')
    
    # 1. Get classification from intent analysis (or fallback to manual classification)
    question_type = state.get('question_type', 'general_leadership')
    detected_styles = state.get('detected_styles', [])
    
    # If not available from intent analysis, do manual classification
    if not question_type:
        logger.info("🔄 Fallback to manual question classification")
        question_type, detected_styles = _classify_leadership_question(user_query)
    
    logger.info(f"🎯 Using question type: {question_type}, Detected styles: {detected_styles}")
    
    # 2. Recherche vectorielle adaptée
    try:
        leadership_results = _execute_goleman_search(user_query, question_type, detected_styles, state)
        
        # 3. Formatage des résultats
        leadership_resources = _format_leadership_results(leadership_results, question_type)
        
        logger.info(f"✅ Leadership vector search completed: {len(leadership_results)} results")
        logger.info(f"📝 Formatted resources length: {len(leadership_resources)} chars")
        logger.info(f"📄 Formatted content preview: {leadership_resources[:200]}...")
        
    except Exception as e:
        logger.warning(f"⚠️ Leadership vector search failed: {str(e)}")
        logger.info("📚 Using Goleman fallback")
        
        leadership_resources = _get_goleman_fallback(question_type, detected_styles)
    
    return {
        **state,
        'leadership_resources': leadership_resources,
        'question_type': question_type,
        'detected_styles': detected_styles,
        'leadership_search_debug': f"Retrieved {len(leadership_results)} items for {question_type}",
        'vector_search_complete': True
    }


def _classify_leadership_question(query: str) -> tuple[str, list]:
    """Classifier le type de question et détecter les styles mentionnés"""
    query_lower = query.lower()
    detected_styles = []
    
    # Détecter les styles mentionnés explicitement
    style_keywords = {
        'coercive': ['coercive', 'commanding', 'directive', 'do what i say', 'authoritarian'],
        'authoritative': ['authoritative', 'visionary', 'come with me', 'vision', 'inspire'],
        'affiliative': ['affiliative', 'people first', 'harmony', 'relationship', 'empathy'],
        'democratic': ['democratic', 'participative', 'what do you think', 'consensus', 'input'],
        'pacesetting': ['pacesetting', 'go as fast as i go', 'high standards', 'excellence', 'performance'],
        'coaching': ['coaching', 'what could you try', 'development', 'mentor', 'growth']
    }
    
    for style, keywords in style_keywords.items():
        if any(keyword in query_lower for keyword in keywords):
            detected_styles.append(style)
    
    # Classifier le type de question
    if any(word in query_lower for word in ['my', 'i am', 'my style', 'natural']):
        return 'personal_style', detected_styles
        
    elif any(word in query_lower for word in ['crisis', 'conflict', 'emergency', 'change', 'when to', 'situation']):
        return 'situational', detected_styles
        
    elif any(word in query_lower for word in ['vs', 'versus', 'difference', 'compare', 'better', 'which']):
        return 'comparative', detected_styles
        
    elif any(word in query_lower for word in ['how to', 'develop', 'improve', 'practice', 'implement', 'build']):
        return 'implementation', detected_styles
        
    elif detected_styles:
        return 'specific_style', detected_styles
        
    else:
        return 'general_leadership', detected_styles


def _execute_goleman_search(query: str, question_type: str, detected_styles: list, state: dict) -> list:
    """Exécuter la recherche vectorielle avec filtres Goleman appropriés"""
    try:
        from ..lencioni.lencioni_analysis import perform_supabase_vector_search
        
        # Définir la stratégie de recherche selon le type
        search_strategies = {
            'personal_style': {
                'two_stage_search': True,  # Special handling for personal_style
                'stage1': {
                    'filters': {
                        'source_type': 'leadership_document',
                        'document_key': 'GOLEMAN_Leadership_6_Styles',
                        'content_type': 'research_foundation'
                    },
                    'k': 2
                },
                'stage2': {
                    'filters': {
                        'source_type': 'leadership_document', 
                        'document_key': 'GOLEMAN_Leadership_6_Styles',
                        'content_type': 'integration_principles',
                        'climate_impact': 'neutral',
                        'usage_context': 'general'
                    },
                    'k': 2
                }
            },
            'situational': {
                'situational_analysis': True,  # Special handling for situational questions
                'base_filters': {
                    'source_type': 'leadership_document',
                    'document_key': 'GOLEMAN_Leadership_6_Styles'
                },
                'priority_content_types': ['practical_template', 'partial_guide'],  # MASSIVE PRIORITY
                'understanding_content_types': ['style_complete'],  # Understanding mechanisms
                'complementary_content_types': ['situational_context'],  # Background complement
                'k_priority_per_style': 2,  # Per style for priority content
                'k_understanding_per_style': 1,  # Per style for understanding mechanisms
                'k_complement_per_style': 1,  # Per style for complement
                'llm_fallback': True,  # Use LLM for style detection if no styles found
                'fallback_all_styles': 4  # If LLM also fails, get general content
            },
            'comparative': {
                'parallel_style_search': True,  # Special handling for comparative
                'base_filters': {
                    'source_type': 'leadership_document',
                    'document_key': 'GOLEMAN_Leadership_6_Styles'
                },
                'content_types': ['style_complete', 'integration_principles'],
                'k_per_style': 2,  # Results per style for balance
                'k_integration': 3  # Integration principles for flexibility context
            },
            'implementation': {
                'style_focused_implementation': True,  # Special handling for implementation with style
                'base_filters': {
                    'source_type': 'leadership_document',
                    'document_key': 'GOLEMAN_Leadership_6_Styles'
                },
                'priority_search': {
                    'content_type': 'practical_template',
                    'goleman_section': 'practical_guides',
                    'k': 5  # Prioritize heavily practical templates
                },
                'complementary_search': {
                    'content_type': 'style_complete', 
                    'k': 2  # Complement with conceptual understanding
                }
            },
            'specific_style': {
                'layered_style_search': True,  # Special handling for specific style
                'base_filters': {
                    'source_type': 'leadership_document',
                    'document_key': 'GOLEMAN_Leadership_6_Styles',
                    'leadership_style': detected_styles[0] if detected_styles else None
                },
                'content_layers': ['practical_template', 'style_complete', 'research_foundation'],
                'k_per_layer': 2,  # Results per content layer
                'include_limitations': True  # For problematic styles
            },
            'general_leadership': {
                'multi_target_search': True,  # Special handling for general leadership
                'base_filters': {
                    'source_type': 'leadership_document',
                    'document_key': 'GOLEMAN_Leadership_6_Styles'
                },
                'targets': [
                    {'leadership_style': 'research', 'content_type': 'research_foundation', 'k': 3},
                    {'leadership_style': 'integration', 'content_type': 'integration_principles', 'k': 3}
                ]
            }
        }
        
        strategy = search_strategies.get(question_type, search_strategies['general_leadership'])
        
        # Handle special multi-stage direct retrieval for personal_style
        if strategy.get('two_stage_search', False):
            logger.info("🔍 Executing multi-stage DIRECT RETRIEVAL for personal_style")
            results = []
            
            # Stage 1: Research foundations - Direct retrieval
            logger.info("📚 Stage 1 - Research foundations: Direct metadata filtering")
            try:
                stage1_response = supabase.table('documents_content_test').select('content, metadata').eq(
                    'metadata->>document_key', 'GOLEMAN_Leadership_6_Styles'
                ).eq(
                    'metadata->>content_type', 'research_foundation'
                ).execute()
                
                stage1_results = []
                for item in stage1_response.data:
                    stage1_results.append({
                        'content': item['content'],
                        'metadata': item['metadata']
                    })
                results.extend(stage1_results)
                logger.info(f"📚 Stage 1 retrieved: {len(stage1_results)} research foundation items")
                
            except Exception as e:
                logger.error(f"❌ Stage 1 direct retrieval failed: {e}")
            
            # Stage 2: Integration principles - Direct retrieval with additional filters
            logger.info("🔗 Stage 2 - Integration principles: Direct metadata filtering")
            try:
                stage2_response = supabase.table('documents_content_test').select('content, metadata').eq(
                    'metadata->>document_key', 'GOLEMAN_Leadership_6_Styles'
                ).eq(
                    'metadata->>content_type', 'integration_principles'
                ).eq(
                    'metadata->>climate_impact', 'neutral'
                ).contains(
                    'metadata->usage_context', '["general"]'
                ).execute()
                
                stage2_results = []
                for item in stage2_response.data:
                    stage2_results.append({
                        'content': item['content'],
                        'metadata': item['metadata']
                    })
                results.extend(stage2_results)
                logger.info(f"🔗 Stage 2 retrieved: {len(stage2_results)} integration principles items")
                
            except Exception as e:
                logger.error(f"❌ Stage 2 direct retrieval failed: {e}")
            
            # Stage 3: Temperament leadership style - Direct retrieval
            user_temperament = state.get('user_temperament', '')
            if user_temperament:
                logger.info(f"🧬 Stage 3 - Temperament leadership style: {user_temperament}")
                try:
                    stage3_response = supabase.table('documents_content_test').select('content, metadata').eq(
                        'metadata->>storage_path', 'A_UnderstandingMyselfAndOthers/A1_PersonalityMBTI/00KEY_Personality_MBTI_Temperaments.pdf'
                    ).eq(
                        'metadata->>temperament', user_temperament
                    ).eq(
                        'metadata->>facet', 'leadership_style'
                    ).execute()
                    
                    stage3_results = []
                    for item in stage3_response.data:
                        stage3_results.append({
                            'content': item['content'],
                            'metadata': item['metadata']
                        })
                    results.extend(stage3_results)
                    logger.info(f"🧬 Stage 3 retrieved: {len(stage3_results)} temperament leadership items")
                    
                except Exception as e:
                    logger.error(f"❌ Stage 3 temperament retrieval failed: {e}")
            else:
                logger.warning("⚠️ No user temperament available for Stage 3")
            
            # Stage 4: Style analysis for detected styles (if any)
            if detected_styles:
                logger.info(f"📊 Stage 4 - Style analysis for detected styles: {detected_styles}")
                for style in detected_styles:
                    try:
                        stage4_response = supabase.table('documents_content_test').select('content, metadata').eq(
                            'metadata->>document_key', 'GOLEMAN_Leadership_6_Styles'
                        ).eq(
                            'metadata->>source_type', 'leadership_document'
                        ).eq(
                            'metadata->>leadership_style', style
                        ).eq(
                            'metadata->>goleman_section', 'style_analysis'
                        ).execute()
                        
                        stage4_results = []
                        for item in stage4_response.data:
                            stage4_results.append({
                                'content': item['content'],
                                'metadata': item['metadata'],
                                'detected_style': style
                            })
                        results.extend(stage4_results)
                        logger.info(f"📊 Stage 4 - {style}: retrieved {len(stage4_results)} style analysis items")
                        
                    except Exception as e:
                        logger.error(f"❌ Stage 4 style analysis failed for {style}: {e}")
            else:
                logger.info("ℹ️ No styles detected, skipping Stage 4 style analysis")
            
            # Add debug info about retrieved content
            content_preview = ""
            for i, result in enumerate(results[:3]):  # Show first 3 items preview
                content_preview += f"Item {i+1}: {result['content'][:100]}...\n"
            
            logger.info(f"✅ Multi-stage direct retrieval completed: {len(results)} total items")
            logger.info(f"📄 Content preview:\n{content_preview}")
            
            return results
            
        elif strategy.get('parallel_style_search', False):
            # Handle comparative questions with parallel style search
            logger.info("🆚 Executing parallel style search for comparative question")
            results = []
            
            base_filters = strategy['base_filters']
            content_types = strategy['content_types']
            k_per_style = strategy['k_per_style']
            k_integration = strategy['k_integration']
            
            # 1. Search for each detected style in parallel
            if detected_styles:
                logger.info(f"🎯 Parallel search for styles: {detected_styles}")
                
                for style in detected_styles:
                    logger.info(f"🔍 Searching for {style} style details")
                    try:
                        # Search for style_complete content for this specific style
                        style_response = supabase.table('documents_content_test').select('content, metadata').eq(
                            'metadata->>document_key', 'GOLEMAN_Leadership_6_Styles'
                        ).eq(
                            'metadata->>source_type', 'leadership_document'
                        ).eq(
                            'metadata->>content_type', 'style_complete'
                        ).eq(
                            'metadata->>leadership_style', style
                        ).limit(k_per_style).execute()
                        
                        style_results = []
                        for item in style_response.data:
                            style_results.append({
                                'content': item['content'],
                                'metadata': item['metadata']
                            })
                        results.extend(style_results)
                        logger.info(f"✅ {style}: retrieved {len(style_results)} style_complete items")
                        
                    except Exception as e:
                        logger.error(f"❌ Failed to search for {style} style: {e}")
            else:
                logger.info("📚 No specific styles detected - retrieving all 6 Goleman styles for general comparison")
                # Fallback: retrieve all 6 Goleman styles for general comparison
                all_styles = ['coercive', 'authoritative', 'affiliative', 'democratic', 'pacesetting', 'coaching']
                
                try:
                    all_styles_response = supabase.table('documents_content_test').select('content, metadata').eq(
                        'metadata->>document_key', 'GOLEMAN_Leadership_6_Styles'
                    ).eq(
                        'metadata->>source_type', 'leadership_document'
                    ).eq(
                        'metadata->>content_type', 'style_complete'
                    ).limit(12).execute()  # 2 per style * 6 styles = 12
                    
                    fallback_results = []
                    for item in all_styles_response.data:
                        fallback_results.append({
                            'content': item['content'],
                            'metadata': item['metadata']
                        })
                    results.extend(fallback_results)
                    logger.info(f"📚 Fallback: retrieved {len(fallback_results)} items covering all leadership styles")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to retrieve all styles fallback: {e}")
            
            # 2. Add integration_principles for flexibility context
            logger.info("🔗 Adding integration principles for style flexibility context")
            try:
                integration_response = supabase.table('documents_content_test').select('content, metadata').eq(
                    'metadata->>document_key', 'GOLEMAN_Leadership_6_Styles'
                ).eq(
                    'metadata->>source_type', 'leadership_document'
                ).eq(
                    'metadata->>content_type', 'integration_principles'
                ).limit(k_integration).execute()
                
                integration_results = []
                for item in integration_response.data:
                    integration_results.append({
                        'content': item['content'],
                        'metadata': item['metadata']
                    })
                results.extend(integration_results)
                logger.info(f"🔗 Added {len(integration_results)} integration principles")
                
            except Exception as e:
                logger.error(f"❌ Failed to retrieve integration principles: {e}")
            
            # 3. Add vector search for additional context (high similarity only)
            logger.info("🔍 Adding high-quality vector search for comparative context")
            try:
                from ..lencioni.lencioni_analysis import perform_supabase_vector_search
                
                # Construct folder path from state
                main_theme = state.get('main_theme', '') or state.get('theme', '')
                sub_theme = state.get('sub_theme', '')
                folder_path = f"{main_theme}/{sub_theme}" if main_theme and sub_theme else None
                
                if folder_path:
                    logger.info(f"🎯 Vector search on folder: {folder_path} (similarity > 0.6)")
                    
                    # Get vector results with high similarity threshold
                    vector_results = perform_supabase_vector_search(
                        query=query,
                        match_function='match_documents',
                        metadata_filters={
                            'folder_path': folder_path,
                            'prioritize': True,
                            'document_key': None
                        },
                        limit=5
                    )
                    
                    # Filter for high similarity only (> 0.6)
                    high_quality_results = [
                        item for item in vector_results 
                        if item.get('similarity', 0.0) > 0.6
                    ]
                    
                    # Add high-quality vector results to main results
                    for item in high_quality_results:
                        results.append({
                            'content': item.get('content', ''),
                            'metadata': item.get('metadata', {}),
                            'target_type': 'vector_context_comparative',
                            'similarity': item.get('similarity', 0.0)
                        })
                    
                    logger.info(f"🔍 Added {len(high_quality_results)} high-quality vector results (>{len(vector_results)-len(high_quality_results)} filtered out for low similarity)")
                else:
                    logger.warning("⚠️ No folder_path available for vector search")
                    
            except Exception as e:
                logger.error(f"❌ Failed to execute comparative vector search: {e}")
            
            logger.info(f"🆚 Parallel comparative search completed: {len(results)} total items")
            return results
            
        elif strategy.get('layered_style_search', False):
            # Handle specific style questions with layered search
            logger.info("🎯 Executing layered style search for specific style question")
            results = []
            
            base_filters = strategy['base_filters']
            content_layers = strategy['content_layers']  # ['practical_template', 'style_complete', 'research_foundation']
            k_per_layer = strategy['k_per_layer']
            include_limitations = strategy['include_limitations']
            target_style = base_filters.get('leadership_style')
            
            if not target_style:
                logger.warning("⚠️ No target style found for specific_style question")
                return results
            
            logger.info(f"🎯 Layered search for '{target_style}' style")
            
            # Search each content layer in order (practical → complete → foundation)
            for layer_idx, content_type in enumerate(content_layers, 1):
                logger.info(f"📚 Layer {layer_idx}/{len(content_layers)}: Searching {content_type}")
                try:
                    # research_foundation is general (not style-specific), others are filtered by style
                    if content_type == 'research_foundation':
                        layer_response = supabase.table('documents_content_test').select('content, metadata').eq(
                            'metadata->>document_key', 'GOLEMAN_Leadership_6_Styles'
                        ).eq(
                            'metadata->>source_type', 'leadership_document'
                        ).eq(
                            'metadata->>content_type', content_type
                        ).limit(k_per_layer).execute()
                    else:
                        layer_response = supabase.table('documents_content_test').select('content, metadata').eq(
                            'metadata->>document_key', 'GOLEMAN_Leadership_6_Styles'
                        ).eq(
                            'metadata->>source_type', 'leadership_document'
                        ).eq(
                            'metadata->>content_type', content_type
                        ).eq(
                            'metadata->>leadership_style', target_style
                        ).limit(k_per_layer).execute()
                    
                    layer_results = []
                    for item in layer_response.data:
                        layer_results.append({
                            'content': item['content'],
                            'metadata': item['metadata'],
                            'layer_priority': layer_idx  # For ordering
                        })
                    results.extend(layer_results)
                    logger.info(f"✅ Layer {layer_idx} ({content_type}): retrieved {len(layer_results)} items")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to search layer {content_type}: {e}")
            
            # Special handling for "problematic" styles (coercive, pacesetting)
            if include_limitations and target_style in ['coercive', 'pacesetting']:
                logger.info(f"⚠️ Including limitations and alternatives for '{target_style}' style")
                try:
                    limitations_response = supabase.table('documents_content_test').select('content, metadata').eq(
                        'metadata->>document_key', 'GOLEMAN_Leadership_6_Styles'
                    ).eq(
                        'metadata->>source_type', 'leadership_document'
                    ).eq(
                        'metadata->>leadership_style', target_style
                    ).ilike(
                        'content', '%limitation%'
                    ).execute()
                    
                    limitations_results = []
                    for item in limitations_response.data:
                        limitations_results.append({
                            'content': item['content'],
                            'metadata': item['metadata'],
                            'is_limitation': True
                        })
                    results.extend(limitations_results)
                    logger.info(f"⚠️ Added {len(limitations_results)} limitation items for {target_style}")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to retrieve limitations for {target_style}: {e}")
            
            logger.info(f"🎯 Layered specific style search completed: {len(results)} total items")
            return results
            
        elif strategy.get('multi_target_search', False):
            # Handle general leadership questions with targeted search
            logger.info("🌐 Executing multi-target search for general leadership question")
            results = []
            
            base_filters = strategy['base_filters']
            targets = strategy['targets']
            
            for target_idx, target in enumerate(targets, 1):
                target_style = target['leadership_style']
                target_content_type = target['content_type']
                target_k = target['k']
                
                logger.info(f"🎯 Target {target_idx}/{len(targets)}: {target_style} + {target_content_type}")
                try:
                    target_response = supabase.table('documents_content_test').select('content, metadata').eq(
                        'metadata->>document_key', base_filters['document_key']
                    ).eq(
                        'metadata->>source_type', base_filters['source_type']
                    ).eq(
                        'metadata->>leadership_style', target_style
                    ).eq(
                        'metadata->>content_type', target_content_type
                    ).limit(target_k).execute()
                    
                    target_results = []
                    for item in target_response.data:
                        target_results.append({
                            'content': item['content'],
                            'metadata': item['metadata'],
                            'target_type': f"{target_style}_{target_content_type}"
                        })
                    results.extend(target_results)
                    logger.info(f"✅ Target {target_idx} ({target_style} + {target_content_type}): retrieved {len(target_results)} items")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to search target {target_style} + {target_content_type}: {e}")
            
            # 3. Add vector search for additional context based on user query
            logger.info("🔍 Adding vector search for additional context")
            try:
                from ..lencioni.lencioni_analysis import perform_supabase_vector_search
                
                # Construct folder path from state
                main_theme = state.get('main_theme', '') or state.get('theme', '')
                sub_theme = state.get('sub_theme', '')
                folder_path = f"{main_theme}/{sub_theme}" if main_theme and sub_theme else None
                
                if folder_path:
                    logger.info(f"🎯 Vector search on folder: {folder_path}")
                    
                    # First try with prioritize = true
                    vector_results = perform_supabase_vector_search(
                        query=query,
                        match_function='match_documents',
                        metadata_filters={
                            'folder_path': folder_path,
                            'prioritize': True,
                            'document_key': None
                        },
                        limit=3
                    )
                    
                    # If no results with prioritize=true, try with prioritize=false
                    if not vector_results:
                        logger.info("⚠️ No prioritized results, trying non-prioritized")
                        vector_results = perform_supabase_vector_search(
                            query=query,
                            match_function='match_documents',
                            metadata_filters={
                                'folder_path': folder_path,
                                'prioritize': False,
                                'document_key': None
                            },
                            limit=3
                        )
                    
                    # Add vector results to main results
                    for item in vector_results:
                        results.append({
                            'content': item.get('content', ''),
                            'metadata': item.get('metadata', {}),
                            'target_type': 'vector_context',
                            'similarity': item.get('similarity', 0.0)
                        })
                    
                    logger.info(f"🔍 Added {len(vector_results)} vector search results")
                else:
                    logger.warning("⚠️ No folder_path available for vector search")
                    
            except Exception as e:
                logger.error(f"❌ Failed to execute vector search: {e}")
            
            logger.info(f"🌐 Multi-target general leadership search completed: {len(results)} total items")
            return results
            
        elif strategy.get('style_focused_implementation', False):
            # Handle implementation questions with style-focused approach
            logger.info("🛠️ Executing implementation search with practical priority")
            results = []
            
            base_filters = strategy['base_filters']
            priority_search = strategy['priority_search']
            complementary_search = strategy['complementary_search']
            
            # STEP 1: Use detected styles or return all 6 Goleman styles if none detected (same as situational)
            logger.info("🔍 Step 1: Determining target styles for implementation")
            
            # Use detected styles if available, otherwise use all 6 Goleman styles  
            if detected_styles:
                target_styles = detected_styles
                logger.info(f"✅ Using intent-detected styles: {target_styles}")
            else:
                # Return all 6 Goleman leadership styles for comprehensive implementation guidance
                target_styles = ['coercive', 'authoritative', 'affiliative', 'democratic', 'pacesetting', 'coaching']
                logger.info("🎯 No specific styles detected - using all 6 Goleman styles for comprehensive guidance")
            
            # Get the primary target style (first one)
            target_style = target_styles[0] if target_styles else None
            
            if target_style:
                logger.info(f"🎯 Implementation search for '{target_style}' style")
                
                # 1. Priority: Practical Templates (massively prioritized)
                logger.info("🛠️ Priority search: Practical templates")
                try:
                    practical_response = supabase.table('documents_content_test').select('content, metadata').eq(
                        'metadata->>document_key', base_filters['document_key']
                    ).eq(
                        'metadata->>source_type', base_filters['source_type']
                    ).eq(
                        'metadata->>content_type', priority_search['content_type']
                    ).eq(
                        'metadata->>goleman_section', priority_search['goleman_section']
                    ).eq(
                        'metadata->>leadership_style', target_style
                    ).limit(priority_search['k']).execute()
                    
                    practical_results = []
                    for item in practical_response.data:
                        practical_results.append({
                            'content': item['content'],
                            'metadata': item['metadata'],
                            'search_type': 'practical_priority'
                        })
                    results.extend(practical_results)
                    logger.info(f"🛠️ Retrieved {len(practical_results)} practical templates")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to retrieve practical templates: {e}")
                
                # 2. Complementary: Style Complete (conceptual understanding)
                logger.info("📊 Complementary search: Style complete")
                try:
                    conceptual_response = supabase.table('documents_content_test').select('content, metadata').eq(
                        'metadata->>document_key', base_filters['document_key']
                    ).eq(
                        'metadata->>source_type', base_filters['source_type']
                    ).eq(
                        'metadata->>content_type', complementary_search['content_type']
                    ).eq(
                        'metadata->>leadership_style', target_style
                    ).limit(complementary_search['k']).execute()
                    
                    conceptual_results = []
                    for item in conceptual_response.data:
                        conceptual_results.append({
                            'content': item['content'],
                            'metadata': item['metadata'],
                            'search_type': 'conceptual_complement'
                        })
                    results.extend(conceptual_results)
                    logger.info(f"📊 Retrieved {len(conceptual_results)} conceptual complements")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to retrieve conceptual complements: {e}")
                    
            
            logger.info(f"🛠️ Implementation search completed: {len(results)} total items")
            return results
        
        elif strategy.get('situational_analysis', False):
            # Handle situational questions with priority on practical_template + partial_guide
            logger.info("🎯 Executing situational analysis with practical priority")
            results = []
            
            base_filters = strategy['base_filters']
            priority_content_types = strategy['priority_content_types']  # ['practical_template', 'partial_guide']
            understanding_content_types = strategy['understanding_content_types']  # ['style_complete']
            complementary_content_types = strategy['complementary_content_types']  # ['situational_context']
            k_priority_per_style = strategy['k_priority_per_style']
            k_understanding_per_style = strategy['k_understanding_per_style']
            k_complement_per_style = strategy['k_complement_per_style']
            llm_fallback_enabled = strategy['llm_fallback']
            fallback_all_styles = strategy['fallback_all_styles']
            
            # STEP 1: Use detected styles or return all 6 Goleman styles if none detected
            logger.info("🔍 Step 1: Determining target styles for situational analysis")
            
            # Use detected styles if available, otherwise use all 6 Goleman styles
            if detected_styles:
                target_styles = detected_styles
                logger.info(f"✅ Using intent-detected styles: {target_styles}")
            else:
                # Return all 6 Goleman leadership styles for comprehensive situational guidance
                target_styles = ['coercive', 'authoritative', 'affiliative', 'democratic', 'pacesetting', 'coaching']
                logger.info("🎯 No specific styles detected - using all 6 Goleman styles for comprehensive guidance")
            
            # Check if we have target styles
            if target_styles:
                logger.info(f"🎯 Situational analysis for target styles: {target_styles}")
                
                # For each target style: PRIORITIZE practical_template + partial_guide
                for style in target_styles:
                    logger.info(f"🛠️ Priority search for {style}: practical_template + partial_guide")
                    
                    # Priority 1: Practical templates
                    try:
                        practical_response = supabase.table('documents_content_test').select('content, metadata').eq(
                            'metadata->>document_key', base_filters['document_key']
                        ).eq(
                            'metadata->>source_type', base_filters['source_type']
                        ).eq(
                            'metadata->>content_type', 'practical_template'
                        ).eq(
                            'metadata->>leadership_style', style
                        ).limit(k_priority_per_style).execute()
                        
                        for item in practical_response.data:
                            results.append({
                                'content': item['content'],
                                'metadata': item['metadata'],
                                'situational_priority': 'practical_template',
                                'target_style': style
                            })
                        logger.info(f"🛠️ {style} practical_template: retrieved {len(practical_response.data)} items")
                        
                    except Exception as e:
                        logger.error(f"❌ Failed practical_template search for {style}: {e}")
                    
                    # Priority 2: Partial guides
                    try:
                        partial_guide_response = supabase.table('documents_content_test').select('content, metadata').eq(
                            'metadata->>document_key', base_filters['document_key']
                        ).eq(
                            'metadata->>source_type', base_filters['source_type']
                        ).eq(
                            'metadata->>goleman_section', 'partial_guide'
                        ).eq(
                            'metadata->>leadership_style', style
                        ).limit(k_priority_per_style).execute()
                        
                        for item in partial_guide_response.data:
                            results.append({
                                'content': item['content'],
                                'metadata': item['metadata'],
                                'situational_priority': 'partial_guide',
                                'target_style': style
                            })
                        logger.info(f"🛠️ {style} partial_guide: retrieved {len(partial_guide_response.data)} items")
                        
                    except Exception as e:
                        logger.error(f"❌ Failed partial_guide search for {style}: {e}")
                    
                    # Understanding layer: Style complete
                    try:
                        style_complete_response = supabase.table('documents_content_test').select('content, metadata').eq(
                            'metadata->>document_key', base_filters['document_key']
                        ).eq(
                            'metadata->>source_type', base_filters['source_type']
                        ).eq(
                            'metadata->>content_type', 'style_complete'
                        ).eq(
                            'metadata->>leadership_style', style
                        ).limit(k_understanding_per_style).execute()
                        
                        for item in style_complete_response.data:
                            results.append({
                                'content': item['content'],
                                'metadata': item['metadata'],
                                'situational_priority': 'style_complete',
                                'target_style': style
                            })
                        logger.info(f"📊 {style} style_complete: retrieved {len(style_complete_response.data)} items")
                        
                    except Exception as e:
                        logger.error(f"❌ Failed style_complete search for {style}: {e}")
                    
                    # Complement layer: Situational context
                    try:
                        situational_response = supabase.table('documents_content_test').select('content, metadata').eq(
                            'metadata->>document_key', base_filters['document_key']
                        ).eq(
                            'metadata->>source_type', base_filters['source_type']
                        ).eq(
                            'metadata->>content_type', 'situational_context'
                        ).eq(
                            'metadata->>leadership_style', style
                        ).limit(k_complement_per_style).execute()
                        
                        for item in situational_response.data:
                            results.append({
                                'content': item['content'],
                                'metadata': item['metadata'],
                                'situational_priority': 'situational_context',
                                'target_style': style
                            })
                        logger.info(f"📋 {style} situational_context: retrieved {len(situational_response.data)} items")
                        
                    except Exception as e:
                        logger.error(f"❌ Failed situational_context search for {style}: {e}")
                        
            
            # VECTOR SEARCH ENRICHMENT: Add high-quality vector search for additional context
            logger.info("🔍 Adding vector search enrichment for situational analysis")
            try:
                from ..lencioni.lencioni_analysis import perform_supabase_vector_search
                
                # Construct folder path from state
                main_theme = state.get('main_theme', '') or state.get('theme', '')
                sub_theme = state.get('sub_theme', '')
                folder_path = f"{main_theme}/{sub_theme}" if main_theme and sub_theme else None
                
                if folder_path:
                    logger.info(f"🎯 Vector search on folder: {folder_path}")
                    
                    # First try with prioritize = true
                    vector_results = perform_supabase_vector_search(
                        query=query,
                        match_function='match_documents',
                        metadata_filters={
                            'folder_path': folder_path,
                            'prioritize': True,
                            'document_key': None
                        },
                        limit=3
                    )
                    
                    # Filter for high quality results first
                    high_quality_results = [
                        item for item in vector_results 
                        if item.get('similarity', 0.0) > 0.6  # Only high-quality matches
                    ]
                    
                    # If no high-quality results with prioritize=true, try with prioritize=false
                    if not high_quality_results:
                        logger.info("⚠️ No high-quality prioritized results, trying non-prioritized")
                        vector_results = perform_supabase_vector_search(
                            query=query,
                            match_function='match_documents',
                            metadata_filters={
                                'folder_path': folder_path,
                                'prioritize': False,
                                'document_key': None
                            },
                            limit=3
                        )
                        
                        # Filter again for high-quality results
                        high_quality_results = [
                            item for item in vector_results 
                            if item.get('similarity', 0.0) > 0.6
                        ]
                    
                    # Add high-quality vector results
                    for item in high_quality_results:
                        results.append({
                            'content': item.get('content', ''),
                            'metadata': item.get('metadata', {}),
                            'situational_priority': 'vector_context',
                            'target_style': 'contextual',
                            'similarity': item.get('similarity', 0.0),
                            'vector_search': True
                        })
                    
                    logger.info(f"🔍 Added {len(high_quality_results)} high-quality vector results")
                else:
                    logger.warning("⚠️ No folder_path available for vector search")
                    
            except Exception as e:
                logger.error(f"❌ Failed to execute situational vector search: {e}")
            
            logger.info(f"🎯 Situational analysis completed: {len(results)} total items")
            return results
            
        else:
            # Standard single-stage search for other question types
            filters = strategy['filters']
            k = strategy['k']
            
            # Si un style spécifique détecté, ajouter le filtre
            if detected_styles and question_type != 'specific_style':
                # Pour les questions comparatives, ne pas filtrer par style
                if question_type != 'comparative':
                    filters['leadership_style'] = detected_styles[0]
            
            # Nettoyer les filtres None
            clean_filters = {k: v for k, v in filters.items() if v is not None}
            
            logger.info(f"🔍 Single-stage search with filters: {clean_filters}")
            
            # Exécuter la recherche
            results = perform_supabase_vector_search(query, k, **clean_filters)
            
            logger.info(f"✅ Single-stage search completed: {len(results)} items")
            return results


def _format_leadership_results(results: list, question_type: str) -> str:
    """Formatter les résultats selon le type de question"""
    if not results:
        return ""
    
    if question_type == 'situational':
        return _format_situational_results(results)
    elif question_type == 'personal_style':
        return _format_personal_style_results(results)
    
    formatted_sections = []
    
    for result in results:
        content = result.get('content', '')
        metadata = result.get('metadata', {})
        
        leadership_style = metadata.get('leadership_style', 'general')
        content_type = metadata.get('content_type', 'unknown')
        climate_impact = metadata.get('climate_impact', 'neutral')
        style_motto = metadata.get('style_motto', '')
        temperament = metadata.get('temperament', '')
        facet = metadata.get('facet', '')
        
        # Formater selon le type de contenu
        if content_type == 'research_foundation':
            section_title = "🔬 **Research Foundation**"
        elif content_type == 'integration_principles':
            section_title = "🔗 **Integration Principles**"
        elif facet == 'leadership_style' and temperament:
            section_title = f"🧬 **{temperament} Temperament - Leadership Style**"
        elif content_type == 'style_complete':
            impact_emoji = "⬆️" if climate_impact == "highly_positive" else "✅" if climate_impact == "positive" else "⚠️" if climate_impact == "negative" else "➡️"
            section_title = f"{impact_emoji} **{leadership_style.title()} Style** - \"{style_motto.replace('_', ' ').title()}\""
        elif content_type == 'practical_template':
            section_title = f"🛠️ **{leadership_style.title()} - Practical Guide**"
        else:
            section_title = f"📄 **{leadership_style.title()}**"
        
        # Pour personal_style, inclure le contenu complet pour l'auto-évaluation
        content_preview = content[:500] if len(content) > 500 else content
        
        # Pas de "..." si contenu complet    
        if question_type == 'personal_style':
            formatted_sections.append(f"{section_title}\n{content_preview}")
        else:
            formatted_sections.append(f"{section_title}\n{content_preview}...")
    
    return "\n\n".join(formatted_sections)


                        
                    except Exception as e:
                        logger.error(f"❌ Failed partial_guide search for {style}: {e}")
                    
                    # Understanding layer: Style complete (for mechanisms comprehension)
                    try:
                        style_complete_response = supabase.table('documents_content_test').select('content, metadata').eq(
                            'metadata->>document_key', base_filters['document_key']
                        ).eq(
                            'metadata->>source_type', base_filters['source_type']
                        ).eq(
                            'metadata->>content_type', 'style_complete'
                        ).eq(
                            'metadata->>leadership_style', style
                        ).limit(k_understanding_per_style).execute()
                        
                        for item in style_complete_response.data:
                            results.append({
                                'content': item['content'],
                                'metadata': item['metadata'],
                                'situational_priority': 'style_complete',
                                'target_style': style
                            })
                        logger.info(f"📊 {style} style_complete: retrieved {len(style_complete_response.data)} items")
                        
                    except Exception as e:
                        logger.error(f"❌ Failed style_complete search for {style}: {e}")
                    
                    # Complement: Situational context
                    try:
                        situational_response = supabase.table('documents_content_test').select('content, metadata').eq(
                            'metadata->>document_key', base_filters['document_key']
                        ).eq(
                            'metadata->>source_type', base_filters['source_type']
                        ).eq(
                            'metadata->>content_type', 'situational_context'
                        ).eq(
                            'metadata->>leadership_style', style
                        ).limit(k_complement_per_style).execute()
                        
                        for item in situational_response.data:
                            results.append({
                                'content': item['content'],
                                'metadata': item['metadata'],
                                'situational_priority': 'situational_context',
                                'target_style': style
                            })
                        logger.info(f"📋 {style} situational_context: retrieved {len(situational_response.data)} items")
                        
                    except Exception as e:
                        logger.error(f"❌ Failed situational_context search for {style}: {e}")
                        
            
            # VECTOR SEARCH ENRICHMENT: Add high-quality vector search for additional context
            logger.info("🔍 Adding vector search enrichment for situational analysis")
            try:
                from ..lencioni.lencioni_analysis import perform_supabase_vector_search
                
                # Construct folder path from state
                main_theme = state.get('main_theme', '') or state.get('theme', '')
                sub_theme = state.get('sub_theme', '')
                folder_path = f"{main_theme}/{sub_theme}" if main_theme and sub_theme else None
                
                if folder_path:
                    logger.info(f"🎯 Vector search on folder: {folder_path}")
                    
                    # First try with prioritize = true
                    vector_results = perform_supabase_vector_search(
                        query=query,
                        match_function='match_documents',
                        metadata_filters={
                            'folder_path': folder_path,
                            'prioritize': True,
                            'document_key': None
                        },
                        limit=3
                    )
                    
                    # Filter for high quality results first
                    high_quality_results = [
                        item for item in vector_results 
                        if item.get('similarity', 0.0) > 0.6  # Only high-quality matches
                    ]
                    
                    # If no high-quality results with prioritize=true, try with prioritize=false
                    if not high_quality_results:
                        logger.info("⚠️ No high-quality prioritized results, trying non-prioritized")
                        vector_results = perform_supabase_vector_search(
                            query=query,
                            match_function='match_documents',
                            metadata_filters={
                                'folder_path': folder_path,
                                'prioritize': False,
                                'document_key': None
                            },
                            limit=3
                        )
                        
                        # Filter again for high quality results
                        high_quality_results = [
                            item for item in vector_results 
                            if item.get('similarity', 0.0) > 0.6  # Only high-quality matches
                        ]
                    
                    for item in high_quality_results:
                        results.append({
                            'content': item.get('content', ''),
                            'metadata': item.get('metadata', {}),
                            'situational_priority': 'vector_context',
                            'target_style': 'contextual',
                            'similarity': item.get('similarity', 0.0),
                            'vector_search': True
                        })
                    
                    logger.info(f"🔍 Added {len(high_quality_results)} high-quality vector results (>{len(vector_results)-len(high_quality_results)} filtered out for low similarity)")
                else:
                    logger.warning("⚠️ No folder_path available for vector search")
                    
            except Exception as e:
                logger.error(f"❌ Failed to execute situational vector search: {e}")
            
            logger.info(f"🎯 Situational analysis completed: {len(results)} total items")
            return results
            
        else:
            # Standard single-stage search for other question types
            filters = strategy['filters']
            k = strategy['k']
            
            # Si un style spécifique détecté, ajouter le filtre
            if detected_styles and question_type != 'specific_style':
                # Pour les questions comparatives, ne pas filtrer par style
                if question_type != 'comparative':
                    filters['leadership_style'] = detected_styles[0]
            
            # Nettoyer les filtres None
            clean_filters = {k: v for k, v in filters.items() if v is not None}
            
            logger.info(f"🔍 Single-stage search with filters: {clean_filters}")
            
            # Exécuter la recherche
            results = perform_supabase_vector_search(query, k, **clean_filters)
            
            return results
        
    except Exception as e:
        logger.error(f"Error in Goleman search: {e}")
        return []


def _analyze_situational_styles_with_llm(user_question: str, state: WorkflowState) -> list:
    """
    Use LLM to analyze situational question and detect appropriate leadership styles
    Returns list of detected styles or empty list if low confidence
    """
    logger.info("🤖 Starting LLM situational style detection")
    
    try:
        prompt = f"""
Analyze this leadership situational question and identify the 1-3 most appropriate Goleman styles.

QUESTION: {user_question}

GOLEMAN STYLES:
- coercive: Crisis situations, urgency, discipline, immediate control, serious problems
- authoritative: Vision, change, clear direction, mobilization, transformation, goals
- affiliative: Conflicts, harmony, low morale, human relations, demoralized team
- democratic: Complex decisions, consensus, team participation, brainstorm, team input
- pacesetting: High performance, excellence, speed, high standards, efficiency
- coaching: Development, training, potential, mentoring, skills, growth

RESPONSE (strict JSON):
{{"styles": ["style1", "style2"], "confidence": "high/medium/low", "reasoning": "brief explanation"}}

IMPORTANT: Respond ONLY with the JSON, no text before or after.
"""

        # Use isolated LLM call for style detection
        response_content = isolated_analysis_call_with_messages(
            system_content="You are a Goleman leadership expert. Respond only with strict JSON.",
            user_content=prompt
        )
        
        logger.info(f"🤖 LLM response: {response_content}")
        
        # Parse JSON response
        import json
        analysis = json.loads(response_content.strip())
        
        confidence = analysis.get("confidence", "low")
        detected_styles = analysis.get("styles", [])
        reasoning = analysis.get("reasoning", "No reasoning provided")
        
        logger.info(f"🤖 LLM analysis - Confidence: {confidence}, Styles: {detected_styles}")
        logger.info(f"🤖 LLM reasoning: {reasoning}")
        
        # Return styles only if confidence is medium or high
        if confidence in ["medium", "high"] and detected_styles:
            logger.info(f"✅ LLM detected styles accepted: {detected_styles}")
            return detected_styles
        else:
            logger.warning(f"❌ LLM confidence too low ({confidence}) or no styles detected")
            return []
            
    except json.JSONDecodeError as e:
        logger.error(f"❌ LLM response JSON parsing failed: {e}")
        logger.error(f"❌ Raw response: {response_content}")
        return []
    except Exception as e:
        logger.error(f"❌ LLM situational analysis failed: {e}")
        return []


def _format_leadership_results(results: list, question_type: str) -> str:
    """Formatter les résultats selon le type de question"""
    if not results:
        return ""
    
    # Special formatting for different question types
    if question_type == 'comparative':
        return _format_comparative_results(results)
    elif question_type == 'specific_style':
        return _format_specific_style_results(results)
    elif question_type == 'general_leadership':
        return _format_general_leadership_results(results)
    elif question_type == 'implementation':
        return _format_implementation_results(results)
    elif question_type == 'personal_style':
        return _format_personal_style_results(results)
    elif question_type == 'situational':
        return _format_situational_results(results)
    
    formatted_sections = []
    
    for result in results:
        content = result.get('content', '')
        metadata = result.get('metadata', {})
        
        leadership_style = metadata.get('leadership_style', 'general')
        content_type = metadata.get('content_type', 'unknown')
        climate_impact = metadata.get('climate_impact', 'neutral')
        style_motto = metadata.get('style_motto', '')
        temperament = metadata.get('temperament', '')
        facet = metadata.get('facet', '')
        
        # Formater selon le type de contenu
        if content_type == 'research_foundation':
            section_title = "🔬 **Research Foundation**"
        elif content_type == 'integration_principles':
            section_title = "🔗 **Integration Principles**"
        elif facet == 'leadership_style' and temperament:
            section_title = f"🧬 **{temperament} Temperament - Leadership Style**"
        elif content_type == 'style_complete':
            impact_emoji = "⬆️" if climate_impact == "highly_positive" else "✅" if climate_impact == "positive" else "⚠️" if climate_impact == "negative" else "➡️"
            section_title = f"{impact_emoji} **{leadership_style.title()} Style** - \"{style_motto.replace('_', ' ').title()}\""
        elif content_type == 'practical_template':
            section_title = f"🛠️ **{leadership_style.title()} - Practical Guide**"
        else:
            section_title = f"📄 **{leadership_style.title()}**"
        
        # Pour personal_style, inclure le contenu complet pour l'auto-évaluation
        if question_type == 'personal_style':
            content_preview = content  # Contenu complet
        elif question_type == 'implementation':
            content_preview = content[:600]
        else:
            content_preview = content[:500]
        
        # Pas de "..." si contenu complet    
        if question_type == 'personal_style':
            formatted_sections.append(f"{section_title}\n{content_preview}")
        else:
            formatted_sections.append(f"{section_title}\n{content_preview}...")
    
    return "\n\n".join(formatted_sections)


def _format_comparative_results(results: list) -> str:
    """
    Special formatting for comparative questions to create comparison matrices
    Exploits style_motto and climate_impact for structured comparisons
    """
    if not results:
        return ""
    
    # Organize results by content type and style
    styles_data = {}
    integration_principles = []
    vector_results = []
    
    for result in results:
        content = result.get('content', '')
        metadata = result.get('metadata', {})
        content_type = metadata.get('content_type', '')
        leadership_style = metadata.get('leadership_style', 'unknown')
        target_type = result.get('target_type', '')
        
        if content_type == 'style_complete':
            if leadership_style not in styles_data:
                styles_data[leadership_style] = {
                    'content': content,
                    'motto': metadata.get('style_motto', '').replace('_', ' ').title(),
                    'climate_impact': metadata.get('climate_impact', 'neutral'),
                    'usage_context': metadata.get('usage_context', []),
                    'approach': metadata.get('approach', ''),
                    'key_behaviors': metadata.get('key_behaviors', ''),
                }
        elif content_type == 'integration_principles':
            integration_principles.append(content)
        elif target_type == 'vector_context_comparative':
            vector_results.append(result)
    
    # Build comparison matrix format
    formatted_output = ""
    
    if styles_data:
        formatted_output += "🆚 **LEADERSHIP STYLES COMPARISON**\n\n"
        
        # Create comparison matrix headers
        style_names = list(styles_data.keys())
        formatted_output += "| Aspect | " + " | ".join([f"**{style.title()}**" for style in style_names]) + " |\n"
        formatted_output += "|" + "---|" * (len(style_names) + 1) + "\n"
        
        # Style mottos row
        motto_row = "| **Motto** | " + " | ".join([f'"{styles_data[style]["motto"]}"' for style in style_names]) + " |\n"
        formatted_output += motto_row
        
        # Climate impact row with emojis
        impact_row = "| **Climate Impact** | "
        impact_cells = []
        for style in style_names:
            impact = styles_data[style]['climate_impact']
            if impact == 'highly_positive':
                impact_cells.append("⬆️ Highly Positive")
            elif impact == 'positive':
                impact_cells.append("✅ Positive")
            elif impact == 'negative':
                impact_cells.append("⚠️ Negative")
            else:
                impact_cells.append("➡️ Neutral")
        impact_row += " | ".join(impact_cells) + " |\n"
        formatted_output += impact_row
        
        # Detailed comparison sections
        formatted_output += "\n"
        for i, style in enumerate(style_names, 1):
            data = styles_data[style]
            impact_emoji = "⬆️" if data['climate_impact'] == "highly_positive" else "✅" if data['climate_impact'] == "positive" else "⚠️" if data['climate_impact'] == "negative" else "➡️"
            
            formatted_output += f"\n**{i}. {impact_emoji} {style.title()} Style** - \"{data['motto']}\"\n"
            formatted_output += f"{data['content']}\n"
    
    # Add integration principles for flexibility context
    if integration_principles:
        formatted_output += "\n\n🔗 **STYLE INTEGRATION & FLEXIBILITY**\n"
        formatted_output += "Understanding when and how to blend leadership styles:\n\n"
        
        for i, principle in enumerate(integration_principles, 1):
            formatted_output += f"**Principle {i}:**\n{principle}\n\n"
    
    # Add high-quality vector search context (supplementary information)
    if vector_results:
        formatted_output += "\n\n📋 **Additional Context**\n"
        formatted_output += "Supplementary information highly relevant to your comparison:\n\n"
        
        for i, result in enumerate(vector_results, 1):
            content = result.get('content', '')
            similarity = result.get('similarity', 0.0)
            formatted_output += f"**Context {i}** (Relevance: {similarity:.2f})\n{content}\n\n"
    
    return formatted_output


def _format_specific_style_results(results: list) -> str:
    """
    Special formatting for specific style questions with layered ordering
    Orders by profundity: practical templates → complete analysis → conceptual foundations
    """
    if not results:
        return ""
    
    # Organize results by layer priority and separate limitations
    layer_results = {'practical': [], 'complete': [], 'foundation': []}
    limitation_results = []
    target_style = None
    
    for result in results:
        content = result.get('content', '')
        metadata = result.get('metadata', {})
        content_type = metadata.get('content_type', '')
        leadership_style = metadata.get('leadership_style', '')
        is_limitation = result.get('is_limitation', False)
        
        # Capture target style
        if leadership_style and not target_style:
            target_style = leadership_style
        
        if is_limitation:
            limitation_results.append(result)
        elif content_type == 'practical_template':
            layer_results['practical'].append(result)
        elif content_type == 'style_complete':
            layer_results['complete'].append(result)
        elif content_type == 'research_foundation':
            layer_results['foundation'].append(result)
    
    # Build formatted output in priority order
    formatted_output = ""
    style_title = target_style.title() if target_style else "Leadership Style"
    
    # Get style metadata for header
    style_metadata = {}
    if layer_results['complete']:
        style_metadata = layer_results['complete'][0].get('metadata', {})
    elif layer_results['practical']:
        style_metadata = layer_results['practical'][0].get('metadata', {})
    
    climate_impact = style_metadata.get('climate_impact', 'neutral')
    style_motto = style_metadata.get('style_motto', '').replace('_', ' ').title()
    
    # Header with impact indicator
    impact_emoji = "⬆️" if climate_impact == "highly_positive" else "✅" if climate_impact == "positive" else "⚠️" if climate_impact == "negative" else "➡️"
    formatted_output += f"# {impact_emoji} **{style_title} Leadership Style**\n"
    if style_motto:
        formatted_output += f"*\"{style_motto}\"*\n\n"
    
    # 1. Practical Templates First (highest priority)
    if layer_results['practical']:
        formatted_output += "## 🛠️ **Practical Implementation**\n"
        formatted_output += "Actionable templates and step-by-step guidance:\n\n"
        for i, result in enumerate(layer_results['practical'], 1):
            content = result.get('content', '')
            formatted_output += f"### Template {i}\n{content}\n\n"
    
    # 2. Complete Style Analysis (medium priority)
    if layer_results['complete']:
        formatted_output += "## 📊 **Complete Style Analysis**\n"
        formatted_output += "Comprehensive understanding of this leadership approach:\n\n"
        for i, result in enumerate(layer_results['complete'], 1):
            content = result.get('content', '')
            formatted_output += f"{content}\n\n"
    
    # 3. Conceptual Foundations (lowest priority)
    if layer_results['foundation']:
        formatted_output += "## 🔬 **Conceptual Foundations**\n"
        formatted_output += "Research-based theoretical background:\n\n"
        for i, result in enumerate(layer_results['foundation'], 1):
            content = result.get('content', '')
            formatted_output += f"### Foundation {i}\n{content}\n\n"
    
    # 4. Limitations and Alternatives (for problematic styles)
    if limitation_results:
        formatted_output += f"## ⚠️ **Important Limitations & Alternatives**\n"
        formatted_output += f"Critical considerations for the {style_title} style:\n\n"
        for i, result in enumerate(limitation_results, 1):
            content = result.get('content', '')
            formatted_output += f"### Limitation {i}\n{content}\n\n"
    
    return formatted_output


def _format_general_leadership_results(results: list) -> str:
    """
    Special formatting for general leadership questions
    Organizes by research foundations and integration principles
    """
    if not results:
        return ""
    
    # Organize results by target type
    research_results = []
    integration_results = []
    vector_results = []
    
    for result in results:
        target_type = result.get('target_type', '')
        if 'research_foundation' in target_type:
            research_results.append(result)
        elif 'integration_principles' in target_type:
            integration_results.append(result)
        elif target_type == 'vector_context':
            vector_results.append(result)
    
    formatted_output = ""
    
    # 1. Scientific Research Foundations
    if research_results:
        formatted_output += "# 🔬 **Scientific Research Foundations**\n"
        formatted_output += "Evidence-based insights from Goleman's leadership research:\n\n"
        
        for i, result in enumerate(research_results, 1):
            content = result.get('content', '')
            formatted_output += f"## Research Finding {i}\n{content}\n\n"
    
    # 2. Integration Frameworks
    if integration_results:
        formatted_output += "# 🔗 **Leadership Integration Frameworks**\n"
        formatted_output += "Global frameworks for applying leadership principles:\n\n"
        
        for i, result in enumerate(integration_results, 1):
            content = result.get('content', '')
            formatted_output += f"## Integration Principle {i}\n{content}\n\n"
    
    # 3. Vector Search Context
    if vector_results:
        formatted_output += "# 🎯 **Contextual Insights**\n"
        formatted_output += "Additional relevant content based on your specific question:\n\n"
        
        for i, result in enumerate(vector_results, 1):
            content = result.get('content', '')
            similarity = result.get('similarity', 0.0)
            formatted_output += f"## Insight {i} (Relevance: {similarity:.2f})\n{content}\n\n"
    
    return formatted_output


def _format_implementation_results(results: list) -> str:
    """
    Special formatting for implementation questions with proper sequencing:
    1. Practical "how-to" templates first
    2. Conceptual "why it works" understanding second
    """
    if not results:
        return ""
    
    # Organize results by search type
    practical_results = []
    conceptual_results = []
    target_style = None
    
    for result in results:
        search_type = result.get('search_type', '')
        metadata = result.get('metadata', {})
        
        # Capture target style for header
        if metadata.get('leadership_style') and not target_style:
            target_style = metadata['leadership_style']
            
        if search_type == 'practical_priority':
            practical_results.append(result)
        elif search_type == 'conceptual_complement':
            conceptual_results.append(result)
    
    formatted_output = ""
    style_title = target_style.title() if target_style else "Leadership"
    
    # Get style metadata for header
    style_metadata = {}
    if practical_results:
        style_metadata = practical_results[0].get('metadata', {})
    elif conceptual_results:
        style_metadata = conceptual_results[0].get('metadata', {})
    
    climate_impact = style_metadata.get('climate_impact', 'neutral')
    style_motto = style_metadata.get('style_motto', '').replace('_', ' ').title()
    
    # Header with impact indicator
    impact_emoji = "⬆️" if climate_impact == "highly_positive" else "✅" if climate_impact == "positive" else "⚠️" if climate_impact == "negative" else "➡️"
    formatted_output += f"# {impact_emoji} **How to Implement {style_title} Leadership**\n"
    if style_motto:
        formatted_output += f"*\"{style_motto}\"*\n\n"
    
    # 1. PRACTICAL "HOW-TO" FIRST (immediate action)
    if practical_results:
        formatted_output += "## 🛠️ **Immediate Implementation Guide**\n"
        formatted_output += "Step-by-step practical templates to start using this style right away:\n\n"
        
        for i, result in enumerate(practical_results, 1):
            content = result.get('content', '')
            formatted_output += f"### Practical Step {i}\n{content}\n\n"
    
    # 2. CONCEPTUAL "WHY IT WORKS" SECOND (deeper understanding)
    if conceptual_results:
        formatted_output += "## 📊 **Why This Approach Works**\n"
        formatted_output += "Understanding the emotional intelligence competencies and psychological foundations:\n\n"
        
        for i, result in enumerate(conceptual_results, 1):
            content = result.get('content', '')
            formatted_output += f"### Conceptual Foundation {i}\n{content}\n\n"
    
    return formatted_output


def _format_personal_style_results(results: list) -> str:
    """
    Special formatting for personal_style questions with multi-stage content
    Organizes: research foundation → integration principles → temperament → detected styles analysis
    """
    if not results:
        return ""
    
    # Organize results by type and source
    research_foundation = []
    integration_principles = []
    temperament_leadership = []
    style_analysis = []
    
    for result in results:
        content = result.get('content', '')
        metadata = result.get('metadata', {})
        content_type = metadata.get('content_type', '')
        facet = metadata.get('facet', '')
        goleman_section = metadata.get('goleman_section', '')
        detected_style = result.get('detected_style', '')
        
        if content_type == 'research_foundation':
            research_foundation.append(result)
        elif content_type == 'integration_principles':
            integration_principles.append(result)
        elif facet == 'leadership_style':  # Temperament leadership
            temperament_leadership.append(result)
        elif goleman_section == 'style_analysis' and detected_style:
            style_analysis.append(result)
    
    formatted_output = ""
    
    # 1. Research Foundation (conceptual base)
    if research_foundation:
        formatted_output += "# 🔬 **Leadership Style Assessment Framework**\n"
        formatted_output += "Scientific foundation for understanding your natural leadership approach:\n\n"
        
        for i, result in enumerate(research_foundation, 1):
            content = result.get('content', '')
            formatted_output += f"## Foundation {i}\n{content}\n\n"
    
    # 2. Integration Principles (flexibility guidance)
    if integration_principles:
        formatted_output += "# 🔗 **Style Flexibility & Integration**\n"
        formatted_output += "Understanding how to adapt and blend leadership approaches:\n\n"
        
        for i, result in enumerate(integration_principles, 1):
            content = result.get('content', '')
            formatted_output += f"## Integration Principle {i}\n{content}\n\n"
    
    # 3. Temperament-Based Leadership Tendencies
    if temperament_leadership:
        formatted_output += "# 🧬 **Your Natural Leadership Tendencies**\n"
        formatted_output += "How your temperament influences your leadership approach:\n\n"
        
        for i, result in enumerate(temperament_leadership, 1):
            content = result.get('content', '')
            metadata = result.get('metadata', {})
            temperament = metadata.get('temperament', 'Unknown')
            formatted_output += f"## {temperament} Leadership Pattern\n{content}\n\n"
    
    # 4. Detected Styles Analysis (if styles were identified)
    if style_analysis:
        formatted_output += "# 🎯 **Your Identified Leadership Styles**\n"
        formatted_output += "Detailed analysis of the leadership styles that resonate with you:\n\n"
        
        # Group by detected style
        styles_content = {}
        for result in style_analysis:
            style = result.get('detected_style', 'Unknown')
            if style not in styles_content:
                styles_content[style] = []
            styles_content[style].append(result)
        
        for style, style_results in styles_content.items():
            formatted_output += f"## {style.title()} Style Analysis\n"
            for result in style_results:
                content = result.get('content', '')
                formatted_output += f"{content}\n\n"
    
    return formatted_output


def _format_situational_results(results: list) -> str:
    """
    Special formatting for situational questions with MASSIVE PRIORITY to practical_template + partial_guide
    Organizes: practical templates → partial guides → situational context
    """
    if not results:
        return ""
    
    # Organize results by priority level and style
    practical_templates = []
    partial_guides = []
    style_completes = []
    situational_contexts = []
    vector_contexts = []
    styles_involved = set()
    llm_detected_flags = []
    fallback_flags = []
    
    for result in results:
        situational_priority = result.get('situational_priority', '')
        target_style = result.get('target_style', 'unknown')
        is_llm_detected = result.get('llm_detected', False)
        is_fallback = result.get('fallback', False)
        is_vector_search = result.get('vector_search', False)
        
        styles_involved.add(target_style)
        
        if is_llm_detected:
            llm_detected_flags.append(target_style)
        if is_fallback:
            fallback_flags.append(target_style)
            
        if situational_priority == 'practical_template':
            practical_templates.append(result)
        elif situational_priority == 'partial_guide':
            partial_guides.append(result)
        elif situational_priority == 'style_complete':
            style_completes.append(result)
        elif situational_priority == 'situational_context':
            situational_contexts.append(result)
        elif situational_priority == 'vector_context':
            vector_contexts.append(result)
    
    # Build formatted output with MASSIVE PRIORITY structure
    formatted_output = ""
    
    # Header with detection method info
    if llm_detected_flags:
        formatted_output += f"# 🤖 **Situational Leadership Guidance** (AI-Enhanced Analysis)\n"
        formatted_output += f"*Leadership styles identified through AI analysis: {', '.join(set(llm_detected_flags)).title()}*\n\n"
    elif fallback_flags:
        formatted_output += f"# 🔄 **General Situational Leadership Guidance**\n"
        formatted_output += f"*Comprehensive approaches across leadership styles*\n\n"
    else:
        styles_list = ', '.join([s.title() for s in styles_involved if s != 'unknown'])
        formatted_output += f"# 🎯 **Situational Leadership Guidance**\n"
        if styles_list:
            formatted_output += f"*Focus on: {styles_list} Leadership*\n\n"
    
    # PRIORITY 1: PRACTICAL TEMPLATES (80% weight - immediate actionable guidance)
    if practical_templates:
        formatted_output += "## 🛠️ **IMMEDIATE ACTION TEMPLATES**\n"
        formatted_output += "Ready-to-use practical approaches for your situation:\n\n"
        
        # Group by style for better organization
        templates_by_style = {}
        for result in practical_templates:
            style = result.get('target_style', 'general')
            if style not in templates_by_style:
                templates_by_style[style] = []
            templates_by_style[style].append(result)
        
        for style, style_templates in templates_by_style.items():
            if len(templates_by_style) > 1:  # Multiple styles, show style headers
                style_emoji = "🤖" if style in llm_detected_flags else "🔄" if style in fallback_flags else "🎯"
                formatted_output += f"### {style_emoji} **{style.title()} Approach**\n"
            
            for i, result in enumerate(style_templates, 1):
                content = result.get('content', '')
                formatted_output += f"**Template {i}:**\n{content}\n\n"
    
    # PRIORITY 2: PARTIAL GUIDES (15% weight - strategic understanding)
    if partial_guides:
        formatted_output += "## 📋 **STRATEGIC GUIDANCE**\n"
        formatted_output += "Deeper strategic understanding for situational leadership:\n\n"
        
        # Group by style
        guides_by_style = {}
        for result in partial_guides:
            style = result.get('target_style', 'general')
            if style not in guides_by_style:
                guides_by_style[style] = []
            guides_by_style[style].append(result)
        
        for style, style_guides in guides_by_style.items():
            if len(guides_by_style) > 1:  # Multiple styles, show style headers
                style_emoji = "🤖" if style in llm_detected_flags else "🔄" if style in fallback_flags else "🎯"
                formatted_output += f"### {style_emoji} **{style.title()} Strategic Guide**\n"
            
            for i, result in enumerate(style_guides, 1):
                content = result.get('content', '')
                formatted_output += f"**Guide {i}:**\n{content}\n\n"
    
    # UNDERSTANDING LAYER: STYLE COMPLETE (mechanisms comprehension)
    if style_completes:
        formatted_output += "## 🧠 **UNDERSTANDING THE MECHANISMS**\n"
        formatted_output += "How and why these leadership approaches work in your situation:\n\n"
        
        # Group by style
        complete_by_style = {}
        for result in style_completes:
            style = result.get('target_style', 'general')
            if style not in complete_by_style:
                complete_by_style[style] = []
            complete_by_style[style].append(result)
        
        for style, style_completes_list in complete_by_style.items():
            if len(complete_by_style) > 1:  # Multiple styles, show style headers
                style_emoji = "🤖" if style in llm_detected_flags else "🔄" if style in fallback_flags else "🎯"
                formatted_output += f"### {style_emoji} **{style.title()} Style Mechanisms**\n"
            
            for i, result in enumerate(style_completes_list, 1):
                content = result.get('content', '')
                metadata = result.get('metadata', {})
                
                # Add style motto if available
                style_motto = metadata.get('style_motto', '').replace('_', ' ').title()
                if style_motto:
                    formatted_output += f"**{style.title()} - \"{style_motto}\"**\n{content}\n\n"
                else:
                    formatted_output += f"**Mechanism {i}:**\n{content}\n\n"
    
    # PRIORITY 3: SITUATIONAL CONTEXT (5% weight - background understanding)
    if situational_contexts:
        formatted_output += "## 📖 **SITUATIONAL CONTEXT**\n"
        formatted_output += "Background context for understanding when to apply these approaches:\n\n"
        
        for i, result in enumerate(situational_contexts, 1):
            content = result.get('content', '')
            target_style = result.get('target_style', 'general')
            formatted_output += f"**Context {i} - {target_style.title()}:**\n{content}\n\n"
    
    # VECTOR SEARCH ENRICHMENT: Additional contextual insights
    if vector_contexts:
        formatted_output += "## 🔍 **ADDITIONAL CONTEXTUAL INSIGHTS**\n"
        formatted_output += "High-relevance supplementary content from your leadership knowledge base:\n\n"
        
        for i, result in enumerate(vector_contexts, 1):
            content = result.get('content', '')
            similarity = result.get('similarity', 0.0)
            metadata = result.get('metadata', {})
            
            # Get document info if available
            doc_title = metadata.get('title', 'Contextual Insight')
            formatted_output += f"**Insight {i}** (Relevance: {similarity:.2f})\n"
            formatted_output += f"*Source: {doc_title}*\n{content}\n\n"
    
    return formatted_output


def _get_goleman_fallback(question_type: str, detected_styles: list) -> str:
    """Fallback avec connaissances Goleman structurées"""
    
    if question_type == 'personal_style':
        return """
🔬 **Research Foundation (Goleman)**
Based on research by Hay/McBer with 3,871 executives, Daniel Goleman identified six distinct leadership styles, each with different impacts on organizational climate and performance.

✅ **Authoritative Style** - "Come with me" 
Most effective overall (+0.54 climate impact). Mobilizes people toward a vision. Best when clear direction is needed.

🛠️ **Coaching Style** - "What could you try?"
Develops people for the future (+0.42 impact). Focus on long-term development and personal growth.

✅ **Affiliative Style** - "People come first"
Creates harmony and builds emotional bonds (+0.32 impact). Excellent for healing rifts and motivating during stress.
        """
        
    elif detected_styles:
        style = detected_styles[0]
        style_info = {
            'authoritative': ('✅ **Authoritative** - "Come with me"', 'Mobilizes toward vision, most effective overall (+0.54 climate impact)'),
            'coaching': ('🛠️ **Coaching** - "What could you try?"', 'Develops people for future success (+0.42 climate impact)'), 
            'affiliative': ('✅ **Affiliative** - "People come first"', 'Creates harmony, builds relationships (+0.32 climate impact)'),
            'democratic': ('✅ **Democratic** - "What do you think?"', 'Builds consensus through participation (+0.28 climate impact)'),
            'pacesetting': ('⚠️ **Pacesetting** - "Go as fast as I go"', 'Sets high standards, can be counterproductive (-0.25 climate impact)'),
            'coercive': ('⚠️ **Coercive** - "Do what I say"', 'Demands immediate compliance, use only in crisis (-0.32 climate impact)')
        }
        
        if style in style_info:
            title, description = style_info[style]
            return f"{title}\n{description}"
    
    return """
🔬 **Goleman's 6 Leadership Styles**

**Positive Impact Styles:**
✅ **Authoritative** - "Come with me" (+0.54 impact)
🛠️ **Coaching** - "What could you try?" (+0.42 impact)  
✅ **Affiliative** - "People come first" (+0.32 impact)
✅ **Democratic** - "What do you think?" (+0.28 impact)

**Use with Caution:**
⚠️ **Pacesetting** - "Go as fast as I go" (-0.25 impact)
⚠️ **Coercive** - "Do what I say" (-0.32 impact, crisis only)
    """