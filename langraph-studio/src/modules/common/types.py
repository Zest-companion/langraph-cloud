"""
Types de données communs
"""
from typing import Dict, List, Optional
from typing_extensions import TypedDict

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
    pcm_base: Optional[str]  # Type PCM de base de l'utilisateur
    pcm_phase: Optional[str]  # Phase PCM actuelle de l'utilisateur
    
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
    
    # Analyse Leadership Intent  
    leadership_intent_analysis: Optional[str]  # 🎯 AJOUTÉ: Analyse d'intent Leadership (Goleman)
    question_type: Optional[str]  # 🎯 AJOUTÉ: Type de question extrait (personal_style, situational, etc.)
    detected_styles: Optional[List[str]]  # 🎨 AJOUTÉ: Styles Goleman détectés
    leadership_resources: Optional[str]  # 📚 AJOUTÉ: Résultats recherche/récupération Leadership
    leadership_search_debug: Optional[str]  # 🔍 AJOUTÉ: Debug info sur la recherche leadership
    debug_leadership_intent: Optional[str]  # 🔍 AJOUTÉ: Debug leadership intent
    
    # Analyse PCM Intent
    pcm_intent_analysis: Optional[Dict]  # 🧠 AJOUTÉ: Analyse d'intent PCM (flow_type + language)
    pcm_classification: Optional[Dict]  # 🎯 NOUVEAU: Classification PCM du flow manager
    flow_type: Optional[str]  # 🎯 AJOUTÉ: Type de flow PCM (general_knowledge/self_focused/coworker_focused)
    language: Optional[str]  # 🌐 AJOUTÉ: Langue détectée (fr/en)
    pcm_base_or_phase: Optional[str]  # 🔄 AJOUTÉ: Classification BASE ou PHASE pour self_focused
    exploration_mode: Optional[str]  # 🔄 AJOUTÉ: systematic|flexible - persiste l'intention d'exploration
    pcm_specific_dimensions: Optional[List[str]]  # 🎯 AJOUTÉ: Dimensions spécifiques demandées (["perception", "strengths"], etc.)
    pcm_explored_dimensions: Optional[List[str]]  # 📋 AJOUTÉ: Liste des dimensions BASE déjà explorées
    pcm_resources: Optional[str]  # 📚 AJOUTÉ: Résultats recherche PCM formatés
    pcm_base_results: Optional[List[Dict]]  # 🎯 AJOUTÉ: Résultats vectoriels pour BASE
    pcm_phase_results: Optional[List[Dict]]  # 🔄 AJOUTÉ: Résultats vectoriels pour PHASE
    pcm_general_results: Optional[List[Dict]]  # 📚 AJOUTÉ: Résultats vectoriels généraux
    pcm_comparison_types: Optional[List[str]]  # 🆚 AJOUTÉ: Types PCM à comparer
    pcm_comparison_results: Optional[Dict]  # 📊 AJOUTÉ: Résultats de comparaison PCM
    pcm_analysis_done: Optional[bool]  # ✅ AJOUTÉ: État de l'analyse PCM
    debug_pcm_intent: Optional[str]  # 🔍 AJOUTÉ: Debug PCM intent
    pcm_search_debug: Optional[str]  # 🔍 AJOUTÉ: Debug recherche PCM
    vector_search_complete: Optional[bool]  # ✅ AJOUTÉ: État recherche vectorielle PCM
    pcm_analysis_result: Optional[str]  # 📝 AJOUTÉ: Résultat final analyse PCM
    analysis_complete: Optional[bool]  # ✅ AJOUTÉ: État analyse complète
    
    # PCM Conversational System (3-context: BASE/PHASE/ACTION_PLAN)
    pcm_conversational_context: Optional[Dict]  # 🎯 AJOUTÉ: Contexte conversationnel PCM
    pcm_context_reasoning: Optional[str]  # 🤔 AJOUTÉ: Raisonnement Chain of Thought
    pcm_transition_suggestions: Optional[Dict]  # 💡 AJOUTÉ: Suggestions de transition
    conversational_analysis_complete: Optional[bool]  # ✅ AJOUTÉ: État analyse conversationnelle
    
    # PCM Context Tracking for COWORKER flows
    has_explored_base: Optional[bool]  # 🏗️ AJOUTÉ: A exploré sa BASE personnelle
    has_explored_phase: Optional[bool]  # 📊 AJOUTÉ: A exploré sa PHASE/stress
    has_explored_action_plan: Optional[bool]  # 🎯 AJOUTÉ: A exploré son ACTION_PLAN
    base_exploration_level: Optional[int]  # 📈 AJOUTÉ: Nombre de dimensions BASE explorées (0-6)
    previous_context: Optional[str]  # 🔄 AJOUTÉ: Contexte précédent ('base', 'phase', 'action_plan', None)
    coworker_context_type: Optional[str]  # 🎯 AJOUTÉ: Type de contexte COWORKER ('contextual' ou 'direct')

    # PCM Coworker-focused flow state tracking
    coworker_step: Optional[int]  # 🎯 AJOUTÉ: Step actuel du flow coworker_focused (1-4)
    coworker_self_ok: Optional[bool]  # 💚 AJOUTÉ: État émotionnel utilisateur (+/+ ou -/-)
    coworker_other_profile: Optional[Dict]  # 👥 AJOUTÉ: Profil PCM du collègue/manager
    coworker_step_2_substep: Optional[int]  # 🔄 AJOUTÉ: Sous-étapes pour step 2 (ACTION_PLAN)
    
    # PCM Coworker transition and info gathering
    coworker_context_detected: Optional[bool]  # 🔄 AJOUTÉ: Transition coworker détectée depuis self_focused
    coworker_gathering_info: Optional[bool]  # 🤔 AJOUTÉ: Mode questions intermédiaires actif
    coworker_info_attempts: Optional[int]  # 📊 AJOUTÉ: Nombre d'échanges d'info gathering
    ready_for_coworker_analysis: Optional[bool]  # ✅ AJOUTÉ: Prêt pour analysis coworker
    coworker_step_1_attempts: Optional[int]  # 🔄 AJOUTÉ: Compteur pour step 1 coworker (éviter boucle infinie)
    
    # Analyse des tempéraments (NODE 3.5)
    temperament_analysis: Optional[Dict]
    temperament_search_results: Optional[List[Dict]]
    
    # Résultats recherche vectorielle
    personalized_content: List[Dict]
    generic_content: List[Dict]
    others_content: List[Dict]
    general_content: List[Dict]
    temperament_content: List[Dict]  # 🏛️ NOUVEAU: Contenu des tempéraments (user et/ou others)
    
    # Résultats recherche General/Introspection
    general_vector_results: Optional[List[Dict]]  # 🔍 Résultats de recherche vectorielle pour contenu général
    general_search_performed: Optional[bool]  # État de la recherche
    general_search_query: Optional[str]  # Query utilisée pour la recherche
    general_folder_filter: Optional[str]  # Filtre de dossier appliqué
    general_search_error: Optional[str]  # Erreur éventuelle
    
    # Réponse finale
    final_response: str
    streaming_active: Optional[bool]
    
    # Debug - Visible dans LangGraph Studio
    system_prompt_debug: Optional[str]
    reformulated_query_debug: Optional[str]
