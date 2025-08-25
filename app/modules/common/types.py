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
    pcm_analysis_done: Optional[bool]  # ✅ AJOUTÉ: État de l'analyse PCM
    debug_pcm_intent: Optional[str]  # 🔍 AJOUTÉ: Debug PCM intent
    pcm_search_debug: Optional[str]  # 🔍 AJOUTÉ: Debug recherche PCM
    vector_search_complete: Optional[bool]  # ✅ AJOUTÉ: État recherche vectorielle PCM
    pcm_analysis_result: Optional[str]  # 📝 AJOUTÉ: Résultat final analyse PCM
    analysis_complete: Optional[bool]  # ✅ AJOUTÉ: État analyse complète
    
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
