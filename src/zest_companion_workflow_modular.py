"""
ZEST MBTI Workflow - Version Modulaire
Workflow minimal utilisant uniquement les modules refactorisés.

Ce fichier est un simple orchestrateur qui utilise les fonctions des modules.
Aucune logique métier n'est dupliquée ici.
"""

import logging
from langgraph.graph import StateGraph, END

# ============================================================================
# IMPORTS MODULAIRES PURS - Toute la logique est dans les modules
# ============================================================================

# Types et configuration
from src.modules.common.types import WorkflowState

# Fonctions d'analyse
from src.modules.mbti.mbti_analysis import mbti_expert_analysis
from src.modules.lencioni.lencioni_analysis import (
    lencioni_intent_analysis, 
    lencioni_analysis, 
    lencioni_vector_search
)
from src.modules.leadership.leadership_analysis import (
    leadership_intent_analysis,
    # leadership_analysis,  # REMOVED - bypassed for direct Goleman approach
    leadership_vector_search
)
from src.modules.pcm.pcm_analysis import (
    pcm_intent_analysis,
    pcm_vector_search,
    update_explored_dimensions
)
from src.modules.general.general_analysis import general_vector_search

# Fonctions d'outils et profil
from src.modules.tools.vector_tools import (
    fetch_user_profile,
    fetch_temperament_description,
    analyze_temperament_facets,
    route_to_tools,
    execute_tools_ab,
    execute_tools_abc,
    execute_tools_c,
    execute_tools_d,
    no_tools
)

# Génération de réponse
from src.modules.response.response_generator import generate_final_response

# ============================================================================
# FONCTIONS DE ROUTAGE SIMPLES
# ============================================================================

def route_by_subtheme(state: WorkflowState) -> str:
    """Route vers différents flux selon le sous-thème"""
    # Le frontend envoie 'main_theme' et 'sub_theme', pas 'theme'
    main_theme = state.get('main_theme', '') or state.get('theme', '')
    sub_theme = state.get('sub_theme', '')
    
    logging.info(f"🔀 MODULAR Routing: main_theme={main_theme}, sub_theme={sub_theme}")
    
    # Routage spécial pour Leadership
    if main_theme == 'A_UnderstandingMyselfAndOthers' and sub_theme == 'A4_LeadershipStyle':
        logging.info("👔 → Routing to Leadership flow")
        return "leadership_flow"
    elif main_theme == 'A_UnderstandingMyselfAndOthers' and sub_theme == 'A2_PersonalityPCM':
        logging.info("🧠 → Routing to PCM flow")
        return "pcm_flow"
    elif sub_theme == 'D6_CollectiveSuccess':
        logging.info("📊 → Routing to Lencioni flow")
        return "lencioni_flow"
    elif sub_theme == 'C8_Introspection':
        logging.info("🔍 → Routing to General flow")
        return "general_flow"
    else:
        logging.info("🧠 → Routing to MBTI flow")
        return "mbti_flow"

# ============================================================================
# CONSTRUCTION DU GRAPHE MODULAIRE
# ============================================================================

# Créer le graphe avec WorkflowState
workflow = StateGraph(WorkflowState)

# Ajouter tous les nodes en utilisant les fonctions des modules
workflow.add_node("fetch_user_profile", fetch_user_profile)
workflow.add_node("fetch_temperament_description", fetch_temperament_description)

# Nodes d'analyse (modulaires)
workflow.add_node("mbti_expert_analysis", mbti_expert_analysis)
workflow.add_node("analyze_temperament_facets", analyze_temperament_facets)

# Nodes d'exécution des outils (modulaires)
workflow.add_node("execute_tools_ab", execute_tools_ab)
workflow.add_node("execute_tools_abc", execute_tools_abc)
workflow.add_node("execute_tools_c", execute_tools_c)
workflow.add_node("execute_tools_d", execute_tools_d)
workflow.add_node("no_tools", no_tools)

# Nodes Lencioni (modulaires)
workflow.add_node("lencioni_intent_analysis", lencioni_intent_analysis)
workflow.add_node("lencioni_analysis", lencioni_analysis)
workflow.add_node("lencioni_vector_search", lencioni_vector_search)

# Nodes Leadership (modulaires)
workflow.add_node("leadership_intent_analysis", leadership_intent_analysis)
# workflow.add_node("leadership_analysis", leadership_analysis)  # REMOVED - bypassed for direct Goleman approach
workflow.add_node("leadership_vector_search", leadership_vector_search)

# Nodes PCM (modulaires)
workflow.add_node("pcm_intent_analysis", pcm_intent_analysis)
workflow.add_node("pcm_vector_search", pcm_vector_search)

# Nodes General (modulaires)
workflow.add_node("general_vector_search", general_vector_search)

# Node de génération de réponse (modulaire)
workflow.add_node("generate_final_response", generate_final_response)

# ============================================================================
# DÉFINITION DES CONNEXIONS (identique à l'original)
# ============================================================================

# Point d'entrée
workflow.set_entry_point("fetch_user_profile")
workflow.add_edge("fetch_user_profile", "fetch_temperament_description")

# Routage conditionnel selon le sous-thème
workflow.add_conditional_edges(
    "fetch_temperament_description",
    route_by_subtheme,
    {
        "leadership_flow": "leadership_intent_analysis",
        "pcm_flow": "pcm_intent_analysis",
        "lencioni_flow": "lencioni_intent_analysis",
        "mbti_flow": "mbti_expert_analysis",
        "general_flow": "general_vector_search"
    }
)

# Flux MBTI
workflow.add_edge("mbti_expert_analysis", "analyze_temperament_facets")

# Routage vers les outils après analyse des tempéraments
workflow.add_conditional_edges(
    "analyze_temperament_facets",
    route_to_tools,
    {
        "execute_tools_ab": "execute_tools_ab",
        "execute_tools_abc": "execute_tools_abc", 
        "execute_tools_c": "execute_tools_c",
        "execute_tools_d": "execute_tools_d",
        "no_tools": "no_tools"
    }
)

# Flux Lencioni
workflow.add_edge("lencioni_intent_analysis", "lencioni_analysis")
workflow.add_edge("lencioni_analysis", "lencioni_vector_search")

# Flux Leadership
workflow.add_edge("leadership_intent_analysis", "leadership_vector_search")

# Flux PCM
workflow.add_edge("pcm_intent_analysis", "pcm_vector_search")

# Tous les chemins mènent à la génération de réponse
workflow.add_edge("execute_tools_ab", "generate_final_response")
workflow.add_edge("execute_tools_abc", "generate_final_response")
workflow.add_edge("execute_tools_c", "generate_final_response")
workflow.add_edge("execute_tools_d", "generate_final_response")
workflow.add_edge("no_tools", "generate_final_response")
workflow.add_edge("lencioni_vector_search", "generate_final_response")
workflow.add_edge("leadership_vector_search", "generate_final_response")
workflow.add_edge("pcm_vector_search", "generate_final_response")

# Flux General - connexion directe vers génération de réponse
workflow.add_edge("general_vector_search", "generate_final_response")

# Fin du workflow
workflow.add_edge("generate_final_response", END)

# Compiler le graphe modulaire
graph = workflow.compile()

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info("🔧 ✅ MODULAR Workflow graph compiled successfully!")
logger.info("📊 Uses 100% modular structure from modules/ directory")
logger.info("🎯 No duplicated logic - pure orchestration")
logger.info("🔗 Identical graph structure to original workflow")