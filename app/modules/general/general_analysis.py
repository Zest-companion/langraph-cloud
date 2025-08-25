"""
Fonctions d'analyse pour contenu général (introspection, etc.)
Recherche vectorielle simple basée sur la question utilisateur
"""
import logging
from typing import Dict, List, Optional
from ..common.types import WorkflowState
from ..common.config import supabase

logger = logging.getLogger(__name__)

def perform_general_vector_search(query: str, folder_filter: str = None, limit: int = 5) -> List[Dict]:
    """
    Effectue une recherche vectorielle dans les documents généraux
    Utilise EXACTEMENT la même fonction que Lencioni pour garantir la cohérence
    
    Args:
        query: Question de l'utilisateur
        folder_filter: Filtre sur le dossier (ex: "C_MaximizingIndividualPerformance/C8_Introspection")  
        limit: Nombre de résultats max
    
    Returns:
        Liste des documents trouvés
    """
    try:
        # Importer la fonction éprouvée de Lencioni
        from ..lencioni.lencioni_analysis import perform_supabase_vector_search
        
        logger.info(f"🔍 General vector search: '{query[:100]}...'")
        
        if not query or not query.strip():
            logger.warning("⚠️ Empty query provided")
            return []
        
        # Préparer les filtres de métadonnées
        metadata_filters = {}
        if folder_filter:
            logger.info(f"📁 Applying filter: folder_path = {folder_filter}")
            metadata_filters["folder_path"] = folder_filter
        
        # DEBUG: Also test without filter to see all documents
        logger.info(f"🔍 DEBUG: Testing search with filter: {metadata_filters}")
        
        # First test without filter to see all available documents
        if folder_filter:
            logger.info("🔍 DEBUG: Also searching WITHOUT filter to see all documents...")
            all_results = perform_supabase_vector_search(
                query=query,
                match_function="match_documents", 
                metadata_filters={},
                limit=10
            )
            logger.info(f"📊 Found {len(all_results)} total documents without filter:")
            for i, result in enumerate(all_results[:5]):  # Show first 5
                metadata = result.get('metadata', {})
                folder = metadata.get('folder_path', 'No folder_path')
                name = metadata.get('name', 'No name')
                logger.info(f"  {i+1}. {name} | folder_path: '{folder}'")
        
        # Utiliser la fonction Lencioni éprouvée (même logique exacte)
        results = perform_supabase_vector_search(
            query=query,
            match_function="match_documents",
            metadata_filters=metadata_filters,
            limit=limit
        )
        
        # DEBUG: Log des résultats pour comprendre le filtrage
        logger.info(f"✅ General search completed: {len(results)} results")
        for i, result in enumerate(results):
            metadata = result.get('metadata', {})
            folder = metadata.get('folder_path', 'No folder_path')
            name = metadata.get('name', 'No name')
            similarity = result.get('similarity', 0.0)
            logger.info(f"  {i+1}. {name} | folder_path: '{folder}' | similarity: {similarity:.3f}")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Error in general vector search: {e}")
        return []

def sanitize_general_results(results: List[Dict]) -> List[Dict]:
    """
    Nettoie et structure les résultats de recherche vectorielle
    
    Args:
        results: Résultats bruts de Supabase
    
    Returns:
        Résultats nettoyés et structurés
    """
    try:
        sanitized = []
        
        for result in results:
            try:
                content = result.get('content', '').strip()
                metadata = result.get('metadata', {})
                
                if not content:
                    continue
                
                # Extraire les métadonnées importantes
                document_name = metadata.get('name', 'Unknown Document')
                theme = metadata.get('theme', '')
                sub_theme = metadata.get('sub_theme', '')
                folder_path = metadata.get('folder_path', '')
                language = metadata.get('language', 'unknown')
                
                # Tronquer le contenu si trop long
                if len(content) > 1500:
                    content = content[:1500] + "..."
                
                sanitized_result = {
                    "content": content,
                    "document_name": document_name,
                    "theme": theme,
                    "sub_theme": sub_theme,
                    "folder_path": folder_path,
                    "language": language,
                    "similarity": result.get('similarity', 0.0)
                }
                
                sanitized.append(sanitized_result)
                
            except Exception as e:
                logger.warning(f"⚠️ Error sanitizing result: {e}")
                continue
        
        logger.info(f"✅ Sanitized {len(sanitized)} results")
        return sanitized
        
    except Exception as e:
        logger.error(f"❌ Error sanitizing general results: {e}")
        return []

def general_vector_search(state: WorkflowState) -> WorkflowState:
    """
    NODE: Recherche vectorielle pour contenu général
    Effectue une recherche simple basée sur la question utilisateur
    """
    logger.info("🔍 NODE: General Vector Search...")
    
    try:
        # Récupérer la question utilisateur
        user_msg = state.get('user_message', '') or (
            state.get('messages', [{}])[-1].get('content', '') if state.get('messages') else ''
        )
        
        if not user_msg or not user_msg.strip():
            logger.warning("⚠️ No user message found for search")
            return {
                **state,
                "general_vector_results": [],
                "general_search_performed": False,
                "general_search_error": "No user message provided"
            }
        
        # Déterminer le filtre de dossier basé sur main_theme + sub_theme
        main_theme = state.get('main_theme', '')
        sub_theme = state.get('sub_theme', '')
        folder_filter = None
        
        logger.info(f"🎯 Mapping: main_theme='{main_theme}', sub_theme='{sub_theme}'")
        
        if main_theme and sub_theme:
            # Mapper main_theme/sub_theme vers folder_path complet
            if main_theme == 'C_MaximizingIndividualPerformance' and sub_theme == 'C8_Introspection':
                folder_filter = "C_MaximizingIndividualPerformance/C8_Introspection"
                logger.info(f"✅ Mapped to folder_filter: '{folder_filter}'")
            # Ajouter d'autres mappings si nécessaire
        else:
            logger.warning(f"⚠️ Missing theme data: main_theme='{main_theme}', sub_theme='{sub_theme}'")
        
        # Effectuer la recherche vectorielle
        results = perform_general_vector_search(
            query=user_msg,
            folder_filter=folder_filter,
            limit=5
        )
        
        # Nettoyer les résultats
        sanitized_results = sanitize_general_results(results)
        
        logger.info(f"✅ General search completed: {len(sanitized_results)} results")
        
        return {
            **state,
            "general_vector_results": sanitized_results,
            "general_search_performed": True,
            "general_search_query": user_msg,
            "general_folder_filter": folder_filter,
            "general_search_error": None
        }
        
    except Exception as e:
        logger.error(f"❌ Error in general vector search: {e}")
        return {
            **state,
            "general_vector_results": [],
            "general_search_performed": False,
            "general_search_error": str(e)
        }