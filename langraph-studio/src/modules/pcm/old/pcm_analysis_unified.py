"""
PCM Analysis Unified - Nouveau système unifié PCM
Remplace PCMFlowManager + PCMConversationalAnalysis par un système unique
"""
import logging
from typing import Dict, Any, List
from ..common.types import WorkflowState

logger = logging.getLogger(__name__)

def pcm_unified_analysis_entry_point(state: WorkflowState) -> Dict[str, Any]:
    """
    Point d'entrée principal du système PCM unifié
    
    SYSTÈME UNIFIÉ COMPLET:
    - Classification globale + analyse fine en une seule passe
    - Transitions dynamiques naturelles (PHASE → COWORKER, etc.)
    - Gestion complète BASE/PHASE/ACTION_PLAN + COWORKER/COMPARISON
    - Performance optimisée (un seul appel LLM au lieu de plusieurs)
    """
    logger.info("🧠 PCM Unified Analysis - New System Entry Point")
    
    try:
        from .pcm_unified_analysis import pcm_unified_intent_analysis
        result = pcm_unified_intent_analysis(state)
        logger.info(f"✅ Unified analysis completed: {result.get('flow_type', 'N/A')}")
        return result
    except Exception as e:
        logger.error(f"❌ Unified system failed: {e}")
        return _fallback_to_current_system(state)

def _fallback_to_current_system(state: WorkflowState) -> Dict[str, Any]:
    """Fallback vers le système actuel (pcm_analysis_new) en cas d'erreur"""
    logger.warning("🔄 Falling back to current working system")
    
    try:
        from .pcm_analysis_new import pcm_analysis_with_flow_manager
        result = pcm_analysis_with_flow_manager(state)
        result['unified_fallback_used'] = True
        return result
    except Exception as e:
        logger.error(f"❌ Even fallback failed: {e}")
        return {
            **state,
            'flow_type': 'self_base',
            'pcm_base_or_phase': 'base',
            'error': 'All PCM systems failed',
            'unified_fallback_used': True
        }

# Fonctions de compatibilité pour tests
def test_unified_vs_current(state: WorkflowState) -> Dict[str, Any]:
    """Compare les résultats du système unifié vs système actuel"""
    logger.info("🧪 Testing Unified vs Current System")
    
    try:
        # Test système unifié
        logger.info("Testing unified system...")
        from .pcm_unified_analysis import pcm_unified_intent_analysis
        unified_result = pcm_unified_intent_analysis(state)
        
        # Test système actuel
        logger.info("Testing current system...")
        from .pcm_analysis_new import pcm_analysis_with_flow_manager
        current_result = pcm_analysis_with_flow_manager(state)
        
        return {
            'unified_result': unified_result,
            'current_result': current_result,
            'comparison_complete': True
        }
        
    except Exception as e:
        logger.error(f"❌ Comparison test failed: {e}")
        return {
            'error': str(e),
            'comparison_complete': False
        }