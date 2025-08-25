"""
Utilitaires de test pour le PCM Agent
Permet de tester et comparer l'agent vs approche traditionnelle
"""

import logging
from typing import Dict, Any
from .pcm_agent import PCMAgent, compare_agent_vs_traditional
from ..common.types import WorkflowState

logger = logging.getLogger(__name__)

def create_test_state(
    user_message: str,
    pcm_base: str = "harmonizer", 
    pcm_phase: str = "imaginer",
    explored_dimensions: list = None,
    exploration_mode: str = "flexible",
    language: str = "en"
) -> WorkflowState:
    """Créer un état de test pour l'agent PCM"""
    
    if explored_dimensions is None:
        explored_dimensions = []
    
    # Simulation de ressources PCM
    mock_pcm_resources = f"""
Perception of your {pcm_base.title()} Base: 
You naturally perceive the world through emotions and personal values...

Strengths of your {pcm_base.title()} Base:
Your main strengths include compassion, empathy, and creating harmony...

Interaction Style of your {pcm_base.title()} Base:
You prefer a benevolent interaction style, caring and supportive...
"""
    
    return {
        'user_message': user_message,
        'pcm_base': pcm_base,
        'pcm_phase': pcm_phase,
        'pcm_explored_dimensions': explored_dimensions,
        'exploration_mode': exploration_mode,
        'language': language,
        'pcm_resources': mock_pcm_resources,
        'messages': [
            {'role': 'user', 'content': 'Tell me about my PCM profile'},
            {'role': 'assistant', 'content': 'I see you have a Harmonizer BASE. This means...'},
            {'role': 'user', 'content': user_message}
        ],
        'flow_type': 'self_focused',
        'pcm_base_or_phase': 'base'
    }

def test_pcm_agent_scenarios():
    """Test l'agent PCM avec différents scénarios"""
    
    scenarios = [
        {
            'name': 'Profile Rejection',
            'user_message': "I don't recognize myself in that description",
            'expected_behavior': 'Should redirect to Jean-Pierre Aerts'
        },
        {
            'name': 'Profile Validation', 
            'user_message': "Yes, that sounds exactly like me!",
            'expected_behavior': 'Should proceed to dimension exploration'
        },
        {
            'name': 'Specific Dimension Request',
            'user_message': "Tell me more about my strengths",
            'expected_behavior': 'Should focus on strengths dimension'
        },
        {
            'name': 'Multiple Dimensions',
            'user_message': "I want to know about my perception and interaction style",
            'expected_behavior': 'Should handle multiple dimensions'
        },
        {
            'name': 'Phase Interest',
            'user_message': "What motivates me right now? I'm feeling stressed lately",
            'explored_dimensions': ['Perception', 'Strengths', 'Interaction Style', 'Personality Parts'],
            'expected_behavior': 'Should suggest PHASE transition'
        },
        {
            'name': 'Systematic Exploration',
            'user_message': "What should I explore next?",
            'explored_dimensions': ['Perception', 'Strengths'],
            'exploration_mode': 'systematic',
            'expected_behavior': 'Should suggest next dimension systematically'
        }
    ]
    
    agent = PCMAgent()
    results = []
    
    for scenario in scenarios:
        logger.info(f"\n🧪 Testing scenario: {scenario['name']}")
        logger.info(f"📝 Input: {scenario['user_message']}")
        logger.info(f"🎯 Expected: {scenario['expected_behavior']}")
        
        try:
            # Créer l'état de test
            test_state = create_test_state(
                user_message=scenario['user_message'],
                explored_dimensions=scenario.get('explored_dimensions', []),
                exploration_mode=scenario.get('exploration_mode', 'flexible')
            )
            
            # Tester l'agent
            response = agent.handle_pcm_interaction(test_state)
            
            result = {
                'scenario': scenario['name'],
                'input': scenario['user_message'],
                'output': response[:200] + '...' if len(response) > 200 else response,
                'full_output': response,
                'success': True,
                'length': len(response)
            }
            
            logger.info(f"✅ Response: {result['output']}")
            
        except Exception as e:
            result = {
                'scenario': scenario['name'],
                'input': scenario['user_message'], 
                'error': str(e),
                'success': False
            }
            logger.error(f"❌ Error: {e}")
        
        results.append(result)
    
    return results

def enable_pcm_agent_for_testing():
    """
    Fonction utilitaire pour activer l'agent PCM dans le workflow
    Modifie temporairement le comportement pour tester
    """
    logger.info("🔬 PCM Agent testing enabled")
    return {
        'experimental_pcm_agent': True,
        'pcm_agent_test_mode': True
    }

if __name__ == "__main__":
    # Test direct de l'agent
    print("🧪 Testing PCM Agent...")
    results = test_pcm_agent_scenarios()
    
    print(f"\n📊 Test Results Summary:")
    successful = sum(1 for r in results if r.get('success', False))
    print(f"✅ Successful: {successful}/{len(results)}")
    
    for result in results:
        status = "✅" if result.get('success') else "❌"
        print(f"{status} {result['scenario']}: {result.get('output', result.get('error', 'Unknown'))[:100]}...")