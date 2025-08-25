"""
PCM Agent - Agent LangChain spécialisé pour le Process Communication Model
Architecture expérimentale pour remplacer la logique PCM actuelle
"""

import logging
from typing import Dict, List, Any, Optional
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

from ..common.types import WorkflowState
from ..common.config import llm

logger = logging.getLogger(__name__)

# ============================================================================
# OUTILS SPÉCIALISÉS PCM
# ============================================================================

@tool
def validate_base_profile(user_response: str, pcm_base: str, pcm_resources: str) -> str:
    """
    Validate if user recognizes themselves in their BASE profile
    Returns: 'recognized' | 'rejected' | 'uncertain'
    """
    user_lower = user_response.lower()
    
    # Patterns de rejet
    rejection_patterns = [
        "don't recognize", "doesn't fit", "not me", "not like me",
        "ne me reconnais pas", "ce n'est pas moi", "ça ne me correspond pas"
    ]
    
    # Patterns de reconnaissance
    recognition_patterns = [
        "that's me", "sounds like me", "recognize myself", "fits me",
        "c'est moi", "ça me correspond", "je me reconnais"
    ]
    
    for pattern in rejection_patterns:
        if pattern in user_lower:
            return f"rejected - User said: '{user_response}' - Need to redirect to Jean-Pierre Aerts"
    
    for pattern in recognition_patterns:
        if pattern in user_lower:
            return f"recognized - User validates their {pcm_base} BASE profile"
    
    return f"uncertain - Need more validation from user about {pcm_base} BASE"

@tool
def suggest_next_dimension(explored_dimensions: List[str], exploration_mode: str, current_context: str) -> str:
    """
    Suggest the next BASE dimension to explore based on systematic or flexible mode
    """
    all_dimensions = [
        "Perception", "Strengths", "Interaction Style", 
        "Personality Parts", "Channels of Communication", "Environmental Preferences"
    ]
    
    remaining = [dim for dim in all_dimensions if dim not in explored_dimensions]
    
    if not remaining:
        return "all_explored - All 6 BASE dimensions have been explored. Time to suggest PHASE transition."
    
    if exploration_mode == "systematic":
        # Ordre suggéré pour exploration systématique
        suggested_order = [
            "Perception", "Strengths", "Interaction Style",
            "Personality Parts", "Channels of Communication", "Environmental Preferences"
        ]
        for dim in suggested_order:
            if dim in remaining:
                return f"systematic_next - Suggest exploring: {dim} ({len(explored_dimensions)+1}/6 dimensions)"
    
    # Mode flexible - donner des choix
    if len(remaining) <= 3:
        return f"flexible_final - Offer remaining dimensions: {', '.join(remaining)}"
    else:
        return f"flexible_choice - Offer choice between: {', '.join(remaining[:3])}"

@tool
def detect_phase_transition_readiness(conversation_history: str, explored_dimensions: List[str]) -> str:
    """
    Detect if user is ready to transition from BASE to PHASE exploration
    """
    history_lower = conversation_history.lower()
    
    # Signaux de transition vers PHASE
    phase_signals = [
        "motivation", "needs", "stress", "current", "now", "lately",
        "these days", "recently", "what drives me"
    ]
    
    signal_count = sum(1 for signal in phase_signals if signal in history_lower)
    
    # Critères de préparation
    base_well_explored = len(explored_dimensions) >= 4  # Au moins 4 dimensions BASE
    phase_interest_shown = signal_count >= 2
    
    if base_well_explored and phase_interest_shown:
        return "ready - User shows PHASE interest and BASE is well explored"
    elif phase_interest_shown:
        return "interested - User shows PHASE interest but BASE needs more exploration"
    elif base_well_explored:
        return "base_complete - BASE well explored, could suggest PHASE"
    else:
        return "continue_base - Continue BASE exploration"

@tool
def handle_dimension_request(user_query: str, pcm_base: str) -> str:
    """
    Detect and handle specific dimension requests from user query
    """
    query_lower = user_query.lower()
    
    dimension_keywords = {
        "perception": ["perception", "see", "view", "filter", "interpret"],
        "strengths": ["strengths", "talents", "abilities", "good at"],
        "interaction_style": ["interaction", "communicate", "relate", "engage"],
        "personality_part": ["personality", "behavior", "energy", "parts"],
        "channel_communication": ["communication", "channel", "talk", "express"],
        "environmental_preferences": ["environment", "setting", "prefer", "comfortable"]
    }
    
    detected_dimensions = []
    for dimension, keywords in dimension_keywords.items():
        if any(keyword in query_lower for keyword in keywords):
            detected_dimensions.append(dimension)
    
    if len(detected_dimensions) == 1:
        return f"single_dimension - User asking about: {detected_dimensions[0]}"
    elif len(detected_dimensions) > 1:
        return f"multiple_dimensions - User asking about: {', '.join(detected_dimensions)}"
    else:
        return f"general_query - No specific dimension detected in: {user_query}"

# ============================================================================
# AGENT PCM PRINCIPAL
# ============================================================================

class PCMAgent:
    """
    Agent LangChain spécialisé pour les interactions PCM
    Remplace la logique complexe de prompt selection
    """
    
    def __init__(self):
        self.tools = [
            validate_base_profile,
            suggest_next_dimension, 
            detect_phase_transition_readiness,
            handle_dimension_request
        ]
        
        # Prompt système pour l'agent
        self.system_prompt = """You are ZEST COMPANION's PCM specialist agent.
        
Your role: Guide users through their Process Communication Model (PCM) exploration with expertise and care.

PCM CORE CONCEPTS:
• BASE: How you naturally perceive the world (never changes) - 6 dimensions to explore
• PHASE: Your current needs and motivations (can change over time)
• The Okay/Not-Okay Matrix: Always guide toward "I'm Okay, You're Okay"

YOUR APPROACH:
1. **Start with BASE validation** - Ensure user recognizes their profile
2. **Systematic or flexible exploration** - Adapt to user preference  
3. **Use your tools** to make decisions and provide personalized guidance
4. **Natural transitions** - Move from BASE to PHASE when ready
5. **Expert fallback** - Redirect to jean-pierre.aerts@zestforleaders.com if profile doesn't fit

CONVERSATION STYLE:
• Warm, professional, and insightful
• Ask follow-up questions that deepen understanding
• Use examples from the PCM resources provided
• Validate user experiences and insights

Available tools: {tool_names}
Use these tools to make informed decisions about the conversation flow."""

        # Créer le prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])
        
        # Créer l'agent
        self.agent = create_openai_tools_agent(llm, self.tools, self.prompt)
        self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=True)
        
    def handle_pcm_interaction(self, state: WorkflowState) -> str:
        """
        Point d'entrée principal pour les interactions PCM
        Remplace select_pcm_prompt et la logique de routage complexe
        """
        logger.info("🤖 PCM Agent: Starting interaction")
        
        # Extraire le contexte du state
        user_query = state.get('user_message', '') or (
            state.get('messages', [{}])[-1].get('content', '') if state.get('messages') else ''
        )
        
        pcm_base = state.get('pcm_base', 'Unknown')
        pcm_phase = state.get('pcm_phase', 'Unknown')  
        explored_dimensions = state.get('pcm_explored_dimensions', [])
        pcm_resources = state.get('pcm_resources', 'No PCM resources available')
        exploration_mode = state.get('exploration_mode', 'flexible')
        
        # Créer l'historique de conversation pour le contexte
        messages = state.get('messages', [])
        conversation_context = self._build_conversation_context(messages)
        
        # Préparer l'input pour l'agent
        agent_input = {
            "input": f"""USER QUERY: "{user_query}"

CONTEXT:
- PCM BASE: {pcm_base}
- PCM PHASE: {pcm_phase} 
- Explored dimensions: {explored_dimensions}
- Exploration mode: {exploration_mode}
- Language: {state.get('language', 'en')}

CONVERSATION HISTORY:
{conversation_context}

PCM RESOURCES:
{pcm_resources}

Please analyze the user's query and provide an appropriate PCM coaching response. 
Use your tools to make decisions about validation, next steps, and conversation flow."""
        }
        
        try:
            # Invoquer l'agent
            result = self.agent_executor.invoke(agent_input)
            response = result.get('output', 'I apologize, but I encountered an issue processing your PCM question.')
            
            logger.info(f"🤖 PCM Agent: Generated response ({len(response)} chars)")
            return response
            
        except Exception as e:
            logger.error(f"❌ PCM Agent error: {e}")
            return f"I apologize, but I encountered a technical issue. Please try rephrasing your question, or contact jean-pierre.aerts@zestforleaders.com for personalized PCM support."
    
    def _build_conversation_context(self, messages: List[Dict]) -> str:
        """Construire le contexte conversationnel pour l'agent"""
        if not messages:
            return "No previous conversation"
            
        context_lines = []
        for msg in messages[-4:]:  # Derniers 4 messages pour contexte
            if isinstance(msg, dict):
                role = "User" if msg.get('role') == 'user' else "Assistant"
                content = msg.get('content', '')[:200] + ('...' if len(msg.get('content', '')) > 200 else '')
                context_lines.append(f"{role}: {content}")
            elif hasattr(msg, 'type') and hasattr(msg, 'content'):
                role = "User" if msg.type == 'human' else "Assistant"  
                content = str(msg.content)[:200] + ('...' if len(str(msg.content)) > 200 else '')
                context_lines.append(f"{role}: {content}")
        
        return "\n".join(context_lines) if context_lines else "No previous conversation"

# ============================================================================
# FONCTION D'INTÉGRATION AVEC LE WORKFLOW EXISTANT
# ============================================================================

def pcm_agent_response(state: WorkflowState) -> str:
    """
    Fonction d'intégration pour tester l'agent PCM
    Peut remplacer les appels à select_pcm_prompt()
    """
    agent = PCMAgent()
    return agent.handle_pcm_interaction(state)

# ============================================================================
# MIGRATION PROGRESSIVE
# ============================================================================

def compare_agent_vs_traditional(state: WorkflowState) -> Dict[str, str]:
    """
    Fonction utilitaire pour comparer l'agent vs approche traditionnelle
    Utile pour tester et valider la migration
    """
    try:
        # Réponse de l'agent
        agent = PCMAgent()
        agent_response = agent.handle_pcm_interaction(state)
        
        # Réponse traditionnelle (si disponible)
        from .pcm_analysis import select_pcm_prompt
        traditional_response = "Traditional approach would be used here"
        
        return {
            "agent_response": agent_response,
            "traditional_response": traditional_response,
            "agent_length": len(agent_response),
            "comparison_ready": True
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "comparison_ready": False
        }