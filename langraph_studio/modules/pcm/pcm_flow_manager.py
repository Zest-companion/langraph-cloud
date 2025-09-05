"""
PCM Flow Manager - Équivalent du système MBTI pour tous les flux PCM
Gère la classification intelligente et le routing vers les bonnes recherches/conversations
"""
import json
import logging
from typing import Dict, List, Optional, Any
from ..common.types import WorkflowState
from ..common.llm_utils import isolated_analysis_call_with_messages

logger = logging.getLogger(__name__)

class PCMFlowManager:
    """Gestionnaire central pour tous les flux PCM (inspiré du système MBTI)"""
    
    # Types de flux PCM (équivalent des classifications MBTI)
    FLOW_TYPES = {
        'SELF_BASE': 'Questions sur ma base PCM personnelle',
        'SELF_PHASE': 'Questions sur ma phase/stress actuel',
        'SELF_ACTION_PLAN': 'Demandes de conseils et plan d\'action',
        'COWORKER': 'Relations avec collègues/coworkers', 
        'TEAM': 'Dynamique d\'équipe/groupe',
        'COMPARISON': 'Comparaisons entre types PCM',
        'GENERAL_PCM': 'Théorie et concepts PCM',
        'GREETING': 'Salutations et remerciements'
    }
    
    # Actions correspondantes (équivalent des CALL_AB, CALL_ABC, etc.)
    FLOW_ACTIONS = {
        'SELF_BASE': 'SEARCH_MY_BASE',
        'SELF_PHASE': 'SEARCH_MY_PHASE',
        'SELF_ACTION_PLAN': 'SEARCH_MY_ACTION_PLAN',
        'COWORKER': 'START_COWORKER_CONVERSATION',
        'TEAM': 'SEARCH_TEAM_DYNAMICS',
        'COMPARISON': 'SEARCH_COMPARISON',
        'GENERAL_PCM': 'SEARCH_GENERAL_THEORY',
        'GREETING': 'NO_SEARCH'
    }
    
    @staticmethod
    def classify_pcm_intent(state: WorkflowState) -> Dict[str, Any]:
        """
        Analyse et classifie l'intention PCM de l'utilisateur
        Retourne le flow type et les actions à entreprendre
        """
        logger.info("🎯 PCM Flow Manager - Classifying user intent")
        
        # Construire le prompt d'analyse (similaire à MBTI)
        analysis_prompt = PCMFlowManager._build_classification_prompt()
        
        try:
            # Appel LLM pour classification
            messages = state.get('messages', [])
            user_message = state.get('user_message', '')
            
            # Contexte utilisateur
            user_context = {
                'user_pcm_base': state.get('pcm_base', ''),
                'user_pcm_phase': state.get('pcm_phase', ''),
                'conversation_length': len(messages),
                'previous_flow': state.get('flow_type')
            }
            
            # Appel LLM isolé avec continuité contextuelle
            llm_messages = [
                {"role": "system", "content": analysis_prompt},
                {"role": "user", "content": f"""
CONTEXTE UTILISATEUR:
- Base PCM: {user_context['user_pcm_base']}
- Phase PCM: {user_context['user_pcm_phase']}
- Messages conversation: {user_context['conversation_length']}
- Flow précédent: {user_context['previous_flow']}

MESSAGE UTILISATEUR: "{user_message}"

HISTORIQUE RÉCENT (5 derniers messages):
{PCMFlowManager._format_recent_messages(messages[-5:])}

RÈGLES DE CONTINUITÉ CONTEXTUELLE:
- Si previous_flow = 'safety_refusal' ET message contient des références au même sujet → **CONTINUER LE REFUS**
- Si previous_flow = 'self_base' ET message contient des références vagues ("all", "tous", "the ones you mentioned", "ceux que tu as dit") → continuer en SELF_BASE
- Si previous_flow = 'self_phase' ET références vagues par raaport à la phase/stress actuel → continuer en SELF_PHASE  
- Si previous_flow = 'comparison' ET (références vagues OU types PCM mentionnés) → continuer en COMPARISON
- Si previous_flow = 'coworker_focused' → **TOUJOURS CONTINUER EN COWORKER** sauf si l'utilisateur change explicitement de sujet (base/phase/théorie/comparaisons/général/...)

**RÈGLE PRIORITAIRE COMPARISON**: Si previous_flow = 'comparison' ET message contient des types PCM (harmonizer, promoteur, empathique, etc.) → TOUJOURS COMPARISON (PAS COWORKER)

**RÈGLE ABSOLUE COWORKER**: Une fois dans coworker_focused, TOUS les messages suivants restent dans COWORKER, incluant:
- **RÉPONSES ÉMOTIONNELLES**: "I feel anxious", "I'm stressed", "I don't feel ok", "It affects me", "I'm not comfortable", "Yes I feel...", "No I don't feel..."
- **CONFIRMATIONS/NÉGATIONS**: "yes", "no", "oui", "non", "exactly", "that's right", "not really"
- **DESCRIPTIONS D'ÉTAT**: "anxious", "stressed", "worried", "uncomfortable", "not ok", "affected"
- **EXPLICATIONS PERSONNELLES**: Toute phrase commençant par "I feel", "I am", "I'm", "It makes me", "It affects me"
- **LETTRES DE CHOIX**: "A", "B", "C", "D", "E", "F" (réponses aux questions du flow)
- **CONTINUATIONS**: "continue", "go on", "tell me more", "next", "what else"

**SEULES EXCEPTIONS pour sortir de coworker_focused**:
- L'utilisateur pose une question TOTALEMENT différente (ex: "What is PCM?", "Tell me about my base", "quelle est ma phase actuelle", ...)
- L'utilisateur dit EXPLICITEMENT: "parlons d'autre chose", "j'ai une autre question", "changeons de sujet", "let's talk about something else"

**PRINCIPE CRITIQUE**: Dans le doute sur une réponse émotionnelle ou personnelle après une question coworker → TOUJOURS rester en COWORKER

Analyse et classifie cette intention PCM selon les règles définies, en priorité la continuité contextuelle.
"""}
            ]
            
            # Extraire system et user content des messages
            system_content = llm_messages[0]["content"]
            user_content = llm_messages[1]["content"]
            
            response = isolated_analysis_call_with_messages(system_content, user_content)
            
            # Nettoyage et validation de la réponse JSON
            response = response.strip()
            
            # Retirer les markdown code blocks si présents
            if '```json' in response:
                start = response.find('```json') + 7
                end = response.find('```', start)
                if end != -1:
                    response = response[start:end].strip()
            elif '```' in response:
                start = response.find('```') + 3
                end = response.find('```', start)
                if end != -1:
                    response = response[start:end].strip()
            
            # Vérification basique du format JSON
            if not response.startswith('{') or not response.endswith('}'):
                logger.warning(f"⚠️ Invalid JSON format from LLM: {response[:100]}...")
                raise ValueError("Invalid JSON format")
            
            result = json.loads(response)
            
            # Validation des champs requis
            required_fields = ['flow_type', 'action']
            for field in required_fields:
                if field not in result or result[field] is None:
                    logger.warning(f"⚠️ Missing or null required field '{field}' in LLM response: {result}")
                    raise ValueError(f"Missing or null required field: {field}")
            
            logger.info(f"✅ PCM Intent classified: {result.get('flow_type')} (confidence: {result.get('confidence', 0.8)})")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error in PCM classification: {e}")
            # Fallback vers classification simple
            return PCMFlowManager._simple_fallback_classification(state)
    
    @staticmethod
    def _build_classification_prompt() -> str:
        """Construit le prompt de classification PCM (inspiré MBTI)"""
        return """Tu es un expert PCM (Process Communication Model) qui analyse les intentions utilisateur.


## CONTEXTE PCM
Le PCM identifie 6 BASES de personnalité:
- THINKER: Logique, facts, analyse
- HARMONIZER: Relations, empathie, émotions
- PERSISTER: Valeurs, convictions, opinions
- REBEL: Créativité, spontanéité, fun
- PROMOTER: Action, résultats, efficacité  
- IMAGINER: Calme, réflexion, imagination

Chaque personne a aussi des PHASES de stress possibles.

## CLASSIFICATIONS PCM

### 1. SELF_BASE → Questions sur ma base personnelle - ma base qui ne change pas comprenant (forces, style d'interaction, perceptions, canal de communication, environnement préféré)
- "Ma base PCM", "Comment je fonctionne", "Mes forces"
- "Tell me about my personality", "What am I like"
- Focus sur SOI sans mention d'autres types
- → Action: SEARCH_MY_BASE

### 2. SELF_PHASE → Questions sur mon mes besoins psychologuques et réactions sous stress/phase (ÉTAT ACTUEL SEULEMENT)
- "Mon stress", "Ma phase actuelle" (sans demande d'aide)
- "I'm feeling stressed", "My current state", "How am I doing"
- Indicateurs: stress, feeling, emotional needs, current, lately (UNIQUEMENT si pas de demande d'action)
- → Action: SEARCH_MY_PHASE

### 3. SELF_ACTION_PLAN → Demandes de conseils/actions PERSONNELLES (par rapport à mes besoins psychologuques et réactions sous stress/phase) ou une SITUATIONNELLE n'impliquant pas une personne spécifique
- **RÈGLE CRITIQUE**: SELF_ACTION_PLAN = actions sur SOI-MÊME uniquement pouvant impliquer des metions à d'autre personnes MAIS PAS IDENTIDIABLES (e.g. my colleagues)
- "What can I do about MY stress", "How can I improve MYSELF", "Help me change MY behavior"
- "How can I manage MY emotions", "What should I do about MY phase"
- "Recommendations for ME", "Help me with MY development"
- "how to handle my stress in front of my colleagues", "I feel anxious at the office" - ... 
- **EXCLUSION IMPORTANTE**: Si mention d'autres personnes (colleague, manager, coworker) → JAMAIS SELF_ACTION_PLAN
- **Si stress/phase + demande d'action SUR SOI** → SELF_ACTION_PLAN
- **Si stress/phase + demande d'action SUR SOI + mention d'und personne spécifique** → COWORKER
- → Action: SEARCH_MY_ACTION_PLAN

### 4. COWORKER → Relations avec UNE PERSONNE SPÉCIFIQUE
- **RÈGLE CRITIQUE**: COWORKER = UNE PERSONNE SPÉCIFIQUE et IDENTIFIÉE (e.g. my manager, my colleague, my boss, someone specific at work)
- **RELATIONS SPÉCIFIQUES**: "My manager [nom]", "My colleague [description spécifique]", "My boss", "Someone specific at work"
- **SITUATIONS RELATIONNELLES PRÉCISES**: "Conflict with my manager", "My boss micromanages me", "Working with John who..."
- **EXEMPLES COWORKER**: "My manager puts pressure", "Colleague X is difficult", "Boss Y is demanding"

**EXCLUSIONS IMPORTANTES - CES CAS RESTENT EN SELF:**
- **SENTIMENTS GÉNÉRAUX**: "Je ne suis pas à l'aise avec mes collègues" → SELF_PHASE
- **GROUPES VAGUES**: "I don't like speaking in public" → SELF_ACTION_PLAN  
- **STRESS GÉNÉRAL**: "I am stressed at the office" (sans personne spécifique) → SELF_PHASE
- **ÉQUIPE ENTIÈRE**: "Mon équipe me stresse" → SELF_PHASE
- **SITUATIONS PUBLIQUES**: "Je n'ose pas parler en public" → SELF_ACTION_PLAN
- **TYPES PCM SEULS**: "empathique", "promoteur", "harmonizer", "thinker" sans contexte relationnel → COMPARISON ou SELF

**RÈGLE DE VALIDATION**: Pour être COWORKER, l'utilisateur doit pouvoir répondre à "Qui précisément ?" avec un nom, un rôle spécifique, ou une description claire d'UNE personne.

- → Action: START_COWORKER_CONVERSATION

### 5. COMPARISON → Comparaisons DIRECTES entre TYPES PCM
- **UNIQUEMENT**: Comparaisons directes entre 2+ types de personnalité PCM
- "Différence entre thinker et harmonizer", "Compare promoter and rebel"
- "Thinker vs Harmonizer", "What's the difference between empathique et promoteur"
- "How is a thinker different from a persister", "Compare ces deux types"
- Mots-clés: "difference", "different", "compare", "versus", "vs", "entre"
- IMPORTANT: Extraire les types PCM mentionnés dans "extracted_pcm_types"

⚠️ **CE QUI N'EST PAS COMPARISON** (= GENERAL_PCM à la place):
- Questions de CLASSIFICATION THÉORIQUE : "X c'est dans base ou phase ?", "Communication channel belongs to base or phase ?"
- Comparaisons de CONCEPTS (pas de types) : "base vs phase", "strengths vs needs"  
- Définitions et explications : "How does communication channel work ?"

⚠️ **DISTINCTION CRITIQUE POUR LE LLM** :
- "Communication channel c'est base ou phase ?" = GENERAL_PCM (classification structurelle, pas comparaison de types)
- "Thinker vs Harmonizer communication" = COMPARISON (comparer comment 2 types utilisent communication)
- "base vs phase" = GENERAL_PCM (explication des concepts fondamentaux)
- "thinker vs harmonizer" = COMPARISON (comparer 2 types de personnalité)

- → Action: SEARCH_COMPARISON

### 6. GENERAL_PCM → Théorie PCM et exploration générale
- **TOUT ASPECT THEORIQUE DU MODELE PCM sauf comparaisons directes entre types**:
- "Qu'est-ce que PCM", "What is PCM", "PCM theory"
- "How does PCM work", "Explain PCM", "Explain communication channels"
- "Les 6 bases", "All PCM types", "Show me everything", "Explore", "Discover"
- Questions de CLASSIFICATION STRUCTURELLE : "Communication channel c'est dans base ou phase ?", "X belongs to base or phase ?"
- Concepts PCM : base, phase, channels, strengths, needs, stress, etc. et leurs définitions, dimensions
- Structure du modèle : où appartiennent les différents éléments PCM
- → Action: SEARCH_GENERAL_THEORY

### 7. GREETING → Salutations et small talk
- "Bonjour", "Merci", "Hello", "Thanks", "Hi", "Good morning"
- "Comment allez-vous", "How are you", "What's up"
- Pas de contenu PCM substantiel, juste interaction sociale
- → Action: NO_SEARCH

## RÈGLES DE PRIORITÉ (ORDRE STRICT)
0. **RÈGLE SUPRÊME**: Si previous_flow = 'coworker_focused' ET le message est une réponse émotionnelle/personnelle → TOUJOURS COWORKER
   - EXEMPLE: Après "How do you feel about your manager?", "Yes I feel anxious" = COWORKER (PAS SELF_PHASE)
   - EXEMPLE: Après question coworker, "I don't feel ok" = COWORKER (PAS SELF_PHASE)
0.5. **RÈGLE CONTINUITÉ COMPARISON**: Si previous_flow = 'comparison' ET message contient des types PCM → RESTER EN COMPARISON
   - EXEMPLE: Après "quelle est la différence entre une base harmonizer et promoteur", "empathique et promoteur" = COMPARISON (PAS COWORKER)
   - EXEMPLE: Réponses de clarification avec types PCM = COMPARISON 
   - Cette règle a PRIORITÉ sur la détection COWORKER
0.6. **RÈGLE CONTINUITÉ SELF**: Si previous_flow = 'self_base'/'self_phase'/'self_action_plan' ET message exprime doute/désaccord → RESTER EN SELF
   - EXEMPLE: Après exploration BASE, "En fait je ne me reconnais pas" = SELF_BASE (PAS COWORKER)
   - EXEMPLE: Après exploration PHASE, "This doesn't sound like me" = SELF_PHASE (PAS COWORKER) 
   - EXEMPLE: Expressions de doute: "je ne me reconnais", "doesn't fit", "not sure", "seems wrong"
1. **RÈGLE ABSOLUE #1**: Si UNE PERSONNE SPÉCIFIQUE mentionnée → COWORKER
   - **DOIT ÊTRE SPÉCIFIQUE**: "My manager", "My boss", "My colleague Marc", "This person", "Someone specific"
   - **EXCLUSIONS - RESTENT EN SELF**: "colleagues" (pluriel vague), "team members", "people at work", "everyone"
   - EXEMPLE: "How can I manage my manager" = COWORKER (personne spécifique)
   - EXEMPLE: "My colleagues stress me" = SELF_PHASE (groupe vague)
   - EXEMPLE: "I'm not comfortable with colleagues" = SELF_PHASE (sentiment général)
2. Comparaisons explicites → COMPARISON  
3. **DEMANDES D'ACTION PERSONNELLES** (action sur soi uniquement, AUCUNE autre personne mentionnée) → SELF_ACTION_PLAN
4. **RÈGLE CRITIQUE**: "stress/phase + action SUR SOI SEUL" → SELF_ACTION_PLAN
5. **RÈGLE ABSOLUE #2**: "stress/phase + UNE PERSONNE SPÉCIFIQUE" → COWORKER (jamais SELF_ACTION_PLAN)
6. Indicateurs stress personnels PURS (sans action, sans autres personnes) → SELF_PHASE
7. Questions personnelles pures → SELF_BASE
8. Théorie/concepts/exploration générale → GENERAL_PCM

**EXEMPLES CRITIQUES:**
- "I want to manage my stress because of my manager" → COWORKER (manager = personne spécifique)
- "How can I handle my manager's pressure" → COWORKER (manager = personne spécifique)
- "I want to manage my stress" → SELF_ACTION_PLAN (aucune personne mentionnée)
- "My colleagues stress me" → SELF_PHASE (groupe vague, pas personne spécifique)
- "Je ne suis pas à l'aise avec mes collègues" → SELF_PHASE (sentiment général)
- "Je n'ose pas parler en public" → SELF_ACTION_PLAN (situation publique, pas personne spécifique)
- "I don't like speaking in meetings" → SELF_ACTION_PLAN (situation générale)
- Après "How do you feel about your manager?", "Yes I feel anxious" → COWORKER (continuation du contexte)
- Après "Tell me about the situation", "I don't feel ok because..." → COWORKER (continuation)
- Après question coworker, "It affects me emotionally" → COWORKER (réponse dans le contexte)
- Sans contexte coworker, "I feel anxious" → SELF_PHASE (pas de contexte coworker établi)

## DÉTECTION DE LANGUE - OBLIGATOIRE
- Détecte la langue du message utilisateur
- Si le message est en français → "language": "fr"
- Si le message est en anglais → "language": "en"
- Si autre langue → "language": "en" (défaut anglais)

## FORMAT DE RÉPONSE
Retourne UNIQUEMENT un JSON valide, sans texte avant ou après.

**IMPORTANT**: Le champ "action" ne doit JAMAIS être null - utilisez toujours une valeur string valide.

EXEMPLE pour "What is my PCM personality structure?":
```json
{
  "flow_type": "SELF_BASE",
  "action": "SEARCH_MY_BASE",
  "confidence": 0.9,
  "reasoning": "L'utilisateur demande des informations sur sa structure de personnalité PCM personnelle",
  "extracted_pcm_types": null,
  "conversation_stage": "initial",
  "language": "fr"
}
```

EXEMPLE pour "How do I work with my colleague who is stressed?":
```json
{
  "flow_type": "COWORKER",
  "action": "START_COWORKER_CONVERSATION",
  "confidence": 0.8,
  "reasoning": "L'utilisateur demande des conseils sur les relations avec un collègue en stress",
  "extracted_pcm_types": null,
  "conversation_stage": "initial", 
  "language": "en"
}
```

EXEMPLE pour "what is the difference with a promoter?":
```json
{
  "flow_type": "COMPARISON",
  "action": "SEARCH_COMPARISON",
  "confidence": 0.9,
  "reasoning": "L'utilisateur demande la différence entre sa base et le type Promoter",
  "extracted_pcm_types": ["promoter"],
  "conversation_stage": "comparison",
  "language": "fr"
}
```

EXEMPLE pour "Hello":
```json
{
  "flow_type": "GREETING",
  "action": "NO_SEARCH",
  "confidence": 0.9,
  "reasoning": "Simple salutation sans contenu PCM",
  "extracted_pcm_types": null,
  "conversation_stage": "greeting",
  "language": "en"
}
```

IMPORTANT: Retourne SEULEMENT le JSON, pas de texte explicatif."""
    
    @staticmethod
    def _format_recent_messages(messages: List) -> str:
        """Formate les messages récents pour le contexte"""
        if not messages:
            return "Aucun historique"
        
        formatted = []
        for msg in messages:  # Prend tous les messages passés (limité par l'appelant)
            if hasattr(msg, 'content'):
                content = msg.content[:150]  # Plus de contexte
                formatted.append(f"- {content}")
            elif isinstance(msg, dict):
                content = msg.get('content', str(msg))[:150]
                formatted.append(f"- {content}")
        
        return "\n".join(formatted)
    
    @staticmethod
    def _simple_fallback_classification(state: WorkflowState) -> Dict[str, Any]:
        """Classification simple en cas d'erreur LLM avec continuité contextuelle"""
        user_message = state.get('user_message', '').lower()
        previous_flow = state.get('flow_type', '')
        
        # PRIORITÉ ABSOLUE: Continuité des refus de sécurité
        if previous_flow == 'safety_refusal':
            # Si on était en refus ET que le message semble continuer le même sujet → CONTINUER LE REFUS
            logger.warning(f"🚫 Previous flow was safety_refusal - continuing refusal for: '{user_message[:50]}...'")
            return {
                "flow_type": "SAFETY_REFUSAL",
                "action": "REFUSE_NON_WORKPLACE", 
                "confidence": 1.0,
                "reasoning": "Continuité du sujet précédemment refusé",
                "extracted_pcm_types": None,
                "conversation_stage": "refused"
            }
        
        # PRIORITÉ 0: Continuité COMPARISON avec types PCM
        pcm_types = ['harmonizer', 'promoteur', 'empathique', 'thinker', 'persister', 'rebel', 'imaginer', 'promoter']
        if previous_flow == 'comparison' and any(pcm_type in user_message for pcm_type in pcm_types):
            logger.info(f"🔄 Fallback: PCM TYPES in comparison context - FORCING continuation as COMPARISON")
            flow_type = 'COMPARISON'
            action = 'SEARCH_COMPARISON'
        # PRIORITÉ 1: Continuité contextuelle FORTE pour coworker
        # Mots émotionnels qui dans un contexte coworker indiquent une continuation
        elif previous_flow == 'coworker_focused':
            emotional_responses = ['feel', 'anxious', 'stressed', 'worried', 'uncomfortable', 'not ok', 
                                 'affects me', 'makes me', 'i am', "i'm", 'nervous', 'afraid', 'scared']
            
            # Si on vient de coworker ET c'est une réponse émotionnelle → CONTINUER EN COWORKER
            if any(emotion in user_message for emotion in emotional_responses):
                logger.info(f"🔄 Fallback: EMOTIONAL RESPONSE in coworker context - FORCING continuation as COWORKER")
                flow_type = 'COWORKER'
                action = 'START_COWORKER_CONVERSATION'
            else:
                # Même si pas émotionnel, rester en coworker si on était dans ce contexte
                logger.info(f"🔄 Fallback: Continuing COWORKER context")
                flow_type = 'COWORKER'
                action = 'START_COWORKER_CONVERSATION'
        # Continuité générale pour références vagues
        elif any(word in user_message for word in ['all', 'tous', 'toutes', 'the ones you mentioned', 'ceux que tu as dit', 
                            'those', 'ces', 'them', 'les', 'everything', 'tout',
                            'yes', 'oui', 'ok', 'sure', 'continue', 'next', 'suivant', 'go on']) and previous_flow:
            logger.info(f"🔄 Fallback: Continuing previous flow '{previous_flow}' due to contextual reference")
            if previous_flow in ['self_base', 'self_focused']:
                flow_type = 'SELF_BASE'
                action = 'SEARCH_MY_BASE'
            elif previous_flow == 'self_phase':
                flow_type = 'SELF_PHASE'
                action = 'SEARCH_MY_PHASE'
            elif previous_flow in ['comparison', 'compare']:
                flow_type = 'COMPARISON'
                action = 'SEARCH_COMPARISON'
            elif previous_flow == 'coworker_focused':
                flow_type = 'COWORKER'
                action = 'START_COWORKER_CONVERSATION'
            else:
                flow_type = 'SELF_BASE'  # Défaut sécurisé
                action = 'SEARCH_MY_BASE'
        # PRIORITÉ 1.5: Si on était dans un flow SELF et qu'on n'est pas sûr → RESTER DANS LE MÊME FLOW  
        elif previous_flow in ['self_base', 'self_focused', 'self_phase', 'self_action_plan']:
            logger.info(f"🔄 Fallback: Uncertain intent but previous was SELF flow '{previous_flow}' - STAYING IN SELF")
            if previous_flow in ['self_base', 'self_focused']:
                flow_type = 'SELF_BASE'
                action = 'SEARCH_MY_BASE'
            elif previous_flow == 'self_phase':
                flow_type = 'SELF_PHASE'  
                action = 'SEARCH_MY_PHASE'
            elif previous_flow == 'self_action_plan':
                flow_type = 'SELF_ACTION_PLAN'
                action = 'SEARCH_MY_ACTION_PLAN'
        # PRIORITÉ 2: Détection simple par mots-clés (ordre d'importance)
        # IMPORTANT: Détecter les greetings EN PREMIER (mots entiers seulement)
        elif any(f' {word} ' in f' {user_message} ' for word in ['hello', 'hi', 'hey', 'bonjour', 'salut', 
                                                               'gello', 'helo', 'hallo', 'allo',  # typos courants
                                                               'good morning', 'good afternoon', 'good evening',
                                                               'thank', 'thanks', 'merci', 'bye', 'goodbye']):
            flow_type = 'GREETING'
            action = 'NO_SEARCH'
        elif any(word in user_message for word in ['difference', 'different', 'compare', 'comparison', 'versus', 'vs', 'différence', 'comparer']):
            flow_type = 'COMPARISON'
            action = 'SEARCH_COMPARISON'
        elif any(phrase in user_message for phrase in ['my manager', 'my boss', 'my colleague', 'mon manager', 'mon chef', 'mon collègue', 'ma collègue']):
            # Personne spécifique → COWORKER
            flow_type = 'COWORKER'
            action = 'START_COWORKER_CONVERSATION'
        elif any(word in user_message for word in ['colleagues', 'collègues', 'team members', 'coworkers']):
            # Groupes vagues → Rester en SELF
            flow_type = 'SELF_PHASE'
            action = 'SEARCH_MY_PHASE'
        # PRIORITÉ: Demandes d'action AVANT indicateurs de stress
        elif any(phrase in user_message for phrase in ['how can i', 'how do i', 'what can i do', 'help me', 'manage', 'handle', 'deal with']):
            flow_type = 'SELF_ACTION_PLAN'
            action = 'SEARCH_MY_ACTION_PLAN'
        elif any(word in user_message for word in ['stress', 'phase', 'feeling', 'current']):
            flow_type = 'SELF_PHASE' 
            action = 'SEARCH_MY_PHASE'
        elif any(word in user_message for word in ['me', 'my', 'myself', 'moi', 'mon', 'ma']):
            flow_type = 'SELF_BASE'
            action = 'SEARCH_MY_BASE'
        else:
            flow_type = 'GENERAL_PCM'
            action = 'SEARCH_GENERAL_THEORY'
        
        return {
            "flow_type": flow_type,
            "action": action,
            "confidence": 0.6,
            "reasoning": "Fallback classification par mots-clés",
            "extracted_pcm_types": None,
            "conversation_stage": "initial"
        }
    
    @staticmethod
    def execute_flow_action(classification: Dict[str, Any], state: WorkflowState) -> Dict[str, Any]:
        """
        Exécute l'action déterminée par la classification en utilisant les outils PCM
        Retourne l'état mis à jour avec la configuration pour les outils
        """
        flow_type = classification.get('flow_type')
        action = classification.get('action')
        
        logger.info(f"🚀 Executing PCM flow: {flow_type} → {action}")
        
        # Configurer l'état pour le routing des outils PCM
        updated_state = {
            **state,
            'pcm_classification': classification,
            'pcm_flow_manager_used': True
        }
        
        # Router vers les outils PCM selon l'action ou flow_type
        # NOTE: Sécurité (safety_refusal) est maintenant gérée en amont dans pcm_analysis_v2.py
        
        # 🎯 PRIORITÉ: COWORKER continuation overrides any other classification
        if flow_type == 'COWORKER':
            updated_state.update({
                'flow_type': 'coworker_focused',
                'search_focus': 'coworker',
                'pcm_tool_target': 'execute_pcm_coworker_tool'
            })
        elif flow_type == 'GREETING':
            # Traitement direct des greetings sans action spécifique
            updated_state.update({
                'flow_type': 'greeting',
                'skip_search': True,
                'pcm_tool_target': 'execute_pcm_no_search'
            })
        elif action == 'START_COWORKER_CONVERSATION':
            updated_state.update({
                'flow_type': 'coworker_focused',
                'search_focus': 'coworker',
                'pcm_tool_target': 'execute_pcm_coworker_tool'
            })
        elif action == 'SEARCH_MY_BASE':
            updated_state.update({
                'flow_type': 'self_base',
                'search_focus': 'user_base',
                'pcm_base_or_phase': 'base',
                'pcm_tool_target': 'execute_pcm_self_tool'
            })
        elif action == 'SEARCH_MY_PHASE':
            updated_state.update({
                'flow_type': 'self_phase',
                'search_focus': 'user_phase',
                'pcm_base_or_phase': 'phase',
                'pcm_tool_target': 'execute_pcm_self_tool'
            })
        elif action == 'SEARCH_MY_ACTION_PLAN':
            updated_state.update({
                'flow_type': 'self_action_plan',
                'search_focus': 'action_plan',
                'pcm_tool_target': 'execute_pcm_action_plan_tool'
            })
        elif action == 'SEARCH_COMPARISON':
            extracted_types = classification.get('extracted_pcm_types', [])
            updated_state.update({
                'flow_type': 'comparison',
                'search_focus': 'comparison',
                'pcm_types_to_compare': extracted_types,
                'pcm_tool_target': 'execute_pcm_comparison_tool'
            })
        elif action == 'SEARCH_ALL_BASES':
            updated_state.update({
                'flow_type': 'exploration',
                'search_focus': 'all_bases',
                'pcm_tool_target': 'execute_pcm_exploration_tool'
            })
        elif action == 'SEARCH_GENERAL_THEORY':
            updated_state.update({
                'flow_type': 'general_pcm',
                'search_focus': 'theory',
                'pcm_tool_target': 'execute_pcm_general_tool'
            })
        elif action == 'NO_SEARCH':
            updated_state.update({
                'flow_type': 'greeting',
                'skip_search': True,
                'pcm_tool_target': 'execute_pcm_no_search'
            })
        elif action == 'REFUSE_NON_WORKPLACE':
            updated_state.update({
                'flow_type': 'safety_refusal',
                'skip_search': True,
                'pcm_tool_target': 'execute_pcm_no_search'  # Utilise no_search pour générer refus
            })
        else:
            logger.warning(f"⚠️ Unknown action: {action}, defaulting to general")
            updated_state.update({
                'flow_type': 'general_pcm',
                'search_focus': 'theory',
                'pcm_tool_target': 'execute_pcm_general_tool'
            })
        
        logger.info(f"✅ PCM Flow configured: {flow_type} → tool: {updated_state.get('pcm_tool_target')}")
        return updated_state
    
