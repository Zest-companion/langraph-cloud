# PCM Module - Architecture & Flow Documentation

## Vue d'Ensemble

Le module PCM (Process Communication Model) fournit une architecture complète pour analyser les intentions utilisateur PCM, effectuer des recherches vectorielles spécialisées, et générer des réponses personnalisées selon le profil PCM de l'utilisateur.

## Architecture Générale

```
User Query → pcm_analysis_v2 → pcm_flow_manager → pcm_tools_router → handlers → generate_final_response
```

### Points d'Entrée
- **Node LangGraph :** `pcm_intent_analysis` dans `zest_companion_workflow_modular.py`
- **Fonction principale :** `pcm_analysis_with_flow_manager` dans `pcm_analysis_v2.py`

## Flow Types Supportés

### 1. SELF_BASE
- **Action :** `SEARCH_MY_BASE`
- **Description :** Questions sur ma base personnelle (forces, fonctionnement)
- **Exemples :** "Ma base PCM", "Comment je fonctionne", "Mes forces"
- **Handler :** `_handle_self_base_search`
- **Recherche :** BASE (6 dimensions)

### 2. SELF_PHASE  
- **Action :** `SEARCH_MY_PHASE`
- **Description :** Questions sur mon stress/phase actuelle (état émotionnel)
- **Exemples :** "Mon stress", "Ma phase actuelle", "I'm feeling stressed"
- **Handler :** `_handle_self_phase_search`
- **Recherche :** PHASE (psychological_needs + negative_satisfaction + distress_sequence)

### 3. SELF_ACTION_PLAN
- **Action :** `SEARCH_MY_ACTION_PLAN`
- **Description :** Demandes de conseils/stratégies pratiques
- **Exemples :** "What should I do", "Help me", "Conseils", "How can I"
- **Handler :** `_handle_self_action_plan_search`
- **Recherche :** BASE + PHASE (4 sections) + ACTION_PLAN
- **Formatage :** `_format_action_plan_results_by_sections` (3 sections)

### 4. COWORKER
- **Action :** `START_COWORKER_CONVERSATION`
- **Description :** Relations collègues/manager (processus 4 étapes)
- **Exemples :** "Mon collègue", "Conflict with", "Working with"
- **Handler :** `_handle_coworker_search`
- **Processus :** 4-step workplace relationship analysis

### 5. COMPARISON
- **Action :** `SEARCH_COMPARISON`
- **Description :** Comparaisons entre types PCM
- **Exemples :** "Différence entre X et Y", "Compare", "Thinker vs Harmonizer"
- **Handler :** `_handle_comparison_search`
- **Recherche :** Types PCM extraits de la requête

### 6. GENERAL_PCM
- **Action :** `SEARCH_GENERAL_THEORY`
- **Description :** Théorie PCM et exploration générale
- **Exemples :** "Qu'est-ce que PCM", "Les 6 bases", "Show me everything"
- **Handler :** Standard PCM flow

### 7. GREETING
- **Action :** `NO_SEARCH`
- **Description :** Salutations sans contenu substantiel
- **Exemples :** "Bonjour", "Hello", "Thanks"
- **Handler :** `_handle_greeting_search`
- **Recherche :** Aucune

### 8. TEAM
- **Action :** `SEARCH_TEAM_DYNAMICS`
- **Description :** Dynamique d'équipe
- **Exemples :** "Mon équipe", "Team dynamics", "Leadership"
- **Handler :** Standard PCM flow

## Transitions Dynamiques

### Système de Détection (`pcm_analysis_v2.py`)

#### Règle 1: PHASE → COWORKER
```python
# Condition: utilisateur en PHASE + mention collègue
if previous_flow in ['self_phase', 'SELF_PHASE']:
    colleague_keywords = ['colleague', 'coworker', 'manager', 'boss', 'collègue', 'chef', 'équipe']
    if any(keyword in user_message for keyword in colleague_keywords):
        # Transition vers COWORKER
```

#### Règle 2: PHASE → ACTION_PLAN  
```python
# Condition: utilisateur en PHASE + demande conseils
if previous_flow in ['self_phase', 'SELF_PHASE']:
    action_keywords = ['what should i do', 'recommendations', 'conseils', 'aide-moi', 'help me']
    if any(keyword in user_message for keyword in action_keywords):
        # Transition vers SELF_ACTION_PLAN
```

### Règles de Continuité
- **Mots de continuation :** "yes", "oui", "ok", "continue", "next" → garder flow précédent
- **Références vagues :** "all", "tous", "the ones you mentioned" → continuer même contexte

## Architecture de Recherche

### Handlers Spécialisés (`pcm_vector_search.py`)

#### `_handle_self_action_plan_search`
**Le plus complet - 3 sections :**
1. **BASE Foundation :** `pcm_base_type` (3 résultats)
2. **PHASE Context :** `pcm_phase_type` + 4 section_types :
   - `psychological_needs`
   - `negative_satisfaction` 
   - `distress_sequence`
   - `action_plan`
3. **Formatage :** `_format_action_plan_results_by_sections` (sans sanitize)

#### `_handle_self_base_search` 
- Recherche BASE (6 dimensions)
- Exploration systématique ou flexible selon le mode

#### `_handle_self_phase_search`
- Recherche PHASE (3 sections : psychological_needs + negative_satisfaction + distress_sequence)

#### `_handle_coworker_search`
- Processus 4 étapes avec gestion des sous-étapes
- Prompts différents selon l'étape (regular vs action_plan)

## Tools et Routing (`pcm_tools.py`)

### Router Principal : `pcm_tools_router`
```python
# Mapping flow_type → tool
if flow_type in ['self_base', 'self_phase']: 
    return "execute_pcm_self_tool"
elif flow_type == 'self_action_plan':
    return "execute_pcm_action_plan_tool"  # ← Utilise _handle_self_action_plan_search
elif flow_type == 'coworker_focused':
    return "execute_pcm_coworker_tool"
# etc.
```

## Prompts Système (`prompt_builder.py`)

### Sélection dans `generate_final_response`
```python
# create_prompt_by_subtheme pour A2_PersonalityPCM
elif flow_type in ['SELF_ACTION_PLAN', 'self_action_plan']:
    return build_pcm_self_focused_action_plan_prompt(state)
elif flow_type in ['self_focused', 'self_base', 'self_phase']:
    return select_pcm_prompt(state)  # Système conversationnel
elif flow_type == 'coworker_focused':
    return build_pcm_coworker_focused_prompt(state)
```

### Prompts Disponibles
1. `build_pcm_self_focused_action_plan_prompt` - **Action plans personnalisés**
2. `build_pcm_self_focused_base_prompt` - BASE exploration  
3. `build_pcm_self_focused_phase_prompt` - PHASE analysis
4. `build_pcm_coworker_focused_prompt` - Relations workplace
5. `build_pcm_coworker_focused_action_plan_prompt` - Action plans coworker
6. `build_pcm_comparison_prompt` - Comparaisons types
7. `build_pcm_general_knowledge_prompt` - Théorie générale
8. `build_pcm_greeting_prompt` - Salutations

## Flow Complet SELF_ACTION_PLAN (Exemple)

```mermaid
User: "I'm stressed when presenting, what should I do?"
    ↓
1. pcm_analysis_v2 détecte transition PHASE → ACTION_PLAN
    ↓  
2. pcm_flow_manager classifie: flow_type="SELF_ACTION_PLAN"
    ↓
3. pcm_tools_router → execute_pcm_action_plan_tool
    ↓
4. _handle_self_action_plan_search fait 3 recherches:
   - BASE harmonizer (3 résultats)
   - PHASE harmonizer (4 sections: psychological_needs, negative_satisfaction, distress_sequence, action_plan)
    ↓
5. _format_action_plan_results_by_sections structure en 3 sections
    ↓
6. generate_final_response utilise build_pcm_self_focused_action_plan_prompt
    ↓
7. Réponse personnalisée avec stratégies concrètes
```

## Cas Particuliers

### 1. Système Conversationnel
- **Contexte :** Continuité des conversations BASE/PHASE
- **Module :** `pcm_conversational_analysis.py`  
- **Fonction :** `analyze_pcm_conversational_intent`

### 2. Coworker 4-Step Process
- **Step 1 :** Analyse relationnelle initiale
- **Step 2 :** Action plan (2 sous-étapes)
- **Step 3-4 :** Suivi et ajustements

### 3. Exploration Dimensionnelle  
- **Mode systematic :** Exploration ordonnée des 6 dimensions
- **Mode flexible :** Exploration libre selon la demande

### 4. Fallbacks et Compatibilité
- **Legacy system :** `_execute_pcm_search` comme fallback
- **Error handling :** Fallback vers analyse standard si classification échoue

### 5. Classification Avancée
- **LLM-based :** Prompt de classification avec exemples et règles
- **Context-aware :** Tient compte de l'historique conversationnel
- **Transition detection :** Détection automatique des changements d'intention

## État et Transmission

### WorkflowState Critical Fields
```python
pcm_classification: Optional[Dict]  # Classification du flow manager
flow_type: Optional[str]           # Type de flow détecté
pcm_base: Optional[str]            # Base utilisateur
pcm_phase: Optional[str]           # Phase utilisateur
```

### Transmission Entre Nodes
- **pcm_analysis_v2** → classification + routing
- **pcm_tools** → préservation pcm_classification
- **pcm_vector_search** → utilisation classification pour handlers
- **generate_final_response** → prompt système adapté

Cette architecture offre une expérience PCM complète avec classification intelligente, recherches spécialisées, transitions dynamiques, et génération de réponses personnalisées selon le profil et l'intention de l'utilisateur.