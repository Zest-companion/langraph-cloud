# Zest Companion - Structure Modulaire

Ce dossier contient la restructuration modulaire du fichier `zest_companion_workflow.py` original (conservé intact).

## Structure

```
modules/
├── common/                 # Configuration et utilitaires partagés
│   ├── config.py          # Configuration LLM, Supabase, logging
│   ├── types.py           # Types de données (WorkflowState)
│   ├── llm_utils.py       # Utilitaires pour appels LLM isolés
│   └── __init__.py
├── mbti/                  # Analyse MBTI
│   ├── mbti_analysis.py   # Fonctions d'analyse MBTI expert
│   └── __init__.py
├── lencioni/              # Analyse Lencioni Five Dysfunctions
│   ├── lencioni_analysis.py # Fonctions d'analyse des dysfonctions
│   └── __init__.py
├── prompts/               # Construction des prompts
│   ├── prompt_builder.py  # Factory de prompts par sous-thème
│   └── __init__.py
├── response/              # Génération des réponses
│   ├── response_generator.py # Générateur unifié de réponses finales
│   └── __init__.py
└── README.md              # Cette documentation
```

## Modules

### 1. Common
Configuration partagée et utilitaires de base :
- **config.py** : Configuration LLM (gpt-4, gpt-3.5-turbo), Supabase, logging
- **types.py** : Définition du type `WorkflowState` avec tous les champs
- **llm_utils.py** : Appels LLM isolés pour éviter les conflits de streaming

### 2. MBTI
Fonctions d'analyse MBTI :
- **mbti_expert_analysis()** : Analyse experte avec classification des questions
- **extract_mbti_profiles_from_text()** : Extraction des types MBTI 4-lettres
- **validate_mbti_type()** : Validation des types MBTI

### 3. Lencioni
Fonctions d'analyse Lencioni Five Dysfunctions :
- **lencioni_intent_analysis()** : Analyse d'intent (REPORT_LOOKUP, GENERAL_KNOWLEDGE, INSIGHT_BLEND)
- **lencioni_analysis()** : Récupération des données d'évaluation équipe
- **get_dysfunction_scores()** : Récupération des scores par dysfonction
- **extract_dysfunctions_from_text()** : Extraction des dysfonctions mentionnées

### 4. Prompts
Construction des prompts par sous-thème :
- **create_prompt_by_subtheme()** : Factory qui choisit le bon prompt
- **build_lencioni_prompt()** : Prompt spécialisé pour D6_CollectiveSuccess
- **build_mbti_prompt()** : Prompt pour A1_PersonalityMBTI

### 5. Response
Génération des réponses finales :
- **generate_final_response()** : Générateur unifié avec streaming
- **generate_final_response_original()** : Version originale conservée
- **create_error_response()** : Réponses d'erreur standardisées

## Usage

```python
# Import des fonctions nécessaires
from modules.mbti import mbti_expert_analysis
from modules.lencioni import lencioni_intent_analysis, lencioni_analysis
from modules.prompts import create_prompt_by_subtheme
from modules.response import generate_final_response

# Dans le workflow principal
state = mbti_expert_analysis(state)
state = lencioni_intent_analysis(state)
state = lencioni_analysis(state)

# Génération de la réponse
prompt = create_prompt_by_subtheme(state.get('sub_theme'), state)
response = generate_final_response(state)
```

## Avantages de cette Structure

1. **Séparation des responsabilités** : Chaque module a un rôle spécifique
2. **Réutilisabilité** : Les fonctions peuvent être importées individuellement
3. **Maintenabilité** : Plus facile de modifier/déboguer des modules spécifiques
4. **Extensibilité** : Facile d'ajouter de nouveaux sous-thèmes ou modules
5. **Tests** : Possibilité de tester chaque module individuellement
6. **Documentation** : Code mieux organisé et documenté

## Migration

Le fichier original `zest_companion_workflow.py` reste intact pour éviter toute régression. Cette structure modulaire peut être adoptée progressivement :

1. Importer les modules dans le fichier original
2. Remplacer progressivement les fonctions inline par les modules
3. Tester chaque migration individuellement
4. Finaliser la transition quand tous les tests passent

## Notes Importantes

- Tous les imports et dépendances sont préservés
- La logique métier reste identique à l'original
- Les types et configurations sont partagés entre modules
- Le streaming LangGraph fonctionne avec cette structure