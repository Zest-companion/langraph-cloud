## ZEST MBTI Workflow — LangGraph Studio

Orchestration LangGraph fidèle au workflow n8n d’origine, avec routage conditionnel, recherche vectorielle multilingue (FR/EN) et garde‑fous pour la génération de réponses.

### Diagramme (haut niveau)
```
[START] → [Fetch User Profile] → [Fetch Temperament] → [MBTI Expert]
                   ↓
             [Conditional Router]
                   ↓
  [Tools A+B] / [Tools A+B+C] / [Tool C] / [Tool D]
                   ↓
          [Generate Final Response] → [END]
```

---

## Prérequis
- Python 3.9+
- Compte Supabase + base Postgres (extension `vector` activée)
- Clé OpenAI (`OPENAI_API_KEY`)
- Tables de contenu et index vectoriels (voir plus bas)

---

## Installation
1) Installer LangGraph Studio
```bash
pip install "langgraph[studio]"
# ou
uv pip install "langgraph[studio]"
```

2) Variables d’environnement (dans `langraph-studio/.env`)
```bash
OPENAI_API_KEY=sk-...
SUPABASE_URL=https://<project>.supabase.co
# Une des deux clés ci‑dessous (éviter SERVICE_ROLE en front)
SUPABASE_SERVICE_ROLE_KEY=...
# ou
SUPABASE_ANON_KEY=...
```

3) Lancer le Studio
```bash
cd langraph-studio
langraph dev
```
Interface: `http://localhost:2024`

---

## Utilisation de l’API Studio
- Endpoint streaming:
```
POST http://localhost:2024/runs/stream
```
- Exemple de payload minimal:
```json
{
  "input": {
    "messages": [{"role": "user", "content": "Comment gérer un conflit d'équipe ?"}],
    "user_message": "Comment gérer un conflit d'équipe ?",
    "main_theme": "A_UnderstandingMyselfAndOthers",
    "sub_theme": "A1_PersonalityMBTI",
    "user_id": "uuid-123",
    "user_name": "Chloé Aerts",
    "client": "Zest Companion",
    "cohort": "AGL_ELV1",
    "filter": "AGL_ELV1/chloe_aerts/A1_PersonalityMBTI",
    "folder_path": "participants/AGL_ELV1/chloe_aerts"
  },
  "config": { "thread_id": "conversation-unique-id" }
}
```

### Champs `input` reconnus (principaux)
- `messages`: historique conversationnel (facultatif)
- `user_message`: message courant de l’utilisateur
- `user_id`, `user_name`, `user_email` (optionnel)
- `main_theme`, `sub_theme` (pour filtrage de contenu)
- `client`, `cohort` (traçabilité)
- `folder_path`: nécessaire pour Tool A (participants)

---

## Nœuds du workflow
- `fetch_user_profile` → Table `profiles` (MBTI, tempérament)
- `fetch_temperament_description` → Table `temperament` (description)
- `mbti_expert_analysis` → Analyse de la requête, classification (PERSONAL_DEVELOPMENT / COMPARISON / OTHER_TYPES / GENERAL_MBTI / GREETING)
- `analyze_temperament_facets` (Node 3.5) → Détection facettes et tempéraments à rechercher (mode debug activable)
- `route_to_tools` → Routage conditionnel vers A/B/C/D
- `execute_tools_ab` → Tool A (participants) + Tool B (documents)
- `execute_tools_abc` → A + B + C (autres profils)
- `execute_tools_c` → C seul (autres profils)
- `execute_tools_d` → D seul (connaissances générales)
- `generate_final_response` → Synthèse finale avec garde‑fous

---

## Données & Recherche vectorielle

### Modèles d’embedding
- Requêtes: `text-embedding-3-small`
- IMPORTANT: utiliser le même modèle pour l’indexation des contenus

### Tables attendues (par défaut)
- `participants_content_test` (Tool A)
- `documents_content_test` (Tools B/C/D)
- Optionnel: `temperaments_content` pour contenus « spécial tempéraments »

### Métadonnées recommandées
- Participants (A): `folder_path`, `language`
- Documents (B/C/D): `sub_theme`, `mbti_type`, `language`
- Tempéraments (spécial): `mbti_family`, `temperament`, `facet`, `language`, `source_type=documents_special`, `document_key`

### RPC Supabase (exemples)
```sql
-- match_documents
begin
  return query
  select id, content, metadata,
         1 - cosine_distance(embedding, query_embedding) as similarity
  from documents_content_test
  where metadata @> filter
  order by similarity desc
  limit match_count;
end;

-- match_participants
begin
  return query
  select id, content, metadata,
         1 - cosine_distance(embedding, query_embedding) as similarity
  from participants_content_test
  where metadata @> filter
  order by similarity desc
  limit match_count;
end;
```

### Index vectoriels
```sql
create extension if not exists vector;

-- Documents
drop index if exists documents_content_test_embedding_cosine_idx;
create index documents_content_test_embedding_cosine_idx
on public.documents_content_test
using ivfflat (embedding vector_cosine_ops)
with (lists = 100);

-- Participants
drop index if exists participants_content_test_embedding_cosine_idx;
create index participants_content_test_embedding_cosine_idx
on public.participants_content_test
using ivfflat (embedding vector_cosine_ops)
with (lists = 100);

analyze public.documents_content_test;
analyze public.participants_content_test;
```

### Stratégies (A/B/C/D)
- **A (Participants)**: filtre strict sur `folder_path`; requêtes base + `language='fr'` + `language='en'`, fusion, tri, dé‑duplication.
- **B (Documents)**: `sub_theme` + `mbti_type`; même logique multilingue.
- **C (Others)**: recherche par profils MBTI listés (ex: `ENTJ,ISFP`).
- **D (General)**: recherche générale (sans `mbti_type`).

### Sanitize des résultats
- Filtrage par métadonnées, tri par similarité, dé‑duplication,
- Budgets de caractères par item et global par tool (voir code).

---

## Contenu spécial « Tempéraments » (optionnel)
Si vous vectorisez `00KEY_Personality_MBTI_Temperaments.pdf` dans une table dédiée (ex: `temperaments_content`):
- Modèle d’entête recommandé par chunk: `[Temperament: <name>][Facet: <facet>][Locale: fr-BE][Source: <label>]`
- Métadonnées stables (ex.):
  - `mbti_family` (SJ/SP/NF/NT), `temperament` (Guardian/Commando/Catalyst/Architect), `facet` (ex: `values`, `strengths`), `document_key`, `source_type='documents_special'`, `version`.
- Assurez-vous que la recherche (Tools) utilise la bonne table/RPC si vous ciblez ces contenus.

---

## Prompt final (garde‑fous clés)
- Utiliser uniquement la description du tempérament + résultats RAG fournis
- Pas de surnoms MBTI ni de connaissances externes hors contexte
- Réponses concises, concrètes; terminer par 1 question de coaching

---

## Tests & Scénarios
- Fichier: `tests/test_scenarios.py` (10 scénarios types)
- Pour lister les scénarios:
```bash
python langraph-studio/tests/test_scenarios.py
```

---

## Débogage rapide
- Pas de résultats RAG:
  - Vérifier les filtres `metadata` (clés/valeurs exactes)
  - Confirmer les RPC (`metadata @> filter`, tri par similarité)
  - Vérifier l’alignement du modèle d’embedding (indexation vs requêtes)
- Téléchargement Supabase (404):
  - Checker le bucket et le chemin exacts; variable `SUPABASE_DOCS_BUCKET` côté scripts d’ingestion
- Performances:
  - Indices IVFFLAT créés + `analyze`

---

## Sécurité & bonnes pratiques
- Ne pas exposer `SUPABASE_SERVICE_ROLE_KEY` côté client
- Limiter les logs en production (métadonnées sensibles possibles)
- Surveiller le budget tokens; ajuster `top_k`/budgets si troncatures

---

## Déploiement (optionnel)
```bash
langraph build
# puis déployer selon votre infra (ex: Cloud Run, Docker, etc.)
```

---

## Fichier de configuration Studio
`langgraph.json`
```json
{
  "dependencies": ["."],
  "graphs": {"zest_mbti_workflow": "./src/zest_mbti_workflow.py:graph"},
  "env": ".env"
}
```

---

## Licence
Voir `LICENSE` à la racine du dépôt.
