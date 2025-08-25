"""
Outils d'exécution de recherches vectorielles pour MBTI
"""
import logging
import os
from typing import Dict, List, Optional
from ..common.types import WorkflowState
from ..common.config import supabase
from ..lencioni.lencioni_analysis import perform_supabase_vector_search, sanitize_vector_results
from ..common.llm_utils import normalize_name_for_metadata

logger = logging.getLogger(__name__)

# Helper function: Récupération directe des tempéraments par filtres (pas de recherche vectorielle)
def search_temperaments_documents(temperament_analysis: Dict, limit: int = 5) -> List[Dict]:
    """
    Récupère directement les documents de tempéraments par filtres métadonnées
    (pas de recherche vectorielle, juste récupération par tempérament + facette)
    """
    if not temperament_analysis:
        return []
    
    logger.info("🏛️ Fetching temperaments documents by metadata filters...")
    
    temperaments = temperament_analysis.get("temperaments_to_search", [])
    facets = temperament_analysis.get("relevant_facets", [])
    
    if not temperaments or not facets:
        logger.info("  ⚠️ No temperaments or facets to fetch")
        return []
    
    logger.info(f"  📊 Processing {len(temperaments)} temperaments: {temperaments}")
    logger.info(f"  📊 Processing {len(facets)} facets: {facets}")
    
    all_results = []
    
    try:
        # Récupération directe par filtres sans recherche vectorielle
        for temperament in temperaments:  # Traiter tous les tempéraments demandés
            for facet in facets[:3]:  # Limiter à 3 facettes max
                logger.info(f"  🔍 Fetching {temperament}/{facet}...")
                
                # Requête directe sur la table documents_content_test
                try:
                    response = supabase.table('documents_content_test').select('content,metadata').eq(
                        'metadata->>mbti_family', temperament
                    ).eq(
                        'metadata->>facet', facet
                    ).eq(
                        'metadata->>document_key', 'MBTI_Temperaments_v2'
                    ).limit(2).execute()
                    
                    if response.data:
                        for item in response.data:
                            # Formater le résultat
                            result = {
                                'content': item.get('content', ''),
                                'metadata': item.get('metadata', {}),
                                'temperament': temperament,
                                'facet': facet,
                                'similarity': 1.0  # Score fixe car pas de recherche vectorielle
                            }
                            all_results.append(result)
                            
                        logger.info(f"    ✅ Found {len(response.data)} chunks for {temperament}/{facet}")
                    else:
                        logger.info(f"    ⚠️ No content found for {temperament}/{facet}")
                        
                except Exception as inner_e:
                    logger.info(f"    ❌ Error fetching {temperament}/{facet}: {inner_e}")
                    
    except Exception as e:
        logger.info(f"  ❌ Error fetching temperaments: {e}")
    
    # Retourner directement tous les résultats (plus de déduplication défectueuse)
    logger.info(f"✅ Total temperament documents found: {len(all_results)}")
    
    return all_results[:limit]



# NODE 1: Récupérer le profil MBTI de l'utilisateur
def fetch_user_profile(state: WorkflowState) -> WorkflowState:
    """
    Étape 1: Récupération du profil MBTI depuis Supabase
    - Input: user_email (email) ou user_name (nom complet) comme fallback
    - Process: Recherche par email dans la table profiles
    - Output: user_mbti, user_temperament
    """
    logger.info("🔍 NODE 1: Fetching user MBTI profile...")
    
    # Priorité à l'email, fallback sur user_name
    user_email = state.get("user_email", "").strip()
    user_name = state.get("user_name", "").strip()
    
    # 🔴 MODE TEST JEAN-PIERRE - DÉCOMMENTER POUR TESTER
    # user_email = "jean-pierre.aerts@zestforleaders.com"
    # user_name = "Jean-Pierre Aerts"
    # logger.info("🔴 TEST MODE: Forçage Jean-Pierre Aerts")
    
    if not user_email and not user_name:
        logger.info("❌ No user_email or user_name provided")
        return {**state, "user_mbti": None, "user_temperament": None, "pcm_base": None, "pcm_phase": None}
    
    try:
        response = None
        
        # PRIORITÉ 1: Recherche par email si disponible
        if user_email:
            logger.info(f"🔍 Searching by email: '{user_email}'")
            
            # Essayer d'abord avec la colonne 'email'
            try:
                
                # Utiliser ilike qui est moins strict que eq
                query = supabase.table("profiles").select("mbti, temperament, pcm_base, pcm_phase, first_name, last_name, email")
                query = query.ilike("email", user_email.lower())  # Forcer en minuscules
                response = query.execute()
                logger.info(f"🔍 Email search (ilike) result: {len(response.data) if response.data else 0} records found")
                
                # Si ça ne marche pas, essayer avec eq
                if not response.data:
                    logger.info(f"🔍 Trying eq search for email...")
                    query = supabase.table("profiles").select("mbti, temperament, pcm_base, pcm_phase, first_name, last_name, email")
                    query = query.eq("email", user_email)
                    response = query.execute()
                    logger.info(f"🔍 Email search (eq) result: {len(response.data) if response.data else 0} records found")
                    
                # Dernière tentative : recherche avec filter
                if not response.data:
                    logger.info(f"🔍 Trying filter search for email...")
                    query = supabase.table("profiles").select("mbti, temperament, pcm_base, pcm_phase, first_name, last_name, email")
                    query = query.filter("email", "eq", user_email)
                    response = query.execute()
                    logger.info(f"🔍 Email filter search result: {len(response.data) if response.data else 0} records found")
            except Exception as email_error:
                logger.info(f"⚠️ Email search failed (column may not exist): {email_error}")
                
                # Essayer avec 'user_email' comme nom de colonne alternative
                try:
                    query = supabase.table("profiles").select("mbti, temperament, pcm_base, pcm_phase, first_name, last_name, user_email")
                    query = query.eq("user_email", user_email)
                    response = query.execute()
                    logger.info(f"🔍 User_email search result: {len(response.data) if response.data else 0} records found")
                except Exception as user_email_error:
                    logger.info(f"⚠️ User_email search also failed: {user_email_error}")
                    response = None
        
        # PRIORITÉ 2: Fallback sur le nom si pas d'email ou pas de résultat
        if (not response or not response.data) and user_name:
            logger.info(f"🔍 Falling back to name search: '{user_name}'")
            
            # Parser le nom complet en prénom/nom
            name_parts = [part.strip() for part in user_name.split() if part.strip()]
            
            if name_parts:
                first_name = name_parts[0]
                last_name = " ".join(name_parts[1:]) if len(name_parts) > 1 else ""
                
                logger.info(f"🔍 Searching by name: first_name='{first_name}', last_name='{last_name}'")
                
                # Recherche par nom avec eq
                query = supabase.table("profiles").select("mbti, temperament, pcm_base, pcm_phase, first_name, last_name")
                query = query.eq("first_name", first_name)
                if last_name:
                    query = query.eq("last_name", last_name)
                
                response = query.execute()
                logger.info(f"🔍 Name search (eq) result: {len(response.data) if response.data else 0} records found")
                
                # Si pas de résultat, essayer avec ilike et wildcards
                if not response.data:
                    logger.info("🔍 Trying with ilike and wildcards...")
                    query = supabase.table("profiles").select("mbti, temperament, pcm_base, pcm_phase, first_name, last_name")
                    query = query.ilike("first_name", f"%{first_name}%")
                    if last_name:
                        query = query.ilike("last_name", f"%{last_name}%")
                    response = query.execute()
                    logger.info(f"🔍 Name search (ilike) result: {len(response.data) if response.data else 0} records found")
        
        # Vérifier si on a trouvé un profil
        if response and response.data and len(response.data) > 0:
            # Prendre le premier résultat si plusieurs matches
            profile = response.data[0]
            user_mbti = profile.get("mbti")
            user_temperament = profile.get("temperament")
            pcm_base = profile.get("pcm_base")
            pcm_phase = profile.get("pcm_phase")
            
            logger.info(f"✅ Profile found: {profile.get('first_name')} {profile.get('last_name')} | MBTI: {user_mbti} | Temperament: {user_temperament} | PCM Base: {pcm_base} | PCM Phase: {pcm_phase}")
            
            # Validation des données récupérées
            if user_mbti and user_temperament:
                return {**state, "user_mbti": user_mbti, "user_temperament": user_temperament, "pcm_base": pcm_base, "pcm_phase": pcm_phase}
            else:
                logger.info(f"⚠️ Profile found but missing data: mbti={user_mbti}, temperament={user_temperament}")
                return {**state, "user_mbti": user_mbti, "user_temperament": user_temperament, "pcm_base": pcm_base, "pcm_phase": pcm_phase}
        else:
            search_info = f"email: {user_email}" if user_email else f"name: {user_name}"
            logger.info(f"❌ No profile found for {search_info}")
            return {**state, "user_mbti": None, "user_temperament": None, "pcm_base": None, "pcm_phase": None}
    
    except Exception as e:
        logger.info(f"❌ Error fetching profile: {type(e).__name__}: {str(e)}")
        # Log plus de détails pour debug
        logger.info(f"   - user_name: '{user_name}'")
        logger.info(f"   - parsed: first_name='{first_name}', last_name='{last_name}'")
        return {**state, "user_mbti": None, "user_temperament": None, "pcm_base": None, "pcm_phase": None}


# NODE 2: Récupérer la description du tempérament
def fetch_temperament_description(state: WorkflowState) -> WorkflowState:
    """
    Étape 2: Récupération de la description du tempérament depuis Supabase
    - Input: user_temperament (ex: "NF", "ST", etc.)
    - Process: Requête table temperament avec matching sur colonne type
    - Output: temperament_description
    """
    logger.info("🔍 NODE 2: Fetching temperament description...")
    
    user_temperament = state.get("user_temperament")
    if not user_temperament:
        logger.info("❌ No user_temperament provided")
        return {**state, "temperament_description": None}
    
    logger.info(f"🔍 Searching temperament: type='{user_temperament}'")
    
    try:
        # Construire la requête Supabase
        query = supabase.table("temperament").select("description, temperament").eq("temperament", user_temperament)
        
        logger.info(f"🔍 Temperament query: temperament = '{user_temperament}'")
        
        # Exécuter la requête
        response = query.execute()
        
        logger.info(f"🔍 Raw temperament response: {response}")
        logger.info(f"🔍 Temperament data: {response.data}")
        logger.info(f"🔍 Temperament count: {response.count}")
        
        if response.data and len(response.data) > 0:
            # Prendre le premier résultat
            temperament_row = response.data[0]
            description = temperament_row.get("description")
            
            logger.info(f"✅ Temperament found: {temperament_row.get('temperament')} | Description: {description[:100]}..." if description else "No description")
            
            # Validation des données récupérées
            if description:
                return {**state, "temperament_description": description}
            else:
                logger.info(f"⚠️ Temperament found but no description: {temperament_row}")
                return {**state, "temperament_description": None}
        else:
            logger.info(f"❌ No temperament found for type: {user_temperament}")
            
            # Debug: afficher quelques tempéraments disponibles
            try:
                debug_response = supabase.table("temperament").select("temperament, description").limit(5).execute()
                logger.info(f"🔍 Available temperaments: {[t.get('temperament') for t in debug_response.data] if debug_response.data else []}")
            except Exception as debug_e:
                logger.info(f"🔍 Debug temperament query failed: {debug_e}")
            
            return {**state, "temperament_description": None}
    
    except Exception as e:
        logger.info(f"❌ Error fetching temperament: {type(e).__name__}: {str(e)}")
        logger.info(f"   - user_temperament: '{user_temperament}'")
        return {**state, "temperament_description": None}


# NODE 3.5: Temperament Facet Analyzer
def analyze_temperament_facets(state: WorkflowState) -> WorkflowState:
    """
    Analyse la query reformulée pour identifier:
    1. Les facettes de tempérament pertinentes (values, strengths, leadership_style, etc.)
    2. Si c'est pour l'utilisateur ou d'autres types MBTI
    3. Les tempéraments à rechercher (SJ, SP, NF, NT)
    """
    # 🔥 DEBUG MODE: Activer pour tester l'analyse sans affecter les recherches existantes
    # True = Mode debug avec logs détaillés, les recherches de tempéraments sont désactivées  
    # False = Mode production, les recherches de tempéraments sont activées
    DEBUG_MODE = False  # ⚠️ Mode production activé
    
    if DEBUG_MODE:
        logger.info("🔥 DEBUG MODE ACTIVÉ - NODE 3.5 en mode log uniquement")
        logger.info("   ⚠️ Les recherches de tempéraments ne seront pas intégrées aux outils A, B, C, D")
        logger.info("   ℹ️ Mettre DEBUG_MODE = False pour activer les recherches")
    
    logger.info("🔍 NODE 3.5: Temperament Facet Analysis...")
    
    # Mapping MBTI types to temperaments
    MBTI_TO_TEMPERAMENT = {
        # SJ - Guardian
        "ISTJ": "SJ", "ISFJ": "SJ", "ESTJ": "SJ", "ESFJ": "SJ",
        # SP - Commando/Artisan
        "ISTP": "SP", "ISFP": "SP", "ESTP": "SP", "ESFP": "SP",
        # NF - Catalyst/Idealist
        "INFJ": "NF", "INFP": "NF", "ENFJ": "NF", "ENFP": "NF",
        # NT - Architect/Rational
        "INTJ": "NT", "INTP": "NT", "ENTJ": "NT", "ENTP": "NT"
    }
    
    # Classification IA des facettes (remplace l'approche par mots-clés)
    def classify_facets_with_ai(query: str, user_mbti: str = None) -> List[str]:
        """
        Utilise l'IA pour identifier intelligemment les facettes de tempérament pertinentes
        Multilingue et adaptable à tout type de question
        """
        try:
            prompt = f"""Vous êtes un expert MBTI. Analysez cette question et identifiez les 3-4 facettes de tempérament les plus pertinentes.

FACETTES DISPONIBLES avec définitions:

FACETTES PRINCIPALES:
• overview – Vue d'ensemble : Intro sur la vision et les priorités du tempérament, ce sur quoi il se focalise
• mottos – Mots d'ordre : Phrases clés ou slogans résumant leur état d'esprit
• values – Valeurs : Principes et croyances fondamentales qu'ils défendent
• desires – Désirs : Ce qu'ils recherchent ou souhaitent accomplir dans leur vie
• needs – Besoins : Conditions essentielles pour donner leur meilleur
• aversions – Aversions : Ce qu'ils évitent ou supportent mal (SP, NF, NT uniquement)
• learning_style – Style d'apprentissage : Manière préférée d'apprendre et d'acquérir des compétences
• leadership_style – Style de leadership : Façon de diriger, de prendre des décisions et de mobiliser les autres
• strengths – Forces : Points forts, talents naturels et atouts principaux
• recognition – Reconnaissance souhaitée : Formes d'appréciation ou de reconnaissance qu'ils valorisent
• general_traits – Traits généraux : Caractéristiques communes à tous les membres du tempérament
• weaknesses – Faiblesses potentielles : Limites ou zones de vulnérabilité typiques
• recommendations – Recommandations : Conseils pratiques pour progresser et mieux interagir avec eux

CONTEXTES SPÉCIFIQUES:
• context_family – Contexte familial : Comportement en famille, avec les enfants, relations parentales
• context_education – Contexte éducatif : Apprentissage, école, formation, développement des compétences
• context_work – Contexte professionnel : Travail, carrière, environnement professionnel, productivité
• context_authority – Exercice de l'autorité : Leadership, hiérarchie, prise de décision, management
• context_sectors – Secteurs d'activité : Domaines professionnels privilégiés, industries, métiers
• context_time – Relation au temps : Gestion du temps, ponctualité, planning, organisation temporelle
• context_money – Relation à l'argent : Gestion financière, attitude envers l'argent, priorités économiques

QUESTION: "{query}"
{f"TYPE MBTI UTILISATEUR: {user_mbti}" if user_mbti else ""}

Analysez l'intention de la question et choisissez les 3-4 facettes les plus pertinentes pour y répondre efficacement.

Répondez uniquement par les noms de facettes séparés par des virgules, sans explication.
Exemple: strengths,recommendations,context_work"""

            from openai import OpenAI
            client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,  # Augmenté de 0.3 à 0.7 pour plus de flexibilité
                max_tokens=50
            )
            
            facets_str = response.choices[0].message.content.strip()
            facets = [f.strip() for f in facets_str.split(',') if f.strip()]
            
            logger.info(f"🤖 IA Classification: {facets}")
            return facets[:3]  # Limiter à 3 facettes max
            
        except Exception as e:
            logger.info(f"⚠️ Erreur classification IA: {e}")
            # Fallback vers facettes par défaut
            return ["overview", "general_traits"]
    
    # Récupérer les données nécessaires
    reformulated_query = state.get('reformulated_query', '')
    user_mbti = state.get('user_mbti')
    mbti_analysis = state.get('mbti_analysis', {})
    other_mbti_profiles = mbti_analysis.get('other_mbti_profiles', [])
    
    # DEBUG: Vérifier pourquoi la query est vide
    logger.info(f"🔍 STATE KEYS: {list(state.keys())}")
    logger.info(f"📝 Reformulated query: '{reformulated_query}'")
    logger.info(f"👤 User MBTI: {user_mbti}")
    logger.info(f"🎯 Other profiles: {other_mbti_profiles}")
    logger.info(f"📊 Full mbti_analysis: {mbti_analysis}")
    
    # Essayer d'utiliser user_message comme fallback si reformulated_query est vide
    if not reformulated_query.strip():
        user_message = state.get('user_message', '')
        logger.info(f"⚠️ Reformulated query vide, utilisation user_message: '{user_message}'")
        reformulated_query = user_message
    
    # Déterminer les tempéraments à rechercher et les cibles
    temperaments_to_search = set()
    search_for_user = False
    search_for_others = False
    question_type = mbti_analysis.get('question_type', '')
    
    logger.info(f"🔍 Analyzing search targets for question type: {question_type}")
    
    # LOGIQUE BASÉE SUR LE TYPE DE QUESTION ET LE CONTENU
    
    # 1. QUESTIONS PERSONNELLES: "How can I...", "Comment puis-je...", etc.
    personal_indicators = ['how can i', 'comment puis-je', 'comment je peux', 'how do i', 'how should i']
    is_personal_question = any(indicator in reformulated_query.lower() for indicator in personal_indicators)
    
    # 2. Si l'utilisateur est mentionné ET d'autres types
    if other_mbti_profiles and is_personal_question:
        # Cas: "How can I manage conflict with ESTP" → Chercher pour USER + OTHERS
        search_for_user = True
        search_for_others = True
        logger.info(f"🎯 Personal question with other types → Search for USER + OTHERS")
        
    elif other_mbti_profiles and question_type == 'COMPARISON':
        # Cas: "Difference between ISFP and ESTP" → Chercher pour USER + OTHERS
        search_for_user = True
        search_for_others = True
        logger.info(f"🎯 Comparison question → Search for USER + OTHERS")
        
    elif other_mbti_profiles and not is_personal_question:
        # Cas: "How do ESTP handle conflict" → Chercher seulement OTHERS
        search_for_user = False
        search_for_others = True
        logger.info(f"🎯 Others-focused question → Search for OTHERS only")
        
    elif not other_mbti_profiles:
        # Cas: "How can I improve my leadership" → Chercher seulement USER
        search_for_user = True
        search_for_others = False
        logger.info(f"🎯 User-only question → Search for USER only")
    
    # 3. AJOUTER LES TEMPÉRAMENTS CORRESPONDANTS
    
    # Ajouter tempérament utilisateur si nécessaire
    if search_for_user and user_mbti and user_mbti in MBTI_TO_TEMPERAMENT:
        user_temperament = MBTI_TO_TEMPERAMENT[user_mbti]
        temperaments_to_search.add(user_temperament)
        logger.info(f"  ➕ Adding user temperament {user_temperament} ({user_mbti})")
    
    # Ajouter tempéraments des autres types si nécessaire  
    if search_for_others and other_mbti_profiles:
        logger.info(f"🔍 Processing other_mbti_profiles: {other_mbti_profiles} (type: {type(other_mbti_profiles)})")
        
        # Gérer différents formats possibles
        if isinstance(other_mbti_profiles, str):
            mbti_types = [t.strip() for t in other_mbti_profiles.split(',')]
        elif isinstance(other_mbti_profiles, list):
            mbti_types = other_mbti_profiles
        else:
            mbti_types = []
            
        logger.info(f"🔍 MBTI types to process: {mbti_types}")
        
        for mbti_type in mbti_types:
            if mbti_type in MBTI_TO_TEMPERAMENT:
                other_temperament = MBTI_TO_TEMPERAMENT[mbti_type]
                temperaments_to_search.add(other_temperament)
                logger.info(f"  ➕ Adding other temperament {other_temperament} ({mbti_type})")
            else:
                logger.info(f"  ⚠️ MBTI type '{mbti_type}' not found in mapping")
    
    # Identifier les facettes pertinentes avec l'IA
    logger.info(f"🔍 Query utilisée pour analyse facettes: '{reformulated_query}'")
    
    # Utiliser la classification IA pour identifier les facettes
    relevant_facets = classify_facets_with_ai(reformulated_query, user_mbti)
    
    # Fallback si la classification IA échoue ou retourne des facettes vides
    if not relevant_facets or relevant_facets == ["overview", "general_traits"]:
        logger.info("⚠️ Classification IA non optimale, utilisation de la logique de fallback")
        query_lower = reformulated_query.lower()
        
        # Logique de fallback simplifiée et intelligente
        if any(word in query_lower for word in ["leader", "manage", "diriger", "authority", "équipe", "team"]):
            relevant_facets = ["leadership_style", "context_authority", "strengths"]
        elif any(word in query_lower for word in ["learn", "apprendre", "étudier", "education", "formation"]):
            relevant_facets = ["learning_style", "context_education"]
        elif any(word in query_lower for word in ["stress", "pressure", "difficile", "challenge", "problème", "struggle"]):
            relevant_facets = ["weaknesses", "recommendations", "aversions"]
        elif any(word in query_lower for word in ["travail", "work", "job", "career", "professional", "productivité", "productivity"]):
            relevant_facets = ["strengths", "recommendations", "context_work"]
        elif any(word in query_lower for word in ["famille", "family", "enfant", "children", "parent", "familial"]):
            relevant_facets = ["context_family", "general_traits"]
        elif any(word in query_lower for word in ["value", "valeur", "belief", "principe", "important", "matter"]):
            relevant_facets = ["values", "needs", "desires"]
        elif any(word in query_lower for word in ["qui suis", "who am", "myself", "me comprendre", "understand me"]):
            relevant_facets = ["overview", "general_traits", "values"]
        else:
            # Défaut basé sur le type de question
            relevant_facets = ["overview", "general_traits"]
    
    # Garder seulement les facettes valides et limiter à 3
    valid_facets = [
        "overview", "mottos", "values", "desires", "needs", "aversions", 
        "learning_style", "leadership_style", "strengths", "recognition", 
        "general_traits", "context_family", "context_education", "context_work",
        "context_authority", "context_sectors", "context_time", "context_money",
        "weaknesses", "recommendations"
    ]
    relevant_facets = [f for f in relevant_facets if f in valid_facets][:3]
    
    logger.info(f"✅ Identified facets: {relevant_facets}")
    logger.info(f"🎯 Temperaments to search: {list(temperaments_to_search)}")
    logger.info(f"👤 Search for user: {search_for_user}")
    logger.info(f"👥 Search for others: {search_for_others}")
    
    # Créer le résultat de l'analyse
    temperament_analysis = {
        "temperaments_to_search": list(temperaments_to_search),
        "relevant_facets": relevant_facets,
        "search_for_user": search_for_user,
        "search_for_others": search_for_others,
        "classification_method": "AI",  # Indiquer que c'est la classification IA
        "debug_mode": DEBUG_MODE
    }
    
    # 🔥 DEBUG LOGGING - Affichage détaillé pour test
    if DEBUG_MODE:
        logger.info("\n" + "="*60)
        logger.info("🔥 DEBUG - ANALYSE DES TEMPÉRAMENTS")
        logger.info("="*60)
        logger.info(f"📝 Query analysée: '{reformulated_query}'")
        logger.info(f"👤 MBTI utilisateur: {user_mbti}")
        logger.info(f"👥 Autres profils MBTI: {other_mbti_profiles}")
        logger.info(f"🎯 Tempéraments identifiés: {list(temperaments_to_search)}")
        logger.info(f"📊 Facettes pertinentes: {relevant_facets}")
        logger.info(f"🔍 Rechercher pour utilisateur: {search_for_user}")
        logger.info(f"🔍 Rechercher pour autres: {search_for_others}")
        logger.info("🤖 Classification IA utilisée pour identifier les facettes pertinentes")
        
        # Simulation de recherche temperament (sans vraiment chercher)
        if list(temperaments_to_search) and relevant_facets:
            logger.info("\n🔍 SIMULATION - Recherches qui seraient effectuées:")
            for temp in list(temperaments_to_search)[:2]:
                for facet in relevant_facets[:2]:
                    logger.info(f"   📄 {temp} + {facet}")
        
        logger.info("="*60)
        logger.info("🔥 FIN DEBUG - NODE 3.5")
        logger.info("="*60 + "\n")
    
    # En mode debug, on peut désactiver les recherches réelles
    if DEBUG_MODE:
        # Désactiver temporairement l'analyse pour les outils
        logger.info("🔥 DEBUG: Analyse des tempéraments enregistrée mais pas utilisée dans les recherches")
        return {
            **state,
            "temperament_analysis": None  # Désactivé pour le debug
        }
    else:
        # Mode normal - ajouter au state
        return {
            **state,
            "temperament_analysis": temperament_analysis
        }

# NODE 4: Router conditionnel basé sur l'analyse MBTI
def route_to_tools(state: WorkflowState) -> str:
    """Router qui détermine quels outils exécuter"""
    logger.info("🔀 NODE 4: Routing to appropriate tools...")
    
    analysis = state.get("mbti_analysis", {})
    instructions = analysis.get("instructions", "").upper()
    question_type = analysis.get("question_type", "").upper()
    
    logger.info(f"🔍 Question type: '{question_type}'")
    logger.info(f"🔍 Analysis instructions: '{instructions}'")
    
    # Routing basé sur les nouvelles instructions standardisées
    if "NO_TOOLS" in instructions or question_type == "GREETING_SMALL_TALK":
        logger.info("➡️  Routing to: no_tools")
        return "no_tools"
    elif "CALL_ABC" in instructions or question_type == "COMPARISON":
        logger.info("➡️  Routing to: execute_tools_abc")
        return "execute_tools_abc"
    elif "CALL_AB" in instructions or question_type == "PERSONAL_DEVELOPMENT":
        logger.info("➡️  Routing to: execute_tools_ab")
        return "execute_tools_ab" 
    elif "CALL_C" in instructions or question_type == "OTHER_TYPES":
        logger.info("➡️  Routing to: execute_tools_c")
        return "execute_tools_c"
    elif "CALL_D" in instructions or question_type == "GENERAL_MBTI":
        logger.info("➡️  Routing to: execute_tools_d")
        return "execute_tools_d"
    else:
        # Fallback intelligent basé sur la présence de données
        if state.get('user_mbti'):
            logger.info(f"⚠️  No clear match, defaulting to user-focused search (execute_tools_ab)")
            return "execute_tools_ab"
        else:
            logger.info(f"⚠️  No clear match, defaulting to general search (execute_tools_d)")
            return "execute_tools_d"


# NODE 5A: Exécuter Tools A + B (User only)
def execute_tools_ab(state: WorkflowState) -> WorkflowState:
    """Execute 2 vector searches: Tool A (participants) + Tool B (documents), then combine results"""
    logger.info("🔧 NODE 5A: Executing 2 vector searches - Tools A + B...")
    
    personalized_content = []  # Tool A: Vector search results from participants collection
    generic_content = []       # Tool B: Vector search results from documents collection
    
    try:
        # Utiliser la query reformulée si disponible, sinon fallback sur le message original
        search_query = state.get('reformulated_query') or state.get('user_message', '') or (state.get('messages', [{}])[-1].get('content', '') if state.get('messages') else '')
        user_msg = search_query  # Pour compatibilité avec le reste du code
        logger.info(f"🔍 Using search query: '{search_query}'")
        
        # Tool A: Vector search in participants_content_test with folder_path filter
        logger.info("🔍 Tool A: Vector search in participants_content_test...")
        folder_path = state.get('folder_path', '')
        
        # NORMALISATION: Convertir jean-pierre_aerts -> jean_pierre_aerts pour matcher les métadonnées
        normalized_folder_path = normalize_name_for_metadata(folder_path)
        logger.info(f"🔍 Tool A - original folder_path: '{folder_path}'")
        logger.info(f"🔍 Tool A - normalized folder_path: '{normalized_folder_path}'")
        logger.info(f"🔍 Tool A - user_msg: '{user_msg}'")
        
        if user_msg and normalized_folder_path:
            base_filters = {'folder_path': normalized_folder_path}
            # 🔄 SIMPLIFIÉ: Une seule recherche sans filtre de langue
            raw_personalized_all = perform_supabase_vector_search(
                query=user_msg,
                match_function='match_participants',
                metadata_filters=base_filters,
                limit=10  # Augmenté car une seule recherche
            )

            # Sanitize with a slightly more permissive threshold to handle multilingual variance
            personalized_content = sanitize_vector_results(
                results=raw_personalized_all,
                required_filters=None,  # Pas de filtrage strict par métadonnées
                top_k=3,
                min_similarity=0.30,
                max_chars_per_item=1800,
                max_total_chars=5000,
            )

            # Add tool identifier to metadata
            for item in personalized_content:
                item.setdefault("metadata", {})
                item["metadata"]["tool"] = "A"
                item["metadata"]["source"] = "participants_content_test"
        else:
            logger.info(f"⚠️ Tool A skipped - missing user_msg or folder_path: msg='{user_msg}', original='{folder_path}', normalized='{normalized_folder_path}'")
        
        # Tool B: Vector search in documents_content_test with sub_theme and mbti_type filters
        logger.info("🔍 Tool B: Vector search in documents_content_test...")
        sub_theme = state.get('sub_theme', '')
        user_mbti = state.get('user_mbti', '')
        logger.info(f"🔍 Tool B - sub_theme: '{sub_theme}'")
        logger.info(f"🔍 Tool B - user_mbti: '{user_mbti}'")
        logger.info(f"🔍 Tool B - user_msg: '{user_msg}'")
        
        if user_msg and sub_theme and user_mbti:
            base_filters = {
                'sub_theme': sub_theme,
                'mbti_type': user_mbti
            }
            # 🔄 SIMPLIFIÉ: Une seule recherche sans filtre de langue
            raw_generic_all = perform_supabase_vector_search(
                query=user_msg,
                match_function='match_documents',
                metadata_filters=base_filters,
                limit=10  # Augmenté car une seule recherche
            )

            # Sanitize with the standard threshold for documents
            generic_content = sanitize_vector_results(
                results=raw_generic_all,
                required_filters=None,  # Pas de filtrage strict par métadonnées
                top_k=3,
                min_similarity=0.30,
                max_chars_per_item=1800,
                max_total_chars=5000,
            )

            # Add tool identifier to metadata
            for item in generic_content:
                item.setdefault("metadata", {})
                item["metadata"]["tool"] = "B"
                item["metadata"]["source"] = "documents_content_test"
        else:
            logger.info(f"⚠️ Tool B skipped - missing required fields: msg='{user_msg}', sub_theme='{sub_theme}', mbti='{user_mbti}'")
        
        logger.info(f"✅ Tool A results: {len(personalized_content)} items")
        logger.info(f"✅ Tool B results: {len(generic_content)} items")
        
        # Debug: show detailed results for LangGraph Studio visibility
        if personalized_content:
            logger.info(f"\n📋 TOOL A RESULTS ({len(personalized_content)} items):")
            for i, item in enumerate(personalized_content):
                logger.info(f"  [{i+1}] Similarity: {item.get('similarity', 'N/A')}")
                logger.info(f"      Content: {item['content'][:200]}...")
                logger.info(f"      Metadata: {item.get('metadata', {})}")
                logger.info("")
        else:
            logger.info("❌ Tool A: No results found")
            
        if generic_content:
            logger.info(f"\n📋 TOOL B RESULTS ({len(generic_content)} items):")
            for i, item in enumerate(generic_content):
                logger.info(f"  [{i+1}] Similarity: {item.get('similarity', 'N/A')}")
                logger.info(f"      Content: {item['content'][:200]}...")
                logger.info(f"      Metadata: {item.get('metadata', {})}")
                logger.info("")
        else:
            logger.info("❌ Tool B: No results found")
    
    except Exception as e:
        logger.info(f"❌ Error in tools A+B: {e}")
        # Fallback to original static content if vector search fails
        if state.get("user_mbti"):
            personalized_content = [{
                "content": f"[Tool A Fallback] En tant que {state.get('user_mbti', '')}, vous avez tendance à... (caractéristiques générales MBTI pour: {user_msg})",
                "metadata": {"source": "participants_content_test_fallback", "mbti": state.get('user_mbti', ''), "tool": "A"},
                "similarity": 0.85
            }]
        
        if state.get("sub_theme"):
            generic_content = [{
                "content": f"[Tool B Fallback] Informations sur le thème {state.get('sub_theme', '')} concernant: {user_msg}",
                "metadata": {"source": "documents_content_test_fallback", "sub_theme": state.get('sub_theme', ''), "tool": "B"},
                "similarity": 0.80
            }]
    
    # 🏛️ NOUVEAU: Recherche des tempéraments si disponible (après Tools A + B)
    temperament_content = []  # Initialiser le contenu des tempéraments
    try:
        temperament_analysis = state.get("temperament_analysis")
        if temperament_analysis:
            logger.info("🏛️ Tool T: Recherche supplémentaire des tempéraments...")
            temperament_results = search_temperaments_documents(temperament_analysis, limit=15)  # 3 temperaments × 3 facettes × max 2 chunks each = 18 max
            if temperament_results:
                logger.info(f"   ✅ Trouvé {len(temperament_results)} résultats de tempéraments")
                
                # Déterminer si c'est pour l'utilisateur ou d'autres profils
                search_for_user = temperament_analysis.get('search_for_user', False)
                search_for_others = temperament_analysis.get('search_for_others', False)
                
                # Récupérer les tempéraments user et others pour assignation intelligente
                user_mbti = state.get('user_mbti', '')
                user_temperament = None
                if user_mbti and user_mbti in {'INTJ': 'NT', 'INTP': 'NT', 'ENTJ': 'NT', 'ENTP': 'NT',
                                              'INFJ': 'NF', 'INFP': 'NF', 'ENFJ': 'NF', 'ENFP': 'NF',
                                              'ISTJ': 'SJ', 'ISFJ': 'SJ', 'ESTJ': 'SJ', 'ESFJ': 'SJ',
                                              'ISTP': 'SP', 'ISFP': 'SP', 'ESTP': 'SP', 'ESFP': 'SP'}:
                    MBTI_TO_TEMPERAMENT = {'INTJ': 'NT', 'INTP': 'NT', 'ENTJ': 'NT', 'ENTP': 'NT',
                                          'INFJ': 'NF', 'INFP': 'NF', 'ENFJ': 'NF', 'ENFP': 'NF',
                                          'ISTJ': 'SJ', 'ISFJ': 'SJ', 'ESTJ': 'SJ', 'ESFJ': 'SJ',
                                          'ISTP': 'SP', 'ISFP': 'SP', 'ESTP': 'SP', 'ESFP': 'SP'}
                    user_temperament = MBTI_TO_TEMPERAMENT.get(user_mbti)
                
                # Ajouter les métadonnées appropriées avec assignation intelligente
                for result in temperament_results:
                    result['source_category'] = 'temperament'
                    result['tool'] = 'T'  # Tool Temperament
                    temperament_name = result.get('temperament', 'Unknown')
                    facet_name = result.get('facet', 'Unknown') 
                    result['temperament_info'] = f"{temperament_name}/{facet_name}"
                    
                    # Assignation intelligente basée sur le tempérament
                    if search_for_user and not search_for_others:
                        result['target'] = 'user'
                        logger.info(f"    → Target: 'user' (user only)")
                    elif search_for_others and not search_for_user:
                        result['target'] = 'others'
                        logger.info(f"    → Target: 'others' (others only)")
                    elif search_for_user and search_for_others:
                        # Si les deux, assigner selon le tempérament du résultat
                        # Mapping des noms de tempéraments vers les codes
                        temperament_to_code = {'Commando': 'SP', 'Guardian': 'SJ', 'Catalyst': 'NF', 'Architect': 'NT'}
                        temperament_code = temperament_to_code.get(temperament_name, temperament_name)
                        
                        logger.info(f"    🔄 Target assignation: '{temperament_name}' (code: {temperament_code}) vs user: {user_temperament}")
                        
                        if temperament_code == user_temperament:
                            result['target'] = 'user'  # Ce tempérament correspond à l'utilisateur
                            logger.info(f"    → Target: 'user' (matches user)")
                        else:
                            result['target'] = 'others'  # Ce tempérament correspond aux autres types
                            logger.info(f"    → Target: 'others' (different from user)")
                    else:
                        result['target'] = 'mixed'
                        logger.info(f"    → Target: 'mixed' (fallback)")
                        
                    logger.info(f"    ✅ Added: {temperament_name}/{facet_name} → target='{result['target']}')")
                    
                    temperament_content.append(result)
                
                logger.info(f"   📈 Total temperament_content: {len(temperament_content)}")
                
                # Debug des résultats de tempéraments
                logger.info(f"\n🏛️ TOOL T RESULTS ({len(temperament_content)} items):")
                for i, item in enumerate(temperament_content):
                    logger.info(f"  [T{i+1}] Temperament: {item.get('temperament_info', 'N/A')}")
                    logger.info(f"       Target: {item.get('target', 'N/A')}")
                    logger.info(f"       Similarity: {item.get('similarity', 'N/A')}")
                    logger.info(f"       Content: {item['content'][:150]}...")
                    logger.info("")
            else:
                logger.info("   ⚠️ Aucun résultat de tempérament trouvé")
        else:
            logger.info("   ℹ️ Pas d'analyse de tempérament disponible (mode debug ou analyse échouée)")
    except Exception as e:
        logger.info(f"⚠️ Erreur recherche tempéraments: {e}")
    
    return {
        **state, 
        "personalized_content": personalized_content, 
        "generic_content": generic_content,
        "temperament_content": temperament_content  # 🏛️ NOUVEAU: Ajouter au state
    }

# NODE 5B: Exécuter Tools A + B + C (User + Others)
def execute_tools_abc(state: WorkflowState) -> WorkflowState:
    """Execute 3 vector searches: Tool A + B + C, then combine all results for synthesis/comparison"""
    logger.info("🔧 NODE 5B: Executing 3 vector searches - Tools A + B + C...")
    
    # Exécuter A + B d'abord (2 recherches vectorielles)
    state = execute_tools_ab(state)
    
    # Ajouter Tool C (3ème recherche vectorielle pour les autres profils MBTI)
    others_content = []
    try:
        logger.info("🔍 Tool C: Vector search for other MBTI profiles...")
        analysis = state.get("mbti_analysis", {})
        other_profiles = analysis.get("other_mbti_profiles")
        # Utiliser la query reformulée pour une meilleure recherche
        search_query = state.get('reformulated_query') or state.get('user_message', '') or (state.get('messages', [{}])[-1].get('content', '') if state.get('messages') else '')
        user_msg = search_query
        sub_theme = state.get('sub_theme', '')
        logger.info(f"🔍 Tool C - Using search query: '{search_query}'")
        
        if other_profiles and user_msg:
            profiles = [p.strip() for p in other_profiles.split(",")]
            # Filtrer les profils pour exclure celui de l'utilisateur
            filtered_profiles = [p for p in profiles if p and p != state.get('user_mbti')]
            
            if filtered_profiles:
                logger.info(f"🔍 Tool C - Searching individually for MBTI types: {filtered_profiles}")
                logger.info(f"🔍 Tool C - sub_theme: '{sub_theme}'")
                
                # Faire une recherche séparée pour chaque profil MBTI
                for profile in filtered_profiles:
                    logger.info(f"🔍 Tool C - Individual search for: {profile}")
                    
                    # Préparer les filtres pour ce profil spécifique
                    filters = {'mbti_type': profile}
                    if sub_theme:
                        filters['sub_theme'] = sub_theme
                    
                    # Recherche individuelle pour ce profil
                    raw_profile_results = perform_supabase_vector_search(
                        query=user_msg,
                        match_function='match_documents',
                        metadata_filters=filters,
                        limit=4  # 4 résultats max par profil pour éviter trop de contenu
                    )
                    
                    if raw_profile_results:
                        # Sanitize pour ce profil spécifique
                        profile_content = sanitize_vector_results(
                            results=raw_profile_results,
                            required_filters=None,
                            top_k=2,  # 2 meilleurs résultats par profil
                            min_similarity=0.30,
                            max_chars_per_item=1200,
                            max_total_chars=2400,
                        )
                        
                        # Ajouter les résultats avec metadata enrichie
                        for item in profile_content:
                            item.setdefault("metadata", {})
                            item["metadata"]["tool"] = "C"
                            item["metadata"]["source"] = "documents_content_test"
                            item["metadata"]["target_mbti"] = profile
                            others_content.append(item)
                        
                        logger.info(f"  ✅ Found {len(profile_content)} results for {profile}")
                    else:
                        logger.info(f"  ❌ No results found for {profile}")
            else:
                logger.info("🔍 Tool C - No valid profiles to search (all filtered out)")
        
        logger.info(f"✅ Tool C results: {len(others_content)} items")
        logger.info(f"📊 Total results for synthesis: A={len(state.get('personalized_content', []))} + B={len(state.get('generic_content', []))} + C={len(others_content)}")
        
        if others_content:
            logger.info(f"\n📋 TOOL C RESULTS ({len(others_content)} items):")
            for i, item in enumerate(others_content):
                logger.info(f"  [{i+1}] MBTI: {item.get('metadata', {}).get('target_mbti', 'N/A')}")
                logger.info(f"      Similarity: {item.get('similarity', 'N/A')}")
                logger.info(f"      Content: {item['content'][:200]}...")
                logger.info("")
    
    except Exception as e:
        logger.info(f"❌ Error in tool C: {e}")
    
    # Récupérer les tempéraments déjà générés par Tool T dans execute_tools_ab
    temperament_content = state.get("temperament_content", [])
    logger.info(f"   ✅ Using temperament content from AB: {len(temperament_content)} items")
    
    return {**state, "others_content": others_content, "temperament_content": temperament_content}

# NODE 5C: Exécuter Tool C uniquement (Others only)
def execute_tools_c(state: WorkflowState) -> WorkflowState:
    """Execute 1 vector search: Tool C only (others collection) for other people's MBTI profiles"""
    logger.info("🔧 NODE 5C: Executing 1 vector search - Tool C only...")
    
    others_content = []
    try:
        logger.info("🔍 Tool C: Vector search for other MBTI profiles (no user profile)...")
        analysis = state.get("mbti_analysis", {})
        other_profiles = analysis.get("other_mbti_profiles")
        # Utiliser la query reformulée pour une meilleure recherche
        search_query = state.get('reformulated_query') or state.get('user_message', '') or (state.get('messages', [{}])[-1].get('content', '') if state.get('messages') else '')
        user_msg = search_query
        sub_theme = state.get('sub_theme', '')
        logger.info(f"🔍 Tool C - Using search query: '{search_query}'")
        
        if other_profiles and user_msg:
            profiles = [p.strip() for p in other_profiles.split(",")]
            # Pas besoin de filtrer par user_mbti car c'est Tool C seul
            filtered_profiles = [p for p in profiles if p]
            
            if filtered_profiles:
                logger.info(f"🔍 Tool C - Searching individually for MBTI types: {filtered_profiles}")
                logger.info(f"🔍 Tool C - sub_theme: '{sub_theme}'")
                
                # Faire une recherche séparée pour chaque profil MBTI
                for profile in filtered_profiles:
                    logger.info(f"🔍 Tool C - Individual search for: {profile}")
                    
                    # Préparer les filtres pour ce profil spécifique
                    filters = {'mbti_type': profile}
                    if sub_theme:
                        filters['sub_theme'] = sub_theme
                    
                    # Recherche individuelle pour ce profil
                    raw_profile_results = perform_supabase_vector_search(
                        query=user_msg,
                        match_function='match_documents',
                        metadata_filters=filters,
                        limit=6  # Plus de résultats par profil car c'est le seul tool
                    )
                    
                    if raw_profile_results:
                        # Sanitize pour ce profil spécifique - plus généreux car Tool C seul
                        profile_content = sanitize_vector_results(
                            results=raw_profile_results,
                            required_filters=None,
                            top_k=3,  # 3 meilleurs résultats par profil
                            min_similarity=0.30,
                            max_chars_per_item=1500,
                            max_total_chars=4000,
                        )
                        
                        # Ajouter avec metadata enrichie
                        for item in profile_content:
                            item.setdefault("metadata", {})
                            item["metadata"]["tool"] = "C"
                            item["metadata"]["source"] = "documents_content_test"
                            item["metadata"]["target_mbti"] = profile
                            others_content.append(item)
                        
                        logger.info(f"  ✅ Found {len(profile_content)} results for {profile}")
                    else:
                        logger.info(f"  ❌ No results found for {profile}")
            else:
                logger.info("🔍 Tool C - No valid profiles to search")
        
        logger.info(f"✅ Tool C results: {len(others_content)} items")
        
        if others_content:
            logger.info(f"\n📋 TOOL C RESULTS ({len(others_content)} items):")
            for i, item in enumerate(others_content):
                logger.info(f"  [{i+1}] MBTI: {item.get('metadata', {}).get('target_mbti', 'N/A')}")
                logger.info(f"      Similarity: {item.get('similarity', 'N/A')}")
                logger.info(f"      Content: {item['content'][:200]}...")
                logger.info("")
    
    except Exception as e:
        logger.info(f"❌ Error in tool C: {e}")
    
    # 🏛️ NOUVEAU: Recherche des tempéraments pour les AUTRES profils uniquement
    temperament_content = []  # Initialiser vide car pas de recherche user
    
    try:
        temperament_analysis = state.get("temperament_analysis")
        if temperament_analysis and temperament_analysis.get('search_for_others'):
            logger.info("🏛️ Tool T (Others only): Recherche des tempéraments pour les autres profils...")
            
            # Cloner l'analyse et forcer la recherche pour others uniquement
            others_temperament_analysis = temperament_analysis.copy()
            others_temperament_analysis['search_for_user'] = False
            others_temperament_analysis['search_for_others'] = True
            
            temperament_results = search_temperaments_documents(others_temperament_analysis, limit=5)
            if temperament_results:
                logger.info(f"   ✅ Trouvé {len(temperament_results)} résultats de tempéraments pour others")
                
                # Ajouter les métadonnées pour others (Tool C context - others only)
                for result in temperament_results:
                    result['source_category'] = 'temperament'
                    result['tool'] = 'T'
                    temperament_name = result.get('temperament', 'Unknown')
                    facet_name = result.get('facet', 'Unknown')
                    result['temperament_info'] = f"{temperament_name}/{facet_name}"
                    result['target'] = 'others'  # Toujours others pour Tool C
                    
                    temperament_content.append(result)
                
                logger.info(f"   📈 Total temperament_content (others only): {len(temperament_content)}")
                
                # Debug des résultats
                logger.info(f"\n🏛️ TOOL T RESULTS (Others only - {len(temperament_content)} items):")
                for i, item in enumerate(temperament_content):
                    logger.info(f"  [T{i+1}] Temperament: {item.get('temperament_info', 'N/A')}")
                    logger.info(f"       Target: {item.get('target', 'N/A')}")
                    logger.info(f"       Content: {item['content'][:150]}...")
                    logger.info("")
            else:
                logger.info("   ⚠️ Aucun résultat de tempérament trouvé pour others")
        else:
            logger.info("   ℹ️ Pas de recherche tempérament nécessaire (pas d'autres profils)")
    except Exception as e:
        logger.info(f"⚠️ Erreur recherche tempéraments others: {e}")
    
    return {**state, "others_content": others_content, "temperament_content": temperament_content}

# NODE 5D: Exécuter Tool D (General)
def execute_tools_d(state: WorkflowState) -> WorkflowState:
    """Execute 1 vector search: Tool D only (general MBTI collection) for general MBTI knowledge"""
    logger.info("🔧 NODE 5D: Executing 1 vector search - Tool D...")
    
    general_content = []
    try:
        logger.info("🔍 Tool D: Vector search for general MBTI knowledge...")
        # Utiliser la query reformulée pour une meilleure recherche
        search_query = state.get('reformulated_query') or state.get('user_message', '') or (state.get('messages', [{}])[-1].get('content', '') if state.get('messages') else '')
        user_msg = search_query
        sub_theme = state.get('sub_theme', '')
        logger.info(f"🔍 Tool D - Using search query: '{search_query}'")
        
        if user_msg:
            # Tool D cherche dans documents_content_test SANS filtre mbti_type
            # pour trouver des informations générales sur le MBTI
            logger.info(f"🔍 Tool D - Searching general MBTI info")
            logger.info(f"🔍 Tool D - sub_theme: '{sub_theme}'")
            
            # Recherche avec seulement sub_theme si disponible
            filters = {}
            if sub_theme:
                filters['sub_theme'] = sub_theme
            
            # 🔄 SIMPLIFIÉ: Une seule recherche sans filtre de langue
            raw_general_all = perform_supabase_vector_search(
                query=user_msg,
                match_function='match_documents',
                metadata_filters=filters if filters else None,
                limit=10  # Augmenté car une seule recherche
            )
            
            # Sanitize avec paramètres généreux car c'est le seul tool
            general_content = sanitize_vector_results(
                results=raw_general_all,
                required_filters=None,  # Pas de filtrage strict par métadonnées
                top_k=4,
                min_similarity=0.25,  # Seuil plus bas pour contenu général
                max_chars_per_item=1800,
                max_total_chars=6000,
            )
            
            # Ajouter metadata
            for item in general_content:
                item.setdefault("metadata", {})
                item["metadata"]["tool"] = "D"
                item["metadata"]["source"] = "documents_content_test"
                item["metadata"]["search_type"] = "general_mbti"
        
        logger.info(f"✅ Tool D results: {len(general_content)} items")
        
        if general_content:
            logger.info(f"\n📋 TOOL D RESULTS ({len(general_content)} items):")
            for i, item in enumerate(general_content):
                logger.info(f"  [{i+1}] Similarity: {item.get('similarity', 'N/A')}")
                logger.info(f"      Content: {item['content'][:200]}...")
                logger.info(f"      Metadata: {item.get('metadata', {})}")
                logger.info("")
    
    except Exception as e:
        logger.info(f"❌ Error in tool D: {e}")
        # Fallback content si erreur
        general_content = [{
            "content": f"Les types MBTI sont basés sur 4 dimensions: Extraversion/Introversion, Sensation/Intuition, Pensée/Sentiment, Jugement/Perception. Chaque combinaison forme un type unique avec ses forces et défis.",
            "metadata": {"source": "fallback", "tool": "D"},
            "similarity": 0.50
        }]
    
    return {**state, "general_content": general_content, "temperament_content": []}

# NODE 5E: Pas d'outils - Réponse directe
def no_tools(state: WorkflowState) -> WorkflowState:
    """Pas d'outils nécessaires - réponse directe basée sur l'analyse"""
    logger.info("🔧 NODE 5E: No tools needed - direct response...")
    
    # Marquer explicitement qu'aucune recherche n'est nécessaire
    # Cela permet au generate_final_response de savoir qu'il doit répondre différemment
    return {**state, 
            "personalized_content": [],
            "generic_content": [],
            "others_content": [],
            "general_content": [],
            "temperament_content": [],  # 🏛️ Initialiser vide
            "no_search_needed": True}
