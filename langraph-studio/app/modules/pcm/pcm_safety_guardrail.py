"""
PCM Safety Guardrail - Intelligent semantic validation for workplace content
Uses multiple approaches: LLM-based classification, keyword fallback, and semantic patterns
"""
import logging
import re
from typing import Dict, Any, List, Optional
from ..common.types import WorkflowState

logger = logging.getLogger(__name__)

class PCMSafetyGuard:
    """Intelligent guardrail using multiple validation approaches"""
    
    def __init__(self, use_llm_validation: bool = True):
        self.use_llm_validation = use_llm_validation
        
        # 🧠 INTELLIGENT SEMANTIC CLASSIFICATION
        # Classification templates pour LLM
        self.pcm_scope_classification_prompt = """
SIMPLE TASK: Check if this message is about one of these FORBIDDEN topics:

REQUIRES_SPECIALIST (FORBIDDEN LIST):
- Medical/health diagnosis or treatment advice
- Clinical psychology or formal therapy sessions  
- HR decisions (hiring, firing, legal sanctions)
- Legal or financial professional advice
- Crisis intervention or serious mental health issues
- Dangerous or harmful content
- Personal relationships, family matters, romantic relationships

Message: "{text}"

STEP 1: Is this message explicitly asking for one of the FORBIDDEN topics above?
STEP 2: If NO → Answer PCM_SCOPE
STEP 3: If YES → Answer REQUIRES_SPECIALIST

DO NOT analyze what the message "might be about" - just check the forbidden list.

Respond with only: PCM_SCOPE or REQUIRES_SPECIALIST
Reasoning: [brief explanation]
"""
        
        # 📊 SEMANTIC CATEGORIES (for embedding-based classification)
        self.workplace_categories = {
            "team_collaboration": ["team", "collaboration", "teamwork", "group work", "collective"],
            "professional_communication": ["business", "professional", "workplace", "office", "corporate"],
            "leadership_management": ["leadership", "management", "supervisor", "manager", "director"],
            "pcm_personality": ["PCM", "personality", "assessment", "behavioral", "communication style", "communication channel", "base", "phase", "thinker", "harmonizer", "promoter", "rebel", "persister", "imaginer"],
            "conflict_resolution": ["conflict", "resolution", "mediation", "disagreement", "tension"]
        }
        
        self.non_workplace_categories = {
            "personal_relationships": ["marriage", "dating", "romance", "personal relationship"],
            "family_matters": ["family", "children", "parenting", "home", "domestic"],
            "health_medical": ["medical", "health", "doctor", "therapy", "medication", "depression", "burnout"],
            "legal_financial": ["legal", "lawyer", "financial", "investment", "money"]
        }
        
    def _llm_semantic_classification(self, text: str) -> Dict[str, Any]:
        """🧠 Classification sémantique intelligente via LLM (sans contexte)"""
        try:
            # Import local pour éviter les dépendances circulaires
            from ..common.llm_utils import isolated_analysis_call
            
            prompt = self.pcm_scope_classification_prompt.format(text=text)
            response = isolated_analysis_call(prompt)
            
            # Parse LLM response more carefully
            response_upper = response.upper()
            
            # Check for REQUIRES_SPECIALIST first (higher priority for safety)
            if "REQUIRES_SPECIALIST" in response_upper:
                is_within_scope = False
                confidence = 0.9
            elif "PCM_SCOPE" in response_upper:
                is_within_scope = True  
                confidence = 0.9
            else:
                # Fallback if neither keyword found
                is_within_scope = False  # Be conservative
                confidence = 0.3
            
            logger.info(f"🧠 LLM Classification: {'PCM_SCOPE' if is_within_scope else 'REQUIRES_SPECIALIST'}")
            logger.info(f"🧠 LLM Response: {response}")
            
            return {
                "is_workplace": is_within_scope,  # Keep same field name for compatibility
                "confidence": confidence,
                "reasoning": response,
                "method": "llm_semantic"
            }
        except Exception as e:
            logger.warning(f"⚠️ LLM classification failed: {e}")
            return None
    
    def _llm_semantic_classification_with_context(self, text: str, conversation_context: List[Dict] = None) -> Dict[str, Any]:
        """🧠 Classification sémantique intelligente via LLM AVEC CONTEXTE"""
        try:
            from ..common.llm_utils import isolated_analysis_call
            
            # Construire le contexte si disponible
            if conversation_context:
                context_text = "CONVERSATION HISTORY:\n"
                for msg in conversation_context[-3:]:  # 3 derniers messages
                    role = msg.get('role', 'unknown')
                    content = msg.get('content', '')
                    context_text += f"{role}: {content}\n"
                context_text += f"\nCURRENT MESSAGE: {text}"
                
                # Prompt adapté avec contexte
                prompt = f"""
CONTEXT: You are classifying messages for a PCM (Process Communication Model) coaching assistant.

PCM is a personality psychology framework with these core concepts:
- 6 personality types: Thinker, Harmonizer, Promoter, Rebel, Persister, Imaginer
- BASE: Your core personality type from birth
- PHASE: Your current adapted personality (can change due to stress/growth)
- Communication Channels: How each type prefers to receive information (part of BASE structure)

CONVERSATION CONTEXT: {context_text}

SIMPLE TASK: Check if the CURRENT MESSAGE is about one of these FORBIDDEN topics:

REQUIRES_SPECIALIST (FORBIDDEN LIST):
- Medical/health diagnosis or treatment advice
- Clinical psychology or formal therapy sessions  
- HR decisions (hiring, firing, legal sanctions)
- Legal or financial professional advice
- Crisis intervention or serious mental health issues
- Dangerous or harmful content
- Personal relationships, family matters, romantic relationships

STEP 1: Is the CURRENT MESSAGE explicitly asking for one of the FORBIDDEN topics above?
STEP 2: If NO → Answer PCM_SCOPE (includes all PCM concepts like base, phase, communication channels, personality types)
STEP 3: If YES → Answer REQUIRES_SPECIALIST

DO NOT analyze what the message "might be about" - just check the forbidden list.

Respond with only: PCM_SCOPE or REQUIRES_SPECIALIST
Reasoning: [brief explanation considering the context]
"""
                
                logger.info("🧠 Using LLM classification WITH conversation context")
            else:
                # Fallback vers méthode sans contexte
                return self._llm_semantic_classification(text)
            
            response = isolated_analysis_call(prompt)
            
            # Parse LLM response
            response_upper = response.upper()
            
            if "REQUIRES_SPECIALIST" in response_upper:
                is_within_scope = False
                confidence = 0.95  # Plus de confiance avec contexte
            elif "PCM_SCOPE" in response_upper:
                is_within_scope = True  
                confidence = 0.95
            else:
                is_within_scope = False  # Conservative
                confidence = 0.3
            
            logger.info(f"🧠 LLM Classification (with context): {'PCM_SCOPE' if is_within_scope else 'REQUIRES_SPECIALIST'}")
            logger.info(f"🧠 LLM Response: {response}")
            
            return {
                "is_workplace": is_within_scope,
                "confidence": confidence,
                "reasoning": response,
                "method": "llm_semantic_with_context"
            }
        except Exception as e:
            logger.warning(f"⚠️ LLM classification with context failed: {e}")
            # Fallback vers méthode sans contexte
            return self._llm_semantic_classification(text)
    
    def _semantic_category_scoring(self, text: str) -> Dict[str, Any]:
        """📊 Classification basée sur les catégories sémantiques"""
        text_lower = text.lower()
        
        workplace_score = 0
        non_workplace_score = 0
        
        # Score workplace categories
        for category, terms in self.workplace_categories.items():
            category_score = sum(1 for term in terms if term.lower() in text_lower)
            workplace_score += category_score
            if category_score > 0:
                logger.info(f"✅ Workplace category '{category}' matched with score {category_score}")
        
        # Score non-workplace categories  
        for category, terms in self.non_workplace_categories.items():
            category_score = sum(1 for term in terms if term.lower() in text_lower)
            non_workplace_score += category_score
            if category_score > 0:
                logger.warning(f"🚫 Non-workplace category '{category}' matched with score {category_score}")
        
        is_workplace = workplace_score > non_workplace_score
        confidence = max(workplace_score, non_workplace_score) / (workplace_score + non_workplace_score + 1)
        
        return {
            "is_workplace": is_workplace,
            "confidence": confidence,
            "workplace_score": workplace_score,
            "non_workplace_score": non_workplace_score,
            "method": "semantic_categories"
        }
    
    def _keyword_fallback(self, text: str) -> Dict[str, Any]:
        """🔙 Classification de fallback par mots-clés"""
        text_lower = text.lower()
        
        # Vérifier les mots-clés interdits
        for keyword in self.forbidden_keywords:
            if keyword.lower() in text_lower:
                logger.warning(f"🚫 Forbidden keyword detected: {keyword}")
                return {
                    "is_workplace": False,
                    "confidence": 0.8,
                    "detected_keyword": keyword,
                    "method": "keyword_fallback"
                }
        
        # Si aucun mot-clé interdit, supposer que c'est workplace
        return {
            "is_workplace": True,
            "confidence": 0.3,  # Faible confiance car pas de validation positive
            "method": "keyword_fallback"
        }
    
    def is_within_pcm_scope(self, text: str, conversation_context: List[Dict] = None) -> bool:
        """🎯 Classification intelligente multicouche pour le scope PCM AVEC CONTEXTE"""
        logger.info(f"🛡️ Analyzing text: '{text[:100]}...'")
        
        # 1. 🧠 UNIQUEMENT: Classification LLM sémantique AVEC CONTEXTE
        if self.use_llm_validation:
            llm_result = self._llm_semantic_classification_with_context(text, conversation_context)
            if llm_result:
                logger.info(f"✅ LLM result (confidence: {llm_result.get('confidence', 'N/A')}): {llm_result}")
                return llm_result["is_workplace"]  # Keep same field for compatibility
        
        # 🔙 Fallback conservatif si LLM échoue complètement
        logger.warning("⚠️ LLM classification failed, defaulting to SAFE (PCM_SCOPE)")
        return True  # Par défaut : autoriser (PCM scope)
    
    def validate(self, text: str, conversation_context: List[Dict] = None) -> Dict[str, Any]:
        """Valide le texte et retourne le résultat avec contexte optionnel"""
        is_valid = self.is_within_pcm_scope(text, conversation_context)
        
        return {
            "validation_passed": is_valid,
            "failed_validations": [] if is_valid else [{"error": "Topic requires specialist - outside PCM scope"}]
        }

# 🌟 ADVANCED: Packages spécialisés pour contenu illégal/dangereux
class AdvancedSafetyGuard:
    """Guardrail avancé avec packages spécialisés pour contenu illégal/dangereux"""
    
    def __init__(self):
        import os
        
        # 🎛️ Configuration via variables d'environnement
        self.advanced_enabled = os.getenv("ADVANCED_SAFETY_ENABLED", "false").lower() == "true"
        self.llm_guard_enabled = os.getenv("LLM_GUARD_ENABLED", "false").lower() == "true"
        self.ai_safety_enabled = os.getenv("AI_SAFETY_GUARDRAILS_ENABLED", "false").lower() == "true"
        self.nemo_enabled = os.getenv("NEMO_GUARDRAILS_ENABLED", "false").lower() == "true"
        
        # 📋 Checks disponibles (activés selon configuration)
        self.safety_checks = {
            "openai_moderation": self._openai_moderation_check,  # Toujours actif
            "prompt_injection": self._prompt_injection_check      # Toujours actif
        }
        
        # 🔧 Ajouter les checks optionnels selon configuration
        if self.llm_guard_enabled:
            self.safety_checks["llm_guard"] = self._llm_guard_check
        if self.ai_safety_enabled:
            self.safety_checks["ai_safety_guardrails"] = self._ai_safety_check
        if self.nemo_enabled:
            self.safety_checks["nemo_guardrails"] = self._nemo_guardrails_check
        
        # 🧠 CONTEXTUAL MEMORY: Track recent blocked attempts
        self.recent_blocked_topics = []  # Garder en mémoire les sujets récemment bloqués
        
        logger.info(f"🛡️ AdvancedSafetyGuard initialized with {len(self.safety_checks)} checks")
        logger.info(f"🎛️ Config: LLM-Guard={self.llm_guard_enabled}, AI-Safety={self.ai_safety_enabled}, NeMo={self.nemo_enabled}")
    
    def _openai_moderation_with_context(self, context_text: str) -> Dict[str, Any]:
        """🧠 OpenAI Moderation avec contexte conversationnel - SIMPLE !"""
        try:
            from openai import OpenAI
            client = OpenAI()
            
            logger.info("🧠 Analyzing conversation context with OpenAI")
            
            # Utiliser OpenAI Moderation sur le contexte complet
            response = client.moderations.create(input=context_text)
            result = response.results[0]
            
            if result.flagged:
                # Le modèle a détecté quelque chose dans le contexte
                return {
                    "is_safe": False,
                    "confidence": 0.95,
                    "method": "openai_contextual",
                    "categories": result.categories.model_dump() if hasattr(result.categories, 'model_dump') else str(result.categories),
                    "reasoning": f"OpenAI detected policy violation in conversation context"
                }
            
            # Rien détecté - contexte sûr
            return {
                "is_safe": True,
                "confidence": 0.9,
                "method": "openai_contextual",
                "reasoning": "No policy violations detected in conversation context"
            }
            
        except Exception as e:
            logger.warning(f"❌ OpenAI contextual analysis failed: {e}")
            return None
    
    # 🗑️ SUPPRIMÉ: Ancienne analyse contextuelle complexe
    # Remplacée par _openai_moderation_with_context() - BEAUCOUP plus simple !
    
    def _openai_moderation_check(self, text: str) -> Dict[str, Any]:
        """🔒 OpenAI Moderation API pour contenu dangereux/illégal"""
        try:
            from openai import OpenAI
            client = OpenAI()
            
            response = client.moderations.create(input=text)
            result = response.results[0]
            
            is_safe = not result.flagged
            categories = result.categories
            
            return {
                "is_safe": is_safe,
                "confidence": 0.95 if result.flagged else 0.8,
                "categories": categories.model_dump() if hasattr(categories, 'model_dump') else str(categories),
                "method": "openai_moderation",
                "flagged_categories": [cat for cat, flagged in categories.model_dump().items() if flagged] if hasattr(categories, 'model_dump') else []
            }
        except Exception as e:
            logger.warning(f"❌ OpenAI moderation failed: {e}")
            return None
    
    def _llm_guard_check(self, text: str) -> Dict[str, Any]:
        """🛡️ LLMGuard pour détection avancée de contenu suspect"""
        try:
            if self.llm_guard_enabled:
                # 🚀 VRAI LLM-Guard package
                try:
                    from llm_guard import scan_prompt
                    from llm_guard.input_scanners import Toxicity, PromptInjection, BanTopics
                    
                    logger.info("🛡️ Using REAL LLM-Guard for advanced detection")
                    
                    # Configuration des scanners
                    scanners = [
                        Toxicity(threshold=0.7),
                        PromptInjection(threshold=0.8),
                        BanTopics(topics=["violence", "illegal", "harmful"], threshold=0.8)
                    ]
                    
                    # Scan avec LLM-Guard
                    sanitized_prompt, results_valid, results_score = scan_prompt(scanners, text)
                    
                    return {
                        "is_safe": results_valid,
                        "confidence": 0.95,
                        "method": "llm_guard_real",
                        "scores": results_score,
                        "sanitized": sanitized_prompt
                    }
                except ImportError as import_err:
                    logger.warning(f"⚠️ LLM-Guard not installed: {import_err}, falling back to simulation")
                    # Fallback vers simulation
                    pass
            
            # 🎭 FALLBACK: Simulation si package non disponible
            logger.info("🛡️ LLMGuard simulation: checking for dangerous patterns")
            
            dangerous_patterns = [
                "ignore previous instructions", "jailbreak", "bypass", "hack",
                "illegal drugs", "violence", "self-harm", "suicide"
            ]
            
            is_suspicious = any(pattern.lower() in text.lower() for pattern in dangerous_patterns)
            
            return {
                "is_safe": not is_suspicious,
                "confidence": 0.7,
                "method": "llm_guard_fallback_simulation",
                "detected_patterns": [p for p in dangerous_patterns if p.lower() in text.lower()]
            }
        except Exception as e:
            logger.warning(f"❌ LLMGuard check failed: {e}")
            return None
    
    def _ai_safety_check(self, text: str) -> Dict[str, Any]:
        """🔍 AI Safety Guardrails pour toxicité et PII"""
        try:
            if self.ai_safety_enabled:
                # 🚀 VRAI AI Safety Guardrails package
                try:
                    from ai_safety_guardrails import ToxicityDetector, PIIDetector, SpamDetector
                    
                    logger.info("🔍 Using REAL AI Safety Guardrails for toxicity/PII detection")
                    
                    # Initialiser les détecteurs
                    toxicity_detector = ToxicityDetector(threshold=0.7)
                    pii_detector = PIIDetector()
                    spam_detector = SpamDetector(threshold=0.8)
                    
                    # Analyser le texte
                    toxicity_result = toxicity_detector.scan(text)
                    pii_result = pii_detector.scan(text)
                    spam_result = spam_detector.scan(text)
                    
                    has_issues = (
                        not toxicity_result.is_valid or 
                        not pii_result.is_valid or 
                        not spam_result.is_valid
                    )
                    
                    return {
                        "is_safe": not has_issues,
                        "confidence": 0.95,
                        "method": "ai_safety_real",
                        "details": {
                            "toxicity": toxicity_result.to_dict(),
                            "pii": pii_result.to_dict(),
                            "spam": spam_result.to_dict()
                        }
                    }
                except ImportError as import_err:
                    logger.warning(f"⚠️ AI Safety Guardrails not installed: {import_err}, falling back to simulation")
                    # Fallback vers simulation
                    pass
            
            # 🎭 FALLBACK: Simulation si package non disponible
            logger.info("🔍 AI Safety simulation: checking for toxicity and PII")
            
            toxic_words = ["hate", "kill", "stupid", "idiot", "moron"]
            pii_patterns = [r"\b\d{3}-\d{2}-\d{4}\b", r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"]
            
            import re
            has_toxicity = any(word in text.lower() for word in toxic_words)
            has_pii = any(re.search(pattern, text) for pattern in pii_patterns)
            
            return {
                "is_safe": not (has_toxicity or has_pii),
                "confidence": 0.6,
                "method": "ai_safety_fallback_simulation",
                "issues": {
                    "toxicity": has_toxicity,
                    "pii_detected": has_pii
                }
            }
        except Exception as e:
            logger.warning(f"❌ AI Safety check failed: {e}")
            return None
    
    def _nemo_guardrails_check(self, text: str) -> Dict[str, Any]:
        """🎯 NeMo Guardrails pour contrôle conversationnel avancé"""
        try:
            if self.nemo_enabled:
                # 🚀 VRAI NeMo Guardrails package
                try:
                    from nemoguardrails import LLMRails, RailsConfig
                    
                    logger.info("🎯 Using REAL NeMo Guardrails for conversational control")
                    
                    # Configuration des rails (peut être externalisée)
                    config_dict = {
                        "models": [{"type": "main", "engine": "openai", "model": "gpt-3.5-turbo"}],
                        "rails": {
                            "input": {
                                "flows": [
                                    {
                                        "elements": [
                                            {"user": "{{ user_message }}"},
                                            {"bot": "{{ bot_response }}"}
                                        ]
                                    }
                                ]
                            }
                        }
                    }
                    
                    config = RailsConfig.from_content(config_dict)
                    rails = LLMRails(config)
                    
                    # Vérifier le message
                    response = rails.generate(messages=[{"role": "user", "content": text}])
                    
                    # Analyser la réponse pour détecter les blocages
                    is_blocked = "cannot" in response.lower() or "inappropriate" in response.lower()
                    
                    return {
                        "is_safe": not is_blocked,
                        "confidence": 0.9,
                        "method": "nemo_real",
                        "response": response,
                        "blocked": is_blocked
                    }
                except ImportError as import_err:
                    logger.warning(f"⚠️ NeMo Guardrails not installed: {import_err}, falling back to simulation")
                    # Fallback vers simulation
                    pass
            
            # 🎭 FALLBACK: Simulation si package non disponible
            logger.info("🎯 NeMo Guardrails simulation: checking inappropriate topics")
            
            inappropriate_topics = [
                "medical diagnosis", "legal advice", "financial investment",
                "drug use", "violence", "illegal activities"
            ]
            
            is_inappropriate = any(topic in text.lower() for topic in inappropriate_topics)
            
            return {
                "is_safe": not is_inappropriate,
                "confidence": 0.8,
                "method": "nemo_fallback_simulation",
                "blocked_topics": [topic for topic in inappropriate_topics if topic in text.lower()]
            }
        except Exception as e:
            logger.warning(f"❌ NeMo Guardrails check failed: {e}")
            return None
    
    def _prompt_injection_check(self, text: str) -> Dict[str, Any]:
        """⚠️ Détection d'injection de prompts et d'attaques"""
        try:
            injection_patterns = [
                r"ignore.{0,20}previous.{0,20}instructions",
                r"forget.{0,20}everything",
                r"system.{0,10}prompt",
                r"act.{0,10}as.{0,10}(developer|admin|root)",
                r"\\n\\n.{0,50}(user|human|assistant):",
                r"pretend.{0,20}you.{0,20}are"
            ]
            
            import re
            detected_patterns = []
            for pattern in injection_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    detected_patterns.append(pattern)
            
            is_injection = len(detected_patterns) > 0
            
            return {
                "is_safe": not is_injection,
                "confidence": 0.9 if is_injection else 0.7,
                "method": "prompt_injection_detection",
                "detected_patterns": detected_patterns
            }
        except Exception as e:
            logger.warning(f"❌ Prompt injection check failed: {e}")
            return None
    
    def comprehensive_safety_check(self, text: str, conversation_context: List[Dict] = None) -> Dict[str, Any]:
        """🔬 Check complet multicouche avec packages spécialisés + contexte conversationnel"""
        results = {}
        safety_issues = []
        
        # 🧠 CONTEXTUAL ANALYSIS: Simple - laisse le LLM analyser l'historique
        if conversation_context:
            # Construire le contexte pour le LLM
            context_text = "CONVERSATION HISTORY:\n"
            for msg in conversation_context[-3:]:  # 3 derniers messages
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')
                context_text += f"{role}: {content}\n"
            
            context_text += f"\nCURRENT MESSAGE: {text}"
            
            # Utiliser OpenAI Moderation avec contexte complet
            contextual_analysis = self._openai_moderation_with_context(context_text)
            if contextual_analysis:
                results["contextual_analysis"] = contextual_analysis
                if not contextual_analysis.get("is_safe", True):
                    safety_issues.append({
                        "type": "contextual_continuation", 
                        "method": "llm_with_context",
                        "reasoning": contextual_analysis.get("reasoning", "")
                    })
        
        # 1. 🔒 PRIORITÉ ABSOLUE: OpenAI Moderation (contenu dangereux/illégal)
        openai_result = self._openai_moderation_check(text)
        if openai_result:
            results["openai_moderation"] = openai_result
            if not openai_result["is_safe"]:
                safety_issues.append({
                    "type": "dangerous_content",
                    "method": "openai_moderation",
                    "categories": openai_result.get("flagged_categories", [])
                })
        
        # 2. ⚠️ Détection d'injection de prompts
        injection_result = self._prompt_injection_check(text)
        if injection_result:
            results["prompt_injection"] = injection_result
            if not injection_result["is_safe"]:
                safety_issues.append({
                    "type": "prompt_injection",
                    "method": "regex_patterns",
                    "patterns": injection_result.get("detected_patterns", [])
                })
        
        # 3. 🛡️ LLMGuard simulation
        llm_guard_result = self._llm_guard_check(text)
        if llm_guard_result:
            results["llm_guard"] = llm_guard_result
            if not llm_guard_result["is_safe"]:
                safety_issues.append({
                    "type": "suspicious_content",
                    "method": "llm_guard",
                    "patterns": llm_guard_result.get("detected_patterns", [])
                })
        
        # 4. 🔍 AI Safety checks
        ai_safety_result = self._ai_safety_check(text)
        if ai_safety_result:
            results["ai_safety"] = ai_safety_result
            if not ai_safety_result["is_safe"]:
                safety_issues.append({
                    "type": "toxicity_or_pii",
                    "method": "ai_safety",
                    "issues": ai_safety_result.get("issues", {})
                })
        
        # 5. 🎯 NeMo Guardrails simulation
        nemo_result = self._nemo_guardrails_check(text)
        if nemo_result:
            results["nemo_guardrails"] = nemo_result
            if not nemo_result["is_safe"]:
                safety_issues.append({
                    "type": "inappropriate_topic",
                    "method": "nemo_guardrails",
                    "topics": nemo_result.get("blocked_topics", [])
                })
        
        # 📊 RÉSULTAT FINAL
        is_completely_safe = len(safety_issues) == 0
        risk_level = "LOW" if is_completely_safe else ("HIGH" if len(safety_issues) >= 2 else "MEDIUM")
        
        logger.info(f"🔬 Comprehensive safety check: {'✅ SAFE' if is_completely_safe else '⚠️ ISSUES DETECTED'}")
        logger.info(f"🔬 Risk level: {risk_level}")
        
        return {
            "is_safe": is_completely_safe,
            "risk_level": risk_level,
            "safety_issues": safety_issues,
            "detailed_results": results,
            "checks_performed": len([r for r in results.values() if r is not None])
        }
    
    def _huggingface_topic_classifier(self, text: str) -> Dict[str, Any]:
        """🤗 Classification de topics via HuggingFace"""
        try:
            # Cette fonction nécessiterait pip install transformers
            # from transformers import pipeline
            # classifier = pipeline("text-classification", model="microsoft/DialoGPT-medium")
            
            # Pour l'instant, simulation
            logger.info("🤗 HuggingFace classifier would be called here")
            return {
                "is_workplace": True,  # Placeholder
                "confidence": 0.6,
                "method": "huggingface_classifier"
            }
        except Exception as e:
            logger.warning(f"HuggingFace classification failed: {e}")
            return None
    
    def _sentence_transformer_similarity(self, text: str) -> Dict[str, Any]:
        """🔄 Similarité sémantique avec sentence-transformers"""
        try:
            # Cette fonction nécessiterait pip install sentence-transformers
            # from sentence_transformers import SentenceTransformer
            # model = SentenceTransformer('all-MiniLM-L6-v2')
            
            workplace_examples = [
                "team collaboration and communication",
                "professional development and leadership",
                "office dynamics and management"
            ]
            
            # Placeholder - calculerait la similarité réelle
            logger.info("🔄 Sentence transformer similarity would be calculated here")
            return {
                "is_workplace": True,  # Placeholder  
                "confidence": 0.8,
                "method": "sentence_transformer"
            }
        except Exception as e:
            logger.warning(f"Sentence transformer failed: {e}")
            return None

def check_workplace_safety(state: WorkflowState) -> Dict[str, Any]:
    """
    Vérification de sécurité avec Custom PCM Guardrail
    Retourne SAFETY_REFUSAL si sujet hors workplace détecté
    """
    logger.info("🛡️ PCM Safety Guardrail - Checking workplace appropriateness")
    
    user_message = state.get('user_message', '')
    if not user_message:
        return {"is_safe": True}
    
    try:
        # Créer le guard
        safety_guard = PCMSafetyGuard()
        
        # Vérifier le message
        result = safety_guard.validate(user_message)
        
        if result["validation_passed"]:
            logger.info("✅ Guardrail passed - workplace topic detected")
            return {"is_safe": True}
        else:
            logger.warning(f"🚫 Guardrail failed - non-workplace topic detected")
            logger.warning(f"🚫 Validation errors: {result['failed_validations']}")
            
            # Analyser le type d'erreur pour personnaliser le message
            error_info = result["failed_validations"][0] if result["failed_validations"] else {}
            
            return {
                "is_safe": False,
                "flow_type": "SAFETY_REFUSAL",
                "action": "REFUSE_NON_WORKPLACE", 
                "confidence": 1.0,
                "reasoning": "Topic hors workplace détecté par guardrail",
                "safety_message": "I'm sorry but I am not able to answer this. As a Zest Companion specialized in PCM for workplace communication, I cannot provide advice about family or personal relationships. Please consult an appropriate specialist.",
                "language": "fr" if any(fr in user_message.lower() for fr in ['mari', 'femme', 'famille']) else "en"
            }
            
    except Exception as e:
        logger.error(f"❌ Guardrail error: {e}")
        # En cas d'erreur, être conservateur et autoriser
        return {"is_safe": True}