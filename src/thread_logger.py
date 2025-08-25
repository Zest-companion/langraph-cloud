"""
Thread-based logging system for ZEST MBTI Workflow
Crée un fichier de log unique par thread_id pour faciliter le debug
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

class ThreadLogger:
    """Logger qui crée un fichier par thread_id"""
    
    def __init__(self, log_dir: str = "logs/threads"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.loggers: Dict[str, logging.Logger] = {}
        self.thread_data: Dict[str, Dict] = {}
    
    def get_logger(self, thread_id: str) -> logging.Logger:
        """Récupère ou crée un logger pour un thread_id"""
        if thread_id not in self.loggers:
            # Créer le logger
            logger = logging.getLogger(f"thread_{thread_id}")
            logger.setLevel(logging.DEBUG)
            
            # Éviter les doublons de handlers
            if not logger.handlers:
                # Créer le fichier de log pour ce thread
                log_file = self.log_dir / f"thread_{thread_id[:8]}.log"
                
                # Handler pour fichier
                file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
                file_handler.setLevel(logging.DEBUG)
                
                # Format détaillé
                formatter = logging.Formatter(
                    '%(asctime)s | %(levelname)s | %(message)s',
                    datefmt='%H:%M:%S'
                )
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
                
                # Log initial
                logger.info(f"🟢 NEW THREAD STARTED: {thread_id}")
                logger.info(f"📁 Log file: {log_file}")
            
            self.loggers[thread_id] = logger
            self.thread_data[thread_id] = {
                "started_at": datetime.now().isoformat(),
                "messages_count": 0,
                "nodes_executed": []
            }
        
        return self.loggers[thread_id]
    
    def log_node_start(self, thread_id: str, node_name: str, state_data: Dict[str, Any]):
        """Log le début d'exécution d'un node"""
        logger = self.get_logger(thread_id)
        
        # Extraire les données importantes du state
        user_msg = state_data.get('user_message', '')
        user_name = state_data.get('user_name', '')
        messages_count = len(state_data.get('messages', []))
        
        logger.info(f"🔵 NODE START: {node_name}")
        logger.info(f"   👤 User: {user_name}")
        logger.info(f"   💬 Message: {user_msg[:100]}...")
        logger.info(f"   📊 Messages in history: {messages_count}")
        
        # Mettre à jour les stats
        if thread_id in self.thread_data:
            self.thread_data[thread_id]["nodes_executed"].append({
                "node": node_name,
                "timestamp": datetime.now().isoformat(),
                "messages_count": messages_count
            })
    
    def log_node_result(self, thread_id: str, node_name: str, result_data: Dict[str, Any]):
        """Log le résultat d'un node"""
        logger = self.get_logger(thread_id)
        
        logger.info(f"✅ NODE COMPLETE: {node_name}")
        
        # Log spécifique selon le node
        if node_name == "fetch_user_profile":
            mbti = result_data.get('user_mbti')
            temperament = result_data.get('user_temperament')
            logger.info(f"   🎯 MBTI: {mbti}, Temperament: {temperament}")
            
        elif node_name == "fetch_temperament_description":
            desc = result_data.get('temperament_description', '')
            logger.info(f"   📝 Description length: {len(desc)} chars")
            
        elif node_name == "mbti_expert_analysis":
            analysis = result_data.get('mbti_analysis', {})
            instructions = analysis.get('instructions', '')[:100]
            other_profiles = analysis.get('other_mbti_profiles')
            logger.info(f"   🧠 Instructions: {instructions}...")
            logger.info(f"   👥 Other profiles: {other_profiles}")
            
        elif node_name == "generate_final_response":
            response = result_data.get('final_response', '')
            logger.info(f"   💡 Response length: {len(response)} chars")
            logger.info(f"   📄 Response preview: {response[:150]}...")
    
    def log_error(self, thread_id: str, node_name: str, error: Exception):
        """Log une erreur"""
        logger = self.get_logger(thread_id)
        logger.error(f"❌ ERROR in {node_name}: {type(error).__name__}: {str(error)}")
    
    def log_routing_decision(self, thread_id: str, decision: str, analysis: Dict[str, Any]):
        """Log la décision de routage"""
        logger = self.get_logger(thread_id)
        instructions = analysis.get('instructions', '')[:100]
        logger.info(f"🔀 ROUTING DECISION: {decision}")
        logger.info(f"   📋 Based on: {instructions}...")
    
    def log_conversation_summary(self, thread_id: str):
        """Log un résumé de la conversation à la fin"""
        if thread_id not in self.thread_data:
            return
            
        logger = self.get_logger(thread_id)
        data = self.thread_data[thread_id]
        
        logger.info("🏁 CONVERSATION SUMMARY:")
        logger.info(f"   ⏱️  Duration: {datetime.now().isoformat()}")
        logger.info(f"   🔢 Messages processed: {data['messages_count']}")
        logger.info(f"   ⚙️  Nodes executed: {len(data['nodes_executed'])}")
        
        for node_exec in data['nodes_executed']:
            logger.info(f"      → {node_exec['node']} (messages: {node_exec['messages_count']})")

# Instance globale
thread_logger = ThreadLogger()

def log_workflow_state(thread_id: str, node_name: str, state: Dict[str, Any], stage: str = "start"):
    """Helper function pour logger facilement depuis le workflow"""
    if stage == "start":
        thread_logger.log_node_start(thread_id, node_name, state)
    elif stage == "result":
        thread_logger.log_node_result(thread_id, node_name, state)
    elif stage == "routing":
        analysis = state.get('mbti_analysis', {})
        decision = "unknown"  # sera déterminé par le router
        thread_logger.log_routing_decision(thread_id, decision, analysis)