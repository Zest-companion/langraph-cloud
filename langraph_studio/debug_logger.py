"""
Debug Logger - Sauvegarde tous les logs par thread/question
"""
import os
import json
from datetime import datetime
from typing import Any, Dict
import logging

class ThreadLogger:
    def __init__(self, base_dir: str = "debug_logs"):
        self.base_dir = base_dir
        self._thread_files = {}  # Cache des fichiers par thread
        os.makedirs(base_dir, exist_ok=True)
        
    def get_log_file(self, thread_id: str) -> str:
        """Retourne le chemin du fichier de log pour un thread (un seul fichier par thread)"""
        if thread_id not in self._thread_files:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"thread_{thread_id}_{timestamp}.log"
            self._thread_files[thread_id] = os.path.join(self.base_dir, filename)
        return self._thread_files[thread_id]
    
    def log_state(self, thread_id: str, step: str, state: Dict[str, Any], extra_info: str = ""):
        """Log l'état complet à une étape donnée"""
        log_file = self.get_log_file(thread_id)
        
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"STEP: {step}\n")
            f.write(f"TIME: {datetime.now().isoformat()}\n")
            if extra_info:
                f.write(f"INFO: {extra_info}\n")
            f.write(f"{'='*80}\n")
            
            # Log état principal
            for key, value in state.items():
                f.write(f"\n[{key}]:\n")
                if isinstance(value, (dict, list)):
                    f.write(json.dumps(value, indent=2, ensure_ascii=False))
                else:
                    f.write(str(value))
                f.write("\n")
    
    def log_message(self, thread_id: str, level: str, message: str, data: Any = None):
        """Log un message simple avec données optionnelles"""
        log_file = self.get_log_file(thread_id)
        
        with open(log_file, 'a', encoding='utf-8') as f:
            timestamp = datetime.now().strftime("%H:%M:%S")
            f.write(f"[{timestamp}] {level.upper()}: {message}\n")
            
            if data:
                if isinstance(data, (dict, list)):
                    f.write(json.dumps(data, indent=2, ensure_ascii=False))
                else:
                    f.write(str(data))
                f.write("\n")
    
    def log_error(self, thread_id: str, error: Exception, context: str = ""):
        """Log une erreur avec contexte"""
        self.log_message(
            thread_id, 
            "ERROR", 
            f"{context}: {type(error).__name__}: {str(error)}"
        )

# Instance globale
thread_logger = ThreadLogger()

def log_pcm_debug(thread_id: str, step: str, state: Dict[str, Any], extra_info: str = ""):
    """Helper pour logger les étapes PCM"""
    thread_logger.log_state(thread_id, f"PCM_{step}", state, extra_info)

def log_pcm_message(thread_id: str, message: str, data: Any = None):
    """Helper pour logger des messages PCM"""
    thread_logger.log_message(thread_id, "PCM", message, data)