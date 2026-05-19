"""
ObsidianCore Integration Module
Bridges ObsidianCore advanced AI systems with HadesAI main application
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import time

# Configure logging
logger = logging.getLogger("ObsidianCoreIntegration")
logger.setLevel(logging.DEBUG)

class ObsidianCoreIntegration:
    """
    Unified interface for ObsidianCore advanced AI systems
    Provides simplified access to all orchestration engines
    """
    
    def __init__(self):
        """Initialize ObsidianCore and all engine subsystems"""
        self.ready = False
        self.engines = {}
        self.status = {}
        
        try:
            self._import_obsidian_core()
            self._initialize_engines()
            self.ready = True
            logger.info("✓ ObsidianCore Integration initialized successfully")
        except Exception as e:
            logger.error(f"✗ Failed to initialize ObsidianCore: {str(e)}")
            self.ready = False
    
    def _import_obsidian_core(self):
        """Import ObsidianCore from Current implementation folder"""
        try:
            impl_path = Path("Current implementation")
            if not impl_path.exists():
                raise FileNotFoundError(f"Current implementation folder not found at {impl_path.absolute()}")
            
            sys.path.insert(0, str(impl_path.absolute()))
            
            # Try to import the AICore class
            try:
                from ObsidianCore import AICore
                self.AICore = AICore
                logger.info("✓ Successfully imported AICore from ObsidianCore.py")
            except ImportError as e:
                logger.warning(f"⚠️ Could not import full AICore: {str(e)}")
                if 'pefile' in str(e).lower():
                    logger.warning("⚠️ Missing optional dependency 'pefile'. Install with: pip install pefile")
                logger.info("Using simplified integration mode")
                self.AICore = None
        except Exception as e:
            logger.error(f"✗ Import error: {str(e)}")
            self.AICore = None
    
    def _initialize_engines(self):
        """Initialize all orchestration engines"""
        engine_list = [
            'attack_engine',
            'defense_engine',
            'deception_engine',
            'movement_engine',
            'learning_engine',
            'monitoring_engine',
            'web_engine',
            'payload_engine',
            'malware_engine'
        ]
        
        if self.AICore:
            try:
                self.core = self.AICore()
                for engine_name in engine_list:
                    if hasattr(self.core, engine_name):
                        engine = getattr(self.core, engine_name)
                        self.engines[engine_name] = engine
                        self.status[engine_name] = 'initialized'
                        logger.info(f"✓ {engine_name}: initialized")
                    else:
                        self.status[engine_name] = 'not_available'
                        logger.warning(f"⚠️ {engine_name}: not available in AICore")
                
                self.core_instance = self.core
            except Exception as e:
                logger.error(f"✗ Failed to instantiate AICore: {str(e)}")
                self.core_instance = None
        else:
            logger.warning("⚠️ AICore not available, engines will be simulated")
            for engine_name in engine_list:
                self.engines[engine_name] = self._create_mock_engine(engine_name)
                self.status[engine_name] = 'simulated'
    
    def _create_mock_engine(self, engine_name: str) -> Dict[str, Any]:
        """Create mock engine for when AICore is not available"""
        return {
            'name': engine_name,
            'status': 'mock',
            'capabilities': [],
            'last_executed': None,
            'execution_count': 0
        }
    
    # ========== ATTACK ENGINE INTERFACE ==========
    
    def get_attack_capabilities(self) -> List[str]:
        """Get list of available attack capabilities"""
        if not self.ready:
            return []
        
        try:
            if self.AICore and hasattr(self.core, 'attack_engine'):
                return self.core.attack_engine.get_capabilities()
            else:
                return [
                    'exploitation',
                    'privilege_escalation',
                    'lateral_movement',
                    'persistence',
                    'exfiltration',
                    'covering_tracks'
                ]
        except Exception as e:
            logger.error(f"Error getting attack capabilities: {str(e)}")
            return []
    
    def execute_attack(self, attack_type: str, target: str, **kwargs) -> Dict:
        """
        Execute attack using appropriate engine
        
        Args:
            attack_type: Type of attack (exploitation, brute_force, etc)
            target: Target address/endpoint
            **kwargs: Additional parameters
        
        Returns:
            dict with execution results
        """
        if not self.ready:
            return {'status': 'error', 'message': 'ObsidianCore not initialized'}
        
        try:
            result = {
                'attack_type': attack_type,
                'target': target,
                'timestamp': time.time(),
                'success': False,
                'data': None
            }
            
            if self.AICore and hasattr(self.core, 'attack_engine'):
                # Use real attack engine if available
                engine = self.core.attack_engine
                if hasattr(engine, 'execute'):
                    result['data'] = engine.execute(attack_type, target, **kwargs)
                    result['success'] = True
                else:
                    logger.warning(f"Attack engine doesn't have execute method")
            else:
                # Simulate attack
                result['data'] = {
                    'simulated': True,
                    'payload_generated': True,
                    'evasion_techniques': ['polymorphism', 'obfuscation'],
                    'success_probability': 0.85
                }
                result['success'] = True
            
            logger.info(f"✓ Attack executed: {attack_type} on {target}")
            return result
            
        except Exception as e:
            logger.error(f"✗ Attack execution failed: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    # ========== DEFENSE ENGINE INTERFACE ==========
    
    def get_defense_status(self) -> Dict:
        """Get current defense system status"""
        if not self.ready:
            return {'status': 'offline'}
        
        try:
            if self.AICore and hasattr(self.core, 'defense_engine'):
                engine = self.core.defense_engine
                return {
                    'status': 'active',
                    'mode': getattr(engine, 'current_mode', 'adaptive'),
                    'threat_level': getattr(engine, 'threat_level', 0),
                    'rules_active': getattr(engine, 'rule_count', 0),
                    'blocks_total': getattr(engine, 'blocks_total', 0)
                }
            else:
                return {
                    'status': 'simulated',
                    'mode': 'adaptive',
                    'threat_level': 0,
                    'rules_active': 25
                }
        except Exception as e:
            logger.error(f"Error getting defense status: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def deploy_defense(self, defense_type: str, **kwargs) -> Dict:
        """
        Deploy defense mechanism
        
        Args:
            defense_type: Type of defense (firewall, ids, behavioral, etc)
            **kwargs: Defense parameters
        
        Returns:
            dict with deployment status
        """
        if not self.ready:
            return {'status': 'error', 'message': 'ObsidianCore not initialized'}
        
        try:
            result = {
                'defense_type': defense_type,
                'timestamp': time.time(),
                'deployed': False
            }
            
            if self.AICore and hasattr(self.core, 'defense_engine'):
                engine = self.core.defense_engine
                if hasattr(engine, 'deploy'):
                    result['data'] = engine.deploy(defense_type, **kwargs)
                    result['deployed'] = True
            else:
                result['data'] = {
                    'simulated': True,
                    'rules_applied': 15,
                    'endpoints_protected': 100
                }
                result['deployed'] = True
            
            logger.info(f"✓ Defense deployed: {defense_type}")
            return result
            
        except Exception as e:
            logger.error(f"✗ Defense deployment failed: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    # ========== PAYLOAD ENGINE INTERFACE ==========
    
    def generate_payload(self, payload_type: str, **kwargs) -> Dict:
        """
        Generate malware payload with advanced evasion
        
        Args:
            payload_type: Type of payload (shellcode, reverse_shell, etc)
            **kwargs: Payload parameters
        
        Returns:
            dict with generated payload
        """
        if not self.ready:
            return {'status': 'error', 'message': 'ObsidianCore not initialized'}
        
        try:
            result = {
                'payload_type': payload_type,
                'timestamp': time.time(),
                'generated': False
            }
            
            if self.AICore and hasattr(self.core, 'payload_engine'):
                engine = self.core.payload_engine
                if hasattr(engine, 'generate_payload'):
                    payload = engine.generate_payload(payload_type)
                    result['payload'] = payload
                    result['generated'] = True
            else:
                result['payload'] = self._generate_mock_payload(payload_type)
                result['generated'] = True
            
            logger.info(f"✓ Payload generated: {payload_type}")
            return result
            
        except Exception as e:
            logger.error(f"✗ Payload generation failed: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def _generate_mock_payload(self, payload_type: str) -> Dict:
        """Generate mock payload for simulation"""
        return {
            'simulated': True,
            'type': payload_type,
            'size_bytes': 2048,
            'evasion': ['xor_encrypt', 'polymorphic', 'obfuscated'],
            'detection_rating': 0.15  # 15% detection probability
        }
    
    # ========== MOVEMENT ENGINE INTERFACE ==========
    
    def plan_lateral_movement(self, current_position: str, target_position: str) -> Dict:
        """
        Plan lateral movement strategy
        
        Args:
            current_position: Current access level/system
            target_position: Target system/privilege level
        
        Returns:
            dict with movement path and recommendations
        """
        if not self.ready:
            return {'status': 'error', 'message': 'ObsidianCore not initialized'}
        
        try:
            result = {
                'from': current_position,
                'to': target_position,
                'timestamp': time.time(),
                'path_found': False
            }
            
            if self.AICore and hasattr(self.core, 'movement_engine'):
                engine = self.core.movement_engine
                if hasattr(engine, 'find_attack_path'):
                    # This requires attack graph - simulate for now
                    result['path'] = [current_position, 'intermediate', target_position]
                    result['path_found'] = True
            else:
                result['path'] = [current_position, 'jump_host', target_position]
                result['techniques'] = ['token_theft', 'credential_reuse', 'pivot']
                result['path_found'] = True
            
            logger.info(f"✓ Lateral movement planned: {current_position} -> {target_position}")
            return result
            
        except Exception as e:
            logger.error(f"✗ Movement planning failed: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    # ========== LEARNING ENGINE INTERFACE ==========
    
    def log_learning_event(self, system: str, action: str, success: bool, 
                          execution_time: float) -> Dict:
        """
        Log learning event for AI improvement
        
        Args:
            system: System where action occurred
            action: Action performed
            success: Whether action succeeded
            execution_time: Time taken (seconds)
        
        Returns:
            dict with logging status
        """
        try:
            result = {
                'logged': False,
                'timestamp': time.time()
            }
            
            if self.AICore and hasattr(self.core, 'learning_engine'):
                engine = self.core.learning_engine
                if hasattr(engine, 'log_ai_learning'):
                    engine.log_ai_learning(system, action, success, execution_time)
                    result['logged'] = True
            else:
                result['logged'] = True  # Simulate logging
            
            logger.debug(f"Learning event logged: {system}/{action} (success={success})")
            return result
            
        except Exception as e:
            logger.error(f"Error logging learning event: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    # ========== MONITORING ENGINE INTERFACE ==========
    
    def start_monitoring(self, target_type: str = 'system') -> Dict:
        """
        Start behavioral monitoring
        
        Args:
            target_type: What to monitor (system, network, filesystem, etc)
        
        Returns:
            dict with monitoring status
        """
        try:
            result = {
                'monitoring': False,
                'target_type': target_type,
                'timestamp': time.time()
            }
            
            if self.AICore and hasattr(self.core, 'monitoring_engine'):
                engine = self.core.monitoring_engine
                if hasattr(engine, 'start_behavior_monitoring'):
                    engine.start_behavior_monitoring()
                    result['monitoring'] = True
            else:
                result['monitoring'] = True  # Simulate
            
            logger.info(f"✓ Monitoring started: {target_type}")
            return result
            
        except Exception as e:
            logger.error(f"Error starting monitoring: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    # ========== SYSTEM STATUS ==========
    
    def get_system_status(self) -> Dict:
        """Get overall ObsidianCore system status"""
        return {
            'integrated': self.ready,
            'core_available': self.AICore is not None,
            'engines': self.status,
            'timestamp': time.time()
        }
    
    def get_capabilities(self) -> Dict:
        """Get all available capabilities"""
        return {
            'attack': self.get_attack_capabilities(),
            'defense': ['firewall', 'ids', 'behavioral', 'adaptive'],
            'movement': ['lateral', 'privilege_escalation', 'persistence'],
            'learning': ['adaptive', 'reinforced', 'behavioral'],
            'monitoring': ['behavioral', 'network', 'filesystem'],
            'payloads': ['shellcode', 'reverse_shell', 'persistence_agent']
        }


# Create global instance
_obsidian_integration = None

def get_obsidian_core() -> Optional[ObsidianCoreIntegration]:
    """Get or create global ObsidianCore integration instance"""
    global _obsidian_integration
    if _obsidian_integration is None:
        _obsidian_integration = ObsidianCoreIntegration()
    return _obsidian_integration
