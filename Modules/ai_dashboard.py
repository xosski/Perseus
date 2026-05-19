# ai_dashboard.py
"""
SingularityDrive Dashboard
Real-time monitoring interface for AI Engine operations
Powered by HadesAI - Self-Learning Pentesting AI
"""

from flask import Flask, render_template
from flask_socketio import SocketIO
import random
import threading
import time
import re
import psutil
import numpy as np
import requests
import sys
import asyncio
import subprocess

# Import HadesAI for chat functionality
try:
    from HadesAI import HadesAI
    HADES_AVAILABLE = True
except ImportError:
    HADES_AVAILABLE = False
    print("Warning: HadesAI not available. Chat functionality will be limited.")

app = Flask(__name__)
socketio = SocketIO(app)

# Initialize HadesAI as the main AI engine
hades_ai = HadesAI() if HADES_AVAILABLE else None

MONITORED_FILES = ["system.log", "access.log", "attack.log", "defense.log"]

class DashboardMonitor:
    def __init__(self):
        self.active_connections = set()
        self.attack_history = []
        self.system_metrics = {}
        self.api_key = self.fetch_current_api_key()
        self.command_queue = asyncio.Queue()
    @socketio.on('operator_input')
    def handle_operator_input(self, message):
        """Process operator commands and generate responses"""
        parsed = self.parse_intent(message)
        response = self.ai_reply(message, parsed)
        
        if parsed["action"] in ["attack", "pivot", "exfil"]:
            response += "\n⚠️ This action requires confirmation. Type 'confirm' to proceed."
            self.operation_state["pending_confirmation"] = parsed
        else:
            result = self.execute_intent(parsed)
            response += "\n" + result
        
        return response
    def operation_state(self):
        """Maintains current operational state and context"""
        return {
            'mode': 'offensive',  # Current operation mode
            'active_target': None,  # Current target if any
            'last_command': None,  # Last executed command
            'pending_confirmation': None,  # Awaiting confirmation
            'history': [],  # Command history
            'metrics': {
                'successful_operations': 0,
                'failed_operations': 0,
                'current_efficiency': 100
            }
        }
    def parse_intent(self, message):
        """Parse operator commands into structured intents"""
        intent_patterns = {
            'attack': r'attack|exploit|breach',
            'defend': r'defend|protect|shield',
            'analyze': r'analyze|scan|check',
            'monitor': r'monitor|watch|track'
        }
        
        parsed_intent = {
            'action': 'unknown',
            'target': 'none',
            'parameters': {}
        }
        
        for action, pattern in intent_patterns.items():
            if re.search(pattern, message.lower()):
                parsed_intent['action'] = action
                break
                
        return parsed_intent

    def ai_reply(self, message, parsed_intent):
        """Generate AI response to operator command"""
        responses = {
            'attack': f"Received '{message}'. Initiating attack sequence with {parsed_intent['target']}",
            'defend': f"Command '{message}' acknowledged. Activating defense protocols",
            'analyze': f"Processing request '{message}'. Running analysis on {parsed_intent['target']}",
            'monitor': f"Executing '{message}'. Beginning monitoring operation"
        }
        
        return responses.get(parsed_intent['action'], f"Command '{message}' understood, proceeding with operation")

    def execute_intent(self, intent):
        """Execute parsed operator intent"""
        execution_results = {
            'attack': self.launch_attack(intent['target']),
            'defend': self.activate_defenses(),
            'analyze': self.analyze_target(intent['target']),
            'monitor': self.start_monitoring()
        }
        
        return execution_results.get(intent['action'], "Operation executed successfully")
    def emit(event, data):
        """Emit events to connected clients"""
        socketio.emit(event, data)
    def on_operator_input(self, message):
        response = self.handle_operator_input(message)
        self.emit('ai_response', response)

    def fetch_current_api_key(self):
        """Fetch API key with fallback for development"""
        try:
            response = requests.get('http://localhost:8000/current-api-key')
            return response.json()['api_key']
        except:
            return "default_api_key_123"

    def validate_connection(self, key):
        return key == self.api_key
    def calculate_attack_velocity(self):
        if len(self.attack_history) < 2:
            return 0
        
        recent_attacks = self.attack_history[-10:]
        time_diff = recent_attacks[-1]['timestamp'] - recent_attacks[0]['timestamp']
        return len(recent_attacks) / time_diff if time_diff > 0 else 0

    def get_defense_rating(self):
        return {
            'overall_rating': random.uniform(0.7, 0.99),
            'shield_strength': random.uniform(85, 100),
            'countermeasure_effectiveness': random.uniform(0.8, 0.95)
        }

    def get_metrics(self):
        return {
            'cpu_usage': psutil.cpu_percent(),
            'memory_usage': psutil.virtual_memory().percent,
            'operational_efficiency': self.calculate_operational_efficiency(),
            'system_load': self.get_system_load(),
            'attack_velocity': self.calculate_attack_velocity(),
            'defense_rating': self.get_defense_rating()
        }
    def start_monitoring(self):
        thread = threading.Thread(target=self.monitor_ai_decisions)
        thread.daemon = True
        thread.start()
    async def execute_command(self, command_data):
        command_type = command_data.get('type')
        if command_type == 'attack':
            target = command_data.get('target')
            return await self.launch_attack(target)
        elif command_type == 'defend':
            return await self.activate_defenses()
        elif command_type == 'analyze':
            return await self.analyze_target(command_data.get('target'))
            
    async def launch_attack(self, target):
        attack_config = {
            'target': target,
            'timestamp': time.time(),
            'attack_vectors': self.generate_attack_vectors()
        }
        self.attack_history.append(attack_config)
        return {'status': 'attack_launched', 'config': attack_config}
        
    def generate_attack_vectors(self):
        return [
            {'type': 'exploit', 'success_rate': random.uniform(0.7, 0.95)},
            {'type': 'backdoor', 'success_rate': random.uniform(0.6, 0.9)},
            {'type': 'payload', 'success_rate': random.uniform(0.8, 1.0)}
        ]
    async def activate_defenses(self):
        defense_layers = {
            'firewall': self.deploy_adaptive_firewall(),
            'honeypots': self.spawn_decoy_systems(),
            'countermeasures': self.engage_active_defense()
        }
        
        defense_status = {
            'status': 'shields_up',
            'active_layers': defense_layers,
            'coverage': random.uniform(85, 100),
            'timestamp': time.time()
        }
        
        return defense_status
        
    def deploy_adaptive_firewall(self):
        return {
            'rules_updated': time.time(),
            'blocked_ips': [f'192.168.1.{random.randint(1,255)}' for _ in range(5)],
            'effectiveness': random.uniform(0.8, 0.99)
        }
        
    def spawn_decoy_systems(self):
        return {
            'active_honeypots': random.randint(3, 8),
            'trapped_attackers': random.randint(0, 5),
            'deception_level': random.uniform(0.7, 0.95)
        }
        
    def engage_active_defense(self):
        return {
            'countermeasures': ['packet_reflection', 'syn_flood_protection', 'zero_day_shield'],
            'response_time_ms': random.randint(10, 50)
        }

    async def analyze_target(self, target):
        scan_results = {
            'target_ip': target,
            'open_ports': self.scan_ports(target),
            'vulnerabilities': self.detect_vulnerabilities(),
            'attack_surface': self.calculate_attack_surface(),
            'recommended_vectors': self.suggest_attack_vectors()
        }
        
        return scan_results
        
    def scan_ports(self, target):
        return [
            {'port': port, 'service': f'service_{port}', 'status': 'open'}
            for port in random.sample(range(1, 1000), 5)
        ]
        
    def detect_vulnerabilities(self):
        vuln_types = ['buffer_overflow', 'sql_injection', 'rce', 'privilege_escalation']
        return [
            {
                'type': random.choice(vuln_types),
                'severity': random.uniform(0.5, 1.0),
                'exploitability': random.uniform(0.3, 0.9)
            }
            for _ in range(random.randint(2, 5))
        ]
        
    def calculate_attack_surface(self):
        return {
            'exposure_level': random.uniform(0.1, 0.9),
            'attack_paths': random.randint(1, 5),
            'critical_assets': random.randint(2, 8)
        }
        
    def suggest_attack_vectors(self):
        return [
            {
                'vector': v,
                'success_probability': random.uniform(0.6, 0.95),
                'stealth_rating': random.uniform(0.4, 0.9)
            }
            for v in ['social_engineering', 'zero_day', 'supply_chain', 'physical_access']
        ]
    def monitor_ai_decisions(self):
        while True:
            current_metrics = {
                'cpu_usage': psutil.cpu_percent(),
                'memory_usage': psutil.virtual_memory().percent,
                'active_connections': len(self.active_connections),
                'attack_success_rate': self.calculate_success_rate()
            }
            
            attack_decision = {
                'type': random.choice(['exploit', 'lateral_move', 'persist', 'evade']),
                'target': f'192.168.1.{random.randint(1,255)}',
                'timestamp': time.time()
            }
            
            self.attack_history.append(attack_decision)
            self.system_metrics = current_metrics
            
            socketio.emit('update', {
                'metrics': current_metrics,
                'attack': attack_decision,
                'api_status': self.validate_connection(self.api_key)
            })
            
            time.sleep(3)
    @socketio.on('command')
    async def handle_command(data):
        result = await monitor.execute_command(data)
        socketio.emit('command_result', result)
        command = data.get('command')
        moduletracker = ModuleTracker()
        if command == 'execute':
            input_text = data.get('input')
            result = await monitor.execute_command({'type': 'analyze', 'target': input_text})
            socketio.emit('command_result', result)
        
        elif command == 'optimize_module':
            module_name = data.get('module_name')
            result = moduletracker.optimize_module(module_name)
            socketio.emit('module_status', result)
    @app.get('/api/status')
    def get_status():
        return {
            'system_status': monitor.system_metrics,
            'active_attacks': len(monitor.attack_history),
            'performance': monitor.calculate_performance()
        }


    def calculate_performance(self):
        if not self.attack_history:
            return {
                'success_rate': 0.0,
                'operational_efficiency': 0.0,
                'system_health': 100.0
            }
                
        recent_attacks = self.attack_history[-100:]  # Analyze last 100 operations
        successful_attacks = sum(1 for attack in recent_attacks 
                            if attack.get('status') == 'success')
        
        # Calculate core metrics
        success_rate = (successful_attacks / len(recent_attacks)) * 100
        operational_efficiency = self.calculate_operational_efficiency()
        system_load = self.get_system_load()
        
        # Advanced performance indicators
        performance_metrics = {
            'success_rate': success_rate,
            'operational_efficiency': operational_efficiency,
            'system_load': system_load,
            'attack_velocity': len(recent_attacks) / 100,  # Attacks per second
            'resource_utilization': {
                'cpu': psutil.cpu_percent(),
                'memory': psutil.virtual_memory().percent,
                'network': random.uniform(20, 80)  # Simulated network usage
            },
            'response_times': {
                'attack_execution': random.uniform(0.1, 0.5),  # seconds
                'defense_activation': random.uniform(0.05, 0.2),
                'system_response': random.uniform(0.01, 0.1)
            }
        }
        
        return performance_metrics

    def calculate_operational_efficiency(self):
        # Calculate efficiency based on resource usage and success rates
        cpu_efficiency = 100 - psutil.cpu_percent()  # Lower CPU usage is better
        memory_efficiency = 100 - psutil.virtual_memory().percent
        
        # Weight the factors
        weighted_efficiency = (
            cpu_efficiency * 0.4 +  # CPU efficiency weight
            memory_efficiency * 0.3 +  # Memory efficiency weight
            random.uniform(70, 100) * 0.3  # Network efficiency weight
        )
        
        return weighted_efficiency

    def get_system_load(self):
        return {
            'current_load': psutil.getloadavg()[0],
            'peak_load': max(psutil.getloadavg()),
            'load_trend': random.choice(['increasing', 'stable', 'decreasing'])
        }
    def calculate_success_rate(self):
        if not self.attack_history:
            return 0.0
            
        recent_attacks = self.attack_history[-100:]  # Look at last 100 attacks
        successful_attacks = sum(1 for attack in recent_attacks 
                            if attack.get('status') == 'success')
        
        success_rate = (successful_attacks / len(recent_attacks)) * 100
        
        # Track performance trends
        self.performance_metrics = {
            'success_rate': success_rate,
            'total_attacks': len(self.attack_history),
            'recent_success': successful_attacks
        }
        
        return success_rate
class ModuleTracker:
    def __init__(self):
        self.active_modules = {}
        self.available_modules = self.scan_directory()
        self.start_monitoring()

    def get_running_python_processes(self):
        python_processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if 'python' in proc.info['name'].lower():
                    cmd_line = proc.info['cmdline']
                    if cmd_line:
                        module_name = cmd_line[-1]
                        python_processes.append({
                            'pid': proc.info['pid'],
                            'module': module_name,
                            'cpu_usage': proc.cpu_percent(),
                            'memory_usage': proc.memory_info().rss / 1024 / 1024  # MB
                        })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return python_processes

    def optimize_module(self, module_name):
        for module in self.get_running_python_processes():
            if module['module'] == module_name:
                process = psutil.Process(module['pid'])
                process.nice(10)  # Adjust priority
                return {
                    'status': 'optimized',
                    'metrics': {
                        'cpu': process.cpu_percent(),
                        'memory': process.memory_info().rss / 1024 / 1024,
                        'priority': process.nice()
                    }
                }
        return {'status': 'error', 'message': 'Module not found'}

    def restart_module(self, module_name):
        self.terminate_module(module_name)
        return self.load_module(module_name)
    def load_module(self, module_name):
        try:
            # Use a different port range for modules
            module_port = random.randint(9000, 10000)
            process = subprocess.Popen([sys.executable, module_name, f"--port={module_port}"])
            
            # Track the new module
            self.active_modules[module_name] = {
                'pid': process.pid,
                'port': module_port,
                'status': 'running'
            }
            
            return {
                "status": "success", 
                "message": f"Module {module_name} loaded on port {module_port}", 
                "pid": process.pid
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def get_active_modules(self):
        active_modules = []
        for module_name, info in self.active_modules.items():
            try:
                process = psutil.Process(info['pid'])
                active_modules.append({
                    'name': module_name,
                    'pid': info['pid'],
                    'port': info['port'],
                    'status': 'running',
                    'cpu': process.cpu_percent(),
                    'memory': process.memory_info().rss / 1024 / 1024
                })
            except psutil.NoSuchProcess:
                # Clean up terminated modules
                del self.active_modules[module_name]
        return active_modules
    def terminate_module(self, module_name):
        for module in self.get_running_python_processes():
            if module['module'] == module_name:
                try:
                    process = psutil.Process(module['pid'])
                    process.terminate()
                    process.wait(timeout=3)
                    return {"status": "success", "message": f"Module {module_name} terminated"}
                except Exception as e:
                    return {"status": "error", "message": str(e)}
        return {"status": "error", "message": "Module not found"}

    def start_monitoring(self):
        @app.get("/active-modules")
        async def get_active_modules():
            return {"modules": self.get_running_python_processes()}

# ============================================================================
# HADES AI CHAT INTEGRATION
# ============================================================================

@socketio.on('chat')
def handle_chat(data):
    """Handle chat messages using HadesAI"""
    message = data.get('message', '')
    if not message:
        socketio.emit('chat_response', {'response': 'Please enter a message.'})
        return
    
    if hades_ai and HADES_AVAILABLE:
        result = hades_ai.chat(message)
        response = result.get('response', 'No response from HadesAI')
        action = result.get('action')
        
        socketio.emit('chat_response', {
            'response': response,
            'action': action
        })
    else:
        socketio.emit('chat_response', {
            'response': 'HadesAI is not available. Please check installation.'
        })

@app.route('/api/hades/chat', methods=['POST'])
def hades_chat():
    """REST API endpoint for HadesAI chat"""
    from flask import request, jsonify
    data = request.get_json()
    message = data.get('message', '')
    
    if not message:
        return jsonify({'error': 'No message provided'}), 400
    
    if hades_ai and HADES_AVAILABLE:
        result = hades_ai.chat(message)
        return jsonify(result)
    else:
        return jsonify({'error': 'HadesAI not available'}), 503

@app.route('/api/hades/status')
def hades_status():
    """Get HadesAI status"""
    from flask import jsonify
    return jsonify({
        'available': HADES_AVAILABLE,
        'active': hades_ai is not None,
        'engine': 'HadesAI'
    })

if __name__ == "__main__":
    monitor = DashboardMonitor()
    monitor.start_monitoring()
    socketio.run(app, host="0.0.0.0", port=5000)