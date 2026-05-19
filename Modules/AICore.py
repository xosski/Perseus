from fastapi.security import APIKeyHeader
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Response
from fastapi.responses import FileResponse
from pathlib import Path
import uvicorn
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import random
import os
import sys
import psutil
import time
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import secrets
import subprocess
import socket
import ast
from typing import Optional
import builtins


def _safe_print(*args, **kwargs):
    """Print status messages without failing on legacy Windows consoles."""
    try:
        builtins.print(*args, **kwargs)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        safe_args = [str(arg).encode(encoding, errors="replace").decode(encoding, errors="replace") for arg in args]
        builtins.print(*safe_args, **kwargs)


print = _safe_print

# Support running this file both as ``python modules/AICore.py`` and as
# ``import modules.AICore``.  When executed as a script, Python only adds the
# modules directory to sys.path, so the project root must be added before
# importing HadesAI from the repository root.
MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from modules.ai_dashboard import DashboardMonitor
except ImportError:
    from ai_dashboard import DashboardMonitor

try:
    from GhostMemory import GhostMemory
except ImportError:
    class GhostMemory:
        """Minimal conversation-memory fallback used when GhostMemory is absent."""

        def __init__(self):
            self.messages = []

        def save(self, role, content):
            self.messages.append({
                "role": role,
                "content": content,
                "timestamp": time.time(),
            })

# Import HadesAI as the main AI engine (replaces Mistral)
try:
    from HadesAI import HadesAI
    HADES_AVAILABLE = True
    logger.info("✅ HadesAI loaded successfully")
except ImportError as e:
    HADES_AVAILABLE = False
    logger.warning(f"⚠️ HadesAI not available: {e}")

_initialized = False
API_KEY = None
class KeyManager:
    _instance = None
    _initialized = False
    KEY_FILE = "api_key.json"
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(KeyManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'active_keys'):
            self.active_keys = set()
            self.API_KEY = self.generate_new_api_key()
            self.save_key(self.API_KEY)
            print("🚀 Launching AI Engine")
            print("📡 Server starting at http://127.0.0.1:8000")
            print(f"🔑 API Key: {self.API_KEY}")
            
    def generate_new_api_key(self):
        new_key = secrets.token_hex(32)
        self.active_keys.add(new_key)
        return new_key
    
    def save_key(self, key):
        with open(self.KEY_FILE, 'w') as f:
            json.dump({'api_key': key}, f)
            
    def validate_key(self, key):
        return key in self.active_keys
        
    def initialize_server(self):
        if not self._initialized:
            self._initialized = True
        return self.API_KEY

key_manager = KeyManager()
api_key_header = APIKeyHeader(name="X-API-Key")
API_KEY = key_manager.initialize_server()
async def get_current_api_key():
    return {"api_key": API_KEY, "generated_at": time.time()}

# ============================================================================
# HADES AI INITIALIZATION (Replaces Mistral)
# ============================================================================

# Initialize HadesAI as the main AI engine
hades_ai = None
if HADES_AVAILABLE:
    try:
        hades_ai = HadesAI()
        print("=" * 50)
        print("🔥 HADES AI ENGINE INITIALIZED")
        print("=" * 50)
        print("✅ Self-learning pentesting AI active")
        print("✅ Knowledge base connected")
        print("✅ Exploitation engine ready")
        print("✅ Web learning enabled")
        print("=" * 50)
    except Exception as e:
        logger.error(f"❌ Failed to initialize HadesAI: {e}")
        hades_ai = None
else:
    print("⚠️ HadesAI not available - chat functionality will be limited")

# Initialize GhostMemory for conversation context
ghost_memory = GhostMemory()

# Thread pool for async operations
ai_executor = ThreadPoolExecutor(max_workers=int(os.getenv("AI_MAX_WORKERS", "2")))


async def ask_hades(user_message: str, module_context: dict = None) -> dict:
    """Query HadesAI with optional module context"""
    if not hades_ai:
        return {"response": "HadesAI is not available.", "action": None}
    
    try:
        # If module context provided, enhance the message
        enhanced_message = user_message
        if module_context:
            enhanced_message = f"[Available modules: {', '.join(module_context.get('modules', []))}] {user_message}"
        
        # Get response from HadesAI
        result = hades_ai.chat(enhanced_message)
        
        # Persist to ghost memory
        ghost_memory.save("user", user_message)
        ghost_memory.save("ghost", result.get('response', ''))
        
        return result
    except Exception as e:
        logger.error(f"❌ HadesAI Error: {e}")
        return {"response": f"Error: {str(e)}", "action": None}


class AIEngine:
    def __init__(self):
        self.status = "ONLINE"
        self.active_strategies = set()
        self.performance_metrics = {}
        self.start_time = time.time()
        
    def get_uptime(self):
        uptime_seconds = time.time() - self.start_time
        return {
            "total_seconds": int(uptime_seconds),
            "formatted": self.format_uptime(uptime_seconds)
        }
    def format_uptime(self, seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    def get_status(self):
        return {
            "status": self.status,
            "active_strategies": list(self.active_strategies),
            "performance": self.calculate_performance(),
            "uptime": self.get_uptime()
        }
    
    def calculate_performance(self):
        return {
            "success_rate": random.uniform(0.7, 0.95),
            "decisions_per_second": random.randint(100, 1000),
            "active_connections": len(self.active_strategies)
        }
# Initialize the AI Engine
ai_engine = AIEngine()
# ... existing code ...
class ModuleTracker:
    def __init__(self):
        self.active_modules = {}
        self.available_modules = self.scan_directory()
        self.importer = DynamicImporter()

    def is_valid_python_module(self, file_path: Path) -> bool:
        """
        Quickly validate whether a .py file contains valid Python syntax.
        Avoids launching TypeScript/JS or malformed Python files.
        """
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                source = f.read()
            ast.parse(source)
            return True
        except SyntaxError:
            return False
        except Exception:
            # If we can't read/parse for any reason, treat it as invalid
            return False

    def scan_directory(self):
        python_files = []
        for file in ROOT_DIR.glob('**/*.py'):
            if file.name != '__init__.py':
                is_valid = self.is_valid_python_module(file)
                python_files.append({
                    'name': file.name,
                    'path': str(file.relative_to(ROOT_DIR)),
                    'status': 'available' if is_valid else 'invalid',
                    'memory': 0.0,  # default memory
                    'cpu': 0.0      # default CPU
                })
        return python_files

    def load_module(self, module_name):
        # Find module metadata
        module_meta = next((m for m in self.available_modules if m['name'] == module_name), None)
        if not module_meta:
            return {"status": "error", "message": f"Module {module_name} not found", "module_name": module_name}

        # Prevent launching invalid Python files
        if module_meta.get('status') == 'invalid':
            return {
                "status": "error",
                "message": f"Module {module_name} has invalid Python syntax and cannot be launched.",
                "module_name": module_name
            }

        module_path = module_meta['path']
        try:
            full_path = str(ROOT_DIR / module_path)
            module_port = random.randint(9000, 10000)
            process = subprocess.Popen([sys.executable, full_path, f"--port={module_port}"])

            # Update module status immediately
            module_meta['status'] = 'running'
            module_meta['pid'] = process.pid
            module_meta['port'] = module_port

            return {
                "status": "success",
                "message": f"Module {module_name} loaded on port {module_port}",
                "pid": process.pid,
                "module_name": module_name
            }
        except Exception as e:
            return {"status": "error", "message": str(e), "module_name": module_name}

    def optimize_module(self, module_name):
        for module in self.available_modules:
            if module['name'] == module_name:
                module['cpu'] = psutil.cpu_percent()
                module['memory'] = psutil.Process().memory_info().rss / 1024 / 1024
                return {"status": "optimized", "metrics": module}
        return {"status": "error", "message": f"Module {module_name} not found"}

    def restart_module(self, module_name):
        self.terminate_module(module_name)
        return self.load_module(module_name)

    def terminate_module(self, module_name):
        """Gracefully terminate a running module"""
        for module in self.available_modules:
            if module['name'] == module_name and module.get('pid'):
                try:
                    process = psutil.Process(module['pid'])
                    process.terminate()
                    process.wait(timeout=3)
                    module['status'] = 'available'
                    module['pid'] = None
                    module['port'] = None
                    return {"status": "success", "message": f"Module {module_name} terminated"}
                except psutil.NoSuchProcess:
                    module['status'] = 'available'
                    return {"status": "success", "message": "Process already terminated"}
                except Exception as e:
                    return {"status": "error", "message": str(e)}
        return {"status": "error", "message": f"Module {module_name} not found"}

    def get_running_python_processes(self):
        python_processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if 'python' in (proc.info['name'] or '').lower():
                    cmd_line = proc.info['cmdline']
                    if cmd_line:
                        module_name = Path(cmd_line[-1]).name
                        python_processes.append({
                            'pid': proc.info['pid'],
                            'name': module_name,
                            'cpu': proc.cpu_percent(),
                            'memory': proc.memory_info().rss / 1024 / 1024,
                            'status': 'running'
                        })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        # include available/invalid modules not currently running
        for module in self.available_modules:
            if not any(p['name'] == module['name'] for p in python_processes):
                python_processes.append(module)

        return python_processes
class AttackMonitor:
    def __init__(self):
        self.attacks = []
        self.active_targets = set()
        self.success_metrics = {
            'successful': 0,
            'failed': 0,
            'ongoing': 0
        }
    
    def get_metrics(self):
        return {
            "total_attacks": len(self.attacks),
            "active_targets": len(self.active_targets),
            "success_rate": self.calculate_success_rate(),
            "current_operations": self.get_active_operations()
        }
    
    def calculate_success_rate(self):
        total = self.success_metrics['successful'] + self.success_metrics['failed']
        return (self.success_metrics['successful'] / total) if total > 0 else 0
    
    def get_active_operations(self):
        return [
            {"target": target, "status": "engaging"} 
            for target in self.active_targets
        ]
ROOT_DIR = Path(__file__).resolve().parent
class DynamicImporter:
    def __init__(self):
        self.common_imports = {
            'Request': 'from fastapi import Request',
            'FastAPI': 'from fastapi import FastAPI',
            'WebSocket': 'from fastapi import WebSocket',
            'random': 'import random',
            'asyncio': 'import asyncio',
            'json': 'import json'
        }
        
    def inject_imports(self, module_path):
        with open(module_path, 'r') as file:
            content = file.read()
            
        needed_imports = []
        for name, import_statement in self.common_imports.items():
            if name in content and import_statement not in content:
                needed_imports.append(import_statement)
                
        if needed_imports:
            new_content = '\n'.join(needed_imports) + '\n' + content
            with open(module_path, 'w') as file:
                file.write(new_content)

class BattleReadyPortManager:
    def __init__(self):
        self.base_port = 8000
        self.port_range = range(8000, 9000)
        self.max_retries = 20
        self.active_ports = set()
        
    def terminate_existing_connections(self, port):
        for conn in psutil.net_connections():
            if conn.laddr.port == port:
                try:
                    psutil.Process(conn.pid).terminate()
                    print(f"🛡️ Cleared conflicting process on port {port}")
                except:
                    pass
                    
    def secure_port(self):
        for port in self.port_range:
            self.terminate_existing_connections(port)
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.bind(('127.0.0.1', port))
                s.close()
                print(f"🔒 Secured command channel on port {port}")
                return port
            except:
                continue
        return random.randint(9000, 10000)
app = FastAPI()
# Initialize the tracker
module_tracker = ModuleTracker()


# Initialize the Attack Monitor
attack_monitor = AttackMonitor()
# Get the absolute path to our project root
ROOT_DIR = Path(__file__).resolve().parent

# Define our command center structure
DIRECTORIES = [
    ROOT_DIR / "static",
    ROOT_DIR / "static/css",
    ROOT_DIR / "static/js",
    ROOT_DIR / "templates"
]

# Create the command center directories
for directory in DIRECTORIES:
    directory.mkdir(parents=True, exist_ok=True)
css_path = ROOT_DIR / "static/css/bridge.css"

css_content = """
/* SingularityDrive Bridge Theme */
body {
    background-color: #0a0f1c;
    color: #00fff2;
    font-family: 'Orbitron', 'Courier New', monospace;
    margin: 0;
    padding: 0;
    background-image: linear-gradient(45deg, #0a0f1c 25%, #111827 25%, #111827 50%, #0a0f1c 50%, #0a0f1c 75%, #111827 75%, #111827 100%);
    background-size: 56.57px 56.57px;
}

.bridge-container {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 20px;
    padding: 20px;
    max-width: 1400px;
    margin: 0 auto;
}

.module {
    background: rgba(16, 24, 39, 0.8);
    border: 2px solid #00fff2;
    border-radius: 10px;
    padding: 20px;
    box-shadow: 0 0 15px rgba(0, 255, 242, 0.2);
    backdrop-filter: blur(5px);
    position: relative;
    overflow: hidden;
}

.module::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 2px;
    background: linear-gradient(90deg, transparent, #00fff2, transparent);
    animation: scan-line 2s linear infinite;
}

@keyframes scan-line {
    0% { transform: translateX(-100%); }
    100% { transform: translateX(100%); }
}

.metrics {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 10px;
}

.metric-value {
    font-size: 24px;
    font-weight: bold;
    text-shadow: 0 0 10px rgba(0, 255, 242, 0.5);
}

h1 {
    color: #00fff2;
    text-align: center;
    font-size: 2.5em;
    text-transform: uppercase;
    letter-spacing: 3px;
    margin-bottom: 30px;
    text-shadow: 0 0 20px rgba(0, 255, 242, 0.5);
}

.status-indicator {
    display: inline-block;
    width: 12px;
    height: 12px;
    border-radius: 50%;
    margin-right: 8px;
}

.status-online {
    background-color: #00ff00;
    box-shadow: 0 0 10px #00ff00;
}

.status-offline {
    background-color: #ff0000;
    box-shadow: 0 0 10px #ff0000;
}
/* Add cache busting for dynamic updates */
.module {
    transition: all 0.3s ease;
}

.module:hover {
    transform: scale(1.02);
    box-shadow: 0 0 25px rgba(0, 255, 242, 0.4);
}

/* Enhanced loading states */
.loading {
    animation: pulse 1.5s infinite;
}

@keyframes pulse {
    0% { opacity: 1; }
    50% { opacity: 0.7; }
    100% { opacity: 1; }
}

/* Chat Interface Styling */
.ai-chat {
    grid-column: span 2;
}

.chat-container {
    height: 400px;
    display: flex;
    flex-direction: column;
}

.chat-messages {
    flex: 1;
    overflow-y: auto;
    padding: 10px;
    background: rgba(0, 0, 0, 0.3);
    border: 1px solid #00fff2;
    border-radius: 5px;
    margin-bottom: 10px;
    max-height: 300px;
}

.chat-message {
    margin-bottom: 10px;
    padding: 8px;
    border-radius: 5px;
    word-wrap: break-word;
}

.chat-message.user {
    background: rgba(0, 255, 242, 0.1);
    border-left: 3px solid #00fff2;
}

.chat-message.ai {
    background: rgba(0, 255, 0, 0.1);
    border-left: 3px solid #00ff00;
}

.chat-message.system {
    background: rgba(255, 165, 0, 0.1);
    border-left: 3px solid #ffa500;
    font-style: italic;
}

.chat-message.error {
    background: rgba(255, 0, 0, 0.1);
    border-left: 3px solid #ff0000;
}

.chat-input-container {
    display: flex;
    gap: 10px;
}

.chat-input-container input {
    flex: 1;
    padding: 10px;
    background: rgba(0, 0, 0, 0.5);
    border: 1px solid #00fff2;
    border-radius: 5px;
    color: #00fff2;
    font-family: 'Orbitron', monospace;
}

.chat-input-container input:focus {
    outline: none;
    box-shadow: 0 0 10px rgba(0, 255, 242, 0.5);
}

.chat-input-container button, .ai-controls button {
    padding: 10px 15px;
    background: linear-gradient(45deg, #00fff2, #0099cc);
    border: none;
    border-radius: 5px;
    color: #000;
    font-weight: bold;
    cursor: pointer;
    font-family: 'Orbitron', monospace;
    transition: all 0.3s ease;
}

.chat-input-container button:hover, .ai-controls button:hover {
    background: linear-gradient(45deg, #0099cc, #00fff2);
    box-shadow: 0 0 15px rgba(0, 255, 242, 0.5);
    transform: translateY(-2px);
}

.ai-controls {
    display: flex;
    gap: 10px;
    margin-top: 10px;
    flex-wrap: wrap;
}
"""
# Write the CSS file
with open(css_path, 'w') as f:
    f.write(css_content)

print("🎨 Bridge CSS styling generated!")
print("🎯 Command center directories created!")
print("💫 The Bridge is ready for deployment!")
templates = Jinja2Templates(directory=str(ROOT_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(ROOT_DIR / "static")), name="static")
# Health check endpoint
@app.get("/")
def health_check():
    return {"status": "online", "service": "AI Engine"}
@app.get("/current-api-key")
async def current_api_key():
    return await get_current_api_key()
@app.get("/current-api-key")
async def current_api_key():
    return await get_current_api_key()
@app.get("/bridge")
async def command_bridge(request: Request):
     # Safe fallback for load averages on Windows
     # Safe fallback for load averages on Windows
    try:
        loadavg = psutil.getloadavg()
        current_load = loadavg[0]
        peak_load = max(loadavg)
    except (AttributeError, OSError):
        current_load = psutil.cpu_percent()
        peak_load = current_load

    # Core system metrics (single definition; removes duplicate block)
    metrics = {
        'cpu_usage': psutil.cpu_percent(),
        'memory_usage': psutil.virtual_memory().percent,
        'network_usage': random.randint(0, 100),
        'operational_efficiency': random.uniform(70, 100),
        'system_load': {
            'current_load': current_load,
            'trend': random.choice(['increasing', 'stable', 'decreasing'])
        }
    }

    # Attack monitoring metrics
    attack_metrics = {
        'total_attacks': len(attack_monitor.attacks),
        'active_targets': len(attack_monitor.active_targets),
        'success_rate': attack_monitor.calculate_success_rate(),
        'attack_velocity': random.uniform(0.5, 2.0),
        'defense_rating': {
            'overall_rating': random.uniform(0.7, 0.99),
            'shield_strength': random.uniform(85, 100),
            'countermeasure_effectiveness': random.uniform(0.8, 0.95)
        }
    }
    
    return templates.TemplateResponse("bridge.html", {
        "request": request,
        "metrics": metrics,
        "attack_metrics": attack_metrics,
        "ai_status": ai_engine.get_status(),
        "active_modules": module_tracker.get_running_python_processes()
    })

@app.get('/favicon.ico')
async def favicon():
    favicon_path = Path(ROOT_DIR) / "static" / "favicon.ico"
    if not favicon_path.exists():
        # Return a default empty favicon if file doesn't exist
        return Response(content=b'', media_type='image/x-icon')
    return FileResponse(favicon_path)

PORT = os.getenv('PORT', 8000)
HOST = os.getenv('HOST', '127.0.0.1')
LOG_FILE = os.getenv('LOG_FILE', 'exfiltrated_data.log')
# 🧠 Portable AI Decision Model
def predict_attack_strategy(user_data):
    if user_data["time_of_day"] in range(9, 17):
        return {"attackType": "phishing"}
    else:
        return {"attackType": "keylogging"}

@app.post("/ai-decision")
async def ai_decision(request: Request):
    user_data = await request.json()
    decision = predict_attack_strategy(user_data)
    return decision

@app.post("/generate-polymorphic")
async def generate_polymorphic(request: Request):
    data = await request.json()
    base_payload = data["payload"]
    mutated_payload = base_payload.replace("formData", "data_" + str(random.randint(1000, 9999)))
    return {"mutatedPayload": mutated_payload}
@app.post("/aiMonitor")
async def ai_monitor(request: Request):
    user_data = await request.json()
    decision = predict_attack_strategy(user_data)
    return decision

@app.post("/chat")
async def chat_with_ai(request: Request):
    """Chat endpoint using HadesAI"""
    data = await request.json()
    user_message = data.get("message", "").strip()
    
    try:
        # Get available modules for context
        module_context = {
            'modules': [m['name'] for m in module_tracker.available_modules if m.get('status') != 'invalid']
        }
        
        # Query HadesAI
        result = await ask_hades(user_message, module_context)
        
        return {
            "response": result.get('response', 'No response'),
            "action": result.get('action'),
            "status": "success"
        }
    except Exception as e:
        return {"response": f"Error: {str(e)}", "status": "error"}

@app.post("/start-ai")
async def start_ai():
    """Start/test HadesAI engine"""
    try:
        if hades_ai:
            result = await ask_hades("Hello, are you online?")
            return {
                "status": "success", 
                "message": "HadesAI engine is online",
                "test_response": result.get('response', 'Ready'),
                "engine": "HadesAI"
            }
        else:
            return {"status": "error", "message": "HadesAI not available"}
    except Exception as e:
        return {"status": "error", "message": f"Failed to start AI: {str(e)}"}

# ============================================================================
# HADES AI FEATURE ENDPOINTS
# ============================================================================

@app.post("/hades/scan-cache")
async def hades_scan_cache():
    """Trigger HadesAI browser cache scan"""
    if not hades_ai:
        return {"status": "error", "message": "HadesAI not available"}
    
    try:
        # This would run in background - return immediately
        return {"status": "started", "message": "Cache scan initiated"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/hades/learn-url")
async def hades_learn_url(request: Request):
    """Learn exploits from a URL"""
    if not hades_ai:
        return {"status": "error", "message": "HadesAI not available"}
    
    data = await request.json()
    url = data.get("url", "").strip()
    
    if not url:
        return {"status": "error", "message": "No URL provided"}
    
    try:
        result = hades_ai.learn_from_url(url)
        return {"status": "success", "result": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/hades/exploits")
async def hades_get_exploits():
    """Get learned exploits"""
    if not hades_ai:
        return {"status": "error", "message": "HadesAI not available"}
    
    try:
        exploits = hades_ai.kb.get_learned_exploits(50)
        return {"status": "success", "exploits": exploits}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/hades/findings")
async def hades_get_findings():
    """Get threat findings"""
    if not hades_ai:
        return {"status": "error", "message": "HadesAI not available"}
    
    try:
        findings = hades_ai.kb.get_threat_findings(100)
        return {"status": "success", "findings": findings}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/hades/status")
async def hades_status():
    """Get HadesAI status"""
    return {
        "available": HADES_AVAILABLE,
        "active": hades_ai is not None,
        "engine": "HadesAI",
        "features": ["chat", "cache_scan", "web_learning", "exploits", "tools"] if hades_ai else []
    }

@app.post("/hades/run-tool")
async def hades_run_tool(request: Request):
    """Run a HadesAI pentesting tool"""
    if not hades_ai:
        return {"status": "error", "message": "HadesAI not available"}
    
    data = await request.json()
    tool = data.get("tool")
    target = data.get("target")
    
    if not tool or not target:
        return {"status": "error", "message": "Tool and target required"}
    
    # Execute via chat command
    result = await ask_hades(f"run {tool} on {target}")
    return {"status": "success", "result": result}

@app.post("/hades/execute-module")
async def hades_execute_module(request: Request):
    """Execute a loaded module via HadesAI"""
    data = await request.json()
    module_name = data.get("module_name")
    action = data.get("action", "run")
    
    if not module_name:
        return {"status": "error", "message": "Module name required"}
    
    # Check if module is available
    module_meta = next((m for m in module_tracker.available_modules if m['name'] == module_name), None)
    if not module_meta:
        return {"status": "error", "message": f"Module {module_name} not found"}
    
    # Load module if not running
    if module_meta.get('status') != 'running':
        load_result = module_tracker.load_module(module_name)
        if load_result.get('status') != 'success':
            return load_result
    
    return {
        "status": "success", 
        "message": f"Module {module_name} executed",
        "module": module_meta
    }

@app.post("/hades/full-scan")
async def hades_full_scan(request: Request):
    """Run comprehensive site scan on a target URL"""
    if not hades_ai:
        return {"status": "error", "message": "HadesAI not available"}
    
    data = await request.json()
    url = data.get("url", "").strip()
    
    if not url:
        return {"status": "error", "message": "No URL provided"}
    
    # Add protocol if missing
    if not url.startswith('http'):
        url = f"https://{url}"
    
    try:
        # Run the comprehensive scan
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            ai_executor,
            lambda: hades_ai.full_site_scan(url)
        )
        return {"status": "success", "result": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/hades/export-pdf")
async def hades_export_pdf(request: Request):
    """Export findings to PDF report"""
    if not hades_ai:
        return {"status": "error", "message": "HadesAI not available"}
    
    data = await request.json()
    clear_after = data.get("clear_after", False)
    
    try:
        from datetime import datetime
        import os
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"hades_security_report_{timestamp}.pdf"
        
        # Save to reports directory
        reports_dir = ROOT_DIR / "reports"
        reports_dir.mkdir(exist_ok=True)
        filepath = str(reports_dir / filename)
        
        # Generate the PDF
        result = hades_ai.export_exploits_to_pdf(filepath, clear_after=clear_after)
        
        if result.get('success'):
            return {
                "status": "success",
                "filename": filename,
                "filepath": filepath,
                "download_url": f"/reports/{filename}",
                "exploits": result.get('exploits_exported', 0),
                "findings": result.get('findings_exported', 0),
                "patterns": result.get('patterns_exported', 0),
                "cleared": result.get('cleared', False)
            }
        else:
            return {"status": "error", "message": result.get('error', 'Unknown error')}
            
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/reports/{filename}")
async def download_report(filename: str):
    """Download a generated PDF report"""
    from fastapi.responses import FileResponse
    
    filepath = ROOT_DIR / "reports" / filename
    if not filepath.exists():
        return {"status": "error", "message": "Report not found"}
    
    return FileResponse(
        path=str(filepath),
        filename=filename,
        media_type='application/pdf'
    )

@app.get("/hades/reports")
async def list_reports():
    """List all generated reports"""
    reports_dir = ROOT_DIR / "reports"
    if not reports_dir.exists():
        return {"status": "success", "reports": []}
    
    reports = []
    for f in reports_dir.glob("*.pdf"):
        reports.append({
            "filename": f.name,
            "size": f.stat().st_size,
            "created": f.stat().st_mtime,
            "download_url": f"/reports/{f.name}"
        })
    
    # Sort by creation time, newest first
    reports.sort(key=lambda x: x['created'], reverse=True)
    return {"status": "success", "reports": reports}

js_path = ROOT_DIR / "static/js/bridge.js"

js_content = """
// Define addChatMessage first since it's used by event handlers
function addChatMessage(sender, message, type) {
    const chatMessages = document.getElementById('chat-messages');
    if (!chatMessages) return;
    const messageDiv = document.createElement('div');
    messageDiv.className = `chat-message ${type}`;
    messageDiv.innerHTML = `<strong>${sender}:</strong> ${message}`;
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// WebSocket connection
const wsUrl = `ws://${window.location.host}/ws`;
console.log('Connecting to WebSocket:', wsUrl);
const socket = new WebSocket(wsUrl);

socket.onopen = () => {
    console.log('✅ WebSocket connected');
    addChatMessage('System', 'Connected to AI Core', 'system');
};

socket.onerror = (error) => {
    console.error('❌ WebSocket error:', error);
    addChatMessage('System', 'Connection error', 'error');
};

socket.onclose = () => {
    console.log('🔌 WebSocket disconnected');
    addChatMessage('System', 'Disconnected from AI Core', 'error');
};

socket.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log('📨 Received:', data.type);
    
    if (data.type === 'chat_response') {
        addChatMessage('AI', data.data.response, 'ai');
    } else if (data.type === 'chat_error') {
        addChatMessage('System', `Error: ${data.data.error}`, 'error');
    } else if (data.type === 'ai_started') {
        addChatMessage('System', 'AI engine started successfully!', 'system');
        addChatMessage('AI', data.data.test_response, 'ai');
    } else if (data.type === 'ai_error') {
        addChatMessage('System', `AI startup error: ${data.data.message}`, 'error');
    } else if (data.type === 'metrics_update') {
        updateMetrics(data);
    } else {
        updateModuleStatus(data);
    }
};

function updateMetrics(data) {
    document.querySelector('#cpu-usage').textContent = `CPU: ${data.metrics.cpu_usage}%`;
    document.querySelector('#memory-usage').textContent = `Memory: ${data.metrics.memory_usage}%`;
    document.querySelector('#ai-status').textContent = `Status: ${data.ai_status.status}`;
    document.querySelector('#uptime').textContent = `Uptime: ${data.ai_status.uptime.formatted}`;
    document.querySelector('#active-targets').textContent = `Active Targets: ${data.attack_metrics.active_targets}`;
    document.querySelector('#success-rate').textContent = `Success Rate: ${(data.attack_metrics.success_rate * 100).toFixed(2)}%`;

    const modulesList = document.querySelector('#modules-list');
    if (modulesList && data.active_modules) {
        modulesList.innerHTML = data.active_modules.map(module => `
            <div class="module-item" onclick="loadModule('${module.name}')" data-name="${module.name}">
                ${module.name} - ${module.status}
                ${module.status === 'running' ? 
                    `(PID: ${module.pid}, CPU: ${module.cpu}%, Memory: ${module.memory.toFixed(2)} MB)` : 
                    '<button class="load-btn">Load</button>'
                }
            </div>
        `).join('');
    }
}

function updateModuleStatus(data) {
    if (data.type === "module_loaded" && data.data && data.data.status === "success") {
        const moduleElement = document.querySelector(`[data-name="${data.data.module_name}"]`);
        if (moduleElement) {
            const btn = moduleElement.querySelector('.load-btn');
            if (btn) {
                btn.textContent = 'Running';
                btn.classList.add('running');
            }
        }
    }
}

function loadModule(moduleName) {
    socket.send(JSON.stringify({ command: 'load_module', module_name: moduleName }));
}

function startAI() {
    addChatMessage('System', 'Starting AI engine...', 'system');
    socket.send(JSON.stringify({ command: 'start_ai' }));
}

function sendChatMessage() {
    const input = document.getElementById('chat-input');
    const message = input.value.trim();
    if (message && socket.readyState === WebSocket.OPEN) {
        addChatMessage('You', message, 'user');
        socket.send(JSON.stringify({ command: 'chat', message: message }));
        input.value = '';
    }
}

function handleChatKeyPress(event) {
    if (event.key === 'Enter') {
        sendChatMessage();
    }
}
"""

# Write the JS file
with open(js_path, 'w', encoding='utf-8') as f:
    f.write(js_content)

print("📡 Bridge WebSocket handler generated!")

# Add to existing WebSocket handlers
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # Create a queue for metrics updates
    metrics_queue = asyncio.Queue()
    
    async def metrics_producer():
        while True:
            try:
                loadavg = psutil.getloadavg()
                current_load = loadavg[0]
            except (AttributeError, OSError):
                current_load = psutil.cpu_percent()

            # Windows-safe load average
            try:
                loadavg = psutil.getloadavg()
                current_load = loadavg[0]
            except (AttributeError, OSError):
                current_load = psutil.cpu_percent()

            metrics = {
                'cpu_usage': psutil.cpu_percent(interval=0.1),
                'memory_usage': psutil.virtual_memory().percent,
                'operational_efficiency': random.uniform(70, 100),
                'system_load': {
                    'current_load': current_load,
                    'trend': random.choice(['increasing', 'stable', 'decreasing'])
                }
            }
            await metrics_queue.put(metrics)
            await asyncio.sleep(0.5)
    async def metrics_consumer():
        while True:
            try:
                metrics = await metrics_queue.get()
                await websocket.send_json({
                    "type": "metrics_update",
                    "metrics": metrics,
                    "active_modules": module_tracker.get_running_python_processes(),
                    "ai_status": ai_engine.get_status(),
                    "attack_metrics": attack_monitor.get_metrics()
                })
            except WebSocketDisconnect:
                break

    async def command_handler():
        while True:
            try:
                data = await websocket.receive_json()
                command = data.get('command')
                
                if command == 'chat':
                    message = data.get('message', '').strip()
                    try:
                        # Get available modules for AI context
                        module_context = {
                            'modules': [m['name'] for m in module_tracker.available_modules if m.get('status') != 'invalid']
                        }
                        
                        # Query HadesAI
                        result = await ask_hades(message, module_context)
                        
                        await websocket.send_json({
                            "type": "chat_response", 
                            "data": {
                                "response": result.get('response', ''),
                                "action": result.get('action'),
                                "user_message": message
                            }
                        })
                    except Exception as e:
                        await websocket.send_json({"type": "chat_error", "data": {"error": str(e)}})
                        
                elif command == 'start_ai':
                    try:
                        if hades_ai:
                            result = await ask_hades("Hello, are you online?")
                            await websocket.send_json({
                                "type": "ai_started", 
                                "data": {
                                    "status": "success", 
                                    "test_response": result.get('response', 'HadesAI online'),
                                    "engine": "HadesAI"
                                }
                            })
                        else:
                            await websocket.send_json({"type": "ai_error", "data": {"status": "error", "message": "HadesAI not available"}})
                    except Exception as e:
                        await websocket.send_json({"type": "ai_error", "data": {"status": "error", "message": str(e)}})
                
                elif command == 'hades_tool':
                    # Execute HadesAI tools
                    tool = data.get('tool')
                    target = data.get('target')
                    if hades_ai and tool and target:
                        result = await ask_hades(f"run {tool} on {target}")
                        await websocket.send_json({"type": "tool_result", "data": result})
                    else:
                        await websocket.send_json({"type": "error", "message": "Tool and target required"})
                
                elif command == 'hades_learn':
                    # Learn from URL
                    url = data.get('url')
                    if hades_ai and url:
                        result = hades_ai.learn_from_url(url)
                        await websocket.send_json({"type": "learn_result", "data": result})
                    else:
                        await websocket.send_json({"type": "error", "message": "URL required"})
                
                elif command == 'hades_scan_cache':
                    # Trigger cache scan
                    if hades_ai:
                        await websocket.send_json({"type": "scan_started", "data": {"message": "Cache scan initiated"}})
                    else:
                        await websocket.send_json({"type": "error", "message": "HadesAI not available"})
                
                elif command == 'hades_full_scan':
                    # Full comprehensive site scan
                    url = data.get('url')
                    if hades_ai and url:
                        await websocket.send_json({"type": "scan_started", "data": {"message": f"Starting full reconnaissance on {url}..."}})
                        try:
                            # Add protocol if missing
                            if not url.startswith('http'):
                                url = f"https://{url}"
                            
                            # Run scan in executor to not block
                            loop = asyncio.get_running_loop()
                            result = await loop.run_in_executor(
                                ai_executor,
                                lambda: hades_ai.full_site_scan(url)
                            )
                            await websocket.send_json({"type": "full_scan_result", "data": result})
                        except Exception as e:
                            await websocket.send_json({"type": "error", "message": str(e)})
                    else:
                        await websocket.send_json({"type": "error", "message": "URL required for full scan"})
                
                elif command == 'hades_export_pdf':
                    # Export findings to PDF
                    if hades_ai:
                        try:
                            clear_after = data.get('clear_after', False)
                            from datetime import datetime
                            
                            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                            filename = f"hades_security_report_{timestamp}.pdf"
                            reports_dir = ROOT_DIR / "reports"
                            reports_dir.mkdir(exist_ok=True)
                            filepath = str(reports_dir / filename)
                            
                            result = hades_ai.export_exploits_to_pdf(filepath, clear_after=clear_after)
                            
                            if result.get('success'):
                                await websocket.send_json({
                                    "type": "pdf_exported", 
                                    "data": {
                                        "filename": filename,
                                        "download_url": f"/reports/{filename}",
                                        "exploits": result.get('exploits_exported', 0),
                                        "findings": result.get('findings_exported', 0),
                                        "cleared": result.get('cleared', False)
                                    }
                                })
                            else:
                                await websocket.send_json({"type": "error", "message": result.get('error', 'PDF export failed')})
                        except Exception as e:
                            await websocket.send_json({"type": "error", "message": str(e)})
                    else:
                        await websocket.send_json({"type": "error", "message": "HadesAI not available"})
                
                elif command == 'execute_module':
                    # Execute module with HadesAI context
                    module_name = data.get('module_name')
                    if module_name:
                        result = module_tracker.load_module(module_name)
                        if result.get('status') == 'success' and hades_ai:
                            # Notify AI that module was loaded
                            await ask_hades(f"Module {module_name} has been loaded and is running")
                        await websocket.send_json({"type": "module_executed", "data": result})
                    else:
                        await websocket.send_json({"type": "error", "message": "Module name required"})
                        
                elif command == 'load_module':
                    module_name = data.get('module_name')
                    result = module_tracker.load_module(module_name)
                    await websocket.send_json({"type": "module_loaded", "data": result})
                    
                elif command == 'activate_defenses':
                    dashboardmonitor = DashboardMonitor()
                    result = await dashboardmonitor.activate_defenses()
                    await websocket.send_json({"type": "defense_status", "data": result})
                    
                elif command == 'analyze_target':
                    target = data.get('target')
                    dashboardmonitor = DashboardMonitor()
                    result = await dashboardmonitor.analyze_target(target)
                    await websocket.send_json({"type": "analysis_result", "data": result})
                    
                elif command == 'optimize_module':
                    module_name = data.get('module_name')
                    result = module_tracker.optimize_module(module_name)
                    await websocket.send_json({"type": "module_optimized", "data": result})
                    
                elif command == 'restart_module':
                    module_name = data.get('module_name')
                    result = module_tracker.restart_module(module_name)
                    await websocket.send_json({"type": "module_restarted", "data": result})
                    
            except WebSocketDisconnect:
                break
            except Exception as e:
                await websocket.send_json({"type": "error", "message": str(e)})

    # Run producer, consumer, and command handler concurrently
    producer = asyncio.create_task(metrics_producer())
    consumer = asyncio.create_task(metrics_consumer())
    commands = asyncio.create_task(command_handler())
    
    try:
        await asyncio.gather(producer, consumer, commands)
    except Exception as e:
        print(f"WebSocket stream ended: {e}")
    finally:
        producer.cancel()
        consumer.cancel()
        commands.cancel()


# 🛠️ Local Exfil & Logging
@app.post("/exfil")
async def receive_exfil(request: Request):
    encrypted_data = await request.json()
    with open("exfiltrated_data.log", "a") as f:
        f.write(json.dumps(encrypted_data) + "\\n")
    return {"status": "Logged"}
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Prevent TF warnings

if __name__ == "__main__":
    port_manager = BattleReadyPortManager()
    battle_port = port_manager.secure_port()
    
    uvicorn.run(
        "AICorePortable:app",
        host="127.0.0.1",
        port=battle_port,
        reload=True,
        reload_dirs=['d:\\Ai Core\\SingularityDrive'],
        reload_delay=0.25,
        workers=1,
        loop="asyncio"
    )
