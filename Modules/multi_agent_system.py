"""
Multi-Agent Collaboration System Module
Coordinates multiple autonomous agents working together on complex tasks

Features:
- Agent coordination and communication
- Task delegation and distribution
- Result aggregation
- Conflict resolution
- Load balancing
- Emergent behavior and cooperation
"""

import sqlite3
import logging
import threading
import time
import json
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Set, Callable, Any, Tuple
from enum import Enum
from queue import Queue, PriorityQueue
import uuid

logger = logging.getLogger("MultiAgent")


class AgentRole(Enum):
    """Agent role definitions"""
    SCOUT = "scout"              # Reconnaissance
    BREACHER = "breacher"        # Exploitation
    LATERAL = "lateral"          # Lateral movement
    ESCALATOR = "escalator"      # Privilege escalation
    EXFILTRATOR = "exfiltrator"  # Data extraction
    PERSISTENCE = "persistence"  # Maintaining access
    CLEANER = "cleaner"          # Covering tracks
    COORDINATOR = "coordinator"  # Orchestrating others


class AgentStatus(Enum):
    """Agent operational status"""
    IDLE = "idle"
    ACTIVE = "active"
    BUSY = "busy"
    WAITING = "waiting"
    ERROR = "error"
    OFFLINE = "offline"


class MessagePriority(Enum):
    """Message priority levels"""
    CRITICAL = 5
    HIGH = 4
    NORMAL = 3
    LOW = 2
    DEFERRED = 1


class TaskStatus(Enum):
    """Collaborative task status"""
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    DELEGATED = "delegated"


@dataclass
class AgentMessage:
    """Inter-agent communication message"""
    message_id: str
    sender_id: str
    recipient_id: str
    message_type: str
    content: Dict
    priority: MessagePriority = MessagePriority.NORMAL
    timestamp: float = field(default_factory=time.time)
    acknowledged: bool = False
    response: Optional[Dict] = None


@dataclass
class Agent:
    """Autonomous agent in multi-agent system"""
    agent_id: str
    name: str
    role: AgentRole
    status: AgentStatus = AgentStatus.IDLE
    capabilities: List[str] = field(default_factory=list)
    current_task: Optional[str] = None
    completed_tasks: List[str] = field(default_factory=list)
    performance_score: float = 0.5
    reliability: float = 0.9
    speed: float = 1.0
    accuracy: float = 0.95
    created_at: float = field(default_factory=time.time)
    last_heartbeat: float = field(default_factory=time.time)
    message_queue: Queue = field(default_factory=Queue)


@dataclass
class CollaborativeTask:
    """Task requiring multi-agent collaboration"""
    task_id: str
    name: str
    description: str
    required_roles: List[AgentRole]
    assigned_agents: Dict[AgentRole, str] = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    priority: int = 3
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    subtasks: List[str] = field(default_factory=list)
    results: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)


class MultiAgentSystem:
    """Coordinates multiple autonomous agents"""
    
    def __init__(self, db_path: str = "hades_knowledge.db"):
        self.db_path = db_path
        self.logger = logging.getLogger("MultiAgentSystem")
        self.enabled = False
        self.agents: Dict[str, Agent] = {}
        self.collaborative_tasks: Dict[str, CollaborativeTask] = {}
        self.message_queue: Queue = Queue()
        self.message_history: List[AgentMessage] = []
        self.coordination_enabled = True
        self.load_balancing_enabled = True
        self.conflict_resolution_enabled = True
        self.coordinator_thread: Optional[threading.Thread] = None
        self.running = False
        self._init_db()
    
    def _init_db(self):
        """Initialize multi-agent database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Agents table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS agents (
                    agent_id TEXT PRIMARY KEY,
                    name TEXT,
                    role TEXT,
                    status TEXT,
                    capabilities TEXT,
                    performance_score REAL,
                    reliability REAL,
                    speed REAL,
                    accuracy REAL,
                    created_at REAL
                )
            """)
            
            # Collaborative tasks table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS collaborative_tasks (
                    task_id TEXT PRIMARY KEY,
                    name TEXT,
                    description TEXT,
                    required_roles TEXT,
                    assigned_agents TEXT,
                    status TEXT,
                    priority INTEGER,
                    created_at REAL,
                    started_at REAL,
                    completed_at REAL
                )
            """)
            
            # Message history table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS agent_messages (
                    message_id TEXT PRIMARY KEY,
                    sender_id TEXT,
                    recipient_id TEXT,
                    message_type TEXT,
                    content TEXT,
                    priority INTEGER,
                    timestamp REAL,
                    acknowledged INTEGER
                )
            """)
            
            conn.commit()
            conn.close()
            self.logger.info("Multi-agent database initialized")
        except Exception as e:
            self.logger.error(f"Failed to init multi-agent database: {e}")
    
    def enable_multi_agent_system(self, auto_start: bool = True) -> bool:
        """Enable multi-agent collaboration"""
        try:
            self.enabled = True
            self.logger.info("Multi-agent system enabled")
            
            if auto_start:
                self.start_coordination()
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to enable multi-agent system: {e}")
            return False
    
    def start_coordination(self):
        """Start agent coordination"""
        if self.running:
            return
        
        self.running = True
        self.coordinator_thread = threading.Thread(
            target=self._coordination_loop,
            daemon=True
        )
        self.coordinator_thread.start()
        self.logger.info("Agent coordination started")
    
    def stop_coordination(self):
        """Stop agent coordination"""
        self.running = False
        if self.coordinator_thread:
            self.coordinator_thread.join(timeout=5)
        self.logger.info("Agent coordination stopped")
    
    def _coordination_loop(self):
        """Main coordination loop"""
        while self.running:
            try:
                self._process_messages()
                self._check_agent_health()
                self._manage_tasks()
                self._resolve_conflicts()
                time.sleep(1)
            except Exception as e:
                self.logger.error(f"Coordination loop error: {e}")
    
    def register_agent(self, agent_id: str, name: str, role: AgentRole,
                      capabilities: Optional[List[str]] = None) -> bool:
        """Register new agent"""
        try:
            agent = Agent(
                agent_id=agent_id,
                name=name,
                role=role,
                capabilities=capabilities or []
            )
            self.agents[agent_id] = agent
            self._store_agent(agent)
            
            self.logger.info(
                f"Registered agent: {name} "
                f"(id={agent_id}, role={role.value})"
            )
            return True
        except Exception as e:
            self.logger.error(f"Failed to register agent: {e}")
            return False
    
    def create_collaborative_task(self, task_name: str, task_description: str,
                                 required_roles: List[AgentRole],
                                 priority: int = 3,
                                 dependencies: Optional[List[str]] = None) -> str:
        """Create task requiring multi-agent collaboration"""
        try:
            task_id = f"task_{int(time.time())}_{uuid.uuid4().hex[:8]}"
            
            task = CollaborativeTask(
                task_id=task_id,
                name=task_name,
                description=task_description,
                required_roles=required_roles,
                priority=priority,
                dependencies=dependencies or []
            )
            
            self.collaborative_tasks[task_id] = task
            self._store_task(task)
            
            self.logger.info(
                f"Created collaborative task: {task_name} "
                f"(id={task_id}, roles={[r.value for r in required_roles]})"
            )
            
            # Attempt assignment
            if self.coordination_enabled:
                self._assign_agents_to_task(task)
            
            return task_id
        except Exception as e:
            self.logger.error(f"Failed to create collaborative task: {e}")
            return ""
    
    def _assign_agents_to_task(self, task: CollaborativeTask):
        """Assign best available agents to task roles"""
        try:
            for required_role in task.required_roles:
                # Find best agent for this role
                best_agent = self._select_agent_for_role(
                    required_role,
                    exclude_ids=set(task.assigned_agents.values())
                )
                
                if best_agent:
                    task.assigned_agents[required_role] = best_agent.agent_id
                    best_agent.current_task = task.task_id
                    best_agent.status = AgentStatus.ASSIGNED
                    
                    self.logger.info(
                        f"Assigned {best_agent.name} to {required_role.value} "
                        f"in task {task.task_id}"
                    )
                else:
                    self.logger.warning(
                        f"No available agent for role {required_role.value} "
                        f"in task {task.task_id}"
                    )
            
            # Update task status
            if len(task.assigned_agents) == len(task.required_roles):
                task.status = TaskStatus.ASSIGNED
                task.started_at = time.time()
                self._store_task(task)
        
        except Exception as e:
            self.logger.error(f"Agent assignment failed: {e}")
    
    def _select_agent_for_role(self, role: AgentRole,
                              exclude_ids: Optional[Set[str]] = None) -> Optional[Agent]:
        """Select best agent for specific role"""
        try:
            candidates = [
                agent for agent in self.agents.values()
                if agent.role == role and
                   agent.status != AgentStatus.OFFLINE and
                   agent.current_task is None and
                   (exclude_ids is None or agent.agent_id not in exclude_ids)
            ]
            
            if not candidates:
                return None
            
            # Sort by performance score
            candidates.sort(
                key=lambda a: (a.performance_score * a.reliability),
                reverse=True
            )
            
            return candidates[0]
        except Exception as e:
            self.logger.error(f"Agent selection failed: {e}")
            return None
    
    def send_message(self, sender_id: str, recipient_id: str,
                    message_type: str, content: Dict,
                    priority: MessagePriority = MessagePriority.NORMAL) -> bool:
        """Send message between agents"""
        try:
            message = AgentMessage(
                message_id=f"msg_{int(time.time())}_{uuid.uuid4().hex[:8]}",
                sender_id=sender_id,
                recipient_id=recipient_id,
                message_type=message_type,
                content=content,
                priority=priority
            )
            
            self.message_queue.put((priority.value * -1, message))
            self.message_history.append(message)
            self._store_message(message)
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to send message: {e}")
            return False
    
    def _process_messages(self):
        """Process pending inter-agent messages"""
        try:
            while not self.message_queue.empty():
                _, message = self.message_queue.get_nowait()
                
                if message.recipient_id in self.agents:
                    agent = self.agents[message.recipient_id]
                    agent.message_queue.put(message)
                    
                    self.logger.debug(
                        f"Message {message.message_type} from "
                        f"{message.sender_id} → {message.recipient_id}"
                    )
                    
                    # Handle critical messages immediately
                    if message.priority == MessagePriority.CRITICAL:
                        self._handle_critical_message(message, agent)
        
        except Exception as e:
            self.logger.error(f"Message processing error: {e}")
    
    def _handle_critical_message(self, message: AgentMessage, agent: Agent):
        """Handle critical messages immediately"""
        try:
            if message.message_type == "abort_task":
                if agent.current_task:
                    self.abort_task(agent.current_task)
            
            elif message.message_type == "emergency_sync":
                # Force synchronization
                self._synchronize_agents()
        
        except Exception as e:
            self.logger.error(f"Critical message handling failed: {e}")
    
    def _check_agent_health(self):
        """Monitor agent health and responsiveness"""
        try:
            current_time = time.time()
            
            for agent_id, agent in self.agents.items():
                # Check heartbeat timeout
                if current_time - agent.last_heartbeat > 60:
                    if agent.status != AgentStatus.OFFLINE:
                        agent.status = AgentStatus.OFFLINE
                        self.logger.warning(
                            f"Agent {agent.name} offline "
                            f"(no heartbeat for 60s)"
                        )
                        
                        # Reassign its task
                        if agent.current_task:
                            self._reassign_task(agent.current_task)
        
        except Exception as e:
            self.logger.error(f"Agent health check failed: {e}")
    
    def _manage_tasks(self):
        """Manage collaborative task lifecycle"""
        try:
            for task_id, task in list(self.collaborative_tasks.items()):
                if task.status == TaskStatus.PENDING:
                    if len(task.assigned_agents) == len(task.required_roles):
                        task.status = TaskStatus.IN_PROGRESS
                        task.started_at = time.time()
                        self._store_task(task)
                
                elif task.status == TaskStatus.IN_PROGRESS:
                    # Check task completion
                    if self._is_task_complete(task):
                        task.status = TaskStatus.COMPLETED
                        task.completed_at = time.time()
                        self._store_task(task)
                        
                        self.logger.info(
                            f"Task completed: {task.name} "
                            f"(duration={(task.completed_at - task.started_at):.1f}s)"
                        )
                        
                        # Free agents
                        for role, agent_id in task.assigned_agents.items():
                            if agent_id in self.agents:
                                agent = self.agents[agent_id]
                                agent.current_task = None
                                agent.status = AgentStatus.IDLE
                                agent.completed_tasks.append(task_id)
        
        except Exception as e:
            self.logger.error(f"Task management error: {e}")
    
    def _is_task_complete(self, task: CollaborativeTask) -> bool:
        """Check if collaborative task is complete"""
        try:
            # All subtasks completed or no subtasks
            if task.subtasks:
                return all(st in task.results for st in task.subtasks)
            
            # All agents reported completion
            return all(
                role in task.results
                for role in task.required_roles
            )
        except Exception as e:
            self.logger.error(f"Task completion check failed: {e}")
            return False
    
    def _resolve_conflicts(self):
        """Resolve conflicts between agents"""
        try:
            if not self.conflict_resolution_enabled:
                return
            
            # Check for agent conflicts (duplicate task assignments, etc.)
            assigned_tasks = {}
            for agent_id, agent in self.agents.items():
                if agent.current_task:
                    if agent.current_task not in assigned_tasks:
                        assigned_tasks[agent.current_task] = []
                    assigned_tasks[agent.current_task].append(agent_id)
            
            # Resolve conflicts
            for task_id, agent_ids in assigned_tasks.items():
                if len(agent_ids) > 1 and task_id in self.collaborative_tasks:
                    task = self.collaborative_tasks[task_id]
                    
                    # Keep highest performing agent
                    agent_ids.sort(
                        key=lambda aid: self.agents[aid].performance_score,
                        reverse=True
                    )
                    
                    for aid in agent_ids[1:]:
                        self.agents[aid].current_task = None
                        self.logger.info(
                            f"Conflict resolution: freed {self.agents[aid].name} "
                            f"from duplicate task"
                        )
        
        except Exception as e:
            self.logger.error(f"Conflict resolution failed: {e}")
    
    def _synchronize_agents(self):
        """Synchronize agent state across system"""
        try:
            self.logger.info("Synchronizing agent states")
            
            # Force heartbeat from all agents
            for agent_id, agent in self.agents.items():
                agent.last_heartbeat = time.time()
        
        except Exception as e:
            self.logger.error(f"Synchronization failed: {e}")
    
    def _reassign_task(self, task_id: str):
        """Reassign task from offline agent"""
        try:
            if task_id not in self.collaborative_tasks:
                return
            
            task = self.collaborative_tasks[task_id]
            
            # Find offline agents and reassign
            for role, agent_id in list(task.assigned_agents.items()):
                if agent_id in self.agents:
                    agent = self.agents[agent_id]
                    if agent.status == AgentStatus.OFFLINE:
                        # Remove and find replacement
                        del task.assigned_agents[role]
                        self._assign_agents_to_task(task)
        
        except Exception as e:
            self.logger.error(f"Task reassignment failed: {e}")
    
    def abort_task(self, task_id: str) -> bool:
        """Abort collaborative task"""
        try:
            if task_id not in self.collaborative_tasks:
                return False
            
            task = self.collaborative_tasks[task_id]
            task.status = TaskStatus.FAILED
            task.completed_at = time.time()
            
            # Free agents
            for role, agent_id in task.assigned_agents.items():
                if agent_id in self.agents:
                    self.agents[agent_id].current_task = None
                    self.agents[agent_id].status = AgentStatus.IDLE
            
            self._store_task(task)
            self.logger.warning(f"Task aborted: {task_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to abort task: {e}")
            return False
    
    def report_agent_result(self, agent_id: str, task_id: str,
                           result: Dict) -> bool:
        """Agent reports result for collaborative task"""
        try:
            if task_id not in self.collaborative_tasks:
                return False
            
            task = self.collaborative_tasks[task_id]
            task.results[agent_id] = result
            
            self.logger.debug(f"Result reported by {agent_id} for task {task_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to report agent result: {e}")
            return False
    
    def get_agent_status(self, agent_id: str) -> Optional[Dict]:
        """Get agent status"""
        if agent_id not in self.agents:
            return None
        
        agent = self.agents[agent_id]
        return {
            "agent_id": agent_id,
            "name": agent.name,
            "role": agent.role.value,
            "status": agent.status.value,
            "current_task": agent.current_task,
            "completed_tasks": len(agent.completed_tasks),
            "performance_score": agent.performance_score,
            "reliability": agent.reliability,
            "speed": agent.speed,
            "accuracy": agent.accuracy
        }
    
    def get_system_status(self) -> Dict:
        """Get multi-agent system status"""
        return {
            "enabled": self.enabled,
            "running": self.running,
            "total_agents": len(self.agents),
            "active_agents": sum(
                1 for a in self.agents.values()
                if a.status != AgentStatus.OFFLINE
            ),
            "total_tasks": len(self.collaborative_tasks),
            "active_tasks": sum(
                1 for t in self.collaborative_tasks.values()
                if t.status == TaskStatus.IN_PROGRESS
            ),
            "message_queue_size": self.message_queue.qsize()
        }
    
    def _store_agent(self, agent: Agent):
        """Store agent in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO agents
                (agent_id, name, role, status, capabilities, performance_score,
                 reliability, speed, accuracy, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (agent.agent_id, agent.name, agent.role.value,
                  agent.status.value, json.dumps(agent.capabilities),
                  agent.performance_score, agent.reliability, agent.speed,
                  agent.accuracy, agent.created_at))
            conn.commit()
            conn.close()
        except Exception as e:
            self.logger.error(f"Failed to store agent: {e}")
    
    def _store_task(self, task: CollaborativeTask):
        """Store task in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO collaborative_tasks
                (task_id, name, description, required_roles, assigned_agents,
                 status, priority, created_at, started_at, completed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (task.task_id, task.name, task.description,
                  json.dumps([r.value for r in task.required_roles]),
                  json.dumps(task.assigned_agents), task.status.value,
                  task.priority, task.created_at, task.started_at,
                  task.completed_at))
            conn.commit()
            conn.close()
        except Exception as e:
            self.logger.error(f"Failed to store task: {e}")
    
    def _store_message(self, message: AgentMessage):
        """Store message in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO agent_messages
                (message_id, sender_id, recipient_id, message_type, content,
                 priority, timestamp, acknowledged)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (message.message_id, message.sender_id, message.recipient_id,
                  message.message_type, json.dumps(message.content),
                  message.priority.value, message.timestamp,
                  int(message.acknowledged)))
            conn.commit()
            conn.close()
        except Exception as e:
            self.logger.error(f"Failed to store message: {e}")
