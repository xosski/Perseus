"""
Autonomous Scheduler Module
Automatically schedules and orchestrates Hades-AI operations based on conditions

Features:
- Task scheduling (cron-like)
- Conditional execution
- Operation orchestration
- Resource-aware scheduling
- Automatic retries
- Priority-based execution
"""

import sqlite3
import logging
import threading
import time
import json
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Callable, Any
from enum import Enum

try:
    import schedule
except ImportError:
    class _FallbackJob:
        """Small fallback for the optional schedule package."""

        def __init__(self, interval: int = 1):
            self.interval = max(1, int(interval or 1))
            self.unit_seconds = 3600
            self.job_func = None
            self.args = ()
            self.kwargs = {}
            self.next_run = time.time() + self.unit_seconds * self.interval

        @property
        def hour(self):
            self.unit_seconds = 3600
            return self

        @property
        def day(self):
            self.unit_seconds = 86400
            return self

        @property
        def week(self):
            self.unit_seconds = 604800
            return self

        @property
        def minutes(self):
            self.unit_seconds = 60
            return self

        def at(self, _time_string: str):
            return self

        def do(self, job_func: Callable, *args, **kwargs):
            self.job_func = job_func
            self.args = args
            self.kwargs = kwargs
            self.next_run = time.time() + self.unit_seconds * self.interval
            return self

        def run_if_due(self):
            if self.job_func and time.time() >= self.next_run:
                self.job_func(*self.args, **self.kwargs)
                self.next_run = time.time() + self.unit_seconds * self.interval

    class _FallbackScheduler:
        def __init__(self):
            self.jobs = []

        def every(self, interval: int = 1):
            job = _FallbackJob(interval)
            self.jobs.append(job)
            return job

        def run_pending(self):
            for job in list(self.jobs):
                job.run_if_due()

    class schedule:
        Scheduler = _FallbackScheduler

logger = logging.getLogger("AutonomousScheduler")


class TaskPriority(Enum):
    """Task execution priority"""
    CRITICAL = 5
    HIGH = 4
    NORMAL = 3
    LOW = 2
    DEFERRED = 1


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELED = "canceled"


class ExecutionCondition(Enum):
    """Conditions for task execution"""
    ALWAYS = "always"
    ON_DEMAND = "on_demand"
    TIME_BASED = "time_based"
    RESOURCE_BASED = "resource_based"
    EVENT_BASED = "event_based"
    DEPENDENCY_BASED = "dependency_based"


@dataclass
class ScheduledTask:
    """Scheduled operation task"""
    task_id: str
    name: str
    operation: Callable
    schedule_time: str  # Cron-like schedule (e.g., "0 */6 * * *")
    priority: TaskPriority = TaskPriority.NORMAL
    enabled: bool = True
    max_retries: int = 3
    timeout: int = 300  # seconds
    parameters: Dict = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    condition: ExecutionCondition = ExecutionCondition.TIME_BASED
    condition_check: Optional[Callable] = None
    created_at: float = field(default_factory=time.time)
    last_execution: Optional[float] = None
    next_execution: Optional[float] = None
    execution_count: int = 0
    success_count: int = 0
    failure_count: int = 0


@dataclass
class ExecutionRecord:
    """Record of task execution"""
    execution_id: str
    task_id: str
    status: TaskStatus
    start_time: float
    end_time: Optional[float] = None
    duration: float = 0.0
    result: Optional[Any] = None
    error: Optional[str] = None
    retry_count: int = 0


class AutonomousScheduler:
    """Autonomous task scheduler and orchestrator"""
    
    def __init__(self, db_path: str = "hades_knowledge.db"):
        self.db_path = db_path
        self.logger = logging.getLogger("AutonomousScheduler")
        self.enabled = False
        self.scheduler = schedule.Scheduler()
        self.scheduled_tasks: Dict[str, ScheduledTask] = {}
        self.execution_history: List[ExecutionRecord] = []
        self.running = False
        self.scheduler_thread: Optional[threading.Thread] = None
        self.task_handlers: Dict[str, Callable] = {}
        self.event_listeners: Dict[str, List[Callable]] = {}
        self._init_db()
    
    def _init_db(self):
        """Initialize scheduler database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Scheduled tasks table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS scheduled_tasks (
                    task_id TEXT PRIMARY KEY,
                    name TEXT,
                    schedule_time TEXT,
                    priority INTEGER,
                    enabled INTEGER,
                    max_retries INTEGER,
                    timeout INTEGER,
                    parameters TEXT,
                    dependencies TEXT,
                    condition TEXT,
                    created_at REAL,
                    last_execution REAL,
                    execution_count INTEGER
                )
            """)
            
            # Execution history table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS execution_history (
                    execution_id TEXT PRIMARY KEY,
                    task_id TEXT,
                    status TEXT,
                    start_time REAL,
                    end_time REAL,
                    duration REAL,
                    retry_count INTEGER,
                    error TEXT
                )
            """)
            
            conn.commit()
            conn.close()
            self.logger.info("Scheduler database initialized")
        except Exception as e:
            self.logger.error(f"Failed to init scheduler database: {e}")
    
    def enable_scheduling(self, auto_start: bool = True) -> bool:
        """Enable autonomous scheduling"""
        try:
            self.enabled = True
            self.logger.info("Autonomous scheduling enabled")
            
            if auto_start:
                self.start_scheduler()
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to enable scheduling: {e}")
            return False
    
    def start_scheduler(self):
        """Start background scheduler"""
        if self.running:
            return
        
        self.running = True
        self.scheduler_thread = threading.Thread(
            target=self._scheduler_loop,
            daemon=True
        )
        self.scheduler_thread.start()
        self.logger.info("Scheduler started")
    
    def stop_scheduler(self):
        """Stop background scheduler"""
        self.running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        self.logger.info("Scheduler stopped")
    
    def _scheduler_loop(self):
        """Main scheduler loop"""
        while self.running:
            try:
                self.scheduler.run_pending()
                time.sleep(1)
            except Exception as e:
                self.logger.error(f"Scheduler loop error: {e}")
    
    def schedule_task(self, task_id: str, name: str, operation: Callable,
                     schedule_time: str, priority: TaskPriority = TaskPriority.NORMAL,
                     parameters: Optional[Dict] = None,
                     dependencies: Optional[List[str]] = None,
                     max_retries: int = 3,
                     timeout: int = 300) -> bool:
        """Schedule a new autonomous task"""
        try:
            task = ScheduledTask(
                task_id=task_id,
                name=name,
                operation=operation,
                schedule_time=schedule_time,
                priority=priority,
                max_retries=max_retries,
                timeout=timeout,
                parameters=parameters or {},
                dependencies=dependencies or []
            )
            
            self.scheduled_tasks[task_id] = task
            self._register_task_with_scheduler(task)
            self._store_task(task)
            
            self.logger.info(
                f"Scheduled task: {name} "
                f"(id={task_id}, schedule={schedule_time})"
            )
            return True
        except Exception as e:
            self.logger.error(f"Failed to schedule task: {e}")
            return False
    
    def _register_task_with_scheduler(self, task: ScheduledTask):
        """Register task with background scheduler"""
        try:
            # Parse schedule_time and register
            if task.schedule_time == "@hourly":
                self.scheduler.every().hour.do(
                    self._execute_task_wrapper, task
                )
            elif task.schedule_time == "@daily":
                self.scheduler.every().day.do(
                    self._execute_task_wrapper, task
                )
            elif task.schedule_time == "@weekly":
                self.scheduler.every().week.do(
                    self._execute_task_wrapper, task
                )
            elif task.schedule_time.startswith("*/"):
                # Every N minutes
                minutes = int(task.schedule_time.split("/")[1])
                self.scheduler.every(minutes).minutes.do(
                    self._execute_task_wrapper, task
                )
            elif ":" in task.schedule_time:
                # Specific time daily
                self.scheduler.every().day.at(task.schedule_time).do(
                    self._execute_task_wrapper, task
                )
            else:
                # Default: hourly
                self.scheduler.every().hour.do(
                    self._execute_task_wrapper, task
                )
        except Exception as e:
            self.logger.error(f"Failed to register task with scheduler: {e}")
    
    def _execute_task_wrapper(self, task: ScheduledTask):
        """Wrapper for task execution with error handling"""
        if not task.enabled:
            return
        
        try:
            # Check dependencies
            if task.dependencies:
                if not self._check_dependencies(task):
                    self.logger.info(
                        f"Task {task.task_id} skipped (dependencies not met)"
                    )
                    return
            
            # Check condition
            if task.condition != ExecutionCondition.ALWAYS:
                if task.condition_check and not task.condition_check():
                    self.logger.debug(
                        f"Task {task.task_id} skipped (condition not met)"
                    )
                    return
            
            # Execute with retries
            self._execute_task(task)
        
        except Exception as e:
            self.logger.error(f"Task wrapper error: {e}")
    
    def _check_dependencies(self, task: ScheduledTask) -> bool:
        """Check if task dependencies are met"""
        try:
            for dep_id in task.dependencies:
                if dep_id not in self.scheduled_tasks:
                    return False
                
                dep_task = self.scheduled_tasks[dep_id]
                if not dep_task.enabled:
                    return False
                
                # Check if dependent task executed recently
                if dep_task.last_execution is None:
                    return False
                
                # Must have executed within last hour
                if time.time() - dep_task.last_execution > 3600:
                    return False
            
            return True
        except Exception as e:
            self.logger.error(f"Dependency check failed: {e}")
            return False
    
    def _execute_task(self, task: ScheduledTask):
        """Execute a scheduled task with retries"""
        execution_id = f"{task.task_id}_{int(time.time())}"
        record = ExecutionRecord(
            execution_id=execution_id,
            task_id=task.task_id,
            status=TaskStatus.RUNNING,
            start_time=time.time()
        )
        
        try:
            self.logger.info(f"Executing task: {task.name}")
            
            # Execute with timeout
            result = self._execute_with_timeout(
                task.operation,
                task.parameters,
                task.timeout
            )
            
            record.status = TaskStatus.SUCCESS
            record.result = result
            
            task.last_execution = time.time()
            task.execution_count += 1
            task.success_count += 1
            
            self.logger.info(
                f"Task completed: {task.name} "
                f"(execution_count={task.execution_count})"
            )
            
            # Trigger success event
            self._trigger_event(f"task_success_{task.task_id}")
        
        except Exception as e:
            # Retry logic
            if record.retry_count < task.max_retries:
                record.retry_count += 1
                self.logger.warning(
                    f"Task {task.name} failed, retrying "
                    f"(attempt {record.retry_count}/{task.max_retries})"
                )
                time.sleep(2 ** record.retry_count)  # Exponential backoff
                self._execute_task(task)
            else:
                record.status = TaskStatus.FAILED
                record.error = str(e)
                
                task.failure_count += 1
                
                self.logger.error(
                    f"Task failed: {task.name} - {e}"
                )
                
                # Trigger failure event
                self._trigger_event(f"task_failed_{task.task_id}")
        
        finally:
            record.end_time = time.time()
            record.duration = record.end_time - record.start_time
            
            self.execution_history.append(record)
            self._store_execution(record)
    
    def _execute_with_timeout(self, operation: Callable, parameters: Dict,
                             timeout: int) -> Any:
        """Execute operation with timeout"""
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Operation timed out after {timeout}s")
        
        try:
            # For Windows, use threading-based timeout
            result = [None]
            error = [None]
            
            def target():
                try:
                    result[0] = operation(**parameters)
                except Exception as e:
                    error[0] = e
            
            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(timeout=timeout)
            
            if thread.is_alive():
                raise TimeoutError(f"Operation timed out after {timeout}s")
            
            if error[0]:
                raise error[0]
            
            return result[0]
        
        except Exception as e:
            raise e
    
    def trigger_task(self, task_id: str) -> bool:
        """Manually trigger a task"""
        try:
            if task_id not in self.scheduled_tasks:
                return False
            
            task = self.scheduled_tasks[task_id]
            self._execute_task(task)
            return True
        except Exception as e:
            self.logger.error(f"Failed to trigger task: {e}")
            return False
    
    def enable_task(self, task_id: str) -> bool:
        """Enable a task"""
        if task_id in self.scheduled_tasks:
            self.scheduled_tasks[task_id].enabled = True
            self.logger.info(f"Task enabled: {task_id}")
            return True
        return False
    
    def disable_task(self, task_id: str) -> bool:
        """Disable a task"""
        if task_id in self.scheduled_tasks:
            self.scheduled_tasks[task_id].enabled = False
            self.logger.info(f"Task disabled: {task_id}")
            return True
        return False
    
    def register_event_listener(self, event: str, callback: Callable):
        """Register callback for event"""
        if event not in self.event_listeners:
            self.event_listeners[event] = []
        self.event_listeners[event].append(callback)
    
    def _trigger_event(self, event: str):
        """Trigger event callbacks"""
        if event in self.event_listeners:
            for callback in self.event_listeners[event]:
                try:
                    callback(event)
                except Exception as e:
                    self.logger.error(f"Event callback error: {e}")
    
    def get_task_status(self, task_id: str) -> Optional[Dict]:
        """Get current task status"""
        if task_id not in self.scheduled_tasks:
            return None
        
        task = self.scheduled_tasks[task_id]
        return {
            "task_id": task_id,
            "name": task.name,
            "enabled": task.enabled,
            "priority": task.priority.name,
            "execution_count": task.execution_count,
            "success_count": task.success_count,
            "failure_count": task.failure_count,
            "last_execution": task.last_execution,
            "next_execution": task.next_execution,
            "schedule": task.schedule_time
        }
    
    def get_all_tasks(self) -> List[Dict]:
        """Get all scheduled tasks"""
        return [self.get_task_status(tid) for tid in self.scheduled_tasks.keys()]
    
    def get_execution_history(self, task_id: Optional[str] = None,
                             limit: int = 100) -> List[Dict]:
        """Get execution history"""
        history = self.execution_history
        
        if task_id:
            history = [e for e in history if e.task_id == task_id]
        
        return [asdict(e) for e in history[-limit:]]
    
    def get_scheduler_status(self) -> Dict:
        """Get scheduler status"""
        return {
            "running": self.running,
            "enabled": self.enabled,
            "total_tasks": len(self.scheduled_tasks),
            "active_tasks": sum(1 for t in self.scheduled_tasks.values() if t.enabled),
            "total_executions": len(self.execution_history),
            "pending_jobs": len(self.scheduler.jobs)
        }
    
    def _store_task(self, task: ScheduledTask):
        """Store task in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO scheduled_tasks
                (task_id, name, schedule_time, priority, enabled, max_retries,
                 timeout, parameters, dependencies, condition, created_at,
                 last_execution, execution_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (task.task_id, task.name, task.schedule_time,
                  task.priority.value, int(task.enabled), task.max_retries,
                  task.timeout, json.dumps(task.parameters),
                  json.dumps(task.dependencies), task.condition.value,
                  task.created_at, task.last_execution, task.execution_count))
            conn.commit()
            conn.close()
        except Exception as e:
            self.logger.error(f"Failed to store task: {e}")
    
    def _store_execution(self, record: ExecutionRecord):
        """Store execution record in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO execution_history
                (execution_id, task_id, status, start_time, end_time,
                 duration, retry_count, error)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (record.execution_id, record.task_id, record.status.value,
                  record.start_time, record.end_time, record.duration,
                  record.retry_count, record.error))
            conn.commit()
            conn.close()
        except Exception as e:
            self.logger.error(f"Failed to store execution: {e}")
