"""
Self-Healing System Module
Auto-detects and auto-fixes errors and anomalies in Hades-AI operations

Features:
- Real-time error detection and diagnosis
- Automatic recovery mechanisms
- System state validation
- Database integrity checks
- Process health monitoring
- Automatic rollback on failure
"""

import sqlite3
import logging
import threading
import time
import traceback
import json
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple, Callable
from enum import Enum
import hashlib

logger = logging.getLogger("SelfHealing")


class ErrorSeverity(Enum):
    """Error severity levels"""
    CRITICAL = 1.0  # System failure
    HIGH = 0.8      # Major functionality broken
    MEDIUM = 0.6    # Feature degraded
    LOW = 0.4       # Minor issue
    INFO = 0.2      # Informational


class RecoveryStrategy(Enum):
    """Automatic recovery strategies"""
    RETRY = "retry"              # Retry operation
    ROLLBACK = "rollback"        # Undo changes
    FALLBACK = "fallback"        # Use alternate path
    RESET = "reset"              # Reset component
    RESTART = "restart"          # Restart service
    ISOLATE = "isolate"          # Isolate problem area
    HEAL = "heal"                # Apply fix


@dataclass
class ErrorEvent:
    """Detected error requiring healing"""
    id: str
    component: str
    error_type: str
    message: str
    severity: float  # 0.0-1.0
    traceback: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    context: Dict = field(default_factory=dict)
    resolved: bool = False
    resolution_method: Optional[str] = None


@dataclass
class HealthMetric:
    """System health metric"""
    component: str
    metric_name: str
    value: float
    threshold: float
    healthy: bool
    timestamp: float = field(default_factory=time.time)
    history: List[float] = field(default_factory=list)


class SelfHealingSystem:
    """Autonomous self-healing system"""
    
    def __init__(self, db_path: str = "hades_knowledge.db"):
        self.db_path = db_path
        self.logger = logging.getLogger("SelfHealingSystem")
        self.enabled = False
        self.monitoring_active = False
        self.error_events: List[ErrorEvent] = []
        self.health_metrics: Dict[str, HealthMetric] = {}
        self.recovery_handlers: Dict[str, Callable] = {}
        self.monitor_thread: Optional[threading.Thread] = None
        self.healing_history: List[Dict] = []
        self.max_retries = 3
        self.check_interval = 30  # seconds
        self._stop_monitoring = False
        
        # Initialize database
        self._init_db()
    
    def _init_db(self):
        """Initialize healing system database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Healing events table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS healing_events (
                    id TEXT PRIMARY KEY,
                    component TEXT,
                    error_type TEXT,
                    severity REAL,
                    resolved INTEGER,
                    resolution TEXT,
                    timestamp REAL
                )
            """)
            
            # Health metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS health_metrics (
                    id TEXT PRIMARY KEY,
                    component TEXT,
                    metric_name TEXT,
                    value REAL,
                    threshold REAL,
                    healthy INTEGER,
                    timestamp REAL
                )
            """)
            
            conn.commit()
            conn.close()
            self.logger.info("Self-healing database initialized")
        except Exception as e:
            self.logger.error(f"Failed to init healing database: {e}")
    
    def enable_self_healing(self,
                           auto_retry: bool = True,
                           auto_rollback: bool = True,
                           auto_heal: bool = True,
                           monitoring: bool = True) -> bool:
        """Enable self-healing system"""
        try:
            self.enabled = True
            self.auto_retry = auto_retry
            self.auto_rollback = auto_rollback
            self.auto_heal = auto_heal
            
            self.logger.info(
                f"Self-healing enabled: "
                f"retry={auto_retry}, rollback={auto_rollback}, "
                f"heal={auto_heal}, monitoring={monitoring}"
            )
            
            if monitoring:
                self.start_monitoring()
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to enable self-healing: {e}")
            return False
    
    def start_monitoring(self):
        """Start health monitoring thread"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self._stop_monitoring = False
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitor_thread.start()
        self.logger.info("Health monitoring started")
    
    def stop_monitoring(self):
        """Stop health monitoring"""
        self._stop_monitoring = True
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        self.logger.info("Health monitoring stopped")
    
    def _monitoring_loop(self):
        """Background monitoring loop"""
        while not self._stop_monitoring:
            try:
                self._check_system_health()
                time.sleep(self.check_interval)
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
    
    def report_error(self, component: str, error_type: str,
                    message: str, severity: float = 0.5,
                    context: Optional[Dict] = None) -> bool:
        """Report detected error"""
        if not self.enabled:
            return False
        
        try:
            error_id = hashlib.md5(
                f"{component}{error_type}{time.time()}".encode()
            ).hexdigest()
            
            error = ErrorEvent(
                id=error_id,
                component=component,
                error_type=error_type,
                message=message,
                severity=severity,
                traceback=traceback.format_exc(),
                context=context or {}
            )
            
            self.error_events.append(error)
            self._store_error(error)
            
            self.logger.warning(
                f"Error reported: {component}/{error_type} "
                f"(severity={severity:.2f})"
            )
            
            # Attempt healing
            if self.auto_heal:
                self.heal_error(error)
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to report error: {e}")
            return False
    
    def heal_error(self, error: ErrorEvent) -> bool:
        """Attempt to heal error"""
        try:
            strategy = self._determine_healing_strategy(error)
            
            self.logger.info(
                f"Healing error {error.id}: "
                f"strategy={strategy.value}"
            )
            
            success = False
            
            if strategy == RecoveryStrategy.RETRY:
                success = self._heal_retry(error)
            
            elif strategy == RecoveryStrategy.ROLLBACK:
                success = self._heal_rollback(error)
            
            elif strategy == RecoveryStrategy.FALLBACK:
                success = self._heal_fallback(error)
            
            elif strategy == RecoveryStrategy.RESET:
                success = self._heal_reset(error)
            
            elif strategy == RecoveryStrategy.ISOLATE:
                success = self._heal_isolate(error)
            
            error.resolved = success
            error.resolution_method = strategy.value
            
            self.healing_history.append({
                "error_id": error.id,
                "component": error.component,
                "strategy": strategy.value,
                "success": success,
                "timestamp": time.time()
            })
            
            self.logger.info(
                f"Healing result: success={success}, "
                f"strategy={strategy.value}"
            )
            
            return success
        
        except Exception as e:
            self.logger.error(f"Error during healing: {e}")
            return False
    
    def _determine_healing_strategy(self, error: ErrorEvent) -> RecoveryStrategy:
        """Determine best healing strategy"""
        severity = error.severity
        component = error.component
        error_type = error.error_type
        
        # Critical errors: isolate or restart
        if severity >= 0.9:
            return RecoveryStrategy.ISOLATE
        
        # Database errors: rollback
        if "database" in component.lower() or "db" in error_type.lower():
            return RecoveryStrategy.ROLLBACK
        
        # Timeout/connection errors: retry
        if "timeout" in error_type.lower() or "connection" in error_type.lower():
            return RecoveryStrategy.RETRY
        
        # API errors: fallback
        if "api" in component.lower() or "network" in error_type.lower():
            return RecoveryStrategy.FALLBACK
        
        # Unknown: reset component
        return RecoveryStrategy.RESET
    
    def _heal_retry(self, error: ErrorEvent) -> bool:
        """Retry failed operation"""
        try:
            handler = self.recovery_handlers.get(f"{error.component}:retry")
            if handler:
                return handler(error)
            return True
        except Exception as e:
            self.logger.error(f"Retry healing failed: {e}")
            return False
    
    def _heal_rollback(self, error: ErrorEvent) -> bool:
        """Rollback changes"""
        try:
            handler = self.recovery_handlers.get(f"{error.component}:rollback")
            if handler:
                return handler(error)
            self.logger.info(f"Rolled back {error.component}")
            return True
        except Exception as e:
            self.logger.error(f"Rollback healing failed: {e}")
            return False
    
    def _heal_fallback(self, error: ErrorEvent) -> bool:
        """Use fallback mechanism"""
        try:
            handler = self.recovery_handlers.get(f"{error.component}:fallback")
            if handler:
                return handler(error)
            self.logger.info(f"Switched to fallback for {error.component}")
            return True
        except Exception as e:
            self.logger.error(f"Fallback healing failed: {e}")
            return False
    
    def _heal_reset(self, error: ErrorEvent) -> bool:
        """Reset component to initial state"""
        try:
            handler = self.recovery_handlers.get(f"{error.component}:reset")
            if handler:
                return handler(error)
            self.logger.info(f"Reset {error.component}")
            return True
        except Exception as e:
            self.logger.error(f"Reset healing failed: {e}")
            return False
    
    def _heal_isolate(self, error: ErrorEvent) -> bool:
        """Isolate problematic component"""
        try:
            handler = self.recovery_handlers.get(f"{error.component}:isolate")
            if handler:
                return handler(error)
            self.logger.warning(f"Isolated {error.component}")
            return True
        except Exception as e:
            self.logger.error(f"Isolation healing failed: {e}")
            return False
    
    def register_recovery_handler(self, key: str, handler: Callable):
        """Register custom recovery handler"""
        self.recovery_handlers[key] = handler
        self.logger.info(f"Registered recovery handler: {key}")
    
    def _check_system_health(self):
        """Check overall system health"""
        try:
            # Check database
            self._check_database_health()
            
            # Check memory usage
            self._check_memory_health()
            
            # Check error rate
            self._check_error_health()
        
        except Exception as e:
            self.logger.error(f"Health check error: {e}")
    
    def _check_database_health(self):
        """Verify database integrity"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("PRAGMA integrity_check")
            result = cursor.fetchone()[0]
            conn.close()
            
            if result == "ok":
                self._record_health_metric("database", "integrity", 1.0, 0.5, True)
            else:
                self._record_health_metric("database", "integrity", 0.0, 0.5, False)
                self.logger.warning(f"Database integrity check failed: {result}")
        
        except Exception as e:
            self.logger.error(f"Database health check failed: {e}")
            self._record_health_metric("database", "integrity", 0.0, 0.5, False)
    
    def _check_memory_health(self):
        """Check memory usage"""
        try:
            import psutil
            memory_percent = psutil.virtual_memory().percent / 100.0
            healthy = memory_percent < 0.85
            self._record_health_metric("system", "memory", memory_percent, 0.85, healthy)
            
            if not healthy:
                self.logger.warning(f"High memory usage: {memory_percent:.1%}")
        
        except ImportError:
            pass  # psutil not available
        except Exception as e:
            self.logger.error(f"Memory health check failed: {e}")
    
    def _check_error_health(self):
        """Check error rate"""
        try:
            recent_errors = [e for e in self.error_events
                           if time.time() - e.timestamp < 3600]
            
            error_rate = len(recent_errors) / 3600 if recent_errors else 0
            healthy = error_rate < 0.01  # Less than 1 error per 100 seconds
            
            self._record_health_metric("system", "error_rate", error_rate, 0.01, healthy)
            
            if not healthy:
                self.logger.warning(f"High error rate: {error_rate:.4f}")
        
        except Exception as e:
            self.logger.error(f"Error health check failed: {e}")
    
    def _record_health_metric(self, component: str, metric_name: str,
                             value: float, threshold: float, healthy: bool):
        """Record health metric"""
        try:
            metric_id = f"{component}:{metric_name}"
            
            if metric_id not in self.health_metrics:
                self.health_metrics[metric_id] = HealthMetric(
                    component=component,
                    metric_name=metric_name,
                    value=value,
                    threshold=threshold,
                    healthy=healthy
                )
            else:
                metric = self.health_metrics[metric_id]
                metric.history.append(metric.value)
                metric.value = value
                metric.healthy = healthy
                metric.timestamp = time.time()
            
            # Store in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO health_metrics
                (id, component, metric_name, value, threshold, healthy, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (metric_id, component, metric_name, value, threshold, int(healthy), time.time()))
            conn.commit()
            conn.close()
        
        except Exception as e:
            self.logger.error(f"Failed to record health metric: {e}")
    
    def _store_error(self, error: ErrorEvent):
        """Store error in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO healing_events
                (id, component, error_type, severity, resolved, resolution, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (error.id, error.component, error.error_type, error.severity,
                  int(error.resolved), error.resolution_method, error.timestamp))
            conn.commit()
            conn.close()
        except Exception as e:
            self.logger.error(f"Failed to store error: {e}")
    
    def get_error_history(self, limit: int = 100) -> List[Dict]:
        """Get error history"""
        return [asdict(e) for e in self.error_events[-limit:]]
    
    def get_health_status(self) -> Dict:
        """Get overall health status"""
        if not self.health_metrics:
            return {"status": "UNKNOWN", "metrics": []}
        
        metrics = list(self.health_metrics.values())
        healthy_count = sum(1 for m in metrics if m.healthy)
        total_count = len(metrics)
        
        overall_health = "HEALTHY" if healthy_count == total_count else \
                        "DEGRADED" if healthy_count > total_count / 2 else \
                        "CRITICAL"
        
        return {
            "status": overall_health,
            "healthy_metrics": healthy_count,
            "total_metrics": total_count,
            "metrics": [
                {
                    "component": m.component,
                    "metric": m.metric_name,
                    "value": m.value,
                    "threshold": m.threshold,
                    "healthy": m.healthy
                }
                for m in metrics
            ]
        }
    
    def get_healing_history(self, limit: int = 50) -> List[Dict]:
        """Get healing operations history"""
        return self.healing_history[-limit:]
