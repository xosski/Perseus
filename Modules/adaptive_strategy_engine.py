"""
Adaptive Strategy Engine Module
Modifies attack approaches based on real-time success rates and environmental feedback

Features:
- Dynamic exploit selection
- Strategy optimization
- Performance-based ranking
- Environmental adaptation
- A/B strategy testing
- Automatic strategy switching
"""

import sqlite3
import logging
import time
import json
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Set, Tuple
from enum import Enum
import numpy as np

logger = logging.getLogger("AdaptiveStrategy")


class StrategyType(Enum):
    """Types of attack strategies"""
    BRUTE_FORCE = "brute_force"
    EXPLOIT_KNOWN = "exploit_known"
    EXPLOIT_ZERO_DAY = "exploit_zero_day"
    SOCIAL_ENGINEERING = "social_engineering"
    RECONNAISSANCE = "reconnaissance"
    LATERAL_MOVEMENT = "lateral_movement"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    PERSISTENCE = "persistence"
    HYBRID = "hybrid"


@dataclass
class StrategyMetric:
    """Performance metric for a strategy"""
    strategy_id: str
    strategy_type: StrategyType
    target_type: str
    success_count: int = 0
    failure_count: int = 0
    total_time: float = 0.0  # seconds
    last_used: float = field(default_factory=time.time)
    confidence: float = 0.5
    adaptation_count: int = 0
    environmental_factors: Dict = field(default_factory=dict)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0
    
    @property
    def average_time(self) -> float:
        """Calculate average execution time"""
        total = self.success_count + self.failure_count
        return self.total_time / total if total > 0 else 0
    
    def update_confidence(self):
        """Recalculate confidence score"""
        success_factor = self.success_rate
        frequency_factor = min(
            (self.success_count + self.failure_count) / 100, 1.0
        )
        self.confidence = (success_factor * 0.7) + (frequency_factor * 0.3)


@dataclass
class StrategyVariant:
    """Variant of a strategy with different parameters"""
    variant_id: str
    base_strategy: str
    parameters: Dict
    success_rate: float = 0.5
    test_count: int = 0
    winner: bool = False
    timestamp: float = field(default_factory=time.time)


@dataclass
class AdaptationEvent:
    """Record of strategy adaptation"""
    event_id: str
    timestamp: float
    strategy_id: str
    reason: str
    old_parameters: Dict
    new_parameters: Dict
    success: bool


class AdaptiveStrategyEngine:
    """Dynamically adapts attack strategies based on performance"""
    
    def __init__(self, db_path: str = "hades_knowledge.db"):
        self.db_path = db_path
        self.logger = logging.getLogger("AdaptiveStrategyEngine")
        self.enabled = False
        self.strategy_metrics: Dict[str, StrategyMetric] = {}
        self.active_strategies: Set[str] = set()
        self.disabled_strategies: Set[str] = set()
        self.variants: Dict[str, List[StrategyVariant]] = {}
        self.adaptation_history: List[AdaptationEvent] = []
        self.ab_testing_enabled = False
        self.auto_switch = True
        self.performance_threshold = 0.3
        self.evaluation_interval = 60  # seconds
        self._init_db()
    
    def _init_db(self):
        """Initialize strategy database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Strategy metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS strategy_metrics (
                    strategy_id TEXT PRIMARY KEY,
                    strategy_type TEXT,
                    target_type TEXT,
                    success_count INTEGER,
                    failure_count INTEGER,
                    total_time REAL,
                    confidence REAL,
                    last_used REAL,
                    timestamp REAL
                )
            """)
            
            # Strategy variants table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS strategy_variants (
                    variant_id TEXT PRIMARY KEY,
                    base_strategy TEXT,
                    parameters TEXT,
                    success_rate REAL,
                    test_count INTEGER,
                    winner INTEGER,
                    timestamp REAL
                )
            """)
            
            # Adaptation events table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS adaptation_events (
                    event_id TEXT PRIMARY KEY,
                    timestamp REAL,
                    strategy_id TEXT,
                    reason TEXT,
                    old_parameters TEXT,
                    new_parameters TEXT,
                    success INTEGER
                )
            """)
            
            conn.commit()
            conn.close()
            self.logger.info("Adaptive strategy database initialized")
        except Exception as e:
            self.logger.error(f"Failed to init strategy database: {e}")
    
    def enable_adaptive_strategies(self,
                                  ab_testing: bool = True,
                                  auto_switch: bool = True,
                                  performance_threshold: float = 0.3) -> bool:
        """Enable adaptive strategy engine"""
        try:
            self.enabled = True
            self.ab_testing_enabled = ab_testing
            self.auto_switch = auto_switch
            self.performance_threshold = performance_threshold
            
            self.logger.info(
                f"Adaptive strategies enabled: "
                f"ab_testing={ab_testing}, auto_switch={auto_switch}, "
                f"threshold={performance_threshold}"
            )
            return True
        except Exception as e:
            self.logger.error(f"Failed to enable adaptive strategies: {e}")
            return False
    
    def register_strategy(self, strategy_id: str, strategy_type: StrategyType,
                         target_type: str) -> bool:
        """Register a new strategy"""
        try:
            metric = StrategyMetric(
                strategy_id=strategy_id,
                strategy_type=strategy_type,
                target_type=target_type
            )
            self.strategy_metrics[strategy_id] = metric
            self.active_strategies.add(strategy_id)
            self._store_metric(metric)
            
            self.logger.info(f"Registered strategy: {strategy_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to register strategy: {e}")
            return False
    
    def record_attempt(self, strategy_id: str, success: bool,
                      execution_time: float = 0.0,
                      environmental_factors: Optional[Dict] = None) -> bool:
        """Record strategy attempt"""
        if strategy_id not in self.strategy_metrics:
            return False
        
        try:
            metric = self.strategy_metrics[strategy_id]
            
            if success:
                metric.success_count += 1
            else:
                metric.failure_count += 1
            
            metric.total_time += execution_time
            metric.last_used = time.time()
            
            if environmental_factors:
                metric.environmental_factors.update(environmental_factors)
            
            metric.update_confidence()
            self._store_metric(metric)
            
            # Check if adaptation needed
            if self.enabled and self.auto_switch:
                self._evaluate_adaptation(strategy_id)
            
            self.logger.debug(
                f"Strategy attempt: {strategy_id} "
                f"(success={success}, time={execution_time:.2f}s)"
            )
            return True
        except Exception as e:
            self.logger.error(f"Failed to record strategy attempt: {e}")
            return False
    
    def _evaluate_adaptation(self, strategy_id: str):
        """Evaluate if strategy needs adaptation"""
        try:
            metric = self.strategy_metrics[strategy_id]
            
            # Check if underperforming
            if metric.success_rate < self.performance_threshold:
                if (metric.success_count + metric.failure_count) >= 10:
                    self.logger.warning(
                        f"Strategy {strategy_id} underperforming "
                        f"({metric.success_rate:.1%})"
                    )
                    self._adapt_strategy(strategy_id)
        
        except Exception as e:
            self.logger.error(f"Adaptation evaluation failed: {e}")
    
    def _adapt_strategy(self, strategy_id: str):
        """Adapt underperforming strategy"""
        try:
            metric = self.strategy_metrics[strategy_id]
            old_params = metric.environmental_factors.copy()
            
            # Strategy: switch to more aggressive approach
            if metric.success_rate < 0.2:
                metric.environmental_factors["aggressiveness"] = "high"
                metric.environmental_factors["speed"] = "maximum"
                reason = "Switched to aggressive mode (low success rate)"
            
            # Strategy: reduce speed to avoid detection
            elif metric.success_rate < 0.3:
                metric.environmental_factors["speed"] = "slow"
                metric.environmental_factors["stealth"] = "high"
                reason = "Switched to stealth mode (detection risk)"
            
            # Strategy: add reconnaissance phase
            else:
                metric.environmental_factors["recon_first"] = True
                reason = "Added reconnaissance phase"
            
            metric.adaptation_count += 1
            new_params = metric.environmental_factors.copy()
            
            # Record adaptation
            event = AdaptationEvent(
                event_id=f"{strategy_id}_{int(time.time())}",
                timestamp=time.time(),
                strategy_id=strategy_id,
                reason=reason,
                old_parameters=old_params,
                new_parameters=new_params,
                success=False  # TBD after next attempt
            )
            self.adaptation_history.append(event)
            self._store_adaptation(event)
            
            self.logger.info(
                f"Adapted strategy {strategy_id}: {reason}"
            )
        
        except Exception as e:
            self.logger.error(f"Strategy adaptation failed: {e}")
    
    def create_variant(self, base_strategy: str, variant_parameters: Dict,
                      variant_id: Optional[str] = None) -> str:
        """Create and test strategy variant"""
        try:
            if variant_id is None:
                variant_id = f"{base_strategy}_var_{int(time.time())}"
            
            variant = StrategyVariant(
                variant_id=variant_id,
                base_strategy=base_strategy,
                parameters=variant_parameters
            )
            
            if base_strategy not in self.variants:
                self.variants[base_strategy] = []
            
            self.variants[base_strategy].append(variant)
            self._store_variant(variant)
            
            self.logger.info(
                f"Created strategy variant: {variant_id} "
                f"(base={base_strategy})"
            )
            return variant_id
        
        except Exception as e:
            self.logger.error(f"Failed to create variant: {e}")
            return ""
    
    def evaluate_variant(self, variant_id: str, success: bool,
                        metrics: Optional[Dict] = None):
        """Evaluate variant performance"""
        try:
            # Find variant
            for base_strategy, variants in self.variants.items():
                for variant in variants:
                    if variant.variant_id == variant_id:
                        variant.test_count += 1
                        
                        if success:
                            variant.success_rate = (
                                (variant.success_rate * (variant.test_count - 1) + 1) /
                                variant.test_count
                            )
                        else:
                            variant.success_rate = (
                                (variant.success_rate * (variant.test_count - 1)) /
                                variant.test_count
                            )
                        
                        self._store_variant(variant)
                        
                        # Check if winner
                        if variant.success_rate > 0.7 and variant.test_count >= 5:
                            variant.winner = True
                            self.logger.info(
                                f"Variant {variant_id} selected as winner "
                                f"({variant.success_rate:.1%} success rate)"
                            )
                            
                            # Apply winning variant
                            self._apply_variant(base_strategy, variant)
                        
                        return True
            
            return False
        
        except Exception as e:
            self.logger.error(f"Variant evaluation failed: {e}")
            return False
    
    def _apply_variant(self, base_strategy: str, variant: StrategyVariant):
        """Apply winning variant as new strategy"""
        try:
            if base_strategy in self.strategy_metrics:
                metric = self.strategy_metrics[base_strategy]
                old_params = metric.environmental_factors.copy()
                metric.environmental_factors.update(variant.parameters)
                
                event = AdaptationEvent(
                    event_id=f"{base_strategy}_apply_variant_{int(time.time())}",
                    timestamp=time.time(),
                    strategy_id=base_strategy,
                    reason=f"Applied winning variant {variant.variant_id}",
                    old_parameters=old_params,
                    new_parameters=metric.environmental_factors,
                    success=True
                )
                self.adaptation_history.append(event)
                self._store_adaptation(event)
                
                self.logger.info(f"Applied variant to strategy {base_strategy}")
        
        except Exception as e:
            self.logger.error(f"Failed to apply variant: {e}")
    
    def get_best_strategy(self, target_type: str) -> Optional[str]:
        """Get best performing strategy for target type"""
        try:
            candidates = [
                (sid, metric) for sid, metric in self.strategy_metrics.items()
                if metric.target_type == target_type and
                   sid in self.active_strategies and
                   sid not in self.disabled_strategies
            ]
            
            if not candidates:
                return None
            
            best = max(candidates, key=lambda x: x[1].confidence)
            return best[0]
        
        except Exception as e:
            self.logger.error(f"Failed to get best strategy: {e}")
            return None
    
    def get_strategy_metrics(self, strategy_id: str) -> Optional[Dict]:
        """Get detailed metrics for strategy"""
        if strategy_id not in self.strategy_metrics:
            return None
        
        metric = self.strategy_metrics[strategy_id]
        return {
            "strategy_id": strategy_id,
            "strategy_type": metric.strategy_type.value,
            "target_type": metric.target_type,
            "success_rate": metric.success_rate,
            "average_time": metric.average_time,
            "confidence": metric.confidence,
            "total_attempts": metric.success_count + metric.failure_count,
            "adaptations": metric.adaptation_count,
            "environmental_factors": metric.environmental_factors,
            "last_used": metric.last_used
        }
    
    def get_all_strategies(self, target_type: Optional[str] = None) -> List[Dict]:
        """Get all strategies, optionally filtered by target"""
        strategies = []
        for sid, metric in self.strategy_metrics.items():
            if target_type is None or metric.target_type == target_type:
                strategies.append(self.get_strategy_metrics(sid))
        
        return sorted(strategies, key=lambda x: x["confidence"], reverse=True)
    
    def disable_strategy(self, strategy_id: str) -> bool:
        """Disable underperforming strategy"""
        try:
            self.disabled_strategies.add(strategy_id)
            self.active_strategies.discard(strategy_id)
            self.logger.warning(f"Disabled strategy: {strategy_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to disable strategy: {e}")
            return False
    
    def enable_strategy(self, strategy_id: str) -> bool:
        """Re-enable disabled strategy"""
        try:
            self.disabled_strategies.discard(strategy_id)
            self.active_strategies.add(strategy_id)
            self.logger.info(f"Enabled strategy: {strategy_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to enable strategy: {e}")
            return False
    
    def get_adaptation_history(self, limit: int = 50) -> List[Dict]:
        """Get adaptation history"""
        return [asdict(e) for e in self.adaptation_history[-limit:]]
    
    def get_performance_summary(self) -> Dict:
        """Get performance summary across all strategies"""
        if not self.strategy_metrics:
            return {"total_strategies": 0}
        
        metrics = list(self.strategy_metrics.values())
        success_rates = [m.success_rate for m in metrics if m.success_count > 0]
        
        return {
            "total_strategies": len(metrics),
            "active_strategies": len(self.active_strategies),
            "disabled_strategies": len(self.disabled_strategies),
            "average_success_rate": np.mean(success_rates) if success_rates else 0,
            "best_success_rate": np.max(success_rates) if success_rates else 0,
            "worst_success_rate": np.min(success_rates) if success_rates else 0,
            "total_adaptations": sum(m.adaptation_count for m in metrics),
            "variants_created": sum(len(v) for v in self.variants.values())
        }
    
    def _store_metric(self, metric: StrategyMetric):
        """Store metric in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO strategy_metrics
                (strategy_id, strategy_type, target_type, success_count,
                 failure_count, total_time, confidence, last_used, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (metric.strategy_id, metric.strategy_type.value,
                  metric.target_type, metric.success_count, metric.failure_count,
                  metric.total_time, metric.confidence, metric.last_used,
                  time.time()))
            conn.commit()
            conn.close()
        except Exception as e:
            self.logger.error(f"Failed to store metric: {e}")
    
    def _store_variant(self, variant: StrategyVariant):
        """Store variant in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO strategy_variants
                (variant_id, base_strategy, parameters, success_rate,
                 test_count, winner, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (variant.variant_id, variant.base_strategy,
                  json.dumps(variant.parameters), variant.success_rate,
                  variant.test_count, int(variant.winner), time.time()))
            conn.commit()
            conn.close()
        except Exception as e:
            self.logger.error(f"Failed to store variant: {e}")
    
    def _store_adaptation(self, event: AdaptationEvent):
        """Store adaptation event in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO adaptation_events
                (event_id, timestamp, strategy_id, reason,
                 old_parameters, new_parameters, success)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (event.event_id, event.timestamp, event.strategy_id,
                  event.reason, json.dumps(event.old_parameters),
                  json.dumps(event.new_parameters), int(event.success)))
            conn.commit()
            conn.close()
        except Exception as e:
            self.logger.error(f"Failed to store adaptation: {e}")
