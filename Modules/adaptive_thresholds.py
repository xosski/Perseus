"""
Adaptive Thresholds Engine
Dynamically adjusts security parameters based on real-time threat analysis.
Enables autonomous response to changing threat landscape.

Features:
- Dynamic threshold calculation based on threat metrics
- Real-time parameter tuning
- Anomaly detection for threshold calibration
- Historical trend analysis
- Automatic escalation/de-escalation
- Threat-specific threshold adjustment
"""

import time
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import deque
import numpy as np
from enum import Enum

logger = logging.getLogger("AdaptiveThresholds")


class ThresholdType(Enum):
    """Types of configurable thresholds"""
    ALERT_THRESHOLD = "alert_threshold"
    BLOCK_THRESHOLD = "block_threshold"
    RATE_LIMIT = "rate_limit"
    DETECTION_SENSITIVITY = "detection_sensitivity"
    RESPONSE_DELAY = "response_delay"
    ESCALATION_LEVEL = "escalation_level"
    TIMEOUT = "timeout"
    MAX_CONNECTIONS = "max_connections"
    RESOURCE_LIMIT = "resource_limit"


class ThreatLevel(Enum):
    """Threat severity levels"""
    MINIMAL = (0, 0.1)
    LOW = (0.1, 0.3)
    MEDIUM = (0.3, 0.6)
    HIGH = (0.6, 0.8)
    CRITICAL = (0.8, 1.0)


@dataclass
class ThresholdConfig:
    """Configuration for a single threshold"""
    name: str
    threshold_type: ThresholdType
    current_value: float
    min_value: float = 0.0
    max_value: float = 1.0
    adjustment_step: float = 0.05
    adjustment_cooldown: int = 30  # seconds
    last_adjusted: float = field(default_factory=time.time)
    adjustment_count: int = 0
    created_at: float = field(default_factory=time.time)
    metadata: Dict = field(default_factory=dict)


@dataclass
class ThreatMetric:
    """A single threat observation"""
    metric_name: str
    value: float
    threat_type: str
    timestamp: float = field(default_factory=time.time)
    context: Dict = field(default_factory=dict)
    severity: str = "MEDIUM"


@dataclass
class ThresholdAdjustment:
    """Records a threshold adjustment"""
    threshold_name: str
    old_value: float
    new_value: float
    reason: str
    timestamp: float = field(default_factory=time.time)
    success: bool = True
    metrics_used: List[str] = field(default_factory=list)


class ThreatAnalyzer:
    """Analyzes threat metrics to determine appropriate thresholds"""
    
    def __init__(self, window_size: int = 100):
        self.metrics: Dict[str, deque] = {}
        self.window_size = window_size
        self.threat_trends: Dict[str, List[float]] = {}
        self.anomaly_scores: Dict[str, float] = {}
    
    def record_metric(self, metric: ThreatMetric) -> None:
        """Record a threat metric"""
        if metric.metric_name not in self.metrics:
            self.metrics[metric.metric_name] = deque(maxlen=self.window_size)
            self.threat_trends[metric.metric_name] = []
        
        self.metrics[metric.metric_name].append(metric.value)
        self.threat_trends[metric.metric_name].append(metric.value)
    
    def get_average_threat_level(self, metric_name: str) -> float:
        """Get average threat level for a metric"""
        if metric_name not in self.metrics or not self.metrics[metric_name]:
            return 0.0
        
        values = list(self.metrics[metric_name])
        return float(np.mean(values)) if values else 0.0
    
    def get_threat_trend(self, metric_name: str) -> str:
        """Determine if threat is increasing, decreasing, or stable"""
        if metric_name not in self.threat_trends or len(self.threat_trends[metric_name]) < 3:
            return "stable"
        
        recent = self.threat_trends[metric_name][-10:]
        if len(recent) < 2:
            return "stable"
        
        # Simple trend analysis
        avg_first_half = np.mean(recent[:len(recent)//2])
        avg_second_half = np.mean(recent[len(recent)//2:])
        
        difference = avg_second_half - avg_first_half
        
        if difference > 0.1:
            return "increasing"
        elif difference < -0.1:
            return "decreasing"
        else:
            return "stable"
    
    def detect_anomalies(self, metric_name: str, sensitivity: float = 2.0) -> List[int]:
        """Detect anomalous values using Z-score"""
        if metric_name not in self.metrics or len(self.metrics[metric_name]) < 2:
            return []
        
        values = list(self.metrics[metric_name])
        mean = np.mean(values)
        std = np.std(values)
        
        if std == 0:
            return []
        
        anomalies = []
        for i, val in enumerate(values):
            z_score = abs((val - mean) / std)
            if z_score > sensitivity:
                anomalies.append(i)
        
        return anomalies
    
    def calculate_threat_composite(self, threat_types: List[str]) -> float:
        """Calculate composite threat level from multiple threat types"""
        if not threat_types:
            return 0.0
        
        values = []
        for threat_type in threat_types:
            avg = self.get_average_threat_level(threat_type)
            values.append(avg)
        
        return float(np.mean(values)) if values else 0.0
    
    def get_threat_volatility(self, metric_name: str) -> float:
        """Calculate how volatile/unstable a threat metric is"""
        if metric_name not in self.metrics or len(self.metrics[metric_name]) < 2:
            return 0.0
        
        values = list(self.metrics[metric_name])
        return float(np.std(values))


class AdaptiveThresholdsEngine:
    """
    Dynamically adjusts security thresholds based on threat analysis.
    Enables autonomous response tuning without manual configuration.
    """
    
    def __init__(self):
        self.thresholds: Dict[str, ThresholdConfig] = {}
        self.analyzer = ThreatAnalyzer()
        self.adjustment_history: List[ThresholdAdjustment] = []
        self.threat_history: List[ThreatMetric] = []
        self.adaptive_enabled = True
        self.aggressiveness = 0.5  # 0.0 = conservative, 1.0 = aggressive
        self.logger = logging.getLogger("AdaptiveThresholdsEngine")
        self._init_default_thresholds()
    
    def _init_default_thresholds(self) -> None:
        """Initialize default threshold configurations"""
        defaults = [
            ThresholdConfig(
                name="detection_sensitivity",
                threshold_type=ThresholdType.DETECTION_SENSITIVITY,
                current_value=0.5,
                min_value=0.1,
                max_value=0.95
            ),
            ThresholdConfig(
                name="alert_threshold",
                threshold_type=ThresholdType.ALERT_THRESHOLD,
                current_value=0.6,
                min_value=0.3,
                max_value=0.95
            ),
            ThresholdConfig(
                name="block_threshold",
                threshold_type=ThresholdType.BLOCK_THRESHOLD,
                current_value=0.7,
                min_value=0.5,
                max_value=0.99
            ),
            ThresholdConfig(
                name="rate_limit",
                threshold_type=ThresholdType.RATE_LIMIT,
                current_value=100.0,
                min_value=10.0,
                max_value=1000.0,
                adjustment_step=10.0
            ),
            ThresholdConfig(
                name="response_delay",
                threshold_type=ThresholdType.RESPONSE_DELAY,
                current_value=5.0,
                min_value=0.1,
                max_value=30.0,
                adjustment_step=0.5
            ),
            ThresholdConfig(
                name="escalation_level",
                threshold_type=ThresholdType.ESCALATION_LEVEL,
                current_value=2,
                min_value=0,
                max_value=3,
                adjustment_step=1
            ),
        ]
        
        for threshold in defaults:
            self.thresholds[threshold.name] = threshold
    
    def record_threat(self, metric_name: str, value: float, 
                     threat_type: str, severity: str = "MEDIUM",
                     context: Dict = None) -> None:
        """
        Record a threat metric
        
        Args:
            metric_name: Name of the metric
            value: Metric value (typically 0-1 for severity)
            threat_type: Type of threat
            severity: Severity level
            context: Additional context
        """
        metric = ThreatMetric(
            metric_name=metric_name,
            value=value,
            threat_type=threat_type,
            severity=severity,
            context=context or {}
        )
        
        self.threat_history.append(metric)
        self.analyzer.record_metric(metric)
        
        # Trigger adaptive adjustments if enabled
        if self.adaptive_enabled:
            self._evaluate_threshold_adjustments(metric)
    
    def _evaluate_threshold_adjustments(self, metric: ThreatMetric) -> None:
        """Evaluate if thresholds need adjustment based on metric"""
        threat_level = self._determine_threat_level(metric.value)
        trend = self.analyzer.get_threat_trend(metric.metric_name)
        volatility = self.analyzer.get_threat_volatility(metric.metric_name)
        
        # Different adjustment strategies based on threat level and trend
        if threat_level == ThreatLevel.CRITICAL:
            self._aggressive_threshold_adjustment(metric)
        elif threat_level == ThreatLevel.HIGH:
            if trend == "increasing" or volatility > 0.2:
                self._moderate_threshold_adjustment(metric)
        elif threat_level == ThreatLevel.MEDIUM:
            if trend == "increasing":
                self._gentle_threshold_adjustment(metric)
        elif threat_level == ThreatLevel.LOW:
            if trend == "decreasing":
                self._relax_threshold_adjustment(metric)
    
    def _determine_threat_level(self, value: float) -> ThreatLevel:
        """Determine threat level from a value"""
        for level in ThreatLevel:
            min_val, max_val = level.value
            if min_val <= value < max_val:
                return level
        return ThreatLevel.CRITICAL
    
    def _aggressive_threshold_adjustment(self, metric: ThreatMetric) -> None:
        """Aggressively lower thresholds for critical threats"""
        self.logger.warning(
            f"🔴 CRITICAL threat detected: {metric.metric_name} = {metric.value:.2f}"
        )
        
        # Lower detection and alert thresholds
        self._adjust_threshold("detection_sensitivity", -0.1, "Critical threat - increase sensitivity")
        self._adjust_threshold("alert_threshold", -0.05, "Critical threat - lower alert level")
        self._adjust_threshold("block_threshold", -0.05, "Critical threat - lower block threshold")
        self._adjust_threshold("response_delay", -1.0, "Critical threat - faster response")
        self._adjust_threshold("escalation_level", 1, "Critical threat - escalate defense")
        
        # Increase rate limits
        self._adjust_threshold("rate_limit", -50.0, "Critical threat - stricter rate limit")
    
    def _moderate_threshold_adjustment(self, metric: ThreatMetric) -> None:
        """Moderately adjust thresholds for high threats"""
        self.logger.warning(
            f"🟠 HIGH threat detected: {metric.metric_name} = {metric.value:.2f}"
        )
        
        self._adjust_threshold("detection_sensitivity", -0.05, "High threat - increase sensitivity")
        self._adjust_threshold("alert_threshold", -0.03, "High threat detected")
        self._adjust_threshold("block_threshold", -0.03, "High threat - lower block threshold")
        self._adjust_threshold("response_delay", -0.5, "High threat - faster response")
    
    def _gentle_threshold_adjustment(self, metric: ThreatMetric) -> None:
        """Gently adjust thresholds for medium threats"""
        self.logger.info(
            f"🟡 MEDIUM threat detected: {metric.metric_name} = {metric.value:.2f}"
        )
        
        self._adjust_threshold("detection_sensitivity", -0.02, "Medium threat trend")
    
    def _relax_threshold_adjustment(self, metric: ThreatMetric) -> None:
        """Relax thresholds when threats are decreasing"""
        self.logger.info(
            f"🟢 Threat decreasing: {metric.metric_name} = {metric.value:.2f}"
        )
        
        # Gradually increase thresholds (make detection less sensitive)
        self._adjust_threshold("detection_sensitivity", 0.02, "Threat trend decreasing")
        self._adjust_threshold("response_delay", 0.2, "Threat level lowering - allow longer responses")
        self._adjust_threshold("rate_limit", 10.0, "Threat decreasing - relax rate limit")
    
    def _adjust_threshold(self, threshold_name: str, adjustment: float, 
                         reason: str) -> bool:
        """
        Adjust a specific threshold
        
        Args:
            threshold_name: Name of threshold to adjust
            adjustment: Amount to adjust (can be negative)
            reason: Reason for adjustment
            
        Returns:
            True if adjustment was made
        """
        if threshold_name not in self.thresholds:
            self.logger.warning(f"Unknown threshold: {threshold_name}")
            return False
        
        threshold = self.thresholds[threshold_name]
        
        # Check cooldown
        if time.time() - threshold.last_adjusted < threshold.adjustment_cooldown:
            return False
        
        old_value = threshold.current_value
        new_value = np.clip(
            old_value + adjustment,
            threshold.min_value,
            threshold.max_value
        )
        
        # Only adjust if value actually changes
        if abs(new_value - old_value) < 0.001:
            return False
        
        threshold.current_value = new_value
        threshold.last_adjusted = time.time()
        threshold.adjustment_count += 1
        
        # Record adjustment
        adj = ThresholdAdjustment(
            threshold_name=threshold_name,
            old_value=old_value,
            new_value=new_value,
            reason=reason,
            success=True,
            metrics_used=[m.metric_name for m in self.threat_history[-5:]]
        )
        self.adjustment_history.append(adj)
        
        change_pct = ((new_value - old_value) / old_value * 100) if old_value != 0 else 0
        self.logger.info(
            f"⚙️  Adjusted {threshold_name}: {old_value:.3f} → {new_value:.3f} "
            f"({change_pct:+.1f}%) | {reason}"
        )
        
        return True
    
    def set_aggressiveness(self, level: float) -> None:
        """
        Set overall aggressiveness of adaptive thresholds
        
        Args:
            level: 0.0 (conservative) to 1.0 (aggressive)
        """
        self.aggressiveness = np.clip(level, 0.0, 1.0)
        self.logger.info(f"Aggressiveness set to {self.aggressiveness:.1%}")
    
    def get_current_thresholds(self) -> Dict[str, float]:
        """Get all current threshold values"""
        return {
            name: threshold.current_value
            for name, threshold in self.thresholds.items()
        }
    
    def get_threshold_status(self, threshold_name: str) -> Dict:
        """Get detailed status of a specific threshold"""
        if threshold_name not in self.thresholds:
            return {}
        
        t = self.thresholds[threshold_name]
        
        return {
            'name': t.name,
            'type': t.threshold_type.value,
            'current_value': t.current_value,
            'min_value': t.min_value,
            'max_value': t.max_value,
            'adjustment_count': t.adjustment_count,
            'last_adjusted': datetime.fromtimestamp(t.last_adjusted).isoformat(),
            'utilization': (t.current_value - t.min_value) / (t.max_value - t.min_value)
        }
    
    def get_all_threshold_status(self) -> List[Dict]:
        """Get status of all thresholds"""
        return [
            self.get_threshold_status(name)
            for name in self.thresholds.keys()
        ]
    
    def get_threat_summary(self) -> Dict:
        """Get summary of recent threat activity"""
        if not self.threat_history:
            return {
                'total_threats': 0,
                'threat_level': 'MINIMAL',
                'recent_trends': {}
            }
        
        recent = self.threat_history[-50:]
        
        threat_types = {}
        for metric in recent:
            if metric.threat_type not in threat_types:
                threat_types[metric.threat_type] = []
            threat_types[metric.threat_type].append(metric.value)
        
        composite = self.analyzer.calculate_threat_composite(
            [m.metric_name for m in recent]
        )
        
        return {
            'total_threats': len(self.threat_history),
            'recent_threats': len(recent),
            'composite_threat_level': composite,
            'threat_level': self._determine_threat_level(composite).name,
            'threat_by_type': {
                ttype: {
                    'count': len(values),
                    'avg': float(np.mean(values)),
                    'max': float(np.max(values))
                }
                for ttype, values in threat_types.items()
            },
            'threshold_adjustments': len(self.adjustment_history)
        }
    
    def get_adjustment_history(self, limit: int = 20) -> List[Dict]:
        """Get recent threshold adjustment history"""
        return [
            {
                'threshold': adj.threshold_name,
                'old_value': adj.old_value,
                'new_value': adj.new_value,
                'change': adj.new_value - adj.old_value,
                'reason': adj.reason,
                'timestamp': datetime.fromtimestamp(adj.timestamp).isoformat()
            }
            for adj in self.adjustment_history[-limit:]
        ]
    
    def enable_adaptive(self, enabled: bool = True) -> None:
        """Enable or disable adaptive threshold adjustment"""
        self.adaptive_enabled = enabled
        status = "ENABLED" if enabled else "DISABLED"
        self.logger.info(f"Adaptive thresholds {status}")
    
    def reset_thresholds(self) -> None:
        """Reset all thresholds to defaults"""
        self._init_default_thresholds()
        self.logger.info("All thresholds reset to defaults")


def main():
    """Demonstration of adaptive thresholds engine"""
    print("=" * 60)
    print("⚙️  ADAPTIVE THRESHOLDS ENGINE")
    print("=" * 60)
    print()
    
    engine = AdaptiveThresholdsEngine()
    
    print("🎯 Default Thresholds:")
    for name, threshold in engine.get_current_thresholds().items():
        print(f"  {name}: {threshold}")
    
    print()
    print("📊 Simulating threat activity...")
    
    # Simulate some threats
    threats = [
        ("port_scan_rate", 0.4, "reconnaissance"),
        ("failed_auth_attempts", 0.5, "brute_force"),
        ("port_scan_rate", 0.6, "reconnaissance"),
        ("failed_auth_attempts", 0.7, "brute_force"),
        ("port_scan_rate", 0.8, "reconnaissance"),  # CRITICAL
    ]
    
    for metric_name, value, threat_type in threats:
        engine.record_threat(metric_name, value, threat_type)
        time.sleep(0.1)
    
    print()
    print("📈 Threat Summary:")
    summary = engine.get_threat_summary()
    for key, value in summary.items():
        if key != 'threat_by_type':
            print(f"  {key}: {value}")
    
    print()
    print("🔧 Adjusted Thresholds:")
    for name, threshold in engine.get_current_thresholds().items():
        print(f"  {name}: {threshold}")
    
    print()
    print("📝 Adjustment History (last 5):")
    for adj in engine.get_adjustment_history(5):
        print(f"  {adj['threshold']}: {adj['old_value']:.3f} → {adj['new_value']:.3f}")
        print(f"    Reason: {adj['reason']}")
    
    print()
    print("✅ Module loaded successfully!")
    print()
    print("=" * 60)


if __name__ == '__main__':
    main()
