"""
Autonomous Intelligence Orchestrator
Integrates Predictive Execution and Adaptive Thresholds for fully autonomous operation.

This module combines:
1. Predictive Execution - Anticipates and executes next actions
2. Adaptive Thresholds - Dynamically tunes security parameters
3. Cognitive Memory - Learns from experiences
4. Strategy Adaptation - Adjusts approach based on feedback

Features:
- Proactive threat response without user intervention
- Self-tuning security parameters
- Continuous learning and improvement
- Multi-agent coordination
- Feedback loop closure
"""

import time
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

# Import specialized modules
try:
    from modules.predictive_executor import PredictiveExecutor, PredictedAction
    HAS_PREDICTIVE = True
except ImportError:
    PredictiveExecutor = None
    HAS_PREDICTIVE = False

try:
    from modules.adaptive_thresholds import AdaptiveThresholdsEngine, ThreatMetric
    HAS_ADAPTIVE = True
except ImportError:
    AdaptiveThresholdsEngine = None
    HAS_ADAPTIVE = False

try:
    from modules.cognitive_memory import CognitiveLayer
    HAS_COGNITIVE = True
except ImportError:
    CognitiveLayer = None
    HAS_COGNITIVE = False

logger = logging.getLogger("AutonomousIntelligence")


class AutonomyLevel(Enum):
    """Levels of autonomous operation"""
    MANUAL = 0          # User-directed only
    ASSISTED = 1        # Suggestions provided
    SEMI_AUTONOMOUS = 2  # Auto-execute low-risk actions
    FULLY_AUTONOMOUS = 3 # Full autonomous operation


class DecisionContext(Enum):
    """Contexts that trigger autonomous decisions"""
    THREAT_DETECTED = "threat_detected"
    ANOMALY_FOUND = "anomaly_found"
    PATTERN_MATCH = "pattern_match"
    RESOURCE_CONSTRAINED = "resource_constrained"
    PERFORMANCE_ISSUE = "performance_issue"
    ROUTINE_OPERATION = "routine_operation"


@dataclass
class AutonomousDecision:
    """Represents an autonomous decision made by the system"""
    decision_id: str
    decision_type: str
    context: DecisionContext
    predicted_action: Optional[PredictedAction]
    threshold_adjustment: Optional[Dict]
    confidence: float
    execution_timestamp: float
    success: Optional[bool] = None
    feedback: Dict = field(default_factory=dict)
    rationale: str = ""


class AutonomousIntelligence:
    """
    Master orchestrator for autonomous decision-making and execution.
    Coordinates predictive actions with adaptive security parameters.
    """
    
    def __init__(self, 
                 cognitive_layer=None,
                 action_executor: Callable = None,
                 autonomy_level: AutonomyLevel = AutonomyLevel.SEMI_AUTONOMOUS):
        """
        Initialize Autonomous Intelligence system
        
        Args:
            cognitive_layer: CognitiveLayer for memory
            action_executor: Callable to execute predicted actions
            autonomy_level: Initial autonomy level
        """
        self.cognitive_layer = cognitive_layer
        self.action_executor = action_executor
        self.autonomy_level = autonomy_level
        
        # Initialize sub-systems
        self.predictor: Optional[PredictiveExecutor] = None
        self.thresholds: Optional[AdaptiveThresholdsEngine] = None
        
        if HAS_PREDICTIVE:
            self.predictor = PredictiveExecutor(
                cognitive_layer=cognitive_layer,
                executor_fn=action_executor
            )
        
        if HAS_ADAPTIVE:
            self.thresholds = AdaptiveThresholdsEngine()
        
        # Decision tracking
        self.decisions: List[AutonomousDecision] = []
        self.decision_stats: Dict[str, int] = {
            'total': 0,
            'successful': 0,
            'failed': 0,
            'executed': 0,
            'suggested': 0
        }
        
        # Performance metrics
        self.performance_metrics: Dict[str, float] = {
            'prediction_accuracy': 0.5,
            'execution_success_rate': 0.5,
            'avg_decision_time_ms': 0.0,
            'autonomous_actions_count': 0
        }
        
        self.logger = logging.getLogger("AutonomousIntelligence")
    
    def process_observation(self, observation: Dict, context: DecisionContext) -> Optional[AutonomousDecision]:
        """
        Process an observation and make autonomous decisions
        
        Args:
            observation: Current system observation/state
            context: Context of the observation
            
        Returns:
            AutonomousDecision if one was made, None otherwise
        """
        decision_time_start = time.time()
        
        try:
            # 1. Analyze threat level and update thresholds
            threat_level = self._analyze_threat_level(observation)
            
            if self.thresholds:
                self._update_adaptive_thresholds(observation, threat_level)
            
            # 2. Predict next action
            predicted_action = None
            if self.predictor:
                predictions = self.predictor.predict_next_actions(
                    current_state=observation,
                    observation=json.dumps(observation, default=str),
                    top_k=1
                )
                
                if predictions:
                    predicted_action = predictions[0]
            
            # 3. Decide on execution
            should_execute = self._should_execute_prediction(
                predicted_action, threat_level, context
            )
            
            # 4. Create decision record
            decision = AutonomousDecision(
                decision_id=f"decision_{int(time.time() * 1000)}",
                decision_type="action_execution" if should_execute else "threshold_adjustment",
                context=context,
                predicted_action=predicted_action,
                threshold_adjustment=self._get_recent_threshold_adjustments(),
                confidence=predicted_action.confidence if predicted_action else 0.0,
                execution_timestamp=time.time(),
                rationale=self._generate_rationale(context, predicted_action, threat_level)
            )
            
            # 5. Execute if appropriate
            if should_execute and predicted_action:
                decision.success = self._execute_decision(decision)
                if decision.success:
                    self.decision_stats['executed'] += 1
                    self.decision_stats['successful'] += 1
                    self.predictor.reinforce_prediction(
                        predicted_action,
                        success=True,
                        feedback={'observation': observation, 'context': context.value}
                    )
                else:
                    self.decision_stats['failed'] += 1
                    if self.predictor:
                        self.predictor.reinforce_prediction(
                            predicted_action,
                            success=False
                        )
            else:
                self.decision_stats['suggested'] += 1
                self.logger.info(
                    f"🤖 Prediction suggested (not executed): {predicted_action.action if predicted_action else 'None'}"
                )
            
            # Track decision
            self.decisions.append(decision)
            self.decision_stats['total'] += 1
            
            # Update performance metrics
            decision_time = (time.time() - decision_time_start) * 1000
            self.performance_metrics['avg_decision_time_ms'] = (
                self.performance_metrics['avg_decision_time_ms'] * 0.8 +
                decision_time * 0.2
            )
            
            return decision
        
        except Exception as e:
            self.logger.error(f"Decision processing failed: {e}")
            return None
    
    def _analyze_threat_level(self, observation: Dict) -> float:
        """Analyze observation to determine threat level (0-1)"""
        threat_level = 0.0
        
        # Extract threat indicators
        if 'threat_count' in observation:
            threat_level = max(threat_level, observation['threat_count'] / 100)
        
        if 'anomaly_score' in observation:
            threat_level = max(threat_level, observation['anomaly_score'])
        
        if 'attack_detected' in observation and observation['attack_detected']:
            threat_level = max(threat_level, 0.8)
        
        if 'resource_usage' in observation:
            resource_usage = observation['resource_usage']
            if resource_usage > 0.8:
                threat_level = max(threat_level, 0.6)
        
        return min(threat_level, 1.0)
    
    def _update_adaptive_thresholds(self, observation: Dict, threat_level: float) -> None:
        """Update adaptive thresholds based on current threat level"""
        if not self.thresholds:
            return
        
        # Record generic threat metric
        self.thresholds.record_threat(
            metric_name="overall_threat_level",
            value=threat_level,
            threat_type="composite",
            context=observation
        )
        
        # Record specific threat types if detected
        if 'threats' in observation:
            for threat in observation['threats']:
                self.thresholds.record_threat(
                    metric_name=threat.get('type', 'unknown'),
                    value=threat.get('severity', 0.5),
                    threat_type=threat.get('type', 'unknown')
                )
    
    def _should_execute_prediction(self, prediction: Optional[PredictedAction],
                                   threat_level: float,
                                   context: DecisionContext) -> bool:
        """Determine if a prediction should be executed"""
        if not prediction:
            return False
        
        # Adjust confidence threshold based on autonomy level
        threshold_map = {
            AutonomyLevel.MANUAL: 1.5,  # Never auto-execute
            AutonomyLevel.ASSISTED: 0.95,
            AutonomyLevel.SEMI_AUTONOMOUS: 0.65,
            AutonomyLevel.FULLY_AUTONOMOUS: 0.5
        }
        
        confidence_threshold = threshold_map.get(self.autonomy_level, 0.65)
        
        # Lower threshold for critical threats
        if threat_level > 0.8:
            confidence_threshold = max(0.4, confidence_threshold - 0.2)
        
        should_exec = prediction.confidence >= confidence_threshold
        
        if should_exec:
            self.logger.info(
                f"✅ Will execute: {prediction.action} "
                f"(confidence: {prediction.confidence:.1%}, threat: {threat_level:.1%})"
            )
        
        return should_exec
    
    def _execute_decision(self, decision: AutonomousDecision) -> bool:
        """Execute an autonomous decision"""
        if not decision.predicted_action or not self.action_executor:
            return False
        
        try:
            self.logger.info(
                f"⚙️ Executing autonomous action: {decision.predicted_action.action}"
            )
            
            result = self.action_executor(
                decision.predicted_action.action,
                decision.predicted_action.metadata
            )
            
            success = result.get('success', False) if isinstance(result, dict) else bool(result)
            
            if success:
                self.logger.info(f"✅ Action executed successfully: {decision.predicted_action.action}")
                self.performance_metrics['autonomous_actions_count'] += 1
            else:
                self.logger.warning(f"❌ Action execution failed: {decision.predicted_action.action}")
            
            return success
        
        except Exception as e:
            self.logger.error(f"Execution error: {e}")
            return False
    
    def _get_recent_threshold_adjustments(self) -> Optional[Dict]:
        """Get most recent threshold adjustment if any"""
        if not self.thresholds or not self.thresholds.adjustment_history:
            return None
        
        latest = self.thresholds.adjustment_history[-1]
        return {
            'threshold': latest.threshold_name,
            'old_value': latest.old_value,
            'new_value': latest.new_value,
            'reason': latest.reason
        }
    
    def _generate_rationale(self, context: DecisionContext, 
                          prediction: Optional[PredictedAction],
                          threat_level: float) -> str:
        """Generate human-readable explanation of decision"""
        rationale = f"Context: {context.value} | Threat level: {threat_level:.1%}"
        
        if prediction:
            rationale += f" | Predicted: {prediction.action} ({prediction.confidence:.1%})"
        
        if self.thresholds:
            threat_summary = self.thresholds.get_threat_summary()
            if threat_summary['threshold_adjustments'] > 0:
                rationale += f" | Adjusted {threat_summary['threshold_adjustments']} thresholds"
        
        return rationale
    
    def set_autonomy_level(self, level: AutonomyLevel) -> None:
        """Set the autonomy level"""
        self.autonomy_level = level
        self.logger.info(f"🤖 Autonomy level set to: {level.name}")
    
    def set_predictor_confidence_threshold(self, threshold: float) -> None:
        """Set minimum confidence for predictions"""
        if self.predictor:
            self.predictor.confidence_threshold = threshold
            self.logger.info(f"Predictor confidence threshold: {threshold:.1%}")
    
    def get_autonomy_status(self) -> Dict:
        """Get current autonomy status"""
        return {
            'autonomy_level': self.autonomy_level.name,
            'total_decisions': self.decision_stats['total'],
            'executed': self.decision_stats['executed'],
            'successful': self.decision_stats['successful'],
            'failed': self.decision_stats['failed'],
            'success_rate': (
                self.decision_stats['successful'] / self.decision_stats['executed']
                if self.decision_stats['executed'] > 0 else 0
            ),
            'prediction_accuracy': self.performance_metrics['prediction_accuracy'],
            'avg_decision_time_ms': self.performance_metrics['avg_decision_time_ms']
        }
    
    def get_decision_log(self, limit: int = 10) -> List[Dict]:
        """Get recent autonomous decisions"""
        return [
            {
                'decision_id': d.decision_id,
                'type': d.decision_type,
                'context': d.context.value,
                'action': d.predicted_action.action if d.predicted_action else None,
                'confidence': d.predicted_action.confidence if d.predicted_action else 0,
                'success': d.success,
                'timestamp': datetime.fromtimestamp(d.execution_timestamp).isoformat(),
                'rationale': d.rationale
            }
            for d in self.decisions[-limit:]
        ]
    
    def get_comprehensive_status(self) -> Dict:
        """Get comprehensive status of autonomous intelligence system"""
        status = {
            'autonomy': self.get_autonomy_status(),
            'modules': {
                'predictive_executor': HAS_PREDICTIVE,
                'adaptive_thresholds': HAS_ADAPTIVE,
                'cognitive_memory': HAS_COGNITIVE
            }
        }
        
        if self.thresholds:
            status['threat_summary'] = self.thresholds.get_threat_summary()
            status['thresholds'] = self.thresholds.get_current_thresholds()
        
        if self.predictor:
            status['prediction_stats'] = self.predictor.get_prediction_stats()
        
        return status
    
    def clear_decision_history(self) -> None:
        """Clear decision history"""
        self.decisions.clear()
        self.logger.info("Decision history cleared")


def main():
    """Demonstration of autonomous intelligence system"""
    print("=" * 70)
    print("🤖 AUTONOMOUS INTELLIGENCE ORCHESTRATOR")
    print("=" * 70)
    print()
    
    system = AutonomousIntelligence()
    
    print("📋 Available Autonomy Levels:")
    for level in AutonomyLevel:
        print(f"   • {level.name} (value: {level.value})")
    
    print()
    print("🎯 Available Decision Contexts:")
    for ctx in DecisionContext:
        print(f"   • {ctx.value}")
    
    print()
    print("✨ Loaded Modules:")
    status = system.get_comprehensive_status()
    for module, available in status['modules'].items():
        marker = "✅" if available else "❌"
        print(f"   {marker} {module}")
    
    print()
    print("⚙️  Current Configuration:")
    autonomy = system.get_autonomy_status()
    for key, value in autonomy.items():
        if not key.endswith('rate'):
            print(f"   {key}: {value}")
    
    print()
    print("✅ Autonomous Intelligence ready!")
    print("   Set autonomy_level to enable autonomous operation:")
    print("   system.set_autonomy_level(AutonomyLevel.SEMI_AUTONOMOUS)")
    print()
    print("=" * 70)


if __name__ == '__main__':
    main()
