"""
Predictive Execution Engine
Anticipates next optimal actions using historical memory and pattern analysis.
Enables proactive autonomous decision-making without explicit user prompts.

Features:
- Action prediction from memory patterns
- Confidence scoring for predicted actions
- Execution of predicted actions automatically
- Learning from prediction accuracy
- Context-aware decision trees
"""

import time
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import Counter
import numpy as np

logger = logging.getLogger("PredictiveExecutor")


@dataclass
class ActionPattern:
    """Represents a learned action pattern"""
    pattern_id: str
    sequence: List[str]  # Sequence of actions taken
    context: Dict[str, Any]  # Context when actions occurred
    success_count: int = 0
    failure_count: int = 0
    last_occurrence: float = field(default_factory=time.time)
    confidence: float = 0.5
    frequency: int = 0  # How many times this pattern occurred
    avg_time_between_actions: float = 0.0  # Average seconds between actions
    metadata: Dict = field(default_factory=dict)


@dataclass
class PredictedAction:
    """Represents a predicted next action"""
    action: str
    confidence: float
    reasoning: str
    estimated_duration: float
    prerequisites: List[str] = field(default_factory=list)
    expected_outcome: str = ""
    alternative_actions: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


class PatternAnalyzer:
    """Analyzes action sequences to identify patterns"""
    
    def __init__(self):
        self.action_sequences: List[List[str]] = []
        self.action_contexts: List[Dict] = []
        self.patterns: Dict[str, ActionPattern] = {}
        self.transition_matrix: Dict[str, Dict[str, int]] = {}  # Markov chain
        self._init_transition_matrix()
    
    def _init_transition_matrix(self):
        """Initialize empty transition matrix for common actions"""
        common_actions = [
            'scan_port', 'probe_service', 'enumerate_version', 'exploit_vulnerability',
            'escalate_privilege', 'establish_persistence', 'steal_data', 'lateral_move',
            'analyze_code', 'test_payload', 'defend_network', 'mitigate_threat'
        ]
        for action in common_actions:
            self.transition_matrix[action] = {}
    
    def record_action_sequence(self, actions: List[str], context: Dict) -> None:
        """Record a sequence of actions with context"""
        if len(actions) >= 2:
            self.action_sequences.append(actions)
            self.action_contexts.append(context)
            
            # Update transition matrix (Markov chain)
            for i in range(len(actions) - 1):
                current = actions[i]
                next_action = actions[i + 1]
                
                if current not in self.transition_matrix:
                    self.transition_matrix[current] = {}
                
                self.transition_matrix[current][next_action] = \
                    self.transition_matrix[current].get(next_action, 0) + 1
    
    def predict_next_action(self, current_action: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Predict next action(s) based on Markov chain"""
        if current_action not in self.transition_matrix:
            return []
        
        transitions = self.transition_matrix[current_action]
        if not transitions:
            return []
        
        total = sum(transitions.values())
        probabilities = [
            (action, count / total)
            for action, count in sorted(
                transitions.items(),
                key=lambda x: x[1],
                reverse=True
            )[:top_k]
        ]
        
        return probabilities
    
    def find_similar_patterns(self, context: Dict, similarity_threshold: float = 0.7) -> List[ActionPattern]:
        """Find patterns similar to given context"""
        similar = []
        for pattern in self.patterns.values():
            if self._context_similarity(pattern.context, context) > similarity_threshold:
                similar.append(pattern)
        
        return sorted(similar, key=lambda p: p.confidence, reverse=True)
    
    def _context_similarity(self, ctx1: Dict, ctx2: Dict) -> float:
        """Calculate context similarity (0-1)"""
        if not ctx1 or not ctx2:
            return 0.0
        
        keys1 = set(ctx1.keys())
        keys2 = set(ctx2.keys())
        
        if not keys1 or not keys2:
            return 0.0
        
        intersection = len(keys1 & keys2)
        union = len(keys1 | keys2)
        
        similarity = intersection / union if union > 0 else 0.0
        
        # Also check value similarity for common keys
        for key in keys1 & keys2:
            if ctx1[key] == ctx2[key]:
                similarity += 0.1
        
        return min(1.0, similarity)


class PredictiveExecutor:
    """
    Proactively suggests and executes next actions based on learned patterns.
    Integrates with cognitive memory for context-aware predictions.
    """
    
    def __init__(self, cognitive_layer=None, executor_fn=None):
        """
        Initialize predictive executor
        
        Args:
            cognitive_layer: CognitiveLayer instance for memory access
            executor_fn: Callable to execute predicted actions
        """
        self.cognitive_layer = cognitive_layer
        self.executor_fn = executor_fn
        self.analyzer = PatternAnalyzer()
        self.prediction_history: List[Dict] = []
        self.prediction_accuracy: float = 0.5
        self.auto_execute = False
        self.confidence_threshold = 0.65
        self.max_auto_actions = 5
        self.logger = logging.getLogger("PredictiveExecutor")
    
    def learn_action_sequence(self, actions: List[str], context: Dict, 
                             success: bool, execution_time: float = 0.0) -> str:
        """
        Learn from an executed action sequence
        
        Args:
            actions: List of action identifiers
            context: Context dict when actions were executed
            success: Whether the sequence succeeded
            execution_time: Total time taken
            
        Returns:
            Pattern ID
        """
        pattern_id = f"pattern_{int(time.time())}_{len(self.analyzer.patterns)}"
        
        pattern = ActionPattern(
            pattern_id=pattern_id,
            sequence=actions,
            context=context,
            success_count=1 if success else 0,
            failure_count=0 if success else 1,
            confidence=0.7 if success else 0.3,
            frequency=1,
            avg_time_between_actions=execution_time / max(1, len(actions) - 1)
        )
        
        self.analyzer.patterns[pattern_id] = pattern
        self.analyzer.record_action_sequence(actions, context)
        
        # Store in cognitive memory if available
        if self.cognitive_layer:
            try:
                self.cognitive_layer.remember(
                    text=f"Action sequence: {' -> '.join(actions)}. Success: {success}",
                    importance=0.8 if success else 0.4,
                    metadata={
                        'type': 'action_sequence',
                        'pattern_id': pattern_id,
                        'context': context,
                        'success': success
                    }
                )
            except Exception as e:
                self.logger.warning(f"Failed to store pattern in memory: {e}")
        
        return pattern_id
    
    def predict_next_actions(self, current_state: Dict, observation: str = "",
                            top_k: int = 3) -> List[PredictedAction]:
        """
        Predict next optimal action(s) based on current state and history
        
        Args:
            current_state: Current system/execution state
            observation: Recent observation/context
            top_k: Number of predictions to return
            
        Returns:
            List of PredictedAction objects
        """
        predictions = []
        
        try:
            # Recall relevant memories
            memory_context = ""
            if self.cognitive_layer:
                query = f"{observation} next step action"
                memories = self.cognitive_layer.recall(query, top_k=3)
                memory_context = "\n".join(
                    f"- {m.content}" for _, m in memories
                )
            
            # Find similar historical patterns
            similar_patterns = self.analyzer.find_similar_patterns(current_state)
            
            # Compile predictions
            seen_actions = set()
            
            # From similar patterns
            for pattern in similar_patterns[:3]:
                if pattern.sequence and len(pattern.sequence) > 0:
                    next_action = pattern.sequence[-1]
                    
                    if next_action not in seen_actions:
                        confidence = min(1.0, pattern.confidence * 0.9)
                        
                        predictions.append(PredictedAction(
                            action=next_action,
                            confidence=confidence,
                            reasoning=f"Similar pattern detected ({pattern.frequency} occurrences, {pattern.confidence:.1%} success)",
                            estimated_duration=pattern.avg_time_between_actions,
                            expected_outcome=f"Based on {pattern.success_count} successful executions",
                            metadata={
                                'pattern_id': pattern.pattern_id,
                                'pattern_success_rate': pattern.confidence
                            }
                        ))
                        seen_actions.add(next_action)
            
            # From Markov chain (transition matrix)
            if current_state.get('last_action'):
                markov_predictions = self.analyzer.predict_next_action(
                    current_state['last_action'],
                    top_k=3
                )
                
                for action, probability in markov_predictions:
                    if action not in seen_actions:
                        predictions.append(PredictedAction(
                            action=action,
                            confidence=probability,
                            reasoning=f"Markov transition probability: {probability:.1%}",
                            estimated_duration=5.0,  # Default estimate
                            expected_outcome="Statistical prediction from action sequences",
                            metadata={'method': 'markov_chain'}
                        ))
                        seen_actions.add(action)
            
            # Sort by confidence
            predictions.sort(key=lambda p: p.confidence, reverse=True)
            
            # Log prediction
            self.prediction_history.append({
                'timestamp': datetime.now().isoformat(),
                'state': current_state,
                'predictions': [
                    {
                        'action': p.action,
                        'confidence': p.confidence,
                        'reasoning': p.reasoning
                    }
                    for p in predictions[:top_k]
                ]
            })
            
            return predictions[:top_k]
        
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            return []
    
    def execute_predicted_action(self, prediction: PredictedAction, 
                                override_threshold: bool = False) -> Dict:
        """
        Execute a predicted action if conditions are met
        
        Args:
            prediction: PredictedAction to execute
            override_threshold: Skip confidence threshold check
            
        Returns:
            Execution result dict
        """
        if not self.executor_fn:
            return {
                'success': False,
                'error': 'No executor function configured'
            }
        
        # Check confidence threshold
        if not override_threshold and prediction.confidence < self.confidence_threshold:
            return {
                'success': False,
                'error': f'Confidence {prediction.confidence:.1%} below threshold {self.confidence_threshold:.1%}'
            }
        
        try:
            self.logger.info(
                f"🤖 Executing predicted action: {prediction.action} "
                f"(confidence: {prediction.confidence:.1%})"
            )
            
            result = self.executor_fn(prediction.action, prediction.metadata)
            
            return {
                'success': result.get('success', False),
                'action': prediction.action,
                'confidence': prediction.confidence,
                'result': result,
                'timestamp': datetime.now().isoformat()
            }
        
        except Exception as e:
            self.logger.error(f"Execution failed: {e}")
            return {
                'success': False,
                'action': prediction.action,
                'error': str(e)
            }
    
    def reinforce_prediction(self, prediction: PredictedAction, 
                            success: bool, feedback: Dict = None) -> float:
        """
        Update prediction accuracy based on outcome
        
        Args:
            prediction: Original prediction
            success: Whether it succeeded
            feedback: Optional feedback dict
            
        Returns:
            Updated confidence score
        """
        # Update accuracy metric
        adjustment = 0.1 if success else -0.1
        self.prediction_accuracy = np.clip(
            self.prediction_accuracy + adjustment,
            0.0, 1.0
        )
        
        # Find and update related patterns
        for pattern in self.analyzer.patterns.values():
            if prediction.action in pattern.sequence:
                if success:
                    pattern.success_count += 1
                    pattern.confidence = min(1.0, pattern.confidence + 0.05)
                else:
                    pattern.failure_count += 1
                    pattern.confidence = max(0.0, pattern.confidence - 0.05)
        
        # Update memory reinforcement if available
        if self.cognitive_layer and feedback:
            try:
                self.cognitive_layer.reinforce_memory(
                    prediction.metadata.get('pattern_id', ''),
                    success_score=1.0 if success else 0.0
                )
            except:
                pass
        
        self.logger.info(
            f"{'✅' if success else '❌'} Prediction reinforced: {prediction.action} "
            f"(accuracy: {self.prediction_accuracy:.1%})"
        )
        
        return self.prediction_accuracy
    
    def get_prediction_stats(self) -> Dict:
        """Get statistics about predictions"""
        return {
            'total_predictions': len(self.prediction_history),
            'overall_accuracy': self.prediction_accuracy,
            'patterns_learned': len(self.analyzer.patterns),
            'auto_execution_enabled': self.auto_execute,
            'confidence_threshold': self.confidence_threshold,
            'recent_predictions': self.prediction_history[-5:]
        }
    
    def enable_auto_execution(self, enabled: bool = True, 
                             confidence_threshold: float = 0.75) -> None:
        """Enable automatic execution of high-confidence predictions"""
        self.auto_execute = enabled
        self.confidence_threshold = confidence_threshold
        
        status = "ENABLED" if enabled else "DISABLED"
        self.logger.info(
            f"🤖 Predictive auto-execution {status} "
            f"(confidence threshold: {confidence_threshold:.1%})"
        )
    
    def get_common_sequences(self, min_frequency: int = 2) -> List[Dict]:
        """Get most common action sequences"""
        sequences = []
        
        for pattern in self.analyzer.patterns.values():
            if pattern.frequency >= min_frequency:
                sequences.append({
                    'sequence': pattern.sequence,
                    'frequency': pattern.frequency,
                    'success_rate': (
                        pattern.success_count / 
                        (pattern.success_count + pattern.failure_count)
                    ) if (pattern.success_count + pattern.failure_count) > 0 else 0,
                    'confidence': pattern.confidence,
                    'pattern_id': pattern.pattern_id
                })
        
        return sorted(sequences, key=lambda x: x['frequency'], reverse=True)
    
    def clear_history(self) -> None:
        """Clear prediction history"""
        self.prediction_history.clear()
        self.logger.info("Prediction history cleared")


def main():
    """Demonstration of predictive executor capabilities"""
    print("=" * 60)
    print("[*] PREDICTIVE EXECUTOR MODULE")
    print("=" * 60)
    print()
    
    executor = PredictiveExecutor()
    
    # Simulate learning some patterns
    print("[+] Training on example patterns...")
    
    context_1 = {'target_type': 'web_app', 'threat_level': 'high'}
    executor.learn_action_sequence(
        actions=['scan_port', 'probe_service', 'enumerate_version', 'exploit_vulnerability'],
        context=context_1,
        success=True,
        execution_time=120
    )
    
    context_2 = {'target_type': 'web_app', 'threat_level': 'medium'}
    executor.learn_action_sequence(
        actions=['scan_port', 'probe_service', 'test_payload'],
        context=context_2,
        success=True,
        execution_time=90
    )
    
    print(f"[OK] Learned {len(executor.analyzer.patterns)} patterns\n")
    
    # Test prediction
    print("[*] Testing prediction...")
    current_state = {
        'target_type': 'web_app',
        'threat_level': 'high',
        'last_action': 'scan_port'
    }
    
    predictions = executor.predict_next_actions(
        current_state,
        observation="Port scan completed, found open ports"
    )
    
    print(f"\n[*] Predicted next actions:")
    for i, pred in enumerate(predictions, 1):
        print(f"  {i}. {pred.action}")
        print(f"     Confidence: {pred.confidence:.1%}")
        print(f"     Reasoning: {pred.reasoning}")
    
    print()
    print(f"[*] Statistics:")
    stats = executor.get_prediction_stats()
    for key, value in stats.items():
        if key != 'recent_predictions':
            print(f"  {key}: {value}")
    
    print()
    print("[OK] Module loaded successfully!")
    print()
    print("=" * 60)


if __name__ == '__main__':
    main()
