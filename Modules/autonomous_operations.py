"""
Autonomous Operations Module
Threat response, continuous learning, and decision-making for HadesAI

Features:
- Auto-respond to detected threats
- Continuous learning from findings
- Intelligent decision-making for exploitation
"""

import sqlite3
import logging
import threading
import time
import json
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import numpy as np

logger = logging.getLogger("AutonomousOps")


class ThreatLevel(Enum):
    """Threat severity levels"""
    CRITICAL = 1.0  # Immediate action required
    HIGH = 0.8      # Quick response needed
    MEDIUM = 0.6    # Standard response
    LOW = 0.4       # Monitoring response
    INFO = 0.2      # Informational


class ResponseAction(Enum):
    """Autonomous response actions"""
    BLOCK_IP = "block_ip"
    ISOLATE = "isolate"
    PATCH = "patch"
    ALERT = "alert"
    INVESTIGATE = "investigate"
    EXPLOIT = "exploit"
    DOCUMENT = "document"


@dataclass
class ThreatEvent:
    """Detected threat requiring response"""
    id: str
    threat_type: str
    severity: float  # 0.0-1.0
    source_ip: Optional[str] = None
    target: Optional[str] = None
    pattern: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    metadata: Dict = field(default_factory=dict)
    
    def threat_level(self) -> ThreatLevel:
        """Get threat level from severity"""
        if self.severity >= 0.9:
            return ThreatLevel.CRITICAL
        elif self.severity >= 0.7:
            return ThreatLevel.HIGH
        elif self.severity >= 0.5:
            return ThreatLevel.MEDIUM
        elif self.severity >= 0.3:
            return ThreatLevel.LOW
        else:
            return ThreatLevel.INFO


@dataclass
class LearningRecord:
    """Record of exploit or finding for learning"""
    id: str
    exploit_name: str
    target_type: str
    success_rate: float = 0.5  # 0.0-1.0
    attempts: int = 0
    successes: int = 0
    last_used: float = field(default_factory=time.time)
    confidence: float = 0.5
    metadata: Dict = field(default_factory=dict)
    
    def update_success(self):
        """Record successful use"""
        self.attempts += 1
        self.successes += 1
        self.success_rate = self.successes / self.attempts if self.attempts > 0 else 0
        self.last_used = time.time()
        self.confidence = min(1.0, self.success_rate * (self.attempts / 100))


class ThreatResponseEngine:
    """Autonomous threat response system"""
    
    def __init__(self, db_path: str = "hades_knowledge.db"):
        self.db_path = db_path
        self.logger = logging.getLogger("ThreatResponseEngine")
        self.enabled = False
        self.response_history: List[Dict] = []
        self.blocked_ips = set()
        self.isolation_list = set()
        self.auto_patch = False
        self.auto_exploit = False
        self.response_threshold = 0.7  # Min severity to auto-respond
    
    def enable_auto_response(self, 
                            block_ips: bool = True,
                            isolate: bool = False,
                            auto_patch: bool = True,
                            auto_exploit: bool = False,
                            threshold: float = 0.7) -> bool:
        """Enable autonomous threat response"""
        try:
            self.enabled = True
            self.auto_patch = auto_patch
            self.auto_exploit = auto_exploit
            self.response_threshold = threshold
            
            self.logger.info(
                f"Autonomous threat response enabled: "
                f"patch={auto_patch}, exploit={auto_exploit}, "
                f"threshold={threshold}"
            )
            return True
        except Exception as e:
            self.logger.error(f"Failed to enable auto-response: {e}")
            return False
    
    def process_threat(self, threat: ThreatEvent) -> List[ResponseAction]:
        """Process threat and determine responses"""
        if not self.enabled or threat.severity < self.response_threshold:
            return []
        
        actions = []
        threat_level = threat.threat_level()
        
        # Determine response actions based on threat level
        if threat_level == ThreatLevel.CRITICAL:
            actions = [ResponseAction.ALERT, ResponseAction.INVESTIGATE]
            if self.auto_patch:
                actions.append(ResponseAction.PATCH)
            if threat.source_ip:
                actions.append(ResponseAction.BLOCK_IP)
        
        elif threat_level == ThreatLevel.HIGH:
            actions = [ResponseAction.ALERT]
            if threat.source_ip:
                actions.append(ResponseAction.BLOCK_IP)
            if self.auto_patch:
                actions.append(ResponseAction.PATCH)
        
        elif threat_level == ThreatLevel.MEDIUM:
            actions = [ResponseAction.INVESTIGATE]
            if self.auto_patch:
                actions.append(ResponseAction.PATCH)
        
        elif threat_level == ThreatLevel.LOW:
            actions = [ResponseAction.DOCUMENT]
        
        # Execute actions
        for action in actions:
            self._execute_action(action, threat)
        
        # Log response
        self.response_history.append({
            "threat_id": threat.id,
            "threat_type": threat.threat_type,
            "severity": threat.severity,
            "actions": [a.value for a in actions],
            "timestamp": time.time()
        })
        
        self.logger.info(
            f"Threat response: {threat.threat_type} "
            f"(severity={threat.severity:.2f}) → {[a.value for a in actions]}"
        )
        
        return actions
    
    def _execute_action(self, action: ResponseAction, threat: ThreatEvent):
        """Execute response action"""
        try:
            if action == ResponseAction.BLOCK_IP and threat.source_ip:
                self.blocked_ips.add(threat.source_ip)
                self.logger.warning(f"Auto-blocked IP: {threat.source_ip}")
            
            elif action == ResponseAction.ISOLATE and threat.target:
                self.isolation_list.add(threat.target)
                self.logger.warning(f"Auto-isolated target: {threat.target}")
            
            elif action == ResponseAction.PATCH:
                self.logger.info(f"Auto-patch recommended for: {threat.pattern}")
                # In real scenario, would apply patch
            
            elif action == ResponseAction.ALERT:
                self.logger.warning(
                    f"ALERT: {threat.threat_type} detected "
                    f"(severity={threat.severity:.2f})"
                )
            
            elif action == ResponseAction.INVESTIGATE:
                self.logger.info(f"Investigation started for: {threat.threat_type}")
            
            elif action == ResponseAction.EXPLOIT:
                self.logger.info(f"Auto-exploitation enabled for: {threat.pattern}")
            
            elif action == ResponseAction.DOCUMENT:
                self.logger.debug(f"Documented threat: {threat.threat_type}")
        
        except Exception as e:
            self.logger.error(f"Error executing action {action}: {e}")
    
    def get_blocked_ips(self) -> List[str]:
        """Get list of auto-blocked IPs"""
        return list(self.blocked_ips)
    
    def get_response_history(self) -> List[Dict]:
        """Get threat response history"""
        return self.response_history[-100:]  # Last 100
    
    def block_ip(self, ip: str):
        """Manually block IP"""
        self.blocked_ips.add(ip)
        self.logger.info(f"Manually blocked IP: {ip}")
    
    def unblock_ip(self, ip: str):
        """Manually unblock IP"""
        self.blocked_ips.discard(ip)
        self.logger.info(f"Unblocked IP: {ip}")


class ContinuousLearningEngine:
    """Continuous learning from findings and exploits"""
    
    def __init__(self, db_path: str = "hades_knowledge.db"):
        self.db_path = db_path
        self.logger = logging.getLogger("LearningEngine")
        self.enabled = False
        self.learning_records: Dict[str, LearningRecord] = {}
        self.pattern_generation = False
        self.success_feedback_loop = False
        self.update_frequency = 3600  # seconds
    
    def enable_continuous_learning(self,
                                   auto_update_exploits: bool = True,
                                   pattern_generation: bool = False,
                                   success_feedback_loop: bool = True) -> bool:
        """Enable continuous learning"""
        try:
            self.enabled = True
            self.pattern_generation = pattern_generation
            self.success_feedback_loop = success_feedback_loop
            
            self.logger.info(
                f"Continuous learning enabled: "
                f"update_exploits={auto_update_exploits}, "
                f"pattern_gen={pattern_generation}, "
                f"feedback={success_feedback_loop}"
            )
            return True
        except Exception as e:
            self.logger.error(f"Failed to enable learning: {e}")
            return False
    
    def record_attempt(self, exploit_name: str, target_type: str, 
                      success: bool, metadata: Dict = None) -> bool:
        """Record exploit attempt"""
        if not self.enabled:
            return False
        
        try:
            key = f"{exploit_name}_{target_type}"
            
            if key not in self.learning_records:
                self.learning_records[key] = LearningRecord(
                    id=key,
                    exploit_name=exploit_name,
                    target_type=target_type,
                    metadata=metadata or {}
                )
            
            record = self.learning_records[key]
            
            if success:
                record.update_success()
                self.logger.info(
                    f"Learning: {exploit_name} success rate "
                    f"{record.success_rate:.2%} ({record.successes}/{record.attempts})"
                )
                
                # Trigger feedback loop if enabled
                if self.success_feedback_loop:
                    self._update_exploit_ranking(record)
            
            return True
        except Exception as e:
            self.logger.error(f"Error recording attempt: {e}")
            return False
    
    def _update_exploit_ranking(self, record: LearningRecord):
        """Update exploit ranking based on success"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Update ranking in database
            cursor.execute("""
                UPDATE security_patterns 
                SET confidence = ? 
                WHERE pattern_type = 'exploit' AND signature LIKE ?
            """, (record.confidence, f"%{record.exploit_name}%"))
            
            conn.commit()
            conn.close()
            
            self.logger.debug(f"Updated ranking for {record.exploit_name}")
        except Exception as e:
            self.logger.debug(f"Could not update ranking: {e}")
    
    def generate_patterns(self) -> List[Dict]:
        """Generate new attack patterns from existing data"""
        if not self.pattern_generation:
            return []
        
        try:
            patterns = []
            
            # Analyze successful exploits for common patterns
            successful = [r for r in self.learning_records.values() 
                         if r.success_rate > 0.7]
            
            if len(successful) > 2:
                # Find common target types
                target_types = {}
                for record in successful:
                    t = record.target_type
                    target_types[t] = target_types.get(t, 0) + 1
                
                # Generate combinations
                for target, count in target_types.items():
                    if count >= 2:
                        pattern = {
                            "generated": True,
                            "target_type": target,
                            "success_rate": np.mean(
                                [r.success_rate for r in successful 
                                 if r.target_type == target]
                            ),
                            "confidence": 0.5  # Conservative for generated
                        }
                        patterns.append(pattern)
                        self.logger.info(f"Generated pattern: {pattern}")
            
            return patterns
        except Exception as e:
            self.logger.error(f"Error generating patterns: {e}")
            return []
    
    def get_top_exploits(self, limit: int = 10) -> List[LearningRecord]:
        """Get top-performing exploits"""
        sorted_exploits = sorted(
            self.learning_records.values(),
            key=lambda x: (x.success_rate, x.confidence),
            reverse=True
        )
        return sorted_exploits[:limit]
    
    def get_learning_stats(self) -> Dict:
        """Get learning statistics"""
        if not self.learning_records:
            return {"total": 0, "average_success": 0}
        
        successes = [r.success_rate for r in self.learning_records.values()]
        return {
            "total_exploits": len(self.learning_records),
            "average_success_rate": np.mean(successes),
            "best_exploit": max(successes) if successes else 0,
            "worst_exploit": min(successes) if successes else 0,
            "learning_enabled": self.enabled
        }


class DecisionMakingAgent:
    """Intelligent decision-making for exploitation"""
    
    def __init__(self, learning_engine: ContinuousLearningEngine,
                 threat_response: ThreatResponseEngine):
        self.learning = learning_engine
        self.threat_response = threat_response
        self.logger = logging.getLogger("DecisionAgent")
        self.enabled = False
        self.vulnerability_threshold = 7.0  # CVSS score
        self.auto_prioritize = False
        self.explain_reasoning = False
        self.decision_history: List[Dict] = []
    
    def enable_autonomous_decisions(self,
                                    vulnerability_threshold: float = 7.0,
                                    auto_prioritize: bool = True,
                                    explain_reasoning: bool = True) -> bool:
        """Enable autonomous decision-making"""
        try:
            self.enabled = True
            self.vulnerability_threshold = vulnerability_threshold
            self.auto_prioritize = auto_prioritize
            self.explain_reasoning = explain_reasoning
            
            self.logger.info(
                f"Decision agent enabled: "
                f"cvss_threshold={vulnerability_threshold}, "
                f"auto_prioritize={auto_prioritize}"
            )
            return True
        except Exception as e:
            self.logger.error(f"Failed to enable decision agent: {e}")
            return False
    
    def evaluate_target(self, target: Dict) -> Dict:
        """Evaluate target and decide exploitation strategy"""
        if not self.enabled:
            return {"decision": "SKIP", "reasoning": "Decision agent disabled"}
        
        try:
            decision = {
                "target": target.get("name"),
                "decision": "SKIP",
                "reasoning": [],
                "recommended_exploits": [],
                "risk_level": "UNKNOWN",
                "confidence": 0.0
            }
            
            # Check vulnerability severity
            cvss = target.get("cvss_score", 0)
            if cvss < self.vulnerability_threshold:
                decision["reasoning"].append(
                    f"CVSS {cvss} below threshold {self.vulnerability_threshold}"
                )
                return decision
            
            decision["reasoning"].append(
                f"High severity vulnerability (CVSS {cvss})"
            )
            
            # Find matching exploits from learning
            target_type = target.get("type", "unknown")
            matching_exploits = [
                e for e in self.learning.get_top_exploits(20)
                if e.target_type == target_type and e.success_rate > 0.5
            ]
            
            if not matching_exploits:
                decision["reasoning"].append(f"No known exploits for {target_type}")
                decision["decision"] = "INVESTIGATE"
                return decision
            
            # Select best exploit
            best_exploit = matching_exploits[0]
            decision["recommended_exploits"] = [
                {
                    "name": e.exploit_name,
                    "success_rate": e.success_rate,
                    "confidence": e.confidence
                }
                for e in matching_exploits[:3]
            ]
            
            decision["reasoning"].append(
                f"Best exploit: {best_exploit.exploit_name} "
                f"({best_exploit.success_rate:.1%} success rate)"
            )
            
            # Determine risk level
            if cvss >= 9.0 and best_exploit.success_rate >= 0.8:
                decision["risk_level"] = "CRITICAL"
                decision["decision"] = "EXPLOIT"
                decision["confidence"] = best_exploit.confidence
            elif cvss >= 7.5 and best_exploit.success_rate >= 0.6:
                decision["risk_level"] = "HIGH"
                decision["decision"] = "EXPLOIT"
                decision["confidence"] = best_exploit.confidence
            elif best_exploit.success_rate >= 0.7:
                decision["risk_level"] = "MEDIUM"
                decision["decision"] = "EXPLOIT"
                decision["confidence"] = best_exploit.confidence
            else:
                decision["risk_level"] = "MEDIUM"
                decision["decision"] = "INVESTIGATE"
                decision["confidence"] = 0.5
            
            # Log decision
            self.decision_history.append(decision)
            
            if self.explain_reasoning:
                self.logger.info(
                    f"Decision: {decision['decision']} for {target.get('name')} | "
                    f"Reasoning: {' | '.join(decision['reasoning'])}"
                )
            
            return decision
        
        except Exception as e:
            self.logger.error(f"Error evaluating target: {e}")
            return {"decision": "ERROR", "reasoning": str(e)}
    
    def recommend_strategy(self, targets: List[Dict]) -> Dict:
        """Recommend overall exploitation strategy"""
        if not self.enabled or not targets:
            return {"strategy": "MANUAL", "targets": []}
        
        try:
            evaluated = [self.evaluate_target(t) for t in targets]
            
            # Prioritize targets
            if self.auto_prioritize:
                evaluated.sort(
                    key=lambda x: (
                        1 if x["decision"] == "EXPLOIT" else 0,
                        x.get("confidence", 0)
                    ),
                    reverse=True
                )
            
            strategy = {
                "strategy": "AUTONOMOUS",
                "timestamp": time.time(),
                "target_count": len(targets),
                "targets": evaluated,
                "total_confidence": np.mean([e.get("confidence", 0) for e in evaluated]),
                "recommended_order": [
                    t.get("target") for t in evaluated
                    if t.get("decision") == "EXPLOIT"
                ]
            }
            
            self.logger.info(
                f"Strategy: {strategy['strategy']} | "
                f"Recommended {len(strategy['recommended_order'])} targets"
            )
            
            return strategy
        
        except Exception as e:
            self.logger.error(f"Error recommending strategy: {e}")
            return {"strategy": "ERROR", "reasoning": str(e)}
    
    def get_decision_history(self) -> List[Dict]:
        """Get decision history"""
        return self.decision_history[-50:]  # Last 50


def main():
    """Module initialization"""
    logger.info("Autonomous Operations module loaded successfully")
    return {
        "status": "ready",
        "module": "autonomous_operations",
        "version": "1.0",
        "description": "Threat Response, Learning, and Decision-Making",
        "components": [
            "ThreatResponseEngine",
            "ContinuousLearningEngine",
            "DecisionMakingAgent"
        ]
    }


if __name__ == "__main__":
    result = main()
    print(json.dumps(result, indent=2))
