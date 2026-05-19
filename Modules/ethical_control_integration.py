"""
EthicalControl Integration Module
Enforces ethical safeguards and authorization checks throughout HadesAI
CRITICAL: This module ensures all operations are authorized and compliant
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
from datetime import datetime
import json

# Configure logging
logger = logging.getLogger("EthicalControl")
logger.setLevel(logging.DEBUG)


class AuthorizationLevel(Enum):
    """Authorization levels for operations"""
    UNAUTHORIZED = 0
    READ_ONLY = 1
    EXECUTE = 2
    ADMIN = 3
    FULL_ACCESS = 4


class ComplianceStatus(Enum):
    """Compliance status levels"""
    COMPLIANT = 'compliant'
    WARNING = 'warning'
    VIOLATION = 'violation'
    BLOCKED = 'blocked'


class EthicalControlIntegration:
    """
    Enforces ethical controls and authorization checks
    Ensures all HadesAI operations comply with policies
    """
    
    def __init__(self):
        """Initialize ethical control system"""
        self.enabled = True
        self.current_authorization = AuthorizationLevel.READ_ONLY
        self.current_environment = os.getenv("ENV_NAME", "unknown")
        self.authorized_environments = [
            "redteam-lab",
            "test-server",
            "authorized-assessment",
            "development"
        ]
        self.whitelist_exploits = set()
        self.whitelist_targets = set()
        self.operation_log = []
        self.policy_violations = []
        self.compliance_history = []
        
        self._initialize_ethical_controls()
        logger.info("✓ EthicalControl Integration initialized")
    
    def _initialize_ethical_controls(self):
        """Initialize ethical control policies"""
        try:
            # Load configuration
            config_path = Path(".hades_config.json")
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    
                    # Load authorized targets if specified
                    if 'authorized_targets' in config:
                        self.whitelist_targets = set(config['authorized_targets'])
                    
                    # Load authorized exploits if specified
                    if 'authorized_exploits' in config:
                        self.whitelist_exploits = set(config['authorized_exploits'])
            
            # Check environment authorization
            if self.current_environment not in self.authorized_environments:
                logger.warning(f"⚠️ Running in unauthorized environment: {self.current_environment}")
                self.current_authorization = AuthorizationLevel.READ_ONLY
                self._log_violation("Unauthorized environment", "CRITICAL")
            else:
                logger.info(f"✓ Authorized environment: {self.current_environment}")
                self.current_authorization = AuthorizationLevel.EXECUTE
                
        except Exception as e:
            logger.error(f"Error initializing ethical controls: {str(e)}")
            self.current_authorization = AuthorizationLevel.READ_ONLY
    
    # ========== AUTHORIZATION CHECKS ==========
    
    def is_authorized(self, operation: str, target: Optional[str] = None,
                     exploit: Optional[str] = None) -> bool:
        """
        Check if operation is authorized
        
        Args:
            operation: Type of operation (execute, generate, deploy, etc)
            target: Target of operation (IP, URL, system)
            exploit: Exploit being used
        
        Returns:
            bool indicating if operation is authorized
        """
        if not self.enabled:
            return True  # Disabled = all operations allowed
        
        # Check environment first
        if not self._is_environment_authorized():
            self._log_violation(
                f"Unauthorized operation in {self.current_environment}",
                "CRITICAL"
            )
            return False
        
        # Check target authorization
        if target and self.whitelist_targets:
            if target not in self.whitelist_targets:
                self._log_violation(f"Unauthorized target: {target}", "HIGH")
                return False
        
        # Check exploit authorization
        if exploit and self.whitelist_exploits:
            if exploit not in self.whitelist_exploits:
                self._log_violation(f"Unauthorized exploit: {exploit}", "HIGH")
                return False
        
        # Check operation type
        if operation.upper() in ['DEPLOY', 'EXECUTE', 'GENERATE_MALWARE']:
            if self.current_authorization.value < AuthorizationLevel.EXECUTE.value:
                self._log_violation(f"Insufficient authorization for: {operation}", "HIGH")
                return False
        
        # Log authorized operation
        self._log_operation(operation, target, exploit, authorized=True)
        return True
    
    def require_authorization(self, operation: str, target: Optional[str] = None,
                            exploit: Optional[str] = None) -> Dict[str, Any]:
        """
        Require explicit authorization for operation
        Returns details about what authorization is needed
        
        Args:
            operation: Type of operation
            target: Target of operation
            exploit: Exploit being used
        
        Returns:
            dict with authorization status and requirements
        """
        result = {
            'authorized': False,
            'operation': operation,
            'target': target,
            'exploit': exploit,
            'timestamp': datetime.now().isoformat(),
            'requirements': []
        }
        
        # Check each requirement
        if not self._is_environment_authorized():
            result['requirements'].append({
                'type': 'environment',
                'current': self.current_environment,
                'required': self.authorized_environments,
                'status': 'failed'
            })
        
        if target and self.whitelist_targets and target not in self.whitelist_targets:
            result['requirements'].append({
                'type': 'target',
                'value': target,
                'status': 'unauthorized'
            })
        
        if exploit and self.whitelist_exploits and exploit not in self.whitelist_exploits:
            result['requirements'].append({
                'type': 'exploit',
                'value': exploit,
                'status': 'unauthorized'
            })
        
        # Determine authorization
        result['authorized'] = len(result['requirements']) == 0
        
        if result['authorized']:
            self._log_operation(operation, target, exploit, authorized=True)
        else:
            self._log_violation(f"Authorization failed for {operation}", "MEDIUM")
        
        return result
    
    def _is_environment_authorized(self) -> bool:
        """Check if current environment is authorized"""
        return self.current_environment in self.authorized_environments
    
    # ========== COMPLIANCE CHECKS ==========
    
    def check_compliance(self, operation_type: str, details: Dict) -> Dict[str, Any]:
        """
        Check if operation complies with policies
        
        Args:
            operation_type: Type of operation being checked
            details: Operation details
        
        Returns:
            dict with compliance status and violations
        """
        violations = []
        warnings = []
        
        # Check for dangerous operations
        dangerous_operations = [
            'deploy_ransomware',
            'exfiltrate_data',
            'delete_evidence',
            'propagate_worm'
        ]
        
        if operation_type in dangerous_operations:
            violations.append(f"Dangerous operation not allowed: {operation_type}")
        
        # Check for unauthorized targets
        if 'target' in details and self.whitelist_targets:
            if details['target'] not in self.whitelist_targets:
                violations.append(f"Unauthorized target: {details['target']}")
        
        # Check scope
        if 'scope' in details and details['scope'] == 'production':
            if self.current_authorization.value < AuthorizationLevel.ADMIN.value:
                violations.append("Production scope requires admin authorization")
        
        # Determine compliance status
        if violations:
            status = ComplianceStatus.BLOCKED
        elif warnings:
            status = ComplianceStatus.WARNING
        else:
            status = ComplianceStatus.COMPLIANT
        
        compliance_record = {
            'operation_type': operation_type,
            'status': status.value,
            'timestamp': datetime.now().isoformat(),
            'violations': violations,
            'warnings': warnings,
            'authorized': len(violations) == 0
        }
        
        self.compliance_history.append(compliance_record)
        
        if violations:
            for violation in violations:
                self._log_violation(violation, "HIGH")
        
        return compliance_record
    
    # ========== AUTHORIZATION MANAGEMENT ==========
    
    def set_authorization_level(self, level: AuthorizationLevel):
        """Set current authorization level"""
        self.current_authorization = level
        logger.info(f"Authorization level set to: {level.name}")
    
    def add_authorized_target(self, target: str):
        """Add target to whitelist"""
        self.whitelist_targets.add(target)
        logger.info(f"✓ Target authorized: {target}")
    
    def add_authorized_exploit(self, exploit: str):
        """Add exploit to whitelist"""
        self.whitelist_exploits.add(exploit)
        logger.info(f"✓ Exploit authorized: {exploit}")
    
    def remove_authorized_target(self, target: str):
        """Remove target from whitelist"""
        self.whitelist_targets.discard(target)
        logger.info(f"Target removed from whitelist: {target}")
    
    def remove_authorized_exploit(self, exploit: str):
        """Remove exploit from whitelist"""
        self.whitelist_exploits.discard(exploit)
        logger.info(f"Exploit removed from whitelist: {exploit}")
    
    def get_authorized_targets(self) -> List[str]:
        """Get list of authorized targets"""
        return list(self.whitelist_targets)
    
    def get_authorized_exploits(self) -> List[str]:
        """Get list of authorized exploits"""
        return list(self.whitelist_exploits)
    
    # ========== AUDIT LOGGING ==========
    
    def _log_operation(self, operation: str, target: Optional[str],
                      exploit: Optional[str], authorized: bool):
        """Log operation for audit trail"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'operation': operation,
            'target': target,
            'exploit': exploit,
            'authorized': authorized,
            'environment': self.current_environment,
            'auth_level': self.current_authorization.name
        }
        self.operation_log.append(log_entry)
        
        status = "✓ AUTHORIZED" if authorized else "✗ DENIED"
        logger.info(f"{status}: {operation} on {target} using {exploit}")
    
    def _log_violation(self, violation: str, severity: str = "MEDIUM"):
        """Log policy violation"""
        violation_entry = {
            'timestamp': datetime.now().isoformat(),
            'violation': violation,
            'severity': severity,
            'environment': self.current_environment
        }
        self.policy_violations.append(violation_entry)
        
        logger.error(f"🚫 VIOLATION [{severity}]: {violation}")
    
    def get_audit_log(self, limit: int = 100) -> List[Dict]:
        """Get recent audit log entries"""
        return self.operation_log[-limit:]
    
    def get_violation_log(self, limit: int = 100) -> List[Dict]:
        """Get recent violation log entries"""
        return self.policy_violations[-limit:]
    
    def get_compliance_history(self, limit: int = 50) -> List[Dict]:
        """Get recent compliance check history"""
        return self.compliance_history[-limit:]
    
    # ========== REPORTING ==========
    
    def generate_compliance_report(self) -> Dict[str, Any]:
        """Generate compliance report"""
        total_operations = len(self.operation_log)
        authorized_operations = sum(1 for op in self.operation_log if op['authorized'])
        denied_operations = total_operations - authorized_operations
        
        total_violations = len(self.policy_violations)
        critical_violations = sum(1 for v in self.policy_violations if v['severity'] == 'CRITICAL')
        high_violations = sum(1 for v in self.policy_violations if v['severity'] == 'HIGH')
        
        return {
            'report_timestamp': datetime.now().isoformat(),
            'environment': self.current_environment,
            'authorization_level': self.current_authorization.name,
            'operations': {
                'total': total_operations,
                'authorized': authorized_operations,
                'denied': denied_operations,
                'authorization_rate': f"{(authorized_operations/max(1,total_operations)*100):.1f}%"
            },
            'violations': {
                'total': total_violations,
                'critical': critical_violations,
                'high': high_violations,
                'compliance_status': 'COMPLIANT' if critical_violations == 0 else 'VIOLATION'
            },
            'authorized_targets': len(self.whitelist_targets),
            'authorized_exploits': len(self.whitelist_exploits)
        }
    
    # ========== SYSTEM STATUS ==========
    
    def get_status(self) -> Dict[str, Any]:
        """Get EthicalControl system status"""
        return {
            'enabled': self.enabled,
            'environment': self.current_environment,
            'authorization_level': self.current_authorization.name,
            'environment_authorized': self._is_environment_authorized(),
            'operations_logged': len(self.operation_log),
            'violations_detected': len(self.policy_violations),
            'whitelisted_targets': len(self.whitelist_targets),
            'whitelisted_exploits': len(self.whitelist_exploits)
        }


# Create global instance
_ethical_control = None

def get_ethical_control() -> EthicalControlIntegration:
    """Get or create global EthicalControl instance"""
    global _ethical_control
    if _ethical_control is None:
        _ethical_control = EthicalControlIntegration()
    return _ethical_control


def require_authorization(operation: str, target: Optional[str] = None,
                         exploit: Optional[str] = None) -> bool:
    """
    Decorator-friendly authorization check
    Use this before critical operations
    """
    ec = get_ethical_control()
    return ec.is_authorized(operation, target, exploit)
