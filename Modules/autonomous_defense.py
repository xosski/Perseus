"""
Autonomous Defense Module
Active defensive countermeasures for network monitoring

Features:
- Automated threat response with multiple defense strategies
- Honeypot deployment for attacker deception
- Rate limiting and connection throttling
- DNS sinkholing for malicious domains
- Network isolation capabilities
- Deceptive responses to waste attacker resources
- Adaptive defense that learns from attacks
"""

import os
import json
import time
import socket
import threading
import subprocess
import platform
import logging
import hashlib
import random
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Callable
from collections import defaultdict
from enum import Enum

logger = logging.getLogger("AutonomousDefense")


class DefenseLevel(Enum):
    """Defense escalation levels"""
    PASSIVE = 0      # Monitor and log only
    REACTIVE = 1     # Respond to detected threats
    PROACTIVE = 2    # Actively hunt and deceive
    AGGRESSIVE = 3   # Maximum defense with countermeasures


class DefenseAction(Enum):
    """Available defensive actions"""
    LOG = "log"
    ALERT = "alert"
    RATE_LIMIT = "rate_limit"
    TARPIT = "tarpit"
    BLOCK_TEMP = "block_temp"
    BLOCK_PERM = "block_perm"
    HONEYPOT = "honeypot"
    PHANTOM_FS = "phantom_fs"  # Deploy phantom filesystem
    SINKHOLE = "sinkhole"
    ISOLATE = "isolate"
    DECEIVE = "deceive"
    COUNTER = "counter"
    KILL_PROCESS = "kill_process"
    QUARANTINE = "quarantine"


@dataclass
class DefenseRule:
    """Defense rule configuration"""
    name: str
    trigger_type: str  # port_scan, brute_force, c2, malware, etc.
    min_threat_level: str  # WARNING, HIGH, CRITICAL
    actions: List[DefenseAction]
    cooldown_seconds: int = 60
    max_triggers_per_hour: int = 100
    enabled: bool = True
    last_triggered: float = 0
    trigger_count: int = 0


@dataclass
class AttackerProfile:
    """Tracked attacker information"""
    ip: str
    first_seen: float
    last_seen: float
    attack_types: List[str] = field(default_factory=list)
    total_attempts: int = 0
    blocked: bool = False
    threat_score: float = 0.0
    responses_sent: List[str] = field(default_factory=list)
    fingerprint: Dict = field(default_factory=dict)


class HoneypotService:
    """Lightweight honeypot for attacker deception"""
    
    FAKE_BANNERS = {
        21: "220 FTP Server (vsFTPd 2.3.4) Ready\r\n",
        22: "SSH-2.0-OpenSSH_7.4p1 Debian-10+deb9u7\r\n",
        23: "\r\nLogin: ",
        25: "220 mail.company.local ESMTP Postfix\r\n",
        80: "HTTP/1.1 200 OK\r\nServer: Apache/2.4.29\r\n\r\n",
        443: "HTTP/1.1 200 OK\r\nServer: nginx/1.14.0\r\n\r\n",
        445: "\x00",  # SMB null
        3306: "5.7.31-0ubuntu0.18.04.1\x00",
        3389: "\x03\x00\x00\x13\x0e\xd0\x00\x00\x12\x34\x00",  # RDP
        5432: "PostgreSQL 12.4\x00",
        6379: "-ERR unknown command\r\n",
        27017: "MongoDB shell version\x00",
    }
    
    def __init__(self, callback: Callable = None):
        self.active_ports: Dict[int, socket.socket] = {}
        self.connections_log: List[Dict] = []
        self.callback = callback
        self.running = False
        self._threads: List[threading.Thread] = []
    
    def start_honeypot(self, port: int) -> bool:
        """Start a honeypot listener on specified port"""
        if port in self.active_ports:
            return True
        
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.settimeout(1.0)
            sock.bind(('0.0.0.0', port))
            sock.listen(5)
            
            self.active_ports[port] = sock
            self.running = True
            
            thread = threading.Thread(
                target=self._honeypot_listener, 
                args=(port, sock),
                daemon=True
            )
            thread.start()
            self._threads.append(thread)
            
            logger.info(f"🍯 Honeypot started on port {port}")
            return True
        except Exception as e:
            logger.warning(f"Failed to start honeypot on port {port}: {e}")
            return False
    
    def _honeypot_listener(self, port: int, sock: socket.socket):
        """Handle honeypot connections"""
        while self.running and port in self.active_ports:
            try:
                conn, addr = sock.accept()
                thread = threading.Thread(
                    target=self._handle_honeypot_connection,
                    args=(port, conn, addr),
                    daemon=True
                )
                thread.start()
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    logger.debug(f"Honeypot listener error: {e}")
                break
    
    def _handle_honeypot_connection(self, port: int, conn: socket.socket, addr: Tuple):
        """Handle individual honeypot connection"""
        try:
            remote_ip = addr[0]
            
            # Log connection attempt
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'port': port,
                'remote_ip': remote_ip,
                'remote_port': addr[1],
                'type': 'honeypot_hit'
            }
            
            # Send fake banner if available
            banner = self.FAKE_BANNERS.get(port, "")
            if banner:
                conn.send(banner.encode() if isinstance(banner, str) else banner)
            
            # Try to capture attacker input
            conn.settimeout(5.0)
            try:
                data = conn.recv(1024)
                log_entry['payload'] = data.hex()[:200]
                log_entry['payload_text'] = data.decode('utf-8', errors='ignore')[:100]
            except:
                pass
            
            self.connections_log.append(log_entry)
            
            # Callback for threat detection
            if self.callback:
                self.callback({
                    'type': 'honeypot_hit',
                    'ip': remote_ip,
                    'port': port,
                    'timestamp': log_entry['timestamp']
                })
            
            logger.warning(f"🍯 Honeypot hit: {remote_ip} → port {port}")
            
            # Waste attacker time with delayed garbage
            time.sleep(random.uniform(2, 5))
            try:
                garbage = os.urandom(64)
                conn.send(garbage)
            except:
                pass
                
        except Exception as e:
            logger.debug(f"Honeypot connection error: {e}")
        finally:
            try:
                conn.close()
            except:
                pass
    
    def stop_honeypot(self, port: int):
        """Stop honeypot on specified port"""
        if port in self.active_ports:
            try:
                self.active_ports[port].close()
            except:
                pass
            del self.active_ports[port]
            logger.info(f"🍯 Honeypot stopped on port {port}")
    
    def stop_all(self):
        """Stop all honeypots"""
        self.running = False
        for port in list(self.active_ports.keys()):
            self.stop_honeypot(port)
    
    def get_connections(self) -> List[Dict]:
        """Get honeypot connection log"""
        return self.connections_log[-100:]


class RateLimiter:
    """Connection rate limiting and throttling"""
    
    def __init__(self):
        self.ip_rates: Dict[str, List[float]] = defaultdict(list)
        self.throttled_ips: Dict[str, float] = {}  # IP -> throttle until time
        self.blocked_ips: Set[str] = set()
        
        # Configurable limits
        self.connections_per_second = 10
        self.connections_per_minute = 100
        self.throttle_duration = 300  # 5 minutes
        self.block_threshold = 500  # Block after this many in a minute
    
    def check_connection(self, ip: str) -> Tuple[bool, str]:
        """
        Check if connection should be allowed
        Returns: (allowed, reason)
        """
        now = time.time()
        
        # Check if blocked
        if ip in self.blocked_ips:
            return False, "blocked"
        
        # Check if throttled
        if ip in self.throttled_ips:
            if now < self.throttled_ips[ip]:
                return False, "throttled"
            else:
                del self.throttled_ips[ip]
        
        # Record this connection
        self.ip_rates[ip].append(now)
        
        # Clean old entries
        cutoff_second = now - 1
        cutoff_minute = now - 60
        self.ip_rates[ip] = [t for t in self.ip_rates[ip] if t > cutoff_minute]
        
        # Count rates
        count_second = sum(1 for t in self.ip_rates[ip] if t > cutoff_second)
        count_minute = len(self.ip_rates[ip])
        
        # Check thresholds
        if count_minute >= self.block_threshold:
            self.blocked_ips.add(ip)
            logger.warning(f"⛔ Rate limit BLOCK: {ip} ({count_minute}/min)")
            return False, "rate_blocked"
        
        if count_minute >= self.connections_per_minute or count_second >= self.connections_per_second:
            self.throttled_ips[ip] = now + self.throttle_duration
            logger.warning(f"🐢 Rate limit THROTTLE: {ip} for {self.throttle_duration}s")
            return False, "rate_throttled"
        
        return True, "allowed"
    
    def unblock(self, ip: str):
        """Remove IP from blocked list"""
        self.blocked_ips.discard(ip)
        self.throttled_ips.pop(ip, None)
    
    def get_stats(self) -> Dict:
        """Get rate limiting statistics"""
        return {
            'blocked_count': len(self.blocked_ips),
            'throttled_count': len(self.throttled_ips),
            'tracked_ips': len(self.ip_rates)
        }


class TarpitHandler:
    """Tarpit to slow down attackers"""
    
    def __init__(self):
        self.tarpitted_connections: Dict[str, List[socket.socket]] = defaultdict(list)
        self.delay_seconds = 30
        self.max_connections_per_ip = 50
    
    def tarpit_connection(self, conn: socket.socket, ip: str):
        """Add connection to tarpit"""
        if len(self.tarpitted_connections[ip]) >= self.max_connections_per_ip:
            # Close oldest
            try:
                old_conn = self.tarpitted_connections[ip].pop(0)
                old_conn.close()
            except:
                pass
        
        self.tarpitted_connections[ip].append(conn)
        
        # Start slow drip in background
        thread = threading.Thread(
            target=self._slow_drip,
            args=(conn, ip),
            daemon=True
        )
        thread.start()
        logger.info(f"🐌 Tarpitting connection from {ip}")
    
    def _slow_drip(self, conn: socket.socket, ip: str):
        """Slowly send data to waste attacker resources"""
        try:
            for _ in range(self.delay_seconds):
                time.sleep(1)
                try:
                    conn.send(b'\x00')  # Send null byte slowly
                except:
                    break
        finally:
            try:
                conn.close()
            except:
                pass
            if conn in self.tarpitted_connections.get(ip, []):
                self.tarpitted_connections[ip].remove(conn)


class DNSSinkhole:
    """DNS sinkhole for malicious domains"""
    
    def __init__(self):
        self.sinkholed_domains: Set[str] = set()
        self.hosts_file_modified = False
        self.original_hosts = ""
        
        # Common malicious domain patterns
        self.malicious_patterns = [
            'malware', 'botnet', 'c2server', 'evil', 'hack',
            'ransom', 'crypto-miner', 'keylog'
        ]
    
    def add_sinkhole(self, domain: str) -> bool:
        """Add domain to sinkhole"""
        domain = domain.lower().strip()
        if domain in self.sinkholed_domains:
            return True
        
        self.sinkholed_domains.add(domain)
        logger.info(f"🕳️ Sinkholed domain: {domain}")
        return True
    
    def apply_to_hosts_file(self) -> bool:
        """Apply sinkhole to hosts file (requires admin)"""
        if not self.sinkholed_domains:
            return True
        
        try:
            system = platform.system().lower()
            if system == 'windows':
                hosts_path = r'C:\Windows\System32\drivers\etc\hosts'
            else:
                hosts_path = '/etc/hosts'
            
            # Read current hosts
            with open(hosts_path, 'r') as f:
                self.original_hosts = f.read()
            
            # Check if our marker exists
            marker = "# HADES SINKHOLE START"
            if marker in self.original_hosts:
                logger.info("Sinkhole entries already present")
                return True
            
            # Add sinkhole entries
            sinkhole_entries = f"\n{marker}\n"
            for domain in self.sinkholed_domains:
                sinkhole_entries += f"127.0.0.1 {domain}\n"
                sinkhole_entries += f"127.0.0.1 www.{domain}\n"
            sinkhole_entries += "# HADES SINKHOLE END\n"
            
            with open(hosts_path, 'a') as f:
                f.write(sinkhole_entries)
            
            self.hosts_file_modified = True
            logger.info(f"🕳️ Applied {len(self.sinkholed_domains)} sinkhole entries to hosts")
            return True
            
        except PermissionError:
            logger.warning("Insufficient permissions to modify hosts file")
            return False
        except Exception as e:
            logger.error(f"Failed to modify hosts file: {e}")
            return False
    
    def restore_hosts_file(self) -> bool:
        """Remove sinkhole entries from hosts file"""
        if not self.hosts_file_modified:
            return True
        
        try:
            system = platform.system().lower()
            if system == 'windows':
                hosts_path = r'C:\Windows\System32\drivers\etc\hosts'
            else:
                hosts_path = '/etc/hosts'
            
            with open(hosts_path, 'r') as f:
                content = f.read()
            
            # Remove our entries
            start_marker = "# HADES SINKHOLE START"
            end_marker = "# HADES SINKHOLE END\n"
            
            start_idx = content.find(start_marker)
            end_idx = content.find(end_marker)
            
            if start_idx != -1 and end_idx != -1:
                content = content[:start_idx] + content[end_idx + len(end_marker):]
                with open(hosts_path, 'w') as f:
                    f.write(content)
                self.hosts_file_modified = False
                logger.info("🕳️ Removed sinkhole entries from hosts file")
            
            return True
        except Exception as e:
            logger.error(f"Failed to restore hosts file: {e}")
            return False


class DeceptiveResponder:
    """Generate deceptive responses to waste attacker resources"""
    
    FAKE_PASSWORDS = [
        "admin123", "password", "letmein", "123456", "qwerty",
        "root", "toor", "master", "secret", "hunter2"
    ]
    
    FAKE_USERS = [
        "admin", "root", "user", "test", "guest", "backup",
        "oracle", "mysql", "postgres", "ftp", "www-data"
    ]
    
    FAKE_FILES = [
        "config.xml", "passwords.txt", "database.sql", "backup.tar.gz",
        "credentials.json", "secrets.env", "id_rsa", "wallet.dat"
    ]
    
    def generate_fake_login_response(self, delay: float = 3.0) -> str:
        """Generate fake failed login response with delay"""
        time.sleep(delay)  # Waste attacker time
        return f"Login failed for user: {random.choice(self.FAKE_USERS)}\n"
    
    def generate_fake_file_list(self) -> str:
        """Generate fake directory listing"""
        files = random.sample(self.FAKE_FILES, min(5, len(self.FAKE_FILES)))
        listing = "-rw-r--r-- 1 root root  1024 Jan 01 00:00 "
        return "\n".join([listing + f for f in files])
    
    def generate_fake_credentials(self) -> Dict:
        """Generate fake credentials to poison attacker data"""
        return {
            'username': random.choice(self.FAKE_USERS),
            'password': random.choice(self.FAKE_PASSWORDS),
            'host': f"192.168.{random.randint(1,254)}.{random.randint(1,254)}",
            'database': f"db_{random.randint(1000,9999)}",
            'api_key': hashlib.md5(os.urandom(16)).hexdigest()
        }
    
    def generate_canary_token(self) -> str:
        """Generate a trackable canary token"""
        token = hashlib.sha256(os.urandom(32)).hexdigest()[:32]
        return f"HADES_CANARY_{token}"


class AutonomousDefenseEngine:
    """Main autonomous defense coordination engine"""
    
    def __init__(self, kb=None, network_monitor=None):
        self.kb = kb
        self.network_monitor = network_monitor
        self.enabled = False
        self.defense_level = DefenseLevel.REACTIVE
        
        # Components
        self.honeypot = HoneypotService(callback=self._on_honeypot_hit)
        self.rate_limiter = RateLimiter()
        self.tarpit = TarpitHandler()
        self.dns_sinkhole = DNSSinkhole()
        self.deceiver = DeceptiveResponder()
        self.phantom_fs = None  # Lazy-loaded phantom filesystem
        self.phantom_deployed = False
        
        # Tracking
        self.attacker_profiles: Dict[str, AttackerProfile] = {}
        self.defense_rules: List[DefenseRule] = []
        self.action_log: List[Dict] = []
        self.stats = {
            'threats_mitigated': 0,
            'ips_blocked': 0,
            'honeypot_hits': 0,
            'connections_throttled': 0,
            'phantom_deployments': 0,
            'active_since': None
        }
        
        # Callbacks
        self.on_action_taken: Optional[Callable] = None
        self.on_threat_mitigated: Optional[Callable] = None
        
        # Initialize default rules
        self._init_default_rules()
    
    def _init_default_rules(self):
        """Initialize default defense rules"""
        self.defense_rules = [
            DefenseRule(
                name="Block Port Scanners",
                trigger_type="PORT_SCAN",
                min_threat_level="HIGH",
                actions=[DefenseAction.BLOCK_TEMP, DefenseAction.LOG, DefenseAction.ALERT]
            ),
            DefenseRule(
                name="Block Brute Force",
                trigger_type="BRUTE_FORCE",
                min_threat_level="HIGH",
                actions=[DefenseAction.BLOCK_PERM, DefenseAction.RATE_LIMIT, DefenseAction.LOG]
            ),
            DefenseRule(
                name="Sinkhole C2",
                trigger_type="POTENTIAL_C2",
                min_threat_level="HIGH",
                actions=[DefenseAction.BLOCK_PERM, DefenseAction.KILL_PROCESS, DefenseAction.SINKHOLE]
            ),
            DefenseRule(
                name="Quarantine Malware",
                trigger_type="SUSPICIOUS_PROCESS",
                min_threat_level="HIGH",
                actions=[DefenseAction.KILL_PROCESS, DefenseAction.QUARANTINE, DefenseAction.ALERT]
            ),
            DefenseRule(
                name="Rate Limit Suspicious",
                trigger_type="SUSPICIOUS_BEHAVIOR",
                min_threat_level="WARNING",
                actions=[DefenseAction.RATE_LIMIT, DefenseAction.LOG]
            ),
            DefenseRule(
                name="Honeypot Known Bad IPs",
                trigger_type="BLOCKED_IP",
                min_threat_level="CRITICAL",
                actions=[DefenseAction.HONEYPOT, DefenseAction.DECEIVE, DefenseAction.LOG]
            ),
            DefenseRule(
                name="Deploy Phantom FS on Intrusion",
                trigger_type="LATERAL_MOVEMENT",
                min_threat_level="HIGH",
                actions=[DefenseAction.PHANTOM_FS, DefenseAction.DECEIVE, DefenseAction.ALERT]
            ),
            DefenseRule(
                name="Phantom FS for File System Access",
                trigger_type="SUSPICIOUS_FILE_ACCESS",
                min_threat_level="HIGH",
                actions=[DefenseAction.PHANTOM_FS, DefenseAction.LOG]
            )
        ]
    
    def enable(self, level: DefenseLevel = DefenseLevel.REACTIVE) -> bool:
        """Enable autonomous defense"""
        try:
            self.enabled = True
            self.defense_level = level
            self.stats['active_since'] = datetime.now().isoformat()
            
            # Start honeypots on common attack ports if proactive
            if level.value >= DefenseLevel.PROACTIVE.value:
                for port in [4444, 5555, 31337, 12345]:
                    self.honeypot.start_honeypot(port)
            
            logger.info(f"🛡️ Autonomous Defense ENABLED at level: {level.name}")
            return True
        except Exception as e:
            logger.error(f"Failed to enable defense: {e}")
            return False
    
    def disable(self):
        """Disable autonomous defense"""
        self.enabled = False
        self.honeypot.stop_all()
        self.dns_sinkhole.restore_hosts_file()
        logger.info("🛡️ Autonomous Defense DISABLED")
    
    def process_threat(self, threat_data: Dict) -> List[DefenseAction]:
        """Process a detected threat and take defensive action"""
        if not self.enabled:
            return []
        
        threat_type = threat_data.get('threat_type', 'UNKNOWN')
        threat_level = threat_data.get('threat_level', 'WARNING')
        remote_ip = threat_data.get('remote_ip')
        
        # Update attacker profile
        if remote_ip:
            self._update_attacker_profile(remote_ip, threat_data)
        
        # Find matching rules
        actions_taken = []
        for rule in self.defense_rules:
            if not rule.enabled:
                continue
            
            if rule.trigger_type == threat_type or threat_type in rule.trigger_type:
                if self._threat_level_matches(threat_level, rule.min_threat_level):
                    # Check cooldown
                    now = time.time()
                    if now - rule.last_triggered < rule.cooldown_seconds:
                        continue
                    
                    # Execute rule actions
                    for action in rule.actions:
                        success = self._execute_action(action, threat_data)
                        if success:
                            actions_taken.append(action)
                    
                    rule.last_triggered = now
                    rule.trigger_count += 1
        
        if actions_taken:
            self.stats['threats_mitigated'] += 1
            self._log_action(threat_data, actions_taken)
            
            if self.on_threat_mitigated:
                self.on_threat_mitigated(threat_data, actions_taken)
        
        return actions_taken
    
    def _threat_level_matches(self, actual: str, minimum: str) -> bool:
        """Check if threat level meets minimum"""
        levels = {'INFO': 0, 'WARNING': 1, 'HIGH': 2, 'CRITICAL': 3}
        return levels.get(actual, 0) >= levels.get(minimum, 0)
    
    def _execute_action(self, action: DefenseAction, threat_data: Dict) -> bool:
        """Execute a defensive action"""
        remote_ip = threat_data.get('remote_ip')
        pid = threat_data.get('pid')
        
        try:
            if action == DefenseAction.LOG:
                logger.info(f"DEFENSE LOG: {threat_data.get('threat_type')} from {remote_ip}")
                return True
            
            elif action == DefenseAction.ALERT:
                logger.warning(f"⚠️ DEFENSE ALERT: {threat_data}")
                return True
            
            elif action == DefenseAction.RATE_LIMIT and remote_ip:
                allowed, reason = self.rate_limiter.check_connection(remote_ip)
                if not allowed:
                    self.stats['connections_throttled'] += 1
                return True
            
            elif action == DefenseAction.BLOCK_TEMP and remote_ip:
                self._apply_firewall_block(remote_ip, permanent=False)
                self.stats['ips_blocked'] += 1
                return True
            
            elif action == DefenseAction.BLOCK_PERM and remote_ip:
                self._apply_firewall_block(remote_ip, permanent=True)
                self.stats['ips_blocked'] += 1
                return True
            
            elif action == DefenseAction.HONEYPOT:
                port = threat_data.get('local_port', 4444)
                return self.honeypot.start_honeypot(port)
            
            elif action == DefenseAction.SINKHOLE:
                domain = threat_data.get('domain')
                if domain:
                    return self.dns_sinkhole.add_sinkhole(domain)
                return False
            
            elif action == DefenseAction.KILL_PROCESS and pid:
                return self._kill_process(pid)
            
            elif action == DefenseAction.DECEIVE and remote_ip:
                # Generate and log fake credentials
                fake_creds = self.deceiver.generate_fake_credentials()
                logger.info(f"🎭 Deployed deception for {remote_ip}: {fake_creds['username']}")
                return True
            
            elif action == DefenseAction.QUARANTINE:
                # Mark for quarantine
                logger.warning(f"🔒 QUARANTINE: Process {pid} flagged")
                return True
            
            elif action == DefenseAction.TARPIT:
                # Would need socket reference
                logger.info(f"🐌 TARPIT mode for {remote_ip}")
                return True
            
            elif action == DefenseAction.PHANTOM_FS:
                return self._deploy_phantom_filesystem(threat_data)
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to execute {action}: {e}")
            return False
    
    def _apply_firewall_block(self, ip: str, permanent: bool = False):
        """Apply firewall block"""
        system = platform.system().lower()
        
        try:
            if system == 'windows':
                rule_name = f"HADES_AUTOBLOCK_{ip.replace('.', '_')}"
                cmd = f'netsh advfirewall firewall add rule name="{rule_name}" dir=in action=block remoteip={ip}'
                subprocess.run(cmd, shell=True, capture_output=True, timeout=10)
                logger.info(f"🛡️ Firewall: Blocked {ip}")
                
            elif system == 'linux':
                cmd = f'iptables -A INPUT -s {ip} -j DROP'
                subprocess.run(cmd, shell=True, capture_output=True, timeout=10)
                logger.info(f"🛡️ Firewall: Blocked {ip} (Linux)")
                
        except Exception as e:
            logger.warning(f"Firewall block failed: {e}")
    
    def _kill_process(self, pid: int) -> bool:
        """Kill a malicious process"""
        try:
            import psutil
            proc = psutil.Process(pid)
            proc_name = proc.name()
            
            # Safety: don't kill system processes
            protected = {'system', 'svchost.exe', 'csrss.exe', 'explorer.exe', 
                        'init', 'systemd', 'kernel', 'launchd'}
            if proc_name.lower() in protected:
                logger.warning(f"Refused to kill protected process: {proc_name}")
                return False
            
            proc.terminate()
            logger.info(f"🔥 Terminated: {proc_name} (PID: {pid})")
            return True
        except Exception as e:
            logger.debug(f"Failed to kill process {pid}: {e}")
            return False
    
    def _deploy_phantom_filesystem(self, threat_data: Dict) -> bool:
        """Deploy phantom filesystem as a deception tactic"""
        try:
            # Import here to avoid circular imports
            from modules.PhantomFilessytemBuilder import (
                PhantomFileSystemBuilder, 
                deploy_phantom_filesystem
            )
            
            trigger = threat_data.get('threat_type', 'unknown')
            remote_ip = threat_data.get('remote_ip', 'unknown')
            
            # Only deploy once per session (don't spam filesystem)
            if self.phantom_deployed:
                logger.debug("[PhantomFS] Already deployed, skipping")
                return True
            
            # Deploy phantom filesystem
            result = deploy_phantom_filesystem(
                trigger_source=f"{trigger} from {remote_ip}",
                base_path="./phantom_root"
            )
            
            if 'error' not in result:
                self.phantom_deployed = True
                self.stats['phantom_deployments'] += 1
                
                # Store canary tokens for later monitoring
                if self.phantom_fs is None:
                    self.phantom_fs = {
                        'deployed_at': datetime.now().isoformat(),
                        'trigger': trigger,
                        'attacker_ip': remote_ip,
                        'canary_tokens': result.get('canary_tokens', {})
                    }
                
                logger.info(f"👻 PhantomFS deployed: {result.get('files_created', 0)} files created")
                return True
            else:
                logger.warning(f"[PhantomFS] Deployment failed: {result.get('error')}")
                return False
                
        except ImportError:
            logger.warning("[PhantomFS] Module not available")
            return False
        except Exception as e:
            logger.error(f"[PhantomFS] Deployment error: {e}")
            return False
    
    def get_phantom_status(self) -> Optional[Dict]:
        """Get phantom filesystem deployment status"""
        return self.phantom_fs
    
    def _update_attacker_profile(self, ip: str, threat_data: Dict):
        """Update or create attacker profile"""
        now = time.time()
        
        if ip not in self.attacker_profiles:
            self.attacker_profiles[ip] = AttackerProfile(
                ip=ip,
                first_seen=now,
                last_seen=now
            )
        
        profile = self.attacker_profiles[ip]
        profile.last_seen = now
        profile.total_attempts += 1
        
        threat_type = threat_data.get('threat_type')
        if threat_type and threat_type not in profile.attack_types:
            profile.attack_types.append(threat_type)
        
        # Update threat score
        score_map = {'WARNING': 1, 'HIGH': 3, 'CRITICAL': 5}
        profile.threat_score += score_map.get(threat_data.get('threat_level', 'WARNING'), 1)
    
    def _on_honeypot_hit(self, data: Dict):
        """Handle honeypot hit"""
        self.stats['honeypot_hits'] += 1
        ip = data.get('ip')
        
        if ip:
            # Auto-escalate to block
            self.process_threat({
                'threat_type': 'HONEYPOT_TRIGGER',
                'threat_level': 'HIGH',
                'remote_ip': ip,
                'local_port': data.get('port')
            })
    
    def _log_action(self, threat_data: Dict, actions: List[DefenseAction]):
        """Log defensive action"""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'threat': threat_data.get('threat_type'),
            'ip': threat_data.get('remote_ip'),
            'actions': [a.value for a in actions]
        }
        self.action_log.append(entry)
        
        # Keep last 1000 entries
        if len(self.action_log) > 1000:
            self.action_log = self.action_log[-500:]
        
        if self.on_action_taken:
            self.on_action_taken(entry)
    
    def add_rule(self, rule: DefenseRule):
        """Add a custom defense rule"""
        self.defense_rules.append(rule)
        logger.info(f"Added defense rule: {rule.name}")
    
    def get_stats(self) -> Dict:
        """Get defense statistics"""
        return {
            **self.stats,
            'attacker_profiles': len(self.attacker_profiles),
            'active_honeypots': len(self.honeypot.active_ports),
            'rate_limiter': self.rate_limiter.get_stats(),
            'sinkholed_domains': len(self.dns_sinkhole.sinkholed_domains),
            'defense_level': self.defense_level.name,
            'enabled': self.enabled
        }
    
    def get_attacker_profiles(self) -> List[Dict]:
        """Get list of attacker profiles"""
        return [
            {
                'ip': p.ip,
                'first_seen': datetime.fromtimestamp(p.first_seen).isoformat(),
                'last_seen': datetime.fromtimestamp(p.last_seen).isoformat(),
                'attack_types': p.attack_types,
                'total_attempts': p.total_attempts,
                'threat_score': p.threat_score,
                'blocked': p.blocked
            }
            for p in self.attacker_profiles.values()
        ]
    
    def get_action_log(self) -> List[Dict]:
        """Get defense action log"""
        return self.action_log[-100:]


# Integration function for NetworkMonitor
def integrate_with_network_monitor(network_monitor, kb=None) -> AutonomousDefenseEngine:
    """
    Integrate autonomous defense with existing NetworkMonitor
    
    Usage:
        from modules.autonomous_defense import integrate_with_network_monitor
        defense = integrate_with_network_monitor(self.network_monitor, self.kb)
        defense.enable(DefenseLevel.PROACTIVE)
    """
    defense = AutonomousDefenseEngine(kb=kb, network_monitor=network_monitor)
    
    # Connect to threat signals
    if hasattr(network_monitor, 'threat_detected'):
        def on_threat(conn_data):
            defense.process_threat(conn_data)
        
        network_monitor.threat_detected.connect(on_threat)
    
    return defense


def main():
    """
    Standalone execution - demonstrates autonomous defense capabilities
    Run from Modules tab or command line for status/testing
    """
    print("=" * 60)
    print("🛡️  AUTONOMOUS DEFENSE MODULE")
    print("=" * 60)
    print()
    
    # Initialize engine for demo
    engine = AutonomousDefenseEngine()
    
    print("📋 Available Defense Levels:")
    for level in DefenseLevel:
        print(f"   • {level.name} (value: {level.value})")
    print()
    
    print("⚔️  Available Defense Actions:")
    for action in DefenseAction:
        print(f"   • {action.value}")
    print()
    
    print("📜 Default Defense Rules:")
    for rule in engine.defense_rules:
        actions_str = ", ".join([a.value for a in rule.actions])
        print(f"   [{rule.min_threat_level}] {rule.name}")
        print(f"       Trigger: {rule.trigger_type}")
        print(f"       Actions: {actions_str}")
    print()
    
    print("🍯 Honeypot Fake Banners Available:")
    for port, banner in list(HoneypotService.FAKE_BANNERS.items())[:5]:
        banner_preview = repr(banner[:30]) if len(banner) > 30 else repr(banner)
        print(f"   Port {port}: {banner_preview}")
    print(f"   ... and {len(HoneypotService.FAKE_BANNERS) - 5} more")
    print()
    
    print("🎭 Deceptive Response Samples:")
    deceiver = DeceptiveResponder()
    fake_creds = deceiver.generate_fake_credentials()
    print(f"   Fake Credentials: {fake_creds['username']}:{fake_creds['password']}")
    print(f"   Canary Token: {deceiver.generate_canary_token()}")
    print()
    
    print("✅ Module loaded successfully!")
    print("   Enable via Network Monitor → '🤖 Autonomous Defense' checkbox")
    print()
    print("=" * 60)
    
    return {
        "status": "ready",
        "defense_levels": [l.name for l in DefenseLevel],
        "defense_actions": [a.value for a in DefenseAction],
        "rules_count": len(engine.defense_rules),
        "honeypot_ports": list(HoneypotService.FAKE_BANNERS.keys())
    }
