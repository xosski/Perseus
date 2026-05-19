"""
Phantom Filesystem Builder
Creates deceptive filesystem structures to trap and deceive attackers

Features:
- Generates fake sensitive directories and files
- Creates honeypot databases with fake credentials
- Backdates file timestamps to appear legitimate
- Integrates with autonomous defense for automatic deployment
"""

import os
import random
import string
import sqlite3
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Dict

logger = logging.getLogger("PhantomFS")


class PhantomFileSystemBuilder:
    """Creates deceptive filesystem structures to trap attackers"""
    
    def __init__(self, base_path: str = "./phantom_root"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.files_created: List[Path] = []
        self.dirs_created: List[Path] = []

        # Enticing directory paths that attackers look for
        self.fake_dirs = [
            "usr/local/.oldconf.d/",
            "mnt/tempfs/cache/legacy/",
            "var/www/.admin_bak_2019/",
            "home/sysadmin/.shadowfs/",
            "C/Temp/Financial/UNSORTED_OLD/",
            "backup/.hidden_vault/",
            "opt/legacy_apps/credentials/",
            "home/admin/.ssh_backup/",
            "var/backups/database_dumps/",
            "srv/ftp/incoming/.private/"
        ]

        # Enticing filenames that attackers target
        self.fake_files = [
            "backup.2022.rclone.errlog",
            "restore_me.lck",
            ".vold_meta.json",
            "session.sqlite3",
            "pswds-vault-legacy.keystore",
            "error_log.old.txt",
            "ghostdrive_02.iso",
            "wallet_backup.dat",
            "id_rsa.bak",
            "credentials.csv.enc",
            "master_password.txt.gpg",
            "database_export.sql",
            "api_keys.json.old",
            ".htpasswd.bak",
            "shadow.backup"
        ]
        
        # Canary tokens for tracking access
        self.canary_tokens: Dict[str, str] = {}

    def generate(self) -> Dict:
        """Generate the phantom filesystem"""
        logger.info("[PhantomFS] Generating phantom filesystem...")
        
        for dir_path in self.fake_dirs:
            full_path = self.base_path / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            self.dirs_created.append(full_path)
            self._scatter_files(full_path)
        
        result = {
            "base_path": str(self.base_path),
            "dirs_created": len(self.dirs_created),
            "files_created": len(self.files_created),
            "canary_tokens": len(self.canary_tokens)
        }
        
        logger.info(f"[PhantomFS] Created {len(self.dirs_created)} dirs, {len(self.files_created)} files")
        return result

    def _scatter_files(self, path: Path):
        """Scatter fake files in a directory"""
        for _ in range(random.randint(3, 7)):
            filename = random.choice(self.fake_files)
            file_path = path / filename
            
            # Avoid duplicates
            if file_path.exists():
                continue
                
            content = self._generate_file_content(filename)
            
            # Generate and store canary token
            canary = self._generate_canary_token()
            self.canary_tokens[str(file_path)] = canary
            
            # Embed canary in content
            if isinstance(content, str):
                content = f"# CANARY:{canary}\n{content}"
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            self.files_created.append(file_path)

            # Set random backdated timestamp to appear old/legitimate
            past_time = datetime.now() - timedelta(days=random.randint(100, 3000))
            mod_time = past_time.timestamp()
            os.utime(file_path, (mod_time, mod_time))

    def _generate_file_content(self, filename: str) -> str:
        """Generate convincing fake content based on filename"""
        if filename.endswith(".sqlite3"):
            return self._fake_sqlite_payload()
        elif "vault" in filename or "keystore" in filename:
            return self._fake_vault_block()
        elif filename.endswith(".json"):
            return self._fake_json_config()
        elif filename.endswith(".iso"):
            return "ISOIMAGEGHOSTHEADER\n" + ''.join(random.choices(string.printable, k=512))
        elif "id_rsa" in filename:
            return self._fake_ssh_key()
        elif filename.endswith(".csv"):
            return self._fake_credentials_csv()
        elif filename.endswith(".sql"):
            return self._fake_sql_dump()
        elif "password" in filename.lower() or "passwd" in filename.lower():
            return self._fake_password_file()
        elif "htpasswd" in filename:
            return self._fake_htpasswd()
        elif "shadow" in filename:
            return self._fake_shadow_file()
        return ''.join(random.choices(string.ascii_letters + string.digits, k=100))

    def _generate_canary_token(self) -> str:
        """Generate a unique canary token for tracking"""
        import hashlib
        token = hashlib.sha256(os.urandom(32)).hexdigest()[:16]
        return f"PHANTOM_{token}"

    def _fake_sqlite_payload(self) -> str:
        """Create a fake SQLite database dump"""
        db_path = self.base_path / f"temp_fake_{random.randint(1000,9999)}.db"
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER, username TEXT, password TEXT, email TEXT);")
            
            fake_users = [
                ("admin", "P@ssw0rd123!", "admin@internal.local"),
                ("backup_user", "Backup2023#", "backup@internal.local"),
                ("dbadmin", "Database!Root", "dba@internal.local"),
                ("svc_account", "ServicePass!", "service@internal.local"),
                ("root", "r00t@ccess!", "root@localhost"),
            ]
            
            for i, (user, pwd, email) in enumerate(fake_users):
                cursor.execute("INSERT INTO users VALUES (?, ?, ?, ?);", (i, user, pwd, email))
            conn.commit()
            conn.close()

            with open(db_path, 'rb') as f:
                dump = f.read().hex()
            os.remove(db_path)
            return dump
        except Exception as e:
            logger.debug(f"SQLite fake generation error: {e}")
            return "SQLITE_CORRUPT_HEADER"

    def _fake_vault_block(self) -> str:
        """Generate fake encrypted vault blocks"""
        lines = ["# ENCRYPTED VAULT - DO NOT MODIFY", "# Version: 2.1.3-legacy"]
        for i in range(10):
            hexstring = ''.join(random.choices('0123456789abcdef', k=64))
            lines.append(f"$BLOCK_{i:03d}::{hexstring}")
        lines.append("# END VAULT")
        return '\n'.join(lines)

    def _fake_json_config(self) -> str:
        """Generate fake JSON configuration"""
        import json
        config = {
            "status": "deprecated",
            "version": "1.0.3-legacy",
            "database": {
                "host": "db.internal.local",
                "port": 5432,
                "username": "app_user",
                "password": "Db@pp2019!",
                "database": "production_bak"
            },
            "api_keys": {
                "stripe": "sk_live_" + ''.join(random.choices(string.ascii_letters + string.digits, k=24)),
                "aws": "AKIA" + ''.join(random.choices(string.ascii_uppercase + string.digits, k=16))
            },
            "error": "config migration incomplete"
        }
        return json.dumps(config, indent=2)

    def _fake_ssh_key(self) -> str:
        """Generate fake SSH private key"""
        key_body = '\n'.join([''.join(random.choices(string.ascii_letters + string.digits + '+/', k=64)) for _ in range(20)])
        return f"""-----BEGIN RSA PRIVATE KEY-----
{key_body}
-----END RSA PRIVATE KEY-----"""

    def _fake_credentials_csv(self) -> str:
        """Generate fake credentials CSV"""
        lines = ["username,password,email,role,last_login"]
        users = [
            ("admin", "Admin123!", "admin@company.com", "superuser", "2023-01-15"),
            ("jsmith", "John$mith99", "john.smith@company.com", "user", "2023-02-20"),
            ("backup", "Bkup2022!", "backup@internal", "service", "2022-12-01"),
            ("api_svc", "ApiK3y!Svc", "api@internal", "service", "2023-03-01"),
        ]
        for user in users:
            lines.append(','.join(user))
        return '\n'.join(lines)

    def _fake_sql_dump(self) -> str:
        """Generate fake SQL dump"""
        return """-- MySQL dump 10.13  Distrib 5.7.32
-- Server version 5.7.32-log

DROP TABLE IF EXISTS `admin_users`;
CREATE TABLE `admin_users` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `username` varchar(50) NOT NULL,
  `password_hash` varchar(255) NOT NULL,
  `api_key` varchar(64) DEFAULT NULL,
  PRIMARY KEY (`id`)
);

INSERT INTO `admin_users` VALUES 
(1,'admin','$2y$10$fakehashvaluehere123456','ak_live_abc123'),
(2,'superuser','$2y$10$anotherfakehash78901','ak_live_xyz789');

-- Dump completed
"""

    def _fake_password_file(self) -> str:
        """Generate fake password file"""
        return """# Password storage - LEGACY - DO NOT USE
admin:P@ssw0rd2023!
root:R00tAccess#1
backup:BackupKey$99
service:Svc!Account42
database:Db@dmin2022
"""

    def _fake_htpasswd(self) -> str:
        """Generate fake htpasswd file"""
        return """admin:$apr1$fakehash$morefakehashdatahere
webmaster:$apr1$xyz12345$abcdefghijklmnop
backup:$apr1$bkp00001$backuphashvalue123
"""

    def _fake_shadow_file(self) -> str:
        """Generate fake shadow file"""
        return """root:$6$rounds=5000$fakesalt$longhashvaluegoesherewithmoredata:18000:0:99999:7:::
admin:$6$rounds=5000$anothersalt$differenthashvalueforthisuser:18500:0:99999:7:::
backup:$6$rounds=5000$backupsalt$backupuserhashvalue:18200:0:99999:7:::
"""

    def cleanup(self):
        """Remove all phantom files and directories"""
        import shutil
        try:
            if self.base_path.exists():
                shutil.rmtree(self.base_path)
                logger.info(f"[PhantomFS] Cleaned up {self.base_path}")
                return True
        except Exception as e:
            logger.warning(f"[PhantomFS] Cleanup failed: {e}")
        return False

    def get_canary_tokens(self) -> Dict[str, str]:
        """Get all canary tokens for monitoring"""
        return self.canary_tokens.copy()


# Singleton instance for integration
_phantom_instance: Optional[PhantomFileSystemBuilder] = None


def get_phantom_builder(base_path: str = "./phantom_root") -> PhantomFileSystemBuilder:
    """Get or create phantom filesystem builder instance"""
    global _phantom_instance
    if _phantom_instance is None:
        _phantom_instance = PhantomFileSystemBuilder(base_path)
    return _phantom_instance


def deploy_phantom_filesystem(trigger_source: Optional[str] = None, base_path: str = "./phantom_root") -> Dict:
    """
    Deploy phantom filesystem - can be triggered by autonomous defense
    
    Args:
        trigger_source: What triggered the deployment (e.g., "honeypot_hit", "port_scan")
        base_path: Where to create the phantom filesystem
    
    Returns:
        Dict with deployment results
    """
    try:
        logger.info(f"[PhantomFS] Deploying due to trigger: {trigger_source or 'manual'}")
        builder = PhantomFileSystemBuilder(base_path)
        result = builder.generate()
        result["trigger"] = trigger_source or "manual"
        result["canary_tokens"] = builder.get_canary_tokens()
        logger.info("[PhantomFS] Deployment complete.")
        return result
    except Exception as e:
        logger.warning(f"[PhantomFS] Failed to deploy: {e}")
        return {"error": str(e), "trigger": trigger_source}


def main():
    """
    Standalone execution - deploy phantom filesystem and show status
    """
    print("=" * 60)
    print("👻 PHANTOM FILESYSTEM BUILDER")
    print("=" * 60)
    print()
    
    builder = PhantomFileSystemBuilder("./phantom_root")
    
    print("📁 Fake Directories to Create:")
    for dir_path in builder.fake_dirs[:5]:
        print(f"   • {dir_path}")
    print(f"   ... and {len(builder.fake_dirs) - 5} more")
    print()
    
    print("📄 Fake Files to Scatter:")
    for filename in builder.fake_files[:5]:
        print(f"   • {filename}")
    print(f"   ... and {len(builder.fake_files) - 5} more")
    print()
    
    # Ask before deploying
    print("⚠️  This will create deceptive files in ./phantom_root/")
    print()
    
    # Deploy
    result = builder.generate()
    
    print("✅ Phantom Filesystem Deployed:")
    print(f"   • Base Path: {result['base_path']}")
    print(f"   • Directories: {result['dirs_created']}")
    print(f"   • Files: {result['files_created']}")
    print(f"   • Canary Tokens: {result['canary_tokens']}")
    print()
    
    print("🎯 Canary Tokens (for monitoring):")
    for path, token in list(builder.canary_tokens.items())[:3]:
        print(f"   {token}: {path}")
    if len(builder.canary_tokens) > 3:
        print(f"   ... and {len(builder.canary_tokens) - 3} more")
    print()
    
    print("=" * 60)
    
    return result
