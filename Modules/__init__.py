"""
HadesAI Modules Package
Exploit loaders, implementations, and integrations
"""

from .known_exploits_loader import ExploitsLoader, POCExploit, ExploitPOCParser
from .exploit_implementations import (
    get_exploit_registry,
    ExploitRegistry,
    BaseExploit,
    DataExtractionExploit,
    CachePoisoningExploit,
    DOMStorageExploit,
    APIAbuseExploit,
    StealthPayloadDeliveryExploit,
    DLLHijackingExploit,
    ProcessInjectionExploit,
    C2CommunicationExploit
)
from .hades_exploits_integration import (
    HadesExploitsIntegration,
    HadesExploitsCLI,
    create_integration
)

__all__ = [
    'ExploitsLoader',
    'POCExploit',
    'ExploitPOCParser',
    'get_exploit_registry',
    'ExploitRegistry',
    'BaseExploit',
    'DataExtractionExploit',
    'CachePoisoningExploit',
    'DOMStorageExploit',
    'APIAbuseExploit',
    'StealthPayloadDeliveryExploit',
    'DLLHijackingExploit',
    'ProcessInjectionExploit',
    'C2CommunicationExploit',
    'HadesExploitsIntegration',
    'HadesExploitsCLI',
    'create_integration'
]
