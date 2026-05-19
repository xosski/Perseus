"""
Hades AI - Clockworks Obfuscation Integration
Integrates obfuscation capabilities into Hades AI core systems
"""

import logging
from typing import Dict, List, Optional, Any
from enum import Enum
from .obfuscation_engine import ClockworksObfuscator, obfuscate, deobfuscate

logger = logging.getLogger(__name__)


class ObfuscationType(Enum):
    """Types of obfuscation operations"""
    LUA = "lua"
    PAYLOAD = "payload"
    SHELLCODE = "shellcode"
    COMMAND = "command"
    SCRIPT = "script"
    BINARY = "binary"


class HadesObfuscationIntegration:
    """Hades AI Clockworks Obfuscation System"""

    def __init__(self, default_seed: int = 7, default_rounds: int = 9):
        """Initialize Hades obfuscation integration"""
        self.obfuscator = ClockworksObfuscator(seed=default_seed, rounds=default_rounds)
        self.default_seed = default_seed
        self.default_rounds = default_rounds
        self.obfuscation_cache: Dict[str, Dict[str, Any]] = {}
        logger.info("Hades Clockworks Obfuscation Integration loaded")

    def obfuscate_payload(
        self,
        payload: str,
        payload_type: ObfuscationType = ObfuscationType.PAYLOAD,
        seed: Optional[int] = None,
        rounds: Optional[int] = None,
        cache_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Obfuscate a security payload for Hades operations

        Args:
            payload: Payload content (string or bytes)
            payload_type: Type of obfuscation
            seed: Custom seed (uses default if None)
            rounds: Custom rounds (uses default if None)
            cache_key: Optional cache key for this payload

        Returns:
            Dictionary with obfuscated payload and metadata
        """
        try:
            seed = seed or self.default_seed
            rounds = rounds or self.default_rounds

            # Check cache
            if cache_key and cache_key in self.obfuscation_cache:
                logger.debug(f"Using cached obfuscation for {cache_key}")
                return self.obfuscation_cache[cache_key]

            if payload_type == ObfuscationType.LUA:
                obfuscated = self.obfuscator.obfuscate_lua(payload)
                result = {
                    "type": "lua_obfuscated",
                    "obfuscated": obfuscated,
                    "original_size": len(payload),
                    "obfuscated_size": len(obfuscated),
                    "seed": seed,
                    "rounds": rounds,
                    "format": "lua",
                }
            else:
                # Binary obfuscation for other types
                payload_bytes = payload.encode() if isinstance(payload, str) else payload
                obfuscated = self.obfuscator.obfuscate_binary(payload_bytes, format="b64")
                result = {
                    "type": f"{payload_type.value}_obfuscated",
                    "obfuscated": obfuscated,
                    "original_size": len(payload_bytes),
                    "obfuscated_size": len(obfuscated),
                    "seed": seed,
                    "rounds": rounds,
                    "format": "b64",
                }

            # Cache if key provided
            if cache_key:
                self.obfuscation_cache[cache_key] = result
                logger.debug(f"Cached obfuscation as {cache_key}")

            logger.info(f"Obfuscated {payload_type.value} payload ({len(payload)} -> {result['obfuscated_size']} bytes)")
            return result

        except Exception as e:
            logger.error(f"Payload obfuscation failed: {e}")
            raise

    def deobfuscate_payload(
        self,
        obfuscated: str,
        payload_type: ObfuscationType = ObfuscationType.PAYLOAD,
        seed: Optional[int] = None,
        rounds: Optional[int] = None,
    ) -> bytes:
        """
        Deobfuscate a security payload

        Args:
            obfuscated: Obfuscated payload
            payload_type: Type of payload
            seed: Seed used (uses default if None)
            rounds: Rounds used (uses default if None)

        Returns:
            Original payload as bytes
        """
        try:
            seed = seed or self.default_seed
            rounds = rounds or self.default_rounds

            if payload_type == ObfuscationType.LUA:
                # For Lua, we can't easily deobfuscate the loader format
                # This is by design - Lua obfuscation is meant to be run, not extracted
                logger.warning("Lua payload deobfuscation not supported (run payload instead)")
                return None

            original = deobfuscate(obfuscated, seed=seed, rounds=rounds, format="b64")
            logger.info(f"Deobfuscated {payload_type.value} payload ({len(obfuscated)} -> {len(original)} bytes)")
            return original

        except Exception as e:
            logger.error(f"Payload deobfuscation failed: {e}")
            raise

    def obfuscate_batch(
        self,
        payloads: List[str],
        payload_type: ObfuscationType = ObfuscationType.PAYLOAD,
        seed: Optional[int] = None,
        rounds: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Obfuscate multiple payloads efficiently

        Args:
            payloads: List of payloads to obfuscate
            payload_type: Type of obfuscation
            seed: Custom seed
            rounds: Custom rounds

        Returns:
            List of obfuscation results
        """
        results = []
        for i, payload in enumerate(payloads):
            try:
                result = self.obfuscate_payload(payload, payload_type, seed, rounds)
                results.append(result)
                logger.debug(f"Batch obfuscation: {i+1}/{len(payloads)}")
            except Exception as e:
                logger.error(f"Batch obfuscation failed for item {i}: {e}")
                results.append({"error": str(e), "index": i})

        return results

    def generate_polymorph_payload(
        self,
        payload: str,
        variations: int = 5,
        payload_type: ObfuscationType = ObfuscationType.PAYLOAD,
    ) -> List[Dict[str, Any]]:
        """
        Generate polymorphic variations of a payload using different seeds

        Args:
            payload: Original payload
            variations: Number of variations to generate
            payload_type: Type of payload

        Returns:
            List of polymorphic obfuscated payloads
        """
        results = []
        for i in range(variations):
            seed = ((i + self.default_seed) % 12) or 12
            rounds = self.default_rounds + (i % 3)
            result = self.obfuscate_payload(payload, payload_type, seed, rounds)
            result["variation"] = i + 1
            results.append(result)
            logger.debug(f"Generated polymorphic variation {i+1}/{variations}")

        return results

    def get_obfuscation_stats(self) -> Dict[str, Any]:
        """Get statistics about cached obfuscations"""
        total_obfuscated = sum(r.get("original_size", 0) for r in self.obfuscation_cache.values())
        total_result = sum(r.get("obfuscated_size", 0) for r in self.obfuscation_cache.values())

        return {
            "cached_payloads": len(self.obfuscation_cache),
            "total_original_size": total_obfuscated,
            "total_obfuscated_size": total_result,
            "compression_ratio": total_result / total_obfuscated if total_obfuscated > 0 else 0,
            "default_seed": self.default_seed,
            "default_rounds": self.default_rounds,
        }

    def clear_cache(self) -> None:
        """Clear the obfuscation cache"""
        self.obfuscation_cache.clear()
        logger.info("Obfuscation cache cleared")

    def update_defaults(self, seed: Optional[int] = None, rounds: Optional[int] = None) -> None:
        """Update default seed and rounds"""
        if seed is not None:
            self.default_seed = seed
            self.obfuscator = ClockworksObfuscator(seed=self.default_seed, rounds=self.default_rounds)
        if rounds is not None:
            self.default_rounds = rounds
            self.obfuscator = ClockworksObfuscator(seed=self.default_seed, rounds=self.default_rounds)
        logger.info(f"Defaults updated: seed={self.default_seed}, rounds={self.default_rounds}")
    
    def set_seed(self, seed: int) -> None:
        """Alias for update_defaults"""
        self.update_defaults(seed=seed)
    
    def set_rounds(self, rounds: int) -> None:
        """Alias for update_defaults"""
        self.update_defaults(rounds=rounds)


# Global integration instance
_hades_obfuscation: Optional[HadesObfuscationIntegration] = None


def get_obfuscation_service() -> HadesObfuscationIntegration:
    """Get or create the global Hades obfuscation service"""
    global _hades_obfuscation
    if _hades_obfuscation is None:
        _hades_obfuscation = HadesObfuscationIntegration()
    return _hades_obfuscation


def obfuscate_for_hades(
    payload: str,
    payload_type: str = "payload",
    seed: Optional[int] = None,
    rounds: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Convenience function to obfuscate payloads through Hades

    Args:
        payload: Payload to obfuscate
        payload_type: Type of payload ('lua', 'payload', 'shellcode', 'command', 'script', 'binary')
        seed: Optional custom seed
        rounds: Optional custom rounds

    Returns:
        Dictionary with obfuscated payload and metadata
    """
    service = get_obfuscation_service()
    ptype = ObfuscationType[payload_type.upper()] if hasattr(ObfuscationType, payload_type.upper()) else ObfuscationType.PAYLOAD
    return service.obfuscate_payload(payload, ptype, seed, rounds)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test integration
    service = get_obfuscation_service()

    test_payload = "print('Hello from Clockworks Obfuscation!')"
    result = service.obfuscate_payload(test_payload, ObfuscationType.LUA)
    print(f"\nObfuscated Lua Payload:\n{result['obfuscated'][:200]}...\n")

    # Test polymorphic generation
    variations = service.generate_polymorph_payload(test_payload, variations=3, payload_type=ObfuscationType.LUA)
    print(f"Generated {len(variations)} polymorphic variations")

    # Test statistics
    stats = service.get_obfuscation_stats()
    print(f"\nObfuscation Statistics: {stats}")
