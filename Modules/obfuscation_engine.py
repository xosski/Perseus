"""
Clockworks Obfuscation Engine
Integrates the Clock-Direction RNG driftwheel keystream obfuscator into Hades AI
"""

from __future__ import annotations
import base64
from dataclasses import dataclass
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class DriftState:
    """State for the driftwheel RNG"""
    shape: int  # 0..5
    color: int  # 0..5
    direction: int  # 1..12


# 12 clock positions -> angle bucket + base "shape/color" impulses
DIRECTIONS = {
    1: (1, 0),
    2: (2, 1),
    3: (3, 2),
    4: (4, 3),
    5: (5, 4),
    6: (0, 5),
    7: (1, 4),
    8: (2, 3),
    9: (3, 2),
    10: (4, 1),
    11: (5, 0),
    12: (0, 1),
}


def _mix(a: int, b: int) -> int:
    """Small nonlinear mixer in 0..255 space"""
    x = (a * 73 + b * 151 + 19) & 0xFF
    x ^= ((x << 3) & 0xFF)
    x ^= (x >> 5)
    return x & 0xFF


def _step(st: DriftState, i: int) -> int:
    """Execute one driftwheel step"""
    base_shape, base_color = DIRECTIONS[st.direction]
    
    # "morph" shape and color using recursive drift
    st.shape = (st.shape + base_shape + (i * 3)) % 6
    st.color = (st.color + base_color + (i * 5)) % 6

    # derive a pseudo "area" value from shape/color/direction
    area = (st.shape + 1) * (st.color + 2) * (st.direction + 7)
    area = _mix(area & 0xFF, (area >> 8) & 0xFF)

    # drift direction (clock step with feedback)
    drift = ((area % 11) + 1)  # 1..12-ish
    st.direction = ((st.direction + drift + (i * 3)) % 12) or 12

    return area


def keystream(seed: int, n: int, rounds: int = 9) -> bytes:
    """Generate deterministic keystream using clock-direction RNG"""
    st = DriftState(shape=0, color=0, direction=((seed - 1) % 12) + 1)
    out = bytearray()
    i = 0
    while len(out) < n:
        # each byte comes from several drift rounds to add diffusion
        v = 0
        for _ in range(rounds):
            v = _mix(v, _step(st, i))
            i += 1
        out.append(v)
    return bytes(out)


def xor_bytes(data: bytes, ks: bytes) -> bytes:
    """XOR data with keystream"""
    return bytes(b ^ ks[i % len(ks)] for i, b in enumerate(data))


class ClockworksObfuscator:
    """Hades AI Clockworks Obfuscation Engine"""

    LUA_LOADER_TEMPLATE = """-- Generated Clockworks Obfuscator (Clock-Drift Packer)
-- NOT real security. Reversible. For basic IP deterrence.

local b64 = [[{B64_PAYLOAD}]]
local SEED = {SEED}
local ROUNDS = {ROUNDS}

local function b64dec(data)
  local b='ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/'
  data = data:gsub('[^'..b..'=]', '')
  return (data:gsub('.', function(x)
    if x == '=' then return '' end
    local r,f='',(b:find(x)-1)
    for i=6,1,-1 do
      r=r..(f%2^i - f%2^(i-1) > 0 and '1' or '0')
    end
    return r
  end):gsub('%d%d%d?%d?%d?%d?%d?%d?', function(x)
    if #x ~= 8 then return '' end
    local c=0
    for i=1,8 do
      c=c + (x:sub(i,i)=='1' and 2^(8-i) or 0)
    end
    return string.char(c)
  end))
end

-- Clock-Direction Driftwheel RNG (Lua side)
local DIRECTIONS = {
  [1]={1,0}, [2]={2,1}, [3]={3,2}, [4]={4,3}, [5]={5,4}, [6]={0,5},
  [7]={1,4}, [8]={2,3}, [9]={3,2}, [10]={4,1}, [11]={5,0}, [12]={0,1},
}

local function mix(a,b)
  local x = (a*73 + b*151 + 19) % 256
  x = (x ~ ((x << 3) % 256)) % 256
  x = (x ~ (x >> 5)) % 256
  return x % 256
end

local function step(st, i)
  local base = DIRECTIONS[st.dir]
  local baseShape, baseColor = base[1], base[2]

  st.shape = (st.shape + baseShape + (i*3)) % 6
  st.color = (st.color + baseColor + (i*5)) % 6

  local area = (st.shape+1) * (st.color+2) * (st.dir+7)
  area = mix(area % 256, math.floor(area / 256) % 256)

  local drift = (area % 11) + 1
  st.dir = ((st.dir + drift + (i*3)) % 12)
  if st.dir == 0 then st.dir = 12 end

  return area
end

local function keystream(seed, n, rounds)
  local st = { shape=0, color=0, dir=((seed-1) % 12)+1 }
  local out = {}
  local i, v = 0, 0
  for k=1,n do
    v = 0
    for _=1,rounds do
      v = mix(v, step(st, i))
      i = i + 1
    end
    out[k] = string.char(v)
  end
  return table.concat(out)
end

local function xor_str(data, ks)
  local out = {}
  local klen = #ks
  for i=1,#data do
    local db = data:byte(i)
    local kb = ks:byte(((i-1) % klen) + 1)
    out[i] = string.char(db ~ kb)
  end
  return table.concat(out)
end

local enc = b64dec(b64)
local ks = keystream(SEED, math.max(64, #enc), ROUNDS)
local src = xor_str(enc, ks)

local fn, err = load(src, "clock_drift_payload", "t", _G)
if not fn then error("decode/load failed: "..tostring(err)) end
return fn()
"""

    def __init__(self, seed: int = 7, rounds: int = 9):
        """Initialize obfuscator with seed and round count"""
        self.seed = seed
        self.rounds = rounds
        logger.info(f"Clockworks Obfuscator initialized (seed={seed}, rounds={rounds})")

    def obfuscate_lua(self, lua_code: str) -> str:
        """
        Obfuscate Lua code using clock-direction RNG

        Args:
            lua_code: Source Lua code as string

        Returns:
            Obfuscated Lua code with embedded loader
        """
        try:
            src_bytes = lua_code.encode("utf-8")
            ks = keystream(self.seed, max(64, len(src_bytes)), self.rounds)
            enc = xor_bytes(src_bytes, ks)
            b64 = base64.b64encode(enc).decode("ascii")

            obf = self.LUA_LOADER_TEMPLATE.format(
                B64_PAYLOAD=b64,
                SEED=int(self.seed),
                ROUNDS=int(self.rounds),
            )
            logger.info(f"Lua obfuscation complete: {len(src_bytes)} -> {len(obf)} bytes")
            return obf
        except Exception as e:
            logger.error(f"Lua obfuscation failed: {e}")
            raise

    def obfuscate_binary(self, data: bytes, format: str = "hex") -> str:
        """
        Obfuscate binary data using clock-direction RNG

        Args:
            data: Binary data to obfuscate
            format: Output format ('hex', 'b64', 'bin')

        Returns:
            Obfuscated data in specified format
        """
        try:
            ks = keystream(self.seed, max(64, len(data)), self.rounds)
            enc = xor_bytes(data, ks)

            if format == "hex":
                result = enc.hex()
            elif format == "b64":
                result = base64.b64encode(enc).decode("ascii")
            elif format == "bin":
                result = enc.decode("latin1")
            else:
                raise ValueError(f"Unknown format: {format}")

            logger.info(f"Binary obfuscation complete: {len(data)} -> {len(result)} bytes")
            return result
        except Exception as e:
            logger.error(f"Binary obfuscation failed: {e}")
            raise

    def deobfuscate(self, obfuscated_data: bytes, format: str = "hex") -> bytes:
        """
        Deobfuscate data using clock-direction RNG (reverse operation)

        Args:
            obfuscated_data: Obfuscated data
            format: Input format ('hex', 'b64', 'bin')

        Returns:
            Original data
        """
        try:
            if format == "hex":
                enc = bytes.fromhex(obfuscated_data.decode() if isinstance(obfuscated_data, bytes) else obfuscated_data)
            elif format == "b64":
                enc = base64.b64decode(obfuscated_data)
            elif format == "bin":
                enc = obfuscated_data
            else:
                raise ValueError(f"Unknown format: {format}")

            ks = keystream(self.seed, max(64, len(enc)), self.rounds)
            original = xor_bytes(enc, ks)
            logger.info(f"Deobfuscation complete: {len(enc)} -> {len(original)} bytes")
            return original
        except Exception as e:
            logger.error(f"Deobfuscation failed: {e}")
            raise

    def set_seed(self, seed: int) -> None:
        """Update the seed value"""
        if not (1 <= seed <= 12):
            logger.warning(f"Seed {seed} out of range 1-12, normalizing...")
            seed = ((seed - 1) % 12) + 1
        self.seed = seed
        logger.info(f"Seed updated to {self.seed}")

    def set_rounds(self, rounds: int) -> None:
        """Update the rounds value"""
        if rounds < 1:
            logger.warning(f"Rounds {rounds} too low, setting to 9")
            rounds = 9
        self.rounds = rounds
        logger.info(f"Rounds updated to {self.rounds}")


# Convenience functions for quick obfuscation
def obfuscate(data: bytes, seed: int = 7, rounds: int = 9, format: str = "hex") -> str:
    """Quick obfuscation function"""
    obfuscator = ClockworksObfuscator(seed=seed, rounds=rounds)
    if isinstance(data, str):
        data = data.encode("utf-8")
    return obfuscator.obfuscate_binary(data, format=format)


def deobfuscate(obfuscated: str, seed: int = 7, rounds: int = 9, format: str = "hex") -> bytes:
    """Quick deobfuscation function"""
    obfuscator = ClockworksObfuscator(seed=seed, rounds=rounds)
    return obfuscator.deobfuscate(obfuscated.encode() if isinstance(obfuscated, str) else obfuscated, format=format)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test example
    test_data = b"Hello, Clockworks Obfuscation!"
    obf = ClockworksObfuscator(seed=7, rounds=9)
    
    encrypted = obf.obfuscate_binary(test_data, format="b64")
    print(f"Original: {test_data}")
    print(f"Obfuscated (b64): {encrypted}")
    
    decrypted = obf.deobfuscate(encrypted, format="b64")
    print(f"Deobfuscated: {decrypted}")
    print(f"Match: {decrypted == test_data}")
