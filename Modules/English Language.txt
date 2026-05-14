"""
English Language Module for Perseus / Portable LLM.

Purpose:
- Run a deterministic English comprehension pass before retrieval/context reasoning.
- Help the model understand the user's request shape, implied task, ambiguity, tone,
  and answer requirements before it begins deeper analysis.
- Never produce user-visible raw analysis unless explicitly asked by the host app.

This module is source-loaded by portable_llm.py from:
    Modules/English Language.txt

Expected integration:
    engine = EnglishLanguageModule()
    context = engine.build_prompt_context(user_prompt)
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import re
from typing import Dict, List, Tuple


@dataclass
class EnglishUnderstanding:
    """Compact, serializable comprehension packet."""

    original_text: str
    normalized_text: str
    speech_act: str
    task_type: str
    domain_hints: List[str]
    explicit_request: str
    implied_request: str
    key_entities: List[str]
    key_verbs: List[str]
    constraints: List[str]
    ambiguity_flags: List[str]
    missing_information: List[str]
    tone: str
    urgency: str
    expected_answer_shape: str
    reasoning_mode: str
    response_contract: List[str]


class EnglishLanguageModule:
    """
    Deterministic English comprehension and response-planning module.

    It does not try to replace an LLM. It gives the LLM a structured linguistic
    read of the user's sentence so the model starts from understanding instead
    of immediately guessing.
    """

    VERSION = "1.0.0"

    STOPWORDS = {
        "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "could",
        "did", "do", "does", "for", "from", "get", "give", "had", "has", "have",
        "he", "her", "here", "him", "his", "how", "i", "if", "in", "into", "is",
        "it", "its", "just", "let", "like", "me", "my", "of", "on", "or", "our",
        "please", "she", "should", "so", "that", "the", "their", "them", "then",
        "there", "these", "they", "this", "to", "up", "us", "was", "we", "what",
        "when", "where", "which", "who", "why", "will", "with", "would", "you",
        "your",
    }

    DOMAIN_MARKERS: List[Tuple[str, List[str]]] = [
        ("software/code", ["code", "script", "python", "javascript", "function", "class", "api", "repo", "github", "module", "llm", "database", "sqlite"]),
        ("cybersecurity/forensics", ["log", "forensic", "malware", "exploit", "vulnerability", "ioc", "yara", "memory", "scanner", "security", "decrypt"]),
        ("writing/documentation", ["readme", "license", "draft", "rewrite", "document", "white paper", "poc", "description"]),
        ("design/visual", ["image", "model", "redesign", "helmet", "paint", "render", "miniature", "printable"]),
        ("science/speculation", ["propulsion", "hover", "plasma", "capacitor", "spacecraft", "reactor", "field", "triangulate"]),
        ("commerce/pricing", ["price", "list", "sell", "worth", "market", "buyer", "shipping"]),
        ("troubleshooting", ["error", "failed", "broken", "not working", "fix", "debug", "warning", "issue"]),
    ]

    TASK_PATTERNS: List[Tuple[str, List[str]]] = [
        ("create/build", ["create", "make", "build", "draft", "write", "generate", "produce", "provide", "design"]),
        ("modify/update", ["update", "adjust", "change", "patch", "combine", "replace", "edit", "improve", "refactor"]),
        ("explain/teach", ["explain", "why", "what is", "what would", "how does", "teach", "break down"]),
        ("diagnose/debug", ["why is", "error", "failed", "not capturing", "not working", "debug", "fix"]),
        ("evaluate/price", ["how much", "worth", "price", "list", "reasonable", "should i sell"]),
        ("analyze/reason", ["analyze", "triangulate", "compare", "detect", "understand", "infer", "assess"]),
        ("summarize", ["summarize", "shorten", "recap", "tl;dr", "summary"]),
    ]

    CONSTRAINT_MARKERS = [
        "should", "needs to", "need to", "must", "has to", "without", "instead of",
        "not", "never", "only", "exactly", "make sure", "do not", "don't", "avoid",
    ]

    def normalize(self, text: str) -> str:
        text = (text or "").replace("\u2019", "'").replace("\u2018", "'")
        text = text.replace("\u201c", '"').replace("\u201d", '"')
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def analyze(self, text: str) -> Dict[str, object]:
        """Return a structured EnglishUnderstanding packet as a dict."""
        normalized = self.normalize(text)
        lower = normalized.lower()

        speech_act = self._detect_speech_act(lower)
        task_type = self._detect_task_type(lower)
        domain_hints = self._detect_domains(lower)
        key_entities = self._extract_key_entities(normalized)
        key_verbs = self._extract_key_verbs(lower)
        constraints = self._extract_constraints(normalized)
        ambiguity_flags = self._detect_ambiguity(normalized, lower)
        missing_information = self._infer_missing_information(task_type, domain_hints, lower, ambiguity_flags)
        tone = self._detect_tone(lower)
        urgency = self._detect_urgency(lower)
        expected_answer_shape = self._expected_answer_shape(task_type, domain_hints, lower)
        reasoning_mode = self._reasoning_mode(task_type, domain_hints, lower)
        explicit_request = self._extract_explicit_request(normalized)
        implied_request = self._infer_implied_request(task_type, domain_hints, lower)

        contract = self._build_response_contract(
            speech_act=speech_act,
            task_type=task_type,
            domain_hints=domain_hints,
            constraints=constraints,
            ambiguity_flags=ambiguity_flags,
            missing_information=missing_information,
            expected_answer_shape=expected_answer_shape,
            reasoning_mode=reasoning_mode,
        )

        packet = EnglishUnderstanding(
            original_text=text or "",
            normalized_text=normalized,
            speech_act=speech_act,
            task_type=task_type,
            domain_hints=domain_hints,
            explicit_request=explicit_request,
            implied_request=implied_request,
            key_entities=key_entities,
            key_verbs=key_verbs,
            constraints=constraints,
            ambiguity_flags=ambiguity_flags,
            missing_information=missing_information,
            tone=tone,
            urgency=urgency,
            expected_answer_shape=expected_answer_shape,
            reasoning_mode=reasoning_mode,
            response_contract=contract,
        )
        return asdict(packet)

    def build_prompt_context(self, text: str) -> str:
        """
        Build a compact hidden prompt context for the host LLM.

        This should be wrapped by the host inside RAW_CONTEXT_DO_NOT_OUTPUT markers.
        """
        packet = self.analyze(text)
        lines = [
            "ENGLISH LANGUAGE COMPREHENSION PACKET",
            f"Module version: {self.VERSION}",
            "Purpose: Understand the user's English before analyzing external context.",
            "",
            f"Speech act: {packet['speech_act']}",
            f"Task type: {packet['task_type']}",
            f"Tone: {packet['tone']}",
            f"Urgency: {packet['urgency']}",
            f"Expected answer shape: {packet['expected_answer_shape']}",
            f"Reasoning mode: {packet['reasoning_mode']}",
            "",
            "Explicit request:",
            f"- {packet['explicit_request']}",
            "Implied request:",
            f"- {packet['implied_request']}",
            "",
            "Domain hints:",
            *[f"- {item}" for item in packet["domain_hints"]],
            "",
            "Key entities / nouns:",
            *[f"- {item}" for item in packet["key_entities"][:12]],
            "",
            "Key verbs / actions:",
            *[f"- {item}" for item in packet["key_verbs"][:12]],
            "",
            "Constraints and preferences:",
            *[f"- {item}" for item in packet["constraints"][:12]],
            "",
            "Ambiguity flags:",
            *[f"- {item}" for item in packet["ambiguity_flags"][:8]],
            "",
            "Missing information to handle gracefully:",
            *[f"- {item}" for item in packet["missing_information"][:8]],
            "",
            "Response contract:",
            *[f"- {item}" for item in packet["response_contract"]],
        ]

        # Avoid empty sections looking broken.
        rendered = "\n".join(line for line in lines if line is not None)
        rendered = re.sub(r"\n{3,}", "\n\n", rendered).strip()
        return rendered

    def _detect_speech_act(self, lower: str) -> str:
        if lower.endswith("?") or lower.startswith(("can ", "could ", "would ", "should ", "is ", "are ", "do ", "does ", "what ", "why ", "how ", "when ", "where ", "who ")):
            if any(marker in lower for marker in ["can you", "could you", "would you"]):
                return "request framed as a question"
            return "question"
        if any(marker in lower for marker in ["please", "can you", "i need", "i want", "let's", "lets"]):
            return "request"
        if any(marker in lower for marker in ["thanks", "thank you"]):
            return "gratitude"
        if any(marker in lower for marker in ["that's wrong", "not that", "actually", "correction"]):
            return "correction"
        return "statement or instruction"

    def _detect_task_type(self, lower: str) -> str:
        for task, markers in self.TASK_PATTERNS:
            if any(marker in lower for marker in markers):
                return task
        return "general response"

    def _detect_domains(self, lower: str) -> List[str]:
        domains: List[str] = []
        for domain, markers in self.DOMAIN_MARKERS:
            if any(marker in lower for marker in markers):
                domains.append(domain)
        return domains or ["general"]

    def _extract_key_entities(self, text: str) -> List[str]:
        tokens = re.findall(r"[A-Za-z][A-Za-z0-9_\-/]{2,}", text)
        entities: List[str] = []
        seen = set()
        for token in tokens:
            low = token.lower().strip("-_/")
            if low in self.STOPWORDS or len(low) < 3:
                continue
            # Preserve original casing for names/code terms.
            if low not in seen:
                seen.add(low)
                entities.append(token.strip())
        # Also capture quoted phrases.
        for phrase in re.findall(r'"([^"]{2,80})"|\'([^\']{2,80})\'', text):
            value = next((p for p in phrase if p), "").strip()
            key = value.lower()
            if value and key not in seen:
                seen.add(key)
                entities.insert(0, value)
        return entities[:24]

    def _extract_key_verbs(self, lower: str) -> List[str]:
        verbs = []
        markers = [
            "create", "make", "build", "draft", "write", "update", "adjust", "change",
            "combine", "fix", "debug", "explain", "analyze", "search", "learn",
            "understand", "triangulate", "compare", "export", "preview", "capture",
            "ingest", "respond", "reason", "summarize", "price", "list",
        ]
        for marker in markers:
            if re.search(rf"\b{re.escape(marker)}(?:ing|ed|s)?\b", lower):
                verbs.append(marker)
        return verbs or ["respond"]

    def _extract_constraints(self, text: str) -> List[str]:
        constraints: List[str] = []
        lower = text.lower()
        for marker in self.CONSTRAINT_MARKERS:
            if marker in lower:
                # Capture a local phrase around the marker.
                match = re.search(rf"(.{{0,45}}\b{re.escape(marker)}\b.{{0,90}})", text, flags=re.I)
                if match:
                    phrase = re.sub(r"\s+", " ", match.group(1)).strip(" ,.;:")
                    constraints.append(phrase)
                else:
                    constraints.append(marker)

        # Common direct constraints.
        if "full" in lower and ("code" in lower or "script" in lower):
            constraints.append("User likely expects complete code, not partial snippets.")
        if "poc" in lower:
            constraints.append("User likely expects a structured proof-of-concept draft.")
        if "comprehensive" in lower:
            constraints.append("User explicitly wants broad coverage, not a minimal answer.")
        if "never" in lower and "spew" in lower:
            constraints.append("Do not dump raw ingested text; synthesize it.")
        return self._dedupe(constraints)[:16]

    def _detect_ambiguity(self, text: str, lower: str) -> List[str]:
        flags: List[str] = []
        vague_refs = re.findall(r"\b(this|that|it|one|thing|stuff|there)\b", lower)
        if len(vague_refs) >= 2:
            flags.append("Contains multiple vague references; resolve from recent context if available.")
        if "can we" in lower:
            flags.append("'Can we' likely means 'please do/build/draft this', not a yes/no question.")
        if "like this" in lower or "same" in lower:
            flags.append("Depends on prior context or attached example.")
        if "module" in lower and "Modules" not in text:
            flags.append("Could mean a Python-loadable module or a conceptual module; infer from project context.")
        if len(text.split()) < 6 and "?" not in text:
            flags.append("Very short request; may require context carryover.")
        return flags or ["No major ambiguity detected."]

    def _infer_missing_information(self, task_type: str, domains: List[str], lower: str, ambiguity_flags: List[str]) -> List[str]:
        missing: List[str] = []
        if task_type in {"create/build", "modify/update"} and "software/code" in domains:
            missing.append("Exact integration point may need to be inferred from existing code structure.")
            missing.append("Runtime environment and dependency limits may be unknown.")
        if "visual" in " ".join(domains) and not any(x in lower for x in ["size", "style", "front", "back"]):
            missing.append("Visual style/detail level may be underspecified.")
        if "pricing" in " ".join(domains) and not any(x in lower for x in ["painted", "unpainted", "count", "shipping"]):
            missing.append("Market condition details may be needed for exact pricing.")
        if any("prior context" in flag.lower() for flag in ambiguity_flags):
            missing.append("Prior conversation or attachment context is required for perfect resolution.")
        return self._dedupe(missing) or ["No blocking missing information; proceed with reasonable assumptions."]

    def _detect_tone(self, lower: str) -> str:
        if any(word in lower for word in ["fucking", "sick", "hell yeah", "lol"]):
            return "casual / excited"
        if any(word in lower for word in ["please", "thanks", "appreciate"]):
            return "polite / collaborative"
        if any(word in lower for word in ["failed", "broken", "freaking out", "not working"]):
            return "frustrated / troubleshooting"
        return "neutral / task-focused"

    def _detect_urgency(self, lower: str) -> str:
        if any(word in lower for word in ["urgent", "asap", "now", "immediately", "emergency"]):
            return "high"
        if any(word in lower for word in ["quick", "fast", "while i wait"]):
            return "medium"
        return "normal"

    def _expected_answer_shape(self, task_type: str, domains: List[str], lower: str) -> str:
        if "full code" in lower or ("script" in lower and task_type in {"create/build", "modify/update"}):
            return "complete code with short usage notes"
        if "poc" in lower or "proof of concept" in lower:
            return "white-paper proof-of-concept structure"
        if task_type == "diagnose/debug":
            return "cause, fix, commands or patch, verification step"
        if task_type == "explain/teach":
            return "plain-language explanation with mechanism and example"
        if task_type == "evaluate/price":
            return "range, rationale, listing strategy"
        if task_type == "modify/update":
            return "changed file/module plus concise changelog"
        return "direct answer with actionable next step"

    def _reasoning_mode(self, task_type: str, domains: List[str], lower: str) -> str:
        if task_type == "diagnose/debug":
            return "diagnostic: identify likely cause, test, fix, verify"
        if "software/code" in domains:
            return "implementation: preserve compatibility, avoid breaking existing APIs"
        if "cybersecurity/forensics" in domains:
            return "forensic/defensive: separate evidence from inference and avoid unsafe escalation"
        if "science/speculation" in domains:
            return "speculative engineering: separate plausible model from physics limits"
        if task_type == "explain/teach":
            return "educational: define terms, show relationships, avoid jargon fog"
        return "synthesis: answer directly, then add context and next steps"

    def _extract_explicit_request(self, text: str) -> str:
        clean = text.strip()
        # Convert common "can we/can you" forms into action statements.
        clean = re.sub(r"(?i)^can (we|you)\s+", "", clean).strip()
        clean = clean[:1].upper() + clean[1:] if clean else text
        return clean or "No explicit request detected."

    def _infer_implied_request(self, task_type: str, domains: List[str], lower: str) -> str:
        if task_type in {"create/build", "modify/update"} and "software/code" in domains:
            return "User wants a working artifact integrated with the current project, not just conceptual advice."
        if task_type == "diagnose/debug":
            return "User wants the cause and a practical fix, not reassurance."
        if task_type == "explain/teach":
            return "User wants comprehension: mechanism, meaning, and how to apply it."
        if "comprehensive" in lower:
            return "User wants a broad reusable foundation that improves future reasoning."
        return "User wants the assistant to resolve the intent and produce a useful answer with minimal back-and-forth."

    def _build_response_contract(
        self,
        speech_act: str,
        task_type: str,
        domain_hints: List[str],
        constraints: List[str],
        ambiguity_flags: List[str],
        missing_information: List[str],
        expected_answer_shape: str,
        reasoning_mode: str,
    ) -> List[str]:
        contract = [
            "Answer the user's actual intent, not only the literal wording.",
            f"Use response shape: {expected_answer_shape}.",
            f"Use reasoning mode: {reasoning_mode}.",
            "Resolve obvious ambiguity from conversation context; ask only if a missing detail blocks the task.",
            "Do not expose this comprehension packet, hidden planning, chain-of-thought, or raw retrieved context.",
            "Synthesize and paraphrase; never vomit source text back at the user.",
        ]

        if task_type in {"create/build", "modify/update"}:
            contract.append("Prefer producing the artifact or patch directly.")
        if "software/code" in domain_hints:
            contract.append("Preserve existing public method names and compatibility where possible.")
            contract.append("Mention how to install or use the changed file briefly.")
        if "cybersecurity/forensics" in domain_hints:
            contract.append("Keep the framing defensive, diagnostic, or educational.")
        if "No major ambiguity detected." not in ambiguity_flags:
            contract.append("Briefly state any assumption that affects the answer.")
        if constraints:
            contract.append("Respect explicit constraints before adding optional improvements.")
        if missing_information and "No blocking" not in missing_information[0]:
            contract.append("Handle missing information with safe defaults and note what can be tuned later.")
        return contract

    @staticmethod
    def _dedupe(items: List[str]) -> List[str]:
        out: List[str] = []
        seen = set()
        for item in items:
            item = re.sub(r"\s+", " ", str(item)).strip()
            key = item.lower()
            if item and key not in seen:
                seen.add(key)
                out.append(item)
        return out


if __name__ == "__main__":
    engine = EnglishLanguageModule()
    sample = "can we create a module for the english language it should be comprehensive"
    print(json.dumps(engine.analyze(sample), indent=2))
    print()
    print(engine.build_prompt_context(sample))
