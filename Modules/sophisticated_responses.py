# modules/sophisticated_responses.py
# Enhanced response generation with OpenAI/Mistral-style sophistication

import random
import datetime
import json
from typing import Dict, List, Optional, Tuple
import re

class SophisticatedResponseEngine:
    """
    Advanced response generation engine with features inspired by:
    - OpenAI's structured reasoning and thinking traces
    - Mistral's technical precision and multi-step reasoning
    - Context-aware response depth and complexity adjustment
    - Intelligent formatting and progression
    """
    
    def __init__(self):
        self.thinking_styles = {
            "analytical": "Breaking down the problem systematically...",
            "creative": "Exploring novel approaches and connections...",
            "empirical": "Weighing evidence and practical outcomes...",
            "theoretical": "Examining fundamental principles and theory...",
            "pragmatic": "Focusing on actionable solutions..."
        }
        
        self.reasoning_markers = [
            "First, consider that",
            "It's worth noting that",
            "The key insight here is",
            "More importantly,",
            "This suggests that",
            "In practice, this means",
            "To illustrate this point,",
            "Building on this logic,",
            "The critical factor is",
            "Additionally,"
        ]
        
        self.structured_formats = {
            "technical": {
                "high": "**Technical Analysis**\n\n{content}\n\n**Key Implications:**\n{summary}",
                "low": "**Overview:**\n{content}"
            },
            "strategic": {
                "high": "**Strategic Approach**\n\n{content}\n\n**Implementation Path:**\n{summary}",
                "low": "**Strategy:**\n{content}"
            },
            "educational": {
                "high": "**Comprehensive Explanation**\n\n{content}\n\n**Practical Application:**\n{summary}",
                "low": "**Simple Explanation:**\n{content}"
            },
            "analytical": {
                "high": "**Detailed Analysis**\n\n{content}\n\n**Conclusions:**\n{summary}",
                "low": "**Analysis:**\n{content}"
            }
        }
    
    def generate_thinking_process(self, user_input: str, thinking_style: str = "analytical") -> str:
        """Generate visible reasoning/thinking process like OpenAI o1"""
        thinking_lines = []
        thinking_lines.append("<thinking>")
        thinking_lines.append(f"Approach: {self.thinking_styles.get(thinking_style, self.thinking_styles['analytical'])}")
        
        # Analyze query complexity
        word_count = len(user_input.split())
        complexity_assessment = "High complexity - multi-faceted query" if word_count > 10 else "Moderate complexity"
        thinking_lines.append(f"Query complexity: {complexity_assessment}")
        
        # Extract key concepts
        key_concepts = self._extract_concepts(user_input)
        if key_concepts:
            thinking_lines.append(f"Key concepts: {', '.join(key_concepts[:3])}")
        
        # Determine reasoning depth
        if any(word in user_input.lower() for word in ['how', 'why', 'what if', 'explain', 'analyze']):
            thinking_lines.append("Reasoning depth: Deep technical analysis required")
        else:
            thinking_lines.append("Reasoning depth: Straightforward explanation sufficient")
        
        # Response structure
        thinking_lines.append("Response structure: Hierarchical progression from concepts to applications")
        thinking_lines.append("</thinking>\n")
        
        return "\n".join(thinking_lines)
    
    def _extract_concepts(self, text: str) -> List[str]:
        """Extract key concepts from input"""
        concepts = []
        tech_terms = {
            'security': ['security', 'secure', 'attack', 'defense'],
            'vulnerability': ['vulnerability', 'vulnerable', 'weakness', 'flaw', 'cve'],
            'network': ['network', 'tcp', 'ip', 'protocol', 'packet'],
            'encryption': ['encryption', 'encrypt', 'cipher', 'key', 'hash'],
            'authentication': ['auth', 'login', 'password', 'token', 'mfa'],
            'sql': ['sql', 'database', 'query', 'injection'],
            'xss': ['xss', 'cross-site', 'script', 'javascript'],
            'performance': ['performance', 'speed', 'optimization', 'latency'],
            'architecture': ['architecture', 'design', 'system', 'microservice']
        }
        
        text_lower = text.lower()
        for category, terms in tech_terms.items():
            for term in terms:
                if term in text_lower:
                    concepts.append(category)
                    break
        
        return concepts[:5]
    
    def generate_structured_response(self, 
                                     user_input: str,
                                     content: str,
                                     response_type: str = "technical",
                                     complexity: str = "high",
                                     include_thinking: bool = True) -> str:
        """Generate a structured response with OpenAI/Mistral-style formatting"""
        
        response_parts = []
        
        # Add thinking process if requested
        if include_thinking:
            thinking = self.generate_thinking_process(user_input)
            response_parts.append(thinking)
        
        # Apply template formatting
        template = self.structured_formats.get(response_type, self.structured_formats["technical"]).get(complexity, "{content}")
        
        formatted = template.format(
            content=content,
            summary=self._generate_summary(content),
        )
        
        response_parts.append(formatted)
        
        # Add reasoning trace
        response_parts.append(self._add_reasoning_trace(user_input))
        
        return "\n\n".join(response_parts)
    
    def _generate_summary(self, content: str) -> str:
        """Generate key takeaways from content"""
        lines = [l.strip() for l in content.split('\n') if l.strip()]
        key_points = []
        
        for line in lines[:4]:
            # Look for meaningful lines
            if len(line) > 20 and not line.startswith('#'):
                key_points.append(f"• {line[:75]}...")
        
        if not key_points:
            key_points = ["• The analysis demonstrates systematic problem-solving"]
        
        return "\n".join(key_points)
    
    def _add_reasoning_trace(self, query: str) -> str:
        """Add a reasoning trace showing multi-step thinking"""
        marker = random.choice(self.reasoning_markers)
        reasoning = f"\n**Reasoning Trace:** {marker} this analysis follows a structured progression from foundational concepts through to practical applications."
        return reasoning
    
    def analyze_context(self, brain_state: Dict, user_input: str) -> Tuple[str, str, str]:
        """Analyze context for appropriate response style"""
        mood = brain_state.get("mood", "neutral")
        tokens = user_input.lower().split()
        
        # Detect sophistication level
        advanced_indicators = sum(1 for term in 
                                   ['vulnerability', 'exploit', 'cvss', 'zero-day', 'privilege', 
                                    'architecture', 'performance', 'algorithm'] 
                                   if term in user_input.lower())
        complexity = "high" if advanced_indicators >= 2 or len(tokens) > 15 else "low"
        
        # Detect request type
        request_type = "technical"
        if any(x in user_input.lower() for x in ['learn', 'explain', 'understand', 'how does', 'tutorial', 'guide']):
            request_type = "educational"
        elif any(x in user_input.lower() for x in ['plan', 'strategy', 'approach', 'best way', 'should', 'recommend']):
            request_type = "strategic"
        elif any(x in user_input.lower() for x in ['analyze', 'assess', 'evaluate', 'compare', 'impact']):
            request_type = "analytical"
        
        return mood, complexity, request_type
    
    def synthesize_response(self, brain_state: Dict, user_input: str, content: str = "") -> str:
        """Synthesize a sophisticated response with all enhancements"""
        mood, complexity, request_type = self.analyze_context(brain_state, user_input)
        
        # Use provided content or generate placeholder
        if not content:
            content = f"Analyzing {request_type} query...\n\nThis response addresses your question about {self._extract_concepts(user_input)[0] if self._extract_concepts(user_input) else 'the topic'} from multiple angles."
        
        # Generate with thinking process
        response = self.generate_structured_response(
            user_input=user_input,
            content=content,
            response_type=request_type,
            complexity=complexity,
            include_thinking=True
        )
        
        return response


# ============================================================================
# Legacy function wrappers for backwards compatibility
# ============================================================================

def analyze_context(brain_state, user_input):
    """Legacy wrapper"""
    engine = SophisticatedResponseEngine()
    mood, complexity, request_type = engine.analyze_context(brain_state, user_input)
    echo = f"Processing: {user_input[:50]}" if user_input else ""
    return mood, complexity, echo


def response_templates():
    """Legacy wrapper"""
    engine = SophisticatedResponseEngine()
    return engine.structured_formats


def synthesize_response(brain_state, user_input, content: str = ""):
    """Legacy wrapper with sophisticated response"""
    engine = SophisticatedResponseEngine()
    mood, complexity, request_type = engine.analyze_context(brain_state, user_input)
    time = datetime.datetime.now().strftime("%H:%M:%S")
    
    response = engine.synthesize_response(brain_state, user_input, content)
    prefix = f"\n[{mood.upper()} @ {time}]\n"
    
    return prefix + response


def main():
    """Test the sophisticated response engine"""
    engine = SophisticatedResponseEngine()
    
    test_brain = {
        "mood": "curious",
        "last_input": "previous test",
        "core_emotions": {"curiosity": 0.6, "frustration": 0.0, "hope": 0.2}
    }
    
    test_input = "How does SQL injection work and how can we prevent it?"
    test_content = """
SQL Injection occurs when attackers insert malicious SQL code into application inputs that interact with a database.

**Attack Mechanism:**
1. Identify user input fields connected to database queries
2. Inject SQL syntax (e.g., ' OR '1'='1) to manipulate query logic
3. Execute unintended queries to extract or modify data

**Prevention Strategies:**
- Use parameterized queries (prepared statements)
- Implement strict input validation and sanitization
- Apply principle of least privilege to database accounts
- Deploy Web Application Firewalls (WAF)
- Monitor and log suspicious database activity
    """
    
    result = engine.synthesize_response(test_brain, test_input, test_content)
    return result


if __name__ == "__main__":
    print(main())
