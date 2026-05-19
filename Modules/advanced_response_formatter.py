"""
Advanced Response Formatter
Applies Mistral/OpenAI-style formatting and structuring to responses
Features:
- Multi-step reasoning display
- Structured markdown formatting
- Progressive information disclosure
- Context integration
- Intelligent emphasis and organization
"""

import re
from typing import Dict, List, Optional, Tuple
from enum import Enum


class ResponseStyle(Enum):
    TECHNICAL = "technical"
    EDUCATIONAL = "educational"
    STRATEGIC = "strategic"
    ANALYTICAL = "analytical"


class AdvancedResponseFormatter:
    """Format responses like Mistral/OpenAI with sophisticated structure"""
    
    def __init__(self):
        self.markdown_enhancements = {
            'emphasis': {
                'critical': '**CRITICAL:**',
                'important': '**Important:**',
                'note': '**Note:**',
                'warning': '⚠️ **Warning:**',
                'tip': '💡 **Tip:**'
            },
            'structure': {
                'problem': '### Problem',
                'solution': '### Solution',
                'implications': '### Implications',
                'considerations': '### Key Considerations'
            }
        }
    
    def format_with_thinking(self, 
                            user_input: str, 
                            response_content: str,
                            thinking_process: Optional[str] = None,
                            style: ResponseStyle = ResponseStyle.TECHNICAL) -> str:
        """Format response with optional thinking process visible"""
        
        formatted_parts = []
        
        # Add thinking process if provided
        if thinking_process:
            formatted_parts.append(self._format_thinking_section(thinking_process))
        
        # Add main response with structure
        formatted_parts.append(self._apply_structure(response_content, style))
        
        # Add conclusion or summary
        formatted_parts.append(self._add_conclusion(response_content))
        
        return "\n\n".join(formatted_parts)
    
    def _format_thinking_section(self, thinking: str) -> str:
        """Format the thinking/reasoning section"""
        lines = [
            "<details>",
            "<summary><strong>Thinking Process</strong></summary>",
            "",
            thinking.strip(),
            "",
            "</details>"
        ]
        return "\n".join(lines)
    
    def _apply_structure(self, content: str, style: ResponseStyle) -> str:
        """Apply intelligent structure based on response style"""
        
        if style == ResponseStyle.TECHNICAL:
            return self._structure_technical(content)
        elif style == ResponseStyle.EDUCATIONAL:
            return self._structure_educational(content)
        elif style == ResponseStyle.STRATEGIC:
            return self._structure_strategic(content)
        elif style == ResponseStyle.ANALYTICAL:
            return self._structure_analytical(content)
        
        return content
    
    def _structure_technical(self, content: str) -> str:
        """Structure for technical responses"""
        lines = []
        lines.append("## Technical Analysis\n")
        
        # Split into logical sections
        sections = self._identify_sections(content)
        for section in sections:
            lines.append(section)
        
        return "\n\n".join(lines)
    
    def _structure_educational(self, content: str) -> str:
        """Structure for educational responses"""
        lines = []
        lines.append("## Learning Path\n")
        
        # Build progressive learning structure
        lines.append("### 1. Foundational Concepts")
        lines.append("This section covers the basic principles...\n")
        
        lines.append("### 2. Detailed Explanation")
        lines.append(content[:len(content)//2])
        lines.append("\n")
        
        lines.append("### 3. Practical Application")
        lines.append(content[len(content)//2:])
        lines.append("\n")
        
        lines.append("### 4. Further Resources")
        lines.append("- Deep dive documentation\n- Advanced case studies\n- Community resources")
        
        return "\n".join(lines)
    
    def _structure_strategic(self, content: str) -> str:
        """Structure for strategic responses"""
        lines = []
        lines.append("## Strategic Approach\n")
        lines.append("### Overview")
        lines.append(content[:150].strip() + "...\n")
        
        lines.append("### Phase 1: Planning")
        lines.append("- Define objectives\n- Assess current state\n- Identify gaps\n")
        
        lines.append("### Phase 2: Implementation")
        lines.append(content)
        lines.append("\n")
        
        lines.append("### Phase 3: Optimization")
        lines.append("- Monitor performance\n- Adjust strategy\n- Document lessons learned")
        
        return "\n".join(lines)
    
    def _structure_analytical(self, content: str) -> str:
        """Structure for analytical responses"""
        lines = []
        lines.append("## Analytical Breakdown\n")
        
        lines.append("### Context Analysis")
        lines.append("Examining the situation from multiple dimensions...\n")
        
        lines.append("### Key Findings")
        findings = self._extract_key_points(content)
        for finding in findings:
            lines.append(f"- {finding}")
        lines.append("")
        
        lines.append("### Interpretation")
        lines.append(content)
        lines.append("")
        
        lines.append("### Recommendations")
        lines.append("Based on the analysis above:")
        lines.append("1. Primary recommendation")
        lines.append("2. Alternative approaches")
        lines.append("3. Risk mitigation strategies")
        
        return "\n".join(lines)
    
    def _identify_sections(self, content: str) -> List[str]:
        """Identify and extract logical sections from content"""
        # Simple section detection
        sections = []
        lines = content.split('\n')
        current_section = []
        
        for line in lines:
            if line.startswith('**') and line.endswith('**'):
                if current_section:
                    sections.append('\n'.join(current_section))
                current_section = [line]
            else:
                current_section.append(line)
        
        if current_section:
            sections.append('\n'.join(current_section))
        
        return sections
    
    def _extract_key_points(self, content: str) -> List[str]:
        """Extract key points from content"""
        points = []
        sentences = content.split('.')
        
        # Take first 3-4 meaningful sentences
        for sentence in sentences[:4]:
            sentence = sentence.strip()
            if len(sentence) > 10 and not sentence.startswith('#'):
                points.append(sentence.strip()[:100])
        
        return points if points else ["See detailed analysis below"]
    
    def _add_conclusion(self, content: str) -> str:
        """Add intelligent conclusion section"""
        return """
---

## Summary

This analysis demonstrates systematic problem-solving through structured reasoning and technical depth. The key takeaway is understanding both the underlying mechanisms and practical applications.

**Next Steps:**
- Review the provided examples
- Test in your specific context
- Adjust approach based on results
"""
    
    def enhance_with_codeblocks(self, content: str, language: str = "python") -> str:
        """Enhance response with properly formatted code blocks"""
        # Find code-like patterns and wrap them
        enhanced = re.sub(
            r'(?<!`)(class |def |import |from ).+(?!`)',
            lambda m: f"```{language}\n{m.group(0)}\n```",
            content
        )
        return enhanced
    
    def add_visual_hierarchy(self, content: str) -> str:
        """Add visual hierarchy markers"""
        # Convert to better formatted hierarchy
        lines = content.split('\n')
        formatted = []
        
        for line in lines:
            if line.startswith('##'):
                formatted.append(f"\n{line}\n{'='*50}\n")
            elif line.startswith('###'):
                formatted.append(f"\n{line}\n{'-'*40}\n")
            else:
                formatted.append(line)
        
        return '\n'.join(formatted)
    
    @staticmethod
    def create_comparison_table(items: Dict[str, List[str]]) -> str:
        """Create formatted comparison table"""
        headers = list(items.keys())
        rows = items[headers[0]]
        
        # Build markdown table
        table = f"| {' | '.join(headers)} |\n"
        table += f"|{' | '.join(['---'] * len(headers))}|\n"
        
        for i, row in enumerate(rows):
            values = [items[h][i] if i < len(items[h]) else '' for h in headers]
            table += f"| {' | '.join(values)} |\n"
        
        return table


def format_security_response(query: str, analysis: str, expertise_level: str = "intermediate") -> str:
    """Format security-focused responses"""
    formatter = AdvancedResponseFormatter()
    
    # Determine style based on query
    style = ResponseStyle.TECHNICAL
    if any(x in query.lower() for x in ['learn', 'explain', 'how']):
        style = ResponseStyle.EDUCATIONAL
    elif any(x in query.lower() for x in ['plan', 'strategy', 'approach']):
        style = ResponseStyle.STRATEGIC
    
    thinking = f"""
    Analyzing security query at {expertise_level} level.
    Detecting query type: {style.value}
    Building response with context awareness and structured progression.
    """
    
    return formatter.format_with_thinking(
        user_input=query,
        response_content=analysis,
        thinking_process=thinking,
        style=style
    )


if __name__ == "__main__":
    formatter = AdvancedResponseFormatter()
    
    test_response = """
**SQL Injection Overview**

SQL Injection is a code injection attack where attackers insert malicious SQL statements into application inputs. This allows them to:
- Extract sensitive data from the database
- Modify or delete database records
- Potentially execute commands on the database server

**How It Works:**
1. Attacker finds an input field connected to a database query
2. Instead of normal input, attacker enters SQL syntax
3. The database executes the modified query
4. Attacker gains unauthorized access to data
"""
    
    formatted = formatter.format_with_thinking(
        user_input="explain SQL injection",
        response_content=test_response,
        thinking_process="Analyzing security query...",
        style=ResponseStyle.EDUCATIONAL
    )
    
    print(formatted)
