"""
LaTeX and problem text normalization.
Converts problem text into LLM-friendly format.
"""

import re
import logging

logger = logging.getLogger(__name__)


def normalize_problem(latex_text: str) -> str:
    """
    Normalize problem text for LLM processing.
    
    Conversions:
    - \\overline{xyz} -> "(digits xyz)"
    - Remove excessive LaTeX markup
    - Standardize math notation
    """
    if not latex_text:
        return ""
    
    text = latex_text
    
    # Convert digit notation
    text = re.sub(r'\\overline\{([a-zA-Z0-9]+)\}', r'(digits \1)', text)
    
    # Remove dollar signs (keep content)
    text = re.sub(r'\$\$(.+?)\$\$', r'\1', text, flags=re.DOTALL)
    text = re.sub(r'\$(.+?)\$', r'\1', text)
    
    # Normalize brackets
    text = re.sub(r'\\left\(', '(', text)
    text = re.sub(r'\\right\)', ')', text)
    text = re.sub(r'\\left\[', '[', text)
    text = re.sub(r'\\right\]', ']', text)
    text = re.sub(r'\\left\\{', '{', text)
    text = re.sub(r'\\right\\}', '}', text)
    
    # Normalize operators
    text = re.sub(r'\\times', '*', text)
    text = re.sub(r'\\cdot', '*', text)
    text = re.sub(r'\\div', '/', text)
    
    # Normalize fractions (iterative for nesting)
    for _ in range(5):
        new_text = re.sub(r'\\(?:d|t)?frac\{([^{}]+)\}\{([^{}]+)\}', r'(\1)/(\2)', text)
        if new_text == text:
            break
        text = new_text
    
    # Unicode to ASCII
    text = text.replace('≤', '<=').replace('≥', '>=')
    text = text.replace('≠', '!=').replace('≈', '~=')
    text = text.replace('≡', 'congruent to')
    
    # Remove remaining LaTeX commands
    text = re.sub(r'\\[a-zA-Z]+', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('"', '"').replace('"', '"')
    text = text.strip()
    
    return text


def extract_modulo(text: str) -> int:
    """Extract modulo value if present."""
    patterns = [
        r'mod(?:ulo)?\s+(\d+)',
        r'%\s*(\d+)',
        r'\(\s*mod\s+(\d+)\s*\)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return int(match.group(1))
    
    return None


def extract_constraints(text: str) -> dict:
    """Extract explicit constraints from problem text."""
    constraints = {
        'modulo': extract_modulo(text),
        'ranges': {},
        'has_constraints': bool(re.search(
            r'(where|such that|given|constraint|modulo|mod)', 
            text, re.IGNORECASE
        ))
    }
    
    # Extract variable ranges (e.g., "1 ≤ x ≤ 100")
    range_patterns = [
        r'(\d+)\s*(?:≤|<=|<)\s*([a-zA-Z])\s*(?:≤|<=|<)\s*(\d+)',
        r'([a-zA-Z])\s*(?:in|∈)\s*\[\s*(\d+)\s*,\s*(\d+)\s*\]',
    ]
    
    for pattern in range_patterns:
        for match in re.finditer(pattern, text):
            groups = match.groups()
            if len(groups) == 3:
                if groups[0].isdigit():
                    lo, var, hi = int(groups[0]), groups[1], int(groups[2])
                else:
                    var, lo, hi = groups[0], int(groups[1]), int(groups[2])
                constraints['ranges'][var] = (lo, hi)
    
    return constraints
