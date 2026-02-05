"""
LaTeX and problem text parsing for AIMO-3.
"""

import re
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class ProblemParser:
    """Parse and normalize AIMO problem statements."""
    
    @staticmethod
    def extract_problem_type(text: str) -> str:
        """Classify problem into domain: algebra, number_theory, combinatorics, geometry."""
        text_lower = text.lower()
        
        # Geometry indicators
        geometry_keywords = ['triangle', 'circle', 'polygon', 'angle', 'perpendicular', 
                           'parallel', 'coordinate', 'distance', 'area', 'circumradius']
        if any(kw in text_lower for kw in geometry_keywords):
            return 'geometry'
        
        # Number theory indicators
        nt_keywords = ['prime', 'divisor', 'modulo', 'gcd', 'lcm', 'congruence', 'factorization']
        if any(kw in text_lower for kw in nt_keywords):
            return 'number_theory'
        
        # Combinatorics indicators
        comb_keywords = ['permutation', 'combination', 'arrange', 'select', 'count', 'ways',
                        'sequence', 'subset', 'partition']
        if any(kw in text_lower for kw in comb_keywords):
            return 'combinatorics'
        
        # Default to algebra
        return 'algebra'
    
    @staticmethod
    def has_constraints(text: str) -> bool:
        """Check if problem has explicit constraints (ranges, modulo, etc)."""
        return bool(re.search(r'(where|such that|given|constraint|modulo|mod|\bmod\b)', 
                             text, re.IGNORECASE))
    
    @staticmethod
    def extract_modulo(text: str) -> Optional[int]:
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
    
    @staticmethod
    def extract_ranges(text: str) -> Dict[str, tuple]:
        """Extract variable ranges from problem text returning dict var_name -> (min, max)."""
        ranges = {}
        
        # Patterns: "a ≤ x ≤ b", "a < x ≤ b", "x in [a, b]", "for x = 1, 2, ..., n"
        range_patterns = [
            r'(\d+)\s*(?:≤|<=|<)\s*([a-zA-Z])\s*(?:≤|<=|<)\s*(\d+)',
            r'([a-zA-Z])\s*(?:≤|<=|<)\s*(\d+)',
            r'([a-zA-Z])\s*(?:≥|>=|>)\s*(\d+)',
            r'([a-zA-Z])\s*(?:in|∈)\s*\[\s*(\d+)\s*,\s*(\d+)\s*\]',
            r'for\s+([a-zA-Z])\s*=\s*(\d+)\s*,\s*(\d+)\s*,\s*\.\.\.\s*,\s*(\d+)',
        ]
        
        for pattern in range_patterns:
            for match in re.finditer(pattern, text):
                if len(match.groups()) == 3:
                    # a <= x <= b or x in [a,b]
                    if match.group(1).isalpha():
                        var = match.group(1)
                        lower = int(match.group(2))
                        upper = int(match.group(3))
                    else:
                        lower = int(match.group(1))
                        var = match.group(2)
                        upper = int(match.group(3))
                    ranges[var] = (lower, upper)
                elif len(match.groups()) == 2:
                    var = match.group(1)
                    bound = int(match.group(2))
                    # Store partial bounds as (min, max) when possible
                    if '>=' in match.group(0) or '≥' in match.group(0) or '>' in match.group(0):
                        ranges[var] = (bound, ranges.get(var, (bound, bound))[1])
                    else:
                        ranges[var] = (ranges.get(var, (bound, bound))[0], bound)
                elif len(match.groups()) == 4:
                    var = match.group(1)
                    lower = int(match.group(2))
                    upper = int(match.group(4))
                    ranges[var] = (lower, upper)
        
        return ranges
    
    @staticmethod
    def clean_latex(text: str) -> str:
        """Remove or normalize LaTeX markup for LLM processing - with safe fallback."""
        if not text:
            return text
        
        try:
            # Convert common math environments to text
            text = re.sub(r'\$\$(.+?)\$\$', r'\1', text, flags=re.DOTALL)
            text = re.sub(r'\$(.+?)\$', r'\1', text)
            text = re.sub(r'\\left\(', '(', text)
            text = re.sub(r'\\right\)', ')', text)
            text = re.sub(r'\\left\[', '[', text)
            text = re.sub(r'\\right\]', ']', text)
            text = re.sub(r'\\left\\{', '{', text)
            text = re.sub(r'\\right\\}', '}', text)
            text = re.sub(r'\\\+', '+', text)
            text = re.sub(r'\\times', '*', text)
            text = re.sub(r'\\cdot', '*', text)
            text = re.sub(r'\\div', '/', text)
            # Iterative fraction replacement to handle nesting
            for _ in range(5):
                new_text = re.sub(r'\\(?:d|t)?frac\{([^{}]+)\}\{([^{}]+)\}', r'(\1)/(\2)', text)
                if new_text == text:
                    break
                text = new_text
            text = text.replace('≤', '<=').replace('≥', '>=').replace('≠', '!=').replace('≈', '~=')
            text = re.sub(r'\^', '^', text)
            text = re.sub(r'_', '_', text)
            text = re.sub(r'\\[a-zA-Z]+', '', text)
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()
            return text
        except Exception as e:
            logger.warning(f"LaTeX parsing failed, using raw text: {e}")
            return text
    
    @staticmethod
    def normalize(problem_text: str) -> str:
        """Full normalization pipeline with safe error handling."""
        try:
            if not problem_text:
                return ""
            text = ProblemParser.clean_latex(problem_text)
            text = re.sub(r'\s+', ' ', text)
            text = text.replace('"', '"').replace('"', '"')
            return text.strip()
        except Exception as e:
            logger.warning(f"Normalization failed, returning raw text: {e}")
            return problem_text
    
    @staticmethod
    def estimate_difficulty(text: str) -> float:
        """Simple heuristic to estimate problem difficulty (0-1)."""
        length_score = min(len(text) / 1000.0, 1.0)
        
        # Complex keywords
        complex_keywords = ['theorem', 'lemma', 'prove', 'induction', 'bijection',
                           'homomorphism', 'eigenvalue', 'manifold', 'derive']
        complexity = sum(1 for kw in complex_keywords if kw in text.lower()) / 10.0
        
        # Mathematical symbols density
        symbols = len(re.findall(r'[∀∃∑∏∫√∞∈∉⊂⊃∩∪≤≥≠≈]', text)) / max(len(text) / 100, 1)
        
        difficulty = (length_score * 0.3 + complexity * 0.5 + symbols * 0.2)
        return min(difficulty, 1.0)
    
    @staticmethod
    def extract_problem_subtype(text: str) -> Dict[str, Any]:
        """
        Extract semantic markers for solver routing.
        Returns dict with routing hints.
        """
        text_lower = text.lower()
        result = {
            'problem_type': ProblemParser.extract_problem_type(text),
            'keywords': [],
            'requires_modular': False,
            'requires_diophantine': False,
            'requires_counting': False,
            'requires_symbolic': False,
            'has_functional_equation': False,
            'is_optimization': False,
            'difficulty': ProblemParser.estimate_difficulty(text),
        }
        
        # Keyword-based routing
        keyword_groups = {
            'modular_arithmetic': ['modulo', 'mod', 'remainder', 'congruent', '≡'],
            'diophantine': ['integer solution', 'diophantine', 'linear', 'equation'],
            'counting': ['number of ways', 'count', 'how many', 'arrange', 'select', 'combination'],
            'optimization': ['maximum', 'minimum', 'find the largest', 'find the smallest', 'optimize'],
            'functional': ['functional equation', 'f(', 'f(x)', 'f(f('],
            'symbolic': ['prove', 'show', 'derive', 'equation', 'polynomial'],
        }
        
        for category, keywords in keyword_groups.items():
            for kw in keywords:
                if kw in text_lower:
                    result['keywords'].append(kw)
                    if category == 'modular_arithmetic':
                        result['requires_modular'] = True
                    elif category == 'diophantine':
                        result['requires_diophantine'] = True
                    elif category == 'counting':
                        result['requires_counting'] = True
                    elif category == 'optimization':
                        result['is_optimization'] = True
                    elif category == 'functional':
                        result['has_functional_equation'] = True
                    elif category == 'symbolic':
                        result['requires_symbolic'] = True
                    break
        
        # Extract modulo if present
        mod_val = ProblemParser.extract_modulo(text)
        if mod_val:
            result['modulo'] = mod_val
            result['requires_modular'] = True
        
        # Extract variable ranges/constraints
        result['ranges'] = ProblemParser.extract_ranges(text)
        result['has_constraints'] = ProblemParser.has_constraints(text)
        
        return result
    
    @staticmethod
    def get_solver_strategy(problem_analysis: Dict[str, Any]) -> str:
        """
        Recommend solver strategy based on problem analysis.
        Returns: 'sympy_first' | 'llm_first' | 'hybrid' | 'tree_search'
        """
        ptype = problem_analysis['problem_type']
        
        # SymPy dominates these
        sympy_domains = {
            'requires_modular': True,
            'requires_diophantine': True,
            'requires_symbolic': True,
        }
        
        if any(problem_analysis.get(k) for k in sympy_domains):
            return 'sympy_first'
        
        # LLM excels at these
        if problem_analysis['requires_counting'] or problem_analysis['is_optimization']:
            return 'llm_first'
        
        # Functional equations need both
        if problem_analysis['has_functional_equation']:
            return 'hybrid'
        
        # Geometry without diagrams: LLM only
        if ptype == 'geometry':
            return 'llm_first'
        
        # Default hybrid
        return 'hybrid'
