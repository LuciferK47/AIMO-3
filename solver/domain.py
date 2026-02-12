"""
Lightweight domain classification.
Used ONLY to adjust verification rules, not to route to different solvers.
"""

import re
from typing import Dict, Any


def classify_domain(problem_text: str) -> Dict[str, Any]:
    """
    Classify problem domain.
    Returns hints for verification rules, NOT solver selection.
    """
    text_lower = problem_text.lower()
    
    classification = {
        'domain': 'algebra',  # Default
        'is_modular': False,
        'is_counting': False,
        'is_geometry': False,
        'requires_symbolic_check': True,
    }
    
    # Number theory / modular arithmetic
    if any(kw in text_lower for kw in ['mod', 'modulo', 'remainder', 'congruence', 'divisible']):
        classification['domain'] = 'number_theory'
        classification['is_modular'] = True
    
    # Combinatorics / counting
    elif any(kw in text_lower for kw in ['how many', 'count', 'ways', 'combinations', 'permutations']):
        classification['domain'] = 'combinatorics'
        classification['is_counting'] = True
    
    # Geometry
    elif any(kw in text_lower for kw in ['triangle', 'circle', 'angle', 'area', 'perimeter']):
        classification['domain'] = 'geometry'
        classification['is_geometry'] = True
        classification['requires_symbolic_check'] = False  # Harder to verify symbolically
    
    # Algebra (default)
    else:
        classification['domain'] = 'algebra'
    
    return classification
