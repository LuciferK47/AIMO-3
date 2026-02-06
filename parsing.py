"""
LaTeX and problem text parsing for AIMO-3.
Compatibility layer: All functionality has been consolidated into ProblemClassifier in solver.py
"""

import re
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class ProblemParser:
    """
    LEGACY COMPATIBILITY LAYER
    All parsing functionality is now in ProblemClassifier (solver.py).
    This class delegates to maintain backward compatibility.
    """
    
    # Lazy import to avoid circular dependency
    _classifier = None
    
    @classmethod
    def _get_classifier(cls):
        """Lazy load ProblemClassifier."""
        if cls._classifier is None:
            try:
                from solver import ProblemClassifier
                cls._classifier = ProblemClassifier
            except ImportError:
                logger.warning("Could not import ProblemClassifier from solver")
                return None
        return cls._classifier
    
    @staticmethod
    def extract_problem_type(text: str) -> str:
        """LEGACY: Use ProblemClassifier.classify() instead."""
        # Map to new classification system
        result = ProblemClassifier.classify(text)
        ptype = result['problem_type']
        
        # Return legacy mapping
        if ptype == 'geometry':
            return 'geometry'
        elif ptype in ['modular', 'diophantine']:
            return 'number_theory'
        elif ptype == 'combinatorics':
            return 'combinatorics'
        else:
            return 'algebra'
    
    @staticmethod
    def has_constraints(text: str) -> bool:
        """LEGACY: Delegates to ProblemClassifier.has_constraints()."""
        classifier = ProblemParser._get_classifier()
        if classifier:
            return classifier.has_constraints(text)
        return False
    
    @staticmethod
    def extract_modulo(text: str) -> Optional[int]:
        """LEGACY: Delegates to ProblemClassifier.extract_modulo()."""
        classifier = ProblemParser._get_classifier()
        if classifier:
            return classifier.extract_modulo(text)
        return None
    
    @staticmethod
    def extract_ranges(text: str) -> Dict[str, tuple]:
        """LEGACY: Delegates to ProblemClassifier.extract_ranges()."""
        classifier = ProblemParser._get_classifier()
        if classifier:
            return classifier.extract_ranges(text)
        return {}
    
    @staticmethod
    def clean_latex(text: str) -> str:
        """LEGACY: Delegates to ProblemClassifier.clean_latex()."""
        classifier = ProblemParser._get_classifier()
        if classifier:
            return classifier.clean_latex(text)
        return text
    
    @staticmethod
    def normalize(problem_text: str) -> str:
        """Normalize problem text using clean_latex and whitespace standardization."""
        if not problem_text:
            return ""
        text = ProblemParser.clean_latex(problem_text)
        text = re.sub(r'\s+', ' ', text)
        text = text.replace('"', '"').replace('"', '"')
        return text.strip()
    
    @staticmethod
    def estimate_difficulty(text: str) -> float:
        """LEGACY: Returns ProblemClassifier.classify() difficulty_estimate."""
        classifier = ProblemParser._get_classifier()
        if classifier:
            return classifier.classify(text)['difficulty_estimate']
        return 0.5
    
    @staticmethod
    def extract_problem_subtype(text: str) -> Dict[str, Any]:
        """LEGACY: Returns routing hints from ProblemClassifier.classify()."""
        classifier = ProblemParser._get_classifier()
        if not classifier:
            return {}
        
        classification = classifier.classify(text)
        return {
            'problem_type': ProblemParser.extract_problem_type(text),
            'keywords': classification.get('keywords', []),
            'requires_modular': classification.get('is_modular', False),
            'requires_diophantine': 'diophantine' in classification.get('keywords', []),
            'requires_counting': classification.get('is_counting', False),
            'requires_symbolic': classification.get('has_equations', False),
            'has_functional_equation': False,
            'is_optimization': 'optimization' in classification.get('keywords', []),
            'difficulty': classification.get('difficulty_estimate', 0.5),
            'modulo': classification.get('modulo'),
            'ranges': classification.get('ranges', {}),
            'has_constraints': classification.get('has_constraints', False),
        }
    
    @staticmethod
    def get_solver_strategy(problem_analysis: Dict[str, Any]) -> str:
        """LEGACY: Strategy recommendation based on problem analysis."""
        ptype = problem_analysis.get('problem_type', 'general')
        
        # SymPy dominates these
        if problem_analysis.get('requires_modular') or problem_analysis.get('requires_diophantine'):
            return 'sympy_first'
        
        # LLM excels at these
        if problem_analysis.get('requires_counting') or problem_analysis.get('is_optimization'):
            return 'llm_first'
        
        # Functional equations need both
        if problem_analysis.get('has_functional_equation'):
            return 'hybrid'
        
        # Geometry: LLM only
        if ptype == 'geometry':
            return 'llm_first'
        
        return 'hybrid'


# Consolidation Note:
# This module is now a compatibility layer. All extraction and classification logic
# has been consolidated into ProblemClassifier in solver.py to reduce duplication.
# Original methods are preserved but delegate to the unified implementation.
# 
# Benefits of consolidation:
# - Single source of truth for problem analysis
# - Reduced code duplication (~200 lines saved)
# - Unified interface: ProblemClassifier.classify() returns all extraction data
# - Better maintainability and consistency
