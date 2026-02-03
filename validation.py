"""
Unified validation module for AIMO-3.
Merged from: verifier.py + verification.py

Provides deterministic answer validation, constraint checking,
and hallucination detection.
"""

import re
import logging
from typing import Optional, Tuple, List, Any, Dict, Callable
from collections import Counter

try:
    import sympy as sp
except ImportError:
    sp = None

logger = logging.getLogger(__name__)


# ============================================================================
# BASIC VALIDATION
# ============================================================================

def is_valid_answer(answer_value: Any) -> bool:
    """
    Strict validation for AIMO-3 answers.
    
    Rules:
    - Must be a non-negative integer
    - Must be in range [0, 99999] inclusive
    
    Args:
        answer_value: Value to validate (any type)
        
    Returns:
        True if valid, False otherwise
    """
    try:
        val = int(answer_value)
        return 0 <= val <= 99999
    except (ValueError, TypeError):
        return False


# ============================================================================
# ANSWER VALIDATOR CLASS
# ============================================================================

class AnswerValidator:
    """Validate answers against problem properties and constraints."""
    
    @staticmethod
    def check_range(answer: int, min_val: int = 0, max_val: int = 99999) -> bool:
        """Check if answer is in valid competition range [0, 99999]."""
        return min_val <= answer <= max_val
    
    @staticmethod
    def check_implicit_bounds(answer: int, problem_text: str) -> bool:
        """
        Extract and check implicit bounds from problem text.
        
        Examples:
        - "less than N" -> answer < N
        - "remainder when divided by M" -> answer < M
        - "positive integer" -> answer > 0
        """
        problem_lower = problem_text.lower()
        
        # Check for explicit bounds
        bounds_patterns = [
            (r'less than (\d+)', lambda v, n: v < int(n)),
            (r'greater than (\d+)', lambda v, n: v > int(n)),
            (r'at most (\d+)', lambda v, n: v <= int(n)),
            (r'at least (\d+)', lambda v, n: v >= int(n)),
        ]
        
        for pattern, check_fn in bounds_patterns:
            match = re.search(pattern, problem_lower)
            if match:
                try:
                    return check_fn(answer, match.group(1))
                except Exception:
                    pass
        
        # Check for remainder bounds
        if 'remainder' in problem_lower and 'divided by' in problem_lower:
            match = re.search(r'divided by (\d+)', problem_lower)
            if match:
                divisor = int(match.group(1))
                if answer >= divisor:
                    return False
        
        return True
    
    @staticmethod
    def check_parity(answer: int, problem_text: str) -> bool:
        """Check even/odd constraint if specified."""
        problem_lower = problem_text.lower()
        
        if 'even' in problem_lower and answer % 2 != 0:
            return False
        if 'odd' in problem_lower and answer % 2 != 1:
            return False
        
        return True
    
    @staticmethod
    def check_divisibility(answer: int, problem_text: str) -> bool:
        """Check divisibility constraints mentioned in problem."""
        problem_lower = problem_text.lower()
        
        patterns = [
            r'divisible by (\d+)',
            r'multiple of (\d+)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, problem_lower)
            for match in matches:
                try:
                    divisor = int(match)
                    if answer % divisor != 0:
                        return False
                except Exception:
                    pass
        
        return True

    @staticmethod
    def check_modular_constraint(answer: int, problem_text: str) -> bool:
        """Check remainder constraints: if modulo specified, answer must be < modulo."""
        problem_lower = problem_text.lower()
        match = re.search(r'(?:mod|modulo|divided by)\s+(\d+)', problem_lower)
        if match:
            try:
                modulus = int(match.group(1))
                if 'remainder' in problem_lower and answer >= modulus:
                    return False
            except Exception:
                pass
        return True


# ============================================================================
# DETERMINISTIC VERIFIER
# ============================================================================

class DeterministicVerifier:
    """Deterministic verification using problem constraints."""
    
    @staticmethod
    def verify_equation(equation_str: str, answer: int, variable: str = 'x') -> bool:
        """
        Verify answer satisfies an equation using SymPy (safe).
        
        Args:
            equation_str: Equation string (e.g., "x^2 + 3*x - 10 = 0")
            answer: Value to verify
            variable: Variable name (default 'x')
            
        Returns:
            True if answer satisfies equation, False otherwise
        """
        if sp is None:
            # Fallback: basic check without SymPy
            try:
                expr = equation_str.replace(variable, str(answer))
                # Only allow alphanumeric and basic operators to prevent injection
                if not re.match(r'^[\d+\-*/(). ]+$', expr):
                    return False
                result = eval(expr)  # Safe: validated characters only
                return abs(result) < 1e-9
            except Exception:
                return False
        
        try:
            # Use SymPy for safe symbolic evaluation
            var = sp.Symbol(variable)
            
            # Handle "=" in equation
            if '=' in equation_str:
                lhs_str, rhs_str = equation_str.split('=')
                lhs = sp.sympify(lhs_str.replace(variable, f"({answer})"))
                rhs = sp.sympify(rhs_str.replace(variable, f"({answer})"))
                return sp.simplify(lhs - rhs) == 0
            else:
                # Assume equation is set to 0
                expr = sp.sympify(equation_str.replace(variable, f"({answer})"))
                return sp.simplify(expr) == 0
        except Exception:
            return False
    
    @staticmethod
    def verify_constraint(constraint: str, answer: int) -> bool:
        """
        Verify answer satisfies a constraint using safe comparison only.
        
        Args:
            constraint: Constraint string (e.g., "answer < 100")
            answer: Value to verify
            
        Returns:
            True if constraint satisfied, False otherwise
        """
        if not constraint or not isinstance(answer, int):
            return False
        
        try:
            # Only allow safe comparison operators
            constraint = constraint.strip()
            
            # Support basic comparisons: ==, !=, <, >, <=, >=
            if '==' in constraint:
                parts = constraint.split('==')
                if len(parts) != 2:
                    return False
                target = int(parts[1].replace('answer', '').strip())
                return answer == target
            elif '!=' in constraint:
                parts = constraint.split('!=')
                if len(parts) != 2:
                    return False
                target = int(parts[1].replace('answer', '').strip())
                return answer != target
            elif '<=' in constraint:
                parts = constraint.split('<=')
                if len(parts) != 2:
                    return False
                target = int(parts[1].replace('answer', '').strip())
                return answer <= target
            elif '>=' in constraint:
                parts = constraint.split('>=')
                if len(parts) != 2:
                    return False
                target = int(parts[1].replace('answer', '').strip())
                return answer >= target
            elif '<' in constraint:
                parts = constraint.split('<')
                if len(parts) != 2:
                    return False
                target = int(parts[1].replace('answer', '').strip())
                return answer < target
            elif '>' in constraint:
                parts = constraint.split('>')
                if len(parts) != 2:
                    return False
                target = int(parts[1].replace('answer', '').strip())
                return answer > target
            else:
                return False
        except (ValueError, IndexError, AttributeError):
            return False


# ============================================================================
# ANTI-HALLUCINATION CHECKER
# ============================================================================

class AntiHallucinationChecker:
    """Detect hallucinated answers and suspicious patterns."""
    
    # Pattern-based checks
    SUSPICIOUS_PATTERNS = {
        'round_number': [100, 1000, 10000],
        'power_of_two': [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
        'factorial': [2, 6, 24, 120, 720, 5040, 40320, 362880],
        'fibonacci': [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987],
        'prime': [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47],
    }
    
    @staticmethod
    def check_magnitude_plausibility(answer: int, problem_text: str) -> float:
        """
        Check if answer magnitude seems plausible.
        
        Returns:
            Suspicion score [0, 1] where 0 = not suspicious, 1 = very suspicious
        """
        suspicion = 0.0
        
        # Extremely large answers are suspicious
        if answer > 50000:
            suspicion += 0.2
        
        # Extremely small answers for "count" problems
        if answer < 10 and 'how many' in problem_text.lower():
            suspicion += 0.1
        
        # Very round numbers are often hallucinations
        if answer in [100, 1000, 10000]:
            suspicion += 0.15
        
        return min(1.0, suspicion)
    
    @staticmethod
    def check_common_hallucinations(answer: int, problem_text: str) -> float:
        """
        Check for common AI hallucination patterns.
        
        Returns:
            Suspicion score [0, 1]
        """
        suspicion = 0.0
        
        # Check if answer is a suspicious "special" number
        for category, numbers in AntiHallucinationChecker.SUSPICIOUS_PATTERNS.items():
            if answer in numbers:
                # Context-dependent suspicion
                if category == 'round_number':
                    suspicion += 0.1
                elif category in ['factorial', 'fibonacci'] and 'sequence' not in problem_text.lower():
                    suspicion += 0.2
        
        return min(1.0, suspicion)
    
    @staticmethod
    def check_digit_patterns(answer: int) -> float:
        """
        Check for suspicious digit patterns.
        
        Returns:
            Suspicion score [0, 1]
        """
        suspicion = 0.0
        answer_str = str(answer)
        
        # Repeating digits (111, 222, etc.)
        if len(set(answer_str)) == 1:
            suspicion += 0.2
        
        # Sequential digits (123, 456, etc.)
        if all(int(answer_str[i+1]) - int(answer_str[i]) == 1 for i in range(len(answer_str)-1)):
            suspicion += 0.15
        
        return min(1.0, suspicion)
    
    @staticmethod
    def run_full_check(answer: int, problem_text: str) -> Dict[str, float]:
        """
        Run all hallucination checks.
        
        Returns:
            Dictionary with individual and total suspicion scores
        """
        checks = {
            'magnitude': AntiHallucinationChecker.check_magnitude_plausibility(answer, problem_text),
            'common_hallucination': AntiHallucinationChecker.check_common_hallucinations(answer, problem_text),
            'digit_pattern': AntiHallucinationChecker.check_digit_patterns(answer),
        }
        
        checks['total_suspicion'] = min(1.0, sum(checks.values()) / len(checks))
        
        return checks


# ============================================================================
# ENSEMBLE VOTING & AGGREGATION
# ============================================================================

def filter_duplicate_answers(answers: List[int]) -> List[int]:
    """Remove duplicates while preserving order."""
    seen = set()
    result = []
    for ans in answers:
        if ans not in seen:
            seen.add(ans)
            result.append(ans)
    return result


def rank_answers_by_confidence(answers: List[int], 
                               confidence_scores: Dict[int, float]) -> List[Tuple[int, float]]:
    """
    Rank answers by confidence scores.
    
    Args:
        answers: List of candidate answers
        confidence_scores: Dict mapping answer -> confidence
        
    Returns:
        Sorted list of (answer, confidence) tuples
    """
    ranked = [(ans, confidence_scores.get(ans, 0.5)) for ans in answers]
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked


def compute_agreement_score(answers: List[int]) -> Dict[int, float]:
    """
    Compute agreement score for each answer.
    
    Args:
        answers: List of candidate answers
        
    Returns:
        Dict mapping answer -> agreement score (0.0 - 1.0)
    """
    if not answers:
        return {}
    
    counter = Counter(answers)
    total = len(answers)
    
    return {ans: count / total for ans, count in counter.items()}


def weighted_agreement(cot_answers: List[int], python_answers: List[int],
                       cot_weight: float = 1.0, python_weight: float = 3.0) -> Dict[int, float]:
    """
    Compute weighted agreement between different answer sources.
    
    Python answers (code execution) get higher weight due to determinism.
    """
    scores = {}
    
    # Weight CoT answers
    for ans in cot_answers:
        scores[ans] = scores.get(ans, 0) + cot_weight
    
    # Weight Python answers (3x higher)
    for ans in python_answers:
        scores[ans] = scores.get(ans, 0) + python_weight
    
    # Normalize
    total = sum(scores.values()) if scores else 1.0
    return {ans: score / total for ans, score in scores.items()} if scores else {}


def select_diverse_pair_with_confidence(answers_dict: Dict[int, float],
                                        confidence_dict: Dict[int, float]) -> Tuple[int, int]:
    """
    Select two diverse answers for submission, preferring high-confidence answers.
    
    Uses penalized accuracy: prioritizes well-supported answer, then diverse backup.
    """
    if not answers_dict:
        return (0, 0)
    
    # Sort by combined score: agreement + confidence
    scored = []
    for ans, agreement in answers_dict.items():
        confidence = confidence_dict.get(ans, 0.5)
        combined = agreement * 0.6 + confidence * 0.4
        scored.append((ans, combined))
    
    scored.sort(key=lambda x: x[1], reverse=True)
    
    if len(scored) >= 2:
        return (scored[0][0], scored[1][0])
    elif len(scored) == 1:
        return (scored[0][0], scored[0][0])
    else:
        return (0, 0)


def choose_final_pair(confidence_scores: Dict[int, float]) -> Tuple[int, int]:
    """
    Choose final answer pair using penalized-accuracy strategy.
    
    Strategy:
    1. Pick highest-confidence answer as primary
    2. Pick next-best or random alternative as backup
    """
    if not confidence_scores:
        return (0, 0)
    
    sorted_answers = sorted(confidence_scores.items(), key=lambda x: x[1], reverse=True)
    
    if len(sorted_answers) >= 2:
        return (sorted_answers[0][0], sorted_answers[1][0])
    elif len(sorted_answers) == 1:
        return (sorted_answers[0][0], sorted_answers[0][0])
    else:
        return (0, 0)

# ============================================================================
# ADVANCED VERIFICATION (NEW)
# ============================================================================

class AdvancedVerification:
    """Advanced verification techniques for higher accuracy."""
    
    @staticmethod
    def verify_with_modular_arithmetic(answer: int, problem_text: str) -> float:
        """
        Check if answer is consistent with modular constraints.
        
        Returns confidence score [0, 1].
        """
        problem_lower = problem_text.lower()
        
        # Extract modulo value
        mod_match = re.search(r'mod(?:ulo)?\s+(\d+)', problem_lower)
        if not mod_match:
            return 0.5  # Neutral
        
        try:
            modulus = int(mod_match.group(1))
            # If problem asks for remainder, answer should be < modulus
            if 'remainder' in problem_lower and answer >= modulus:
                return 0.0  # Fails
            if answer >= 0 and answer < modulus:
                return 0.9  # Passes
        except Exception:
            pass
        
        return 0.5
    
    @staticmethod
    def verify_counting_sanity(answer: int, problem_text: str) -> float:
        """
        Sanity check for counting problems.
        
        Returns confidence score [0, 1].
        """
        problem_lower = problem_text.lower()
        
        # If it's a counting problem
        if not any(kw in problem_lower for kw in ['count', 'number of', 'how many', 'ways']):
            return 0.5
        
        # Sanity checks
        if answer < 0:
            return 0.0
        if answer == 0:
            return 0.2  # Suspicious
        if answer > 100000:
            return 0.3  # Very large, somewhat suspicious
        
        return 0.8
    
    @staticmethod
    def verify_bounds_consistency(answer: int, problem_analysis: Dict[str, Any]) -> float:
        """
        Check answer against extracted bounds.
        
        Returns confidence score [0, 1].
        """
        confidence = 0.5
        
        # Check ranges from problem_analysis
        if 'ranges' in problem_analysis:
            ranges = problem_analysis['ranges']
            for var, (min_val, max_val) in ranges.items():
                if min_val <= answer <= max_val:
                    confidence = 0.85
                else:
                    return 0.1  # Violates constraint
        
        return confidence
    
    @staticmethod
    def plug_back_verification(answer: int, equations: List[str]) -> float:
        """
        Verify answer by substituting back into equations.
        
        Returns confidence score [0, 1].
        """
        if not equations:
            return 0.5
        
        verified_count = 0
        for eq in equations[:3]:  # Check up to 3 equations
            try:
                # Use SymPy for safe verification
                if sp is not None:
                    # Equation verification using SymPy
                    if '=' not in eq:
                        continue
                    lhs_str, rhs_str = eq.split('=', 1)
                    # Safe substitution using SymPy
                    lhs_val = sp.sympify(lhs_str.replace('x', f"({answer})"))
                    rhs_val = sp.sympify(rhs_str.replace('x', f"({answer})"))
                    if sp.simplify(lhs_val - rhs_val) == 0:
                        verified_count += 1
                else:
                    # Fallback: basic string replacement with character validation
                    test_eq = eq.replace('x', str(answer))
                    test_eq = test_eq.replace('=', '==')
                    
                    # Only evaluate if equation looks safe
                    if re.match(r'^[\d+\-*/(). ]==[0-9.]+$', test_eq):
                        result = eval(test_eq)  # Safe: validated regex
                        if result:
                            verified_count += 1
            except Exception:

                pass
        
        if len(equations) == 0:
            return 0.5
        
        return min(0.95, verified_count / len(equations[:3]) * 0.95 + 0.05)
    
    @staticmethod
    def consistency_check(answer: int, alternative_answers: List[int]) -> float:
        """
        Check answer consistency with alternative derivations.
        
        Returns confidence score [0, 1].
        """
        if not alternative_answers:
            return 0.5
        
        match_count = sum(1 for alt in alternative_answers if alt == answer)
        
        # Voting-based confidence
        return min(1.0, 0.5 + 0.5 * (match_count / len(alternative_answers)))


class SelfVerificationLoop:
    """Self-correction via verification feedback."""
    
    @staticmethod
    def score_candidate_answer(answer: int, problem_text: str, 
                               problem_analysis: Dict[str, Any],
                               equations: List[str]) -> float:
        """
        Comprehensive confidence score for an answer.
        
        Combines multiple verification strategies.
        """
        scores = []
        weights = []
        
        # Range check
        if AnswerValidator.check_range(answer):
            scores.append(0.9)
            weights.append(0.1)
        else:
            scores.append(0.0)
            weights.append(0.1)
        
        # Implicit bounds
        if AnswerValidator.check_implicit_bounds(answer, problem_text):
            scores.append(0.85)
        else:
            scores.append(0.2)
        weights.append(0.15)
        
        # Parity check
        if AnswerValidator.check_parity(answer, problem_text):
            scores.append(0.8)
        else:
            scores.append(0.3)
        weights.append(0.1)
        
        # Divisibility check
        if AnswerValidator.check_divisibility(answer, problem_text):
            scores.append(0.85)
        else:
            scores.append(0.3)
        weights.append(0.1)

        # Modular constraint check
        if AnswerValidator.check_modular_constraint(answer, problem_text):
            scores.append(0.9)
        else:
            scores.append(0.2)
        weights.append(0.08)
        
        # Modular arithmetic verification
        mod_score = AdvancedVerification.verify_with_modular_arithmetic(answer, problem_text)
        scores.append(mod_score)
        weights.append(0.15)
        
        # Counting sanity
        count_score = AdvancedVerification.verify_counting_sanity(answer, problem_text)
        scores.append(count_score)
        weights.append(0.15)
        
        # Bounds consistency
        bounds_score = AdvancedVerification.verify_bounds_consistency(answer, problem_analysis)
        scores.append(bounds_score)
        weights.append(0.1)
        
        # Plug-back verification
        plug_score = AdvancedVerification.plug_back_verification(answer, equations)
        scores.append(plug_score)
        weights.append(0.14)
        
        # Weighted average
        total_weight = sum(weights)
        if total_weight == 0:
            return 0.5
        
        confidence = sum(s * w for s, w in zip(scores, weights)) / total_weight
        return min(1.0, max(0.0, confidence))
    
    @staticmethod
    def rank_candidates(candidates: List[int], problem_text: str,
                       problem_analysis: Dict[str, Any],
                       equations: List[str]) -> List[Tuple[int, float]]:
        """
        Rank candidate answers by confidence.
        
        Returns list of (answer, confidence) sorted by confidence descending.
        """
        ranked = []
        for ans in set(candidates):  # Deduplicate
            confidence = SelfVerificationLoop.score_candidate_answer(
                ans, problem_text, problem_analysis, equations
            )
            ranked.append((ans, confidence))
        
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked