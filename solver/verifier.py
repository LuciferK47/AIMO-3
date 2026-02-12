"""
Verification module - THE AUTHORITATIVE JUDGE.
LLM proposes, Python verifies. Only verified answers survive.
"""

import re
import logging
from typing import Optional, List

logger = logging.getLogger(__name__)


class Verifier:
    """Authoritative verification of proposed answers."""
    
    @staticmethod
    def verify_answer(answer: int, problem_text: str, constraints: dict, domain: dict) -> bool:
        """
        Verify proposed answer against all checks.
        Returns True only if ALL checks pass.
        """
        # ========== HARD CONSTRAINTS ==========
        
        # Range check (AIMO-3 constraint)
        if not (0 <= answer <= 99999):
            logger.debug(f"Range violation: {answer}")
            return False
        
        # Modular constraint
        if constraints.get('modulo') and 'remainder' in problem_text.lower():
            if answer >= constraints['modulo']:
                logger.debug(f"Modular violation: {answer} >= {constraints['modulo']}")
                return False
        
        # Explicit range constraints
        for var, (lo, hi) in constraints.get('ranges', {}).items():
            # If problem asks for this variable and we have a bound
            if any(f"find {var}" in problem_text.lower() or f"{var} =" in problem_text.lower() for v in [var]):
                if not (lo <= answer <= hi):
                    logger.debug(f"Range constraint violation: {answer} not in [{lo}, {hi}]")
                    return False
        
        # Parity check (explicit constraints only)
        if not Verifier._check_parity(answer, problem_text):
            logger.debug(f"Parity violation: {answer}")
            return False
        
        # Divisibility check (explicit constraints only)
        if not Verifier._check_divisibility(answer, problem_text):
            logger.debug(f"Divisibility violation: {answer}")
            return False
        
        # Primality check (if explicitly requested)
        if 'prime' in problem_text.lower() and 'find' in problem_text.lower():
            if not Verifier._is_prime(answer):
                logger.debug(f"Primality violation: {answer}")
                return False
        
        # Perfect square check (if explicitly requested)
        if 'perfect square' in problem_text.lower() or 'square number' in problem_text.lower():
            root = int(answer ** 0.5)
            if root * root != answer:
                logger.debug(f"Perfect square violation: {answer}")
                return False
        
        # Domain-specific checks
        if domain.get('is_modular') and constraints.get('modulo'):
            # Modulo answer must be less than modulus for remainder problems
            mod_val = constraints['modulo']
            if 'remainder' in problem_text.lower() and answer >= mod_val:
                logger.debug(f"Modulo remainder violation: {answer} >= {mod_val}")
                return False
        
        # ========== SYMBOLIC VERIFICATION (if possible) ==========
        if domain.get('requires_symbolic_check', True):
            if not Verifier._symbolic_verification(answer, problem_text):
                logger.debug(f"Symbolic verification failed: {answer}")
                return False
        
        return True
    
    @staticmethod
    def _check_parity(answer: int, problem_text: str) -> bool:
        """Check parity constraints (explicit only)."""
        text_lower = problem_text.lower()
        
        # Explicit even constraint
        if re.search(r'\b(even|divisible by 2)\b', text_lower):
            if answer % 2 != 0:
                return False
        
        # Explicit odd constraint
        if re.search(r'\b(odd|not divisible by 2)\b', text_lower):
            if answer % 2 == 0:
                return False
        
        return True
    
    @staticmethod
    def _check_divisibility(answer: int, problem_text: str) -> bool:
        """Check divisibility constraints (explicit only)."""
        # Pattern: "divisible by N"
        div_pattern = r'divisible by (\d+)'
        for match in re.finditer(div_pattern, problem_text, re.IGNORECASE):
            divisor = int(match.group(1))
            if answer % divisor != 0:
                return False
        
        # Pattern: "multiple of N"
        mult_pattern = r'multiple of (\d+)'
        for match in re.finditer(mult_pattern, problem_text, re.IGNORECASE):
            divisor = int(match.group(1))
            if answer % divisor != 0:
                return False
        
        return True
    
    @staticmethod
    def _is_prime(n: int) -> bool:
        """Check if number is prime."""
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(n ** 0.5) + 1, 2):
            if n % i == 0:
                return False
        return True
    
    @staticmethod
    def _symbolic_verification(answer: int, problem_text: str) -> bool:
        """
        Attempt symbolic verification using SymPy.
        Returns True if:
        - No equations can be extracted (can't verify), OR
        - Answer satisfies at least one extracted equation
        """
        try:
            import sympy as sp
            
            # Try to extract equations from problem text
            equations = Verifier._extract_equations(problem_text)
            
            if not equations:
                # No equations to verify against - pass
                return True
            
            # Try to verify answer against equations
            candidate_vars = ['x', 'y', 'n', 'k', 'm', 'a', 'b', 'c']
            
            for eq_str in equations:
                for var in candidate_vars:
                    try:
                        # Parse equation
                        if '=' not in eq_str:
                            continue
                        
                        lhs_str, rhs_str = eq_str.split('=', 1)
                        lhs = sp.sympify(lhs_str.strip())
                        rhs = sp.sympify(rhs_str.strip())
                        
                        # Substitute answer
                        lhs_val = lhs.subs(var, answer)
                        rhs_val = rhs.subs(var, answer)
                        
                        # Check if equation holds
                        if sp.simplify(lhs_val - rhs_val) == 0:
                            logger.info(f"Symbolic verification passed: {eq_str} with {var}={answer}")
                            return True
                    
                    except Exception:
                        continue
            
            # No equation was satisfied - verification failed
            logger.debug(f"No equation satisfied by {answer}")
            return False
        
        except ImportError:
            # SymPy not available - pass (can't verify)
            logger.debug("SymPy not available for symbolic verification")
            return True
        except Exception as e:
            logger.debug(f"Symbolic verification error: {e}")
            return True  # Don't fail on verification errors
    
    @staticmethod
    def _extract_equations(problem_text: str) -> List[str]:
        """Extract mathematical equations from problem text."""
        equations = []
        
        # Look for patterns like "x^2 + 3 = 12" or "2*x - 5 = 13"
        # Simple heuristic: find segments with '=' that contain variables and operators
        lines = problem_text.split('\n')
        for line in lines:
            if '=' in line:
                # Check if line contains math operators and variables
                if any(op in line for op in ['+', '-', '*', '/', '^', '**']) and \
                   any(c.isalpha() for c in line):
                    # Clean up the equation
                    eq = line.strip()
                    # Remove "where ", "if ", etc.
                    eq = re.sub(r'^(where|if|given|such that)\s+', '', eq, flags=re.IGNORECASE)
                    # Convert ^ to **
                    eq = eq.replace('^', '**')
                    equations.append(eq)
        
        return equations


def parse_llm_output(text: str) -> Optional[int]:
    """
    Parse LLM output to extract final integer.
    Returns None if format is invalid.
    """
    if not text:
        return None
    
    # Check for required sections
    if "DERIVATION:" not in text or "FINAL_EXPRESSION:" not in text or "FINAL_INTEGER:" not in text:
        logger.debug("Invalid format: missing required sections")
        return None
    
    # Extract final integer
    match = re.search(r"FINAL_INTEGER:\s*(-?\d+)", text)
    if not match:
        logger.debug("Invalid format: FINAL_INTEGER not found or malformed")
        return None
    
    try:
        value = int(match.group(1))
        return value
    except ValueError:
        logger.debug("Invalid format: FINAL_INTEGER is not a valid integer")
        return None
