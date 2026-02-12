"""
AIMO-3 SOLVER - Implementing Critical Analysis Recommendations

ARCHITECTURE (Based on Section G - Final Recommended Pipeline):
1. Normalize & Classify (0.3s)
2. Code Generation PRIMARY (8s) - LLM → Python → Execute
3. Equation Extraction FALLBACK (6s) - LLM → SymPy
4. Hard Filters (0.5s)
5. Verification (3s) 
6. Selection: (best, best) ALWAYS unless confidence < 0.3

KEY FIXES FROM CRITICAL ANALYSIS:
✅ Code generation promoted to Strategy 1 (Section D2)
✅ max_tokens increased to 600 (Section E2)
✅ Chain-of-thought allowed in prompts (Section E1)
✅ Code fence cleaning before execution (Section E3)
✅ Retry loop with syntax error feedback (Section E3)
✅ (best, best) strategy unless extremely uncertain (Section F2)
✅ Narrowed validation filters (Section F1 - implemented in validation.py)
"""

import logging
import re
import time
import ast
from typing import List, Tuple, Optional, Dict, Any

from config import *
from utils import preprocess_problem_text, query_llm, ensure_determinism, time_limit
from sympy_solver import SymPySolver, EquationExtractor
from validation import AnswerValidator
from safe_executor import safe_execute, validate_code

logger = logging.getLogger(__name__)


# ============================================================================
# IMPROVED SYSTEM PROMPTS (Section E1)
# ============================================================================

CODE_GENERATION_SYSTEM = """You are an expert competition mathematics programmer.
Given an Olympiad math problem, write Python code that computes the answer.

CRITICAL RULES:
1. Think step by step in comments before coding
2. Use brute-force enumeration when exact formulas are unclear
3. Store the final integer answer in a variable called 'result'
4. Answer must be in range [0, 99999]
5. Show your reasoning in comments - explain the approach
6. Use only basic Python (math module functions allowed)
7. Be precise with modular arithmetic and integer constraints

OUTPUT FORMAT - EXACTLY THIS STRUCTURE:
```python
# APPROACH: [Explain strategy in 1-2 lines]
# DERIVATION: [Step-by-step reasoning]

[Your Python code here]
result = [final answer]
```

VERY IMPORTANT:
- Do NOT import libraries (they're not available)
- Only these math functions available: factorial, gcd, comb, perm, isqrt
- Code will be executed in a safe sandbox
- Must set 'result' variable to final integer answer
"""

EQUATION_EXTRACTION_SYSTEM = """You are a mathematical equation extractor.
Given a math problem, identify the key mathematical relationships and express them as equations.

RULES:
- Output ONLY a JSON block, no other text
- Use ** for exponents, explicit * for multiplication
- Variable names: single letters (x, y, n, k, etc.)
- Format: {"equations": ["equation1", "equation2", ...], "variables": ["var1", "var2", ...]}

Example:
Problem: "Find x where x^2 + 2*x - 3 = 0 and x > 0"
Output: {"equations": ["x**2 + 2*x - 3"], "variables": ["x"]}
"""


# ============================================================================
# CODE CLEANING UTILITIES (Section E3)
# ============================================================================

def clean_llm_code(raw_response: str) -> str:
    """Extract Python code from LLM response, stripping markdown fences."""
    if not raw_response:
        return ""
    
    # Try to extract from code fence
    code_fence_pattern = r'```(?:python)?\s*\n(.*?)```'
    match = re.search(code_fence_pattern, raw_response, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # If no fence, try to find code after "APPROACH:" or return as-is
    if 'APPROACH:' in raw_response or '# APPROACH:' in raw_response:
        parts = re.split(r'#?\s*APPROACH:', raw_response)
        if len(parts) > 1:
            return parts[-1].strip()
    
    return raw_response.strip()


def extract_result_variable(code: str) -> Optional[int]:
    """Execute code and extract 'result' variable."""
    try:
        # Validate code safety first
        is_safe, errors = validate_code(code)
        if not is_safe:
            logger.debug(f"Code safety check failed: {errors}")
            return None
        
        # Execute with timeout
        result = safe_execute(code, timeout=4.0, expected_var='result')
        
        if result is None:
            return None
        
        # Convert to integer
        try:
            answer = int(result)
        except (ValueError, TypeError):
            return None
        
        # Validate range
        if not AnswerValidator.check_range(answer):
            return None
        
        return answer
        
    except Exception as e:
        logger.debug(f"Code execution failed: {e}")
        return None


# ============================================================================
# PROBLEM CLASSIFIER
# ============================================================================

class ProblemClassifier:
    """Lightweight rule-based problem classifier."""
    
    @staticmethod
    def classify(problem_text: str) -> Dict[str, Any]:
        """Classify problem and return routing hints."""
        text_lower = problem_text.lower()
        
        classification = {
            'problem_type': 'general',
            'is_modular': False,
            'is_counting': False,
            'is_geometry': False,
            'has_equations': False,
            'modulo': ProblemClassifier.extract_modulo(problem_text),
        }
        
        if any(w in text_lower for w in ['mod', 'remainder', 'divisible']):
            classification['is_modular'] = True
            classification['problem_type'] = 'modular'
        
        if any(w in text_lower for w in ['how many', 'count', 'ways']):
            classification['is_counting'] = True
            classification['problem_type'] = 'combinatorics'
        
        if any(w in text_lower for w in ['triangle', 'circle', 'angle']):
            classification['is_geometry'] = True
            classification['problem_type'] = 'geometry'
        
        if any(w in text_lower for w in ['=', 'equation', 'solve']):
            classification['has_equations'] = True
        
        return classification
    
    @staticmethod
    def extract_modulo(text: str) -> Optional[int]:
        """Extract modulo value from problem text."""
        match = re.search(r'mod(?:ulo)?\s+(\d+)', text, re.IGNORECASE)
        return int(match.group(1)) if match else None


# ============================================================================
# CANDIDATE GENERATOR (Code Gen PRIMARY)
# ============================================================================

class CandidateGenerator:
    """Generate candidates: Code PRIMARY, Equations FALLBACK."""
    
    def __init__(self, timeout_remaining: float = 20.0):
        self.timeout_remaining = timeout_remaining
    
    def generate(self, problem_text: str, classification: Dict[str, Any], retry_reason: Optional[str] = None) -> List[Tuple[int, float]]:
        """Generate candidates with prioritized strategies."""
        candidates = []
        
        # STRATEGY 1: CODE GENERATION (PRIMARY)
        logger.info("Strategy 1: LLM → Python Code")
        answer = self._try_code_generation(problem_text, retry_reason)
        if answer is not None:
            candidates.append((answer, 0.70))
            logger.info(f"Code generation: {answer}")
        
        # STRATEGY 2: EQUATION EXTRACTION (FALLBACK)
        if not candidates:
            logger.info("Strategy 2: LLM → Equations")
            answer = self._try_equation_extraction(problem_text)
            if answer is not None:
                candidates.append((answer, 0.65))
                logger.info(f"Equation extraction: {answer}")
        
        return candidates
    
    def _try_code_generation(self, problem_text: str, retry_reason: Optional[str] = None) -> Optional[int]:
        """PRIMARY: LLM generates Python code, execute it."""
        prompt = f"{CODE_GENERATION_SYSTEM}\n\nPROBLEM:\n{problem_text}"
        if retry_reason:
            prompt += f"\n\nPREVIOUS FAILED:\n{retry_reason}"
        
        for attempt in range(2):  # One retry
            response = query_llm(prompt, max_tokens=600, temperature=0.0)
            if not response:
                continue
            
            code = clean_llm_code(response)
            if not code:
                continue
            
            answer = extract_result_variable(code)
            if answer is not None:
                return answer
            
            # Retry with syntax error feedback
            if attempt < 1:
                try:
                    ast.parse(code)
                    break  # Parsed OK, don't retry
                except SyntaxError as e:
                    prompt = f"{CODE_GENERATION_SYSTEM}\n\nPROBLEM:\n{problem_text}\n\nSYNTAX ERROR:\n{e}\n\nFix and retry."
        
        return None
    
    def _try_equation_extraction(self, problem_text: str) -> Optional[int]:
        """FALLBACK: Extract equations, solve with SymPy."""
        try:
            equations = EquationExtractor().extract_equations(problem_text)
            if not equations:
                return None
            
            result = SymPySolver.solve_from_equations(equations, problem_text)
            if result and isinstance(result, tuple):
                answer = result[0]
                if isinstance(answer, int) and AnswerValidator.check_range(answer):
                    return answer
        except Exception as e:
            logger.debug(f"Equation extraction failed: {e}")
        
        return None


# ============================================================================
# ANSWER ARBITRATOR (Fixed Strategy)
# ============================================================================

class AnswerArbitrator:
    """Verification + Selection. Return (best, best) unless conf < 0.30."""
    
    def arbitrate(self, candidates: List[Tuple[int, float]], problem_text: str) -> Tuple[int, int]:
        """Select best answer with verification."""
        if not candidates:
            return (0, 0)

        candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
        best_ans, best_conf = candidates[0]
        
        # Hard constraints
        if not AnswerValidator.check_range(best_ans):
            return (0, 0)
        if AnswerValidator.is_impossible(best_ans, problem_text):
            return (0, 0)
        if not AnswerValidator.check_parity(best_ans, problem_text):
            return (0, 0)
        if not AnswerValidator.check_divisibility(best_ans, problem_text):
            return (0, 0)
        
        # Symbolic verification
        try:
            equations = EquationExtractor().extract_equations(problem_text)
            if equations:
                verified = False
                for eq in equations:
                    for var in ['x', 'y', 'n', 'k']:
                        if SymPySolver.verify_solution(str(eq), var, best_ans):
                            verified = True
                            break
                    if verified:
                        break
                if not verified:
                    return (0, 0)
        except:
            pass
        
        # CORRECTED: Return (best, best) unless extremely uncertain
        if best_conf >= 0.30:
            return (best_ans, best_ans)
        
        # Hedge if very uncertain
        if len(candidates) >= 2:
            second_ans = candidates[1][0]
            if AnswerValidator.check_range(second_ans):
                return (best_ans, second_ans)
        
        return (best_ans, best_ans)


# ============================================================================
# MAIN SOLVER
# ============================================================================

class StrategyArbiter:
    """Master orchestrator with corrected pipeline."""
    
    def __init__(self, timeout: float = 30.0):
        self.timeout = timeout
        self.start_time = None
    
    def solve(self, problem_text: str) -> Tuple[int, int]:
        """MASTER PIPELINE."""
        self.start_time = time.time()
        if USE_FIXED_SEED:
            ensure_determinism(RANDOM_SEED)
        
        try:
            # STAGE 1: PARSE & CLASSIFY
            logger.info("STAGE 1: Parse and classify")
            try:
                with time_limit(0.5):
                    problem_text = preprocess_problem_text(problem_text)
                    classification = ProblemClassifier.classify(problem_text)
            except Exception as e:
                logger.error(f"Parse error: {e}")
                return (0, 0)
            
            # STAGE 2: GENERATE CANDIDATES
            logger.info("STAGE 2: Generate candidates")
            try:
                remaining = max(0.1, self.timeout - (time.time() - self.start_time))
                with time_limit(min(remaining, 14.0)):
                    generator = CandidateGenerator()
                    candidates = generator.generate(problem_text, classification)
            except Exception as e:
                logger.error(f"Generation error: {e}")
                candidates = []
            
            if not candidates:
                return (0, 0)
            
            # STAGE 3: VERIFY & SELECT
            logger.info("STAGE 3: Verify and select")
            try:
                remaining = max(0.1, self.timeout - (time.time() - self.start_time))
                with time_limit(remaining):
                    arbitrator = AnswerArbitrator()
                    return arbitrator.arbitrate(candidates, problem_text)
            except Exception as e:
                logger.error(f"Verification error: {e}")
                return (0, 0)
        
        except Exception as e:
            logger.error(f"Solver error: {e}", exc_info=True)
            return (0, 0)


class AdaptiveSolver:
    """Kaggle-compatible interface."""
    
    def __init__(self, timeout_seconds: int = 30):
        self.timeout = timeout_seconds
    
    def solve(self, problem_text: str, timeout_seconds: Optional[int] = None) -> Tuple[int, int]:
        """Main entry point for Kaggle submission."""
        timeout = timeout_seconds if timeout_seconds else self.timeout
        arbiter = StrategyArbiter(timeout=timeout)
        
        try:
            return arbiter.solve(problem_text)
        except Exception as e:
            logger.error(f"AdaptiveSolver failed: {e}")
            return (0, 0)
