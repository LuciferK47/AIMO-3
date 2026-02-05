"""
AIMO-3 MASTER SOLVER - Strategy Arbiter Architecture

MASTER PROMPT IMPLEMENTATION

Target Pipeline:
LaTeX problem
  ↓
Semantic parsing + problem classification
  ↓
Parallel candidate generation (LLM × k)
  ↓
Equation extraction + symbolic solving
  ↓
Verification + consistency checks
  ↓
Answer arbitration (vote / confidence)
  ↓
Final integer pair

Key Principles:
- Deterministic execution
- Symbolic solving is PRIMARY (not fallback)
- Multi-candidate generation for robustness
- Strict verification gates
- Bounded time and depth
"""

import logging
import re
import time
from typing import List, Tuple, Optional, Dict, Any, Set

from config import *
from parsing import ProblemParser
from utils import preprocess_problem_text, query_llm, ensure_determinism, time_limit
from sympy_solver import SymPySolver, EquationExtractor, DiophantineSolver
from validation import AnswerValidator, SelfVerificationLoop
from utils import preprocess_problem_text, query_llm, query_llm_json, ensure_determinism

logger = logging.getLogger(__name__)


class ProblemClassifier:
    """Lightweight rule-based problem classifier (no ML)."""
    
    @staticmethod
    def classify(problem_text: str) -> Dict[str, Any]:
        """
        Classify problem and return routing hints.
        
        Returns:
            {
                'problem_type': str,  # 'modular', 'combinatorics', 'diophantine', 'geometry', 'optimization', 'general'
                'keywords': List[str],
                'allows_symbolic': bool,
                'has_equations': bool,
                'is_modular': bool,
                'is_counting': bool,
                'is_geometry': bool,
                'difficulty_estimate': float,  # 0-1
            }
        """
        text_lower = problem_text.lower()
        
        # Keyword detection
        keywords = []
        classification = {
            'problem_type': 'general',
            'keywords': [],
            'allows_symbolic': True,
            'has_equations': False,
            'is_modular': False,
            'is_counting': False,
            'is_geometry': False,
            'difficulty_estimate': 0.5,
        }
        
        # MULTI-LABEL CLASSIFICATION (not mutually exclusive)
        # Modular Arithmetic
        if any(w in text_lower for w in ['mod', 'remainder', 'divisible', 'congruence', 'modular']):
            classification['is_modular'] = True
            classification['allows_symbolic'] = True
            keywords.extend(['modular', 'arithmetic'])
            if not classification['problem_type']:
                classification['problem_type'] = 'modular'
        
        # Combinatorics
        if any(w in text_lower for w in ['how many', 'count', 'combinations', 'permutations', 'ways', 'arrangements', 'sequences']):
            classification['is_counting'] = True
            classification['allows_symbolic'] = True
            keywords.extend(['counting', 'combinatorics'])
            if not classification['problem_type']:
                classification['problem_type'] = 'combinatorics'
        
        # Diophantine / Number Theory
        if any(w in text_lower for w in ['integer solutions', 'find all integers', 'diophantine', 'number of pairs', 'pairs (', 'positive integers']):
            classification['allows_symbolic'] = True
            keywords.extend(['number_theory', 'diophantine'])
            if not classification['problem_type']:
                classification['problem_type'] = 'diophantine'
        
        # Geometry
        if any(w in text_lower for w in ['triangle', 'circle', 'angle', 'area', 'perimeter', 'distance', 'radius', 'diameter']):
            classification['is_geometry'] = True
            # Geometry without diagrams is harder for SymPy
            if classification['is_geometry'] and not classification['is_modular']:
                classification['allows_symbolic'] = False
            keywords.extend(['geometry'])
            if not classification['problem_type']:
                classification['problem_type'] = 'geometry'
        
        # Optimization
        if any(w in text_lower for w in ['maximum', 'minimum', 'maximize', 'minimize', 'least', 'greatest', 'optimal']):
            classification['allows_symbolic'] = True
            keywords.extend(['optimization'])
            if not classification['problem_type']:
                classification['problem_type'] = 'optimization'
        
        # EQUATIONS / ALGEBRA
        if any(w in text_lower for w in ['equation', 'solve', '=', 'satisfy', 'satisfy']):
            classification['has_equations'] = True
        
        classification['keywords'] = keywords
        
        # Difficulty estimate based on keywords
        hard_keywords = ['diophantine', 'complex', 'involving', 'system']
        easy_keywords = ['simple', 'basic', 'find']
        
        difficulty = 0.5
        for kw in hard_keywords:
            if kw in text_lower:
                difficulty = min(1.0, difficulty + 0.15)
        for kw in easy_keywords:
            if kw in text_lower:
                difficulty = max(0.0, difficulty - 0.1)
        
        classification['difficulty_estimate'] = difficulty
        
        return classification


class CandidateGenerator:
    """Generate multiple candidate answers using different strategies."""
    
    def __init__(self, timeout_remaining: float = 20.0):
        self.timeout_remaining = timeout_remaining
        self.candidates = []
    
    def generate(self, problem_text: str, classification: Dict[str, Any]) -> List[Tuple[int, int]]:
        """
        Generate 3-5 candidate answers.
        
        CRITICAL INVERSION:
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        LLM ROLE: Translator (formalize, enumerate, derive)
        SYMPY ROLE: Solver (compute, verify, decide)
        
        Pipeline:
        1. SYMBOLIC-FIRST: Direct SymPy (modular, Diophantine)
        2. LLM TRANSLATION → SYMPY SOLVE (PRIMARY)
        3. LLM CASE ENUMERATION → SYMPY COMPUTE
        4. LLM FORMALIZATION → SYMPY EVALUATE
        
        LLM NEVER directly solves - only translates.
        """
        candidates = []
        
        # STRATEGY 1: SYMBOLIC-FIRST (no LLM)
        if classification['allows_symbolic']:
            try:
                sym_candidates = self._try_symbolic(problem_text, classification)
                candidates.extend(sym_candidates)
                unique_sym = {c[0] for c in sym_candidates if isinstance(c, tuple) and c[0] != 0}
                if len(unique_sym) == 1 and len(sym_candidates) >= 1:
                    logger.info("SymPy solved confidently - skipping LLM")
                    seen = set()
                    unique_candidates = []
                    for c in candidates:
                        if c not in seen and c != (0, 0):
                            seen.add(c)
                            unique_candidates.append(c)
                    return unique_candidates[:MAX_CANDIDATES_PER_PROBLEM]
            except Exception as e:
                logger.debug(f"Direct symbolic failed: {e}")
        
        # STRATEGY 2: LLM TRANSLATION → SYMPY SOLVE (PRIMARY PATH)
        try:
            eq_candidates = self._try_equation_extraction(problem_text, classification)
            candidates.extend(eq_candidates)
        except Exception as e:
            logger.debug(f"Translation→solve failed: {e}")
        
        # STRATEGY 3: LLM CASE ENUMERATION → SYMPY COMPUTE
        try:
            case_candidates = self._try_case_enumeration(problem_text, classification)
            candidates.extend(case_candidates)
        except Exception as e:
            logger.debug(f"Case enumeration failed: {e}")
        
        # STRATEGY 4: LLM FORMALIZATION → SYMPY EVALUATE
        try:
            formal_candidates = self._try_formalization(problem_text, classification)
            candidates.extend(formal_candidates)
        except Exception as e:
            logger.debug(f"Formalization failed: {e}")
        
        # STRATEGY 5: LLM PYTHON GENERATION → SANDBOXED EXECUTION (NEW)
        try:
            python_candidates = self._try_python_execution(problem_text, classification)
            candidates.extend(python_candidates)
        except Exception as e:
            logger.debug(f"Python execution failed: {e}")
        
        # Remove duplicates
        seen = set()
        unique_candidates = []
        for c in candidates:
            if c not in seen and c != (0, 0):
                seen.add(c)
                unique_candidates.append(c)
        
        return unique_candidates[:MAX_CANDIDATES_PER_PROBLEM]
    
    def _try_python_execution(self, problem_text: str, classification: Dict[str, Any]) -> List[Tuple[int, int]]:
        """
        LLM PYTHON GENERATION → SANDBOXED EXECUTION
        
        LLM: "Generate Python code to solve this problem"
        Executor: Safely execute code and extract integer result
        
        Useful for: enumeration, brute-force, iterative solutions
        """
        candidates = []
        
        try:
            from safe_executor import safe_execute, extract_integer_from_code
            
            python_prompt = f"""
You are a Python code generator for mathematical problem solving.

Task: Write Python code to solve this problem. Output the answer in a variable called 'result'.

Rules:
- Use only basic Python (loops, arithmetic, lists)
- Do NOT use imports or eval()
- Assign final answer to 'result'
- Code should complete quickly (< 5 seconds)

Problem:
{problem_text}

Python code:
"""
            
            code = query_llm(python_prompt, max_tokens=300, temperature=0.2)
            if code:
                # Try to extract answer from generated code
                answer = extract_integer_from_code(code, timeout=3.0)
                if answer is not None and AnswerValidator.check_range(answer):
                    candidates.append((answer, 0.6))  # Moderate confidence for executed code
        
        except ImportError:
            logger.debug("safe_executor module not available")
        except Exception as e:
            logger.debug(f"Python execution generation failed: {e}")
        
        return candidates
    
    def _try_symbolic(self, problem_text: str, classification: Dict[str, Any]) -> List[Tuple[int, int]]:
        """Try pure symbolic solving."""
        candidates = []
        
        try:
            # For diophantine equations
            problem_type = classification.get('problem_type', '')
            if problem_type == 'diophantine' or 'diophantine' in problem_text.lower():
                from sympy_solver import DiophantineSolver
                import math
                # Try to extract bounds from problem text (e.g., "1 ≤ x ≤ 100" or "1 <= x <= 10")
                bounds: Dict[str, Tuple[int, int]] = {}
                text_lower = problem_text.lower()

                # Pattern: 1 <= x <= 10
                range_pattern = r'(\d+)\s*(?:<=|≤)\s*([a-zA-Z])\s*(?:<=|≤)\s*(\d+)'
                for match in re.finditer(range_pattern, problem_text):
                    lo, var, hi = match.groups()
                    bounds[var] = (int(lo), int(hi))

                # Pattern: x >= 1, x <= 10
                ge_pattern = r'([a-zA-Z])\s*(?:>=|≥)\s*(\d+)'
                le_pattern = r'([a-zA-Z])\s*(?:<=|≤)\s*(\d+)'
                for match in re.finditer(ge_pattern, problem_text):
                    var, lo = match.groups()
                    lo = int(lo)
                    if var in bounds:
                        bounds[var] = (max(bounds[var][0], lo), bounds[var][1])
                    else:
                        bounds[var] = (lo, 99999)
                for match in re.finditer(le_pattern, problem_text):
                    var, hi = match.groups()
                    hi = int(hi)
                    if var in bounds:
                        bounds[var] = (bounds[var][0], min(bounds[var][1], hi))
                    else:
                        bounds[var] = (0, hi)

                # Pattern: x between 1 and 10
                between_pattern = r'([a-zA-Z])\s*between\s*(\d+)\s*and\s*(\d+)'
                for match in re.finditer(between_pattern, text_lower):
                    var, lo, hi = match.groups()
                    bounds[var] = (int(lo), int(hi))
                
                # Try linear diophantine pattern: ax + by = c
                linear_pattern = r'(\d+)\s*([a-zA-Z])\s*\+\s*(\d+)\s*([a-zA-Z])\s*=\s*(\d+)'
                linear_match = re.search(linear_pattern, problem_text)
                if linear_match:
                    a, var1, b, var2, c = linear_match.groups()
                    a_int, b_int, c_int = int(a), int(b), int(c)
                    solution = DiophantineSolver.linear_diophantine(a_int, b_int, c_int)
                    if solution:
                        x0, y0 = solution
                        g = math.gcd(a_int, b_int)
                        dx = b_int // g
                        dy = -a_int // g

                        # Implicit bounds for positivity
                        if 'positive integer' in text_lower or 'positive integers' in text_lower:
                            bounds.setdefault(var1, (1, 99999))
                            bounds.setdefault(var2, (1, 99999))
                        elif 'nonnegative integer' in text_lower or 'non-negative integer' in text_lower:
                            bounds.setdefault(var1, (0, 99999))
                            bounds.setdefault(var2, (0, 99999))

                        # Default bounds to keep solutions in valid answer range
                        bounds.setdefault(var1, (0, 99999))
                        bounds.setdefault(var2, (0, 99999))

                        def update_t_range(x_start: int, step: int, lo: int, hi: int, t_low: int, t_high: int) -> Tuple[int, int]:
                            if step == 0:
                                if lo <= x_start <= hi:
                                    return t_low, t_high
                                return 1, 0
                            t1 = (lo - x_start) / step
                            t2 = (hi - x_start) / step
                            if step > 0:
                                lo_t = math.ceil(t1)
                                hi_t = math.floor(t2)
                            else:
                                lo_t = math.ceil(t2)
                                hi_t = math.floor(t1)
                            return max(t_low, lo_t), min(t_high, hi_t)

                        t_low, t_high = -10**9, 10**9
                        if var1 in bounds:
                            lo, hi = bounds[var1]
                            t_low, t_high = update_t_range(x0, dx, lo, hi, t_low, t_high)
                        if var2 in bounds:
                            lo, hi = bounds[var2]
                            t_low, t_high = update_t_range(y0, dy, lo, hi, t_low, t_high)

                        if t_low <= t_high:
                            num_solutions = max(0, t_high - t_low + 1)
                            wants_count = any(kw in text_lower for kw in ['how many', 'number of solutions', 'count'])
                            wants_sum = any(kw in text_lower for kw in ['sum of', 'x+y', 'x + y'])
                            wants_product = 'product' in text_lower

                            if wants_count:
                                if 0 <= num_solutions <= 99999:
                                    candidates.append((int(num_solutions), 0.75))
                            else:
                                # Pick a representative solution within bounds
                                t_pick = t_low
                                x_val = x0 + dx * t_pick
                                y_val = y0 + dy * t_pick
                                if bounds.get(var1, (0, 99999))[0] <= x_val <= bounds.get(var1, (0, 99999))[1] and \
                                   bounds.get(var2, (0, 99999))[0] <= y_val <= bounds.get(var2, (0, 99999))[1]:
                                    if wants_product:
                                        candidates.append((x_val * y_val, 0.75))
                                    elif wants_sum:
                                        candidates.append((x_val + y_val, 0.75))
                                    else:
                                        # Default to x-value if problem doesn't specify
                                        candidates.append((x_val, 0.7))
            
            # For modular problems
            if classification['is_modular']:
                result = SymPySolver.solve_modular_equation(problem_text)
                if result and result != (0, 0):
                    candidates.append(result)
            
            # For combinatorics
            if classification['is_counting']:
                result = SymPySolver.solve_combinatorics(problem_text)
                if result and result != (0, 0):
                    candidates.append(result)
            
            # Try general equation extraction for all problems
            from sympy_solver import EquationExtractor
            extractor = EquationExtractor()
            test_equations = extractor.extract_equations(problem_text)
            if test_equations:
                result = SymPySolver.solve_from_equations(test_equations, problem_text)
                if result and result != (0, 0):
                    candidates.append(result)
        
        except Exception as e:
            logger.debug(f"Symbolic solving error: {e}")
        
        return candidates
    
    def _try_equation_extraction(self, problem_text: str, classification: Dict[str, Any]) -> List[Tuple[int, int]]:
        """
        LLM TRANSLATION → SYMPY SOLVE (PRIMARY PIPELINE) with temperature sweep retry
        
        Critical separation:
        - LLM: Translate natural language → formal equations (ONLY)
        - SymPy: Solve equations → numeric answer
        
        This is NOT a fallback. This is the MAIN path for 30-50% of problems.
        Uses temperature sweep [0.1, 0.3, 0.5] for robustness against malformed equations.
        """
        candidates = []
        
        try:
            # Temperature sweep: try deterministic first, then increase creativity if failed
            temperatures = [0.1, 0.3, 0.5]
            
            for temp in temperatures:
                translation_prompt = f"""
You are a mathematical translator, NOT a solver.

Task: Convert this problem into formal mathematical equations ONLY.

Rules:
- Output ONLY JSON: {{"equations": ["x + y = 10", "2*x - 3*y = 5"]}}
- Use standard variables: x, y, z, n, k, a, b
- Do NOT solve
- Do NOT compute answers
- Do NOT explain

Problem:
{problem_text}
"""
                response = query_llm_json(translation_prompt, max_tokens=250, temperature=temp)
                equations = []
                if response and isinstance(response.get("equations"), list):
                    equations = [str(e).strip() for e in response["equations"] if str(e).strip()]

                if not equations:
                    equations_text = query_llm(translation_prompt, max_tokens=250, temperature=temp)
                    if not equations_text:
                        continue  # Try next temperature
                    extractor = EquationExtractor()
                    equations = extractor.extract_equations(equations_text)
                    if not equations:
                        continue  # Try next temperature
                
                # Validate equations are well-formed
                valid_eqs = []
                for eq in equations:
                    eq_str = str(eq)
                    has_var = any(c.isalpha() for c in eq_str)
                    has_operator = any(op in eq_str for op in ['+', '-', '*', '/', '=', '^'])
                    if has_var and has_operator:
                        valid_eqs.append(eq)
                
                if not valid_eqs:
                    continue  # Try next temperature
                
                # STEP 3: SYMPY SOLVES (PRIMARY SOLVER)
                result = SymPySolver.solve_from_equations(valid_eqs, problem_text)
                if result and result != (0, 0):
                    answer, base_conf = result
                    # Adjust confidence based on temperature (lower temp = higher confidence)
                    adjusted_conf = base_conf * (1.0 - temp * 0.2)
                    candidates.append((answer, adjusted_conf))
                    logger.info(f"SymPy solved from LLM-translated equations (temp={temp}): {result}")
                    break  # Success, no need to try higher temperatures
        
        except Exception as e:
            logger.debug(f"Translation→solve pipeline error: {e}")
        
        return candidates
    
    def _try_case_enumeration(self, problem_text: str, classification: Dict[str, Any]) -> List[Tuple[int, int]]:
        """
        LLM CASE ENUMERATION → SYMPY COMPUTE
        
        LLM: "List all cases/values to check"
        SymPy: Compute/validate each case (enumerate enumeration problem constraints)
        
        Useful for: counting problems, modular arithmetic, optimization
        
        NOTE: Case enumeration returns CANDIDATE ANSWERS (not intermediate cases).
        These candidates are verified in the arbitration stage via constraint checks.
        """
        candidates = []
        
        try:
            enumeration_prompt = f"""
You are a case enumerator, NOT a solver.

Task: List ALL cases or values that need to be checked.

Rules:
- Output ONLY JSON: {{"cases": [val1, val2, val3, ...]}}
- Do NOT compute final answer
- Just enumerate the search space
- Each case should be a candidate answer to evaluate

Problem:
{problem_text}
"""
            response = query_llm_json(enumeration_prompt, max_tokens=200, temperature=0.2)
            if response and isinstance(response.get("cases"), list):
                for case in response["cases"][:10]:
                    # Case enumeration returns CANDIDATE answers (not intermediate values)
                    # These are verified via AnswerArbitrator.arbitrate() → SelfVerificationLoop
                    if isinstance(case, int) and AnswerValidator.check_range(case):
                        # Case is a candidate; confidence is 0 (needs verification)
                        candidates.append((case, 0))
                    elif isinstance(case, str):
                        # Try to parse string as integer
                        try:
                            case_int = int(case)
                            if AnswerValidator.check_range(case_int):
                                candidates.append((case_int, 0))
                        except (ValueError, TypeError):
                            pass
        
        except Exception as e:
            logger.debug(f"Case enumeration error: {e}")
        
        return candidates
    
    def _try_formalization(self, problem_text: str, classification: Dict[str, Any]) -> List[Tuple[int, int]]:
        """
        LLM FORMALIZATION → SYMPY EVALUATE
        
        LLM: "Convert to a single mathematical expression"
        SymPy: Evaluate the expression
        
        Useful for: combinatorial formulas, closed-form expressions
        """
        candidates = []
        
        try:
            formalization_prompt = f"""
You are a mathematical formalizer, NOT a calculator.

Task: Convert this problem into a SINGLE mathematical expression.

Rules:
- Output ONLY JSON: {{"expression": "2**10 % 7"}}
- Use Python syntax: **, //, %, factorial(n)
- Do NOT compute the answer
- Do NOT explain

Problem:
{problem_text}
"""
            response = query_llm_json(formalization_prompt, max_tokens=150, temperature=0.1)
            expression = None
            if response and isinstance(response.get("expression"), str):
                expression = response["expression"].strip()
            if not expression:
                raw = query_llm(formalization_prompt, max_tokens=150, temperature=0.1)
                expression = raw.strip() if raw else None

            if expression:
                try:
                    result = SymPySolver.evaluate_expression(expression)
                    if result is not None:
                        answer = int(result)
                        if 0 <= answer <= 99999:
                            candidates.append((answer, 0))
                            logger.info(f"SymPy evaluated: {answer}")
                except Exception:
                    pass
        
        except Exception as e:
            logger.debug(f"Formalization error: {e}")
        
        return candidates


class AnswerArbitrator:
    """Verify candidates and select best answer using arbitration."""
    
    def __init__(self):
        self.verifier = SelfVerificationLoop()
    
    def arbitrate(self, candidates: List[Tuple[int, int]], problem_text: str) -> Tuple[int, int]:
        """
        Verify all candidates and return best pair using multi-stage verification pipeline.
        
        Stages:
        1. HARD CONSTRAINT FILTERING - Range, modular, parity checks
        2. SYMBOLIC RE-VERIFICATION - Plug back into equations
        3. CROSS-VALIDATION - Agreement checking across methods
        4. WEIGHTED VOTE - Confidence-based scoring
        """
        if not candidates:
            return (0, 0)

        # Extract primary answers
        primary_answers = [c[0] for c in candidates if isinstance(c, tuple) and len(c) >= 1]
        primary_answers = [a for a in primary_answers if isinstance(a, int)]

        if not primary_answers:
            return (0, 0)

        # ========== STAGE 1: HARD CONSTRAINT FILTERING ==========
        filtered_answers = []
        for ans in set(primary_answers):
            # Range check (0-99999)
            if not AnswerValidator.check_range(ans):
                logger.debug(f"Answer {ans} failed range check")
                continue
            # Modular bound check
            if not AnswerValidator.check_modular_constraint(ans, problem_text):
                logger.debug(f"Answer {ans} failed modular constraint")
                # Don't reject - modular constraints can be soft
            # Parity check (only if explicitly required)
            if not AnswerValidator.check_parity(ans, problem_text):
                logger.debug(f"Answer {ans} failed parity check")
                # Don't reject - parity might be red herring
            filtered_answers.append(ans)
        
        if not filtered_answers:
            # All candidates filtered - return best from original set anyway
            logger.warning("All candidates filtered by hard constraints, falling back to original set")
            filtered_answers = [a for a in set(primary_answers) if AnswerValidator.check_range(a)]
        
        if not filtered_answers:
            logger.warning("No candidates in valid range [0, 99999]")
            return (0, 0)

        # ========== BUILD PROBLEM ANALYSIS & EQUATIONS ==========
        # Build problem analysis + equations for verification (cached)
        from cache import get_intermediate_cache
        intermediate_cache = get_intermediate_cache()
        problem_analysis = intermediate_cache.get_problem_analysis(problem_text)
        if not problem_analysis:
            problem_analysis = ProblemParser.extract_problem_subtype(problem_text)
            intermediate_cache.put_problem_analysis(problem_text, problem_analysis)

        equations = intermediate_cache.get_equations(problem_text)
        if equations is None:
            try:
                equations = EquationExtractor().extract_equations(problem_text)
            except Exception:
                equations = []
            intermediate_cache.put_equations(problem_text, equations)

        # ========== STAGE 2 & 3 & 4: VERIFICATION & SCORING ==========
        scored_candidates = []
        for ans in filtered_answers:
            confidence = SelfVerificationLoop.score_candidate_answer(
                ans, problem_text, problem_analysis, equations
            )
            scored_candidates.append((ans, confidence))
            logger.debug(f"Answer {ans}: confidence={confidence:.3f}")

        if not scored_candidates:
            return (0, 0)

        # Sort by confidence and pick top two unique answers
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        # Arbitration Strategy: Select pair based on confidence
        # If high confidence on best answer with clear margin: return (best, best) for hedging
        # Otherwise: return (best, second_best) for diversity
        
        ordered_answers = [ans for ans, _ in scored_candidates]
        ordered_scores = [score for _, score in scored_candidates]
        
        if len(ordered_answers) == 0:
            return (0, 0)
        
        if len(ordered_answers) == 1:
            # Only one candidate - return it twice
            return (ordered_answers[0], ordered_answers[0])
        
        # Two or more candidates
        best_ans = ordered_answers[0]
        best_score = ordered_scores[0]
        second_best_ans = ordered_answers[1] if len(ordered_answers) > 1 else best_ans
        second_score = ordered_scores[1] if len(ordered_answers) > 1 else best_score
        
        # Confidence-based hedging strategy
        # If very high confidence and large margin: bet on same answer twice
        # Otherwise: diversity bet
        confidence_threshold = 0.75  # High confidence threshold
        score_margin_threshold = 0.15  # Margin between best and second-best
        
        if best_score >= confidence_threshold and (best_score - second_score) >= score_margin_threshold:
            # High confidence with clear leader → return same answer twice
            logger.info(f"High confidence arbitration: ({best_ans}, {best_ans}) [score={best_score:.3f}]")
            return (best_ans, best_ans)
        else:
            # Moderate confidence or close race → diversity strategy
            logger.info(f"Diversity arbitration: ({best_ans}, {second_best_ans}) [scores={best_score:.3f}, {second_score:.3f}]")
            return (best_ans, second_best_ans)
 

class StrategyArbiter:
    """Master orchestrator implementing the target pipeline."""
    
    def __init__(self, timeout: float = 30.0):
        self.timeout = timeout
        self.start_time = None
        self.time_per_stage = {
            'parse': 0.5,
            'classify': 0.2,
            'generate': 15.0,
            'verify': 5.0,
            'arbitrate': 1.0,
        }
    
    def solve(self, problem_text: str) -> Tuple[int, int]:
        """
        MASTER PIPELINE with per-stage timeout enforcement:
        
        1. Parse & Classify (0.5s budget)
        2. Multi-Candidate Generation (15.0s budget)
        3. Verification & Scoring (5.0s budget)
        4. Answer Arbitration (1.0s budget)
        """
        self.start_time = time.time()
        if USE_FIXED_SEED:
            ensure_determinism(RANDOM_SEED)
        
        try:
            # STAGE 1: PARSE & CLASSIFY (0.5s max)
            logger.info("STAGE 1: Parsing and classifying...")
            try:
                with time_limit(self.time_per_stage['parse']):
                    problem_text = preprocess_problem_text(problem_text)
                    classification = ProblemClassifier.classify(problem_text)
                    logger.info(f"Classification: {classification['problem_type']}")
            except TimeoutError:
                logger.warning("Parse/classify timeout, returning (0, 0)")
                return (0, 0)
            except Exception as e:
                logger.error(f"Parse/classify error: {e}")
                return (0, 0)
            
            # STAGE 2: MULTI-CANDIDATE GENERATION (15.0s max)
            logger.info("STAGE 2: Generating candidates...")
            try:
                remaining_time = self.timeout - (time.time() - self.start_time)
                gen_timeout = min(remaining_time, self.time_per_stage['generate'])
                
                with time_limit(gen_timeout):
                    generator = CandidateGenerator(timeout_remaining=gen_timeout)
                    candidates = generator.generate(problem_text, classification)
                    logger.info(f"Generated {len(candidates)} candidates: {candidates}")
            except TimeoutError:
                logger.warning("Candidate generation timeout")
                candidates = []
            except Exception as e:
                logger.error(f"Candidate generation error: {e}")
                candidates = []
            
            if not candidates:
                logger.warning("No candidates generated, returning (0, 0)")
                return (0, 0)
            
            # STAGE 3 & 4: VERIFY & ARBITRATE (6.0s max combined)
            logger.info("STAGE 3-4: Verifying and arbitrating...")
            try:
                remaining_time = self.timeout - (time.time() - self.start_time)
                arb_timeout = min(remaining_time, self.time_per_stage['verify'] + self.time_per_stage['arbitrate'])
                
                with time_limit(arb_timeout):
                    arbitrator = AnswerArbitrator()
                    final_answer = arbitrator.arbitrate(candidates, problem_text)
                    logger.info(f"Final answer: {final_answer}")
            except TimeoutError:
                logger.warning("Verification/arbitration timeout, returning first two candidates")
                # Fallback: return top 2 candidates as-is
                if len(candidates) >= 2:
                    final_answer = (candidates[0][0], candidates[1][0])
                else:
                    final_answer = (candidates[0][0], candidates[0][0])
            except Exception as e:
                logger.error(f"Verification/arbitration error: {e}")
                # Fallback: return first two candidates
                if len(candidates) >= 2:
                    final_answer = (candidates[0][0], candidates[1][0])
                else:
                    final_answer = (candidates[0][0], candidates[0][0])
            
            return final_answer
        
        except Exception as e:
            logger.error(f"Solver error: {e}", exc_info=True)
            return (0, 0)
    
    def get_elapsed_time(self) -> float:
        """Get elapsed time since solve started."""
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time
    
    def time_remaining(self) -> float:
        """Get remaining time before timeout."""
        if self.start_time is None:
            return self.timeout
        elapsed = self.get_elapsed_time()
        return max(0.0, self.timeout - elapsed)


class AdaptiveSolver:
    """Kaggle-compatible interface wrapping StrategyArbiter."""
    
    def __init__(self, timeout_seconds: int = 30):
        self.timeout = timeout_seconds
        self.arbiter = StrategyArbiter(timeout=timeout_seconds)
    
    def solve(self, problem_text: str, timeout_seconds: Optional[int] = None) -> Tuple[int, int]:
        """
        Main entry point for Kaggle submission.
        
        Args:
            problem_text: LaTeX problem
            timeout_seconds: Override default timeout
        
        Returns:
            Tuple of two integers (answer1, answer2)
        """
        timeout = timeout_seconds if timeout_seconds else self.timeout
        self.arbiter = StrategyArbiter(timeout=timeout)
        
        try:
            result = self.arbiter.solve(problem_text)
            return result
        except Exception as e:
            logger.error(f"AdaptiveSolver failed: {e}")
            return (0, 0)


# Global instance for submission.py
_solver = None

def get_solver() -> AdaptiveSolver:
    """Get or create global solver instance."""
    global _solver
    if _solver is None:
        _solver = AdaptiveSolver(timeout_seconds=30)
    return _solver


def solve(problem_text: str, timeout_seconds: int = 30) -> Tuple[int, int]:
    """Entry point for submission (backward compatible)."""
    solver = get_solver()
    return solver.solve(problem_text, timeout_seconds=timeout_seconds)
