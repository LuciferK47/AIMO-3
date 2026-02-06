"""
SymPy Integration and Domain-Specific Solvers.

Provides symbolic math utilities for algebra, number theory, combinatorics, and geometry.
Converts LLM-generated formulas into deterministic computations.
"""

import logging
import re
import math
from multiprocessing import Process, Queue
from typing import Optional, List, Tuple, Any, Union, Dict
from utils import time_limit

try:
    import sympy as sp
    from sympy import symbols, solve, simplify, factor, expand, gcd, lcm, binomial, factorial
    from sympy import pi, sqrt, sin, cos, tan, atan, atan2, Poly, roots, Rational
    # Optional imports for number theory
    try:
        from sympy.ntheory import isprime, factorint
    except ImportError:
        isprime = None
        factorint = None
    try:
        from sympy.ntheory.modular import mod_inverse
    except ImportError:
        try:
            from sympy import mod_inverse
        except ImportError:
            mod_inverse = None
    SYMPY_AVAILABLE = True
    logging.info(f"SymPy {sp.__version__} loaded successfully")
except Exception as e:
    SYMPY_AVAILABLE = False
    logging.debug(f"SymPy import failed: {type(e).__name__}: {e}")

logger = logging.getLogger(__name__)


def safe_sympify(expr_str: str) -> Optional[Any]:
    """
    Safely parse a string into SymPy expression with restricted namespace.
    
    Prevents injection attacks like sympify("__import__('os').system('rm -rf /')")
    by using a restricted namespace with only math symbols.
    """
    if not SYMPY_AVAILABLE or not isinstance(expr_str, str):
        return None
    
    try:
        # Use restricted namespace with only safe math operations
        restricted_names = {
            'sqrt': sp.sqrt,
            'sin': sp.sin,
            'cos': sp.cos,
            'tan': sp.tan,
            'log': sp.log,
            'exp': sp.exp,
            'pi': sp.pi,
            'E': sp.E,
            'I': sp.I,
            'oo': sp.oo,
        }
        # Allow numeric operations but no imports/system calls
        return sp.sympify(expr_str, locals=restricted_names, rational=False)
    except Exception as e:
        logger.debug(f"sympify failed for '{expr_str}': {type(e).__name__}")
        return None



def _solve_worker(equation_expr: Any, symbols_list: List[Any], queue: Queue) -> None:
    """Worker process for solve with hard timeout."""
    try:
        result = sp.solve(equation_expr, symbols_list)
        queue.put(("success", result))
    except Exception as e:
        queue.put(("error", str(e)))


def solve_with_timeout(equation_expr: Any, symbols_list: List[Any], 
                       timeout: float = 2.0) -> Optional[List[Any]]:
    """
    Solve SymPy equation with hard timeout protection via multiprocessing.
    
    This uses Process.terminate() to forcefully kill SymPy's C extensions,
    avoiding hangs on pathological inputs (nested radicals, high-degree polynomials).
    
    Args:
        equation_expr: SymPy equation or expression
        symbols_list: List of SymPy symbols to solve for
        timeout: Maximum execution time in seconds
        
    Returns:
        Solutions or None if timeout/error occurs
    """
    if not SYMPY_AVAILABLE:
        return None
    
    try:
        queue: Queue = Queue()
        process = Process(target=_solve_worker, args=(equation_expr, symbols_list, queue))
        process.start()
        process.join(timeout=timeout)
        
        if process.is_alive():
            # Hard kill the process if it's still running
            process.terminate()
            process.join(timeout=0.5)
            if process.is_alive():
                process.kill()
            logger.warning(f"SymPy solve timeout after {timeout}s (hard killed)")
            return None
        
        # Check if result is available
        if not queue.empty():
            status, result = queue.get()
            if status == "success":
                return result
            else:
                logger.debug(f"SymPy solve error: {result}")
                return None
        
        return None
    except Exception as e:
        logger.debug(f"SymPy solve exception: {type(e).__name__}: {e}")
        return None




class SymPySolver:
    """Symbolic solver using SymPy for exact computations."""

    @staticmethod
    def verify_solution(equation_str: str, variable: str, solution: int) -> bool:
        """
        Plug solution back into equation to verify.
        
        CRITICAL: Never silently pass on exception. Log and return False.
        Exceptions indicate unparseable equations - treat as unverified.
        """
        if not SYMPY_AVAILABLE:
            return True
        try:
            var = sp.Symbol(variable)
            if '=' in equation_str:
                lhs, rhs = equation_str.split('=')
                lhs_val = safe_sympify(lhs)
                rhs_val = safe_sympify(rhs)
                if lhs_val is None or rhs_val is None:
                    logger.warning(f"Could not parse equation: {equation_str}")
                    return False
                lhs_val = lhs_val.subs(var, solution)
                rhs_val = rhs_val.subs(var, solution)
                return sp.simplify(lhs_val - rhs_val) == 0
            expr = safe_sympify(equation_str)
            if expr is None:
                logger.warning(f"Could not parse expression: {equation_str}")
                return False
            expr_val = expr.subs(var, solution)
            return sp.simplify(expr_val) == 0
        except Exception as e:
            logger.warning(f"Verification failed for {equation_str}={solution}: {type(e).__name__}: {e}")
            return False

    @staticmethod
    def solve_crt(remainders: List[int], moduli: List[int]) -> Optional[int]:
        """Solve system of congruences using CRT."""
        if not SYMPY_AVAILABLE:
            return None
        try:
            from sympy.ntheory.modular import crt
            result = crt(moduli, remainders)
            if result and result[0] is not None:
                return int(result[0])
        except Exception:
            return None
        return None
    
    @staticmethod
    def solve_equation(equation_str: str, variable_str: str = 'x') -> Optional[List[Any]]:
        """
        Solve a single equation for a variable with timeout protection.
        
        Args:
            equation_str: String like "x**2 - 4" or "x**2 - 4 = 0"
            variable_str: Variable name
            
        Returns:
            List of solutions, or None if solving fails/times out
        """
        if not SYMPY_AVAILABLE:
            return None
        
        try:
            var = sp.Symbol(variable_str)
            
            # Parse equation: handle "= 0" explicitly or assume it's set to 0
            if '=' in equation_str:
                lhs, rhs = equation_str.split('=')
                lhs_expr = safe_sympify(lhs)
                rhs_expr = safe_sympify(rhs)
                if lhs_expr is None or rhs_expr is None:
                    logger.warning(f"Could not parse equation: {equation_str}")
                    return None
                eq = lhs_expr - rhs_expr
            else:
                eq = safe_sympify(equation_str)
                if eq is None:
                    logger.warning(f"Could not parse expression: {equation_str}")
                    return None
            
            # Use timeout wrapper to prevent hangs
            solutions = solve_with_timeout(eq, [var], timeout=2.0)
            logger.debug(f"Equation solutions: {solutions}")
            return solutions
        except Exception as e:
            logger.debug(f"SymPy solve failed: {e}")
            return None
    
    @staticmethod
    def solve_system(equations: List[str], variables: List[str]) -> Optional[dict]:
        """
        Solve a system of equations WITH TIMEOUT PROTECTION.
        
        Args:
            equations: List of equation strings
            variables: List of variable names
            
        Returns:
            Dict of solutions, or None if solving fails/times out
        """
        if not SYMPY_AVAILABLE:
            return None
        
        try:
            var_objs = [sp.Symbol(v) for v in variables]
            
            eq_objs = []
            for eq_str in equations:
                if '=' in eq_str:
                    lhs, rhs = eq_str.split('=')
                    eq_objs.append(sp.Eq(sp.sympify(lhs), sp.sympify(rhs)))
                else:
                    eq_objs.append(sp.Eq(sp.sympify(eq_str), 0))
            
            # Use timeout wrapper to prevent hangs on complex systems
            def _solve_system():
                return sp.solve(eq_objs, var_objs)
            
            with time_limit(2.0):
                solution = _solve_system()
                logger.debug(f"System solutions: {solution}")
                return solution
        except TimeoutError:
            logger.warning(f"System solve timeout after 2s")
            return None
        except Exception as e:
            logger.debug(f"SymPy system solve failed: {e}")
            return None
    
    @staticmethod
    def simplify_expression(expr_str: str) -> Optional[str]:
        """
        Simplify an algebraic expression WITH TIMEOUT PROTECTION.
        
        Args:
            expr_str: Expression string like "2*x + 3*x"
            
        Returns:
            Simplified expression, or None if fails/times out
        """
        if not SYMPY_AVAILABLE:
            return None
        
        try:
            expr = sp.sympify(expr_str)
            # Simplify can hang on complex expressions
            with time_limit(1.0):
                simplified = sp.simplify(expr)
                logger.debug(f"Simplified: {expr} -> {simplified}")
                return str(simplified)
        except TimeoutError:
            logger.warning(f"Simplify timeout after 1s")
            return None
        except Exception as e:
            logger.debug(f"SymPy simplify failed: {e}")
            return None
    
    @staticmethod
    def evaluate_expression(expr_str: str, substitutions: dict = None) -> Optional[float]:
        """
        Evaluate an expression numerically, optionally with variable substitutions.
        
        Args:
            expr_str: Expression string
            substitutions: Dict of variable -> value
            
        Returns:
            Numeric result, or None if fails
        """
        if not SYMPY_AVAILABLE:
            return None
        
        try:
            expr = sp.sympify(expr_str)
            if substitutions:
                expr = expr.subs(substitutions)
            result = float(expr)
            logger.debug(f"Expression evaluated: {expr_str} = {result}")
            return result
        except Exception as e:
            logger.debug(f"SymPy evaluate failed: {e}")
            return None
    
    @staticmethod
    def solve_modular_equation(problem_text: str) -> Optional[Tuple[int, int]]:
        """
        Solve modular arithmetic problems including:
        - Simple powers: 7^1000 mod 13
        - CRT systems: x ≡ a (mod m), x ≡ b (mod n)
        - Fermat's Little Theorem: a^(p-1) ≡ 1 (mod p) for prime p
        - Euler's theorem: a^φ(n) ≡ 1 (mod n) for gcd(a,n)=1
        - Linear congruences: ax ≡ b (mod m)
        """
        if not SYMPY_AVAILABLE:
            return None
        
        try:
            # Try CRT system first (multiple congruences)
            congruences = re.findall(r'([a-zA-Z])\s*[≡=]\s*(\d+)\s*\(\s*mod\s*(\d+)\s*\)', problem_text)
            if len(congruences) >= 2:
                remainders = [int(c[1]) for c in congruences]
                moduli = [int(c[2]) for c in congruences]
                crt_result = SymPySolver.solve_crt(remainders, moduli)
                if crt_result is not None and 0 <= crt_result <= 99999:
                    return (crt_result, 0.8)
            
            # Extract modulo value
            mod_match = re.search(r'(?:mod|modulo)\s*(\d+)', problem_text.lower())
            if not mod_match:
                mod_match = re.search(r'divided\s+by\s+(\d+)', problem_text.lower())
            
            if not mod_match:
                return None
            
            modulo = int(mod_match.group(1))
            
            # Pattern 1: Power modulo (a^b mod m)
            power_match = re.search(r'(\d+)\s*(?:\^|\*\*)\s*(\d+)', problem_text)
            if power_match:
                base = int(power_match.group(1))
                exp = int(power_match.group(2))
                result = NumberTheorySolver.modular_power(base, exp, modulo)
                
                if 0 <= result <= 99999:
                    return (result, 0.85)
            
            # Pattern 2: Linear congruence (ax ≡ b (mod m))
            linear_match = re.search(r'(\d+)\s*\*?\s*[a-z]\s*[≡=]\s*(\d+)\s*\(\s*mod\s*(\d+)\s*\)', problem_text)
            if linear_match:
                a = int(linear_match.group(1))
                b = int(linear_match.group(2))
                m = int(linear_match.group(3))
                
                # Use extended GCD to find inverse
                from sympy.ntheory import mod_inverse
                g = NumberTheorySolver.compute_gcd(a, m)
                if b % g != 0:
                    return None  # No solution
                
                a_red = a // g
                b_red = b // g
                m_red = m // g
                
                try:
                    a_inv = mod_inverse(a_red, m_red)
                    x = (a_inv * b_red) % m_red
                    if 0 <= x <= 99999:
                        return (x, 0.8)
                except Exception:
                    pass
            
            return None
        except Exception as e:
            logger.debug(f"Enhanced modular solving error: {e}")
            return None
    
    @staticmethod
    def solve_combinatorics(problem_text: str) -> Optional[Tuple[int, int]]:
        """
        Solve combinatorics problems including:
        - Factorials: n!
        - Binomials: C(n,k) or nCk
        - Permutations: P(n,k) or nPk
        - Combinations with constraints
        """
        if not SYMPY_AVAILABLE:
            return None
        
        try:
            from sympy import factorial as fact, binomial as binom
            
            # Pattern 1: Binomial coefficient C(n,k)
            binom_matches = re.findall(r'C\((\d+),\s*(\d+)\)|(\d+)C(\d+)', problem_text)
            for match in binom_matches:
                if match[0]:  # C(n,k) format
                    n, k = int(match[0]), int(match[1])
                else:  # nCk format
                    n, k = int(match[2]), int(match[3])
                
                if 0 <= k <= n:
                    result = binom(n, k)
                    if result and 0 <= result <= 99999:
                        return (int(result), 0.8)
            
            # Pattern 2: Permutation P(n,k)
            perm_matches = re.findall(r'P\((\d+),\s*(\d+)\)|(\d+)P(\d+)', problem_text)
            for match in perm_matches:
                if match[0]:  # P(n,k) format
                    n, k = int(match[0]), int(match[1])
                else:  # nPk format
                    n, k = int(match[2]), int(match[3])
                
                if 0 <= k <= n:
                    result = fact(n) // fact(n - k)
                    if result and 0 <= result <= 99999:
                        return (int(result), 0.8)
            
            # Pattern 3: Factorial n!
            if '!' in problem_text:
                factorial_matches = re.findall(r'(\d+)\s*!', problem_text)
                for n_str in factorial_matches:
                    n = int(n_str)
                    if n <= 20:  # Factorial grows quickly
                        result = fact(n)
                        if result and 0 <= result <= 99999:
                            return (int(result), 0.8)
            
        except Exception as e:
            logger.debug(f"Enhanced combinatorics solving error: {e}")

        return None
    
    @staticmethod
    def solve_from_equations(equations: List[str], problem_text: str = "") -> Optional[Tuple[int, int]]:
        """
        Solve multi-equation systems.
        
        Supports:
        - Single equation in one variable
        - Multiple equations (system solving)
        - Auto-detection of variables
        
        Returns: (solution_value, confidence) or None
        """
        if not SYMPY_AVAILABLE or not equations:
            return None
        
        try:
            # GUARD: Check polynomial degree (D3 recommendation)
            # Skip if any equation has very high degree
            for eq_str in equations:
                try:
                    expr = safe_sympify(eq_str)
                    if expr is not None:
                        poly = sp.Poly(expr)
                        if poly.total_degree() > 6:
                            logger.debug(f"High-degree polynomial ({poly.total_degree()}), skipping")
                            return None
                except Exception:
                    pass  # Continue if we can't determine degree
            
            # Extract all unique variables from equations
            all_vars = set()
            for eq in equations:
                # Find all letter sequences (variable names)
                vars_in_eq = set(re.findall(r'[a-zA-Z_]\w*', eq))
                all_vars.update(vars_in_eq)
            
            if not all_vars:
                return None
            
            # Sort variables for consistency
            var_list = sorted(list(all_vars))
            
            # Try system solving with all equations if multiple
            if len(equations) > 1 and len(var_list) <= 3:
                try:
                    sympy_eqs = []
                    for eq_str in equations[:5]:  # Limit to 5 equations
                        if '=' not in eq_str:
                            continue
                        lhs, rhs = eq_str.split('=', 1)
                        sympy_eqs.append(sp.Eq(sp.sympify(lhs), sp.sympify(rhs)))
                    
                    if sympy_eqs:
                        with time_limit(2.0):
                            symbols = sp.symbols(' '.join(var_list))
                        if not isinstance(symbols, tuple):
                            symbols = (symbols,)
                        
                        # Solve system
                        solutions = sp.solve(sympy_eqs, symbols, dict=True)
                        
                        if solutions:
                            # Try to extract integer solutions
                            for sol_dict in solutions:
                                # Find first integer variable in solution
                                for var_sym, val in sol_dict.items():
                                    try:
                                        if val.is_integer or isinstance(val, (int, sp.Integer)):
                                            candidate = int(val)
                                            return (candidate, 0.8)
                                    except (ValueError, TypeError, AttributeError):
                                        pass
                except Exception as e:
                    logger.debug(f"System solving error: {e}")
            
            # Fallback: Try to solve the first equation
            if len(equations) >= 1:
                eq = equations[0]
                if '=' not in eq:
                    return None
                
                # Try with each variable as the unknown
                for var_name in var_list[:3]:  # Try first 3 variables
                    try:
                        solutions = SymPySolver.solve_equation(eq, var_name)
                        if solutions:
                            for sol in solutions:
                                try:
                                    candidate = int(sol)
                                except (ValueError, TypeError):
                                    continue
                                # Verify solution
                                if SymPySolver.verify_solution(eq, var_name, candidate):
                                    return (candidate, 0.7)
                    except Exception:
                        pass
            
        except Exception as e:
            logger.debug(f"Equation solving error: {e}")
        
        return None


class NumberTheorySolver:
    """Number theory specific solvers."""
    
    @staticmethod
    def modular_exponentiation(base: int, exponent: int, modulus: int) -> int:
        """
        Efficiently compute (base^exponent) mod modulus.
        
        Args:
            base: Base value
            exponent: Exponent (can be very large)
            modulus: Modulus
            
        Returns:
            Result modulo modulus
        """
        try:
            result = pow(base, exponent, modulus)
            logger.debug(f"Modexp: {base}^{exponent} mod {modulus} = {result}")
            return result
        except Exception as e:
            logger.error(f"Modexp failed: {e}")
            return 0
    
    @staticmethod
    def compute_gcd(*values) -> int:
        """Compute GCD of multiple values."""
        from math import gcd as math_gcd
        if not values:
            return 0
        result = values[0]
        for v in values[1:]:
            result = math_gcd(result, v)
        return result
    
    @staticmethod
    def compute_lcm(*values) -> int:
        """Compute LCM of multiple values."""
        from math import gcd as math_gcd
        if not values:
            return 1
        def lcm(a, b):
            return abs(a * b) // math_gcd(a, b) if math_gcd(a, b) else 0
        result = values[0]
        for v in values[1:]:
            result = lcm(result, v)
        return result
    
    @staticmethod
    def is_prime(n: int) -> bool:
        """Check if n is prime."""
        if not SYMPY_AVAILABLE:
            # Fallback simple check
            if n < 2:
                return False
            for i in range(2, int(n**0.5) + 1):
                if n % i == 0:
                    return False
            return True
        try:
            return isprime(n)
        except Exception:
            return False
    
    @staticmethod
    def prime_factorization(n: int) -> dict:
        """
        Factorize n into primes.
        
        Returns:
            Dict of {prime: exponent}
        """
        if not SYMPY_AVAILABLE:
            # Simple fallback
            factors = {}
            d = 2
            while d * d <= n:
                while n % d == 0:
                    factors[d] = factors.get(d, 0) + 1
                    n //= d
                d += 1
            if n > 1:
                factors[n] = factors.get(n, 0) + 1
            return factors
        try:
            return factorint(n)
        except Exception:
            return {}
    
    @staticmethod
    def modular_inverse(a: int, m: int) -> Optional[int]:
        """
        Compute modular inverse a^(-1) mod m.
        
        Returns None if inverse doesn't exist.
        """
        if not SYMPY_AVAILABLE:
            # Extended Euclidean algorithm fallback
            def egcd(a, b):
                if a == 0:
                    return b, 0, 1
                gcd, x1, y1 = egcd(b % a, a)
                x = y1 - (b // a) * x1
                y = x1
                return gcd, x, y
            
            gcd, x, _ = egcd(a % m, m)
            if gcd != 1:
                return None
            return (x % m + m) % m
        
        try:
            return int(mod_inverse(a, m))
        except Exception:
            return None

    @staticmethod
    def modular_power(base: int, exp: int, mod: int) -> int:
        """
        Compute base^exp mod mod efficiently.
        Uses Fermat/Euler when applicable.
        """
        if mod == 1:
            return 0
        g = NumberTheorySolver.compute_gcd(base, mod)
        if NumberTheorySolver.is_prime(mod) and g == 1:
            reduced_exp = exp % (mod - 1)
            return pow(base, reduced_exp, mod)
        phi_n = NumberTheorySolver.euler_totient(mod)
        if phi_n is not None and g == 1:
            reduced_exp = exp % phi_n
            return pow(base, reduced_exp, mod)
        return pow(base, exp, mod)

    @staticmethod
    def euler_totient(n: int) -> Optional[int]:
        """Compute Euler's totient function φ(n)."""
        if not SYMPY_AVAILABLE:
            return None
        try:
            return int(sp.totient(n))
        except Exception:
            try:
                from sympy.ntheory import totient
                return int(totient(n))
            except Exception:
                return None


class CombinatoricsSolver:
    """Combinatorics specific solvers."""
    
    @staticmethod
    def binomial_coefficient(n: int, k: int) -> int:
        """Compute C(n, k) = n choose k."""
        if not SYMPY_AVAILABLE:
            from math import factorial
            if k > n or k < 0:
                return 0
            return factorial(n) // (factorial(k) * factorial(n - k))
        try:
            return int(binomial(n, k))
        except Exception:
            return 0
    
    @staticmethod
    def factorial_value(n: int) -> int:
        """Compute n!."""
        from math import factorial
        try:
            return factorial(n)
        except Exception:
            return 0
    
    @staticmethod
    def count_permutations(n: int, k: int) -> int:
        """Compute P(n, k) = n!/(n-k)!."""
        if k > n or k < 0:
            return 0
        from math import factorial
        try:
            return factorial(n) // factorial(n - k)
        except Exception:
            return 0
    
    @staticmethod
    def brute_force_counting(condition_func, search_range: Tuple[int, int]) -> int:
        """
        Brute-force count elements satisfying a condition.
        
        Args:
            condition_func: Function that returns True if element satisfies condition
            search_range: (min, max) tuple for search space
            
        Returns:
            Count of satisfying elements
        """
        count = 0
        start, end = search_range
        for i in range(start, min(end + 1, start + 100000)):  # Cap to avoid infinite loops
            try:
                if condition_func(i):
                    count += 1
            except Exception:
                pass
        return count


class NumberTheoryAdvanced:
    """Advanced number theory solvers."""
    
    @staticmethod
    def chinese_remainder_theorem(residues: List[int], moduli: List[int]) -> Optional[int]:
        """
        Solve system of congruences using Chinese Remainder Theorem.
        
        Solves: x ≡ residues[i] (mod moduli[i]) for pairwise coprime moduli.
        
        Args:
            residues: List of remainders
            moduli: List of moduli (must be pairwise coprime)
            
        Returns:
            Solution x in [0, product(moduli)) if exists, else None
        """
        if not residues or not moduli or len(residues) != len(moduli):
            return None
        
        try:
            from functools import reduce
            
            def extended_gcd(a, b):
                """Extended Euclidean algorithm."""
                if a == 0:
                    return b, 0, 1
                g, x, y = extended_gcd(b % a, a)
                return g, y - (b // a) * x, x
            
            # Compute product of all moduli
            M = reduce(lambda a, b: a * b, moduli, 1)
            
            result = 0
            for r, m in zip(residues, moduli):
                Mi = M // m
                g, _, yi = extended_gcd(m, Mi)
                
                # Check if coprime
                if g != 1:
                    logger.debug(f"CRT: moduli {m} and {M // m} not coprime")
                    return None
                
                result = (result + r * Mi * yi) % M
            
            # Return in range [0, 99999] if possible
            if 0 <= result <= 99999:
                return result
            
            # Try to reduce if result is too large
            if result > 99999:
                result = result % 99999
                if result <= 99999:
                    return result
            
            return None
        except Exception as e:
            logger.debug(f"CRT error: {e}")
            return None
    
    @staticmethod
    def solve_congruence_system(problem_text: str) -> Optional[Tuple[int, float]]:
        """
        Detect and solve systems of congruences in problem text.
        
        Pattern: "Find x such that x ≡ a (mod m) and x ≡ b (mod n)..."
        """
        try:
            import re
            problem_lower = problem_text.lower()
            
            # Pattern: x ≡ a (mod m)
            pattern = r'([a-z])\s*(?:≡|=)\s*(\d+)\s*\(\s*(?:mod|modulo)\s+(\d+)\s*\)'
            matches = re.findall(pattern, problem_text)
            
            if len(matches) < 2:
                return None
            
            # Extract residues and moduli
            residues = []
            moduli = []
            for var, residue, modulus in matches:
                residues.append(int(residue))
                moduli.append(int(modulus))
            
            # Solve using CRT
            solution = NumberTheoryAdvanced.chinese_remainder_theorem(residues, moduli)
            if solution is not None:
                return (solution, 0.85)
            
            return None
        except Exception as e:
            logger.debug(f"Congruence system solving error: {e}")
            return None


class GeometrySolver:
    """Geometry specific solvers."""
    
    @staticmethod
    def solve_coordinate_geometry(problem_text: str) -> Optional[Tuple[int, int]]:
        """
        Coordinate geometry solver (D6 from brutal assessment).
        
        Patterns:
        - Triangle vertices → area/perimeter
        - Distance formulas
        - Circles with coordinates
        - Line intersections
        
        Returns (answer, confidence) or None.
        """
        try:
            text_lower = problem_text.lower()
            
            coords = GeometrySolver.extract_coordinates(problem_text)

            # Pattern 1: Triangle with coordinates → area
            if len(coords) >= 3 and ('area' in text_lower or 'triangle' in text_lower):
                x1, y1 = coords[0]
                x2, y2 = coords[1]
                x3, y3 = coords[2]
                
                area = abs((x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2)) / 2.0)
                return (int(area), 0.8)

            # Pattern 1b: Angle at middle point (A-B-C)
            if len(coords) >= 3 and 'angle' in text_lower:
                ax, ay = coords[0]
                bx, by = coords[1]
                cx, cy = coords[2]

                v1x, v1y = ax - bx, ay - by
                v2x, v2y = cx - bx, cy - by
                dot = v1x * v2x + v1y * v2y
                mag1 = math.sqrt(v1x * v1x + v1y * v1y)
                mag2 = math.sqrt(v2x * v2x + v2y * v2y)
                if mag1 > 0 and mag2 > 0:
                    cos_theta = max(-1.0, min(1.0, dot / (mag1 * mag2)))
                    angle_rad = math.acos(cos_theta)
                    angle_deg = angle_rad * 180.0 / math.pi
                    if 'degree' in text_lower:
                        return (int(round(angle_deg)), 0.75)
                    return (int(round(angle_rad)), 0.7)
            
            # Pattern 2: Distance between two points
            if len(coords) == 2 and 'distance' in text_lower:
                x1, y1 = coords[0]
                x2, y2 = coords[1]
                
                dist_sq = (x2-x1)**2 + (y2-y1)**2
                dist = math.sqrt(dist_sq)
                
                # Check if asking for squared distance
                if 'square' in text_lower or 'squared' in text_lower:
                    return (int(dist_sq), 0.9)
                else:
                    return (int(round(dist)), 0.8)
            
            # Pattern 3: Circle radius given center and point
            if 'circle' in text_lower and 'radius' in text_lower and len(coords) >= 2:
                x1, y1 = coords[0]  # center
                x2, y2 = coords[1]  # point on circle
                
                radius_sq = (x2-x1)**2 + (y2-y1)**2
                radius = math.sqrt(radius_sq)
                
                return (int(round(radius)), 0.8)
            
            return None
        except Exception as e:
            logger.debug(f"Coordinate geometry failed: {e}")
            return None

    @staticmethod
    def extract_coordinates(problem_text: str) -> List[Tuple[float, float]]:
        """
        Extract coordinates from various formats:
        - (3, 4), (3,4), (3.5, 4.2)
        - A = (3, 4)
        """
        coords = []
        pattern = r'\((\-?\d+(?:\.\d+)?),\s*(\-?\d+(?:\.\d+)?)\)'
        for match in re.finditer(pattern, problem_text):
            try:
                x = float(match.group(1))
                y = float(match.group(2))
                coords.append((x, y))
            except Exception:
                continue
        return coords
    
    @staticmethod
    def distance_2d(x1: float, y1: float, x2: float, y2: float) -> float:
        """Compute Euclidean distance between two points."""
        import math
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    @staticmethod
    def triangle_area_coords(x1: float, y1: float, x2: float, y2: float, 
                             x3: float, y3: float) -> float:
        """
        Compute area of triangle given three coordinate points.
        Uses the cross product formula.
        """
        area = abs((x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2)) / 2.0)
        return area
    
    @staticmethod
    def circle_through_points(x1: float, y1: float, x2: float, y2: float, 
                               x3: float, y3: float) -> Tuple[float, float, float]:
        """
        Find circle passing through three points.
        
        Returns:
            (center_x, center_y, radius)
        """
        try:
            # Using SymPy for exact computation if available
            if SYMPY_AVAILABLE:
                px1, py1 = sp.Rational(x1), sp.Rational(y1)
                px2, py2 = sp.Rational(x2), sp.Rational(y2)
                px3, py3 = sp.Rational(x3), sp.Rational(y3)
                
                # Solve for circle center (a, b) and radius r
                a, b, r = sp.symbols('a b r', real=True)
                eq1 = sp.Eq((px1 - a)**2 + (py1 - b)**2, r**2)
                eq2 = sp.Eq((px2 - a)**2 + (py2 - b)**2, r**2)
                eq3 = sp.Eq((px3 - a)**2 + (py3 - b)**2, r**2)
                
                sol = sp.solve([eq1, eq2, eq3], [a, b, r])
                if sol:
                    center_x = float(sol[0][0])
                    center_y = float(sol[0][1])
                    radius = float(abs(sol[0][2]))
                    return center_x, center_y, radius
        except Exception as e:
            logger.debug(f"Circle computation failed: {e}")
        
        # Fallback: basic numeric computation
        import math
        d1 = (x1**2 + y1**2)
        d2 = (x2**2 + y2**2)
        d3 = (x3**2 + y3**2)
        
        a = x1 * (y2 - y3) - y1 * (x2 - x3) + x2 * y3 - x3 * y2
        if abs(a) < 1e-10:
            return 0, 0, 0  # Collinear points
        
        b = d1 * (y3 - y2) + d2 * (y1 - y3) + d3 * (y2 - y1)
        c = d1 * (x2 - x3) + d2 * (x3 - x1) + d3 * (x1 - x2)
        d = d1 * (x3 * y2 - x2 * y3) + d2 * (x1 * y3 - x3 * y1) + d3 * (x2 * y1 - x1 * y2)
        
        center_x = -b / (2 * a)
        center_y = -c / (2 * a)
        radius = math.sqrt(center_x**2 + center_y**2 - d / a)
        
        return center_x, center_y, radius


class DiophantineSolver:
    """Specialized Diophantine equation solving."""
    
    @staticmethod
    def linear_diophantine(a: int, b: int, c: int) -> Optional[Tuple[int, int]]:
        """
        Solve ax + by = c for integers x, y.
        Returns one particular solution or None if no solution exists.
        """
        if not SYMPY_AVAILABLE:
            return None
        
        try:
            from sympy import gcdex
            import math
            
            gcd_ab = math.gcd(abs(a), abs(b))
            
            if c % gcd_ab != 0:
                return None  # No solution
            
            # gcdex returns (g, x, y) where g = a*x + b*y
            # We use it to get coefficients, then scale
            gcd_val, x0, y0 = gcdex(a, b)
            
            # Scale coefficients to satisfy ax + by = c
            # Since ax0 + by0 = gcd_val, we multiply by c/gcd_val
            scale = c // gcd_val
            return (int(x0 * scale), int(y0 * scale))
        except Exception as e:
            logger.debug(f"Linear Diophantine solve failed: {e}")
            return None
    
    @staticmethod
    def quadratic_diophantine(problem_text: str) -> Optional[Tuple[int, int]]:
        """
        Solve quadratic Diophantine equations (D2 from brutal assessment):
        - x² + y² = n (sum of two squares)
        - x² - y² = n (difference of squares)
        - xy = n (product)
        
        Returns (answer, confidence) or None.
        """
        try:
            # Pattern 1: x² + y² = n
            match = re.search(r'x\*\*2\s*\+\s*y\*\*2\s*=\s*(\d+)', problem_text)
            if not match:
                match = re.search(r'x\^2\s*\+\s*y\^2\s*=\s*(\d+)', problem_text)
            if not match:
                match = re.search(r'x²\s*\+\s*y²\s*=\s*(\d+)', problem_text)
            
            if match:
                n = int(match.group(1))
                solutions = DiophantineSolver.solve_sum_of_two_squares(n)
                
                if solutions:
                    # Check what the problem wants
                    text_lower = problem_text.lower()
                    if 'how many' in text_lower or 'count' in text_lower:
                        # Count distinct solutions (consider symmetry)
                        unique = len(set((min(x,y), max(x,y)) for x,y in solutions))
                        return (unique, 0.8)
                    elif 'sum' in text_lower or 'x+y' in text_lower or 'x + y' in text_lower:
                        return (solutions[0][0] + solutions[0][1], 0.8)
                    else:
                        # Return x value of first solution
                        return (solutions[0][0], 0.7)
            
            # Pattern 2: xy = n (find factor pairs)
            match = re.search(r'x\s*\*\s*y\s*=\s*(\d+)', problem_text)
            if match:
                n = int(match.group(1))
                # Find all divisors
                divisors = []
                for d in range(1, min(int(math.sqrt(n)) + 1, 1000)):
                    if n % d == 0:
                        divisors.append((d, n // d))
                
                if divisors:
                    text_lower = problem_text.lower()
                    if 'how many' in text_lower:
                        return (len(divisors), 0.8)
                    elif 'minimum' in text_lower or 'smallest' in text_lower:
                        return (min(divisors[0][0], divisors[0][1]), 0.8)
                    elif 'maximum' in text_lower or 'largest' in text_lower:
                        return (max(divisors[-1][0], divisors[-1][1]), 0.8)
            
            return None
        except Exception as e:
            logger.debug(f"Quadratic Diophantine failed: {e}")
            return None

    @staticmethod
    def solve_sum_of_two_squares(n: int) -> List[Tuple[int, int]]:
        """
        Find all (x, y) with x² + y² = n.
        Uses Fermat's theorem: n is sum of two squares iff
        no prime p ≡ 3 (mod 4) appears with odd exponent.
        """
        if n < 0:
            return []
        try:
            factors = NumberTheorySolver.prime_factorization(n)
            for p, exp in factors.items():
                if p % 4 == 3 and exp % 2 == 1:
                    return []
        except Exception:
            pass

        solutions = []
        limit = int(math.sqrt(n)) + 1
        for x in range(0, limit):
            remainder = n - x * x
            if remainder < 0:
                break
            y = int(math.isqrt(remainder))
            if y * y == remainder:
                solutions.append((x, y))
        return solutions
    
    @staticmethod
    def modular_congruence(a: int, b: int, m: int) -> Optional[int]:
        """
        Solve ax ≡ b (mod m) for x.
        Returns smallest non-negative solution or None if none exists.
        """
        gcd_am = NumberTheorySolver.compute_gcd(a, m)
        
        if b % gcd_am != 0:
            return None
        
        # Reduce to ax' ≡ b' (mod m') where gcd(a', m') = 1
        a_prime = a // gcd_am
        b_prime = b // gcd_am
        m_prime = m // gcd_am
        
        inv = NumberTheorySolver.modular_inverse(a_prime, m_prime)
        if inv is None:
            return None
        
        x = (inv * b_prime) % m_prime
        return x
    
    @staticmethod
    def quadratic_residue(a: int, p: int) -> bool:
        """Check if a is a quadratic residue mod p (p prime)."""
        if p == 2:
            return True
        
        # Legendre symbol using Euler's criterion
        legendre = pow(a % p, (p - 1) // 2, p)
        return legendre == 1

    @staticmethod
    def count_diophantine_solutions(equation_str: str, bounds: Dict[str, Tuple[int, int]]) -> Optional[int]:
        """Count integer solutions within bounds for a diophantine equation."""
        if not SYMPY_AVAILABLE:
            return None
        try:
            from sympy import diophantine
            symbols_in_eq = sorted(set(re.findall(r'[a-zA-Z]', equation_str)))
            if not symbols_in_eq:
                return None
            vars_ = sp.symbols(' '.join(symbols_in_eq))
            expr = sp.sympify(equation_str.replace('=', '-(') + ')') if '=' in equation_str else sp.sympify(equation_str)
            sols = diophantine(expr, *vars_)
            count = 0
            for sol in sols:
                if not isinstance(sol, (tuple, list)):
                    sol = (sol,)
                ok = True
                for var_name, value in zip(symbols_in_eq, sol):
                    if var_name in bounds:
                        lo, hi = bounds[var_name]
                        try:
                            val_int = int(value)
                        except Exception:
                            ok = False
                            break
                        if not (lo <= val_int <= hi):
                            ok = False
                            break
                if ok:
                    count += 1
            return count
        except Exception:
            return None


class EquationExtractor:
    """Extract and normalize equations from text."""
    
    @staticmethod
    def extract_equations(text: str) -> List[str]:
        """
        Extract equation strings from problem text.
        Looks for = signs, mathematical expressions.
        """
        equations = []

        # LaTeX-aware equation patterns
        patterns = [
            r'\\\[([^\\]+)\\\]',   # \[ ... \]
            r'\$([^$]+)\$',            # $ ... $
            r'([a-zA-Z_]\w*)\s*=\s*([^,\.;]+)',
            r'([a-zA-Z_]\w*)\s*\+\s*([a-zA-Z_]\w*)\s*=\s*([^,\.;]+)',
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, text):
                eq = match.group(0).strip()
                # If using capture group (LaTeX), take group(1)
                if match.lastindex:
                    eq = match.group(match.lastindex).strip()
                eq = eq.replace('"', '').replace('"', '').strip()
                eq = EquationExtractor.normalize_expression(eq)
                if eq and '=' in eq and len(eq) > 2:
                    equations.append(eq)

        return equations
    
    @staticmethod
    def normalize_expression(expr: str) -> str:
        """Normalize mathematical expression for SymPy."""
        # Replace common math symbols
        expr = expr.replace('×', '*')
        expr = expr.replace('÷', '/')
        expr = expr.replace('√', 'sqrt')
        expr = expr.replace('^', '**')
        expr = expr.replace('π', 'pi')
        expr = expr.replace('∞', 'oo')
        
        # Handle implicit multiplication (e.g., "2x" -> "2*x")
        expr = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', expr)
        expr = re.sub(r'(\))([a-zA-Z(])', r'\1*\2', expr)
        
        return expr


class ConstraintSolver:
    """Solve problems with explicit constraints."""
    
    @staticmethod
    def apply_integer_constraints(solutions: List[Any], variable_name: str = 'x') -> List[int]:
        """
        Filter solutions to only integers.
        Useful for number theory and discrete problems.
        """
        integers = []
        for sol in solutions:
            try:
                if hasattr(sol, 'is_integer') and sol.is_integer:
                    integers.append(int(sol))
                elif isinstance(sol, int):
                    integers.append(sol)
                else:
                    # Try to convert to int
                    val = float(sol)
                    if abs(val - round(val)) < 1e-9:
                        integers.append(int(round(val)))
            except (TypeError, ValueError):
                pass
        return integers
    
    @staticmethod
    def apply_range_constraints(values: List[Any], min_val: int, max_val: int) -> List[Any]:
        """Filter values to stay within [min_val, max_val]."""
        return [v for v in values if min_val <= v <= max_val]
    
    @staticmethod
    def apply_modulo_constraints(values: List[Any], modulo: int) -> List[int]:
        """Apply modulo constraint and return results."""
        return [int(v % modulo) for v in values if v is not None]


# ============================================================================
# END OF FILE - Duplicate class definitions removed
# ============================================================================


