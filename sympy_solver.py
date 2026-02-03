"""
SymPy Integration and Domain-Specific Solvers.

Provides symbolic math utilities for algebra, number theory, combinatorics, and geometry.
Converts LLM-generated formulas into deterministic computations.
"""

import logging
import re
from typing import Optional, List, Tuple, Any, Union, Dict

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


class SymPySolver:
    """Symbolic solver using SymPy for exact computations."""

    @staticmethod
    def verify_solution(equation_str: str, variable: str, solution: int) -> bool:
        """Plug solution back into equation to verify."""
        if not SYMPY_AVAILABLE:
            return True
        try:
            var = sp.Symbol(variable)
            if '=' in equation_str:
                lhs, rhs = equation_str.split('=')
                lhs_val = sp.sympify(lhs).subs(var, solution)
                rhs_val = sp.sympify(rhs).subs(var, solution)
                return sp.simplify(lhs_val - rhs_val) == 0
            expr_val = sp.sympify(equation_str).subs(var, solution)
            return sp.simplify(expr_val) == 0
        except Exception:
            return True

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
        Solve a single equation for a variable.
        
        Args:
            equation_str: String like "x**2 - 4" or "x**2 - 4 = 0"
            variable_str: Variable name
            
        Returns:
            List of solutions, or None if solving fails
        """
        if not SYMPY_AVAILABLE:
            return None
        
        try:
            var = sp.Symbol(variable_str)
            
            # Parse equation: handle "= 0" explicitly or assume it's set to 0
            if '=' in equation_str:
                lhs, rhs = equation_str.split('=')
                eq = sp.sympify(lhs) - sp.sympify(rhs)
            else:
                eq = sp.sympify(equation_str)
            
            solutions = sp.solve(eq, var)
            logger.debug(f"Equation solutions: {solutions}")
            return solutions
        except Exception as e:
            logger.debug(f"SymPy solve failed: {e}")
            return None
    
    @staticmethod
    def solve_system(equations: List[str], variables: List[str]) -> Optional[dict]:
        """
        Solve a system of equations.
        
        Args:
            equations: List of equation strings
            variables: List of variable names
            
        Returns:
            Dict of solutions, or None if solving fails
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
            
            solution = sp.solve(eq_objs, var_objs)
            logger.debug(f"System solutions: {solution}")
            return solution
        except Exception as e:
            logger.debug(f"SymPy system solve failed: {e}")
            return None
    
    @staticmethod
    def simplify_expression(expr_str: str) -> Optional[str]:
        """
        Simplify an algebraic expression.
        
        Args:
            expr_str: Expression string like "2*x + 3*x"
            
        Returns:
            Simplified expression, or None if fails
        """
        if not SYMPY_AVAILABLE:
            return None
        
        try:
            expr = sp.sympify(expr_str)
            simplified = sp.simplify(expr)
            logger.debug(f"Simplified: {expr} -> {simplified}")
            return str(simplified)
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
        """Solve modular arithmetic problems like '7^1000 mod 13' or 'remainder when divided by'."""
        if not SYMPY_AVAILABLE:
            return None
        
        try:
            # Detect multiple congruences for CRT: x ≡ a (mod m)
            congruences = re.findall(r'([a-zA-Z])\s*[≡=]\s*(\d+)\s*\(\s*mod\s*(\d+)\s*\)', problem_text)
            if len(congruences) >= 2:
                remainders = [int(c[1]) for c in congruences]
                moduli = [int(c[2]) for c in congruences]
                crt_result = SymPySolver.solve_crt(remainders, moduli)
                if crt_result is not None:
                    return (crt_result, 0)

            # Extract modulo value - look for "mod X", "modulo X", or "divided by X"
            mod_match = re.search(r'(?:mod|modulo)\s*(\d+)', problem_text.lower())
            if not mod_match:
                # Try "divided by" pattern
                mod_match = re.search(r'divided\s+by\s+(\d+)', problem_text.lower())
            
            if not mod_match:
                return None
            
            modulo = int(mod_match.group(1))
            
            # Extract base and exponent for power mod
            power_match = re.search(r'(\d+)\s*\^\s*(\d+)', problem_text)
            if power_match:
                base = int(power_match.group(1))
                exp = int(power_match.group(2))
                result = pow(base, exp, modulo)
                return (result, 0)
            
            return None
        except Exception as e:
            logger.debug(f"Modular solving error: {e}")
            return None
    
    @staticmethod
    def solve_combinatorics(problem_text: str) -> Optional[Tuple[int, int]]:
        """Solve combinatorics problems."""
        if not SYMPY_AVAILABLE:
            return None
        
        try:
            # This is heuristic; would need better parsing in production
            from sympy import factorial as fact
            
            # Look for patterns like "n!" or factorial
            if '!' in problem_text:
                n_match = re.search(r'(\d+)\s*!', problem_text)
                if n_match:
                    n = int(n_match.group(1))
                    result = fact(n)
                    return (int(result), 0)
        except Exception as e:
            logger.debug(f"Combinatorics solving error: {e}")
        
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


class GeometrySolver:
    """Geometry specific solvers."""
    
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
            gcd_ab = NumberTheorySolver.compute_gcd(abs(a), abs(b))
            
            if c % gcd_ab != 0:
                return None  # No solution
            
            x0, y0 = gcdex(a, b)
            # Scale to c
            scale = c // gcd_ab
            return (x0 * scale, y0 * scale)
        except Exception as e:
            logger.debug(f"Linear Diophantine solve failed: {e}")
            return None
    
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
        
        # Find patterns like "x = ...", "x^2 + y = 5", etc.
        # Pattern: variable(s) = expression
        patterns = [
            r'([a-zA-Z_]\w*)\s*=\s*([^,\.;]+)',  # x = expression
            r'([a-zA-Z_]\w*)\s*\+\s*([a-zA-Z_]\w*)\s*=\s*([^,\.;]+)',  # x + y = expression
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                eq = match.group(0).strip()
                # Clean up
                eq = eq.replace('"', '').replace('"', '').strip()
                if eq and len(eq) > 2:
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


