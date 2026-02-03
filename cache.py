"""
Caching and Memoization Infrastructure.

Implements result caching, computation memoization, and intermediate result storage.
"""

import logging
import hashlib
import json
from typing import Any, Optional, Dict, Callable
from functools import wraps

logger = logging.getLogger(__name__)


class ResultCache:
    """Simple in-memory cache for problem solutions."""
    
    def __init__(self, max_size: int = 1000):
        """
        Initialize cache.
        
        Args:
            max_size: Maximum number of cached items
        """
        self.cache: Dict[str, Any] = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
    
    def _hash_key(self, key: str) -> str:
        """Generate hash key for storage."""
        return hashlib.md5(key.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve cached value.
        
        Returns:
            Cached value or None if not found
        """
        hash_key = self._hash_key(key)
        if hash_key in self.cache:
            self.hits += 1
            logger.debug(f"Cache hit: {key[:50]}...")
            return self.cache[hash_key]
        
        self.misses += 1
        return None
    
    def put(self, key: str, value: Any) -> None:
        """
        Store value in cache.
        """
        if len(self.cache) >= self.max_size:
            # Simple eviction: remove first item
            self.cache.pop(next(iter(self.cache)))
            logger.debug("Cache full; evicted oldest item")
        
        hash_key = self._hash_key(key)
        self.cache[hash_key] = value
        logger.debug(f"Cache put: {key[:50]}...")
    
    def clear(self) -> None:
        """Clear all cached values."""
        self.cache.clear()
        logger.info("Cache cleared")
    
    def stats(self) -> Dict[str, Any]:
        """
        Return cache statistics.
        """
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0.0
        
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'total_requests': total_requests
        }


class MemoizationCache:
    """Decorator-based memoization for function results."""
    
    def __init__(self):
        self.cache = {}
    
    def memoize(self, func: Callable) -> Callable:
        """
        Memoization decorator.
        
        Usage:
            @memo_cache.memoize
            def expensive_computation(x, y):
                ...
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            cache_key = (func.__name__, args, tuple(sorted(kwargs.items())))
            cache_key_str = str(cache_key)
            
            # Check cache
            if cache_key_str in self.cache:
                logger.debug(f"Memoization hit: {func.__name__}{args}")
                return self.cache[cache_key_str]
            
            # Compute and cache
            result = func(*args, **kwargs)
            self.cache[cache_key_str] = result
            logger.debug(f"Memoization store: {func.__name__}{args}")
            
            return result
        
        return wrapper
    
    def clear(self) -> None:
        """Clear memoization cache."""
        self.cache.clear()


class SubproblemCache:
    """Cache for intermediate subproblem solutions."""
    
    def __init__(self):
        self.subproblems: Dict[str, Dict[str, Any]] = {}
    
    def store_subproblem(self, problem_id: str, subproblem_name: str, 
                        solution: Any, metadata: Dict = None) -> None:
        """
        Store solution to a subproblem.
        
        Args:
            problem_id: ID of parent problem
            subproblem_name: Name/key of subproblem
            solution: Solution value
            metadata: Optional metadata (method used, confidence, etc.)
        """
        if problem_id not in self.subproblems:
            self.subproblems[problem_id] = {}
        
        self.subproblems[problem_id][subproblem_name] = {
            'solution': solution,
            'metadata': metadata or {}
        }
        
        logger.debug(f"Stored subproblem: {problem_id}.{subproblem_name} = {solution}")
    
    def retrieve_subproblem(self, problem_id: str, subproblem_name: str) -> Optional[Any]:
        """
        Retrieve cached subproblem solution.
        """
        if problem_id in self.subproblems:
            if subproblem_name in self.subproblems[problem_id]:
                solution = self.subproblems[problem_id][subproblem_name]['solution']
                logger.debug(f"Retrieved cached subproblem: {problem_id}.{subproblem_name}")
                return solution
        
        return None
    
    def get_problem_solutions(self, problem_id: str) -> Dict[str, Any]:
        """
        Get all subproblem solutions for a problem.
        """
        return self.subproblems.get(problem_id, {})
    
    def clear(self) -> None:
        """Clear all subproblem cache."""
        self.subproblems.clear()


class ComputationCache:
    """Cache expensive computations like symbolic solves or modular exponentiation."""
    
    def __init__(self):
        self.sym_solve_cache: Dict[str, Any] = {}
        self.modexp_cache: Dict[Tuple[int, int, int], int] = {}
        self.gcd_cache: Dict[Tuple[int, ...], int] = {}
    
    def cache_symbolic_solve(self, equation: str, solution: Any) -> None:
        """Cache symbolic equation solving."""
        self.sym_solve_cache[equation] = solution
        logger.debug(f"Cached symbolic solve: {equation}")
    
    def get_symbolic_solve(self, equation: str) -> Optional[Any]:
        """Retrieve cached symbolic solve."""
        return self.sym_solve_cache.get(equation)
    
    def cache_modexp(self, base: int, exp: int, mod: int, result: int) -> None:
        """Cache modular exponentiation."""
        self.modexp_cache[(base, exp, mod)] = result
        logger.debug(f"Cached modexp: {base}^{exp} mod {mod} = {result}")
    
    def get_modexp(self, base: int, exp: int, mod: int) -> Optional[int]:
        """Retrieve cached modexp."""
        return self.modexp_cache.get((base, exp, mod))
    
    def cache_gcd(self, *values, result: int) -> None:
        """Cache GCD computation."""
        self.gcd_cache[tuple(sorted(values))] = result
        logger.debug(f"Cached GCD: gcd{values} = {result}")
    
    def get_gcd(self, *values) -> Optional[int]:
        """Retrieve cached GCD."""
        return self.gcd_cache.get(tuple(sorted(values)))
    
    def clear(self) -> None:
        """Clear all computation caches."""
        self.sym_solve_cache.clear()
        self.modexp_cache.clear()
        self.gcd_cache.clear()


# Global cache instances
_result_cache = ResultCache()
_memo_cache = MemoizationCache()
_subproblem_cache = SubproblemCache()
_computation_cache = ComputationCache()


def get_result_cache() -> ResultCache:
    """Get global result cache."""
    return _result_cache


def get_memoization_cache() -> MemoizationCache:
    """Get global memoization cache."""
    return _memo_cache


def get_subproblem_cache() -> SubproblemCache:
    """Get global subproblem cache."""
    return _subproblem_cache


def get_computation_cache() -> ComputationCache:
    """Get global computation cache."""
    return _computation_cache


def cache_result(func: Callable) -> Callable:
    """
    Decorator to cache function results using ResultCache.
    
    Usage:
        @cache_result
        def solve_problem(problem_text):
            ...
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Create cache key from arguments
        cache_key = json.dumps({'args': args, 'kwargs': kwargs}, default=str)
        
        # Check cache
        cached = _result_cache.get(cache_key)
        if cached is not None:
            return cached
        
        # Compute and cache
        result = func(*args, **kwargs)
        _result_cache.put(cache_key, result)
        
        return result
    
    return wrapper

# ============================================================================
# INTERMEDIATE FORM CACHING (NEW)
# ============================================================================

class IntermediateFormCache:
    """
    Cache intermediate problem representations to reduce re-parsing.
    
    Stores:
    - Parsed structure
    - Extracted equations
    - Symbolic forms
    - Problem analysis
    """
    
    def __init__(self, max_size: int = 500):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.max_size = max_size
    
    def _hash_problem(self, problem_text: str) -> str:
        """Generate hash for problem."""
        return hashlib.md5(problem_text.strip().encode()).hexdigest()
    
    def get_parsed_form(self, problem_text: str) -> Optional[Dict[str, Any]]:
        """Get cached parsed form if available."""
        h = self._hash_problem(problem_text)
        if h in self.cache:
            return self.cache[h].get('parsed_form')
        return None
    
    def put_parsed_form(self, problem_text: str, parsed: Dict[str, Any]) -> None:
        """Cache parsed form."""
        h = self._hash_problem(problem_text)
        if h not in self.cache:
            self.cache[h] = {}
        self.cache[h]['parsed_form'] = parsed
        self._evict_if_full()
    
    def get_equations(self, problem_text: str) -> Optional[List[str]]:
        """Get cached equations if available."""
        h = self._hash_problem(problem_text)
        if h in self.cache:
            return self.cache[h].get('equations')
        return None
    
    def put_equations(self, problem_text: str, equations: List[str]) -> None:
        """Cache extracted equations."""
        h = self._hash_problem(problem_text)
        if h not in self.cache:
            self.cache[h] = {}
        self.cache[h]['equations'] = equations
        self._evict_if_full()
    
    def get_problem_analysis(self, problem_text: str) -> Optional[Dict[str, Any]]:
        """Get cached problem analysis."""
        h = self._hash_problem(problem_text)
        if h in self.cache:
            return self.cache[h].get('analysis')
        return None
    
    def put_problem_analysis(self, problem_text: str, analysis: Dict[str, Any]) -> None:
        """Cache problem analysis."""
        h = self._hash_problem(problem_text)
        if h not in self.cache:
            self.cache[h] = {}
        self.cache[h]['analysis'] = analysis
        self._evict_if_full()
    
    def get_intermediate_forms(self, problem_text: str) -> Optional[Dict[str, Any]]:
        """Get all cached intermediate forms."""
        h = self._hash_problem(problem_text)
        if h in self.cache:
            return self.cache[h]
        return None
    
    def _evict_if_full(self) -> None:
        """Simple eviction if cache is full."""
        if len(self.cache) >= self.max_size:
            # Remove oldest (first) entry
            oldest_key = next(iter(self.cache))
            self.cache.pop(oldest_key)
            logger.debug("Intermediate cache evicted oldest entry")


# Global instances
_result_cache = ResultCache()
_intermediate_cache = IntermediateFormCache()


def get_result_cache() -> ResultCache:
    """Get global result cache instance."""
    return _result_cache


def get_intermediate_cache() -> IntermediateFormCache:
    """Get global intermediate form cache instance."""
    return _intermediate_cache