"""
Simplified caching for AIMO-3.

Keeps a single ResultCache plus a minimal IntermediateCache
for parsed forms and equations used by arbitration.
"""

import logging
import hashlib
import json
from typing import Any, Optional, Dict, Callable, List
from functools import wraps

logger = logging.getLogger(__name__)


class ResultCache:
    """Simple in-memory cache for problem solutions."""

    def __init__(self, max_size: int = 1000):
        self.cache: Dict[str, Any] = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    def _hash_key(self, key: str) -> str:
        return hashlib.md5(key.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        hash_key = self._hash_key(key)
        if hash_key in self.cache:
            self.hits += 1
            logger.debug("Cache hit")
            return self.cache[hash_key]
        self.misses += 1
        return None

    def put(self, key: str, value: Any) -> None:
        if len(self.cache) >= self.max_size:
            self.cache.pop(next(iter(self.cache)))
            logger.debug("Cache full; evicted oldest item")
        hash_key = self._hash_key(key)
        self.cache[hash_key] = value

    def clear(self) -> None:
        self.cache.clear()

    def stats(self) -> Dict[str, Any]:
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0.0
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'total_requests': total_requests,
        }


class IntermediateCache:
    """Minimal cache for parsed forms and equations."""

    def __init__(self, max_size: int = 500):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.max_size = max_size

    def _hash_problem(self, problem_text: str) -> str:
        return hashlib.md5(problem_text.strip().encode()).hexdigest()

    def _get_entry(self, problem_text: str) -> Dict[str, Any]:
        h = self._hash_problem(problem_text)
        return self.cache.get(h, {})

    def _set_entry(self, problem_text: str, key: str, value: Any) -> None:
        h = self._hash_problem(problem_text)
        if h not in self.cache:
            self.cache[h] = {}
        self.cache[h][key] = value
        self._evict_if_full()

    def get_parsed_form(self, problem_text: str) -> Optional[Dict[str, Any]]:
        return self._get_entry(problem_text).get('parsed_form')

    def put_parsed_form(self, problem_text: str, parsed: Dict[str, Any]) -> None:
        self._set_entry(problem_text, 'parsed_form', parsed)

    def get_equations(self, problem_text: str) -> Optional[List[str]]:
        return self._get_entry(problem_text).get('equations')

    def put_equations(self, problem_text: str, equations: List[str]) -> None:
        self._set_entry(problem_text, 'equations', equations)

    def get_problem_analysis(self, problem_text: str) -> Optional[Dict[str, Any]]:
        return self._get_entry(problem_text).get('analysis')

    def put_problem_analysis(self, problem_text: str, analysis: Dict[str, Any]) -> None:
        self._set_entry(problem_text, 'analysis', analysis)

    def get_intermediate_forms(self, problem_text: str) -> Optional[Dict[str, Any]]:
        h = self._hash_problem(problem_text)
        return self.cache.get(h)

    def _evict_if_full(self) -> None:
        if len(self.cache) >= self.max_size:
            oldest_key = next(iter(self.cache))
            self.cache.pop(oldest_key)
            logger.debug("Intermediate cache evicted oldest entry")


def cache_result(func: Callable) -> Callable:
    """Decorator to cache function results using ResultCache."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        cache_key = json.dumps({'args': args, 'kwargs': kwargs}, default=str)
        cached = get_result_cache().get(cache_key)
        if cached is not None:
            return cached
        result = func(*args, **kwargs)
        get_result_cache().put(cache_key, result)
        return result

    return wrapper


_result_cache = ResultCache()
_intermediate_cache = IntermediateCache()


def get_result_cache() -> ResultCache:
    return _result_cache


def get_intermediate_cache() -> IntermediateCache:
    return _intermediate_cache


class ResultCache:
    """Simple in-memory cache for problem solutions."""

    def __init__(self, max_size: int = 1000):
        self.cache: Dict[str, Any] = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    def _hash_key(self, key: str) -> str:
        return hashlib.md5(key.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        hash_key = self._hash_key(key)
        if hash_key in self.cache:
            self.hits += 1
            logger.debug("Cache hit")
            return self.cache[hash_key]
        self.misses += 1
        return None

    def put(self, key: str, value: Any) -> None:
        if len(self.cache) >= self.max_size:
            self.cache.pop(next(iter(self.cache)))
            logger.debug("Cache full; evicted oldest item")
        hash_key = self._hash_key(key)
        self.cache[hash_key] = value

    def clear(self) -> None:
        self.cache.clear()

    def stats(self) -> Dict[str, Any]:
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0.0
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'total_requests': total_requests,
        }


class IntermediateCache:
    """Minimal cache for parsed forms and equations."""

    def __init__(self, max_size: int = 500):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.max_size = max_size

    def _hash_problem(self, problem_text: str) -> str:
        return hashlib.md5(problem_text.strip().encode()).hexdigest()

    def _get_entry(self, problem_text: str) -> Dict[str, Any]:
        h = self._hash_problem(problem_text)
        return self.cache.get(h, {})

    def _set_entry(self, problem_text: str, key: str, value: Any) -> None:
        h = self._hash_problem(problem_text)
        if h not in self.cache:
            self.cache[h] = {}
        self.cache[h][key] = value
        self._evict_if_full()

    def get_parsed_form(self, problem_text: str) -> Optional[Dict[str, Any]]:
        return self._get_entry(problem_text).get('parsed_form')

    def put_parsed_form(self, problem_text: str, parsed: Dict[str, Any]) -> None:
        self._set_entry(problem_text, 'parsed_form', parsed)

    def get_equations(self, problem_text: str) -> Optional[List[str]]:
        return self._get_entry(problem_text).get('equations')

    def put_equations(self, problem_text: str, equations: List[str]) -> None:
        self._set_entry(problem_text, 'equations', equations)

    def get_problem_analysis(self, problem_text: str) -> Optional[Dict[str, Any]]:
        return self._get_entry(problem_text).get('analysis')

    def put_problem_analysis(self, problem_text: str, analysis: Dict[str, Any]) -> None:
        self._set_entry(problem_text, 'analysis', analysis)

    def get_intermediate_forms(self, problem_text: str) -> Optional[Dict[str, Any]]:
        h = self._hash_problem(problem_text)
        return self.cache.get(h)

    def _evict_if_full(self) -> None:
        if len(self.cache) >= self.max_size:
            oldest_key = next(iter(self.cache))
            self.cache.pop(oldest_key)
            logger.debug("Intermediate cache evicted oldest entry")


def cache_result(func: Callable) -> Callable:
    """Decorator to cache function results using ResultCache."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        cache_key = json.dumps({'args': args, 'kwargs': kwargs}, default=str)
        cached = get_result_cache().get(cache_key)
        if cached is not None:
            return cached
        result = func(*args, **kwargs)
        get_result_cache().put(cache_key, result)
        return result

    return wrapper


_result_cache = ResultCache()
_intermediate_cache = IntermediateCache()


def get_result_cache() -> ResultCache:
    return _result_cache


def get_intermediate_cache() -> IntermediateCache:
    return _intermediate_cache