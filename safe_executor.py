"""
Safe execution sandbox for LLM-generated Python code.

Olympiad problems often require brute-force enumeration that LLMs can express
as Python code. This module provides sandboxed execution with:
- AST safety validation (whitelist approach)
- Timeout enforcement
- Restricted built-in functions
- Return value validation
"""

import ast
import logging
import math
import sys
from typing import Optional, List, Tuple, Any
from io import StringIO

try:
    import signal
except ImportError:
    signal = None

logger = logging.getLogger(__name__)

# Whitelisted safe built-in functions
SAFE_BUILTINS = {
    'len': len,
    'range': range,
    'sum': sum,
    'min': min,
    'max': max,
    'abs': abs,
    'int': int,
    'float': float,
    'str': str,
    'bool': bool,
    'list': list,
    'dict': dict,
    'set': set,
    'tuple': tuple,
    'zip': zip,
    'enumerate': enumerate,
    'sorted': sorted,
    'reversed': reversed,
    'any': any,
    'all': all,
    'pow': pow,
    'divmod': divmod,
    'round': round,
    'factorial': __import__('math').factorial,
    'gcd': __import__('math').gcd,
}

# Forbidden AST node types
FORBIDDEN_NODES = {
    ast.Import,      # No imports
    ast.ImportFrom,  # No relative imports
    # ast.Exec removed in Python 3.9+
    ast.Global,      # No global modifications (conservative)
    ast.Nonlocal,    # No nonlocal escapes
}


class ASTValidator(ast.NodeVisitor):
    """Validates that AST contains only safe operations."""
    
    def __init__(self):
        self.safe = True
        self.errors = []
    
    def visit(self, node):
        """Check node type before visiting."""
        if type(node) in FORBIDDEN_NODES:
            self.safe = False
            self.errors.append(f"Forbidden operation: {node.__class__.__name__}")
            return
        
        # Check for dangerous function calls
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in ['eval', 'exec', 'compile', '__import__', 'open', 'input']:
                    self.safe = False
                    self.errors.append(f"Forbidden function: {node.func.id}")
                    return
        
        # Check for attribute access (restrict to prevent __getattribute__ tricks)
        if isinstance(node, ast.Attribute):
            if node.attr.startswith('_'):
                self.safe = False
                self.errors.append(f"Forbidden attribute access: _{node.attr}")
                return
        
        self.generic_visit(node)
    
    def is_safe(self) -> bool:
        """Return whether AST is safe to execute."""
        return self.safe


def validate_code(code: str) -> Tuple[bool, List[str]]:
    """
    Validate Python code for safety.
    
    Args:
        code: Python source code to validate
        
    Returns:
        (is_safe, error_messages)
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, [f"Syntax error: {e}"]
    
    validator = ASTValidator()
    validator.visit(tree)
    
    return validator.safe, validator.errors


def safe_execute(code: str, 
                 timeout: float = 5.0,
                 expected_var: str = 'result') -> Optional[Any]:
    """
    Safely execute Python code in a restricted environment.
    
    Args:
        code: Python source code to execute
        timeout: Maximum execution time in seconds
        expected_var: Variable name to return (e.g., 'result', 'answer')
        
    Returns:
        Value of expected_var after execution, or None if execution fails
        
    Examples:
        >>> code = '''
        ... result = sum(i for i in range(100) if i % 2 == 0)
        ... '''
        >>> answer = safe_execute(code, timeout=1.0)
        >>> answer
        2450
    """
    # Validate code before execution
    is_safe, errors = validate_code(code)
    if not is_safe:
        logger.warning(f"Code failed safety check: {errors}")
        return None
    
    # Create restricted environment
    restricted_globals = {
        '__builtins__': SAFE_BUILTINS,
    }
    restricted_locals = {}
    
    try:
        # Execute with timeout (signal-based for Unix)
        if signal and hasattr(signal, 'SIGALRM'):
            def timeout_handler(signum, frame):
                raise TimeoutError(f"Code execution exceeded {timeout}s timeout")
            
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            # Use math.ceil to avoid truncating fractional timeouts (0.5s → 1s not 0s)
            signal.alarm(math.ceil(timeout) + 1)  # +1 for signal delay tolerance
            
            try:
                exec(code, restricted_globals, restricted_locals)
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
        else:
            # Fallback: execute without timeout (less safe on Windows)
            # In production, use concurrent.futures.ProcessPoolExecutor
            exec(code, restricted_globals, restricted_locals)
        
        # Extract result
        if expected_var in restricted_locals:
            result = restricted_locals[expected_var]
            logger.debug(f"Code execution successful: {expected_var} = {result}")
            return result
        else:
            logger.warning(f"Code executed but {expected_var} not found in locals")
            return None
            
    except TimeoutError as e:
        logger.warning(f"Code execution timeout: {e}")
        return None
    except Exception as e:
        logger.warning(f"Code execution failed: {type(e).__name__}: {e}")
        return None


def extract_integer_from_code(code: str, timeout: float = 5.0) -> Optional[int]:
    """
    Execute code and extract integer result.
    
    Tries common variable names: 'result', 'answer', 'ans', 'n', 'x'
    
    Args:
        code: Python source code
        timeout: Execution timeout
        
    Returns:
        Integer result, or None if execution fails or result is not integer
    """
    for var_name in ['result', 'answer', 'ans', 'n', 'x']:
        value = safe_execute(code, timeout=timeout, expected_var=var_name)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
    
    return None


# Test safe execution
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    # Test 1: Safe code
    safe_code = """
result = sum(i for i in range(10) if i % 2 == 0)
"""
    ans = safe_execute(safe_code)
    print(f"Test 1 (safe code): {ans}")  # Should be 20
    
    # Test 2: Unsafe code (should fail)
    unsafe_code = """
result = open('/etc/passwd').read()
"""
    ans = safe_execute(unsafe_code)
    print(f"Test 2 (unsafe code): {ans}")  # Should be None
    
    # Test 3: Timeout
    slow_code = """
result = sum(1 for _ in range(10**9))
"""
    ans = safe_execute(slow_code, timeout=1.0)
    print(f"Test 3 (timeout): {ans}")  # Should be None
