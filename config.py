"""
Centralized configuration for AIMO-3 solver.
Consolidates: models.py, configuration.py, prompt_lib.py, constants
"""

import os

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

DEFAULT_MODEL = "Qwen/Qwen2.5-Math-14B-Instruct"
ALTERNATIVE_MODELS = {
    "qwen_7b": "Qwen/Qwen2.5-Math-7B-Instruct",
    "qwen_14b": "Qwen/Qwen2.5-Math-14B-Instruct",
    "deepseek_distill": "DeepSeek-R1-Distill-14B",
    "openai_gpt4": "gpt-4-turbo",
}

# ============================================================================
# LLM CONFIGURATION (REQUIRED)
# ============================================================================

# Supported values: "openai", "anthropic", "huggingface"
# Use "huggingface" for Kaggle offline mode with local Qwen models
LLM_CLIENT = os.environ.get("LLM_CLIENT", "openai")
# Default model for the selected client
# For huggingface: "Qwen/Qwen2.5-Math-7B-Instruct" or "Qwen/Qwen2.5-Math-14B-Instruct"
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4-turbo")
# API key sourced from environment (not needed for huggingface local models)
LLM_API_KEY = (
    os.environ.get("OPENAI_API_KEY")
    or os.environ.get("ANTHROPIC_API_KEY")
    or os.environ.get("LLM_API_KEY")
)

# ============================================================================
# HYPERPARAMETERS
# ============================================================================

# CRITICAL: Set to 0.0 for reproducibility on double-run format
# Temperature > 0 introduces variance across submissions
LLM_TEMPERATURE_DETERMINISTIC = 0.0  # For all LLM calls (deterministic)
MAX_TOKENS_EQUATION = 300
MAX_TOKENS_PYTHON = 256
MAX_TOKENS_ENUM = 200

# Timeouts
TIMEOUT_PER_STRATEGY = 5.0  # Maximum time per strategy
TIMEOUT_PYTHON_EXECUTION = 4.0
TIMEOUT_SYMPY_SOLVE = 2.0
TIMEOUT_GEOMETRY_ATTEMPT = 2.0
DEFAULT_TIMEOUT_PER_PROBLEM = 30.0  # Total time budget

# Answer validation
MIN_ANSWER = 0
MAX_ANSWER = 99999

# ============================================================================
# MASTER PROMPT HARD LIMITS (Determinism > Cleverness)
# ============================================================================

# Candidate generation limits
MAX_CANDIDATES_PER_PROBLEM = 5  # Hard cap on candidates generated
MAX_RETRIES_PER_STRATEGY = 1    # Max retries if strategy fails (reduced from 2)

# Symbolic solving limits
MAX_SYMBOLIC_DEPTH = 10         # Max recursion depth for symbolic solving
MAX_EQUATION_COMPLEXITY = 5     # Max number of equations in a system

# LLM limits
MAX_LLM_CALLS_PER_PROBLEM = 6   # Total LLM API calls: 2 per strategy × 3 strategies

# Time budgets per stage
TIME_BUDGET_PARSE = 0.5         # seconds
TIME_BUDGET_CLASSIFY = 0.2      # seconds
TIME_BUDGET_GENERATE = 12.0     # seconds (main work)
TIME_BUDGET_VERIFY = 3.0        # seconds
TIME_BUDGET_ARBITRATE = 1.0     # seconds

# Random seed for reproducibility
RANDOM_SEED = 42  # Fixed seed for deterministic behavior
USE_FIXED_SEED = True  # Set to False for exploration mode

# ============================================================================
# SYSTEM PROMPTS
# ============================================================================

SYSTEM_PROMPT_MATH = """You are a mathematical problem-solving expert competing in AIMO-3.
Your task is to solve integer-based mathematical problems from various domains:
- Algebra: Equations, polynomials, systems
- Number Theory: Modular arithmetic, divisibility, primes
- Geometry: Coordinate geometry, angles, areas
- Combinatorics: Counting, permutations, graph theory

CRITICAL RULES:
1. Think step-by-step
2. Show all reasoning
3. Final answer must be an integer in [0, 99999]
4. Format your final answer as: "The answer is [number]"
5. Verify your answer satisfies all problem constraints
"""

SYSTEM_PROMPT_EQUATION = """You are an equation extractor for AIMO-3 competition.

TASK: Extract the core mathematical relationship as Python-parseable equations.

OUTPUT FORMAT (strict JSON only, no other text):
{"equations": ["x**2 - 5*x + 6 = 0"], "variables": ["x"]}

SYNTAX RULES:
- Exponentiation: ** (NOT ^)
- Multiplication: * (explicit, e.g., 2*x not 2x)
- Fractions: use a/b or Rational(a, b)
- Square root: sqrt(...)
- Exactly ONE equals sign per equation

WORKED EXAMPLES:

Problem: "Find x if x squared plus three equals twelve."
Output: {"equations": ["x**2 + 3 = 12"], "variables": ["x"]}

Problem: "If 2x - 5 = 13, what is x?"
Output: {"equations": ["2*x - 5 = 13"], "variables": ["x"]}

Problem: "Find the remainder when x^3 + 2x + 1 is divided by x - 2."
Output: {"equations": ["x**3 + 2*x + 1 = (x - 2)*q + r"], "variables": ["x", "q", "r"]}

NOW EXTRACT FROM:
{problem}
"""

SYSTEM_PROMPT_PYTHON = """Generate Python code that computes the answer.

TEMPLATE (MUST follow exactly):
```python
# Functions available: gcd, factorial, comb, isqrt, min, max, sum, range, list, set
# NO imports allowed
# NO recursion depth > 100

# [Your computation here - max 30 lines]

result = <integer_answer>  # MUST be an integer in [0, 99999]
```

WORKED EXAMPLES:

Problem: "How many divisors does 120 have?"
Code:
```python
n = 120
count = 0
for i in range(1, n + 1):
    if n % i == 0:
        count += 1
result = count
```

Problem: "Find the last two digits of 7^100."
Code:
```python
result = pow(7, 100, 100)
```

Problem: "Find the sum of the first 100 positive integers."
Code:
```python
result = 100 * 101 // 2
```

NOW SOLVE:
{problem}
"""

# ============================================================================
# PREAMBLE CODE (Execution setup)
# ============================================================================

PREAMBLE_CODE = """
import math
from fractions import Fraction
from math import gcd, lcm, factorial, comb, sqrt, sin, cos, tan, pi
from collections import Counter, defaultdict, deque

def floor(x):
    return math.floor(x)

def ceil(x):
    return math.ceil(x)

def frac(x):
    return x - floor(x)

# Additional math utilities
def mod_inverse(a, m):
    def extended_gcd(a, b):
        if a == 0:
            return b, 0, 1
        gcd_val, x1, y1 = extended_gcd(b % a, a)
        x = y1 - (b // a) * x1
        y = x1
        return gcd_val, x, y
    
    _, x, _ = extended_gcd(a % m, m)
    return (x % m + m) % m

def euler_phi(n):
    result = n
    p = 2
    while p * p <= n:
        if n % p == 0:
            while n % p == 0:
                n //= p
            result -= result // p
        p += 1
    if n > 1:
        result -= result // n
    return result
"""

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# ============================================================================
# CACHING CONFIGURATION
# ============================================================================

USE_CACHING = True
CACHE_RESULT_EXPIRY = 3600  # seconds
MAX_CACHE_SIZE = 1000
# ============================================================================
# CONFIDENCE THRESHOLDS (Principled Scoring)
# ============================================================================

CONFIDENCE_THRESHOLDS = {
    'AGREEMENT_HIGH': 0.70,      # 70%+ of candidates agree → return (best, best)
    'AGREEMENT_MEDIUM': 0.50,    # 50-70% → return (best, second_best)
    'AGREEMENT_LOW': 0.30,       # <30% → uncertain, use diversity
    
    'SYMBOLIC_VERIFIED': 0.95,   # Answer verified by equation substitution
    'PYTHON_EXECUTED': 0.85,     # Answer from safe_execute
    'LLM_DIRECT': 0.60,          # Answer from LLM text extraction only
    'FALLBACK': 0.20,            # Guessed/default answer
}