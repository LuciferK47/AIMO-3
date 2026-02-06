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

SYSTEM_PROMPT_PYTHON = """You are a Python expert solving math problems.
Write clean, efficient code that:
1. Solves the problem correctly
2. Outputs the final integer answer
3. Uses efficient algorithms
4. Includes error handling

Format: Output only the final integer answer on the last line.
"""

# ============================================================================
# DOMAIN-SPECIFIC PROMPTS
# ============================================================================

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
# ADVANCED FEATURES
# ============================================================================

# Feature flags
USE_PRM_FILTERING = True
USE_DYNAMIC_ALLOCATION = True
USE_OPTIMAL_WEIGHTING = True
USE_CACHING = True

# Cache configuration
CACHE_RESULT_EXPIRY = 3600  # seconds
MAX_CACHE_SIZE = 1000

# DEBUG flag for logging configuration load (avoid polluting Kaggle logs)
_DEBUG = False
if _DEBUG:
    print("✓ Configuration loaded successfully")
