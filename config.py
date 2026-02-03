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

# Supported values: "openai", "anthropic"
LLM_CLIENT = os.environ.get("LLM_CLIENT", "openai")
# Default model for the selected client
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4-turbo")
# API key sourced from environment
LLM_API_KEY = (
    os.environ.get("OPENAI_API_KEY")
    or os.environ.get("ANTHROPIC_API_KEY")
    or os.environ.get("LLM_API_KEY")
)

# ============================================================================
# HYPERPARAMETERS
# ============================================================================

# Generation parameters
NUM_REASONING_ATTEMPTS = 4
NUM_PYTHON_ATTEMPTS = 2
NUM_DECOMPOSED_ATTEMPTS = 1

TEMPERATURE_RANGE = (0.3, 1.0)
MAX_TOKENS_REASONING = 512
MAX_TOKENS_PYTHON = 256
MAX_TOKENS_DECOMPOSED = 512

# Timeouts
TIMEOUT_PER_ATTEMPT = 6.0
TIMEOUT_PYTHON_EXECUTION = 4.0
TIMEOUT_GEOMETRY_ATTEMPT = 2.0
DEFAULT_TIMEOUT_PER_PROBLEM = 60.0

# PRM Configuration
PRM_QUALITY_THRESHOLD = 0.6
PRM_AGGREGATION_METHOD = "min"  # mean, min, harmonic_mean, product

# Resource Manager Configuration
TOTAL_COMPETITION_BUDGET = 18000.0  # 5 hours
BASE_BUDGET_PER_PROBLEM = 180.0    # 3 minutes
DIFFICULTY_EASY_THRESHOLD = 0.3
DIFFICULTY_HARD_THRESHOLD = 0.7
EASY_MULTIPLIER = 0.5
HARD_MULTIPLIER = 2.0
CONSENSUS_THRESHOLD = 0.8
MAX_BUDGET_FRACTION = 0.4

# Ensemble Configuration
PYTHON_EXECUTION_WEIGHT = 3.0
DEFAULT_MODEL_RELIABILITY = 0.5
SYMPY_RELIABILITY = 0.90
PYTHON_RELIABILITY = 0.85
DECOMPOSED_RELIABILITY = 0.70
COT_RELIABILITY = 0.65
GEOMETRY_RELIABILITY = 0.60

# Answer validation
MIN_ANSWER = 0
MAX_ANSWER = 99999

# ============================================================================
# MASTER PROMPT HARD LIMITS (Determinism > Cleverness)
# ============================================================================

# Candidate generation limits
MAX_CANDIDATES_PER_PROBLEM = 5  # Hard cap on candidates generated
MAX_RETRIES_PER_STRATEGY = 2    # Max retries if strategy fails

# Symbolic solving limits
MAX_SYMBOLIC_DEPTH = 10         # Max recursion depth for symbolic solving
MAX_EQUATION_COMPLEXITY = 5     # Max number of equations in a system

# LLM limits
MAX_LLM_CALLS_PER_PROBLEM = 10  # Total LLM API calls allowed per problem
LLM_TEMPERATURE_DETERMINISTIC = 0.1  # For translator mode (low randomness)
LLM_TEMPERATURE_CREATIVE = 0.3       # For case enumeration (slight variation)

# Time budgets per stage
TIME_BUDGET_PARSE = 0.5         # seconds
TIME_BUDGET_CLASSIFY = 0.2      # seconds
TIME_BUDGET_GENERATE = 15.0     # seconds (main work)
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

ALGEBRAIC_DECOMPOSITION_PROMPT = """Solve this algebra problem step-by-step:
1. Identify variables and unknowns
2. Write all equations/constraints
3. Simplify each equation
4. Solve the system
5. Verify the solution satisfies all constraints
6. State the final integer answer

Problem: {problem}

SOLUTION:"""

NUMBER_THEORY_DECOMPOSITION_PROMPT = """Solve this number theory problem:
1. Identify what we're looking for (count, value, property)
2. Apply relevant theorems (Chinese Remainder, Fermat's Little, etc.)
3. Use modular arithmetic where applicable
4. Break into cases if needed
5. Count or compute the final answer

Problem: {problem}

SOLUTION:"""

GEOMETRY_DECOMPOSITION_PROMPT = """Solve this geometry problem:
1. Set up a coordinate system if helpful
2. Identify key distances/angles
3. Use distance formula, law of cosines, etc.
4. Solve for the target quantity
5. Verify using alternative method if possible

Problem: {problem}

SOLUTION:"""

# ============================================================================
# GEOMETRY HINTS
# ============================================================================

GEOMETRY_NUMERIC_PROMPT = """For this geometry problem, assume convenient coordinates and compute numerically.

Problem: {problem}

Use coordinates like:
- Unit circle or simple fractions for angles
- Integer or simple decimal coordinates for points
- Standard orientations for shapes

Compute the final numeric answer step-by-step.
Final answer (integer): """

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

print("✓ Configuration loaded successfully")
