"""
Single canonical prompt for AIMO-3.
Enforces structured output and no-guessing policy.
"""

# Deterministic settings
TEMPERATURE = 0.0
MAX_TOKENS = 420

# Single canonical prompt (strict format enforcement)
CANONICAL_PROMPT = """You are a mathematical theorem prover for AIMO-3 competition.

CRITICAL RULES (MANDATORY):
1. Do NOT guess. If uncertain, state assumptions explicitly.
2. Provide structured derivation only.
3. Final answer must be an integer in [0, 99999].
4. Output must follow the exact format below.
5. Do NOT include any other text outside this format.

OUTPUT FORMAT (STRICT):
ASSUMPTIONS:
- [List any assumptions made, or write "None"]

DERIVATION:
- Step 1: [Clear mathematical step]
- Step 2: [Clear mathematical step]
- ...

FINAL_EXPRESSION:
[Explicit formula or expression]

FINAL_INTEGER:
[Single integer answer]

PROBLEM:
{problem}

BEGIN YOUR RESPONSE:
"""


def build_prompt(problem_text: str, retry_reason: str = None) -> str:
    """
    Build the prompt for LLM.
    
    Args:
        problem_text: Normalized problem text
        retry_reason: Optional reason for retry (format violation, verification failure)
    
    Returns:
        Complete prompt string
    """
    prompt = CANONICAL_PROMPT.format(problem=problem_text)
    
    if retry_reason:
        prompt += f"\n\nRETRY REASON: {retry_reason}\nCORRECT THE ISSUE AND FOLLOW THE FORMAT EXACTLY.\n"
    
    return prompt
