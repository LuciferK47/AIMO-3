"""
Shared utilities for AIMO-3 pipeline.
"""

import re
import json
import logging
import threading
import _thread
from typing import List, Set, Optional, Tuple, Any, Dict
from collections import Counter
import functools
from contextlib import contextmanager

logger = logging.getLogger(__name__)


def preprocess_problem_text(latex_text: str) -> str:
    """
    Preprocess problem text to convert confusing LaTeX notation 
    into explicit natural language descriptions for LLM clarity.
    
    Conversions:
    - \\overline{xyz} -> "(the integer formed by digits xyz)"
    - Helps LLMs interpret digit notation correctly
    """
    # Pattern: \overline{xyz} -> "the integer formed by digits xyz"
    # Matches \overline followed by content in curly braces
    pattern = r'\\overline\{([a-zA-Z0-9]+)\}'
    
    def replace_overline(match):
        content = match.group(1)
        # Clarify it's digits, not multiplication
        return f"(the integer formed by digits {content})"
    
    processed_text = re.sub(pattern, replace_overline, latex_text)
    
    return processed_text


def ensure_determinism(seed: int = 42) -> None:
    """Ensure deterministic behavior across common RNGs."""
    try:
        import random
        random.seed(seed)
    except Exception:
        pass
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass


@contextmanager
def time_limit(seconds: float):
    """
    Context manager for enforcing hard time limits on code blocks.
    
    Usage:
        with time_limit(2.0):
            do_something()
    
    Raises TimeoutError if time limit exceeded.
    """
    if seconds is None or seconds <= 0:
        yield
        return

    timed_out = {"flag": False}

    def interrupt_main():
        timed_out["flag"] = True
        try:
            _thread.interrupt_main()
        except Exception:
            pass

    timer = threading.Timer(seconds, interrupt_main)
    timer.daemon = True
    timer.start()

    try:
        yield
    except KeyboardInterrupt:
        if timed_out["flag"]:
            raise TimeoutError(f"Operation exceeded {seconds}s time limit")
        raise
    finally:
        timer.cancel()



def extract_integers(text: str) -> List[int]:
    """Extract all integers from text output. Clamped to [0, 99999]."""
    if not text:
        return []
    
    answers = []
    
    boxed_match = re.search(r'\\boxed\{(-?\d+)\}', text)
    if boxed_match:
        val = int(boxed_match.group(1))
        if 0 <= val <= 99999:
            answers.append(val)
    
    answer_match = re.search(r'(?:answer|result|solution)[\s:]*(-?\d+)', text, re.IGNORECASE)
    if answer_match:
        val = int(answer_match.group(1))
        if 0 <= val <= 99999:
            answers.append(val)
    
    all_numbers = re.findall(r'-?\d+', text)
    if all_numbers:
        last_num = int(all_numbers[-1])
        if 0 <= last_num <= 99999:
            answers.append(last_num)
    
    valid_numbers = []
    for num_str in re.findall(r'\d+', text):
        num = int(num_str)
        if 0 <= num <= 99999:
            valid_numbers.append(num)
    
    seen = set()
    unique_answers = []
    for ans in answers + valid_numbers:
        if ans not in seen:
            unique_answers.append(ans)
            seen.add(ans)
    
    return unique_answers if unique_answers else [0]


def clamp_to_range(value: Any, min_val: int = 0, max_val: int = 99999) -> int:
    """Convert value to integer and clamp to [min_val, max_val]."""
    try:
        val = int(float(str(value).strip()))
        return max(min_val, min(max_val, val))
    except (ValueError, TypeError, AttributeError):
        return min_val


def find_most_common(answers: List[int], weights: Optional[List[float]] = None) -> int:
    """Find most common answer from list."""
    if not answers:
        return 0
    
    if weights is None:
        weights = [1.0] * len(answers)
    
    counter = Counter()
    for ans, weight in zip(answers, weights):
        counter[ans] += weight
    
    if counter:
        return counter.most_common(1)[0][0]
    return answers[0]


def select_diverse_pair(answers: List[int], 
                        confidence_scores: Optional[List[float]] = None) -> Tuple[int, int]:
    """Select two answers for double-run evaluation."""
    if not answers:
        return (0, 0)
    
    if confidence_scores is None:
        confidence_scores = [1.0] * len(answers)
    
    weighted = [(ans, conf) for ans, conf in zip(answers, confidence_scores)]
    weighted.sort(key=lambda x: x[1], reverse=True)
    
    unique_answers = list(dict.fromkeys([ans for ans, _ in weighted]))
    
    if len(unique_answers) == 0:
        return (0, 0)
    elif len(unique_answers) == 1:
        return (unique_answers[0], unique_answers[0])
    else:
        return (unique_answers[0], unique_answers[1])


def normalize_latex(text: str) -> str:
    """Remove or normalize LaTeX markup."""
    text = text.replace('\\.', '.')
    text = text.replace('\\,', ',')
    text = text.replace('\\:', ':')
    text = text.replace('\\!', '')
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


def safe_timeout(func, timeout_seconds: float = 10.0, default_return: Any = None):
    """Execute function with timeout. Returns default_return if timeout occurs."""
    import threading
    import queue
    
    result_queue = queue.Queue()
    exception_queue = queue.Queue()
    
    def wrapper():
        try:
            result = func()
            result_queue.put(result)
        except Exception as e:
            exception_queue.put(e)
    
    thread = threading.Thread(target=wrapper, daemon=True)
    thread.start()
    thread.join(timeout=timeout_seconds)
    
    if thread.is_alive():
        logger.warning(f"Function timed out after {timeout_seconds}s")
        return default_return
    
    if not exception_queue.empty():
        e = exception_queue.get()
        logger.error(f"Function raised exception: {e}")
        return default_return
    
    if not result_queue.empty():
        return result_queue.get()
    
    return default_return


def agreement_score(answers: List[int]) -> float:
    """Compute how much agreement there is among answers (0-1)."""
    if not answers:
        return 0.0
    if len(answers) == 1:
        return 1.0
    counter = Counter(answers)
    max_count = max(counter.values())
    return max_count / len(answers)


def query_llm(prompt: str, max_tokens: int = 500, temperature: float = 0.2) -> Optional[str]:
    """
    Query LLM with disciplined prompt.
    
    MASTER PROMPT DISCIPLINE:
    - Force equation-only output
    - No prose
    - Explicit integer answers
    
    Args:
        prompt: Disciplined prompt (should enforce equation/answer format)
        max_tokens: Maximum response length
        temperature: Sampling temperature (0.2 = deterministic, 0.5 = creative)
    
    Returns:
        LLM response text (may be None if API fails)
    """
    try:
        # Import LLM client (uses config.py settings)
        from config import LLM_CLIENT, LLM_MODEL, LLM_API_KEY
        
        if not LLM_API_KEY:
            logger.warning("LLM_API_KEY is not set; skipping LLM call")
            return None

        if LLM_CLIENT == "openai":
            try:
                import openai
                openai.api_key = LLM_API_KEY
                
                response = openai.ChatCompletion.create(
                    model=LLM_MODEL,
                    messages=[
                        {"role": "system", "content": "You are a math formalizer. Output only equations or expressions. No explanation."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=0.95,
                )
                
                return response.choices[0].message.content.strip()
            except Exception as e:
                logger.debug(f"OpenAI query failed: {e}")
                return None
        
        elif LLM_CLIENT == "anthropic":
            try:
                import anthropic
                client = anthropic.Anthropic(api_key=LLM_API_KEY)
                
                response = client.messages.create(
                    model=LLM_MODEL,
                    max_tokens=max_tokens,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                )
                
                return response.content[0].text.strip()
            except Exception as e:
                logger.debug(f"Anthropic query failed: {e}")
                return None
        
        else:
            logger.warning(f"Unknown LLM client: {LLM_CLIENT}")
            return None
    
    except Exception as e:
        logger.debug(f"LLM query error: {e}")
        return None


def parse_json_response(text: str) -> Optional[Dict[str, Any]]:
    """Parse the first valid JSON object from text."""
    if not text:
        return None
    try:
        start = text.find('{')
        end = text.rfind('}')
        if start == -1 or end == -1 or end <= start:
            return None
        return json.loads(text[start:end + 1])
    except Exception:
        return None


def query_llm_json(prompt: str, max_tokens: int = 300, temperature: float = 0.1) -> Optional[Dict[str, Any]]:
    """Query LLM and parse a JSON response."""
    response = query_llm(prompt, max_tokens=max_tokens, temperature=temperature)
    return parse_json_response(response)

