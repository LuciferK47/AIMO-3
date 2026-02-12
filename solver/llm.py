"""
Single LLM inference interface.
Supports OpenAI API with deterministic settings.
"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Configuration
LLM_CLIENT = os.environ.get("LLM_CLIENT", "openai")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4-turbo")
LLM_API_KEY = os.environ.get("OPENAI_API_KEY") or os.environ.get("LLM_API_KEY")


def query_llm(prompt: str, max_tokens: int = 420, temperature: float = 0.0) -> Optional[str]:
    """
    Query LLM with deterministic settings.
    
    Args:
        prompt: Input prompt
        max_tokens: Maximum response length
        temperature: Sampling temperature (0.0 = deterministic)
    
    Returns:
        LLM response text or None if API fails
    """
    try:
        if LLM_CLIENT == "openai":
            return _query_openai(prompt, max_tokens, temperature)
        else:
            logger.error(f"Unsupported LLM client: {LLM_CLIENT}")
            return None
    except Exception as e:
        logger.error(f"LLM query failed: {e}")
        return None


def _query_openai(prompt: str, max_tokens: int, temperature: float) -> Optional[str]:
    """Query OpenAI API."""
    try:
        import openai
        
        if not LLM_API_KEY:
            logger.error("OPENAI_API_KEY not set")
            return None
        
        client = openai.OpenAI(api_key=LLM_API_KEY)
        
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        return None
