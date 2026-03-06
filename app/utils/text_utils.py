"""Tokenization and text processing helpers."""

import re
from typing import Optional


def count_tokens_approx(text: str) -> int:
    """Approximate token count using whitespace + punctuation splitting.

    Rough heuristic: ~1.3 tokens per word for English text.
    """
    words = len(text.split())
    return int(words * 1.3)


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """Truncate text to max character length."""
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


def clean_text(text: str) -> str:
    """Basic text cleaning: normalize whitespace, strip."""
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def format_alpaca_prompt(
    instruction: str,
    input_text: Optional[str] = None,
    output: Optional[str] = None,
) -> str:
    """Format an Alpaca-style prompt."""
    if input_text and input_text.strip():
        prompt = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            f"### Response:\n"
        )
    else:
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"

    if output:
        prompt += output

    return prompt


def format_chat_prompt(messages: list[dict[str, str]]) -> str:
    """Format a list of chat messages into a prompt string."""
    parts = []
    for msg in messages:
        role = msg.get("role", msg.get("from", "user"))
        content = msg.get("content", msg.get("value", ""))

        if role in ("system",):
            parts.append(f"<|system|>\n{content}")
        elif role in ("user", "human"):
            parts.append(f"<|user|>\n{content}")
        elif role in ("assistant", "gpt"):
            parts.append(f"<|assistant|>\n{content}")

    return "\n".join(parts)


def extract_model_size(model_name: str) -> Optional[str]:
    """Extract model size from model name (e.g., '7b' from 'mistral-7b')."""
    match = re.search(r"(\d+\.?\d*)[bB]", model_name)
    if match:
        return match.group(0).upper()
    return None
