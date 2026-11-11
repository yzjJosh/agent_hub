"""How many tokens a piece of text takes."""

__all__ = ["count_tokens"]

import tiktoken


def count_tokens(text: str) -> int:
    """Count how many tokens a text takes.

    Args:
        text: The text to measure.

    Returns:
        The number of tokens.
    """
    if not text:
        return 0
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception:
        pass
    return len(text) // 4
