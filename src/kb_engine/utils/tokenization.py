"""Tokenization utilities.

Uses a simple word-based estimation when tiktoken is not available.
"""


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """Count the approximate number of tokens in text.

    Uses tiktoken if available, otherwise falls back to a simple
    word-based estimation (1 token ~ 0.75 words for English).
    """
    try:
        import tiktoken

        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except (ImportError, KeyError):
        # Rough estimation: ~4 chars per token on average
        return max(1, len(text) // 4)


def truncate_to_tokens(text: str, max_tokens: int, model: str = "gpt-4") -> str:
    """Truncate text to a maximum number of tokens."""
    try:
        import tiktoken

        encoding = tiktoken.encoding_for_model(model)
        tokens = encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return encoding.decode(tokens[:max_tokens])
    except (ImportError, KeyError):
        # Rough estimation: ~4 chars per token
        max_chars = max_tokens * 4
        if len(text) <= max_chars:
            return text
        return text[:max_chars]
