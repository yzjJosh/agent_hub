"""A check that a URL string is well formed."""

__all__ = ["validate_url"]

from urllib.parse import urlparse


def validate_url(url: str) -> str | None:
    """Check that a string is a well formed URL.

    Args:
        url: The string to check.

    Returns:
        An error message when the URL is bad, or None when it is fine.
    """
    parsed = urlparse(url)
    if not parsed.scheme:
        return f"Invalid URL: '{url}' is missing a scheme"
    if not parsed.netloc:
        return f"Invalid URL: '{url}' is missing a hostname"
    return None
