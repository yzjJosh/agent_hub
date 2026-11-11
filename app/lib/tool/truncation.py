DEFAULT_TOOL_OUTPUT_LEN_LIMIT = 25_000
MIN_TOOL_OUTPUT_LEN_LIMIT = 500

MAX_CHAR_DESCRIPTION = (
    "Maximum number of characters to return. The full content is stored "
    "regardless. When the returned content is longer it is truncated and the "
    "note names the task_id — fetch the remaining chars with "
    "task(action='result', task_id='<id>', offset=<max_char>, max_char=<remaining>)."
)

__all__ = [
    "DEFAULT_TOOL_OUTPUT_LEN_LIMIT",
    "MAX_CHAR_DESCRIPTION",
    "MIN_TOOL_OUTPUT_LEN_LIMIT",
    "truncate",
]


def truncate(result: str, params: dict, task_id: str) -> str:
    """The result cut to max_char, with a note saying how to read the rest."""
    max_char = params.get("max_char")
    if not isinstance(max_char, int) or isinstance(max_char, bool) or len(result) <= max_char:
        return result
    total = len(result)
    remaining = total - max_char
    return result[:max_char] + (
        f"\n[Truncated: showing the first {max_char} of {total} chars. "
        f"task_id={task_id} — fetch the remaining {remaining} chars with "
        f"task(action='result', task_id='{task_id}', offset={max_char}, max_char={remaining}).]"
    )
