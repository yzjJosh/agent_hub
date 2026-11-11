__all__ = ["split_prompt_message"]

_BAD_CODE = {
    "а": "a",
    "е": "e",
    "e": "e",
    "i": "i",
    "і": "i",
    "ο": "o",
    "с": "c",
    "ԁ": "d",
    "ѕ": "s",
    "һ": "h",
    "у": "y",
    "р": "p",
    "ϳ": "j",
    "х": "x",
}


def _label_cleaning(raw_label: str) -> str:
    """Swap look alike letters for plain ones."""
    clean_label = raw_label
    for c in _BAD_CODE:
        clean_label = clean_label.replace(c, _BAD_CODE[c])
    return clean_label


def split_prompt_message(label: str) -> str:
    """Pull the thing to look for out of the prompt."""
    if "with" not in label:
        unclean_label = label.strip()
    elif " a " in label:
        unclean_label = label.split("with a ")[1].split()[0]
    else:
        unclean_label = label.split("with")[1].split()[0]

    return _label_cleaning(unclean_label)
