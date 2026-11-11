"""Tests for the set of thumbnails being made right now."""

from app.lib.render.gen_state import gen_add, gen_remove, generating_keys


def test_generating_set_adds_and_drops_keys() -> None:
    """A key is listed while it is being made and gone once it is dropped."""
    key = "e" * 64
    gen_add(key)
    assert key in generating_keys()
    gen_remove(key)
    gen_remove(key)  # dropping it twice is fine
    assert key not in generating_keys()


def test_generating_keys_is_a_snapshot() -> None:
    """The list handed out does not change when the set does."""
    key = "a" * 64
    gen_add(key)
    try:
        snapshot = generating_keys()
        gen_remove(key)
        assert key in snapshot  # the snapshot kept what it had
        assert key not in generating_keys()
    finally:
        gen_remove(key)
