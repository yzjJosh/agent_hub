"""Shared fixtures for the tests of the host scripts."""

import pytest

from tests import read_env_example


@pytest.fixture
def host_script_vars() -> set[str]:
    """The names of the host script entries in the repository's .env.example."""
    return read_env_example().host_scripts
