from collections.abc import AsyncIterator
from typing import Protocol

import pytest


class _TaskStore(Protocol):
    """The task record store the fixture gives to a test."""


@pytest.fixture(autouse=True)
async def tool_infra(task_store: _TaskStore) -> AsyncIterator[_TaskStore]:
    """Give every mcp test a throwaway record store."""
    yield task_store
