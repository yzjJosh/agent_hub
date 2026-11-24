from collections.abc import AsyncIterator

import pytest

from tests.app.conftest import TaskRecordStore


@pytest.fixture(autouse=True)
async def tool_infra(task_store: TaskRecordStore) -> AsyncIterator[TaskRecordStore]:
    """Give every mcp test a throwaway record store."""
    yield task_store
