from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from tests.app.conftest import store_infra


@pytest.fixture(autouse=True)
async def agent_store_infra(tmp_path: Path) -> AsyncIterator[None]:
    """Point the process-wide stores at a throwaway database for the test."""
    async for _ in store_infra(tmp_path):
        yield
