from collections.abc import AsyncIterator
from pathlib import Path

import pytest

import app.lib.db.store as store_module


@pytest.fixture(autouse=True)
async def task_store_infra(tmp_path: Path) -> AsyncIterator[None]:
    """Point the process-wide task store at a throwaway database for the test."""
    previous = store_module.task_store
    store_module.init_store(str(tmp_path / "records.db"))
    await store_module.task_store.open()
    yield
    await store_module.task_store.close()
    store_module.task_store = previous
