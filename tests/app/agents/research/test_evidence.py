from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import app.lib.db.store as store_module
from app.agents.research.evidence import EvidenceStore, SearchEvidence
from app.lib.shared.config import MissingSettingError
from app.mcp.registry import tool_class
from tests.app.lib.tool.conftest import wait_terminal

_EMBEDDING_URL = "http://embed.test:8888"
_RERANKER_URL = "http://rerank.test:8888"
_VECTOR_DIM = 1024
_OPTIONAL_VARS = ("EMBEDDING_MODEL", "EMBEDDING_API_KEY", "RERANKER_MODEL", "RERANKER_API_KEY")
_FAKE_VECTOR = [0.1] * _VECTOR_DIM
_TOP_HIT = [{"index": 0, "score": 0.9}]


def _fake_embedding_client(vectors: list[list[float]]) -> MagicMock:
    """A fake AsyncOpenAI whose embeddings.create returns the vectors in order."""
    responses = []
    for vector in vectors:
        item = MagicMock()
        item.embedding = vector
        response = MagicMock()
        response.data = [item]
        responses.append(response)
    client = MagicMock()
    client.embeddings.create = AsyncMock(side_effect=responses)
    return client


def _fake_rerank_http(results: list[dict]) -> MagicMock:
    """A fake httpx module whose AsyncClient.post answers a rerank payload."""
    httpx_stub, _ = _fake_rerank_http_with_post(results)
    return httpx_stub


def _fake_rerank_http_with_post(results: list[dict]) -> tuple[MagicMock, AsyncMock]:
    """A fake httpx module that answers a rerank payload, and the post mock it records on."""
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json.return_value = {"results": results}
    client = MagicMock()
    client.post = AsyncMock(return_value=response)
    context = MagicMock()
    context.__aenter__ = AsyncMock(return_value=client)
    context.__aexit__ = AsyncMock(return_value=False)
    httpx_stub = MagicMock()
    httpx_stub.AsyncClient = MagicMock(return_value=context)
    return httpx_stub, client.post


@pytest.fixture
def _server_envs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Point both server URLs at throwaway test hosts, with no model or key set."""
    monkeypatch.setenv("EMBEDDING_SERVER_URL", _EMBEDDING_URL)
    monkeypatch.setenv("RERANKER_SERVER_URL", _RERANKER_URL)
    for name in _OPTIONAL_VARS:
        monkeypatch.delenv(name, raising=False)


async def _search_with_rerank(store: EvidenceStore, query: str) -> str:
    """Run store.search with the reranker answered by a single top hit."""
    with patch("app.agents.research.evidence.httpx", _fake_rerank_http([{"index": 0, "score": 0.95}])):
        return await store.search(query)


@pytest.mark.usefixtures("_server_envs")
class TestEvidenceStore:
    async def test_open_creates_evidence_lance(self, tmp_path: Path) -> None:
        store = EvidenceStore(tmp_path)
        await store.open()
        assert (tmp_path / "evidence.lance").is_dir()
        await store.close()

    async def test_write_and_search(self, tmp_path: Path) -> None:
        store = EvidenceStore(tmp_path)
        client = _fake_embedding_client([_FAKE_VECTOR, _FAKE_VECTOR])
        with patch("app.agents.research.evidence.AsyncOpenAI", return_value=client):
            await store.open()
        await store.write("test content", tool_name="brave_search", source_task_id="run-1", call_params='{"query": "test"}')
        assert client.embeddings.create.call_count == 1
        result = await _search_with_rerank(store, "test query")
        assert "brave_search" in result
        assert "run-1" in result
        await store.close()

    async def test_write_dedup_skips_duplicate_chunk(self, tmp_path: Path) -> None:
        store = EvidenceStore(tmp_path)
        client = _fake_embedding_client([_FAKE_VECTOR])
        with patch("app.agents.research.evidence.AsyncOpenAI", return_value=client):
            await store.open()
            await store.write("duplicate content", tool_name="brave_search", source_task_id="run-1", call_params="q1")
            await store.write("duplicate content", tool_name="brave_search", source_task_id="run-2", call_params="q2")
        assert client.embeddings.create.call_count == 1
        await store.close()

    async def test_write_splits_long_content_into_small_chunks(self, tmp_path: Path) -> None:
        """A long document is embedded and reranked as chunks of at most 5000 characters."""
        chunks = ["a" * 5000, "b" * 5000, "c" * 1000]
        store = EvidenceStore(tmp_path)
        client = _fake_embedding_client([_FAKE_VECTOR] * 4)
        with patch("app.agents.research.evidence.AsyncOpenAI", return_value=client):
            await store.open()
        await store.write("".join(chunks), tool_name="web_fetch", source_task_id="run-1", call_params="q")
        embedded = [call.kwargs["input"] for call in client.embeddings.create.call_args_list]
        assert embedded == chunks
        httpx_stub, post = _fake_rerank_http_with_post(_TOP_HIT)
        with patch("app.agents.research.evidence.httpx", httpx_stub):
            await store.search("query")
        assert sorted(post.await_args.kwargs["json"]["documents"]) == chunks
        await store.close()

    async def test_write_failure_swallowed(self, tmp_path: Path) -> None:
        ok_item = MagicMock()
        ok_item.embedding = _FAKE_VECTOR
        ok_response = MagicMock()
        ok_response.data = [ok_item]
        client = MagicMock()
        client.embeddings.create = AsyncMock(side_effect=[RuntimeError("embed failed"), ok_response])
        store = EvidenceStore(tmp_path)
        with patch("app.agents.research.evidence.AsyncOpenAI", return_value=client):
            await store.open()
            await store.write("content", tool_name="brave_search", source_task_id="run-1", call_params="q")
            result = await store.search("content")
        assert result == "No matching evidence found for: content"
        await store.close()

    async def test_search_no_results(self, tmp_path: Path) -> None:
        store = EvidenceStore(tmp_path)
        client = _fake_embedding_client([_FAKE_VECTOR])
        with patch("app.agents.research.evidence.AsyncOpenAI", return_value=client):
            await store.open()
        result = await store.search("nonexistent query")
        assert "No matching evidence" in result
        await store.close()

    async def test_search_drops_candidates_beyond_distance(self, tmp_path: Path) -> None:
        store = EvidenceStore(tmp_path)
        far_vector = [0.5] * _VECTOR_DIM
        client = _fake_embedding_client([_FAKE_VECTOR, far_vector])
        with patch("app.agents.research.evidence.AsyncOpenAI", return_value=client):
            await store.open()
        await store.write("content", tool_name="brave_search", source_task_id="run-1", call_params="q")
        result = await store.search("far query")
        assert "No matching evidence" in result
        await store.close()

    async def test_search_formats_results(self, tmp_path: Path) -> None:
        store = EvidenceStore(tmp_path)
        client = _fake_embedding_client([_FAKE_VECTOR, _FAKE_VECTOR])
        with patch("app.agents.research.evidence.AsyncOpenAI", return_value=client):
            await store.open()
        await store.write("hello world", tool_name="brave_search", source_task_id="run-1", call_params="q")
        result = await _search_with_rerank(store, "test query")
        expected = (
            "## Evidence search: test query (1 result)\n"
            "\n"
            "### 1. brave_search - score 0.95\n"
            "run_id: `run-1`\n"
            "> hello world\n"
        )
        assert result == expected
        await store.close()

    async def test_search_formats_multiple_results_with_multiline_snippet(self, tmp_path: Path) -> None:
        store = EvidenceStore(tmp_path)
        near_vector = [0.1] * _VECTOR_DIM
        close_vector = [0.12] * _VECTOR_DIM
        client = _fake_embedding_client([near_vector, close_vector, near_vector])
        with patch("app.agents.research.evidence.AsyncOpenAI", return_value=client):
            await store.open()
        await store.write("hello world", tool_name="brave_search", source_task_id="run-1", call_params="q1")
        await store.write("line one\nline two", tool_name="web_fetch", source_task_id="run-2", call_params="q2")
        rerank_results = [{"index": 0, "score": 0.95}, {"index": 1, "score": 0.88}]
        with patch("app.agents.research.evidence.httpx", _fake_rerank_http(rerank_results)):
            result = await store.search("test query")
        expected = (
            "## Evidence search: test query (2 results)\n"
            "\n"
            "### 1. brave_search - score 0.95\n"
            "run_id: `run-1`\n"
            "> hello world\n"
            "\n"
            "### 2. web_fetch - score 0.88\n"
            "run_id: `run-2`\n"
            "> line one\n"
            "> line two\n"
        )
        assert result == expected
        await store.close()


@pytest.mark.usefixtures("_server_envs")
class TestEvidenceBackends:
    async def _open_and_write(self, tmp_path: Path, client: MagicMock) -> tuple[EvidenceStore, MagicMock]:
        """Open a store on the fake client, write one chunk, and give back the store and the client class."""
        store = EvidenceStore(tmp_path)
        with patch("app.agents.research.evidence.AsyncOpenAI", return_value=client) as client_class:
            await store.open()
        await store.write("hello world", tool_name="brave_search", source_task_id="run-1", call_params="q")
        return store, client_class

    async def test_embeddings_use_the_defaults_when_model_and_key_are_unset(self, tmp_path: Path) -> None:
        """With only the address set, the embeddings client gets one /v1 and the default model and key."""
        client = _fake_embedding_client([_FAKE_VECTOR])
        store, client_class = await self._open_and_write(tmp_path, client)
        client_class.assert_called_once_with(base_url=f"{_EMBEDDING_URL}/v1", api_key="not-needed")
        client.embeddings.create.assert_awaited_once_with(input="hello world", model="default")
        await store.close()

    async def test_embeddings_send_the_set_model_and_key(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A set embedding model and key reach the client."""
        monkeypatch.setenv("EMBEDDING_SERVER_URL", f"{_EMBEDDING_URL}/v1/")
        monkeypatch.setenv("EMBEDDING_MODEL", "bge-m3")
        monkeypatch.setenv("EMBEDDING_API_KEY", "sk-e")
        client = _fake_embedding_client([_FAKE_VECTOR])
        store, client_class = await self._open_and_write(tmp_path, client)
        client_class.assert_called_once_with(base_url=f"{_EMBEDDING_URL}/v1", api_key="sk-e")
        client.embeddings.create.assert_awaited_once_with(input="hello world", model="bge-m3")
        await store.close()

    async def test_open_fails_without_the_embedding_address(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                            capsys: pytest.CaptureFixture[str]) -> None:
        """With no embedding address, opening the store names the variable."""
        monkeypatch.delenv("EMBEDDING_SERVER_URL")
        message = "EMBEDDING_SERVER_URL is not set: needed by research"
        with pytest.raises(MissingSettingError, match=f"^{message}$"):
            await EvidenceStore(tmp_path).open()
        assert f"[config] {message}\n" in capsys.readouterr().err

    async def test_rerank_sends_the_default_model_and_no_header(self, tmp_path: Path) -> None:
        """With only the address set, the rerank request has the default model and no key."""
        store, _ = await self._open_and_write(tmp_path, _fake_embedding_client([_FAKE_VECTOR, _FAKE_VECTOR]))
        httpx_stub, post = _fake_rerank_http_with_post(_TOP_HIT)
        with patch("app.agents.research.evidence.httpx", httpx_stub):
            await store.search("test query")
        post.assert_awaited_once_with(
            f"{_RERANKER_URL}/v1/rerank",
            json={"model": "default", "query": "test query", "documents": ["hello world"]},
            headers={},
        )
        await store.close()

    async def test_rerank_sends_the_set_model_and_key(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A set reranker model and key reach the rerank request."""
        monkeypatch.setenv("RERANKER_MODEL", "bge-reranker")
        monkeypatch.setenv("RERANKER_API_KEY", "sk-r")
        store, _ = await self._open_and_write(tmp_path, _fake_embedding_client([_FAKE_VECTOR, _FAKE_VECTOR]))
        httpx_stub, post = _fake_rerank_http_with_post(_TOP_HIT)
        with patch("app.agents.research.evidence.httpx", httpx_stub):
            await store.search("test query")
        post.assert_awaited_once_with(
            f"{_RERANKER_URL}/v1/rerank",
            json={"model": "bge-reranker", "query": "test query", "documents": ["hello world"]},
            headers={"Authorization": "Bearer sk-r"},
        )
        await store.close()

    async def test_search_fails_without_the_reranker_address(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                             capsys: pytest.CaptureFixture[str]) -> None:
        """With no reranker address, a search that has candidates names the variable."""
        store, _ = await self._open_and_write(tmp_path, _fake_embedding_client([_FAKE_VECTOR, _FAKE_VECTOR]))
        monkeypatch.delenv("RERANKER_SERVER_URL")
        message = "RERANKER_SERVER_URL is not set: needed by research"
        with pytest.raises(MissingSettingError, match=f"^{message}$"):
            await store.search("test query")
        assert f"[config] {message}\n" in capsys.readouterr().err
        await store.close()


class TestSearchEvidence:
    async def test_call_delegates_to_store_search(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EMBEDDING_SERVER_URL", _EMBEDDING_URL)
        store = EvidenceStore(tmp_path)
        await store.open()
        tool = SearchEvidence(store)
        with patch.object(store, "search", new_callable=AsyncMock, return_value="some results") as mock_search:
            result = await tool.call({"query": "test", "max_results": 5})
            mock_search.assert_called_once_with("test", 5)
        assert result.text == "some results"
        await store.close()

    async def test_run_persists_class_default_without_registration(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The tool's class default is stored on its run row."""
        assert SearchEvidence.result_render_type == "markdown"          # the declaration exists
        assert tool_class("search_evidence") is None                    # this tool is not registered
        monkeypatch.setenv("EMBEDDING_SERVER_URL", _EMBEDDING_URL)
        store = EvidenceStore(tmp_path)
        await store.open()
        tool = SearchEvidence(store)
        assert tool.result_render_type == "markdown"
        with patch.object(store, "search", new_callable=AsyncMock,
                          return_value="## Evidence search: q (1 results)") as mock_search:
            task = await tool.run({"query": "q"})
            await wait_terminal(store_module.task_store, str(task.id))
        assert mock_search.await_count == 1
        view = await store_module.task_view(str(task.id))
        assert view is not None and view.render_type == "markdown"
        await store.close()
