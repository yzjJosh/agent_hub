"""Per-run evidence store for the research agent."""

from __future__ import annotations

import asyncio
import hashlib
import os
import sys
import time
from datetime import timedelta
from typing import ClassVar, override

import httpx
import lancedb
from lancedb.table import LanceTable
from openai import AsyncOpenAI

from app.lib.shared import config
from app.lib.tool.base import Param, Tool, ToolResult

__all__ = ["EvidenceStore", "SearchEvidence"]

_EMBEDDING_DIM = 1024
_CHUNK_MAX_CHARS = 5000
_MAX_RESULTS_CAP = 50
_CANDIDATE_FETCH_LIMIT = 50
_VECTOR_DISTANCE_THRESHOLD = 1.2
_SNIPPET_MAX_CHARS = 300
_RERANK_TIMEOUT_SECS = 30
_TABLE_NAME = "evidence"


async def _embed_text(client: AsyncOpenAI, model: str, text: str) -> list[float]:
    """Get the embedding vector for one piece of text.

    Args:
        client: The embedding server client.
        model: The embedding model to ask for.
        text: The text to embed.

    Returns:
        The vector.
    """
    response = await client.embeddings.create(input=text, model=model)
    return response.data[0].embedding


async def _rerank(query: str, documents: list[str]) -> list[dict]:
    """Score documents against a query with the reranker.

    Args:
        query: What the documents should match.
        documents: The texts to score.

    Returns:
        The scores in best-first order, each paired with its document index.

    Raises:
        MissingSettingError: If RERANKER_SERVER_URL is not set.
    """
    reranker = config.backend(config.RERANKER)
    payload = {"model": reranker.model or config.DEFAULT_MODEL, "query": query, "documents": documents}
    async with httpx.AsyncClient(timeout=_RERANK_TIMEOUT_SECS) as client:
        resp = await client.post(f"{reranker.url}/rerank", json=payload, headers=reranker.auth_headers())
        resp.raise_for_status()
        data = resp.json()
    results = data.get("results", [])
    return sorted(results, key=lambda r: r.get("score", r.get("relevance_score", 0)), reverse=True)


def _format_search_results(query: str, top_items: list[tuple[dict, float]]) -> str:
    """Lay the ranked hits out as markdown.

    Args:
        query: The query the results answer.
        top_items: The ranked rows with their rerank scores.

    Returns:
        The markdown the caller sends back.
    """
    count = len(top_items)
    unit = "result" if count == 1 else "results"
    lines: list[str] = [f"## Evidence search: {query} ({count} {unit})", ""]
    for i, (chunk, score) in enumerate(top_items, 1):
        content = chunk["content"]
        snippet = content[:_SNIPPET_MAX_CHARS]
        if len(content) > _SNIPPET_MAX_CHARS:
            snippet += "..."
        lines.append(f"### {i}. {chunk['tool_name']} - score {score:.2f}")
        lines.append(f"run_id: `{chunk['source_task_id']}`")
        lines.extend(f"> {line}" for line in snippet.splitlines())
        lines.append("")
    return "\n".join(lines)


class EvidenceStore:
    """Vector store over the content gathered in one run."""

    def __init__(self, run_folder: str | os.PathLike) -> None:
        """Bind the store to the folder it keeps its data in.

        Args:
            run_folder: The run folder that holds the store data.
        """
        self._run_folder: str = os.fspath(run_folder)
        self._db: lancedb.DBConnection | None = None
        self._client: AsyncOpenAI | None = None
        self._embedding_model: str = config.DEFAULT_MODEL

    async def open(self) -> None:
        """Open the store and make sure its table exists.

        Raises:
            MissingSettingError: If EMBEDDING_SERVER_URL is not set.
        """
        embedding = config.backend(config.EMBEDDING)
        self._client = AsyncOpenAI(base_url=embedding.url, api_key=embedding.api_key or config.DEFAULT_API_KEY)
        self._embedding_model = embedding.model or config.DEFAULT_MODEL
        self._db = await asyncio.to_thread(lancedb.connect, self._run_folder)
        await asyncio.to_thread(self._init_table_sync)

    def _init_table_sync(self) -> None:
        """Create the evidence table if it is missing."""
        assert self._db is not None
        try:
            self._db.open_table(_TABLE_NAME)
            return
        except Exception:
            pass
        sample_row = {
            "chunk_hash": "",
            "tool_name": "",
            "source_task_id": "",
            "call_params": "",
            "content": "",
            "timestamp": 0.0,
            "vector": [0.0] * _EMBEDDING_DIM,
        }
        self._db.create_table(_TABLE_NAME, data=[sample_row])
        self._db.open_table(_TABLE_NAME).delete("chunk_hash = ''")

    async def write(self, content: str, *, tool_name: str, source_task_id: str, call_params: str) -> None:
        """Store one piece of gathered content. A failed write is logged and dropped.

        Args:
            content: The text to chunk and embed.
            tool_name: The tool that produced the content.
            source_task_id: The run id of the run that produced the content.
            call_params: The params JSON of that run.
        """
        try:
            if self._client is None:
                return
            chunks = [content[i : i + _CHUNK_MAX_CHARS] for i in range(0, len(content), _CHUNK_MAX_CHARS)]
            table = await asyncio.to_thread(self._open_table)
            for chunk in chunks:
                chunk_hash = hashlib.sha256(chunk.encode()).hexdigest()
                if await asyncio.to_thread(self._find_chunk, table, chunk_hash):
                    continue
                embedding = await _embed_text(self._client, self._embedding_model, chunk)
                row = {
                    "chunk_hash": chunk_hash,
                    "tool_name": tool_name,
                    "source_task_id": source_task_id,
                    "call_params": call_params,
                    "content": chunk,
                    "timestamp": time.time(),
                    "vector": embedding,
                }
                await asyncio.to_thread(table.add, [row])
        except Exception as exc:
            print(
                f"[EvidenceStore.write] Failed to write evidence for run {source_task_id}: {exc}",
                file=sys.stderr,
                flush=True,
            )

    async def search(self, query: str, max_results: int = 10) -> str:
        """Find the stored content that matches a query best.

        Args:
            query: What to look for.
            max_results: How many items to return.

        Returns:
            The ranked hits as markdown, or a note that nothing matched.
        """
        max_results = min(max(max_results, 1), _MAX_RESULTS_CAP)
        if self._client is None:
            return f"No matching evidence found for: {query}"
        query_embedding = await _embed_text(self._client, self._embedding_model, query)
        table = await asyncio.to_thread(self._open_table)
        candidates = await asyncio.to_thread(self._search_table, table, query_embedding)
        candidates = [c for c in candidates if c.get("_distance", float("inf")) <= _VECTOR_DISTANCE_THRESHOLD]
        if not candidates:
            return f"No matching evidence found for: {query}"
        documents = [c["content"] for c in candidates]
        ranked = await _rerank(query, documents)
        top_items = [
            (candidates[r["index"]], r.get("score", r.get("relevance_score", 0)))
            for r in ranked[:max_results]
            if r["index"] < len(candidates)
        ]
        return _format_search_results(query, top_items)

    async def close(self) -> None:
        """Drop the store's open handles. The store is unusable after a close."""
        self._db = None
        self._client = None

    def _open_table(self) -> LanceTable:
        """The run's evidence table."""
        assert self._db is not None
        return self._db.open_table(_TABLE_NAME)

    def _find_chunk(self, table: LanceTable, chunk_hash: str) -> bool:
        """Whether a stored chunk already carries this hash."""
        zero_vector = [0.0] * _EMBEDDING_DIM
        found = table.search(zero_vector).where(f"chunk_hash = '{chunk_hash}'").limit(1).to_list()
        return len(found) > 0

    def _search_table(self, table: LanceTable, query_embedding: list[float]) -> list[dict]:
        """The closest stored rows to a query vector."""
        return table.search(query_embedding).limit(_CANDIDATE_FETCH_LIMIT).to_list()


class SearchEvidence(Tool):
    """In-loop tool that searches one run's evidence store."""

    name: ClassVar[str] = "search_evidence"
    staleness: ClassVar[timedelta | None] = None
    tracked: ClassVar[bool] = True
    result_render_type: ClassVar[str] = "markdown"
    description: ClassVar[str] = (
        "Search the evidence already gathered in this run. "
        "Use it to find relevant findings before searching the web again."
    )
    summary: ClassVar[str] = "Search the run's evidence store"
    params: ClassVar[tuple[Param, ...]] = (
        Param("query", str, description="What to look for in the gathered evidence."),
        Param("max_results", int, 10, max=_MAX_RESULTS_CAP, description="How many results to return."),
    )

    def __init__(self, store: EvidenceStore) -> None:
        """Bind the tool to the store it searches.

        Args:
            store: The run's evidence store.
        """
        self._store = store

    @override
    async def call(self, params: dict) -> ToolResult:
        """Search the bound store.

        Args:
            params: The call arguments.

        Returns:
            The ranked results as text.
        """
        query = params["query"]
        max_results = int(params.get("max_results", 10))
        return ToolResult(text=await self._store.search(query, max_results))
