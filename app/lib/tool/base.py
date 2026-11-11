from __future__ import annotations

import abc
import asyncio
import inspect
import json
import os
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import ClassVar, Literal, Protocol, final, get_args

import aiofiles
import pydantic

from app.lib.db import store as store_module
from app.lib.render.detect import TEXT_FAMILY_KINDS
from app.lib.shared import activity
from app.lib.shared.paths import resolve_path_sync, resources_dir
from app.lib.task.task_manager import Task, TaskCancelledError, task_manager

__all__ = [
    "Contract",
    "Param",
    "ParamValue",
    "Tool",
    "ToolMessageError",
    "ToolResult",
    "parse_files_sidecar",
    "read_run_mcp_text",
    "format_staleness",
    "param_signature",
]


def _now_utc() -> datetime:
    """Return the current UTC time as an aware datetime."""
    return datetime.now(timezone.utc)


class ToolMessageError(Exception):
    """A call that never started, whose str is the exact text the client shows."""


_SECONDS_PER_MINUTE = 60
_SECONDS_PER_HOUR = 3_600
_SECONDS_PER_DAY = 86_400
_MIN_DAYS_FOR_DAYS_TIER = 2

# This note is added to a tool description in one place.
# Never write it into a tool's own description.
_CACHE_NOTE = "Duplicate calls within {duration} reuse the stored result silently."

_TEXT_FAMILY_TOKENS = {k.value for k in TEXT_FAMILY_KINDS}


def format_staleness(staleness: timedelta) -> str:
    """Write a time window out in words.

    Args:
        staleness: How long the window is. Must be more than zero.

    Returns:
        The window written in days, hours, minutes, or seconds.
    """
    total = int(staleness.total_seconds())
    if total % _SECONDS_PER_DAY == 0 and total // _SECONDS_PER_DAY >= _MIN_DAYS_FOR_DAYS_TIER:
        return f"{total // _SECONDS_PER_DAY} days"
    if total % _SECONDS_PER_HOUR == 0:
        hours = total // _SECONDS_PER_HOUR
        return "1 hour" if hours == 1 else f"{hours} hours"
    if total % _SECONDS_PER_MINUTE == 0:
        minutes = total // _SECONDS_PER_MINUTE
        return "1 minute" if minutes == 1 else f"{minutes} minutes"
    return "1 second" if total == 1 else f"{total} seconds"


def _compose_description(description: str, staleness: timedelta | None) -> str:
    """Build the description a tool shows on the wire.

    Args:
        description: The tool's own description.
        staleness: How long a stored result is reused, or None.

    Returns:
        The description with the cache note added when it applies.
    """
    parts: list[str] = [description]
    if staleness is not None:
        parts.append(_CACHE_NOTE.format(duration=format_staleness(staleness)))
    return " ".join(parts)


# Notes added to a param's wire description, picked by the param kind.
# Never write one of these into a param's own description by hand.
# The word for the resource is always "resource", never "share".
_RESOURCE_NOTE = (
    "Each path must already exist in the resource. If you created a file, "
    "call the send_or_receive_file tool first for upload instructions, then "
    "use the uploaded file's path, usually 'upload/<filename>'. List available "
    "files with your MCP client's resource listing tool (entries look like "
    "'resources:///upload/video.mp4')."
)

_PARAM_KIND_NOTES: dict[str, str] = {
    "date": "Use the YYYY-MM-DD format.",
    "url": "Use an http or https URL.",
    "resource": _RESOURCE_NOTE,
    "resource_list": _RESOURCE_NOTE,
}

_PARAM_DEFAULT_NOTE = "Optional (default: {})."
_PARAM_OPTIONAL_NOTE = "Optional."
_PARAM_ENUM_NOTE = "Valid values: {}."


def _compose_param_description(p: Param) -> str:
    """Build the wire description for one param.

    The param's own description comes first, then the kind note, then the
    optional note, then the enum note, joined by single spaces. Each note is
    added only when the param has that thing. An optional param always gets
    the optional note, and the note names the default only when the default
    is a plain value that is not empty. A required param gets no note. The
    min and max never show up in the text.

    Args:
        p: The param to describe.

    Returns:
        The description the model sees.
    """
    parts: list[str] = [p.description]
    note = _PARAM_KIND_NOTES.get(p.kind or "")
    if note is not None:
        parts.append(note)
    if not p.required:
        if p.default is None or p.default == "" or isinstance(p.default, (list, dict, tuple, set)):
            parts.append(_PARAM_OPTIONAL_NOTE)
        else:
            parts.append(_PARAM_DEFAULT_NOTE.format(p.default))
    if p.enum:
        parts.append(_PARAM_ENUM_NOTE.format(", ".join(p.enum)))
    return " ".join(parts)


async def _read_file(path: str) -> str:
    async with aiofiles.open(path, "r", encoding="utf-8") as fh:
        return await fh.read()


async def read_run_mcp_text(run_path: str) -> str:
    """Read the text a run shows to the model.

    Args:
        run_path: Folder holding the run files.

    Returns:
        The text of the run result.

    Raises:
        FileNotFoundError: If the run has no result file.
    """
    sidecar = os.path.join(run_path, "result.text")
    if await asyncio.to_thread(os.path.exists, sidecar):
        return await _read_file(sidecar)
    return await _read_file(os.path.join(run_path, "result"))


def _manifest_line(share_path: str) -> str:
    size = mtime = None
    full = None if "\x00" in share_path else resolve_path_sync(resources_dir(), share_path)
    if full is not None:
        try:
            st = os.stat(full)
            size, mtime = st.st_size, int(st.st_mtime)
        except OSError:
            pass
    return json.dumps({"path": share_path, "size": size, "mtime": mtime})


async def _write_files_sidecar(run_path: str, files: tuple[str, ...]) -> None:
    """Write the run's files sidecar, one JSON manifest line per file.

    Args:
        run_path: Folder holding the run files.
        files: The resource files the result produced.
    """
    lines = [await asyncio.to_thread(_manifest_line, p) for p in files]
    async with aiofiles.open(os.path.join(run_path, "files"), "w", encoding="utf-8") as fh:
        await fh.write("\n".join(lines) + "\n")


def _parse_line(line: str) -> dict:
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return {"path": line, "size": None, "mtime": None}
    if isinstance(obj, dict) and isinstance(obj.get("path"), str):
        size, mtime = obj.get("size"), obj.get("mtime")
        return {"path": obj["path"],
                "size": size if isinstance(size, int) and not isinstance(size, bool) else None,
                "mtime": mtime if isinstance(mtime, int) and not isinstance(mtime, bool) else None}
    return {"path": line, "size": None, "mtime": None}


def _parse_files_sidecar_sync(run_path: str) -> list[dict]:
    out: list[dict] = []
    try:
        with open(os.path.join(run_path, "files"), "r", encoding="utf-8") as fh:
            for raw in fh:
                line = raw.strip()
                if line:
                    out.append(_parse_line(line))
    except OSError:
        pass
    return out


async def parse_files_sidecar(run_path: str) -> list[dict]:
    """Read the run's files sidecar as path/size/mtime records.

    Args:
        run_path: Folder holding the run files.

    Returns:
        One record per non-empty line. A line that parses as a JSON object with a
        string path is a manifest entry. Any other non-empty line is a pre-migration
        bare path with null size and mtime. An absent file yields no records.
    """
    return await asyncio.to_thread(_parse_files_sidecar_sync, run_path)


def _validate_result(result: ToolResult) -> str | None:
    if not isinstance(result, ToolResult):
        return "tool must return a ToolResult"
    if "\x00" in result.text:
        return "result text must not contain a NUL byte"
    if result.render_type is not None and result.render_type not in _TEXT_FAMILY_TOKENS:
        return f"result render_type must be one of: {', '.join(sorted(_TEXT_FAMILY_TOKENS))}"
    return None


def _stringify_tool_result(result: ToolResult) -> str:
    """The text body a ToolResult persists, for the run's result file."""
    return result.text


def _error_task(error: str) -> Task[ToolResult]:
    """The failed handle for a call whose params did not check out."""
    return task_manager.create_completed_task(error=ToolMessageError(error))


class ParamValue(Protocol):
    """A value one param takes. A JSON scalar or a container."""


class _Annotation(Protocol):
    """A value that names a Python type."""


class _Required:
    """Marks a Param that has no default value."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "<required>"


_REQUIRED = _Required()


@dataclass(frozen=True)
class Param:
    """One tool parameter."""

    name: str
    type: type
    default: _Required | ParamValue = _REQUIRED
    description: str = ""
    # description is the plain purpose of the param. The wire form adds a note
    # for the kind, a note when the param is optional, and a note listing the
    # enum values.
    # The console reads kind, min, max, and step. Only min and max also reach the wire.
    kind: str | None = None
    min: int | float | None = None
    max: int | float | None = None
    step: int | float | None = None
    enum: list[str] | tuple[str, ...] | None = None

    @property
    def required(self) -> bool:
        return self.default is _REQUIRED

    @property
    def wire_description(self) -> str:
        """The description the model sees.

        It is this param's own description plus the kind, default, and enum
        notes built from the param's own data. The console keeps serving the
        plain description.
        """
        return _compose_param_description(self)

    @property
    def annotation(self) -> _Annotation:
        """The Python type for this param.

        A param with an enum is limited to its enum values.
        """
        if self.enum:
            return Literal[tuple(self.enum)]
        return self.type


@dataclass(frozen=True)
class Contract:
    """Everything a tool declares about itself."""

    name: str
    description: str
    summary: str
    params: tuple[Param, ...]
    rules: tuple[str, ...]
    tracked: bool
    result_render_type: str | None = None
    is_result_long: bool = False


@dataclass(frozen=True)
class ToolResult:
    """A run result, the text body plus the resource files it produced."""

    text: str
    files: tuple[str, ...] | None = None
    render_type: str | None = None


_validation_adapters: dict[type, pydantic.TypeAdapter[dict]] = {}
_numeric_adapters: dict[type, pydantic.TypeAdapter[int | float]] = {}


def param_signature(params: tuple[Param, ...]) -> inspect.Signature:
    """Build the call signature for a list of params.

    All required params must come before the ones with a default.

    Args:
        params: The params to turn into a signature.

    Returns:
        A signature with the params in declared order that returns a string.

    Raises:
        ValueError: If a param with a default comes before a required one.
    """
    ps = [
        inspect.Parameter(
            p.name,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            default=inspect.Parameter.empty if p.required else p.default,
            annotation=p.annotation,
        )
        for p in params
    ]
    return inspect.Signature(ps, return_annotation=str)


def _validation_adapter(tool_cls: type) -> pydantic.TypeAdapter[dict]:
    """The checker for one tool's params."""
    adapter = _validation_adapters.get(tool_cls)
    if adapter is None:
        adapter = pydantic.TypeAdapter(tool_cls.validation_stub())
        _validation_adapters[tool_cls] = adapter
    return adapter


def _numeric_adapter(numeric: type) -> pydantic.TypeAdapter[int | float]:
    """The checker for one number type."""
    adapter = _numeric_adapters.get(numeric)
    if adapter is None:
        adapter = pydantic.TypeAdapter(numeric)
        _numeric_adapters[numeric] = adapter
    return adapter


def _numeric_type(annotation: type) -> type | None:
    """The number type a param uses, or None when the param is not a number."""
    if annotation in (int, float):
        return annotation
    for arg in get_args(annotation):
        if arg in (int, float):
            return arg
    return None


def _drop_tolerated_none(values: dict, none_tolerant: set[str]) -> dict:
    """Leave out the entries whose value is None and whose name is tolerated."""
    return {k: v for k, v in values.items() if not (v is None and k in none_tolerant)}


def _tool_view(params: tuple[Param, ...], values: dict) -> dict:
    """The values to check for a set the tool's own run handed over.

    A None is left out for a param whose type allows None.
    """
    tolerant = {p.name for p in params if type(None) in get_args(p.type)}
    return _drop_tolerated_none(values, tolerant)


def _format_bound(value: int | float) -> str:
    """Write a min or max value for an error message.

    A float with no fraction is written as a whole number.
    """
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _bound_violation(p: Param, value: ParamValue) -> str | None:
    """The error text when a number falls outside the param's min or max, or None when it fits."""
    if p.min is None and p.max is None:
        return None
    numeric = _numeric_type(p.type)
    if numeric is None:
        return None
    try:
        derived = _numeric_adapter(numeric).validate_python(value)
    except pydantic.ValidationError:
        return None
    if p.min is not None and not (derived >= p.min):
        return f"Input should be greater than or equal to {_format_bound(p.min)}."
    if p.max is not None and not (derived <= p.max):
        return f"Input should be less than or equal to {_format_bound(p.max)}."
    return None


def _validation_error(exc: pydantic.ValidationError) -> str:
    """Build the error text for the first thing that failed."""
    first = exc.errors()[0]
    name = ".".join(str(part) for part in first["loc"])
    return f"Error: {name}: {first['msg']}."


class Tool(abc.ABC):
    """Base class for tools. Every call is saved as a task."""

    name: ClassVar[str]
    is_slow: ClassVar[bool] = False
    staleness: ClassVar[timedelta | None] = None
    description: ClassVar[str]
    summary: ClassVar[str]
    params: ClassVar[tuple[Param, ...]]
    rules: ClassVar[tuple[str, ...]] = ()
    tracked: ClassVar[bool] = True
    result_render_type: ClassVar[str | None] = None
    is_result_long: ClassVar[bool] = False

    def contract(self) -> Contract:
        """Collect what this tool declares.

        Returns:
            The tool contract.

        Raises:
            AttributeError: If the tool is missing something it must declare.
        """
        return Contract(
            self.name,
            self.description,
            self.summary,
            self.params,
            self.rules,
            self.tracked,
            self.result_render_type,
            self.is_result_long,
        )

    @classmethod
    def validation_stub(cls) -> Callable[..., dict]:
        """A function taking this tool's params and handing them back.

        Returns:
            A function with one argument per declared param.
        """

        def stub(**kwargs: ParamValue) -> dict:
            return dict(kwargs)

        stub.__signature__ = param_signature(cls.params)
        stub.__annotations__ = {p.name: p.annotation for p in cls.params} | {"return": str}
        return stub

    def _check_params(self, params: dict, view: dict) -> tuple[dict | None, str | None]:
        """Check one set of values against what this tool declares.

        It looks at missing params, types, enum values, extra keys, and min
        and max limits. Defaults fill in the params a call left out, and each
        value is set to its declared type. Only the first problem is reported.

        Args:
            params: The values to check.
            view: The values the checker reads, with allowed Nones left out.

        Returns:
            The bound params and no error when all is well, or None and the
            error text for the first problem.
        """
        try:
            bound = _validation_adapter(type(self)).validate_python(view)
        except pydantic.ValidationError as exc:
            return None, _validation_error(exc)
        for p in self.params:
            if p.name in params and params[p.name] is not None:
                if (violation := _bound_violation(p, params[p.name])) is not None:
                    return None, f"Error: {p.name}: {violation}"
        return bound, None

    def validate_params(self, params: dict) -> str | None:
        """Check a set of params the tool's own run handed over.

        The params are left as they are.

        Args:
            params: The call arguments to check.

        Returns:
            The error text for the first problem, or None when all is well.
        """
        return self._check_params(params, _tool_view(self.params, params))[1]

    @property
    def dynamic_description(self) -> str:
        """The description sent on the wire, with the cache note added when it applies."""
        return _compose_description(self.description, self.staleness)

    @abc.abstractmethod
    async def call(self, params: dict) -> ToolResult:
        """Do the tool's work.

        Args:
            params: The call arguments.

        Returns:
            The whole result.
        """

    async def cache_key(self, _params: dict) -> str | None:
        """The cache key for this call, or None when the call is not cached.

        Work done here must not block the caller.

        Args:
            _params: The call arguments.

        Returns:
            A key that stands for this call, or None.
        """
        return None

    @final
    async def run(self, params: dict) -> Task[ToolResult]:
        """Start one tool call from a raw params dict and hand back its handle.

        The params are checked once. A bad set comes back as a failed immediate
        handle carrying this tool's own words. A resource input whose use cannot be
        saved comes back as a failed immediate handle too. A run that starts comes
        back as an AsyncTask, and a reused stored result comes back as a resolving
        immediate.

        Args:
            params: The raw call arguments.

        Returns:
            The call's task.
        """
        if (error := self.validate_params(params)) is not None:
            return _error_task(error)
        full = {p.name: params.get(p.name, p.default) for p in self.params}
        try:
            await self._record_resource_inputs(full)
        except OSError as exc:
            return _error_task(self.failure_message(self.name, exc))
        key: str | None = None
        if self.staleness is not None:
            key = await self.cache_key(full)
            if key is not None:
                oldest = (_now_utc() - self.staleness).isoformat()
                stored = await self._stored_result(key, oldest)
                if stored is not None:
                    return stored
        task_id = uuid.uuid4()
        run_path = os.path.join(
            store_module.tool_store.data_dir, f"{store_module.RUNS_DIR}/{task_id}"
        )

        async def _runner() -> ToolResult:
            result = await self.call(full)
            if (err := _validate_result(result)) is not None:
                raise RuntimeError(err)
            if self.tracked:
                declared = result.render_type or self.result_render_type
                if declared in _TEXT_FAMILY_TOKENS:
                    await store_module.tool_store.set_render_type(str(task_id), declared)
                if result.files:
                    await _write_files_sidecar(run_path, result.files)
            return result

        task = await task_manager.create_task(
            _runner,
            task_id=task_id,
            should_persist_record=self.tracked,
            stringify_result=_stringify_tool_result,
        )
        if self.tracked:
            await store_module.tool_store.insert(str(task_id), self.name, key, full)
        await task.wait_started()
        return task

    async def _record_resource_inputs(self, params: dict) -> None:
        """Record interest in every resource file given as input.

        Args:
            params: The call arguments.

        Raises:
            OSError: If the interest cannot be saved.
        """
        for param in self.params:
            value = params.get(param.name)
            if param.kind == "resource" and isinstance(value, str) and value:
                await activity.record(value)
            elif param.kind == "resource_list" and isinstance(value, list):
                for path in value:
                    if isinstance(path, str) and path:
                        await activity.record(path)

    @staticmethod
    def failure_message(name: str, exc: Exception) -> str:
        """The message returned when a tool call fails."""
        return f"Tool '{name}' failed: {exc}"

    async def result_text(self, task: Task[ToolResult]) -> str:
        """Wait for what a call did and give back the text a client shows.

        A success hands back the result body. A cancelled run names its task. A
        tool message error hands back its own words. Any other error hands back
        this tool's failure message.

        Args:
            task: The handle the call returned.

        Returns:
            The text for the caller.
        """
        try:
            return (await task.result()).text
        except TaskCancelledError:
            return f"Task {task.id} was cancelled."
        except ToolMessageError as exc:
            return str(exc)
        except Exception as exc:
            return self.failure_message(self.name, exc)

    async def _stored_result(self, key: str, oldest: str) -> Task[ToolResult] | None:
        """The handle for an earlier run this call reuses, or None when there is none.

        Args:
            key: The cache key of this call.
            oldest: The earliest start time an earlier run may have.

        Returns:
            A completed handle carrying the stored text under the reused run's
            own id, or None when no earlier run fits.
        """
        hit = await store_module.tool_store.find_cache(self.name, key, oldest)
        if hit is None:
            return None
        try:
            cached = await read_run_mcp_text(os.path.join(store_module.tool_store.data_dir, hit.run_dir))
        except FileNotFoundError:
            return None
        return task_manager.create_completed_task(
            ToolResult(text=cached), task_id=uuid.UUID(hit.task_id)
        )
