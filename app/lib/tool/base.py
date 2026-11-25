"""Base classes and helpers for declaring, checking, and running tools."""
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
from typing import ClassVar, Literal, Protocol, TypeAlias, final, get_args

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
    "build_param_signature",
    "format_staleness",
    "parse_files_sidecar",
    "read_run_mcp_text",
]


class ToolMessageError(Exception):
    """Error for a call that never started, whose message is shown directly to clients."""


_SECONDS_PER_MINUTE = 60
_SECONDS_PER_HOUR = 3_600
_SECONDS_PER_DAY = 86_400
_MIN_DAYS_FOR_DAYS_TIER = 2

_CACHE_NOTE = "Duplicate calls within {duration} reuse the stored result silently."

_TEXT_FAMILY_TOKENS = {kind.value for kind in TEXT_FAMILY_KINDS}

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


def _format_cache_cutoff(staleness: timedelta) -> str:
    """Return the earliest UTC ISO timestamp eligible for cache reuse."""
    return (datetime.now(timezone.utc) - staleness).isoformat()


def format_staleness(staleness: timedelta) -> str:
    """Write a time window out in words.

    Args:
        staleness: Length of the window, greater than zero.

    Returns:
        The window written in days, hours, minutes, or seconds.
    """
    total_seconds = int(staleness.total_seconds())
    if (
        total_seconds % _SECONDS_PER_DAY == 0
        and total_seconds // _SECONDS_PER_DAY >= _MIN_DAYS_FOR_DAYS_TIER
    ):
        return f"{total_seconds // _SECONDS_PER_DAY} days"
    if total_seconds % _SECONDS_PER_HOUR == 0:
        hours = total_seconds // _SECONDS_PER_HOUR
        return "1 hour" if hours == 1 else f"{hours} hours"
    if total_seconds % _SECONDS_PER_MINUTE == 0:
        minutes = total_seconds // _SECONDS_PER_MINUTE
        return "1 minute" if minutes == 1 else f"{minutes} minutes"
    return "1 second" if total_seconds == 1 else f"{total_seconds} seconds"


def _compose_description(description: str, staleness: timedelta | None) -> str:
    """Build the tool description sent on the wire."""
    parts: list[str] = [description]
    if staleness is not None:
        parts.append(_CACHE_NOTE.format(duration=format_staleness(staleness)))
    return " ".join(parts)


def _compose_param_description(param: Param) -> str:
    """Build the parameter description sent on the wire."""
    parts: list[str] = [param.description]
    kind_note = _PARAM_KIND_NOTES.get(param.kind or "")
    if kind_note is not None:
        parts.append(kind_note)
    if not param.is_required:
        has_empty_or_container_default = (
            param.default is None
            or param.default == ""
            or isinstance(param.default, (list, dict, tuple, set))
        )
        if has_empty_or_container_default:
            parts.append(_PARAM_OPTIONAL_NOTE)
        else:
            parts.append(_PARAM_DEFAULT_NOTE.format(param.default))
    if param.enum:
        parts.append(_PARAM_ENUM_NOTE.format(", ".join(param.enum)))
    return " ".join(parts)


async def _read_file(path: str) -> str:
    async with aiofiles.open(path, "r", encoding="utf-8") as file_handle:
        return await file_handle.read()


async def read_run_mcp_text(run_path: str) -> str:
    """Read the result text a run exposes to clients.

    Args:
        run_path: Folder holding the run files.

    Returns:
        The text of the run result.

    Raises:
        FileNotFoundError: If the run has no result file.
    """
    sidecar_path = os.path.join(run_path, "result.text")
    if await asyncio.to_thread(os.path.exists, sidecar_path):
        return await _read_file(sidecar_path)
    return await _read_file(os.path.join(run_path, "result"))


def _format_manifest_line(resource_path: str) -> str:
    size: int | None = None
    modified_time: int | None = None
    resolved_path = (
        None if "\x00" in resource_path else resolve_path_sync(resources_dir(), resource_path)
    )
    if resolved_path is not None:
        try:
            file_stat = os.stat(resolved_path)
            size = file_stat.st_size
            modified_time = int(file_stat.st_mtime)
        except OSError:
            pass
    return json.dumps({"path": resource_path, "size": size, "mtime": modified_time})


def _write_files_sidecar_sync(run_path: str, files: tuple[str, ...]) -> None:
    lines = [_format_manifest_line(resource_path) for resource_path in files]
    sidecar_path = os.path.join(run_path, "files")
    with open(sidecar_path, "w", encoding="utf-8") as file_handle:
        file_handle.write("\n".join(lines) + "\n")


async def _write_files_sidecar(run_path: str, files: tuple[str, ...]) -> None:
    """Write the run files sidecar with one JSON manifest line per file."""
    await asyncio.to_thread(_write_files_sidecar_sync, run_path, files)


def _parse_manifest_line(line: str) -> dict:
    try:
        parsed = json.loads(line)
    except json.JSONDecodeError:
        return {"path": line, "size": None, "mtime": None}
    if isinstance(parsed, dict) and isinstance(parsed.get("path"), str):
        size = parsed.get("size")
        modified_time = parsed.get("mtime")
        return {
            "path": parsed["path"],
            "size": size if isinstance(size, int) and not isinstance(size, bool) else None,
            "mtime": (
                modified_time
                if isinstance(modified_time, int) and not isinstance(modified_time, bool)
                else None
            ),
        }
    return {"path": line, "size": None, "mtime": None}


def _parse_files_sidecar_sync(run_path: str) -> list[dict]:
    records: list[dict] = []
    try:
        with open(os.path.join(run_path, "files"), "r", encoding="utf-8") as file_handle:
            for raw_line in file_handle:
                line = raw_line.strip()
                if line:
                    records.append(_parse_manifest_line(line))
    except OSError:
        pass
    return records


async def parse_files_sidecar(run_path: str) -> list[dict]:
    """Read the run files sidecar as path, size, and modified time records.

    Args:
        run_path: Folder holding the run files.

    Returns:
        One record per non-empty line. A JSON object with a string path becomes
        a manifest record, and any other non-empty line becomes a bare path
        record with None size and modified time. A missing file returns an empty
        list.
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
    """Return the text body persisted to the run result file."""
    return result.text


def _format_failure_message(tool_name: str, error: Exception) -> str:
    """Format the message returned when a tool call fails."""
    return f"Tool '{tool_name}' failed: {error}"


def _build_error_task(error_message: str) -> Task[ToolResult]:
    """Build a completed failed task for a rejected call."""
    return task_manager.create_completed_task(error=ToolMessageError(error_message))


ParamValue: TypeAlias = str | int | float | bool | list | tuple | dict | None


class _ParameterizedType(Protocol):
    """A generic, union, or literal type form that carries type arguments."""

    @property
    def __args__(self) -> tuple: ...


_ParamType: TypeAlias = type | _ParameterizedType


class _Required:
    """Sentinel for a parameter that has no default value."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "<required>"


_REQUIRED = _Required()


@dataclass(frozen=True)
class Param:
    """Declaration of one tool parameter."""

    name: str
    type: _ParamType
    default: _Required | ParamValue = _REQUIRED
    description: str = ""
    kind: str | None = None
    min: int | float | None = None
    max: int | float | None = None
    step: int | float | None = None
    enum: list[str] | tuple[str, ...] | None = None

    @property
    def is_required(self) -> bool:
        """Return True when this parameter has no default value."""
        return self.default is _REQUIRED

    @property
    def wire_description(self) -> str:
        """Return the parameter description sent on the wire."""
        return _compose_param_description(self)

    @property
    def annotation(self) -> _ParamType:
        """Return the Python type annotation for this parameter."""
        if self.enum:
            return Literal[tuple(self.enum)]  # type: ignore[valid-type, return-value]
        return self.type


@dataclass(frozen=True)
class Contract:
    """Static metadata declared by a tool."""

    name: str
    description: str
    summary: str
    params: tuple[Param, ...]
    rules: tuple[str, ...]
    is_tracked: bool
    result_render_type: str | None = None
    is_result_long: bool = False


@dataclass(frozen=True)
class ToolResult:
    """Result returned by a tool call, including text and optional resource files."""

    text: str
    files: tuple[str, ...] | None = None
    render_type: str | None = None


_validation_adapters: dict[type, pydantic.TypeAdapter[dict]] = {}
_numeric_adapters: dict[type, pydantic.TypeAdapter[int | float]] = {}


def build_param_signature(params: tuple[Param, ...]) -> inspect.Signature:
    """Build a call signature from declared tool parameters.

    Args:
        params: Declared parameters with required parameters before optional ones.

    Returns:
        A signature with the parameters in declared order and a string return type.

    Raises:
        ValueError: If an optional parameter comes before a required parameter.
    """
    parameters = [
        inspect.Parameter(
            param.name,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            default=inspect.Parameter.empty if param.is_required else param.default,
            annotation=param.annotation,
        )
        for param in params
    ]
    return inspect.Signature(parameters, return_annotation=str)


def _build_validation_adapter(tool_cls: type[Tool]) -> pydantic.TypeAdapter[dict]:
    """Return the cached validator for a tool class."""
    adapter = _validation_adapters.get(tool_cls)
    if adapter is None:
        adapter = pydantic.TypeAdapter(tool_cls.build_validation_stub())
        _validation_adapters[tool_cls] = adapter
    return adapter


def _build_numeric_adapter(numeric_type: type) -> pydantic.TypeAdapter[int | float]:
    """Return the cached validator for a numeric type."""
    adapter = _numeric_adapters.get(numeric_type)
    if adapter is None:
        adapter = pydantic.TypeAdapter(numeric_type)
        _numeric_adapters[numeric_type] = adapter
    return adapter


def _extract_numeric_type(annotation: _ParamType) -> type | None:
    """Return int or float when a parameter annotation accepts numbers, or None."""
    if annotation in (int, float) and isinstance(annotation, type):
        return annotation
    for type_arg in get_args(annotation):
        if type_arg in (int, float) and isinstance(type_arg, type):
            return type_arg
    return None


def _build_tool_view(params: tuple[Param, ...], values: dict) -> dict:
    """Drop None entries for parameters whose declared type allows None."""
    nullable_names = {
        param.name for param in params if type(None) in get_args(param.type)
    }
    return {
        key: value
        for key, value in values.items()
        if not (value is None and key in nullable_names)
    }


def _format_bound(value: int | float) -> str:
    """Format a numeric limit for a validation error message."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _find_bound_violation(param: Param, value: ParamValue) -> str | None:
    """Return the limit error text when a numeric value falls out of range, or None."""
    if param.min is None and param.max is None:
        return None
    numeric_type = _extract_numeric_type(param.type)
    if numeric_type is None:
        return None
    try:
        numeric_value = _build_numeric_adapter(numeric_type).validate_python(value)
    except pydantic.ValidationError:
        return None
    if param.min is not None and not (numeric_value >= param.min):
        return f"Input should be greater than or equal to {_format_bound(param.min)}."
    if param.max is not None and not (numeric_value <= param.max):
        return f"Input should be less than or equal to {_format_bound(param.max)}."
    return None


def _format_validation_error(error: pydantic.ValidationError) -> str:
    """Format the first validation error as a single message string."""
    first_error = error.errors()[0]
    field_path = ".".join(str(part) for part in first_error["loc"])
    return f"Error: {field_path}: {first_error['msg']}."


class Tool(abc.ABC):
    """Base class for tools that validate inputs and run as tasks."""

    name: ClassVar[str]
    is_slow: ClassVar[bool] = False
    staleness: ClassVar[timedelta | None] = None
    description: ClassVar[str]
    summary: ClassVar[str]
    params: ClassVar[tuple[Param, ...]]
    rules: ClassVar[tuple[str, ...]] = ()
    is_tracked: ClassVar[bool] = True
    result_render_type: ClassVar[str | None] = None
    is_result_long: ClassVar[bool] = False

    def build_contract(self) -> Contract:
        """Collect the static metadata declared by this tool.

        Returns:
            The tool contract.

        Raises:
            AttributeError: If a required tool attribute is missing.
        """
        return Contract(
            self.name,
            self.description,
            self.summary,
            self.params,
            self.rules,
            self.is_tracked,
            self.result_render_type,
            self.is_result_long,
        )

    @classmethod
    def build_validation_stub(cls) -> Callable[..., dict]:
        """Build a function whose signature matches this tool's declared parameters.

        Returns:
            A function accepting the tool's declared parameters and returning them as a dict.
        """

        def _stub(**kwargs: ParamValue) -> dict:
            return dict(kwargs)

        _stub.__signature__ = build_param_signature(cls.params)  # type: ignore[attr-defined]
        _stub.__annotations__ = {
            param.name: param.annotation for param in cls.params
        } | {"return": str}
        return _stub

    def validate_params(self, params: dict) -> str | None:
        """Check call arguments against this tool's declared parameters.

        Args:
            params: The call arguments to check.

        Returns:
            The error text for the first problem, or None when all arguments are valid.
        """
        validation_input = _build_tool_view(self.params, params)
        try:
            _build_validation_adapter(type(self)).validate_python(validation_input)
        except pydantic.ValidationError as error:
            return _format_validation_error(error)
        for param in self.params:
            if param.name in params and params[param.name] is not None:
                violation = _find_bound_violation(param, params[param.name])
                if violation is not None:
                    return f"Error: {param.name}: {violation}"
        return None

    @property
    def dynamic_description(self) -> str:
        """Return the description sent on the wire, including the cache note when set."""
        return _compose_description(self.description, self.staleness)

    @abc.abstractmethod
    async def call(self, params: dict) -> ToolResult:
        """Execute the tool's work.

        Args:
            params: The call arguments.

        Returns:
            The tool result.
        """

    async def build_cache_key(self, _params: dict) -> str | None:
        """Build the cache key for a call, or return None when caching is disabled.

        Args:
            _params: The call arguments.

        Returns:
            A cache key string for this call, or None.
        """
        return None

    @final
    async def run(self, params: dict) -> Task[ToolResult]:
        """Validate params, reuse a cached run when eligible, or start a new task.

        Args:
            params: The raw call arguments.

        Returns:
            The task handle for this call.
        """
        validation_error = self.validate_params(params)
        if validation_error is not None:
            return _build_error_task(validation_error)
        completed_params = {
            param.name: params.get(param.name, param.default) for param in self.params
        }
        try:
            await self._record_resource_inputs(completed_params)
        except OSError as error:
            return _build_error_task(_format_failure_message(self.name, error))
        cache_key = await self._resolve_cache_key(completed_params)
        if cache_key is not None and self.staleness is not None:
            cached_task = await self._read_stored_result(
                cache_key, _format_cache_cutoff(self.staleness)
            )
            if cached_task is not None:
                return cached_task
        return await self._start_task(completed_params, cache_key)

    async def _resolve_cache_key(self, completed_params: dict) -> str | None:
        if self.staleness is None:
            return None
        return await self.build_cache_key(completed_params)

    async def _execute_call(
        self, completed_params: dict, task_id: uuid.UUID, run_path: str
    ) -> ToolResult:
        result = await self.call(completed_params)
        result_error = _validate_result(result)
        if result_error is not None:
            raise RuntimeError(result_error)
        if self.is_tracked:
            declared_render_type = result.render_type or self.result_render_type
            if declared_render_type in _TEXT_FAMILY_TOKENS:
                await store_module.tool_store.set_render_type(
                    str(task_id), declared_render_type
                )
            if result.files:
                await _write_files_sidecar(run_path, result.files)
        return result

    async def _start_task(
        self, completed_params: dict, cache_key: str | None
    ) -> Task[ToolResult]:
        task_id = uuid.uuid4()
        run_path = os.path.join(
            store_module.tool_store.data_dir, f"{store_module.RUNS_DIR}/{task_id}"
        )

        async def _run_body() -> ToolResult:
            return await self._execute_call(completed_params, task_id, run_path)

        task = await task_manager.create_task(
            _run_body,
            task_id=task_id,
            should_persist_record=self.is_tracked,
            stringify_result=_stringify_tool_result,
        )
        if self.is_tracked:
            await store_module.tool_store.insert(
                str(task_id), self.name, cache_key, completed_params
            )
        await task.wait_started()
        return task

    async def _record_resource_inputs(self, params: dict) -> None:
        for param in self.params:
            value = params.get(param.name)
            if param.kind == "resource" and isinstance(value, str) and value:
                await activity.record(value)
            elif param.kind == "resource_list" and isinstance(value, list):
                for resource_path in value:
                    if isinstance(resource_path, str) and resource_path:
                        await activity.record(resource_path)

    async def read_result_text(self, task: Task[ToolResult]) -> str:
        """Wait for a task and return the text shown to clients.

        Args:
            task: The task handle returned by run.

        Returns:
            The result text, cancellation notice, or error message.
        """
        try:
            result = await task.result()
            if result is None:
                raise RuntimeError("tool returned no result")
            return result.text
        except TaskCancelledError:
            return f"Task {task.id} was cancelled."
        except ToolMessageError as error:
            return str(error)
        except Exception as error:
            return _format_failure_message(self.name, error)

    async def _read_stored_result(
        self, cache_key: str, oldest_started_at: str
    ) -> Task[ToolResult] | None:
        cached_record = await store_module.tool_store.find_cache(
            self.name, cache_key, oldest_started_at
        )
        if cached_record is None:
            return None
        try:
            cached_text = await read_run_mcp_text(
                os.path.join(store_module.tool_store.data_dir, cached_record.run_dir)
            )
        except FileNotFoundError:
            return None
        return task_manager.create_completed_task(
            ToolResult(text=cached_text), task_id=uuid.UUID(cached_record.task_id)
        )
