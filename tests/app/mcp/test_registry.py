from __future__ import annotations

import ast
import inspect
import json
import os
import pathlib
import re
import textwrap
import types
import typing
from collections.abc import Callable
from typing import Literal, get_args

import pydantic
import pytest
from fastmcp.tools import FunctionTool

import app.lib.db.store as store_module
from app.lib.render.detect import TEXT_FAMILY_KINDS
from app.lib.tool.base import (
    Param,
    Tool,
    ToolMessageError,
    ToolResult,
    param_signature,
)
from app.lib.tool.truncation import (
    DEFAULT_TOOL_OUTPUT_LEN_LIMIT,
    MAX_CHAR_DESCRIPTION,
    MIN_TOOL_OUTPUT_LEN_LIMIT,
)
from app.mcp.registry import mcp, registered_tools, tool_class
from app.tools.brave_search import brave_search_tool
from app.tools.present_to_user import present_to_user_tool
from app.tools.read_media_or_doc import read_media_or_doc_tool
from app.tools.redfin import redfin_search_tool
from app.tools.web_fetch import web_fetch_tool
from tests.app.lib.tool.conftest import wait_terminal

# The expected client-layer max_char the wire appends to a long tool.
_MAX_CHAR_PARAM = Param(
    "max_char",
    int,
    DEFAULT_TOOL_OUTPUT_LEN_LIMIT,
    min=MIN_TOOL_OUTPUT_LEN_LIMIT,
    description=MAX_CHAR_DESCRIPTION,
)


async def test_wire_validation_header_names_tool_name() -> None:
    """The wire reports a bad call itself, naming the tool in the error header."""
    with pytest.raises(pydantic.ValidationError) as ei:
        await mcp.call_tool("web_fetch", {})
    err = ei.value.errors()[0]
    assert err["loc"] == ("url",)
    assert err["msg"] == "Missing required argument"
    assert str(ei.value).splitlines()[0] == "1 validation error for call[web_fetch]"


@pytest.mark.parametrize(
    ("instance", "name", "arguments", "loc", "msg"),
    [
        (redfin_search_tool, "redfin_search", {}, ("location",), "Missing required argument"),
        (
            redfin_search_tool,
            "redfin_search",
            {"location": "r", "zz": 1},
            ("zz",),
            "Unexpected keyword argument",
        ),
        (
            brave_search_tool,
            "brave_search",
            {"query": "q", "freshness": "x"},
            ("freshness",),
            "Input should be 'd', 'w', 'm' or 'y'",
        ),
    ],
)
async def test_wire_checks_args_before_the_tool_runs(
    monkeypatch: pytest.MonkeyPatch,
    instance: Tool,
    name: str,
    arguments: dict,
    loc: tuple,
    msg: str,
) -> None:
    """The wire reports a bad call itself and never hands it to the tool."""
    seen: list[dict] = []

    async def _spy(self: Tool, params: dict) -> ToolResult:
        seen.append(params)
        return ToolResult(text="should not run")

    monkeypatch.setattr(type(instance), "call", _spy)
    with pytest.raises(pydantic.ValidationError) as ei:
        await mcp.call_tool(name, arguments)
    err = ei.value.errors()[0]
    assert err["loc"] == loc
    assert err["msg"] == msg
    assert seen == []


async def test_served_tool_round_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _call(self: Tool, params: dict) -> ToolResult:
        return ToolResult(text=f"ok:{params['url']}")

    monkeypatch.setattr(type(web_fetch_tool), "call", _call)
    result = await mcp.call_tool("web_fetch", {"url": "http://round-trip.example"})
    assert result.content[0].text == "ok:http://round-trip.example"


# --- invariant suite: the tool class is the single source of truth ---

_REGISTERED_NAMES = sorted(registered_tools())


async def _wire_tool(name: str) -> FunctionTool:
    """The wire tool served under the given name."""
    return next(t for t in await mcp.list_tools() if t.name == name)


# The keys the registry adds to the wire schema.
_INJECTED_KEYS = ("description", "minimum", "maximum")


def _strip_injected[T](node: T) -> T:
    """The schema node with the added keys taken out."""
    if isinstance(node, dict):
        return {k: _strip_injected(v) for k, v in node.items() if k not in _INJECTED_KEYS}
    if isinstance(node, list):
        return [_strip_injected(v) for v in node]
    return node


# --- invariant 1: the single base run is the only entry ---


@pytest.mark.parametrize("name", _REGISTERED_NAMES)
def test_run_is_the_base_final_entry(name: str) -> None:
    """No tool overrides the single @final run. Every entry is the base."""
    cls = tool_class(name)
    assert cls.run is Tool.run, name


# --- invariant 2a: closure-signature integrity ---


def _assert_closure_lockstep(name: str, sig: inspect.Signature, cls: type[Tool]) -> None:
    """Check each declared param against the matching argument of the registered function.

    A long-result tool gains the client-layer max_char argument last.
    """
    expected = [p.name for p in cls.params] + (["max_char"] if cls.is_result_long else [])
    assert list(sig.parameters) == expected, name
    for p in cls.params:
        sp = sig.parameters[p.name]
        expected_type = Literal[tuple(p.enum)] if p.enum else p.type
        assert sp.annotation == expected_type, f"{name}.{p.name}"
        assert sp.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD, f"{name}.{p.name}"
        assert (sp.default is inspect.Parameter.empty) == p.required, f"{name}.{p.name}"
        if not p.required:
            assert sp.default == p.default, f"{name}.{p.name}"
    if cls.is_result_long:
        sp = sig.parameters["max_char"]
        assert sp.annotation is int, name
        assert sp.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD, name
        assert sp.default == DEFAULT_TOOL_OUTPUT_LEN_LIMIT, name


@pytest.mark.parametrize("name", _REGISTERED_NAMES)
async def test_closure_signature_synthesized_from_params(name: str) -> None:
    """Invariant 2a. The registered function takes the class params.

    An enum param is typed as its value set.
    """
    cls = tool_class(name)
    wire = await _wire_tool(name)
    _assert_closure_lockstep(name, inspect.signature(wire.fn), cls)


# --- invariant 2c: the validation stub shares one signature home with the wire ---


@pytest.mark.parametrize("name", _REGISTERED_NAMES)
def test_validation_stub_signature_matches_tool_params(name: str) -> None:
    """Invariant 2c. The stub used for checking params takes the tool's declared params.

    Those are the registered function's arguments minus the client-layer
    max_char the closure strips before the run.
    """
    cls = tool_class(name)
    assert inspect.signature(cls.validation_stub()) == param_signature(cls.params), name


# --- invariant 2b: served-schema differential (read the wire) ---


def _assert_served_param_descs_and_bounds(name: str, props: dict[str, dict], cls: type[Tool]) -> None:
    """Check that the served param descriptions and bounds match the class."""
    for p in cls.params:
        assert props[p.name]["description"] == p.wire_description, f"{name}.{p.name}"
        assert props[p.name].get("minimum") == p.min, f"{name}.{p.name}"
        assert props[p.name].get("maximum") == p.max, f"{name}.{p.name}"


@pytest.mark.parametrize("name", _REGISTERED_NAMES)
async def test_served_schema_tracks_wire_derivation(name: str) -> None:
    """Invariant 2b. The served schema adds only descriptions and bounds to what the tool declares.

    The served tool description is the one built from the class settings.
    """
    cls = tool_class(name)
    wire = await _wire_tool(name)
    rederived = FunctionTool.from_function(wire.fn).to_mcp_tool().inputSchema
    assert _strip_injected(wire.parameters) == _strip_injected(rederived), name
    _assert_served_param_descs_and_bounds(name, wire.parameters.get("properties", {}), cls)
    assert wire.to_mcp_tool().description == cls().dynamic_description, name


# --- invariant 2e: the client-layer max_char on the wire ---


@pytest.mark.parametrize("name", _REGISTERED_NAMES)
async def test_max_char_is_served_only_as_a_client_param(name: str) -> None:
    """The served max_char is the client's for a long tool, and the tool's own otherwise.

    A long tool declares no max_char of its own, and the client-appended one
    carries the shared smallest value and the resolved default. A tool that is
    not long serves a max_char only when it declares one, like the task tool.
    """
    cls = tool_class(name)
    wire = await _wire_tool(name)
    props = wire.parameters.get("properties", {})
    declared = any(p.name == "max_char" for p in cls.params)
    if cls.is_result_long:
        assert not declared, name
        prop = props["max_char"]
        assert prop["minimum"] == MIN_TOOL_OUTPUT_LEN_LIMIT, name
        assert str(DEFAULT_TOOL_OUTPUT_LEN_LIMIT) in prop["description"], name
    else:
        assert ("max_char" in props) == declared, name


@pytest.mark.parametrize("name", _REGISTERED_NAMES)
def test_max_char_is_undeclared_unless_it_is_a_tool_param(name: str) -> None:
    """No tool description names max_char, and only a tool that declares it serves one of its own."""
    cls = tool_class(name)
    assert "max_char" not in cls().dynamic_description, name


# The exact note text a cached tool adds to its description.
_EXPECTED_CACHE_NOTE_24H = "Duplicate calls within 24 hours reuse the stored result silently."


async def test_wire_carries_synthesized_notes_on_real_tools() -> None:
    """Real tools carry the cache note their reuse window calls for, and no async note."""
    redfin = (await _wire_tool("redfin_search")).to_mcp_tool().description
    assert redfin == tool_class("redfin_search")().dynamic_description
    assert "24 hours" in redfin
    assert "asynchronous" not in redfin
    read_media = (await _wire_tool("read_media_or_doc")).to_mcp_tool().description
    assert read_media == tool_class("read_media_or_doc")().dynamic_description
    assert _EXPECTED_CACHE_NOTE_24H in read_media
    assert "asynchronous" not in read_media


async def _freshness_wire(name: str) -> tuple[Param, dict]:
    """The declared freshness param and its served wire schema property."""
    cls = tool_class(name)
    freshness = next(p for p in cls.params if p.name == "freshness")
    wire = await _wire_tool(name)
    prop = wire.to_mcp_tool().model_dump()["inputSchema"]["properties"]["freshness"]
    return freshness, prop


@pytest.mark.parametrize("name", ("brave_search", "google_search"))
async def test_freshness_wire_carries_enum_note(name: str) -> None:
    """The wire description carries the enum note for both freshness params.

    The list of values comes from the enum, not from the param's own
    description.
    """
    freshness, prop = await _freshness_wire(name)
    assert "Valid values: d, w, m, y." in freshness.wire_description
    assert "Valid values: d, w, m, y." in prop["description"]


@pytest.mark.parametrize("name", ("brave_search", "google_search"))
async def test_freshness_wire_carries_declared_enum(name: str) -> None:
    """The freshness enum reaches the wire schema.

    The schema is flat, with a null default, the enum present, and no anyOf.
    The wire description carries the enum note.
    """
    freshness, prop = await _freshness_wire(name)
    assert prop["enum"] == ["d", "w", "m", "y"]
    assert prop["default"] is None
    assert prop["type"] == "string"
    assert prop["description"] == freshness.wire_description


# --- invariant 3d: the tool-level description restates no param-owned fact ---

# Split after a sentence ends, but not after a short form like e.g. or i.e.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])(?<![a-z]\.[a-z]\.)\s+|\n+")
_BULLET = re.compile(r"^(?:[-*\u2022]|\d+[.)])\s+")
_DELIMITERS = "`\"'"
_PLACEHOLDER_CHARS = "<>"
_MIN_SET_MATCH = 3
# Backticked text may hold spaces, quoted text may not.
_QUOTED_LITERAL = re.compile(r"``(.+?)``|`([^`]+)`|'([^'\s]+)'|\"([^\"\s]+)\"")
_PIPE_ALTERNATION = re.compile(r"[^\s|'\"]+(?:\|[^\s|'\"]+)+")
_FORMAT_TEMPLATE = re.compile(r"\b[A-Z]{2,}(?:-[A-Z0-9]{1,}){1,}\b")
_NUMERIC_DESCRIPTOR = re.compile(r"\b\d+-[A-Za-z][A-Za-z0-9]*\b")
_PIPE_JUNK = re.compile(r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$")
_EXAMPLE_FORM = re.compile(r"\bfor\s+(?:a\s+|an\s+)?example\b|\be\.g\.\b")


def _split_units(description: str) -> list[str]:
    """The sentences in a description."""
    return [u.strip() for u in _SENTENCE_SPLIT.split(description) if u.strip()]


def _whole_word(token: str, text: str) -> bool:
    """Whether the token shows up as a whole word in the text, ignoring case."""
    return re.search(r"(?<!\w)" + re.escape(token.casefold()) + r"(?!\w)", text.casefold()) is not None


def _strip_delimiters(token: str) -> str:
    """The token with backticks and quotes taken off both ends."""
    return token.strip().strip(_DELIMITERS).strip()


def _value_in(token: str, text: str) -> bool:
    """Whether the token shows up as a whole value in the text.

    A placeholder never matches.
    """
    stripped = _strip_delimiters(token).casefold()
    if not stripped or any(c in _PLACEHOLDER_CHARS for c in stripped):
        return False
    return re.search(r"(?<!\w)" + re.escape(stripped) + r"(?!\w)", text.casefold()) is not None


def _param_scopes(description: str, name: str) -> list[str]:
    """The parts of the description that talk about the named param."""
    units = _split_units(description)
    scopes: list[str] = []
    for index, unit in enumerate(units):
        if not _whole_word(name, unit):
            continue
        scopes.append(unit)
        if unit.endswith(":"):
            scopes.extend(_consecutive_bullets(units, index + 1))
    return scopes


def _consecutive_bullets(units: list[str], start: int) -> list[str]:
    """The run of bullet lines that starts at the given spot."""
    block: list[str] = []
    for follow in units[start:]:
        if not _BULLET.match(follow):
            break
        block.append(follow)
    return [" ".join(block)] if block else []


def _clause_i_default(p: Param, scope: str) -> bool:
    """Whether the scope spells out the param's own plain default value."""
    if p.required or p.default is None:
        return False
    if isinstance(p.default, (list, dict, tuple, set)):
        return False
    return bool(str(p.default)) and _value_in(str(p.default), scope)


# A value run is a row of words where every pair of neighbours is split only
# by a comma, a pipe, a slash, or the word "or" or "and". Any other word
# between two of them ends the run. Words in quotes or backticks still count.
_LIST_DELIM_GAP = re.compile(r"^\W*(?:,|\||/)\W*$")
_PUNCT_GAP = re.compile(r"^\W*$")
_RUN_CONNECTIVES = frozenset({"and", "or"})
_WORD_TOKEN = re.compile(r"\w+")


def _value_runs(scope: str) -> list[list[str]]:
    """The value runs found in the scope."""
    words = list(_WORD_TOKEN.finditer(scope))
    runs: list[list[str]] = []
    current: list[str] = []

    def flush() -> None:
        if current:
            runs.append(list(current))
            current.clear()

    for i, match in enumerate(words):
        if i > 0:
            gap = scope[words[i - 1].end():match.start()]
            connects = bool(_LIST_DELIM_GAP.match(gap))
            if not connects and match.group(0).casefold() in _RUN_CONNECTIVES and i + 1 < len(words):
                connects = (
                    bool(_PUNCT_GAP.match(gap))
                    and bool(_PUNCT_GAP.match(scope[match.end():words[i + 1].start()]))
                )
            if not connects and words[i - 1].group(0).casefold() in _RUN_CONNECTIVES:
                connects = bool(_PUNCT_GAP.match(gap))
            if not connects:
                flush()
        current.append(match.group(0))
    flush()
    return runs


def _clause_ii_enum_set(p: Param, scope: str) -> bool:
    """Whether the scope lists the param's allowed values."""
    if p.enum is None or not p.enum:
        return False
    values = {str(v).casefold() for v in p.enum}
    for run in _value_runs(scope):
        if sum(1 for token in run if token.casefold() in values) >= min(_MIN_SET_MATCH, len(p.enum)):
            return True
    return False


def _nullable_resolved(type_: type) -> type:
    """The type with None taken out of it."""
    origin = typing.get_origin(type_)
    if origin in (types.UnionType, typing.Union):
        args = [a for a in typing.get_args(type_) if a is not type(None)]
        return _nullable_resolved(args[0]) if len(args) == 1 else type_
    return type_


_SCALAR_NAMES: dict[type, str] = {
    str: "str", int: "int", float: "float", bool: "bool",
    list: "list", tuple: "tuple", set: "set", dict: "dict",
}


def _scalar_name(type_: type) -> str | None:
    """The plain name of the type, like str or list."""
    resolved = _nullable_resolved(type_)
    if resolved is None or resolved is type(None):
        return None
    origin = typing.get_origin(resolved)
    base = origin if origin is not None else resolved
    return _SCALAR_NAMES.get(base)


def _clause_iii_type_position(p: Param, scope: str) -> bool:
    """Whether the scope names the param's type in a type spot."""
    name = _scalar_name(p.type)
    if name is None:
        return False
    return re.search(r"[(,]\s*" + name + r"\s*[\),:]", scope) is not None


def _dedup_tokens(tokens: list[str]) -> tuple[str, ...]:
    """The tokens with repeats dropped, in the order they came."""
    seen: set[str] = set()
    out: list[str] = []
    for token in tokens:
        if token and token not in seen:
            seen.add(token)
            out.append(token)
    return tuple(out)


def _owned_families(description: str) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """The token families a param description owns.

    Returns:
        Quoted text, pipe choices, and format tokens, each with repeats dropped.
    """
    literals: list[str] = []
    for match in _QUOTED_LITERAL.finditer(description):
        stripped = _strip_delimiters(next(g for g in match.groups() if g is not None)).casefold()
        if stripped and not any(c in _PLACEHOLDER_CHARS for c in stripped):
            literals.append(stripped)
    pipe: list[str] = []
    for match in _PIPE_ALTERNATION.finditer(description):
        for member in match.group(0).split("|"):
            stripped = _PIPE_JUNK.sub("", member).casefold()
            if stripped and not any(c in _PLACEHOLDER_CHARS for c in stripped):
                pipe.append(stripped)
    formats: list[str] = []
    for pattern in (_FORMAT_TEMPLATE, _NUMERIC_DESCRIPTOR):
        for token in pattern.findall(description):
            stripped = token.casefold()
            if stripped and not any(c in _PLACEHOLDER_CHARS for c in stripped):
                formats.append(stripped)
    return _dedup_tokens(literals), _dedup_tokens(pipe), _dedup_tokens(formats)


def _clause_iv_owned_tokens(p: Param, scope: str) -> bool:
    """Whether the scope repeats a token the param description owns.

    The param's own name is never counted.
    """
    self_name = p.name.casefold()
    for family in _owned_families(p.description):
        tokens = tuple(t for t in family if t != self_name)
        if not tokens:
            continue
        threshold = min(_MIN_SET_MATCH, len(tokens)) if len(tokens) >= _MIN_SET_MATCH else 1
        if sum(1 for token in tokens if _value_in(token, scope)) >= threshold:
            return True
    return False


def _is_pinned_value(p: Param, value: str) -> bool:
    """Whether the value is one the param itself can take."""
    v = value.casefold()
    if _scalar_name(p.type) == "bool":
        return v in ("true", "false")
    if p.enum is not None:
        return v in {str(e).casefold() for e in p.enum}
    return (not p.required) and str(p.default).casefold() == v


def _clause_v_pinned_assignment(p: Param, scope: str) -> bool:
    """Whether the scope tells the reader to set the param to one of its own values.

    A call and an example do not count.
    """
    low = scope.casefold()
    name = re.escape(p.name.casefold())
    if re.search(r"(?<!\w)" + name + r"\s*\(", low):
        return False
    if _EXAMPLE_FORM.search(low) and re.search(r"(?<!\w)" + name + r"\s*=", low):
        return False
    match = re.search(r"(?<!\w)" + name + r"\s*=\s*('([^']*)'|\"([^\"]*)\"|(\S+))", low)
    if match is None:
        return False
    return _is_pinned_value(p, next(g for g in match.groups()[1:] if g is not None))


def _clause_vi_bound(p: Param, scope: str) -> bool:
    """Whether the scope spells out the param's smallest or largest value."""
    return any(bound is not None and _value_in(str(bound), scope) for bound in (p.min, p.max))


_CLAUSES: tuple[tuple[str, Callable[[Param, str], bool]], ...] = (
    ("default", _clause_i_default),
    ("enum-set", _clause_ii_enum_set),
    ("type-position", _clause_iii_type_position),
    ("owned-token", _clause_iv_owned_tokens),
    ("pinned-assignment", _clause_v_pinned_assignment),
    ("bound", _clause_vi_bound),
)


def _per_action_block(description: str, names: frozenset[str]) -> bool:
    """Whether the description has a required or optional line that lists param names."""
    for unit in _split_units(description):
        if unit.casefold().startswith(("required:", "optional:")) and any(_whole_word(n, unit) for n in names):
            return True
    return False


def _description_violations(description: str, params: tuple[Param, ...]) -> list[str]:
    """Every place the tool description repeats a fact a param already owns."""
    violations: list[str] = []
    for p in params:
        for scope in _param_scopes(description, p.name):
            for label, clause in _CLAUSES:
                if clause(p, scope):
                    violations.append(f"{p.name} {label}: {scope}")
    if _per_action_block(description, frozenset(p.name for p in params)):
        violations.append("per-action required:/optional: block enumerates param names")
    return violations


@pytest.mark.parametrize("name", _REGISTERED_NAMES)
def test_description_restates_no_param_fact(name: str) -> None:
    """Invariant 3d. The tool description repeats no fact a param already owns."""
    cls = tool_class(name)
    assert _description_violations(cls.description, cls.params) == [], name


# Each case holds a test id, the start of the violation text, a tool name, the param names
# to check, a description, and whether the case must be flagged.
_RESTATE_CASES: tuple[tuple[str, str, str, tuple[str, ...], str, bool], ...] = (
    ("limit default", "limit default", "github", ("limit",), "The limit parameter defaults to 10 results.", True),
    (
        "action enum-set",
        "action enum-set",
        "github",
        ("action",),
        "Use the action parameter with list_issues, search_issues or view_issue.",
        True,
    ),
    ("number type-position", "number type-position", "github", ("number",), "Pass number as (int) when paging results.", True),
    ("state owned-token", "state owned-token", "github", ("state",), "Pick state from 'open', 'closed' or 'all'.", True),
    (
        "nonstop pinned-assignment",
        "nonstop pinned-assignment",
        "google_flights",
        ("nonstop",),
        "Set nonstop=True to filter for non-stop flights.",
        True,
    ),
    ("max_char bound", "max_char bound", "github", ("max_char",), "The max_char parameter has a minimum of 500.", True),
    (
        "per-action",
        "per-action",
        "github",
        ("repo", "query", "limit"),
        "GitHub operations.\n"
        "Actions:\n"
        "  list_issues    List issues.\n"
        "                 required: repo, query\n"
        "                 optional: limit",
        True,
    ),
    # a lone format token counts on its own
    ("origin 3-letter standalone", "origin owned-token", "google_flights", ("origin",), "Origin must be a 3-letter airport code.", True),
    ("origin clean mention", "origin owned-token", "google_flights", ("origin",), "The origin airport is required for the search.", False),
    (
        "date_from clean mention",
        "date_from owned-token",
        "redfin_search",
        ("date_from",),
        "date_from narrows the active-listing window.",
        False,
    ),
    # an apostrophe must not swallow the quoted example
    (
        "files upload literal",
        "files owned-token",
        "present_to_user",
        ("files",),
        "Optional files are share paths like 'upload/cat.jpg'.",
        True,
    ),
    (
        "files clean mention",
        "files owned-token",
        "present_to_user",
        ("files",),
        "The files you send are resolved from the upload share.",
        False,
    ),
    # a sentence that tells you to set a value still counts
    (
        "oneway directive pinned",
        "oneway pinned-assignment",
        "google_flights",
        ("oneway",),
        "Use oneway=True when you need one-way fares.",
        True,
    ),
)


@pytest.mark.parametrize(("case_id", "prefix", "tool_name", "param_names", "description", "should_fire"), _RESTATE_CASES, ids=[c[0] for c in _RESTATE_CASES])
def test_clauses_flag_their_restatement_shapes(case_id: str, prefix: str, tool_name: str, param_names: tuple[str, ...], description: str, should_fire: bool) -> None:
    """Each check fires on a description that repeats a param fact and stays quiet on a clean one."""
    cls = tool_class(tool_name)
    wire = cls.params + ((_MAX_CHAR_PARAM,) if cls.is_result_long else ())
    params = tuple(p for p in wire if p.name in param_names)
    violations = _description_violations(description, params)
    assert any(v.startswith(prefix) for v in violations) is should_fire, (case_id, tool_name, prefix, violations)


def test_owned_token_format_template_family_fires_on_synthetic_param() -> None:
    """A format token in a param's own description fires the format check.

    A made-up param stands in. The live redfin_search date_from description
    carries no format token.
    """
    p = Param("date_probe", str, kind="date", description="Travel date in YYYY-MM-DD format.")
    assert _clause_iv_owned_tokens(p, "date_probe must be a YYYY-MM-DD date.") is True
    assert _clause_iv_owned_tokens(p, "The date narrows the window.") is False
    live = next(p for p in tool_class("redfin_search").params if p.name == "date_from")
    assert _clause_iv_owned_tokens(live, "date_from must be a YYYY-MM-DD date.") is False


# --- invariant 3g: a param base restates none of its own synthesized facts ---

_DEFAULT_WORD = re.compile(r"(?<!\w)defaults?\b")


def _clause_pg_default(p: Param, scope: str) -> bool:
    """Whether the scope repeats the param's own default value.

    The default must show up as a whole word, and the word default must be
    somewhere in the scope too.
    """
    if p.required or p.default is None:
        return False
    if isinstance(p.default, (list, dict, tuple, set)):
        return False
    if not (bool(str(p.default)) and _value_in(str(p.default), scope)):
        return False
    return _DEFAULT_WORD.search(scope) is not None


def _param_restate_violations(p: Param) -> list[str]:
    """Every added note fact that the param's own description repeats.

    An empty list means the description says only what the param is for.
    """
    violations: list[str] = []
    if _clause_pg_default(p, p.description):
        violations.append("default")
    if _clause_ii_enum_set(p, p.description):
        violations.append("enum-set")
    if _clause_vi_bound(p, p.description):
        violations.append("bound")
    return violations


def _param_restate_findings(cls: type[Tool]) -> list[str]:
    """Every param of cls whose own description repeats one of its note facts.

    Each line names the param and the facts it repeats.
    """
    findings: list[str] = []
    for p in cls.params:
        violations = _param_restate_violations(p)
        if violations:
            findings.append(f"{p.name}: {violations}")
    return findings


@pytest.mark.parametrize("name", _REGISTERED_NAMES)
def test_param_description_restates_no_synthesized_fact(name: str) -> None:
    """Invariant 3g. A param's own description repeats none of its note facts.

    The facts are the default value, the list of enum values, and the smallest
    or largest allowed value.
    """
    cls = tool_class(name)
    assert _param_restate_findings(cls) == [], name


# Each case checks one 3g clause against a made-up param.
# A case holds an id, a param, a scope, a clause, and whether it must fire.
_PARAM_RESTATE_CASES: tuple[tuple[str, Param, str, str, bool], ...] = (
    ("default fires", Param("n", int, 10, description="probe"), "defaults to 10", "default", True),
    ("default word missing", Param("n", str, "open", description="probe"),
     "Used by: list_issues ('open'|'closed'|'all').", "default", False),
    ("none default clean", Param("n", str, None, description="probe"),
     "Defaults to the repo's default branch.", "default", False),
    ("empty default clean", Param("n", str, "", description="probe"),
     "Set when you want an answer.", "default", False),
    ("enum flat run", Param("n", str, None, enum=("d", "w", "m", "y"), description="probe"),
     "d, w, m, y", "enum-set", True),
    ("enum cued run", Param("n", str, None, enum=("d", "w", "m", "y"), description="probe"),
     "One of d, w, m, y", "enum-set", True),
    ("enum per-value meanings clean", Param("n", str, None, enum=("d", "w", "m", "y"), description="probe"),
     "d is 24 hours, w is 7 days, m is 1 month, y is 1 year", "enum-set", False),
    ("bound fires", Param("n", int, None, min=1, description="probe"), "minimum 1", "bound", True),
)


@pytest.mark.parametrize(("case_id", "p", "scope", "clause", "should_fire"), _PARAM_RESTATE_CASES,
                         ids=[c[0] for c in _PARAM_RESTATE_CASES])
def test_param_restate_clauses_flag_their_shapes(case_id: str, p: Param, scope: str, clause: str, should_fire: bool) -> None:
    """Each 3g clause fires on a description that repeats a fact and stays quiet on a clean one.

    A flat list of values fires, and per value meanings do not.
    """
    assert sorted(registered_tools()) == _REGISTERED_NAMES
    fired = clause in _param_restate_violations(Param(p.name, p.type, p.default, scope,
                                                      kind=p.kind, min=p.min, max=p.max,
                                                      step=p.step, enum=p.enum))
    assert fired is should_fire, (case_id, scope)


# --- invariant 3h: no hand-written kind boilerplate in any param base ---

# The marker each kind bans from a param's own description.
_KIND_BOILERPLATE_MARKERS: dict[str, str] = {
    "date": "YYYY-MM-DD",
    "url": "http",
    "resource": "send_or_receive_file",
    "resource_list": "send_or_receive_file",
}


def _kind_boilerplate_findings(cls: type[Tool]) -> list[str]:
    """Every kinded param base of cls carrying its kind's distinctive marker."""
    findings: list[str] = []
    for p in cls.params:
        marker = _KIND_BOILERPLATE_MARKERS.get(p.kind or "")
        if marker is None:
            continue
        if marker.isascii() and marker.islower():
            found = re.search(marker, p.description, re.IGNORECASE) is not None
        else:
            found = marker in p.description
        if found:
            findings.append(f"{p.name}: kind {p.kind!r} marker {marker!r}")
    return findings


@pytest.mark.parametrize("name", _REGISTERED_NAMES)
def test_param_base_has_no_kind_boilerplate(name: str) -> None:
    """Invariant 3h. A param's own description carries none of its kind's marker.

    The kind note is added for it.
    """
    cls = tool_class(name)
    assert _kind_boilerplate_findings(cls) == [], name


# --- invariant 3i: no failure strings in any param base ---

# Failure wording a param's own description may not use.
_PARAM_FAILURE_BOILERPLATE = re.compile(
    r"Tool\s*'[^']+'\s+failed"
    r"|on\s+failure"
    r"|(?<!\w)rejected(?!\w)"
    r"|(?<!\w)failed(?!\w)"
    r"|is\s+an\s+error"
    r"|returns\s+an\s+error",
    re.IGNORECASE,
)


def _param_failure_findings(cls: type[Tool]) -> list[str]:
    """Every param base of cls carrying a failure-string marker."""
    return [p.name for p in cls.params if _PARAM_FAILURE_BOILERPLATE.search(p.description) is not None]


@pytest.mark.parametrize("name", _REGISTERED_NAMES)
def test_param_base_has_no_failure_strings(name: str) -> None:
    """Invariant 3i. A param's own description says nothing about failures."""
    cls = tool_class(name)
    assert _param_failure_findings(cls) == [], name


# --- invariant 3j: "resource" is the only client-visible term for the resource ---

_SHARE_TERM = re.compile(r"(?<!\w)share(?!\w)")


def _share_term_findings(cls: type[Tool]) -> list[str]:
    """Every text a client sees on cls that uses the word "share".

    The list holds description, summary, and the name of each param whose
    text uses the word.
    """
    findings: list[str] = []
    if _SHARE_TERM.search(cls().dynamic_description) is not None:
        findings.append("description")
    if _SHARE_TERM.search(cls.summary) is not None:
        findings.append("summary")
    findings.extend(p.name for p in cls.params if _SHARE_TERM.search(p.wire_description) is not None)
    return findings


@pytest.mark.parametrize("name", _REGISTERED_NAMES)
def test_client_visible_text_has_no_share_term(name: str) -> None:
    """Invariant 3j. No text a client sees uses the word "share" for the resource.

    The tool description, the summary, and every param description are checked
    with their notes already added. Result text made while a tool runs is not
    checked. The only word for the resource is "resource", and the WebDAV
    service is called "the WebDAV endpoint at /dav".
    """
    cls = tool_class(name)
    assert _share_term_findings(cls) == [], name


# --- invariant 3k: a param base self-describes no optionality ---

# A param description that starts with "optional", "an optional", or "the
# optional" says what the added optional note already says.
_OPTIONAL_SELF_DESCRIBE = re.compile(r"^(?:the\s+|an\s+)?optional\b", re.IGNORECASE)


def _optional_self_describe_findings(cls: type[Tool]) -> list[str]:
    """Every param of cls whose own description calls itself optional."""
    return [p.name for p in cls.params if _OPTIONAL_SELF_DESCRIBE.match(p.description.strip()) is not None]


@pytest.mark.parametrize("name", _REGISTERED_NAMES)
def test_param_base_self_describes_no_optionality(name: str) -> None:
    """Invariant 3k. A param's own description never calls itself optional.

    It may not start with "optional", "an optional", or "the optional". The
    optional note is added for it.
    """
    cls = tool_class(name)
    assert _optional_self_describe_findings(cls) == [], name


# --- invariant 3l: a param whose wire annotation takes None defaults to None ---

def _none_tolerant_default_findings(cls: type[Tool]) -> list[str]:
    """Every non-required param of cls whose projected annotation takes None without a None default."""
    findings: list[str] = []
    for p in cls.params:
        if type(None) not in get_args(p.annotation):
            continue
        if not p.required and p.default is not None:
            findings.append(f"{p.name}: {p.annotation} with default {p.default!r}")
    return findings


@pytest.mark.parametrize("name", _REGISTERED_NAMES)
def test_param_none_tolerant_annotation_defaults_to_none(name: str) -> None:
    """Invariant 3l. A param whose wire annotation takes None defaults to None.

    The wire invites a null to such a param, so the null must be the value the
    param settles on. A required param carries no default to clash with the
    null, and a param with an enum projects a bare Literal that takes no null.
    """
    cls = tool_class(name)
    assert _none_tolerant_default_findings(cls) == [], name


# --- invariant 3e: no hand-written behavioural notes in any base description ---

_HANDWRITTEN_NOTE = re.compile(r"asynchronous|cached result|within\s+\d+\s+hours", re.IGNORECASE)


@pytest.mark.parametrize("name", _REGISTERED_NAMES)
def test_description_has_no_handwritten_behavioural_notes(name: str) -> None:
    """No tool description is written with its own async or cache note."""
    cls = tool_class(name)
    assert _HANDWRITTEN_NOTE.search(cls.description) is None, name


# --- invariant 3f: no failure-string boilerplate in any base description ---

_FAILURE_BOILERPLATE = re.compile(r"Tool\s*'[^']+'\s+failed|on\s+failure", re.IGNORECASE)


@pytest.mark.parametrize("name", _REGISTERED_NAMES)
def test_description_has_no_failure_boilerplate(name: str) -> None:
    """No tool description spells out the failure message."""
    cls = tool_class(name)
    assert _FAILURE_BOILERPLATE.search(cls.description) is None, name


# --- invariant 4: contract completeness ---

_KIND_VOCABULARY = frozenset({"text", "longtext", "url", "date", "resource", "resource_list"})
_MAX_SUMMARY_LEN = 80


def _contains_literal(type_: type) -> bool:
    """Whether the type holds a Literal."""
    if typing.get_origin(type_) is typing.Literal:
        return True
    if typing.get_origin(type_) in (types.UnionType, typing.Union):
        return any(_contains_literal(a) for a in typing.get_args(type_))
    return False


def _assert_param_contract(where: str, p: Param) -> None:
    """Check one param against the invariant 4 rules for kind, type, enum, and bounds."""
    if p.kind is not None:
        assert p.kind in _KIND_VOCABULARY, f"{where}: kind {p.kind!r}"
    assert not _contains_literal(p.type), f"{where}: Literal in Param.type"
    resolved = _nullable_resolved(p.type)
    if p.kind == "date":
        assert resolved is str, f"{where}: kind date requires a str type"
    if p.kind == "resource_list":
        assert typing.get_origin(resolved) is list, f"{where}: kind resource_list requires a list type"
    if p.min is not None or p.max is not None:
        assert resolved in (int, float), f"{where}: min/max require a numeric type"
    for bound in (p.min, p.max, p.step):
        if bound is not None:
            assert isinstance(bound, (int, float)) and not isinstance(bound, bool), f"{where}: bound {bound!r}"
    if p.enum is not None:
        assert isinstance(p.enum, (list, tuple)) and p.enum, f"{where}: set enum needs a non-empty enum"
        assert all(isinstance(v, str) for v in p.enum), f"{where}: enum values must be strings"


def _assert_all_params(name: str, params: tuple[Param, ...]) -> None:
    """Check every param of the contract against the invariant 4 rules."""
    for p in params:
        _assert_param_contract(f"{name}.{p.name}", p)


@pytest.mark.parametrize("name", _REGISTERED_NAMES)
async def test_contract_is_complete_and_consistent(name: str) -> None:
    """Invariant 4. Every tool declares a full contract.

    It has a description, a one line summary of at most 80 characters, and a description for
    every param. Its served schema stands on its own.
    """
    cls = tool_class(name)
    contract = cls().contract()
    assert isinstance(contract.params, tuple), name
    assert contract.description, name
    assert contract.summary, name
    assert "\n" not in contract.summary, name
    assert len(contract.summary) <= _MAX_SUMMARY_LEN, name
    assert all(p.description for p in contract.params), name
    _assert_all_params(name, contract.params)
    wire = await _wire_tool(name)
    dumped = json.dumps(wire.parameters)
    assert "$defs" not in dumped, name
    assert "$ref" not in dumped, name


# --- invariant 4b: the declared result render kind is a text-family token ---


@pytest.mark.parametrize("name", _REGISTERED_NAMES)
def test_result_render_type_is_a_text_family_token(name: str) -> None:
    token = tool_class(name).result_render_type
    if token is not None:
        assert token in {k.value for k in TEXT_FAMILY_KINDS}


# --- invariant 7: call() mapping coverage ---

# The task tool reads its params in run, not in call.
_RUN_ENTRY_PASSTHROUGH: dict[str, frozenset[str]] = {
    "task": frozenset({"action", "task_id", "wait", "wait_timeout", "offset", "max_char", "tail"}),
}


def _params_dict_reads(node: ast.AST) -> str | None:
    """The param name a params lookup reads, or None."""
    if (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id == "params"
        and isinstance(node.slice, ast.Constant)
        and isinstance(node.slice.value, str)
    ):
        return node.slice.value
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "params"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    ):
        return node.args[0].value
    return None


def _call_param_reads(cls: type[Tool]) -> set[str]:
    """The param names a tool reads inside call."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(cls.call)))
    return {read for node in ast.walk(tree) if (read := _params_dict_reads(node)) is not None}


@pytest.mark.parametrize("name", _REGISTERED_NAMES)
def test_call_reads_every_declared_param(name: str) -> None:
    """Invariant 7. Every declared param is read by call, unless the base class handles it or run reads it."""
    cls = tool_class(name)
    reads = _call_param_reads(cls)
    exempt = _RUN_ENTRY_PASSTHROUGH.get(name, frozenset())
    missing = {p.name for p in cls.params if p.name not in reads and p.name not in exempt}
    assert not missing, f"{name}: call() never reads {sorted(missing)}"


# --- invariant 8: every tool is exercised ---

_TOOLS_TEST_DIR = pathlib.Path(__file__).resolve().parents[3] / "tests" / "app" / "tools"


def _tool_referenced_in_tests(name: str) -> bool:
    """Whether a tool test file names the tool."""
    pattern = re.compile(r"(?<!\w)" + re.escape(name) + r"(?!\w)")
    return any(pattern.search(path.read_text()) for path in _TOOLS_TEST_DIR.glob("test_*.py"))


@pytest.mark.parametrize("name", _REGISTERED_NAMES)
def test_every_tool_is_exercised_by_a_tool_test(name: str) -> None:
    """Invariant 8. Every registered tool is named in at least one tool test file."""
    assert tool_class(name).name == name, name
    assert _tool_referenced_in_tests(name), f"no test module under tests/app/tools/ references {name!r}"


# --- invariant 8b: the slow-tool set is fixed ---

_SLOW_TOOL_NAMES = {
    "browser_use",
    "edit_image",
    "generate_image",
    "generate_video",
    "redfin_search",
    "rednote_research",
    "research",
}


def test_the_slow_tool_set_is_the_ack_answering_tools() -> None:
    """Invariant 8b. Exactly these registered tools answer a call with the ack while their run continues."""
    slow = {name for name in _REGISTERED_NAMES if tool_class(name).is_slow}
    assert slow == _SLOW_TOOL_NAMES


# --- invariant 9: run-creation pipeline contract + structural guard ---


async def _run_state(tool_name: str) -> tuple[int, int, int]:
    """How many rows and run folders the store holds for one tool."""
    stats = [s for s in await store_module.tool_stats() if s["tool_name"] == tool_name]
    tool_rows = sum(s["completed"] + s["running"] + s["failed"] + s["cancelled"] for s in stats)
    task_rows = await store_module.total_count()
    runs = os.path.join(store_module.tool_store.data_dir, "runs")
    run_dirs = len(os.listdir(runs)) if os.path.isdir(runs) else 0
    return tool_rows, task_rows, run_dirs


async def test_tracked_invoke_creates_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """Invariant 9a. A tracked run makes a tool row, a task row, and a run folder."""
    async def _call(self: Tool, params: dict) -> ToolResult:
        return ToolResult(text=f"done:{params['url']}")

    monkeypatch.setattr(type(web_fetch_tool), "call", _call)
    before = await _run_state("web_fetch")
    result = await mcp.call_tool("web_fetch", {"url": "http://tracked-run.example"})
    assert result.content[0].text == "done:http://tracked-run.example"
    after = await _run_state("web_fetch")
    assert (after[0] - before[0], after[1] - before[1], after[2] - before[2]) == (1, 1, 1)


async def test_untracked_creates_no_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """Invariant 9a. An untracked run makes no row and no folder."""
    async def _call(self: Tool, params: dict) -> ToolResult:
        return ToolResult(text=f"direct:{params['title']}")

    monkeypatch.setattr(type(present_to_user_tool), "call", _call)
    before = await _run_state("present_to_user")
    result = await mcp.call_tool("present_to_user", {"title": "hello"})
    assert result.content[0].text == "direct:hello"
    assert await _run_state("present_to_user") == before


async def test_precheck_guard_is_a_failed_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """Invariant 9a. A guard before work makes a failed run with the tool's own words, no result file."""
    async def _call(self: Tool, params: dict) -> ToolResult:
        if params["url"] == "http://precheck.example":
            raise ToolMessageError("pre-check string")
        return ToolResult(text=f"done:{params['url']}")

    monkeypatch.setattr(type(web_fetch_tool), "call", _call)
    before = await _run_state("web_fetch")
    result = await mcp.call_tool("web_fetch", {"url": "http://precheck.example"})
    assert result.content[0].text == "pre-check string"
    after = await _run_state("web_fetch")
    assert (after[0] - before[0], after[1] - before[1], after[2] - before[2]) == (1, 1, 1)
    row = (await store_module.task_store.list())[-1]
    assert row.status == "failed"
    files = os.listdir(os.path.join(store_module.tool_store.data_dir, row.run_dir))
    assert "result" not in files


async def test_cache_hit_adds_no_new_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """Invariant 9a. A reused result comes back from the first run and adds no row or folder."""
    async def _call(self: Tool, params: dict) -> ToolResult:
        return ToolResult(text=f"cached:{params['resource_path']}")

    monkeypatch.setattr(type(read_media_or_doc_tool), "call", _call)
    path = "upload/probe-cache.txt"
    first = await mcp.call_tool("read_media_or_doc", {"resource_path": path})
    assert first.content[0].text == f"cached:{path}"
    before = await _run_state("read_media_or_doc")
    result = await mcp.call_tool("read_media_or_doc", {"resource_path": path})
    assert result.content[0].text == f"cached:{path}"
    assert await _run_state("read_media_or_doc") == before


async def test_slow_cache_hit_returns_the_result_not_the_ack(monkeypatch: pytest.MonkeyPatch) -> None:
    """A slow tool that reuses a stored run hands back the stored result, not the ack."""
    async def _call(self: Tool, params: dict) -> ToolResult:
        return ToolResult(text=f"cached:{params['location']}")

    monkeypatch.setattr(type(redfin_search_tool), "call", _call)
    location = "12 Cache Lane"
    first = await mcp.call_tool("redfin_search", {"location": location})
    assert first.content[0].text.startswith("Task started")
    task_id = first.content[0].text.split("task_id=")[1].split(".")[0]
    await wait_terminal(store_module.task_store, task_id)
    before = await _run_state("redfin_search")
    result = await mcp.call_tool("redfin_search", {"location": location})
    assert result.content[0].text == f"cached:{location}"
    assert "Task started" not in result.content[0].text
    assert await _run_state("redfin_search") == before





def _ack_text(task_id: str) -> str:
    """The task acknowledgement the wire answers a running slow call with."""
    return (
        f"Task started. task_id={task_id}. Retrieve the result with the task tool: "
        f"task(action='result', task_id='{task_id}')."
    )


async def test_slow_ack_is_byte_identical(monkeypatch: pytest.MonkeyPatch) -> None:
    """The wire answers a running slow call with the task acknowledgement, character for character."""
    async def _call(self: Tool, params: dict) -> ToolResult:
        return ToolResult(text=f"slow:{params['location']}")

    monkeypatch.setattr(type(redfin_search_tool), "call", _call)
    result = await mcp.call_tool("redfin_search", {"location": "1 Slow Lane"})
    text = result.content[0].text
    task_id = text.split("task_id=")[1].split(".")[0]
    assert text == _ack_text(task_id)
    await wait_terminal(store_module.task_store, task_id)


async def test_slow_ack_is_never_truncated(monkeypatch: pytest.MonkeyPatch) -> None:
    """A long-result slow tool gets the plain ack, with no truncation note anywhere in it."""
    async def _call(self: Tool, _params: dict) -> ToolResult:
        return ToolResult(text="x" * 30_000)

    monkeypatch.setattr(type(redfin_search_tool), "is_result_long", True)
    monkeypatch.setattr(type(redfin_search_tool), "call", _call)
    result = await mcp.call_tool("redfin_search", {"location": "2 Long Lane"})
    text = result.content[0].text
    task_id = text.split("task_id=")[1].split(".")[0]
    assert text == _ack_text(task_id)
    assert "[Truncated" not in text
    await wait_terminal(store_module.task_store, task_id)


async def test_slow_finished_run_truncates_with_the_run_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """A finished long slow run is cut for the client, with the run's id in the note.

    A finished run reaching the client is the reused stored result, so the run
    is made by one acked call and fetched back with a repeat inside the window.
    """
    limit = DEFAULT_TOOL_OUTPUT_LEN_LIMIT
    total = 30_000

    async def _call(self: Tool, _params: dict) -> ToolResult:
        return ToolResult(text="x" * total)

    monkeypatch.setattr(type(redfin_search_tool), "is_result_long", True)
    monkeypatch.setattr(type(redfin_search_tool), "call", _call)
    first = await mcp.call_tool("redfin_search", {"location": "3 Long Lane"})
    task_id = first.content[0].text.split("task_id=")[1].split(".")[0]
    await wait_terminal(store_module.task_store, task_id)
    result = await mcp.call_tool("redfin_search", {"location": "3 Long Lane"})
    assert result.content[0].text == (
        "x" * limit
        + f"\n[Truncated: showing the first {limit} of {total} chars. "
        f"task_id={task_id} \u2014 fetch the remaining {total - limit} chars with "
        f"task(action='result', task_id='{task_id}', offset={limit}, max_char={total - limit}).]"
    )


# --- MCP-path truncation: the wire cuts the result for the client ---

_LONG_CHARS = 30_000


def _truncated_note(text: str, limit: int, total: int, task_id: str) -> str:
    """The truncation note appended to a cut result."""
    return (
        text
        + f"\n[Truncated: showing the first {limit} of {total} chars. "
        f"task_id={task_id} \u2014 fetch the remaining {total - limit} chars with "
        f"task(action='result', task_id='{task_id}', offset={limit}, max_char={total - limit}).]"
    )


async def _first_task_id() -> str:
    """The task id of the run, read from its task row."""
    rows = await store_module.task_store.list()
    return rows[0].task_id


class TestMcpPathTruncation:
    """The wire's truncation note on a served long tool, asserted byte for byte."""

    async def test_long_result_truncated_at_the_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        limit = DEFAULT_TOOL_OUTPUT_LEN_LIMIT
        total = _LONG_CHARS

        async def _call(self: Tool, _params: dict) -> ToolResult:
            return ToolResult(text="x" * total)

        monkeypatch.setattr(type(web_fetch_tool), "call", _call)
        result = await mcp.call_tool("web_fetch", {"url": "http://trunc-default.example"})
        assert result.content[0].text == _truncated_note("x" * limit, limit, total, await _first_task_id())

    async def test_long_result_truncated_at_a_custom_limit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        limit = 1_000
        total = _LONG_CHARS

        async def _call(self: Tool, _params: dict) -> ToolResult:
            return ToolResult(text="x" * total)

        monkeypatch.setattr(type(web_fetch_tool), "call", _call)
        result = await mcp.call_tool(
            "web_fetch", {"url": "http://trunc-custom.example", "max_char": limit}
        )
        assert result.content[0].text == _truncated_note("x" * limit, limit, total, await _first_task_id())

    async def test_max_char_below_the_floor_behaves_as_the_floor(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A max_char under the floor is raised to the floor, not rejected."""
        limit = MIN_TOOL_OUTPUT_LEN_LIMIT
        total = _LONG_CHARS

        async def _call(self: Tool, _params: dict) -> ToolResult:
            return ToolResult(text="x" * total)

        monkeypatch.setattr(type(web_fetch_tool), "call", _call)
        result = await mcp.call_tool("web_fetch", {"url": "http://trunc-floor.example", "max_char": 50})
        assert result.content[0].text == _truncated_note("x" * limit, limit, total, await _first_task_id())

    async def test_short_result_comes_back_whole(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def _call(self: Tool, _params: dict) -> ToolResult:
            return ToolResult(text="ok")

        monkeypatch.setattr(type(web_fetch_tool), "call", _call)
        result = await mcp.call_tool("web_fetch", {"url": "http://trunc-short.example"})
        assert result.content[0].text == "ok"
        assert "[Truncated" not in result.content[0].text

    async def test_cache_hit_truncates_with_the_first_run_id(self, monkeypatch: pytest.MonkeyPatch) -> None:
        limit = DEFAULT_TOOL_OUTPUT_LEN_LIMIT
        total = _LONG_CHARS
        path = "upload/trunc-cache.txt"
        calls = 0

        async def _call(self: Tool, _params: dict) -> ToolResult:
            nonlocal calls
            calls += 1
            return ToolResult(text="x" * total)

        monkeypatch.setattr(type(read_media_or_doc_tool), "call", _call)
        await mcp.call_tool("read_media_or_doc", {"resource_path": path})
        first_id = await _first_task_id()
        result = await mcp.call_tool("read_media_or_doc", {"resource_path": path})
        assert result.content[0].text == _truncated_note("x" * limit, limit, total, first_id)
        rows = await store_module.task_store.list()
        assert len(rows) == 1
        assert calls == 1


# --- registration source shape: one register line per tool, no tool text ---

_REGISTRY_SOURCE = (pathlib.Path(__file__).resolve().parents[3] / "app" / "mcp" / "registry.py").read_text()
_SERVER_SOURCE = (pathlib.Path(__file__).resolve().parents[3] / "app" / "server.py").read_text()


def _classic_registration_violations(source: str, names: set[str]) -> list[str]:
    """Find each direct tool registration block that names one of the given tools.

    Args:
        source: Text of the registry file.
        names: Tool names to look for.

    Returns:
        The first line of every block that names one of them.
    """
    violations: list[str] = []
    block: list[str] = []
    in_call = False
    for line in source.splitlines():
        if in_call:
            block.append(line)
            if line.startswith(")("):
                if names & set(re.findall(r"\b\w+\b", " ".join(block))):
                    violations.append(block[0])
                in_call = False
                block = []
        elif "mcp.tool(" in line:
            in_call = True
            block = [line]
    return violations


async def test_registry_py_carries_only_register_tool_lines() -> None:
    """The registry file registers each tool with one line and holds no tool text.

    Every tool gets one register line, none is registered the direct way, no name or
    description is written on those lines, and the served tools match the registry.
    The server file carries no registration line and no tool import.
    """
    names = {cls.name for cls in registered_tools().values()}
    lines = _REGISTRY_SOURCE.splitlines()
    assert sum(1 for line in lines if "register_tool(mcp," in line) == len(registered_tools())
    assert _classic_registration_violations(_REGISTRY_SOURCE, names) == []
    assert all(
        "name=" not in line and "description=" not in line for line in lines if "register_tool(" in line
    )
    assert all(cls.description not in _REGISTRY_SOURCE for cls in registered_tools().values())
    assert {t.name for t in await mcp.list_tools()} == set(registered_tools())
    server_lines = _SERVER_SOURCE.splitlines()
    assert not any("register_tool(" in line for line in server_lines)
    assert not any(line.startswith("from app.tools.") for line in server_lines)
