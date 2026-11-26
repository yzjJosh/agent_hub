"""Tests for fragment building."""

import re
from pathlib import Path

import app.lib.render.fragments as frag_module
from app.lib.render.classify import RenderKind
from app.lib.render.fragments import build_fragment, build_fragment_sync

FIX = Path(__file__).parent / "fixtures"
_CSS = Path(__file__).resolve().parents[4] / "app" / "console" / "static" / "css" / "view.css"
_SCRIPT_RE = re.compile(r"<script", re.IGNORECASE)
_STYLE_RE = re.compile(r"<style>(.*?)</style>", re.DOTALL)
# The rich limit is 1 MB and the raw window is 256 KB.
_KB = 1024
_RICH_MAX = _KB * _KB
_RAW_MAX = _KB * 256
# Python source with 12 lines.
_PY_SOURCE = '''def summarize(items: list[str]) -> str:
    """Join items into a numbered summary line."""
    parts: list[str] = []
    for i, item in enumerate(items):
        parts.append(f"{i + 1}. {item}")
    joined = "; ".join(parts)
    total = len(parts)
    return f"total={total}: {joined}"


if __name__ == "__main__":
    print(summarize(["alpha", "beta", "gamma"]))
'''


def _read(name: str) -> str:
    return (FIX / name).read_text(encoding="utf-8")


async def test_markdown_gfm_fragment() -> None:
    built = await build_fragment(RenderKind.MARKDOWN, str(FIX / "sample.md"), len(_read("sample.md")))
    assert built.metadata["mode"] == "doc"
    assert built.html.startswith('<div class="md-doc">')
    assert "<table>" in built.html and "<s>" in built.html
    assert 'type="checkbox"' in built.html
    assert _SCRIPT_RE.search(built.html) is None


async def test_markdown_tables_wrapped_in_scroll_div() -> None:
    built = await build_fragment(RenderKind.MARKDOWN, str(FIX / "sample.md"), len(_read("sample.md")))
    # every table sits in a horizontal scroll container
    assert '<div class="md-table-scroll"><table>' in built.html
    assert "</table></div>" in built.html
    assert built.html.count('<div class="md-table-scroll"><table>') == built.html.count("</table></div>") == 1
    # the css lets the table outgrow the wrapper on phones, so the scroll bar can appear
    css = _CSS.read_text(encoding="utf-8")
    i = css.index("@media (max-width:640px)")
    block = css[i:css.index("\n}", i)]
    assert "width:auto" in block
    assert "table-layout:auto" in block
    assert "min-width:max-content" in block
    assert "white-space:nowrap" in block


async def test_code_fragment_cl_wrappers_and_line_count(tmp_path: Path) -> None:
    p = tmp_path / "sample.py"
    p.write_text(_PY_SOURCE, encoding="utf-8")
    built = await build_fragment(RenderKind.CODE, str(p), p.stat().st_size)
    assert built.metadata["mode"] == "code"
    assert built.metadata["language"] == "python"
    assert built.metadata["line_count"] == 12
    assert built.html.count('<div class="cl">') == 12
    assert _SCRIPT_RE.search(built.html) is None


async def test_html_fragment_is_code(tmp_path: Path) -> None:
    p = tmp_path / "input.html"
    p.write_text("<p>hi</p>", encoding="utf-8")
    built = await build_fragment(RenderKind.HTML, str(p), p.stat().st_size)
    assert "code-block" in built.html and built.metadata["mode"] == "code" and built.metadata["language"] == "html"


async def test_code_unknown_extension_language_null() -> None:
    built = await build_fragment(RenderKind.CODE, str(FIX / "sample.weird"), len(_read("sample.weird")))
    assert built.metadata["language"] is None
    assert built.metadata["mode"] == "code"
    assert built.html.count('<div class="cl">') == built.metadata["line_count"]


async def test_text_kind_language_null() -> None:
    built = await build_fragment(RenderKind.TEXT, str(FIX / "sample.txt"), len(_read("sample.txt")))
    assert built.metadata["language"] is None
    assert built.metadata["mode"] == "plain"
    assert "plain-pre" in built.html
    assert 'class="cl"' not in built.html


async def test_csv_full_table_no_cap_and_extra_rows() -> None:
    built = await build_fragment(RenderKind.CSV, str(FIX / "sample.csv"), len(_read("sample.csv")))
    assert built.metadata["mode"] == "table"
    assert built.metadata["row_count"] == 30
    assert built.metadata["column_count"] == 5
    assert built.html.count("<tr") == 31  # one header row plus all 30 data rows
    assert built.html.count(' class="csv-extra"') == 22  # rows 9 through 30
    assert "O&#x27;Neil, Sam" in built.html  # the quoted cell stays whole


async def test_empty_csv_raw_note() -> None:
    built = await build_fragment(RenderKind.CSV, str(FIX / "empty.csv"), 0)
    assert built.metadata["mode"] == "raw"
    assert built.metadata["note"] == "No rows found."


async def test_json_pretty_printed_minified_provenance() -> None:
    built = await build_fragment(RenderKind.JSON, str(FIX / "sample.json"), len(_read("sample.json")))
    assert built.metadata["mode"] == "code"
    assert built.metadata["provenance"] == "pretty-printed · source was minified"
    assert built.metadata["line_count"] is not None and built.metadata["line_count"] > 1
    assert built.metadata.get("language") is None  # json has no language
    assert built.html.count('<div class="cl">') == built.metadata["line_count"]


async def test_jsonl_one_doc_per_line() -> None:
    built = await build_fragment(RenderKind.JSONL, str(FIX / "sample.jsonl"), len(_read("sample.jsonl")))
    assert built.metadata["mode"] == "code"
    assert built.html.count('"id"') == 2  # both lines are there
    assert built.html.count('<div class="cl">') == built.metadata["line_count"]


async def test_yaml_and_xml_pretty_printed() -> None:
    yaml_built = await build_fragment(RenderKind.YAML, str(FIX / "sample.yaml"), len(_read("sample.yaml")))
    xml_built = await build_fragment(RenderKind.XML, str(FIX / "sample.xml"), len(_read("sample.xml")))
    assert yaml_built.metadata["mode"] == "code" and xml_built.metadata["mode"] == "code"
    assert '<div class="cl">' in yaml_built.html and '<div class="cl">' in xml_built.html
    # A yaml file with one document does not render as a list.
    first_cl = re.search(r'<div class="cl">(.*?)</div>', yaml_built.html).group(1)
    assert not first_cl.strip().startswith("- ")


async def test_yaml_multi_doc_preserves_boundaries(tmp_path: Path) -> None:
    p = tmp_path / "multi.yaml"
    p.write_text("a: 1\n---\nb: 2\n", encoding="utf-8")
    built = await build_fragment(RenderKind.YAML, str(p), p.stat().st_size)
    assert built.metadata["mode"] == "code"
    # The document separator stays.
    assert "---" in built.html


async def test_json_parse_failure_raw_note() -> None:
    built = await build_fragment(RenderKind.JSON, str(FIX / "broken.json"), len(_read("broken.json")))
    assert built.metadata["mode"] == "raw"
    assert built.metadata["note"] == "parse error, showing raw"


async def test_undecodable_text_raw_na_note(tmp_path: Path) -> None:
    p = tmp_path / "bin"
    p.write_bytes(b"\x00\x01\x02\x03")
    built = await build_fragment(RenderKind.TEXT, str(p), 4)
    assert built.html is None  # text that cannot be read has no fragment
    assert built.metadata["mode"] == "raw"
    assert built.metadata["note"] == "No preview available for this file."


async def test_oversize_raw_fallback(tmp_path: Path) -> None:
    big = tmp_path / "big.csv"
    big.write_text("a,b\n" + "1,2\n" * _RICH_MAX, encoding="utf-8")
    built = await build_fragment(RenderKind.CSV, str(big), big.stat().st_size)
    assert built.metadata["mode"] == "raw"
    assert built.metadata["note"] == "Too large to render — showing first 256 KB."
    assert len(built.html) < _RAW_MAX + 512

    bigc = tmp_path / "big.py"
    bigc.write_text("x = 1\n" * (_RICH_MAX // 2), encoding="utf-8")
    code_built = await build_fragment(RenderKind.CODE, str(bigc), bigc.stat().st_size)
    assert code_built.metadata["note"] == "Too large to highlight — showing first 256 KB."


async def test_pygments_stylesheet_cached_and_dual_theme(tmp_path: Path) -> None:
    p = tmp_path / "sample.py"
    p.write_text(_PY_SOURCE, encoding="utf-8")
    path, size = str(p), p.stat().st_size
    built_a = await build_fragment(RenderKind.CODE, path, size)
    built_b = await build_fragment(RenderKind.CODE, path, size)
    style_a = _STYLE_RE.search(built_a.html).group(1)
    style_b = _STYLE_RE.search(built_b.html).group(1)
    assert style_a == style_b  # the same style comes back both times
    assert '[data-theme="light"]' in style_a
    assert '[data-theme="dark"]' in style_a


def test_build_fragment_sync_runs_without_an_event_loop() -> None:
    built = build_fragment_sync(
        RenderKind.MARKDOWN, str(FIX / "sample.md"), len(_read("sample.md")))
    assert built.metadata["mode"] == "doc"
    assert built.html.startswith('<div class="md-doc">')


async def test_build_fragment_bytes_json() -> None:
    data = b'{"name": "Paris", "confidence": 0.99}'
    built = await frag_module.build_fragment_bytes(RenderKind.JSON, data)
    assert '"name"' in built.html
    assert built.metadata["mode"] == "code"


async def test_build_fragment_bytes_markdown() -> None:
    data = b"# Title\n\nSome **bold** text."
    built = await frag_module.build_fragment_bytes(RenderKind.MARKDOWN, data)
    assert "<h1>" in built.html
    assert built.metadata["mode"] == "doc"


async def test_build_fragment_bytes_text() -> None:
    data = b"plain text here"
    built = await frag_module.build_fragment_bytes(RenderKind.TEXT, data)
    assert "plain text here" in built.html
    assert built.metadata["mode"] == "plain"


async def test_build_fragment_bytes_over_cap_returns_note() -> None:
    data = b"x" * (_RICH_MAX + 1)
    built = await frag_module.build_fragment_bytes(RenderKind.TEXT, data)
    assert built.html is None
    assert "Too large" in built.metadata.get("note", "")
