from pathlib import Path

import pytest

from app.agents.research.models import CitedRun
from app.agents.research.references import process_references


_CONSOLE_BASE_URL = "http://console.example.com"


@pytest.fixture(autouse=True)
def _console_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """Point every console link in these tests at one known address."""
    monkeypatch.setenv("CONSOLE_BASE_URL", _CONSOLE_BASE_URL)


def _run(
    tmp_path: Path, run_id: str, tool_name: str, params: dict, result_text: str = ""
) -> CitedRun:
    run_dir = tmp_path / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "result").write_text(result_text, encoding="utf-8")
    return CitedRun(run_id=run_id, tool_name=tool_name, params=params, run_dir=str(run_dir))


class TestNumbering:
    async def test_first_appearance_ordering(self, tmp_path: Path) -> None:
        draft = "Fact A [[ref:run1]]. Fact B [[ref:run2]]. Fact A again [[ref:run1]]."
        runs: dict[str, CitedRun] = {
            "run1": _run(
                tmp_path,
                "run1",
                "brave_search",
                {"query": "q1"},
                "## Results\n1. **Title1**\n   URL: https://a.com",
            ),
            "run2": _run(tmp_path, "run2", "web_fetch", {"url": "https://b.com"}, "# Heading B"),
        }
        result = await process_references(draft, runs)
        assert "[[ref:" not in result
        assert "[[1]](https://a.com)" in result
        assert "[[2]](https://b.com)" in result
        # run1 gets [1] at first appearance, run2 gets [2], the repeat reuses [1].
        assert result.count("[[1]]") == 2
        assert result.count("[[2]]") == 1
        assert result.index("[[1]](https://a.com)") < result.index("[[2]](https://b.com)")

    async def test_no_placeholders_returns_unmodified(self) -> None:
        draft = "A report with no citations at all."
        runs: dict[str, CitedRun] = {}
        result = await process_references(draft, runs)
        assert result == "A report with no citations at all."
        assert "References" not in result


class TestTitleLinkDerivation:
    async def test_brave_search_first_hit(self, tmp_path: Path) -> None:
        draft = "Claim [[ref:run1]]."
        runs = {
            "run1": _run(
                tmp_path,
                "run1",
                "brave_search",
                {"query": "q"},
                "## Results\n1. **My Title**\n   URL: https://example.com\n   desc",
            )
        }
        result = await process_references(draft, runs)
        assert "My Title" in result
        assert "[[1]](https://example.com)" in result

    async def test_google_search_first_hit(self, tmp_path: Path) -> None:
        draft = "Claim [[ref:run1]]."
        runs = {
            "run1": _run(
                tmp_path,
                "run1",
                "google_search",
                {"query": "q"},
                "## Results\n1. **Google Title**\n   URL: https://g.example.com\n   desc",
            )
        }
        result = await process_references(draft, runs)
        assert "Google Title" in result
        assert "[[1]](https://g.example.com)" in result

    async def test_web_fetch_heading_and_url(self, tmp_path: Path) -> None:
        draft = "Claim [[ref:run1]]."
        runs = {
            "run1": _run(tmp_path, "run1", "web_fetch", {"url": "https://page.com"}, "# Article Heading\nSome text")
        }
        result = await process_references(draft, runs)
        assert "Article Heading" in result
        assert "[[1]](https://page.com)" in result

    async def test_github_view_issue(self, tmp_path: Path) -> None:
        draft = "Claim [[ref:run1]]."
        runs = {
            "run1": _run(
                tmp_path,
                "run1",
                "github",
                {"action": "view_issue", "repo": "o/r", "number": "42"},
                "# Bug: crash on startup\n...",
            )
        }
        result = await process_references(draft, runs)
        assert "Bug: crash on startup" in result
        assert "[[1]](https://github.com/o/r/issues/42)" in result

    async def test_github_view_pr(self, tmp_path: Path) -> None:
        draft = "Claim [[ref:run1]]."
        runs = {
            "run1": _run(
                tmp_path,
                "run1",
                "github",
                {"action": "view_pr", "repo": "o/r", "number": "7"},
                "# Fix: patch\n...",
            )
        }
        result = await process_references(draft, runs)
        assert "Fix: patch" in result
        assert "[[1]](https://github.com/o/r/pull/7)" in result

    async def test_github_view_issue_without_heading(self, tmp_path: Path) -> None:
        draft = "Claim [[ref:run1]]."
        runs = {
            "run1": _run(
                tmp_path,
                "run1",
                "github",
                {"action": "view_issue", "repo": "o/r", "number": "42"},
                "plain body with no heading",
            )
        }
        result = await process_references(draft, runs)
        assert "GitHub view_issue o/r#42" in result

    async def test_github_view_file(self, tmp_path: Path) -> None:
        draft = "Claim [[ref:run1]]."
        runs = {
            "run1": _run(
                tmp_path,
                "run1",
                "github",
                {"action": "view_file", "repo": "o/r", "path": "src/main.py"},
                "def main(): pass",
            )
        }
        result = await process_references(draft, runs)
        assert "o/r/src/main.py" in result
        assert "[[1]](https://github.com/o/r)" in result

    async def test_github_view_file_with_ref(self, tmp_path: Path) -> None:
        draft = "Claim [[ref:run1]]."
        runs = {
            "run1": _run(
                tmp_path,
                "run1",
                "github",
                {"action": "view_file", "repo": "o/r", "path": "src/main.py", "ref": "main"},
                "def main(): pass",
            )
        }
        result = await process_references(draft, runs)
        assert "o/r/src/main.py@main" in result
        assert "[[1]](https://github.com/o/r/blob/main/src/main.py)" in result

    async def test_github_list_issues(self, tmp_path: Path) -> None:
        draft = "Claim [[ref:run1]]."
        runs = {
            "run1": _run(
                tmp_path, "run1", "github", {"action": "list_issues", "repo": "o/r", "query": "bug"}, ""
            )
        }
        result = await process_references(draft, runs)
        assert "list_issues on o/r" in result
        assert "[[1]](https://github.com/o/r/issues)" in result

    async def test_github_search_code_uses_console_url(self, tmp_path: Path) -> None:
        draft = "Claim [[ref:run1]]."
        runs = {
            "run1": _run(
                tmp_path, "run1", "github", {"action": "search_code", "repo": "o/r", "query": "def main"}, ""
            )
        }
        result = await process_references(draft, runs)
        assert "search_code o/r def main" in result
        assert "[[1]](http://console.example.com/tools/github/run1)" in result

    async def test_github_search_repos_title(self, tmp_path: Path) -> None:
        draft = "Claim [[ref:run1]]."
        runs = {
            "run1": _run(
                tmp_path, "run1", "github", {"action": "search_repos", "query": "fastmcp tutorial"}, ""
            )
        }
        result = await process_references(draft, runs)
        assert "search_repos fastmcp tutorial" in result
        assert "[[1]](http://console.example.com/tools/github/run1)" in result

    async def test_github_list_releases_title(self, tmp_path: Path) -> None:
        draft = "Claim [[ref:run1]]."
        runs = {
            "run1": _run(tmp_path, "run1", "github", {"action": "list_releases", "repo": "o/r"}, "")
        }
        result = await process_references(draft, runs)
        assert "list_releases o/r" in result
        assert "[[1]](http://console.example.com/tools/github/run1)" in result

    async def test_x_search(self, tmp_path: Path) -> None:
        draft = "Claim [[ref:run1]]."
        runs = {
            "run1": _run(
                tmp_path, "run1", "x_search", {"query": "cuda 13"}, "tweet text\nhttps://x.com/user/status/123"
            )
        }
        result = await process_references(draft, runs)
        assert "X search: cuda 13" in result
        assert "[[1]](https://x.com/user/status/123)" in result

    async def test_x_search_empty_query_has_no_trailing_space(self, tmp_path: Path) -> None:
        draft = "Claim [[ref:run1]]."
        runs = {"run1": _run(tmp_path, "run1", "x_search", {"query": ""}, "no link here")}
        result = await process_references(draft, runs)
        assert "1. [X search](http://console.example.com/tools/x_search/run1)" in result
        assert "X search: " not in result

    async def test_rednote_research(self, tmp_path: Path) -> None:
        draft = "Claim [[ref:run1]]."
        runs = {
            "run1": _run(
                tmp_path,
                "run1",
                "rednote_research",
                {"query": "skincare"},
                "report text\nhttps://www.xiaohongshu.com/explore/abc",
            )
        }
        result = await process_references(draft, runs)
        assert "Rednote search: skincare" in result
        assert "[[1]](https://www.xiaohongshu.com/explore/abc)" in result


class TestFallback:
    async def test_no_link_uses_console_url(self, tmp_path: Path) -> None:
        draft = "Claim [[ref:run1]]."
        runs = {"run1": _run(tmp_path, "run1", "brave_search", {"query": "q"}, "no url here at all")}
        result = await process_references(draft, runs)
        assert "Search: q" in result
        assert "[[1]](http://console.example.com/tools/brave_search/run1)" in result

    async def test_empty_result_uses_neutral_label(self, tmp_path: Path) -> None:
        draft = "Claim [[ref:run1]]."
        runs = {"run1": _run(tmp_path, "run1", "web_fetch", {"url": "https://x.com"}, "")}
        result = await process_references(draft, runs)
        assert "Web fetch of https://x.com" in result
        assert "[[1]](https://x.com)" in result

    async def test_missing_result_file_degrades_to_fallback(self, tmp_path: Path) -> None:
        draft = "Claim [[ref:run1]]."
        runs = {
            "run1": CitedRun(
                run_id="run1",
                tool_name="brave_search",
                params={"query": "q"},
                run_dir=str(tmp_path / "runs" / "run1"),
            )
        }
        result = await process_references(draft, runs)
        assert "Search: q" in result
        assert "[[1]](http://console.example.com/tools/brave_search/run1)" in result

    async def test_unknown_run_id_uses_neutral_label(self) -> None:
        draft = "Claim [[ref:ghost]]."
        runs: dict[str, CitedRun] = {}
        result = await process_references(draft, runs)
        assert "[[ref:" not in result
        assert "Run ghost" in result
        assert "[[1]](http://console.example.com/tools/unknown/ghost)" in result

    async def test_empty_title_entry_is_bare_url(self, tmp_path: Path) -> None:
        draft = "Claim [[ref:run1]]."
        runs = {"run1": _run(tmp_path, "run1", "github", {}, "")}
        result = await process_references(draft, runs)
        assert "[[1]](http://console.example.com/tools/github/run1)" in result
        assert "1. http://console.example.com/tools/github/run1" in result


class TestConsoleBase:
    async def test_links_use_console_base_url(self, tmp_path: Path) -> None:
        """CONSOLE_BASE_URL sets the origin of console links."""
        runs = {"run1": _run(tmp_path, "run1", "github", {}, "")}
        result = await process_references("Claim [[ref:run1]].", runs)
        assert "[[1]](http://console.example.com/tools/github/run1)" in result

    async def test_trailing_slash_is_dropped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A trailing slash on CONSOLE_BASE_URL does not double the slash."""
        monkeypatch.setenv("CONSOLE_BASE_URL", "http://console.example.com/")
        runs = {"run1": _run(tmp_path, "run1", "github", {}, "")}
        result = await process_references("Claim [[ref:run1]].", runs)
        assert "[[1]](http://console.example.com/tools/github/run1)" in result

    @pytest.mark.parametrize("console_base_url", [None, ""])
    async def test_unset_falls_back_to_dav_base_url(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, console_base_url: str | None
    ) -> None:
        """With CONSOLE_BASE_URL unset or empty, links use DAV_BASE_URL."""
        if console_base_url is None:
            monkeypatch.delenv("CONSOLE_BASE_URL")
        else:
            monkeypatch.setenv("CONSOLE_BASE_URL", console_base_url)
        monkeypatch.setenv("DAV_BASE_URL", "http://dav.example.com:8315")
        runs = {"run1": _run(tmp_path, "run1", "github", {}, "")}
        result = await process_references("Claim [[ref:run1]].", runs)
        assert "[[1]](http://dav.example.com:8315/tools/github/run1)" in result

    async def test_both_unset_use_localhost(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With neither variable set, links use the local default address."""
        monkeypatch.delenv("CONSOLE_BASE_URL")
        monkeypatch.delenv("DAV_BASE_URL", raising=False)
        result = await process_references("Claim [[ref:ghost]].", {})
        assert "[[1]](http://localhost:8315/tools/unknown/ghost)" in result


class TestReferencesSection:
    async def test_references_section_appended(self, tmp_path: Path) -> None:
        draft = "A [[ref:r1]]. B [[ref:r2]]."
        runs = {
            "r1": _run(
                tmp_path, "r1", "brave_search", {"query": "a"}, "## Results\n1. **T1**\n   URL: https://a.com"
            ),
            "r2": _run(
                tmp_path, "r2", "brave_search", {"query": "b"}, "## Results\n1. **T2**\n   URL: https://b.com"
            ),
        }
        result = await process_references(draft, runs)
        assert result.index("[[1]](https://a.com)") < result.index("## References")
        assert result.endswith(
            "## References\n"
            "\n"
            "1. [T1](https://a.com)\n"
            "2. [T2](https://b.com)"
        )
