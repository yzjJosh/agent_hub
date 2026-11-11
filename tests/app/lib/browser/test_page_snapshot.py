"""Tests for a captured page."""

import json
import re
from bs4 import BeautifulSoup
import pytest

from app.lib.browser.page_snapshot import PageSnapshot
from app.lib.browser.browser_manager import Viewport
from unittest.mock import AsyncMock, MagicMock

_CLICK_AT_X = 150.5
_CLICK_AT_Y = 250.75
_AIM_CONFIG = {"click_input_x_range": (0.4, 0.75)}
_WINDOW_WIDTH = 1280.0
_WINDOW_HEIGHT = 800.0
_WINDOW = Viewport(
    client_width=_WINDOW_WIDTH,
    client_height=_WINDOW_HEIGHT,
    page_x=120.0,
    page_y=40.0,
    content_width=_WINDOW_WIDTH,
    content_height=_WINDOW_HEIGHT,
    window_width=_WINDOW_WIDTH,
    window_height=_WINDOW_HEIGHT,
)
_SHARE_LEFT = 0.25
_SHARE_CENTER = 0.5
_SHARE_THREE_QUARTERS = 0.75
_REL_CLICK_X = 0.35
_REL_CLICK_Y = 0.62
_DUPLICATE_ID_HTML = (
    "<html><body>"
    "<nav><input id='search-box-input' type='search' name='searchInputBox'"
    " aria-label='City, neighborhood, ZIP' placeholder='City, neighborhood, ZIP'/></nav>"
    "<main><input id='search-box-input' type='search' name='searchInputBox'"
    " aria-label='Search for properties' placeholder='City, Address, School, Agent, ZIP'/></main>"
    "</body></html>"
)


class TestPageSnapshotConstruction:
    """Tests for making a snapshot from page text."""

    async def test_construction_from_valid_html(self, mock_browser_mgr: AsyncMock) -> None:
        """A snapshot keeps the page text and starts with no address or title."""
        html = "<html><body><h1>Title</h1><button>Click</button></body></html>"
        snapshot = await PageSnapshot.create(html, mock_browser_mgr, "tab1")
        assert snapshot.html == html
        assert snapshot.url == ""
        assert snapshot.title == ""

    async def test_construction_from_empty_html(self, mock_browser_mgr: AsyncMock) -> None:
        """An empty page makes a snapshot with empty text."""
        snapshot = await PageSnapshot.create("", mock_browser_mgr, "tab1")
        assert snapshot.html == ""

    async def test_construction_from_malformed_html(self, mock_browser_mgr: AsyncMock) -> None:
        """A broken page still makes a snapshot."""
        html = "<div><span>unclosed"
        snapshot = await PageSnapshot.create(html, mock_browser_mgr, "tab1")
        assert snapshot.html == html

    async def test_construction_preserves_html(self, mock_browser_mgr: AsyncMock) -> None:
        """A snapshot gives back the page text it was made from."""
        html = "<html><body><h1>Title</h1></body></html>"
        snapshot = await PageSnapshot.create(html, mock_browser_mgr, "tab1")
        assert snapshot.html == html

    async def test_construction_with_none_html(self, mock_browser_mgr: AsyncMock) -> None:
        """Missing page text becomes empty text."""
        snapshot = await PageSnapshot.create(None, mock_browser_mgr, "tab1")
        assert snapshot.html == ""


class TestFilteredTree:
    """Tests for the trimmed page tree."""

    async def test_filtered_tree_has_role_name_children_structure(self, page_snapshot_simple: PageSnapshot) -> None:
        """The tree says what the top part is."""
        result = await page_snapshot_simple.filtered_tree()
        assert "role" in result

    async def test_filtered_tree_no_ref_ids(self, page_snapshot_button: PageSnapshot) -> None:
        """The tree holds no element ids."""
        result = await page_snapshot_button.filtered_tree()
        result_str = json.dumps(result)
        assert '"ref_id"' not in result_str
        assert not re.search(r'"e\d+"', result_str), "filtered_tree must not contain ref IDs"

    async def test_filtered_tree_no_raw_html(self, page_snapshot_simple: PageSnapshot) -> None:
        """The tree holds no page tags."""
        result = await page_snapshot_simple.filtered_tree()
        result_str = json.dumps(result, ensure_ascii=False)
        assert "<h1>" not in result_str, "filtered_tree must not contain raw HTML tags"
        assert "<html>" not in result_str, "filtered_tree must not contain raw HTML tags"

    async def test_filtered_tree_no_markdown(self, page_snapshot_simple: PageSnapshot) -> None:
        """The tree holds no markdown marks."""
        result = await page_snapshot_simple.filtered_tree()
        result_str = json.dumps(result, ensure_ascii=False)
        assert "#" not in result_str
        assert "`" not in result_str, "filtered_tree must not contain markdown code syntax"

    async def test_filtered_tree_fallback_on_filter_failure(self, page_snapshot_simple: PageSnapshot) -> None:
        """A simple page gives back a tree with a top part."""
        result = await page_snapshot_simple.filtered_tree()
        assert "role" in result, "filtered_tree fallback must contain role"

    async def test_filtered_tree_removes_noise(self, page_snapshot_filter_html: PageSnapshot) -> None:
        """The main content of a page stays in the tree."""
        result = await page_snapshot_filter_html.filtered_tree()
        result_str = json.dumps(result, ensure_ascii=False)
        assert "Important Content" in result_str

    async def test_filtered_tree_empty_html(self, page_snapshot_empty: PageSnapshot) -> None:
        """An empty page gives back no tree."""
        result = await page_snapshot_empty.filtered_tree()
        assert result is None

    async def test_filtered_tree_caches_result(self, page_snapshot_simple: PageSnapshot) -> None:
        """Asking for the tree twice gives back the same tree."""
        result1 = await page_snapshot_simple.filtered_tree()
        result2 = await page_snapshot_simple.filtered_tree()
        assert result1 is result2, "filtered_tree should cache and return the same object"


class TestActionableElements:
    """Tests for the list of things you can act on."""

    async def test_actionable_elements_returns_list(self, page_snapshot_button: PageSnapshot) -> None:
        """The things you can act on come back as a list."""
        result = page_snapshot_button.actionable_elements()
        assert isinstance(result, list)

    async def test_actionable_elements_has_ref_id_role_name(self, mock_browser_mgr: AsyncMock) -> None:
        """Each thing you can act on has an id, a kind, and a name."""
        html = "<html><body><button>Click Me</button></body></html>"
        snapshot = await PageSnapshot.create(html, mock_browser_mgr, "tab1")
        result = snapshot.actionable_elements()
        if result:
            elem = result[0]
            assert "ref_id" in elem
            assert "role" in elem
            assert "name" in elem

    async def test_actionable_elements_only_interactive(self, page_snapshot_interactive: PageSnapshot) -> None:
        """Headings and plain text are left out of the list."""
        result = page_snapshot_interactive.actionable_elements()
        roles = {elem["role"] for elem in result}
        assert "heading" not in roles
        assert "paragraph" not in roles
        interactive_roles = {"button", "link", "textbox"}
        assert roles & interactive_roles, f"Expected interactive roles, got {roles}"

    async def test_actionable_elements_is_flat_list(self, page_snapshot_button: PageSnapshot) -> None:
        """The list is flat with nothing nested inside."""
        result = page_snapshot_button.actionable_elements()
        assert all("children" not in elem for elem in result), "actionable_elements must be flat, not nested"

    async def test_actionable_elements_ref_ids_unique(self, mock_browser_mgr: AsyncMock) -> None:
        """Every listed thing has its own id."""
        html = """<html><body>
            <button>Button 1</button>
            <button>Button 2</button>
            <a href='/'>Link</a>
        </body></html>"""
        snapshot = await PageSnapshot.create(html, mock_browser_mgr, "tab1")
        result = snapshot.actionable_elements()
        ref_ids = [elem["ref_id"] for elem in result]
        assert len(ref_ids) == len(set(ref_ids)), "ref_ids must be unique within a snapshot"

    async def test_actionable_elements_empty_page(self, page_snapshot_simple: PageSnapshot) -> None:
        """A page with nothing to act on gives back an empty list."""
        result = page_snapshot_simple.actionable_elements()
        assert isinstance(result, list)
        assert len(result) == 0

    async def test_actionable_elements_includes_various_types(self, mock_browser_mgr: AsyncMock) -> None:
        """Buttons, links, text boxes, and drop downs all get listed."""
        html = """<html><body>
            <button>Submit</button>
            <a href='/page'>Go</a>
            <input type='text' aria-label='Name' />
            <textarea aria-label='Comment'></textarea>
            <select aria-label='Choice'><option>A</option></select>
        </body></html>"""
        snapshot = await PageSnapshot.create(html, mock_browser_mgr, "tab1")
        result = snapshot.actionable_elements()
        assert len(result) >= 4, f"Expected at least 4 interactive elements, got {len(result)}"


class TestRefIdIsolation:
    """Tests that ids belong to just one snapshot."""

    async def test_ref_ids_scoped_to_snapshot(self, mock_browser_mgr: AsyncMock) -> None:
        """An id from one snapshot does not work on another snapshot."""
        html1 = "<html><body><button>Button A</button></body></html>"
        html2 = "<html><body><button>Button B</button></body></html>"

        snapshot1 = await PageSnapshot.create(html1, mock_browser_mgr, "tab1")
        snapshot2 = await PageSnapshot.create(html2, mock_browser_mgr, "tab2")

        elements1 = snapshot1.actionable_elements()
        if elements1:
            ref_id_1 = elements1[0]["ref_id"]
            with pytest.raises(ValueError):
                await snapshot2.click(ref_id_1)

    async def test_independent_snapshots_no_shared_state(self, mock_browser_mgr: AsyncMock) -> None:
        """Two snapshots each list the things on their own page."""
        html1 = "<html><body><button>Button A</button></body></html>"
        html2 = "<html><body><a href='/'>Link B</a></body></html>"

        snapshot1 = await PageSnapshot.create(html1, mock_browser_mgr, "tab1")
        snapshot2 = await PageSnapshot.create(html2, mock_browser_mgr, "tab2")

        elements1 = snapshot1.actionable_elements()
        elements2 = snapshot2.actionable_elements()

        if elements1 and elements2:
            assert elements1[0]["name"] == "Button A"
            assert elements2[0]["name"] == "Link B"


class TestPageSnapshotUrlTitle:
    """Tests for the saved address and title."""

    async def test_url_returns_cached_value(self, mock_browser_mgr: AsyncMock) -> None:
        """A snapshot gives back the address it was made with."""
        snapshot = await PageSnapshot.create("<html><body><h1>Title</h1></body></html>", mock_browser_mgr, "tab1", url="https://example.com/page")
        assert snapshot.url == "https://example.com/page"

    async def test_title_returns_cached_value(self, mock_browser_mgr: AsyncMock) -> None:
        """A snapshot gives back the title it was made with."""
        snapshot = await PageSnapshot.create("<html><body><h1>Title</h1></body></html>", mock_browser_mgr, "tab1", title="Example Domain")
        assert snapshot.title == "Example Domain"

    async def test_url_empty_when_not_provided(self, page_snapshot_simple: PageSnapshot) -> None:
        """A snapshot made with no address gives back empty text."""
        assert page_snapshot_simple.url == ""

    async def test_title_empty_when_not_provided(self, page_snapshot_simple: PageSnapshot) -> None:
        """A snapshot made with no title gives back empty text."""
        assert page_snapshot_simple.title == ""

    async def test_url_is_property_not_method(self, mock_browser_mgr: AsyncMock) -> None:
        """The address is a plain value, not something you call."""
        snapshot = await PageSnapshot.create("<html><body><h1>Title</h1></body></html>", mock_browser_mgr, "tab1", url="https://example.com")
        assert snapshot.url == "https://example.com"
        assert not callable(snapshot.url)

    async def test_title_is_property_not_method(self, mock_browser_mgr: AsyncMock) -> None:
        """The title is a plain value, not something you call."""
        snapshot = await PageSnapshot.create("<html><body><h1>Title</h1></body></html>", mock_browser_mgr, "tab1", title="Test Page")
        assert snapshot.title == "Test Page"
        assert not callable(snapshot.title)

    async def test_url_and_title_cached_together(self, mock_browser_mgr: AsyncMock) -> None:
        """A snapshot keeps both the address and the title."""
        snapshot = await PageSnapshot.create(
            "<html><body><h1>Title</h1></body></html>", mock_browser_mgr, "tab1",
            url="https://example.com/path?query=1",
            title="Example Domain"
        )
        assert snapshot.url == "https://example.com/path?query=1"
        assert snapshot.title == "Example Domain"


class TestPageSnapshotClick:
    """Tests for clicking a listed thing."""

    async def test_click_delegates_to_browser_manager(self, mock_browser_mgr: AsyncMock) -> None:
        """Clicking a listed thing clicks it in the browser and gives back true."""
        mock_browser_mgr.click.return_value = True
        snap = await PageSnapshot.create("<html><body><button aria-label='Submit'>Submit</button></body></html>", mock_browser_mgr, "tab1")
        elements = snap.actionable_elements()
        if elements:
            assert await snap.click(elements[0]["ref_id"]) is True
            mock_browser_mgr.click.assert_called()

    async def test_click_invalid_ref_id_raises(self, page_snapshot_button: PageSnapshot) -> None:
        """Clicking an id that is not there raises an error."""
        with pytest.raises(ValueError):
            await page_snapshot_button.click("e999")

    async def test_click_returns_false_on_failure(self, mock_browser_mgr: AsyncMock) -> None:
        """A click that fails in the browser gives back false."""
        mock_browser_mgr.click.return_value = False
        snap = await PageSnapshot.create("<html><body><button>Click</button></body></html>", mock_browser_mgr, "tab1")
        elements = snap.actionable_elements()
        if elements:
            assert await snap.click(elements[0]["ref_id"]) is False

    async def test_click_passes_target_id(self, mock_browser_mgr: AsyncMock) -> None:
        """A click goes to the tab the snapshot came from."""
        mock_browser_mgr.click.return_value = True
        snap = await PageSnapshot.create("<html><body><button>Click</button></body></html>", mock_browser_mgr, "tab1")
        elements = snap.actionable_elements()
        if elements:
            await snap.click(elements[0]["ref_id"])
            assert mock_browser_mgr.click.call_args[0][0] == "tab1"

    async def test_click_forwards_human_config(self, mock_browser_mgr: AsyncMock) -> None:
        """Click settings are passed on to the browser."""
        mock_browser_mgr.click.return_value = True
        snap = await PageSnapshot.create("<html><body><button>Click</button></body></html>", mock_browser_mgr, "tab1")
        elements = snap.actionable_elements()
        if elements:
            await snap.click(elements[0]["ref_id"], human_config=_AIM_CONFIG)
            assert mock_browser_mgr.click.call_args.kwargs["human_config"] is _AIM_CONFIG

    async def test_click_without_human_config_omits_kwarg(self, mock_browser_mgr: AsyncMock) -> None:
        """A click with no settings passes none on."""
        mock_browser_mgr.click.return_value = True
        snap = await PageSnapshot.create("<html><body><button>Click</button></body></html>", mock_browser_mgr, "tab1")
        elements = snap.actionable_elements()
        if elements:
            await snap.click(elements[0]["ref_id"])
            assert "human_config" not in mock_browser_mgr.click.call_args.kwargs


class TestPageSnapshotFill:
    """Tests for typing into a listed thing."""

    async def test_fill_delegates_to_browser_manager(self, mock_browser_mgr: AsyncMock) -> None:
        """Typing into a listed thing types it in the browser and gives back true."""
        mock_browser_mgr.type_text.return_value = True
        snap = await PageSnapshot.create("<html><body><input type='text' aria-label='Search' /></body></html>", mock_browser_mgr, "tab1")
        elements = snap.actionable_elements()
        if elements:
            assert await snap.fill(elements[0]["ref_id"], "hello") is True
            mock_browser_mgr.type_text.assert_called()

    async def test_fill_with_submit(self, mock_browser_mgr: AsyncMock) -> None:
        """Typing with submit turned on gives back true."""
        mock_browser_mgr.type_text.return_value = True
        snap = await PageSnapshot.create("<html><body><input type='text' aria-label='Search' /></body></html>", mock_browser_mgr, "tab1")
        elements = snap.actionable_elements()
        if elements:
            assert await snap.fill(elements[0]["ref_id"], "query", submit=True) is True

    async def test_fill_invalid_ref_id_raises(self, mock_browser_mgr: AsyncMock) -> None:
        """Typing into an id that is not there raises an error."""
        snap = await PageSnapshot.create("<html><body><input type='text' /></body></html>", mock_browser_mgr, "tab1")
        with pytest.raises(ValueError):
            await snap.fill("e999", "text")

    async def test_fill_returns_false_on_failure(self, mock_browser_mgr: AsyncMock) -> None:
        """Typing that fails in the browser gives back false."""
        mock_browser_mgr.type_text.return_value = False
        snap = await PageSnapshot.create("<html><body><input type='text' aria-label='Name' /></body></html>", mock_browser_mgr, "tab1")
        elements = snap.actionable_elements()
        if elements:
            assert await snap.fill(elements[0]["ref_id"], "text") is False

    async def test_fill_empty_text(self, mock_browser_mgr: AsyncMock) -> None:
        """Typing empty text works."""
        mock_browser_mgr.type_text.return_value = True
        snap = await PageSnapshot.create("<html><body><input type='text' aria-label='Name' /></body></html>", mock_browser_mgr, "tab1")
        elements = snap.actionable_elements()
        if elements:
            assert await snap.fill(elements[0]["ref_id"], "") is True

    async def test_fill_forwards_human_config(self, mock_browser_mgr: AsyncMock) -> None:
        """Typing settings are passed on to the browser."""
        mock_browser_mgr.type_text.return_value = True
        snap = await PageSnapshot.create("<html><body><input type='text' aria-label='Name' /></body></html>", mock_browser_mgr, "tab1")
        elements = snap.actionable_elements()
        if elements:
            await snap.fill(elements[0]["ref_id"], "text", human_config=_AIM_CONFIG)
            assert mock_browser_mgr.type_text.call_args.kwargs["human_config"] is _AIM_CONFIG

    async def test_fill_without_human_config_omits_kwarg(self, mock_browser_mgr: AsyncMock) -> None:
        """Typing with no settings passes none on."""
        mock_browser_mgr.type_text.return_value = True
        snap = await PageSnapshot.create("<html><body><input type='text' aria-label='Name' /></body></html>", mock_browser_mgr, "tab1")
        elements = snap.actionable_elements()
        if elements:
            await snap.fill(elements[0]["ref_id"], "text")
            assert "human_config" not in mock_browser_mgr.type_text.call_args.kwargs


class TestPageSnapshotUpload:
    """Tests for putting files into a listed file box."""

    async def test_upload_delegates_to_browser_manager(self, mock_browser_mgr: AsyncMock) -> None:
        """Picking a file for a listed box sends it to the browser and gives back true."""
        mock_browser_mgr.upload_files.return_value = True
        snap = await PageSnapshot.create("<html><body><input type='file' aria-label='Upload' /></body></html>", mock_browser_mgr, "tab1")
        elements = snap.actionable_elements()
        if elements:
            assert await snap.upload(elements[0]["ref_id"], ["/tmp/img.jpg"]) is True
            mock_browser_mgr.upload_files.assert_called()

    async def test_upload_empty_paths(self, mock_browser_mgr: AsyncMock) -> None:
        """Picking no files still works."""
        mock_browser_mgr.upload_files.return_value = True
        snap = await PageSnapshot.create("<html><body><input type='file' aria-label='Upload' /></body></html>", mock_browser_mgr, "tab1")
        elements = snap.actionable_elements()
        if elements:
            assert await snap.upload(elements[0]["ref_id"], []) is True

    async def test_upload_invalid_ref_id_raises(self, mock_browser_mgr: AsyncMock) -> None:
        """Picking a file for an id that is not there raises an error."""
        snap = await PageSnapshot.create("<html><body><input type='file' /></body></html>", mock_browser_mgr, "tab1")
        with pytest.raises(ValueError):
            await snap.upload("e999", ["/tmp/img.jpg"])

    async def test_upload_returns_false_on_failure(self, mock_browser_mgr: AsyncMock) -> None:
        """A file that fails to go in gives back false."""
        mock_browser_mgr.upload_files.return_value = False
        snap = await PageSnapshot.create("<html><body><input type='file' aria-label='Upload' /></body></html>", mock_browser_mgr, "tab1")
        elements = snap.actionable_elements()
        if elements:
            assert await snap.upload(elements[0]["ref_id"], ["/tmp/img.jpg"]) is False


class TestPageSnapshotUploadChooser:
    """Tests for sending files through a file picker window."""

    async def test_upload_with_chooser_relative_passes_captured_viewport(self, mock_browser_mgr: AsyncMock) -> None:
        """A screen point uses the window captured with the page."""
        snap = await PageSnapshot.create("<html><body></body></html>", mock_browser_mgr, "tab1", viewport=_WINDOW)
        mock_browser_mgr.upload_with_chooser_relative.return_value = True

        assert await snap.upload_with_chooser_relative(_REL_CLICK_X, _REL_CLICK_Y, ["/tmp/img.jpg"]) is True
        mock_browser_mgr.upload_with_chooser_relative.assert_awaited_once_with(
            "tab1", _REL_CLICK_X, _REL_CLICK_Y, ["/tmp/img.jpg"], viewport=_WINDOW, verbose=False,
        )


class TestPageSnapshotFilteredTreeNoMocking:
    """Tests for the trimmed tree built from real page text."""

    async def test_filtered_tree_returns_dict_with_valid_html(self, page_snapshot_simple: PageSnapshot) -> None:
        """A good page gives back a tree."""
        result = await page_snapshot_simple.filtered_tree()
        assert isinstance(result, dict)

    async def test_filtered_tree_exercises_filter_pipeline(self, mock_browser_mgr: AsyncMock) -> None:
        """A page with a menu and a footer still keeps its main content."""
        html = "<html><body><nav>Sidebar nav links and more</nav><main><h1>Main content</h1><p>Important paragraph text that should be preserved</p></main><footer>Footer information</footer></body></html>"
        snapshot = await PageSnapshot.create(html, mock_browser_mgr, "tab1")
        result = await snapshot.filtered_tree()
        assert isinstance(result, dict)
        result_str = json.dumps(result, ensure_ascii=False)
        assert "Main content" in result_str

    async def test_filtered_tree_fallback_no_crash_on_edge_case(self, mock_browser_mgr: AsyncMock) -> None:
        """An odd page still gives back a tree."""
        html = "<html><body><div data-widget='chart' id='widget-1'><span>Label</span></div></body></html>"
        snapshot = await PageSnapshot.create(html, mock_browser_mgr, "tab1")
        result = await snapshot.filtered_tree()
        assert isinstance(result, dict)

    async def test_verbose_does_not_change_return_value(self, mock_browser_mgr: AsyncMock) -> None:
        """Turning on logging does not change the tree."""
        html = "<html><body><h1>Title</h1><p>Content paragraph</p></body></html>"
        snapshot = await PageSnapshot.create(html, mock_browser_mgr, "tab1")
        result_default = await snapshot.filtered_tree()
        snapshot2 = await PageSnapshot.create(html, mock_browser_mgr, "tab1")
        result_verbose = await snapshot2.filtered_tree(verbose=True)
        assert result_default == result_verbose

    async def test_verbose_cached_path_returns_same_object(self, mock_browser_mgr: AsyncMock) -> None:
        """Asking again with logging on gives back the same tree."""
        html = "<html><body><h1>Title</h1><p>Content paragraph</p></body></html>"
        snapshot = await PageSnapshot.create(html, mock_browser_mgr, "tab1")
        result = await snapshot.filtered_tree()
        result_verbose = await snapshot.filtered_tree(verbose=True)
        assert result is result_verbose


class TestBrowserManagerSnapshotReturnsPageSnapshot:
    """Tests that a snapshot is its own kind of thing."""

    async def test_snapshot_returns_page_snapshot_instance(self, mock_browser_mgr: AsyncMock) -> None:
        """A snapshot is a page snapshot and not a plain dict."""
        html = "<html><body>test</body></html>"
        snapshot = await PageSnapshot.create(html, mock_browser_mgr, "tab1")
        assert isinstance(snapshot, PageSnapshot)
        assert not isinstance(snapshot, dict)


class TestPageSnapshotMultipleTabs:
    """Tests that a snapshot belongs to one tab."""

    async def test_snapshot_scoped_to_single_tab(self, mock_browser_mgr: AsyncMock) -> None:
        """Acting on a snapshot goes to the tab it came from."""
        html = "<html><body><button>Click</button></body></html>"
        snapshot1 = await PageSnapshot.create(html, mock_browser_mgr, "tab1")
        elements1 = snapshot1.actionable_elements()
        if elements1:
            await snapshot1.click(elements1[0]["ref_id"])
            assert mock_browser_mgr.click.call_args[0][0] == "tab1"


class TestPageSnapshotSelectOption:
    """Tests for picking an option in a drop down."""

    @staticmethod
    def _assert_select_option_call(mock_mgr: MagicMock, expected_kwargs: dict) -> None:
        mock_mgr.select_option.assert_called_once()
        call_args = mock_mgr.select_option.call_args[0]
        call_kwargs = mock_mgr.select_option.call_args[1]
        assert call_args[0] == "tab1"
        assert isinstance(call_args[1], str) and len(call_args[1]) > 0
        assert call_kwargs == expected_kwargs

    async def test_page_snapshot_select_option_by_value(self, mock_browser_mgr: AsyncMock) -> None:
        """Picking by value sends the value to the browser."""
        mock_browser_mgr.select_option.return_value = True
        snap = await PageSnapshot.create("<html><body><select id='test'><option>red</option></select></body></html>", mock_browser_mgr, "tab1")
        elements = snap.actionable_elements()
        ref_id = elements[0]["ref_id"]
        result = await snap.select_option(ref_id, value="red")
        assert result is True
        self._assert_select_option_call(mock_browser_mgr, {"value": "red", "index": None, "label": None, "verbose": False})

    async def test_page_snapshot_select_option_by_index(self, mock_browser_mgr: AsyncMock) -> None:
        """Picking by position sends the position to the browser."""
        mock_browser_mgr.select_option.return_value = True
        snap = await PageSnapshot.create("<html><body><select id='test'><option>red</option></select></body></html>", mock_browser_mgr, "tab1")
        elements = snap.actionable_elements()
        ref_id = elements[0]["ref_id"]
        result = await snap.select_option(ref_id, index=2)
        assert result is True
        self._assert_select_option_call(mock_browser_mgr, {"value": None, "index": 2, "label": None, "verbose": False})

    async def test_page_snapshot_select_option_by_label(self, mock_browser_mgr: AsyncMock) -> None:
        """Picking by label sends the label to the browser."""
        mock_browser_mgr.select_option.return_value = True
        snap = await PageSnapshot.create("<html><body><select id='test'><option>red</option></select></body></html>", mock_browser_mgr, "tab1")
        elements = snap.actionable_elements()
        ref_id = elements[0]["ref_id"]
        result = await snap.select_option(ref_id, label="Option 1")
        assert result is True
        self._assert_select_option_call(mock_browser_mgr, {"value": None, "index": None, "label": "Option 1", "verbose": False})

    async def test_page_snapshot_select_option_invalid_ref(self, mock_browser_mgr: AsyncMock) -> None:
        """Picking on an id that is not there raises an error."""
        snap = await PageSnapshot.create("<html><body></body></html>", mock_browser_mgr, "tab1")
        with pytest.raises(ValueError):
            await snap.select_option("invalid_ref", value="x")

    async def test_page_snapshot_select_option_verbose(self, mock_browser_mgr: AsyncMock) -> None:
        """Picking with logging on passes that on."""
        mock_browser_mgr.select_option.return_value = True
        snap = await PageSnapshot.create("<html><body><select id='test'><option>red</option></select></body></html>", mock_browser_mgr, "tab1")
        elements = snap.actionable_elements()
        ref_id = elements[0]["ref_id"]
        result = await snap.select_option(ref_id, value="red", verbose=True)
        assert result is True
        call_kwargs = mock_browser_mgr.select_option.call_args[1]
        assert call_kwargs["verbose"] is True


class TestPageSnapshotPressKey:
    """Tests for pressing a key on the page."""

    async def test_page_snapshot_press_key_default(self, mock_browser_mgr: AsyncMock) -> None:
        """Pressing a key with none named presses Enter."""
        mock_browser_mgr.press_key.return_value = True
        snap = await PageSnapshot.create("<html><body>test</body></html>", mock_browser_mgr, "tab1")
        result = await snap.press_key()
        assert result is True
        mock_browser_mgr.press_key.assert_called_once_with("tab1", "Enter", verbose=False)

    async def test_page_snapshot_press_key_custom(self, mock_browser_mgr: AsyncMock) -> None:
        """A named key is the one that gets pressed."""
        mock_browser_mgr.press_key.return_value = True
        snap = await PageSnapshot.create("<html><body>test</body></html>", mock_browser_mgr, "tab1")
        result = await snap.press_key(key="Escape")
        assert result is True
        mock_browser_mgr.press_key.assert_called_once_with("tab1", "Escape", verbose=False)

    async def test_page_snapshot_press_key_verbose(self, mock_browser_mgr: AsyncMock) -> None:
        """Pressing a key with logging on passes that on."""
        mock_browser_mgr.press_key.return_value = True
        snap = await PageSnapshot.create("<html><body>test</body></html>", mock_browser_mgr, "tab1")
        result = await snap.press_key(verbose=True)
        assert result is True
        call_kwargs = mock_browser_mgr.press_key.call_args[1]
        assert call_kwargs["verbose"] is True


# ---------------------------------------------------------------------------
# Picking options and pressing keys together
# ---------------------------------------------------------------------------


class TestIntegrationSelectOption:
    """Tests for picking options one after another."""

    async def test_select_option_full_workflow(self, mock_browser_mgr: AsyncMock) -> None:
        """Two drop downs on one page can each be set on their own."""
        mock_browser_mgr.select_option.return_value = True
        snap = await PageSnapshot.create("<html><body><select name='dom'><option>7-</option></select><select id='beds'><option>3+</option></select></body></html>", mock_browser_mgr, "tab1")
        elements = snap.actionable_elements()
        selects = [e for e in elements if e["role"] == "combobox"]
        result = await snap.select_option(selects[0]["ref_id"], value="7-")
        assert result is True
        call_args = mock_browser_mgr.select_option.call_args
        assert call_args[0][0] == "tab1"
        assert call_args[1]["value"] == "7-"
        assert call_args[1]["index"] is None
        assert call_args[1]["label"] is None
        result = await snap.select_option(selects[1]["ref_id"], label="3+")
        assert result is True
        call_args = mock_browser_mgr.select_option.call_args
        assert call_args[0][0] == "tab1"
        assert call_args[1]["value"] is None
        assert call_args[1]["label"] == "3+"

    async def test_press_key_after_fill_workflow(self, mock_browser_mgr: AsyncMock) -> None:
        """Pressing Enter goes to the browser."""
        mock_browser_mgr.press_key.return_value = True
        snap = await PageSnapshot.create("<html><body>test</body></html>", mock_browser_mgr, "tab1")
        result = await snap.press_key("Enter")
        assert result is True
        mock_browser_mgr.press_key.assert_called_once_with("tab1", "Enter", verbose=False)


# ---------------------------------------------------------------------------
# Clicking a spot on the page
# ---------------------------------------------------------------------------


class TestPageSnapshotClickAt:
    """Tests for clicking a spot on the page."""

    async def test_page_snapshot_click_at_delegates(self, mock_browser_mgr: AsyncMock) -> None:
        """Clicking a spot sends the spot to the browser."""
        mock_browser_mgr.click_at.return_value = True
        snap = await PageSnapshot.create("<html><body>test</body></html>", mock_browser_mgr, "tab1")
        result = await snap.click_at(300, 500)
        assert result is True
        mock_browser_mgr.click_at.assert_called_once_with("tab1", 300, 500, verbose=False)

    async def test_page_snapshot_click_at_verbose(self, mock_browser_mgr: AsyncMock) -> None:
        """Clicking a spot with logging on passes that on."""
        mock_browser_mgr.click_at.return_value = True
        snap = await PageSnapshot.create("<html><body>test</body></html>", mock_browser_mgr, "tab1")
        result = await snap.click_at(300, 500, verbose=True)
        assert result is True
        call_kwargs = mock_browser_mgr.click_at.call_args[1]
        assert call_kwargs["verbose"] is True

    async def test_page_snapshot_click_at_float_coords(self, mock_browser_mgr: AsyncMock) -> None:
        """A spot given with decimals is passed on as it is."""
        mock_browser_mgr.click_at.return_value = True
        snap = await PageSnapshot.create("<html><body>test</body></html>", mock_browser_mgr, "tab1")
        result = await snap.click_at(_CLICK_AT_X, _CLICK_AT_Y)
        assert result is True
        mock_browser_mgr.click_at.assert_called_once_with("tab1", _CLICK_AT_X, _CLICK_AT_Y, verbose=False)


class TestPageSnapshotSolveCaptcha:
    """Tests for solving a captcha on the page."""

    async def test_page_snapshot_solve_captcha_delegates(self, mock_browser_mgr: AsyncMock) -> None:
        """Solving a captcha passes the step limit and the logging flag to the browser."""
        mock_browser_mgr.solve_captcha.return_value = True
        snap = await PageSnapshot.create("<html><body>test</body></html>", mock_browser_mgr, "tab1")
        result = await snap.solve_captcha(max_steps=3, verbose=True)
        assert result is True
        mock_browser_mgr.solve_captcha.assert_called_once_with("tab1", max_steps=3, verbose=True)


class TestPageSnapshotScrollTo:
    """Tests for scrolling to a spot."""

    async def test_page_snapshot_scroll_to_delegates(self, mock_browser_mgr: AsyncMock) -> None:
        """Scrolling to a spot sends the spot to the browser."""
        mock_browser_mgr.scroll_to.return_value = True
        snap = await PageSnapshot.create("<html><body>test</body></html>", mock_browser_mgr, "tab1")
        result = await snap.scroll_to(100, 500)
        assert result is True
        mock_browser_mgr.scroll_to.assert_called_once_with("tab1", 100, 500, verbose=False)

    async def test_page_snapshot_scroll_to_verbose(self, mock_browser_mgr: AsyncMock) -> None:
        """Scrolling with logging on passes that on."""
        mock_browser_mgr.scroll_to.return_value = True
        snap = await PageSnapshot.create("<html><body>test</body></html>", mock_browser_mgr, "tab1")
        result = await snap.scroll_to(100, 500, verbose=True)
        assert result is True
        call_kwargs = mock_browser_mgr.scroll_to.call_args[1]
        assert call_kwargs["verbose"] is True

    async def test_page_snapshot_scroll_to_float_coords(self, mock_browser_mgr: AsyncMock) -> None:
        """A spot given with decimals is passed on as it is."""
        mock_browser_mgr.scroll_to.return_value = True
        snap = await PageSnapshot.create("<html><body>test</body></html>", mock_browser_mgr, "tab1")
        result = await snap.scroll_to(_CLICK_AT_X, _CLICK_AT_Y)
        assert result is True
        mock_browser_mgr.scroll_to.assert_called_once_with("tab1", _CLICK_AT_X, _CLICK_AT_Y, verbose=False)

    async def test_page_snapshot_scroll_to_returns_false(self, mock_browser_mgr: AsyncMock) -> None:
        """A scroll that fails gives back false."""
        mock_browser_mgr.scroll_to.return_value = False
        snap = await PageSnapshot.create("<html><body>test</body></html>", mock_browser_mgr, "tab1")
        result = await snap.scroll_to(100, 500)
        assert result is False


class TestPageSnapshotScroll:
    """Tests for scrolling a listed thing into view."""

    async def test_page_snapshot_scroll_delegates(self, mock_browser_mgr: AsyncMock) -> None:
        """Scrolling to a listed thing scrolls it in the browser and gives back true."""
        mock_browser_mgr.scroll_to_view.return_value = True
        snap = await PageSnapshot.create("<html><body><button aria-label='Submit'>Submit</button></body></html>", mock_browser_mgr, "tab1")
        elements = snap.actionable_elements()
        if elements:
            assert await snap.scroll_to_view(elements[0]["ref_id"]) is True
            mock_browser_mgr.scroll_to_view.assert_called()

    async def test_page_snapshot_scroll_verbose(self, mock_browser_mgr: AsyncMock) -> None:
        """Scrolling into view with logging on passes that on."""
        mock_browser_mgr.scroll_to_view.return_value = True
        snap = await PageSnapshot.create("<html><body><button>Click</button></body></html>", mock_browser_mgr, "tab1")
        elements = snap.actionable_elements()
        if elements:
            assert await snap.scroll_to_view(elements[0]["ref_id"], verbose=True) is True
            call_kwargs = mock_browser_mgr.scroll_to_view.call_args[1]
            assert call_kwargs["verbose"] is True

    async def test_page_snapshot_scroll_invalid_ref_raises(self, page_snapshot_button: PageSnapshot) -> None:
        """Scrolling to an id that is not there raises an error."""
        with pytest.raises(ValueError):
            await page_snapshot_button.scroll_to_view("e999")

    async def test_page_snapshot_scroll_returns_false(self, mock_browser_mgr: AsyncMock) -> None:
        """A scroll into view that fails gives back false."""
        mock_browser_mgr.scroll_to_view.return_value = False
        snap = await PageSnapshot.create("<html><body><button>Click</button></body></html>", mock_browser_mgr, "tab1")
        elements = snap.actionable_elements()
        if elements:
            assert await snap.scroll_to_view(elements[0]["ref_id"]) is False


# ---------------------------------------------------------------------------
# Finding things to act on and reaching them by id
# ---------------------------------------------------------------------------


class TestActionableElementsPublicAPI:
    """Tests for which parts of a page count as things you can act on."""

    async def test_div_with_tabindex_is_actionable(self, mock_browser_mgr: AsyncMock) -> None:
        """A box you can tab to is listed."""
        snap = await PageSnapshot.create("<html><body><div tabindex='0' role='button'>Open</div></body></html>", mock_browser_mgr, "tab1")
        elements = snap.actionable_elements()
        assert len(elements) >= 1
        assert any(elem.get("name") == "Open" for elem in elements)

    async def test_div_with_onclick_is_actionable(self, mock_browser_mgr: AsyncMock) -> None:
        """A box with a click handler is listed."""
        snap = await PageSnapshot.create("<html><body><div onclick='doStuff()'>Click me</div></body></html>", mock_browser_mgr, "tab1")
        elements = snap.actionable_elements()
        assert len(elements) >= 1
        assert any(elem.get("name") == "Click me" for elem in elements)

    async def test_div_with_cursor_pointer_is_actionable(self, mock_browser_mgr: AsyncMock) -> None:
        """A box that shows a hand cursor is listed."""
        snap = await PageSnapshot.create("<html><body><div style='cursor:pointer'>Hover</div></body></html>", mock_browser_mgr, "tab1")
        elements = snap.actionable_elements()
        assert len(elements) >= 1
        assert any(elem.get("name") == "Hover" for elem in elements)

    async def test_div_with_aria_role_not_duplicated(self, mock_browser_mgr: AsyncMock) -> None:
        """A box that acts as a button is listed once."""
        snap = await PageSnapshot.create("<html><body><div role='button' aria-label='Toggle'>Toggle</div></body></html>", mock_browser_mgr, "tab1")
        elements = snap.actionable_elements()
        toggle_elements = [e for e in elements if e.get("name") == "Toggle"]
        assert len(toggle_elements) == 1

    async def test_hidden_div_not_actionable(self, mock_browser_mgr: AsyncMock) -> None:
        """A hidden box is not listed."""
        snap = await PageSnapshot.create("<html><body><div hidden>Hidden</div></body></html>", mock_browser_mgr, "tab1")
        elements = snap.actionable_elements()
        hidden_elements = [e for e in elements if "Hidden" in e.get("name", "")]
        assert len(hidden_elements) == 0

    async def test_div_with_long_text_not_actionable(self, mock_browser_mgr: AsyncMock) -> None:
        """A box with a lot of text is not listed."""
        html = "<html><body><div>This is a very long paragraph of text that exceeds the character limit for actionable elements and should therefore be excluded from the results</div></body></html>"
        snap = await PageSnapshot.create(html, mock_browser_mgr, "tab1")
        elements = snap.actionable_elements()
        assert len(elements) == 0

    async def test_regular_div_not_actionable(self, mock_browser_mgr: AsyncMock) -> None:
        """A plain box you cannot act on is not listed."""
        snap = await PageSnapshot.create("<html><body><div>Just text</div></body></html>", mock_browser_mgr, "tab1")
        elements = snap.actionable_elements()
        regular_divs = [e for e in elements if "Just text" in e.get("name", "")]
        assert len(regular_divs) == 0

    async def test_div_without_text_not_actionable(self, mock_browser_mgr: AsyncMock) -> None:
        """An empty box is not listed."""
        snap = await PageSnapshot.create("<html><body><div></div></body></html>", mock_browser_mgr, "tab1")
        elements = snap.actionable_elements()
        assert len(elements) == 0


class TestSelectorResolutionPublicAPI:
    """Tests that an id points at the right thing on the page."""

    async def test_ref_resolves_to_selector_with_attributes(self, mock_browser_mgr: AsyncMock) -> None:
        """A button with an id is found by that id."""
        mock_browser_mgr.click.return_value = True
        snap = await PageSnapshot.create("<html><body><button id='my-btn' aria-label='Submit'>Submit</button></body></html>", mock_browser_mgr, "tab1")
        elements = snap.actionable_elements()
        if elements:
            await snap.click(elements[0]["ref_id"])
            selector = mock_browser_mgr.click.call_args[0][1]
            assert "my-btn" in selector, f"Expected 'my-btn' in selector, got: {selector}"

    @pytest.mark.parametrize("label", ["Search for properties", "City, neighborhood, ZIP"])
    async def test_shared_id_gives_a_selector_that_matches_one_part(
        self, mock_browser_mgr: AsyncMock, label: str
    ) -> None:
        """A part whose id is used twice is pointed at by a selector that matches it alone."""
        mock_browser_mgr.click.return_value = True
        snap = await PageSnapshot.create(_DUPLICATE_ID_HTML, mock_browser_mgr, "tab1")
        elements = [elem for elem in snap.actionable_elements() if elem.get("name") == label]
        assert len(elements) == 1, f"Expected one part named '{label}', got: {elements}"
        await snap.click(elements[0]["ref_id"])
        selector = mock_browser_mgr.click.call_args[0][1]
        matches = BeautifulSoup(_DUPLICATE_ID_HTML, "html.parser").select(selector)
        assert len(matches) == 1, f"Selector '{selector}' matches {len(matches)} parts, expected 1"
        assert matches[0].get("aria-label") == label

    async def test_selector_handles_special_chars(self, mock_browser_mgr: AsyncMock) -> None:
        """A button whose label has odd characters can still be clicked."""
        mock_browser_mgr.click.return_value = True
        snap = await PageSnapshot.create("<html><body><button aria-label='Price: $5 & up'>Price</button></body></html>", mock_browser_mgr, "tab1")
        elements = snap.actionable_elements()
        if elements:
            await snap.click(elements[0]["ref_id"])
            mock_browser_mgr.click.assert_called()

    async def test_selector_handles_unicode(self, mock_browser_mgr: AsyncMock) -> None:
        """A button labeled in another writing system can still be clicked."""
        mock_browser_mgr.click.return_value = True
        snap = await PageSnapshot.create("<html><body><button>検索</button></body></html>", mock_browser_mgr, "tab1")
        elements = snap.actionable_elements()
        if elements:
            await snap.click(elements[0]["ref_id"])
            mock_browser_mgr.click.assert_called()

    async def test_selector_handles_newline_in_text(self, mock_browser_mgr: AsyncMock) -> None:
        """A button with a line break in its text can still be clicked."""
        mock_browser_mgr.click.return_value = True
        snap = await PageSnapshot.create("<html><body><button>Line1\nLine2</button></body></html>", mock_browser_mgr, "tab1")
        elements = snap.actionable_elements()
        if elements:
            await snap.click(elements[0]["ref_id"])
            mock_browser_mgr.click.assert_called()

    async def test_selector_with_long_text_truncated(self, mock_browser_mgr: AsyncMock) -> None:
        """A button with very long text can still be clicked."""
        long_text = "A" * 200
        html = f"<html><body><button>{long_text}</button></body></html>"
        mock_browser_mgr.click.return_value = True
        snap = await PageSnapshot.create(html, mock_browser_mgr, "tab1")
        elements = snap.actionable_elements()
        if elements:
            await snap.click(elements[0]["ref_id"])
            mock_browser_mgr.click.assert_called()

    async def test_selector_without_attributes_or_text(self, mock_browser_mgr: AsyncMock) -> None:
        """A button with no label and no text can still be clicked."""
        mock_browser_mgr.click.return_value = True
        snap = await PageSnapshot.create("<html><body><button></button></body></html>", mock_browser_mgr, "tab1")
        elements = snap.actionable_elements()
        if elements:
            await snap.click(elements[0]["ref_id"])
            mock_browser_mgr.click.assert_called()


# ---------------------------------------------------------------------------
# Acting through the snapshot
# ---------------------------------------------------------------------------


class TestPageSnapshotCounterparts:
    """Tests that the snapshot's actions go to the tab it came from."""

    async def test_click_at_relative_delegates_with_viewport(self, mock_browser_mgr: AsyncMock) -> None:
        """A window-share click goes to the tab with the snapshot's window."""
        mock_browser_mgr.click_at_relative.return_value = True
        snap = await PageSnapshot.create("<html><body>test</body></html>", mock_browser_mgr, "tab1", viewport=_WINDOW)
        result = await snap.click_at_relative(_REL_CLICK_X, _REL_CLICK_Y, verbose=True)
        assert result is True
        mock_browser_mgr.click_at_relative.assert_called_once_with("tab1", _REL_CLICK_X, _REL_CLICK_Y, viewport=_WINDOW, verbose=True)

    async def test_click_at_relative_without_viewport_passes_none(self, mock_browser_mgr: AsyncMock) -> None:
        """A window-share click with no window hands none on."""
        mock_browser_mgr.click_at_relative.return_value = True
        snap = await PageSnapshot.create("<html><body>test</body></html>", mock_browser_mgr, "tab1")
        await snap.click_at_relative(_SHARE_CENTER, _SHARE_CENTER)
        assert mock_browser_mgr.click_at_relative.call_args.kwargs["viewport"] is None

    async def test_scroll_by_relative_delegates_with_viewport(self, mock_browser_mgr: AsyncMock) -> None:
        """A window-share scroll goes to the tab with the snapshot's window."""
        mock_browser_mgr.scroll_by_relative.return_value = True
        snap = await PageSnapshot.create("<html><body>test</body></html>", mock_browser_mgr, "tab1", viewport=_WINDOW)
        result = await snap.scroll_by_relative(_SHARE_LEFT, _SHARE_CENTER)
        assert result is True
        mock_browser_mgr.scroll_by_relative.assert_called_once_with("tab1", _SHARE_LEFT, _SHARE_CENTER, viewport=_WINDOW, verbose=False)

    async def test_type_text_at_delegates(self, mock_browser_mgr: AsyncMock) -> None:
        """Typing at a spot on the page goes to the tab the snapshot came from."""
        mock_browser_mgr.type_text_at.return_value = True
        snap = await PageSnapshot.create("<html><body>test</body></html>", mock_browser_mgr, "tab1")
        result = await snap.type_text_at(_CLICK_AT_X, _CLICK_AT_Y, "hello")
        assert result is True
        mock_browser_mgr.type_text_at.assert_called_once_with("tab1", _CLICK_AT_X, _CLICK_AT_Y, "hello", submit=False, verbose=False)

    async def test_type_text_at_relative_delegates_with_viewport(self, mock_browser_mgr: AsyncMock) -> None:
        """Typing at a window-share spot goes to the tab with the snapshot's window."""
        mock_browser_mgr.type_text_at_relative.return_value = True
        snap = await PageSnapshot.create("<html><body>test</body></html>", mock_browser_mgr, "tab1", viewport=_WINDOW)
        result = await snap.type_text_at_relative(_SHARE_CENTER, _SHARE_CENTER, "hello")
        assert result is True
        mock_browser_mgr.type_text_at_relative.assert_called_once_with(
            "tab1", _SHARE_CENTER, _SHARE_CENTER, "hello", submit=False, viewport=_WINDOW, verbose=False
        )

    async def test_type_text_at_relative_with_submit_passes_it_on(self, mock_browser_mgr: AsyncMock) -> None:
        """Typing with submit on hands the Enter press to the tab."""
        mock_browser_mgr.type_text_at_relative.return_value = True
        snap = await PageSnapshot.create("<html><body>test</body></html>", mock_browser_mgr, "tab1", viewport=_WINDOW)
        result = await snap.type_text_at_relative(_SHARE_CENTER, _SHARE_CENTER, "hello", submit=True)
        assert result is True
        assert mock_browser_mgr.type_text_at_relative.call_args.kwargs["submit"] is True

    async def test_replace_input_text_at_delegates(self, mock_browser_mgr: AsyncMock) -> None:
        """Replacing text at a spot on the page goes to the tab the snapshot came from."""
        mock_browser_mgr.replace_input_text_at.return_value = True
        snap = await PageSnapshot.create("<html><body>test</body></html>", mock_browser_mgr, "tab1")
        result = await snap.replace_input_text_at(_CLICK_AT_X, _CLICK_AT_Y, "helo", "hello", verbose=True)
        assert result is True
        mock_browser_mgr.replace_input_text_at.assert_called_once_with("tab1", _CLICK_AT_X, _CLICK_AT_Y, "helo", "hello", verbose=True)

    async def test_replace_input_text_at_relative_delegates_with_viewport(self, mock_browser_mgr: AsyncMock) -> None:
        """Replacing text at a window-share spot goes to the tab with the snapshot's window."""
        mock_browser_mgr.replace_input_text_at_relative.return_value = False
        snap = await PageSnapshot.create("<html><body>test</body></html>", mock_browser_mgr, "tab1", viewport=_WINDOW)
        result = await snap.replace_input_text_at_relative(_SHARE_CENTER, _SHARE_CENTER, "helo", "")
        assert result is False
        mock_browser_mgr.replace_input_text_at_relative.assert_called_once_with(
            "tab1", _SHARE_CENTER, _SHARE_CENTER, "helo", "", viewport=_WINDOW, verbose=False
        )

    async def test_go_forward_delegates(self, mock_browser_mgr: AsyncMock) -> None:
        """Going forward goes to the tab the snapshot came from."""
        mock_browser_mgr.go_forward.return_value = True
        snap = await PageSnapshot.create("<html><body>test</body></html>", mock_browser_mgr, "tab1")
        result = await snap.go_forward(verbose=True)
        assert result is True
        mock_browser_mgr.go_forward.assert_called_once_with("tab1", verbose=True)

    async def test_go_back_delegates(self, mock_browser_mgr: AsyncMock) -> None:
        """Going back goes to the tab the snapshot came from."""
        mock_browser_mgr.go_back.return_value = False
        snap = await PageSnapshot.create("<html><body>test</body></html>", mock_browser_mgr, "tab1")
        result = await snap.go_back()
        assert result is False
        mock_browser_mgr.go_back.assert_called_once_with("tab1", verbose=False)

    async def test_close_page_delegates(self, mock_browser_mgr: AsyncMock) -> None:
        """Closing the tab goes to the tab the snapshot came from."""
        mock_browser_mgr.close_page.return_value = True
        snap = await PageSnapshot.create("<html><body>test</body></html>", mock_browser_mgr, "tab1")
        result = await snap.close_page()
        assert result is True
        mock_browser_mgr.close_page.assert_called_once_with("tab1", verbose=False)


class TestPageSnapshotMoveMouseToRelativeViewport:
    """Tests that a window-share mouse move carries the snapshot's window."""

    async def test_move_mouse_to_relative_passes_viewport(self, mock_browser_mgr: AsyncMock) -> None:
        """A window-share mouse move hands the snapshot's window to the browser."""
        mock_browser_mgr.move_mouse_to_relative.return_value = True
        snap = await PageSnapshot.create("<html><body>test</body></html>", mock_browser_mgr, "tab1", viewport=_WINDOW)
        result = await snap.move_mouse_to_relative(_SHARE_LEFT, _SHARE_THREE_QUARTERS)
        assert result is True
        mock_browser_mgr.move_mouse_to_relative.assert_called_once_with("tab1", _SHARE_LEFT, _SHARE_THREE_QUARTERS, viewport=_WINDOW, verbose=False)

    async def test_move_mouse_to_relative_without_viewport_passes_none(self, mock_browser_mgr: AsyncMock) -> None:
        """A window-share mouse move with no window hands none on."""
        mock_browser_mgr.move_mouse_to_relative.return_value = True
        snap = await PageSnapshot.create("<html><body>test</body></html>", mock_browser_mgr, "tab1")
        await snap.move_mouse_to_relative(_SHARE_LEFT, _SHARE_THREE_QUARTERS)
        assert mock_browser_mgr.move_mouse_to_relative.call_args.kwargs["viewport"] is None


class TestPageSnapshotNoWaitForDownload:
    """Tests that a snapshot has no download wait of its own."""

    async def test_snapshot_has_no_wait_for_download(self) -> None:
        """A snapshot cannot wait for a file download."""
        assert "wait_for_download" not in vars(PageSnapshot)


class TestPageSnapshotNoNavigate:
    """Tests that a snapshot has no navigation of its own."""

    async def test_snapshot_has_no_navigate(self) -> None:
        """A snapshot cannot send its tab to a new address."""
        assert "navigate" not in vars(PageSnapshot)


