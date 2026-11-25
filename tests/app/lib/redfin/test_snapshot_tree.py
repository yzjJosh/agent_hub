"""Tests for Redfin snapshot tree traversal helpers."""

from __future__ import annotations

import pytest

from app.lib.redfin.snapshot_tree import (
    collect_node_text,
    find_actionable_elements,
    find_descendant_text,
    find_label_sibling_buttons,
    find_matching_nodes,
    gather_node_texts,
    is_listing_card_node,
)
from tests.app.lib.redfin.conftest import make_fake_snapshot


def test_find_actionable_elements_filters_by_role_and_name() -> None:
    """Actionable elements can be narrowed by role, exact name, contains, and prefix."""
    snapshot = make_fake_snapshot(
        [
            {"ref_id": "1", "role": "textbox", "name": "Search for properties"},
            {"ref_id": "2", "role": "button", "name": "Filters"},
            {"ref_id": "3", "role": "button", "name": "All Filters"},
            {"ref_id": "4", "role": "link", "name": "Download results"},
        ]
    )

    assert find_actionable_elements(snapshot, role="textbox", name_prefix="Search") == [
        {"ref_id": "1", "role": "textbox", "name": "Search for properties"}
    ]
    assert find_actionable_elements(snapshot, role="button", name="Filters") == [
        {"ref_id": "2", "role": "button", "name": "Filters"}
    ]
    assert find_actionable_elements(snapshot, role="button", name_contains="filter") == [
        {"ref_id": "2", "role": "button", "name": "Filters"},
        {"ref_id": "3", "role": "button", "name": "All Filters"},
    ]


def test_find_matching_nodes_walks_tree_and_skips_text_leaves() -> None:
    """Matching nodes are collected recursively while text-only nodes are ignored."""
    tree = {
        "role": "WebArea",
        "children": [
            {"text": "Property at 123 Main St"},
            {
                "role": "button",
                "name": "Property at 123 Main St, 3 beds",
                "children": [
                    {"role": "link", "name": "123 Main St", "url": "/home/1"},
                ],
            },
            {"role": "heading", "name": "SOLD JUN 5, 2026"},
        ],
    }

    cards = find_matching_nodes(tree, role="button", name_prefix="Property at")
    assert len(cards) == 1
    assert cards[0]["name"] == "Property at 123 Main St, 3 beds"

    sold_nodes = find_matching_nodes(tree, name_pattern=r"SOLD\s+[A-Z]{3}")
    assert sold_nodes == [{"role": "heading", "name": "SOLD JUN 5, 2026"}]

    assert find_matching_nodes(tree) == []


def test_find_label_sibling_buttons_reads_until_next_filter_label() -> None:
    """Sibling buttons and nested option words after a label stop at the next label."""
    tree = {
        "role": "WebArea",
        "children": [
            {"role": "label", "name": "Beds"},
            {"role": "button", "name": "Any"},
            {
                "role": "group",
                "children": [
                    {"text": "2 3"},
                    {"role": "button", "name": "4"},
                ],
            },
            {"role": "label", "name": "Baths"},
            {"role": "button", "name": "2+"},
            {"role": "button", "name": "3+"},
        ],
    }

    assert find_label_sibling_buttons(tree, "Beds") == ["Any", "2", "3", "4"]
    assert find_label_sibling_buttons(tree, "Baths") == ["2+", "3+"]
    assert find_label_sibling_buttons(tree, "Garage") == []


def test_find_descendant_text_returns_first_matching_text_leaf() -> None:
    """Descendant search finds the first matching text leaf or returns None."""
    card = {
        "role": "button",
        "name": "Property at 123 Main St",
        "children": [
            {"role": "generic", "children": [{"text": "$850,000"}]},
            {"text": "1,600 sq ft"},
        ],
    }

    assert find_descendant_text(card, r"^\$\d") == "$850,000"
    assert find_descendant_text(card, r"sq ft") == "1,600 sq ft"
    assert find_descendant_text(card, r"acres") is None


@pytest.mark.parametrize(
    ("node", "expected"),
    [
        ({"role": "button", "name": "Property at 123 Main St, 3 beds"}, True),
        ({"role": "link", "name": "Property at 123 Main St"}, False),
        ({"role": "button", "name": "Filters"}, False),
    ],
)
def test_is_listing_card_node(node: dict, expected: bool) -> None:
    """Listing card check matches only button nodes starting with the card prefix."""
    assert is_listing_card_node(node) is expected


def test_gather_and_collect_node_text_support_skipping_listing_cards() -> None:
    """Text gathering preserves page order and can omit nested listing cards."""
    tree = {
        "role": "WebArea",
        "children": [
            {"text": "For sale $1,200,000"},
            {
                "role": "button",
                "name": "Property at 999 Other St, 2 beds",
                "children": [{"text": "$500,000"}],
            },
            {"text": "San Jose, CA 95117"},
        ],
    }

    assert gather_node_texts(tree) == [
        "For sale $1,200,000",
        "$500,000",
        "San Jose, CA 95117",
    ]
    assert gather_node_texts(tree, should_skip_listing_cards=True) == [
        "For sale $1,200,000",
        "San Jose, CA 95117",
    ]
    assert collect_node_text(tree) == "For sale $1,200,000 $500,000 San Jose, CA 95117"
    assert collect_node_text(None) == ""
