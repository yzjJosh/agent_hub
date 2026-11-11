"""Page and error types from the browser drivers.

A name stands for an empty class when its driver is not installed.
"""
from __future__ import annotations

from typing import Type

__all__ = [
    "PlaywrightPage",
    "PlaywrightRequest",
    "PlaywrightRoute",
    "PlaywrightLocator",
    "PlaywrightFrameLocator",
    "PlaywrightElementHandle",
    "PlaywrightError",
    "PlaywrightTimeoutError",
    "PlaywrightAsyncLocator",
    "PlaywrightAsyncFrameLocator",
    "PlaywrightAsyncElementHandle",
    "PatchrightPage",
    "PatchrightRequest",
    "PatchrightRoute",
    "PatchrightLocator",
    "PatchrightFrameLocator",
    "PatchrightElementHandle",
    "PatchrightError",
    "PatchrightTimeoutError",
    "PatchrightAsyncLocator",
    "PatchrightAsyncFrameLocator",
    "PatchrightAsyncElementHandle",
]

try:
    from playwright.sync_api import Page as PlaywrightPage
except ImportError:
    PlaywrightPage: Type = type("Page", (), {})  # type: ignore[assignment]

try:
    from playwright.sync_api import Request as PlaywrightRequest
except ImportError:
    PlaywrightRequest: Type = type("Request", (), {})  # type: ignore[assignment]

try:
    from playwright.sync_api import Route as PlaywrightRoute
except ImportError:
    PlaywrightRoute: Type = type("Route", (), {})  # type: ignore[assignment]

try:
    from playwright.sync_api import Locator as PlaywrightLocator
except ImportError:
    PlaywrightLocator: Type = type("Locator", (), {})  # type: ignore[assignment]

try:
    from playwright.sync_api import FrameLocator as PlaywrightFrameLocator
except ImportError:
    PlaywrightFrameLocator: Type = type("FrameLocator", (), {})  # type: ignore[assignment]

try:
    from playwright.sync_api import ElementHandle as PlaywrightElementHandle
except ImportError:
    PlaywrightElementHandle: Type = type("ElementHandle", (), {})  # type: ignore[assignment]

try:
    from playwright.async_api import Error as PlaywrightError
except ImportError:
    PlaywrightError: Type = type("Error", (), {})  # type: ignore[assignment]

try:
    from playwright.async_api import TimeoutError as PlaywrightTimeoutError
except ImportError:
    PlaywrightTimeoutError: Type = type("TimeoutError", (), {})  # type: ignore[assignment]

try:
    from playwright.async_api import Locator as PlaywrightAsyncLocator
except ImportError:
    PlaywrightAsyncLocator: Type = type("Locator", (), {})  # type: ignore[assignment]

try:
    from playwright.async_api import FrameLocator as PlaywrightAsyncFrameLocator
except ImportError:
    PlaywrightAsyncFrameLocator: Type = type("FrameLocator", (), {})  # type: ignore[assignment]

try:
    from playwright.async_api import ElementHandle as PlaywrightAsyncElementHandle
except ImportError:
    PlaywrightAsyncElementHandle: Type = type("ElementHandle", (), {})  # type: ignore[assignment]

try:
    from patchright.sync_api import Page as PatchrightPage
except ImportError:
    PatchrightPage: Type = type("Page", (), {})  # type: ignore[assignment]

try:
    from patchright.sync_api import Request as PatchrightRequest
except ImportError:
    PatchrightRequest: Type = type("Request", (), {})  # type: ignore[assignment]

try:
    from patchright.sync_api import Route as PatchrightRoute
except ImportError:
    PatchrightRoute: Type = type("Route", (), {})  # type: ignore[assignment]

try:
    from patchright.sync_api import Locator as PatchrightLocator
except ImportError:
    PatchrightLocator: Type = type("Locator", (), {})  # type: ignore[assignment]

try:
    from patchright.sync_api import FrameLocator as PatchrightFrameLocator
except ImportError:
    PatchrightFrameLocator: Type = type("FrameLocator", (), {})  # type: ignore[assignment]

try:
    from patchright.sync_api import ElementHandle as PatchrightElementHandle
except ImportError:
    PatchrightElementHandle: Type = type("ElementHandle", (), {})  # type: ignore[assignment]

try:
    from patchright.async_api import Error as PatchrightError
except ImportError:
    PatchrightError: Type = type("Error", (), {})  # type: ignore[assignment]

try:
    from patchright.async_api import TimeoutError as PatchrightTimeoutError
except ImportError:
    PatchrightTimeoutError: Type = type("TimeoutError", (), {})  # type: ignore[assignment]

try:
    from patchright.async_api import Locator as PatchrightAsyncLocator
except ImportError:
    PatchrightAsyncLocator: Type = type("Locator", (), {})  # type: ignore[assignment]

try:
    from patchright.async_api import FrameLocator as PatchrightAsyncFrameLocator
except ImportError:
    PatchrightAsyncFrameLocator: Type = type("FrameLocator", (), {})  # type: ignore[assignment]

try:
    from patchright.async_api import ElementHandle as PatchrightAsyncElementHandle
except ImportError:
    PatchrightAsyncElementHandle: Type = type("ElementHandle", (), {})  # type: ignore[assignment]
