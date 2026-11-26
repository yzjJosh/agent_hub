"""Turns files into rich views for the resource viewer."""

from app.lib.render.classify import EXT_TO_KIND, RenderKind, classify
from app.lib.render.detect import language_extension
from app.lib.render.thumbs import get_thumbnail_engine

__all__ = [
    "EXT_TO_KIND",
    "RenderKind",
    "classify",
    "get_thumbnail_engine",
    "language_extension",
]
