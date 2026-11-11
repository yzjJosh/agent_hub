"""Serve a folder over WebDAV and drop stale render entries as it changes.

After a file is written, deleted, or moved, the render entries for the paths
it touched are dropped. A failure while dropping them never fails the WebDAV
response. A GET of a file counts as a use of it, unless the request asks for
an ephemeral read.
"""

import os
import sys
from typing import BinaryIO, override
from urllib.parse import parse_qs

from wsgidav.fs_dav_provider import FileResource, FilesystemProvider, FolderResource

from app.lib.render.cache import invalidate_sync
from app.lib.shared import activity
from app.lib.shared.paths import EPHEMERAL_PARAM

__all__ = ["CacheInvalidatingProvider"]


def _invalidate(rel: str | None) -> None:
    """Drop the render entries for one path under the share.

    Nothing is raised. Each dropped path is printed, and so is a failure.

    Args:
        rel: Path under the share root, or None to do nothing.
    """
    if rel is None:
        return
    try:
        for path in invalidate_sync(rel):
            print(path, file=sys.stderr)
    except Exception as exc:
        print(f"[resources.dav_hook] invalidation failed for {rel}: {exc}", file=sys.stderr)


def _share_rel(share_root: str, provider_root: str, dav_path: str) -> str | None:
    """Turn a WebDAV path into a path under the share root.

    Args:
        share_root: Root folder of the share.
        provider_root: Root folder this provider serves.
        dav_path: Path from the WebDAV request.

    Returns:
        The path under the share root, or None when it points outside the
        share.
    """
    file_path = os.path.realpath(os.path.join(provider_root, dav_path.lstrip("/")))
    rel = os.path.relpath(file_path, os.path.realpath(share_root))
    if rel in (".", os.pardir) or rel.startswith(os.pardir + os.sep):
        return None
    return rel


class _InvalidatingResource:
    """Shared behavior that drops render entries after a file changes."""

    _share_root: str

    def _rel(self, dav_path: str) -> str | None:
        return _share_rel(self._share_root, self.provider.root_folder_path, dav_path)

    def delete(self) -> None:
        super().delete()
        _invalidate(self._rel(self.path))

    def copy_move_single(self, dest_path: str, *, is_move: bool) -> None:
        super().copy_move_single(dest_path, is_move=is_move)
        _invalidate(self._rel(dest_path))
        if is_move:
            _invalidate(self._rel(self.path))

    def move_recursive(self, dest_path: str) -> None:
        super().move_recursive(dest_path)
        _invalidate(self._rel(self.path))
        _invalidate(self._rel(dest_path))


class _InvalidatingFileResource(_InvalidatingResource, FileResource):
    def __init__(self, path: str, environ: dict, file_path: str, share_root: str) -> None:
        super().__init__(path, environ, file_path)
        self._share_root = share_root

    @override
    def get_content(self) -> BinaryIO:
        content = super().get_content()
        try:
            if self._is_use():
                activity.record_sync(self._rel(self.path) or "", self._share_root)
        except OSError:
            content.close()
            raise
        return content

    def _is_use(self) -> bool:
        """Tell whether this request reads the file as a use of it."""
        query = parse_qs(self.environ.get("QUERY_STRING", ""), keep_blank_values=True)
        return self.environ.get("REQUEST_METHOD") == "GET" and EPHEMERAL_PARAM not in query

    def end_write(self, *, with_errors: bool) -> None:
        super().end_write(with_errors=with_errors)
        _invalidate(self._rel(self.path))


class _InvalidatingFolderResource(_InvalidatingResource, FolderResource):
    def __init__(self, path: str, environ: dict, file_path: str, share_root: str) -> None:
        super().__init__(path, environ, file_path)
        self._share_root = share_root


class CacheInvalidatingProvider(FilesystemProvider):
    """Serves a folder over WebDAV and drops render entries after changes."""

    def __init__(
        self,
        root_folder: str,
        *,
        share_root: str,
        readonly: bool = False,
        fs_opts: dict | None = None,
    ) -> None:
        super().__init__(root_folder, readonly=readonly, fs_opts=fs_opts)
        self._share_root = share_root

    def get_resource_inst(
        self, path: str, environ: dict
    ) -> _InvalidatingFileResource | _InvalidatingFolderResource | None:
        resource = super().get_resource_inst(path, environ)
        if resource is None:
            return None
        file_path = os.path.realpath(os.path.join(self.root_folder_path, path.lstrip("/")))
        if resource.is_collection:
            return _InvalidatingFolderResource(path, environ, file_path, self._share_root)
        return _InvalidatingFileResource(path, environ, file_path, self._share_root)
