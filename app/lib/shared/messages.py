"""Messages the tools return to the client about files."""

from __future__ import annotations

import os

from app.lib.shared.paths import resources_dir
from app.resources.resources import dav_base_url

__all__ = ["saved_file_message", "missing_reference_message"]


def saved_file_message(mode: str, output_path: str) -> str:
    """Return the message telling where the new file is and how to fetch it."""
    resource_path = os.path.relpath(output_path, resources_dir())
    return (
        f"{mode.capitalize()} generated and saved to resource path: {resource_path}\n"
        f"Download it with:\n"
        f"  curl -o /tmp/{os.path.basename(output_path)} "
        f"{dav_base_url()}/dav/{resource_path}\n"
        "See the send_or_receive_file tool for file transfer details."
    )


def missing_reference_message(resource_path: str) -> str:
    """Return the message for a reference image that is not in the resource."""
    return (
        f"Resource not found: {resource_path}. The resource path must already "
        "exist in this server's resources share. Call the send_or_receive_file "
        "tool for instructions to upload the file, then retry this tool with "
        "the uploaded file's resource path (e.g. 'upload/<filename>')."
    )
