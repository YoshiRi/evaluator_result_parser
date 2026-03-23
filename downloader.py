"""T4 dataset downloader — pluggable interface.

This module defines the contract for downloading T4 datasets by ID and provides
a stub implementation that you can replace with your actual download logic.

How to plug in your real downloader
------------------------------------
Option A — Replace the function body of ``_download_impl``:

    def _download_impl(t4dataset_id: str, dest_dir: Path) -> Path:
        # call your existing download script here
        import subprocess
        subprocess.run(["your_download_script.py", t4dataset_id, str(dest_dir)], check=True)
        return dest_dir / t4dataset_id

Option B — Point DOWNLOAD_SCRIPT_PATH to your CLI script:

    DOWNLOAD_SCRIPT_PATH = "/path/to/download_t4dataset.py"

    The script must accept:  <script> <t4dataset_id> <dest_dir>
    and download the dataset into ``dest_dir/<t4dataset_id>/``.

Option C — Set the environment variable T4_DOWNLOAD_CMD:

    T4_DOWNLOAD_CMD="my_download_tool {t4dataset_id} {dest_dir}"
    The placeholders ``{t4dataset_id}`` and ``{dest_dir}`` are expanded at runtime.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Configuration — edit these to match your environment
# ---------------------------------------------------------------------------

# Path to an existing CLI download script.
# Set to None to use the Python function _download_impl() instead.
DOWNLOAD_SCRIPT_PATH: Optional[str] = None

# Shell command template. Overrides DOWNLOAD_SCRIPT_PATH if set.
# Example: "my_downloader --id {t4dataset_id} --out {dest_dir}"
DOWNLOAD_CMD_TEMPLATE: Optional[str] = os.environ.get("T4_DOWNLOAD_CMD")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class DownloadError(RuntimeError):
    """Raised when a dataset download fails."""


def download_dataset(t4dataset_id: str, dest_dir: Path) -> Path:
    """Download a T4 dataset by its ID into *dest_dir* and return the dataset path.

    The downloaded dataset is expected to end up at::

        dest_dir / <t4dataset_id> /

    Args:
        t4dataset_id: Dataset identifier string (UUID or name).
        dest_dir: Directory where the dataset should be placed.

    Returns:
        Path to the downloaded dataset root (``dest_dir / t4dataset_id``).

    Raises:
        DownloadError: If the download fails or the expected directory is not found.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    expected_path = dest_dir / t4dataset_id

    # Skip download if already present
    if expected_path.exists() and _looks_like_t4dataset(expected_path):
        print(f"  [downloader] Dataset already exists, skipping download: {expected_path}")
        return expected_path

    print(f"  [downloader] Downloading {t4dataset_id} → {dest_dir}")

    if DOWNLOAD_CMD_TEMPLATE:
        _run_cmd_template(DOWNLOAD_CMD_TEMPLATE, t4dataset_id, dest_dir)
    elif DOWNLOAD_SCRIPT_PATH:
        _run_script(DOWNLOAD_SCRIPT_PATH, t4dataset_id, dest_dir)
    else:
        _download_impl(t4dataset_id, dest_dir)

    if not expected_path.exists():
        raise DownloadError(
            f"Download finished but expected path not found: {expected_path}\n"
            "Please check that your download logic places the dataset at "
            f"<dest_dir>/<t4dataset_id>/ (i.e. {expected_path})."
        )
    if not _looks_like_t4dataset(expected_path):
        raise DownloadError(
            f"Downloaded path exists but does not look like a T4 dataset: {expected_path}\n"
            "Expected an 'annotation' or 'v1.0-*' subdirectory with JSON files."
        )

    print(f"  [downloader] Ready: {expected_path}")
    return expected_path


def dataset_is_cached(t4dataset_id: str, dest_dir: Path) -> bool:
    """Return True if the dataset already exists and looks valid."""
    path = Path(dest_dir) / t4dataset_id
    return path.exists() and _looks_like_t4dataset(path)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _looks_like_t4dataset(path: Path) -> bool:
    """Heuristic check: does this directory contain T4 annotation files?"""
    # T4 datasets have an annotation directory (or versioned annotation dir)
    # containing sample.json / scene.json etc.
    for candidate in [
        path / "annotation",
        *sorted(path.glob("v1.0-*")),
        *sorted(path.glob("annotation/*")),
    ]:
        if (candidate / "sample.json").exists() or (candidate / "scene.json").exists():
            return True
    # Also accept if sample.json is directly in the path (flat layout)
    return (path / "sample.json").exists() or (path / "scene.json").exists()


def _run_cmd_template(template: str, t4dataset_id: str, dest_dir: Path) -> None:
    cmd = template.format(t4dataset_id=t4dataset_id, dest_dir=str(dest_dir))
    print(f"  [downloader] Running: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        raise DownloadError(f"Download command failed (exit {result.returncode}): {cmd}")


def _run_script(script_path: str, t4dataset_id: str, dest_dir: Path) -> None:
    cmd = [sys.executable, script_path, t4dataset_id, str(dest_dir)]
    print(f"  [downloader] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise DownloadError(
            f"Download script failed (exit {result.returncode}): {script_path}"
        )


def _download_impl(t4dataset_id: str, dest_dir: Path) -> None:
    """Stub implementation — replace this with your actual download logic.

    This stub raises an error to remind you to plug in the real downloader.
    """
    raise DownloadError(
        f"No downloader is configured for dataset '{t4dataset_id}'.\n\n"
        "To fix this, edit downloader.py and either:\n"
        "  1. Set DOWNLOAD_SCRIPT_PATH to your existing download script path.\n"
        "  2. Set DOWNLOAD_CMD_TEMPLATE or the T4_DOWNLOAD_CMD env variable.\n"
        "  3. Implement _download_impl() directly.\n\n"
        "See the module docstring in downloader.py for details."
    )
