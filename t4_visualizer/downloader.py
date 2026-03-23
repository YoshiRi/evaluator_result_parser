"""T4 dataset downloader using ``webauto data annotation-dataset pull``.

Default download command
------------------------
    webauto data annotation-dataset pull \\
        --project-id <project_id> \\
        --annotation-dataset-id <t4dataset_id> \\
        --output <dest_dir>/<t4dataset_id>

Configuration (in order of precedence)
---------------------------------------
1. Environment variable ``T4_DOWNLOAD_CMD`` (full shell template):

       T4_DOWNLOAD_CMD="webauto data annotation-dataset pull \\
           --project-id my_proj \\
           --annotation-dataset-id {t4dataset_id} \\
           --output {dataset_path}"

   Placeholders: ``{t4dataset_id}``, ``{dest_dir}``, ``{dataset_path}``
   (``{dataset_path}`` == ``{dest_dir}/{t4dataset_id}``)

2. Module-level constants ``WEBAUTO_PROJECT_ID`` / ``DOWNLOAD_CMD_TEMPLATE``.

3. Environment variable ``WEBAUTO_PROJECT_ID`` to override the project ID only.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Configuration — edit these to match your environment
# ---------------------------------------------------------------------------

# webauto project ID.  Override with env var WEBAUTO_PROJECT_ID.
WEBAUTO_PROJECT_ID: str = os.environ.get("WEBAUTO_PROJECT_ID", "x2_dev")

# Full shell command template.  Overrides the default webauto command when set.
# Available placeholders: {t4dataset_id}, {dest_dir}, {dataset_path}
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

    print(f"  [downloader] Downloading {t4dataset_id} → {expected_path}")

    if DOWNLOAD_CMD_TEMPLATE:
        _run_cmd_template(DOWNLOAD_CMD_TEMPLATE, t4dataset_id, dest_dir, expected_path)
    else:
        _download_impl(t4dataset_id, dest_dir, expected_path)

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


def _run_cmd_template(
    template: str, t4dataset_id: str, dest_dir: Path, dataset_path: Path
) -> None:
    cmd = template.format(
        t4dataset_id=t4dataset_id,
        dest_dir=str(dest_dir),
        dataset_path=str(dataset_path),
    )
    print(f"  [downloader] Running: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        raise DownloadError(f"Download command failed (exit {result.returncode}): {cmd}")


def _download_impl(t4dataset_id: str, dest_dir: Path, dataset_path: Path) -> None:
    """Download via ``webauto data annotation-dataset pull``.

    Downloads the dataset directly into ``dataset_path``
    (= ``dest_dir / t4dataset_id``) so that the expected path check passes.
    """
    cmd = [
        "webauto", "data", "annotation-dataset", "pull",
        "--project-id", WEBAUTO_PROJECT_ID,
        "--annotation-dataset-id", t4dataset_id,
        "--output", str(dataset_path),
    ]
    print(f"  [downloader] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise DownloadError(
            f"webauto download failed (exit {result.returncode}) for dataset '{t4dataset_id}'.\n"
            f"Command: {' '.join(cmd)}\n"
            "Check that `webauto` is installed and you are logged in."
        )
