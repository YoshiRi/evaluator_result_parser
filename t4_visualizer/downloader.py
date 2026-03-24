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

import fcntl
import json
import os
import shutil
import subprocess
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional


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
# LRU dataset cache
# ---------------------------------------------------------------------------

class DatasetCache:
    """LRU disk cache for downloaded T4 datasets.

    Tracks last-access times in a JSON index file and evicts the least
    recently used dataset directories when the cache exceeds *max_cached*.

    Index file: ``{data_dir}/.cache_index.json``
    Format:     ``{ "<t4dataset_id>": "<ISO-8601 last_accessed UTC>" }``

    A file lock (``{data_dir}/.cache_lock``) serialises concurrent
    updates from parallel batch workers.

    Args:
        data_dir: Directory where datasets are stored.
        max_cached: Maximum number of datasets to keep on disk.
            ``0`` disables eviction entirely.
    """

    _INDEX = ".cache_index.json"
    _LOCK  = ".cache_lock"

    def __init__(self, data_dir: Path, max_cached: int = 10):
        self.data_dir = Path(data_dir)
        self.max_cached = max_cached

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def ensure(self, t4dataset_id: str) -> Path:
        """Return the dataset path, downloading it first if necessary.

        Evicts LRU datasets before downloading a new one when the cache
        is at capacity.  Updates the last-accessed timestamp afterwards.
        """
        with self._lock():
            already_cached = self._on_disk(t4dataset_id)
            if not already_cached and self.max_cached > 0:
                self._evict_to(self.max_cached - 1)
            path = download_dataset(t4dataset_id, self.data_dir)
            self._touch(t4dataset_id)
        return path

    def touch(self, t4dataset_id: str) -> None:
        """Record that *t4dataset_id* was accessed right now."""
        with self._lock():
            self._touch(t4dataset_id)

    def evict_lru(self, keep: int) -> List[str]:
        """Evict datasets until at most *keep* remain.  Returns evicted IDs."""
        with self._lock():
            return self._evict_to(keep)

    def clear(self) -> List[str]:
        """Delete all cached datasets.  Returns list of deleted IDs."""
        with self._lock():
            return self._evict_to(0)

    def status(self) -> List[dict]:
        """Return cache entries sorted from LRU to MRU.

        Each entry is a dict with keys:
        ``t4dataset_id``, ``last_accessed``, ``size_mb``, ``on_disk``.
        """
        with self._lock():
            index = self._read_index()

        rows = []
        for did, ts in sorted(index.items(), key=lambda x: x[1]):
            path = self.data_dir / did
            on_disk = path.exists()
            size_mb = _dir_size_mb(path) if on_disk else 0.0
            rows.append({
                "t4dataset_id": did,
                "last_accessed": ts,
                "size_mb": round(size_mb, 1),
                "on_disk": on_disk,
                "path": str(path),
            })
        return rows

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _on_disk(self, t4dataset_id: str) -> bool:
        path = self.data_dir / t4dataset_id
        return path.exists() and _looks_like_t4dataset(path)

    def _touch(self, t4dataset_id: str) -> None:
        index = self._read_index()
        index[t4dataset_id] = datetime.now(timezone.utc).isoformat()
        self._write_index(index)

    def _evict_to(self, keep: int) -> List[str]:
        """Evict LRU entries until len(on-disk entries) <= keep."""
        index = self._read_index()
        # Sync: keep only entries whose directory actually exists
        on_disk = {k: v for k, v in index.items()
                   if (self.data_dir / k).exists()}
        evicted = []
        while len(on_disk) > keep:
            lru_id = min(on_disk, key=lambda k: on_disk[k])
            ts = on_disk.pop(lru_id)
            index.pop(lru_id, None)
            self._delete(lru_id)
            evicted.append(lru_id)
            print(f"  [cache] Evicted {lru_id}  (last accessed: {ts})")
        self._write_index(index)
        return evicted

    def _delete(self, t4dataset_id: str) -> None:
        path = self.data_dir / t4dataset_id
        if path.exists():
            shutil.rmtree(path)

    def _read_index(self) -> dict:
        index_path = self.data_dir / self._INDEX
        if not index_path.exists():
            return {}
        try:
            return json.loads(index_path.read_text())
        except Exception:
            return {}

    def _write_index(self, index: dict) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / self._INDEX).write_text(
            json.dumps(index, indent=2, sort_keys=True)
        )

    @contextmanager
    def _lock(self):
        """Exclusive file lock — serialises index access across processes."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        lock_path = self.data_dir / self._LOCK
        with open(lock_path, "w") as lf:
            fcntl.flock(lf, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lf, fcntl.LOCK_UN)


# ---------------------------------------------------------------------------
# t4-cache CLI entry point
# ---------------------------------------------------------------------------

def cache_main() -> None:
    """Entry point for the ``t4-cache`` command."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="t4-cache",
        description="Manage the local T4 dataset cache.",
    )
    parser.add_argument(
        "--data-dir",
        default="t4datasets",
        metavar="PATH",
        help="Dataset cache directory (default: ./t4datasets).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        metavar="N",
        help="Cache size limit used for 'evict' (default: 10).",
    )

    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status", help="Show cached datasets (LRU first).")

    evict_p = sub.add_parser("evict", help="Evict LRU datasets until --limit remain.")
    evict_p.add_argument(
        "--keep",
        type=int,
        default=None,
        metavar="N",
        help="Keep this many datasets (overrides --limit).",
    )

    sub.add_parser("clear", help="Delete all cached datasets.")

    args = parser.parse_args()
    cache = DatasetCache(Path(args.data_dir), max_cached=args.limit)

    if args.cmd == "status":
        rows = cache.status()
        if not rows:
            print("Cache is empty.")
            return
        total_mb = sum(r["size_mb"] for r in rows)
        print(f"{'#':<4} {'t4dataset_id':<40} {'last_accessed':<28} {'size_mb':>8}  on_disk")
        print("-" * 90)
        for i, r in enumerate(rows, 1):
            flag = "yes" if r["on_disk"] else "MISSING"
            print(f"{i:<4} {r['t4dataset_id']:<40} {r['last_accessed']:<28} "
                  f"{r['size_mb']:>8.1f}  {flag}")
        print("-" * 90)
        print(f"Total: {len(rows)} datasets, {total_mb:.1f} MB")
        print(f"Limit: {cache.max_cached if cache.max_cached > 0 else 'unlimited'}")

    elif args.cmd == "evict":
        keep = args.keep if args.keep is not None else args.limit
        evicted = cache.evict_lru(keep)
        if evicted:
            print(f"Evicted {len(evicted)} dataset(s): {evicted}")
        else:
            print("Nothing to evict.")

    elif args.cmd == "clear":
        deleted = cache.clear()
        if deleted:
            print(f"Cleared {len(deleted)} dataset(s): {deleted}")
        else:
            print("Cache was already empty.")


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


def _dir_size_mb(path: Path) -> float:
    """Return total size of a directory tree in megabytes."""
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / (1024 * 1024)


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

