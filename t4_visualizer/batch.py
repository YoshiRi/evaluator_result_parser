"""Batch visualization pipeline for T4 dataset scenes.

Reads a CSV or Parquet file whose rows each identify one scene to visualize,
downloads the required datasets (if not already present), and produces
camera-image + LiDAR-BEV plots for every row.

Required columns in the input file
------------------------------------
- ``t4dataset_id`` : dataset identifier used to download / locate the data
- ``uuid``         : scene / sample UUID (used as part of the output filename)
- ``timestamp``    : target timestamp in **microseconds** (Unix time)

Optional columns (used when present)
--------------------------------------
- ``status``       : group label (e.g. "degrade", "improved") — images are placed
                    in a sub-directory named after this value; rows without a status
                    (or with NaN) go into an "unknown" sub-directory
- ``cameras``      : comma-separated camera channel names to render
- ``description``  : free-text label added to plot titles

Output structure
-----------------
All images for a given status are placed in the **same flat directory**:

    <output_dir>/<status>/<t4dataset_id>_<uuid>_<timestamp>_cameras.png
    <output_dir>/<status>/<t4dataset_id>_<uuid>_<timestamp>_pointcloud.png
    <output_dir>/<status>/<t4dataset_id>_<uuid>_<timestamp>_meta.txt

If no ``status`` column is present every image goes directly into <output_dir>.

Storage modes
--------------
``--temp``       Download datasets into a system temp directory that is
                 automatically deleted when the process exits.
``--data-dir``   Download / read datasets from a persistent directory (default:
                 ``./t4datasets/``). Already-downloaded datasets are reused.
``--data-dir`` can be combined with ``--keep`` to prevent deletion on exit.

Usage examples
--------------
    # Persistent storage (default)
    python batch_visualize.py scenes.csv --output-dir results/

    # Temporary storage — data vanishes after the script finishes
    python batch_visualize.py scenes.parquet --temp --output-dir results/

    # Custom persistent data directory
    python batch_visualize.py scenes.csv --data-dir /mnt/ssd/t4data --output-dir results/

    # Dry-run: skip download, assume datasets are already in ./t4datasets/
    python batch_visualize.py scenes.csv --no-download --output-dir results/

    # Parallel workers
    python batch_visualize.py scenes.csv --output-dir results/ --workers 4
"""

from __future__ import annotations

import argparse
import concurrent.futures
import sys
import tempfile
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Row dataclass
# ---------------------------------------------------------------------------

@dataclass
class SceneRow:
    """One row from the input CSV/Parquet."""
    t4dataset_id: str
    uuid: str
    timestamp_us: int
    status: Optional[str] = None        # e.g. "degrade", "improved"; None = no grouping
    cameras: Optional[List[str]] = field(default=None)
    description: str = ""


# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------

@dataclass
class RowResult:
    row: SceneRow
    success: bool
    output_dir: Optional[Path] = None
    error: str = ""


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = {"t4dataset_id", "uuid", "timestamp"}


def load_input(path: str) -> pd.DataFrame:
    """Load a CSV or Parquet file and validate required columns."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {p}")

    if p.suffix.lower() in (".parquet", ".pq"):
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Input file is missing required columns: {missing}\n"
            f"Found columns: {list(df.columns)}"
        )

    df["timestamp"] = df["timestamp"].astype("int64")
    return df


def df_to_rows(df: pd.DataFrame) -> List[SceneRow]:
    rows = []
    for _, r in df.iterrows():
        cameras = None
        if "cameras" in df.columns and pd.notna(r.get("cameras", None)):
            raw = str(r["cameras"]).strip()
            if raw:
                cameras = [c.strip() for c in raw.split(",") if c.strip()]

        status = None
        if "status" in df.columns:
            raw_status = r.get("status", None)
            if pd.notna(raw_status) and str(raw_status).strip():
                status = str(raw_status).strip()

        description = str(r.get("description", "")) if "description" in df.columns else ""
        rows.append(
            SceneRow(
                t4dataset_id=str(r["t4dataset_id"]),
                uuid=str(r["uuid"]),
                timestamp_us=int(r["timestamp"]),
                status=status,
                cameras=cameras,
                description=description,
            )
        )
    return rows


# ---------------------------------------------------------------------------
# Download management
# ---------------------------------------------------------------------------

def resolve_dataset_path(
    t4dataset_id: str,
    data_dir: Path,
    do_download: bool,
) -> Path:
    """Return the local path to a dataset, downloading it first if needed."""
    from t4_visualizer.downloader import download_dataset, dataset_is_cached, DownloadError

    if do_download:
        return download_dataset(t4dataset_id, data_dir)
    else:
        path = data_dir / t4dataset_id
        if not path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {path} and --no-download was specified."
            )
        return path


# ---------------------------------------------------------------------------
# Single-row visualization
# ---------------------------------------------------------------------------

def _status_dir(output_root: Path, status: Optional[str], has_status_column: bool) -> Path:
    """Return the output directory for a given status value.

    - If the input has no ``status`` column → images go directly in output_root.
    - If it has a status column but the value is blank/NaN → "unknown" sub-dir.
    - Otherwise → sub-dir named after the status value.
    """
    if not has_status_column:
        return output_root
    return output_root / (status if status else "unknown")


def _filename_prefix(row: SceneRow) -> str:
    """Build the flat filename prefix: <t4dataset_id>_<uuid>_<timestamp_us>."""
    # Sanitize components so they are safe as filename parts
    safe = lambda s: str(s).replace("/", "-").replace(" ", "_")
    return f"{safe(row.t4dataset_id)}_{safe(row.uuid)}_{row.timestamp_us}"


def visualize_row(
    row: SceneRow,
    dataset_path: Path,
    output_root: Path,
    show_annotations: bool = True,
    version: Optional[str] = None,
    has_status_column: bool = False,
) -> Path:
    """Visualize one scene row. Returns the output directory used.

    All files for this row are written into a single flat directory
    (``<output_root>/<status>/``) with a shared filename prefix
    ``<t4dataset_id>_<uuid>_<timestamp_us>`` so they sort together.
    """
    try:
        from t4_devkit import Tier4
    except ImportError:
        raise ImportError(
            "t4_devkit is not installed. "
            "Install with: pip install git+https://github.com/tier4/t4-devkit.git"
        )

    from t4_visualizer.visualize import (
        find_closest_sample,
        list_camera_channels,
        list_lidar_channels,
        visualize_static,
    )

    # Flat output directory: <output_root>/[<status>/]
    row_out = _status_dir(output_root, row.status, has_status_column)
    row_out.mkdir(parents=True, exist_ok=True)

    prefix = _filename_prefix(row)

    # Load dataset
    kwargs = {}
    if version:
        kwargs["version"] = version
    t4 = Tier4(str(dataset_path), **kwargs)

    # Find closest sample
    sample = find_closest_sample(t4, row.timestamp_us)
    if sample is None:
        raise RuntimeError(f"No samples found in dataset: {dataset_path}")

    delta_ms = abs(sample.timestamp - row.timestamp_us) / 1e3
    print(
        f"  [{row.uuid}] sample={sample.token} "
        f"Δ={delta_ms:.1f} ms  ts={sample.timestamp}"
    )

    # Write metadata sidecar (same prefix, .txt extension)
    meta_path = row_out / f"{prefix}_meta.txt"
    with open(meta_path, "w") as f:
        f.write(
            f"t4dataset_id : {row.t4dataset_id}\n"
            f"uuid         : {row.uuid}\n"
            f"status       : {row.status}\n"
            f"target_ts_us : {row.timestamp_us}\n"
            f"sample_token : {sample.token}\n"
            f"sample_ts_us : {sample.timestamp}\n"
            f"delta_ms     : {delta_ms:.3f}\n"
            f"description  : {row.description}\n"
            f"cameras      : {list_camera_channels(t4, sample)}\n"
            f"lidars       : {list_lidar_channels(t4, sample)}\n"
        )

    visualize_static(
        t4,
        sample,
        cameras=row.cameras,
        show_annotations=show_annotations,
        save_dir=str(row_out),
        filename_prefix=prefix,
    )

    return row_out


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

@dataclass
class BatchConfig:
    input_path: str
    output_dir: Path
    data_dir: Optional[Path]       # None = use temp
    use_temp: bool
    do_download: bool
    show_annotations: bool
    version: Optional[str]
    workers: int
    fail_fast: bool


def run_batch(cfg: BatchConfig) -> List[RowResult]:
    """Main batch pipeline. Returns list of per-row results."""
    # Load input
    print(f"Loading input: {cfg.input_path}")
    df = load_input(cfg.input_path)
    has_status_column = "status" in df.columns
    rows = df_to_rows(df)
    print(f"  {len(rows)} rows found.")
    if has_status_column:
        status_counts = df["status"].fillna("unknown").value_counts().to_dict()
        print(f"  Status groups: {status_counts}")

    # Unique dataset IDs
    unique_ids = list(dict.fromkeys(r.t4dataset_id for r in rows))
    print(f"  {len(unique_ids)} unique t4dataset_id(s): {unique_ids}")

    # Setup data directory
    _tmp_ctx = None
    if cfg.use_temp:
        _tmp_ctx = tempfile.TemporaryDirectory(prefix="t4batch_")
        data_dir = Path(_tmp_ctx.name)
        print(f"  Using TEMP directory: {data_dir}")
    else:
        data_dir = cfg.data_dir or Path("t4datasets")
        data_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Using persistent data directory: {data_dir.resolve()}")

    results: List[RowResult] = []

    try:
        # Download all required datasets first (sequential to avoid conflicts)
        dataset_paths: dict[str, Path] = {}
        for did in unique_ids:
            try:
                path = resolve_dataset_path(did, data_dir, cfg.do_download)
                dataset_paths[did] = path
                print(f"  Dataset ready: {did} → {path}")
            except Exception as e:
                print(f"  ERROR downloading {did}: {e}")
                # Mark all rows for this dataset as failed
                for row in rows:
                    if row.t4dataset_id == did:
                        results.append(RowResult(row=row, success=False, error=str(e)))
                if cfg.fail_fast:
                    raise

        # Filter rows whose dataset downloaded successfully
        ok_rows = [r for r in rows if r.t4dataset_id in dataset_paths]

        cfg.output_dir.mkdir(parents=True, exist_ok=True)

        def _process(row: SceneRow) -> RowResult:
            dataset_path = dataset_paths[row.t4dataset_id]
            try:
                out = visualize_row(
                    row,
                    dataset_path,
                    cfg.output_dir,
                    show_annotations=cfg.show_annotations,
                    version=cfg.version,
                    has_status_column=has_status_column,
                )
                return RowResult(row=row, success=True, output_dir=out)
            except Exception as e:
                tb = traceback.format_exc()
                print(f"  ERROR [{row.uuid} / {row.timestamp_us}]: {e}\n{tb}")
                return RowResult(row=row, success=False, error=str(e))

        if cfg.workers <= 1:
            for row in ok_rows:
                print(
                    f"\n--- Processing [{row.t4dataset_id}] uuid={row.uuid} "
                    f"ts={row.timestamp_us} ---"
                )
                result = _process(row)
                results.append(result)
                if not result.success and cfg.fail_fast:
                    raise RuntimeError(result.error)
        else:
            print(f"\nProcessing {len(ok_rows)} rows with {cfg.workers} parallel workers...")
            with concurrent.futures.ThreadPoolExecutor(max_workers=cfg.workers) as pool:
                futures = {pool.submit(_process, row): row for row in ok_rows}
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    results.append(result)
                    status = "OK" if result.success else "FAIL"
                    print(
                        f"  [{status}] {result.row.uuid} / {result.row.timestamp_us}"
                        + (f" → {result.output_dir}" if result.success else f": {result.error}")
                    )
                    if not result.success and cfg.fail_fast:
                        pool.shutdown(wait=False, cancel_futures=True)
                        raise RuntimeError(result.error)

    finally:
        if _tmp_ctx is not None:
            print(f"\nCleaning up temp directory: {_tmp_ctx.name}")
            _tmp_ctx.cleanup()

    return results


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def print_summary(results: List[RowResult], output_dir: Path) -> None:
    ok = [r for r in results if r.success]
    fail = [r for r in results if not r.success]

    print(f"\n{'='*60}")
    print(f"  Batch complete: {len(ok)} succeeded / {len(fail)} failed")
    print(f"  Output directory: {output_dir.resolve()}")
    print(f"{'='*60}")

    if fail:
        print("\nFailed rows:")
        for r in fail:
            print(f"  t4dataset_id={r.row.t4dataset_id}  uuid={r.row.uuid}  "
                  f"ts={r.row.timestamp_us}\n    Error: {r.error}")

    # Write CSV summary
    summary_path = output_dir / "batch_summary.csv"
    summary_df = pd.DataFrame([
        {
            "t4dataset_id": r.row.t4dataset_id,
            "uuid": r.row.uuid,
            "timestamp_us": r.row.timestamp_us,
            "status": r.row.status,
            "success": r.success,
            "output_dir": str(r.output_dir) if r.output_dir else "",
            "filename_prefix": _filename_prefix(r.row) if r.success else "",
            "error": r.error,
        }
        for r in results
    ])
    summary_df.to_csv(summary_path, index=False)
    print(f"\n  Summary CSV: {summary_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Batch T4 dataset visualization pipeline.\n"
            "Reads a CSV/Parquet with columns [t4dataset_id, uuid, timestamp] "
            "and produces camera + LiDAR plots for each row."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "input",
        help="CSV or Parquet file with columns: t4dataset_id, uuid, timestamp.",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="batch_output",
        metavar="DIR",
        help="Root directory for visualization outputs (default: ./batch_output/).",
    )

    # Storage mode (mutually exclusive)
    storage = parser.add_mutually_exclusive_group()
    storage.add_argument(
        "--temp",
        action="store_true",
        default=False,
        help=(
            "Download datasets into a temporary directory that is deleted on exit. "
            "Useful for CI or one-shot jobs where storage is limited."
        ),
    )
    storage.add_argument(
        "--data-dir",
        default=None,
        metavar="DIR",
        help=(
            "Persistent directory for downloaded datasets (default: ./t4datasets/). "
            "Already-present datasets are reused without re-downloading."
        ),
    )

    parser.add_argument(
        "--no-download",
        action="store_true",
        default=False,
        help=(
            "Skip downloading; assume datasets already exist under --data-dir. "
            "Fails if a required dataset is missing."
        ),
    )
    parser.add_argument(
        "--no-annotations",
        dest="show_annotations",
        action="store_false",
        default=True,
        help="Disable bounding box overlays on images and point clouds.",
    )
    parser.add_argument(
        "--version",
        default=None,
        metavar="VERSION",
        help="Dataset annotation version directory name (default: auto-detect).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        metavar="N",
        help="Number of parallel visualization workers (default: 1).",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        default=False,
        help="Abort the batch on the first error instead of continuing.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    cfg = BatchConfig(
        input_path=args.input,
        output_dir=Path(args.output_dir),
        data_dir=Path(args.data_dir) if args.data_dir else None,
        use_temp=args.temp,
        do_download=not args.no_download,
        show_annotations=args.show_annotations,
        version=args.version,
        workers=args.workers,
        fail_fast=args.fail_fast,
    )

    results = run_batch(cfg)
    print_summary(results, cfg.output_dir)

    n_fail = sum(1 for r in results if not r.success)
    sys.exit(1 if n_fail > 0 else 0)


if __name__ == "__main__":
    main()
