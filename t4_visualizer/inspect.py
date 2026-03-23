"""Inspect a T4 dataset: print available scenes, samples, sensors, and timestamp range.

Usage:
    python inspect_dataset.py <dataset_path> [--version VERSION]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Inspect a T4 dataset: list scenes, sample timestamps, and sensor channels."
    )
    parser.add_argument("dataset_path", help="Path to T4 dataset root directory.")
    parser.add_argument(
        "--version",
        default=None,
        help="Annotation version directory name (default: auto-detect).",
    )
    args = parser.parse_args()

    try:
        from t4_devkit import Tier4
    except ImportError:
        print("ERROR: t4_devkit is not installed. See visualize_by_uuid.py for install instructions.")
        sys.exit(1)

    dataset_path = Path(args.dataset_path).resolve()
    kwargs = {}
    if args.version:
        kwargs["version"] = args.version

    print(f"Loading: {dataset_path}")
    t4 = Tier4(str(dataset_path), **kwargs)

    # ------------------------------------------------------------------
    # Scenes
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  Scenes ({len(t4.scene)})")
    print(f"{'='*60}")
    for scene in t4.scene:
        print(f"  name        : {scene.name}")
        print(f"  token       : {scene.token}")
        print(f"  description : {scene.description}")
        print(f"  nbr_samples : {scene.nbr_samples}")
        print()

    # ------------------------------------------------------------------
    # Samples — timestamp range
    # ------------------------------------------------------------------
    samples = t4.sample
    print(f"{'='*60}")
    print(f"  Samples ({len(samples)})")
    print(f"{'='*60}")
    if samples:
        ts_sorted = sorted(s.timestamp for s in samples)
        print(f"  Timestamp range (µs)  : {ts_sorted[0]}  →  {ts_sorted[-1]}")
        print(f"  Timestamp range  (s)  : {ts_sorted[0]/1e6:.3f}  →  {ts_sorted[-1]/1e6:.3f}")
        print(f"  Duration              : {(ts_sorted[-1]-ts_sorted[0])/1e6:.3f} s")
        print()
        print("  First 5 sample tokens & timestamps:")
        for s in sorted(samples, key=lambda x: x.timestamp)[:5]:
            print(f"    {s.token}  ts={s.timestamp} µs")

    # ------------------------------------------------------------------
    # Sensor channels
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("  Sensor Channels (from first sample)")
    print(f"{'='*60}")
    if samples:
        first = min(samples, key=lambda s: s.timestamp)
        for channel, token in sorted(first.data.items()):
            sd = t4.get("sample_data", token)
            print(f"  {channel:40s} format={sd.fileformat.value}")

    print(f"\n{'='*60}")
    print("  Usage Example")
    print(f"{'='*60}")
    if samples:
        ts_example = min(samples, key=lambda s: s.timestamp).timestamp
        print(f"  python visualize_by_uuid.py {dataset_path} {ts_example}")
        print(f"  python visualize_by_uuid.py {dataset_path} {ts_example} --save-dir output/")
        print(f"  python visualize_by_uuid.py {dataset_path} {ts_example} --rerun")


if __name__ == "__main__":
    main()
