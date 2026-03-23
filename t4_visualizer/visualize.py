"""Visualize T4 dataset images and point clouds for a specific UUID and timestamp.

Usage:
    python visualize_by_uuid.py <dataset_path> <timestamp_us> [options]

Examples:
    # Visualize closest sample to a Unix timestamp (microseconds)
    python visualize_by_uuid.py /path/to/t4dataset 1609459200000000

    # Use Rerun for interactive 3D visualization
    python visualize_by_uuid.py /path/to/t4dataset 1609459200000000 --rerun

    # Save static plots to a directory
    python visualize_by_uuid.py /path/to/t4dataset 1609459200000000 --save-dir output/

    # Specify a dataset version (default: latest)
    python visualize_by_uuid.py /path/to/t4dataset 1609459200000000 --version annotation

Args:
    dataset_path: Path to the T4 dataset root directory (the UUID directory)
    timestamp_us: Target timestamp in microseconds (Unix time)
    --rerun: Use Rerun for interactive visualization instead of matplotlib
    --save-dir: Directory to save visualization outputs
    --version: Dataset version/annotation directory name (default: auto-detect)
    --cameras: Comma-separated list of camera channels to show (default: all)
    --no-annotations: Skip rendering 3D bounding boxes
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Set

import numpy as np


@dataclass
class TargetObject:
    """One target object to highlight in the visualization (from CSV row)."""
    uuid: str           # instance_token in T4 dataset
    x: float = 0.0     # detection position in ego frame (from CSV)
    y: float = 0.0
    z: float = 0.0
    label: str = ""
    width: float = 0.0   # BBOX dimensions in ego frame [m]
    length: float = 0.0
    height: float = 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _box_token(box) -> str:
    """Extract the annotation token from a Box2D or Box3D object."""
    for attr in ("token", "annotation_token"):
        val = getattr(box, attr, None)
        if val:
            return str(val)
    return ""


def _project_ego_to_cam(t4, sample_data_token: str, point_ego) -> Optional[tuple]:
    """Project a 3D point in ego (base_link) frame to camera pixel (u, v).

    Uses calibrated_sensor from the T4 dataset (no rosbag / TF needed).

    Args:
        t4: Tier4 instance.
        sample_data_token: Token of the camera SampleData.
        point_ego: (x, y, z) in ego / base_link frame.

    Returns:
        (u, v) float pixel coordinates, or None if behind the camera or
        if the calibrated sensor has no intrinsics (i.e. LiDAR).
    """
    try:
        sd = t4.get("sample_data", sample_data_token)
        cs = t4.get("calibrated_sensor", sd.calibrated_sensor_token)

        # camera_intrinsic is empty for LiDAR
        K_raw = cs.camera_intrinsic
        if K_raw is None or len(K_raw) == 0:
            return None
        K = np.array(K_raw, dtype=float)
        if K.shape != (3, 3):
            return None

        t_ego = np.array(cs.translation, dtype=float)
        p = np.array(point_ego, dtype=float)

        # Rotation: sensor → ego (stored as [w, x, y, z] in T4)
        rot = cs.rotation
        if hasattr(rot, "inverse"):
            # Already a pyquaternion Quaternion
            q = rot
        else:
            # List/array [w, x, y, z]
            from pyquaternion import Quaternion
            q = Quaternion(np.array(rot, dtype=float))

        # ego → sensor:  p_cam = q^{-1} * (p_ego - t)
        p_cam = q.inverse.rotate(p - t_ego)

        if p_cam[2] <= 0.1:          # behind (or too close to) camera
            return None

        uvw = K @ p_cam
        return (float(uvw[0] / uvw[2]), float(uvw[1] / uvw[2]))

    except Exception:
        return None


def _get_target_ann_tokens(t4, sample, target_objects: List[TargetObject]) -> Set[str]:
    """Return sample_annotation tokens for target instance_tokens in this sample."""
    if not target_objects:
        return set()
    instance_tokens = {obj.uuid for obj in target_objects}
    result = set()
    try:
        for ann in t4.sample_annotation:
            if ann.sample_token == sample.token and ann.instance_token in instance_tokens:
                result.add(ann.token)
    except Exception as e:
        print(f"  WARNING: Could not resolve annotation tokens: {e}")
    if not result:
        print(
            f"  WARNING: No annotation found for instance_tokens {instance_tokens}. "
            "Detection position markers will be drawn in BEV as fallback."
        )
    return result


def find_closest_sample(t4, timestamp_us: int):
    """Return the Sample record whose timestamp is closest to *timestamp_us*.

    Args:
        t4: Tier4 instance already loaded with the dataset.
        timestamp_us: Target timestamp in **microseconds**.

    Returns:
        The closest Sample object, or None if no samples exist.
    """
    samples = t4.sample  # list[Sample]
    if not samples:
        return None
    best = min(samples, key=lambda s: abs(s.timestamp - timestamp_us))
    return best


def find_sample_by_scene_and_index(t4, scene_name: str, frame_index: int):
    """Return the frame_index-th sample in the named scene.

    Walks the linked list of samples starting from scene.first_sample_token.

    Args:
        t4: Tier4 instance already loaded with the dataset.
        scene_name: Value of the ``name`` field in the scene table.
        frame_index: 0-based index of the frame within the scene.

    Returns:
        The Sample object at position *frame_index*.

    Raises:
        ValueError: If no scene with *scene_name* is found.
        IndexError: If *frame_index* exceeds the number of samples in the scene.
    """
    scene = next((s for s in t4.scene if s.name == scene_name), None)
    if scene is None:
        available = t4.scene
        if len(available) == 1:
            print(
                f"  WARNING: Scene '{scene_name}' not found; "
                f"falling back to the only scene in dataset: '{available[0].name}'"
            )
            scene = available[0]
        else:
            raise ValueError(
                f"Scene '{scene_name}' not found in dataset.\n"
                f"Available scenes: {[s.name for s in available]}"
            )

    token = scene.first_sample_token
    for i in range(frame_index):
        sample = t4.get("sample", token)
        if not sample.next:
            raise IndexError(
                f"frame_index {frame_index} is out of range for scene '{scene_name}' "
                f"(scene contains only {i + 1} sample(s))."
            )
        token = sample.next
    return t4.get("sample", token)


def list_camera_channels(t4, sample) -> List[str]:
    """Return camera channel names available for the given sample."""
    channels = []
    for channel, token in sample.data.items():
        sd = t4.get("sample_data", token)
        if sd.fileformat.value in ("jpg", "png"):
            channels.append(channel)
    return sorted(channels)


def list_lidar_channels(t4, sample) -> List[str]:
    """Return LiDAR channel names available for the given sample."""
    channels = []
    for channel, token in sample.data.items():
        sd = t4.get("sample_data", token)
        if sd.fileformat.value in ("pcd", "bin", "pcd.bin"):
            channels.append(channel)
    return sorted(channels)


# ---------------------------------------------------------------------------
# Static matplotlib visualization
# ---------------------------------------------------------------------------

def visualize_static(
    t4,
    sample,
    cameras: Optional[List[str]] = None,
    show_annotations: bool = True,
    save_dir: Optional[str] = None,
    filename_prefix: Optional[str] = None,
    target_objects: Optional[List[TargetObject]] = None,
) -> None:
    """Render images and a bird's-eye-view point cloud with matplotlib.

    Args:
        t4: Tier4 instance.
        sample: Target Sample record.
        cameras: Camera channels to display (None = all cameras).
        show_annotations: Overlay 2D/3D bounding boxes on images.
        save_dir: If given, save figures here instead of showing interactively.
        filename_prefix: Prefix for saved filenames (default: timestamp).
            E.g. "datasetA_sceneX_1609459200000000" produces
            "datasetA_sceneX_1609459200000000_cameras.png".
    """
    import matplotlib
    if save_dir:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from PIL import Image  # Pillow

    ts_us = sample.timestamp
    prefix = filename_prefix if filename_prefix else str(ts_us)
    print(f"[visualize_static] Rendering sample token={sample.token}, timestamp={ts_us} us")

    available_cameras = list_camera_channels(t4, sample)
    selected_cameras = cameras if cameras else available_cameras
    selected_cameras = [c for c in selected_cameras if c in available_cameras]

    target_ann_tokens = _get_target_ann_tokens(t4, sample, target_objects or [])

    if not selected_cameras:
        print("  WARNING: No matching camera channels found in this sample.")
    else:
        _plot_images(
            t4, sample, selected_cameras, show_annotations, save_dir, prefix,
            target_ann_tokens=target_ann_tokens,
            target_objects=target_objects,
        )

    lidar_channels = list_lidar_channels(t4, sample)
    if lidar_channels:
        _plot_bev_pointcloud(
            t4, sample, lidar_channels[0], show_annotations, save_dir, prefix,
            target_ann_tokens=target_ann_tokens,
            target_objects=target_objects or [],
        )
    else:
        print("  WARNING: No LiDAR data found in this sample.")

    if not save_dir:
        plt.show()


def _plot_images(t4, sample, camera_channels, show_annotations, save_dir, filename_prefix=None, target_ann_tokens=None, target_objects=None):
    """Create a figure with one subplot per camera."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from PIL import Image

    n = len(camera_channels)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 4))
    axes = np.array(axes).flatten() if n > 1 else [axes]

    for idx, channel in enumerate(camera_channels):
        ax = axes[idx]
        token = sample.data.get(channel)
        if token is None:
            ax.set_visible(False)
            continue

        # Get data path + 2D boxes
        if show_annotations:
            data_path, boxes_2d, cam_intrinsic = t4.get_sample_data(
                token, as_3d=False, as_sensor_coord=True
            )
        else:
            data_path, _, _ = t4.get_sample_data(token, as_3d=False)
            boxes_2d = []

        img = Image.open(data_path)
        ax.imshow(img)
        ax.set_title(channel, fontsize=9)
        ax.axis("off")

        for box in boxes_2d:
            # box is a Box2D with .roi attribute (l, t, r, b) or similar
            try:
                roi = box.roi  # (left, top, right, bottom)
                bx, by, w, h = roi[0], roi[1], roi[2] - roi[0], roi[3] - roi[1]
                is_target = bool(target_ann_tokens) and _box_token(box) in target_ann_tokens
                color = "red" if is_target else "lime"
                lw = 2.5 if is_target else 1.0
                rect = patches.Rectangle(
                    (bx, by), w, h,
                    linewidth=lw, edgecolor=color, facecolor="none",
                    zorder=3 if is_target else 2,
                )
                ax.add_patch(rect)
                if is_target:
                    label_text = getattr(box, "label", None) or ""
                    ax.text(
                        bx, by - 3, f"[TARGET] {label_text}",
                        color="red", fontsize=7, fontweight="bold",
                        va="bottom", zorder=4,
                    )
            except Exception:
                pass

        # Project detection positions (ego frame) onto this camera
        if target_objects:
            img_w, img_h = img.size
            for obj in target_objects:
                uv = _project_ego_to_cam(t4, token, [obj.x, obj.y, obj.z])
                if uv is None:
                    continue
                u, v = uv
                if not (0 <= u < img_w and 0 <= v < img_h):
                    continue
                # Cross + circle marker for detection position
                ax.plot(u, v, "+", color="yellow", markersize=18,
                        markeredgewidth=2.5, zorder=7)
                circle = patches.Circle(
                    (u, v), radius=12,
                    fill=False, edgecolor="yellow", linewidth=2.0, zorder=7,
                )
                ax.add_patch(circle)
                ax.text(
                    u + 14, v, f"det:{obj.label or '?'}",
                    color="yellow", fontsize=7, fontweight="bold",
                    va="center", zorder=7,
                )

    # Hide unused subplots
    for i in range(len(camera_channels), len(axes)):
        axes[i].set_visible(False)

    ts_us = sample.timestamp
    fig.suptitle(
        f"Cameras — timestamp: {ts_us} µs  ({ts_us / 1e6:.3f} s)\ntoken: {sample.token}",
        fontsize=10,
    )
    fig.tight_layout()

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        prefix = filename_prefix if filename_prefix else str(ts_us)
        out = Path(save_dir) / f"{prefix}_cameras.png"
        fig.savefig(out, dpi=150)
        print(f"  Saved: {out}")
    plt.close(fig)


def _plot_bev_pointcloud(t4, sample, lidar_channel, show_annotations, save_dir, filename_prefix=None, target_ann_tokens=None, target_objects=None):
    """Bird's-eye-view scatter plot of the LiDAR point cloud."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    token = sample.data.get(lidar_channel)
    if token is None:
        print(f"  WARNING: Channel {lidar_channel} not in sample.data")
        return

    if show_annotations:
        data_path, boxes_3d, _ = t4.get_sample_data(
            token, as_3d=True, as_sensor_coord=True
        )
    else:
        data_path, _, _ = t4.get_sample_data(token, as_3d=True)
        boxes_3d = []

    points = _load_pointcloud(data_path)
    if points is None or points.shape[0] == 0:
        print(f"  WARNING: Could not load point cloud from {data_path}")
        return

    x_all, y_all, z_all = points[:, 0], points[:, 1], points[:, 2]
    intensity_all = points[:, 3] if points.shape[1] > 3 else np.ones(len(x_all))

    # Crop to ±10 m around the first target object when available
    CROP_RADIUS = 10.0
    crop_center = None
    if target_objects:
        cx, cy = target_objects[0].x, target_objects[0].y
        crop_center = (cx, cy)
        mask = (np.abs(x_all - cx) <= CROP_RADIUS) & (np.abs(y_all - cy) <= CROP_RADIUS)
        x, y, z = x_all[mask], y_all[mask], z_all[mask]
        intensity = intensity_all[mask]
    else:
        x, y, z, intensity = x_all, y_all, z_all, intensity_all

    # Down-sample for speed
    max_pts = 80_000
    if len(x) > max_pts:
        idx = np.random.choice(len(x), max_pts, replace=False)
        x, y, z, intensity = x[idx], y[idx], z[idx], intensity[idx]

    fig, ax = plt.subplots(figsize=(10, 10))
    sc = ax.scatter(x, y, c=intensity, s=0.5, cmap="viridis", vmin=0, vmax=255)
    plt.colorbar(sc, ax=ax, label="Intensity")

    # Draw 3D box footprints (BEV)
    for box in boxes_3d:
        try:
            is_target = bool(target_ann_tokens) and _box_token(box) in target_ann_tokens
            _draw_box_bev(ax, box, highlight=is_target)
        except Exception:
            pass

    # Draw detection BBOX and position marker for each target object
    if target_objects:
        for obj in target_objects:
            # BBOX rectangle (axis-aligned, using width × length footprint)
            if obj.width > 0 and obj.length > 0:
                half_w, half_l = obj.width / 2.0, obj.length / 2.0
                rect = patches.Rectangle(
                    (obj.x - half_w, obj.y - half_l),
                    obj.width, obj.length,
                    linewidth=2.0, edgecolor="yellow", facecolor="none",
                    linestyle="--", zorder=6,
                )
                ax.add_patch(rect)
            # Center marker
            ax.plot(
                obj.x, obj.y, "*",
                color="yellow", markersize=16,
                markeredgecolor="black", markeredgewidth=0.8,
                zorder=7,
            )
            ax.annotate(
                f"{obj.label or 'target'}\n({obj.x:.1f},{obj.y:.1f})",
                (obj.x, obj.y),
                xytext=(6, 6), textcoords="offset points",
                color="yellow", fontsize=7, fontweight="bold",
                zorder=7,
            )

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_aspect("equal")
    ax.set_title(f"LiDAR BEV — {lidar_channel}")

    # Fix axis limits to crop region
    if crop_center is not None:
        cx, cy = crop_center
        ax.set_xlim(cx - CROP_RADIUS, cx + CROP_RADIUS)
        ax.set_ylim(cy - CROP_RADIUS, cy + CROP_RADIUS)

    ts_us = sample.timestamp
    fig.suptitle(
        f"Point Cloud — timestamp: {ts_us} µs  ({ts_us / 1e6:.3f} s)\ntoken: {sample.token}",
        fontsize=10,
    )
    fig.tight_layout()

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        prefix = filename_prefix if filename_prefix else str(ts_us)
        out = Path(save_dir) / f"{prefix}_pointcloud.png"
        fig.savefig(out, dpi=150)
        print(f"  Saved: {out}")
    plt.close(fig)


def _draw_box_bev(ax, box, highlight: bool = False):
    """Draw the BEV footprint of a Box3D on a matplotlib Axes."""
    import matplotlib.patches as patches

    color = "yellow" if highlight else "cyan"
    lw = 2.5 if highlight else 1.0
    arrow_color = "red" if highlight else "orange"
    zorder = 4 if highlight else 2

    # box.center: [x, y, z], box.size: [w, l, h], box.rotation: Quaternion
    center = box.center
    size = box.size  # (w, l, h) or (l, w, h) depending on convention

    # Get 4 corners in BEV using the box's rotation
    try:
        # t4-devkit Box3D corners: shape (3, 8) — top 4 + bottom 4
        corners = box.corners  # (3, 8)
        # Use only the bottom 4 corners (indices 4-7) for BEV footprint
        bev = corners[:2, 4:]  # (2, 4) — x and y of 4 bottom corners
        xs = np.append(bev[0], bev[0, 0])
        ys = np.append(bev[1], bev[1, 0])
        ax.plot(xs, ys, color=color, linewidth=lw, zorder=zorder)
        # Draw heading arrow from center to front midpoint
        front_mid = (bev[:, 0] + bev[:, 1]) / 2
        ax.annotate(
            "",
            xy=(front_mid[0], front_mid[1]),
            xytext=(center[0], center[1]),
            arrowprops=dict(arrowstyle="->", color=arrow_color, lw=lw),
            zorder=zorder,
        )
        if highlight:
            ax.text(
                center[0], center[1], "[T]",
                color="yellow", fontsize=7, fontweight="bold",
                ha="center", va="center", zorder=5,
            )
    except AttributeError:
        # Fallback: draw a simple rectangle from center + size
        w, l = float(size[0]), float(size[1])
        rect = patches.Rectangle(
            (center[0] - w / 2, center[1] - l / 2), w, l,
            linewidth=lw, edgecolor=color, facecolor="none", zorder=zorder,
        )
        ax.add_patch(rect)


def _load_pointcloud(data_path: str) -> Optional[np.ndarray]:
    """Load a point cloud file and return Nx4 (x,y,z,intensity) array."""
    path = Path(data_path)
    if not path.exists():
        print(f"  WARNING: Point cloud file not found: {data_path}")
        return None

    suffix = path.suffix.lower()
    if suffix == ".bin" or data_path.endswith(".pcd.bin"):
        # Binary float32 format: x, y, z, intensity[, ring, time, ...]
        raw = np.fromfile(data_path, dtype=np.float32)
        for ncols in (4, 5, 6, 3):
            if raw.size % ncols == 0:
                pts = raw.reshape(-1, ncols)
                # Always return Nx4 (pad intensity with zeros if only 3 cols)
                if ncols >= 4:
                    return pts[:, :4]
                else:
                    return np.column_stack([pts, np.zeros(len(pts), dtype=np.float32)])
        print(f"  WARNING: Cannot determine point cloud stride for size {raw.size}: {data_path}")
        return None
    elif suffix == ".pcd":
        return _load_pcd(data_path)
    else:
        print(f"  WARNING: Unrecognized point cloud format: {data_path}")
        return None


def _load_pcd(filepath: str) -> Optional[np.ndarray]:
    """Minimal PCD (ASCII / binary) loader returning Nx4 array."""
    import struct

    header = {}
    fields = []
    size = []
    type_ = []
    count = []
    data_type = "ascii"
    header_end_byte = 0

    with open(filepath, "rb") as f:
        raw = f.read()

    lines = raw.split(b"\n")
    header_lines = []
    data_start_line = 0
    for i, line in enumerate(lines):
        decoded = line.decode("utf-8", errors="replace").strip()
        header_lines.append(decoded)
        if decoded.startswith("FIELDS"):
            fields = decoded.split()[1:]
        elif decoded.startswith("SIZE"):
            size = [int(v) for v in decoded.split()[1:]]
        elif decoded.startswith("TYPE"):
            type_ = decoded.split()[1:]
        elif decoded.startswith("COUNT"):
            count = [int(v) for v in decoded.split()[1:]]
        elif decoded.startswith("DATA"):
            data_type = decoded.split()[1]
            data_start_line = i + 1
            break

    # Byte offset to data
    offset = sum(len(l) + 1 for l in lines[:data_start_line])

    try:
        xi = fields.index("x")
        yi = fields.index("y")
        zi = fields.index("z")
        ii = fields.index("intensity") if "intensity" in fields else None
    except ValueError:
        return None

    if data_type == "ascii":
        pts_text = b"\n".join(lines[data_start_line:])
        arr = np.frombuffer(pts_text, dtype=np.uint8)
        # Parse float from text
        from io import StringIO
        pts = np.loadtxt(StringIO(pts_text.decode("utf-8", errors="replace")))
        if pts.ndim == 1:
            pts = pts[np.newaxis, :]
        x = pts[:, xi]
        y = pts[:, yi]
        z = pts[:, zi]
        intensity = pts[:, ii] if ii is not None else np.zeros(len(x))

    elif data_type == "binary":
        row_size = sum(s * c for s, c in zip(size, count))
        data_bytes = raw[offset:]
        n_pts = len(data_bytes) // row_size
        pts_bytes = np.frombuffer(data_bytes[: n_pts * row_size], dtype=np.uint8)

        # Build structured dtype
        dt_map = {"F": "f", "I": "i", "U": "u"}
        dtype_fields = []
        for f_name, s, t, c in zip(fields, size, type_, count):
            np_type = f"{dt_map.get(t, 'f')}{s}"
            if c == 1:
                dtype_fields.append((f_name, np_type))
            else:
                dtype_fields.append((f_name, np_type, (c,)))
        dt = np.dtype(dtype_fields)
        structured = np.frombuffer(data_bytes[: n_pts * row_size], dtype=dt)

        x = structured["x"].astype(np.float32)
        y = structured["y"].astype(np.float32)
        z = structured["z"].astype(np.float32)
        intensity = (
            structured["intensity"].astype(np.float32)
            if ii is not None
            else np.zeros(len(x), dtype=np.float32)
        )
    else:
        print(f"  WARNING: PCD data type '{data_type}' not fully supported.")
        return None

    return np.column_stack([x, y, z, intensity])


# ---------------------------------------------------------------------------
# Rerun visualization (interactive)
# ---------------------------------------------------------------------------

def visualize_rerun(
    t4,
    sample,
    cameras: Optional[List[str]] = None,
    show_annotations: bool = True,
    save_dir: Optional[str] = None,
) -> None:
    """Use t4-devkit's built-in Rerun integration for interactive visualization.

    This renders the full scene via Rerun and then jumps to the closest
    timestamp. For Rerun, the native t4.render_scene() is the easiest path.

    Args:
        t4: Tier4 instance.
        sample: Closest Sample to the target timestamp.
        cameras: (unused — Rerun renders all cameras)
        show_annotations: (unused — Rerun always shows annotations)
        save_dir: Directory to save Rerun recording (.rrd) file.
    """
    ts_us = sample.timestamp
    ts_sec = ts_us / 1e6
    print(f"[visualize_rerun] Launching Rerun for timestamp {ts_us} µs ({ts_sec:.3f} s)")
    print(f"  Sample token: {sample.token}")
    print("  NOTE: Rerun will render the full scene; use the timeline to jump to the sample.")

    t4.render_scene(save_dir=save_dir)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize T4 dataset images and point clouds at a specific timestamp.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "dataset_path",
        help="Path to T4 dataset root directory (the UUID/scene directory).",
    )
    parser.add_argument(
        "timestamp_us",
        type=float,
        help=(
            "Target timestamp in microseconds (Unix time). "
            "The closest sample will be selected. "
            "Tip: use 0 to visualize the first sample."
        ),
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        default=False,
        help="Use Rerun for interactive 3D visualization (requires t4-devkit with Rerun).",
    )
    parser.add_argument(
        "--save-dir",
        default=None,
        metavar="DIR",
        help="Save visualization outputs to this directory instead of displaying interactively.",
    )
    parser.add_argument(
        "--version",
        default=None,
        metavar="VERSION",
        help="Dataset annotation version directory name (default: auto-detect latest).",
    )
    parser.add_argument(
        "--cameras",
        default=None,
        metavar="CAM1,CAM2",
        help="Comma-separated camera channel names to display (default: all cameras).",
    )
    parser.add_argument(
        "--no-annotations",
        dest="show_annotations",
        action="store_false",
        default=True,
        help="Disable bounding box overlays.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose output from Tier4 loader.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # Import t4-devkit (must be installed)
    # ------------------------------------------------------------------
    try:
        from t4_devkit import Tier4
    except ImportError:
        print(
            "ERROR: t4_devkit is not installed.\n"
            "Install it with:\n"
            "  pip install git+https://github.com/tier4/t4-devkit.git\n"
            "or clone and install locally:\n"
            "  git clone https://github.com/tier4/t4-devkit && cd t4-devkit && pip install -e ."
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    dataset_path = Path(args.dataset_path).resolve()
    if not dataset_path.exists():
        print(f"ERROR: Dataset path does not exist: {dataset_path}")
        sys.exit(1)

    print(f"Loading T4 dataset from: {dataset_path}")
    kwargs = {"verbose": args.verbose}
    if args.version:
        kwargs["version"] = args.version

    try:
        t4 = Tier4(str(dataset_path), **kwargs)
    except Exception as e:
        print(f"ERROR: Failed to load dataset: {e}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Find closest sample
    # ------------------------------------------------------------------
    timestamp_us = int(args.timestamp_us)
    print(f"Target timestamp: {timestamp_us} µs ({timestamp_us / 1e6:.6f} s)")

    sample = find_closest_sample(t4, timestamp_us)
    if sample is None:
        print("ERROR: No samples found in the dataset.")
        sys.exit(1)

    delta_ms = abs(sample.timestamp - timestamp_us) / 1e3
    print(
        f"Closest sample: token={sample.token}\n"
        f"  sample timestamp: {sample.timestamp} µs\n"
        f"  time delta: {delta_ms:.1f} ms"
    )

    # ------------------------------------------------------------------
    # Parse camera list
    # ------------------------------------------------------------------
    cameras = None
    if args.cameras:
        cameras = [c.strip() for c in args.cameras.split(",") if c.strip()]

    # ------------------------------------------------------------------
    # Print available channels for this sample
    # ------------------------------------------------------------------
    all_cameras = list_camera_channels(t4, sample)
    all_lidars = list_lidar_channels(t4, sample)
    print(f"Available cameras : {all_cameras}")
    print(f"Available LiDAR   : {all_lidars}")

    # ------------------------------------------------------------------
    # Visualize
    # ------------------------------------------------------------------
    if args.rerun:
        visualize_rerun(
            t4,
            sample,
            cameras=cameras,
            show_annotations=args.show_annotations,
            save_dir=args.save_dir,
        )
    else:
        visualize_static(
            t4,
            sample,
            cameras=cameras,
            show_annotations=args.show_annotations,
            save_dir=args.save_dir,
        )


if __name__ == "__main__":
    main()
