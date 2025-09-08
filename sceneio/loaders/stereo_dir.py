from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import rerun as rr

from ..api import SceneIO


@dataclass
class CameraCsv:
    t_ns: List[int]
    frame_idx: List[int]


def _require_file(path: str) -> str:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return path


def _flatten_row_major_3x3(mat: List[List[float]]) -> List[float]:
    return [
        float(mat[0][0]), float(mat[0][1]), float(mat[0][2]),
        float(mat[1][0]), float(mat[1][1]), float(mat[1][2]),
        float(mat[2][0]), float(mat[2][1]), float(mat[2][2]),
    ]


def _read_calibration(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _read_ts_idx_csv(path: str) -> CameraCsv:
    df = pd.read_csv(path)
    if "ts_ns" not in df.columns or "frame_idx" not in df.columns:
        raise ValueError(f"CSV {path} must contain columns 'ts_ns' and 'frame_idx'")
    t_ns = df["ts_ns"].astype("int64").tolist()
    frame_idx = df["frame_idx"].astype("int64").tolist()
    if len(t_ns) != len(frame_idx):
        raise ValueError(f"CSV {path} has mismatched lengths")
    return CameraCsv(t_ns=t_ns, frame_idx=frame_idx)


def _read_poses_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = ["frame_idx", "tx", "ty", "tz", "qx", "qy", "qz", "qw"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"CSV {path} missing required column '{col}'")
    return df


def _quat_xyzw_to_rotation_matrix(qx: float, qy: float, qz: float, qw: float) -> List[float]:
    # Normalize to be safe
    norm = float(np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw))
    if norm == 0.0:
        # Identity
        return [1.0,0.0,0.0, 0.0,1.0,0.0, 0.0,0.0,1.0]
    x = qx / norm
    y = qy / norm
    z = qz / norm
    w = qw / norm
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    # Row-major 3x3
    return [
        1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz),       2.0 * (xz + wy),
        2.0 * (xy + wz),       1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx),
        2.0 * (xz - wy),       2.0 * (yz + wx),       1.0 - 2.0 * (xx + yy),
    ]


def load_stereo_from_directory(
    directory: str,
    *,
    left_id: str = "left",
    right_id: str = "right",
    root: str = "/scene",
    camera_xyz: str = "RDF",
    store_frame_index_in_video_ts: bool = True,
) -> SceneIO:
    """
    Build a SceneIO from a stereo directory containing:
      - calibration.json
      - left.csv, right.csv (columns: ts_ns,frame_idx)
      - left.mp4, right.mp4

    The CSV maps global timestamps to per-video frame indices. By default, we
    store the frame index in the 'video_ts_ns' field to preserve this mapping.
    """
    directory = os.path.abspath(directory)

    calib_path = _require_file(os.path.join(directory, "calibration.json"))
    left_csv_path = _require_file(os.path.join(directory, "left.csv"))
    right_csv_path = _require_file(os.path.join(directory, "right.csv"))
    left_mp4_path = _require_file(os.path.join(directory, "left.mp4"))
    right_mp4_path = _require_file(os.path.join(directory, "right.mp4"))

    calib = _read_calibration(calib_path)

    # Intrinsics
    K_left = _flatten_row_major_3x3(calib["left"]["intrinsics"])  # 3x3
    K_right = _flatten_row_major_3x3(calib["right"]["intrinsics"])  # 3x3
    width_left = int(calib["left"]["width"]) 
    height_left = int(calib["left"]["height"]) 
    width_right = int(calib["right"]["width"]) 
    height_right = int(calib["right"]["height"]) 

    # Build scene and cameras (mono loads)
    scene = SceneIO(root=root)
    scene.load_mono_camera(
        cam_id=left_id,
        video_path=left_mp4_path,
        K=K_left,
        width=width_left,
        height=height_left,
        camera_xyz=camera_xyz,
        label=calib.get("left", {}).get("socket"),
    )
    scene.load_mono_camera(
        cam_id=right_id,
        video_path=right_mp4_path,
        K=K_right,
        width=width_right,
        height=height_right,
        camera_xyz=camera_xyz,
        label=calib.get("right", {}).get("socket"),
    )

    # Frame mappings
    left_csv = _read_ts_idx_csv(left_csv_path)
    right_csv = _read_ts_idx_csv(right_csv_path)

    # Derive media PTS (ns) via rerun AssetVideo if available; fall back to index
    try:
        left_pts = rr.AssetVideo(path=left_mp4_path).read_frame_timestamps_nanos()
        right_pts = rr.AssetVideo(path=right_mp4_path).read_frame_timestamps_nanos()
    except Exception:
        left_pts, right_pts = [], []

    def _map_idx_to_pts(idx_list: List[int], pts) -> np.ndarray:
        # Accept list or numpy array; empty -> fallback
        try:
            pts_arr = np.asarray(pts, dtype=np.int64)
        except Exception:
            pts_arr = np.empty((0,), dtype=np.int64)
        if pts_arr.size == 0:
            # fallback: index as monotonic surrogate
            return np.array(idx_list if store_frame_index_in_video_ts else list(range(len(idx_list))), dtype=np.int64)
        base = int(pts_arr[0])
        out = []
        n = int(pts_arr.shape[0])
        for fi in idx_list:
            if fi < 0 or fi >= n:
                out.append(0)
            else:
                out.append(int(pts_arr[fi]) - base)
        return np.array(out, dtype=np.int64)

    left_video_ts_ns = _map_idx_to_pts(left_csv.frame_idx, left_pts)
    right_video_ts_ns = _map_idx_to_pts(right_csv.frame_idx, right_pts)

    scene.add_video_frames(
        cam_id=left_id,
        t_ns=np.array(left_csv.t_ns, dtype=np.int64),
        video_ts_ns=left_video_ts_ns,
    )
    scene.add_video_frames(
        cam_id=right_id,
        t_ns=np.array(right_csv.t_ns, dtype=np.int64),
        video_ts_ns=right_video_ts_ns,
    )

    # Optional poses as extrinsics
    left_poses_path = os.path.join(directory, "left_poses.csv")
    right_poses_path = os.path.join(directory, "right_poses.csv")

    # Build frame_idx -> t_ns maps for fast lookup
    left_idx_to_t: Dict[int, int] = {int(fi): int(t) for fi, t in zip(left_csv.frame_idx, left_csv.t_ns)}
    right_idx_to_t: Dict[int, int] = {int(fi): int(t) for fi, t in zip(right_csv.frame_idx, right_csv.t_ns)}

    if os.path.isfile(left_poses_path):
        df = _read_poses_csv(left_poses_path)
        t_list: List[int] = []
        translations: List[List[float]] = []
        rotations: List[List[float]] = []
        for _, row in df.iterrows():
            fi = int(row["frame_idx"]) 
            if fi not in left_idx_to_t:
                continue
            t_list.append(int(left_idx_to_t[fi]))
            translations.append([float(row["tx"]), float(row["ty"]), float(row["tz"])])
            rotations.append(_quat_xyzw_to_rotation_matrix(float(row["qx"]), float(row["qy"]), float(row["qz"]), float(row["qw"])) )
        if t_list:
            scene.add_extrinsics(
                cam_id=left_id,
                t_ns=np.array(t_list, dtype=np.int64),
                translation=translations,
                rotation_mat=rotations,
            )

    if os.path.isfile(right_poses_path):
        df = _read_poses_csv(right_poses_path)
        t_list: List[int] = []
        translations: List[List[float]] = []
        rotations: List[List[float]] = []
        for _, row in df.iterrows():
            fi = int(row["frame_idx"]) 
            if fi not in right_idx_to_t:
                continue
            t_list.append(int(right_idx_to_t[fi]))
            translations.append([float(row["tx"]), float(row["ty"]), float(row["tz"])])
            rotations.append(_quat_xyzw_to_rotation_matrix(float(row["qx"]), float(row["qy"]), float(row["qz"]), float(row["qw"])) )
        if t_list:
            scene.add_extrinsics(
                cam_id=right_id,
                t_ns=np.array(t_list, dtype=np.int64),
                translation=translations,
                rotation_mat=rotations,
            )

    return scene


def _cli() -> None:
    ap = argparse.ArgumentParser(description="Load stereo rig from directory into a .rrd")
    ap.add_argument("directory", help="Path containing calibration.json, left/right csv & mp4")
    ap.add_argument("--out", default="out.rrd", help="Output .rrd path")
    ap.add_argument("--root", default="/scene", help="Root entity path")
    ap.add_argument("--left-id", default="left", help="Left camera id")
    ap.add_argument("--right-id", default="right", help="Right camera id")
    ap.add_argument(
        "--store-frame-index-in-video-ts",
        action="store_true",
        help="Preserve CSV frame_idx by storing it in video_ts_ns",
    )
    args = ap.parse_args()

    scene = load_stereo_from_directory(
        args.directory,
        left_id=args.left_id,
        right_id=args.right_id,
        root=args.root,
        store_frame_index_in_video_ts=bool(args.store_frame_index_in_video_ts),
    )
    scene.save(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    _cli()


