# api.py
# Domain layer: manages entity paths & columnar state. No direct rerun imports.

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pyarrow as pa

from .models.tables import (
    SceneTables,
    CAMERAS_SCHEMA,
    PINHOLE_SCHEMA,
    VIDEO_ASSETS_SCHEMA,
    VIDEO_FRAMES_SCHEMA,
    EXTRINSICS_SCHEMA,
    new_cameras_table,
    new_pinhole_table,
    new_video_assets_table,
    new_video_frames_table,
    new_extrinsics_table,
    append_rows,
    ensure_schema,
)
from . import rrd_io as rrd_io


# --------------------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------------------

class EntityLayout(Enum):
    COLOCATED = 1      # pinhole, frames, extrinsics live on the same entity_path
    SPLIT_IMAGE_NODE = 2  # optional future mode


# --------------------------------------------------------------------------------------
# Camera handle
# --------------------------------------------------------------------------------------

@dataclass
class Camera:
    """Lightweight handle bound to a SceneIO. Mutates the scene tables."""
    scene: "SceneIO"
    cam_id: str
    entity_path: str
    label: Optional[str] = None

    # Convenience: add extrinsics rows for this camera.
    def add_extrinsics(
        self,
        t_ns: Union[Sequence[int], np.ndarray],
        *,
        translation: Optional[Sequence[Sequence[float]]] = None,
        quaternion_xyzw: Optional[Sequence[Sequence[float]]] = None,
        rotation_mat: Optional[Sequence[Sequence[float]]] = None,
        relation: Optional[Sequence[str]] = None,
    ) -> None:
        self.scene.add_extrinsics(
            self.cam_id,
            t_ns,
            translation=translation,
            quaternion_xyzw=quaternion_xyzw,
            rotation_mat=rotation_mat,
            relation=relation,
        )

    # Convenience: add/replace video for this camera.
    def set_video(
        self,
        video_path: str,
        *,
        media_type: Optional[str] = "video/mp4",
        fps_hint: Optional[float] = None,
        duration_ns: Optional[int] = None,
    ) -> None:
        self.scene.set_video(self.cam_id, video_path, media_type=media_type, fps_hint=fps_hint, duration_ns=duration_ns)

    # Convenience: add frame references for this camera.
    def add_video_frames(
        self,
        t_ns: Union[Sequence[int], np.ndarray],
        video_ts_ns: Union[Sequence[int], np.ndarray],
        *,
        source_video_entity_path: Optional[str] = None,
    ) -> None:
        self.scene.add_video_frames(self.cam_id, t_ns, video_ts_ns, source_video_entity_path=source_video_entity_path)


# --------------------------------------------------------------------------------------
# SceneIO
# --------------------------------------------------------------------------------------

class SceneIO:
    """
    Columnar scene state with timestamp-only policy.
    Public API normalizes inputs to pyarrow Tables defined in tables.py.
    rrd_io handles serialization to/from .rrd.
    """

    def __init__(
        self,
        rrd_path: Optional[str] = None,
        *,
        root: str = "/scene",
        layout: EntityLayout = EntityLayout.COLOCATED,
        entity_paths: Optional[Sequence[str]] = None,  # optional explicit set to load
    ):
        self.root = root.rstrip("/")
        self.layout = layout

        # Tables
        self.cameras: pa.Table = new_cameras_table()
        self.pinhole: pa.Table = new_pinhole_table()
        self.video_assets: pa.Table = new_video_assets_table()
        self.video_frames: pa.Table = new_video_frames_table()
        self.extrinsics: pa.Table = new_extrinsics_table()

        # Index: cam_id -> paths
        self._cam_index: Dict[str, Dict[str, str]] = {}

        # Load from file if requested
        if rrd_path is not None:
            eps = list(entity_paths) if entity_paths is not None else self._discover_entities(rrd_path)
            print("EPS", eps)
            if eps:
                tables = rrd_io.read_all_for_entities(rrd_path, eps)
                self._ingest_tables(tables)
                self._rebuild_index()

    # --------------------------------------------
    # Public API
    # --------------------------------------------

    def load_mono_camera(
        self,
        *,
        cam_id: str,
        video_path: Optional[str] = None,
        # intrinsics (any one form)
        K: Optional[Sequence[float]] = None,              # length 9 row-major
        fx: Optional[float] = None,
        fy: Optional[float] = None,
        cx: Optional[float] = None,
        cy: Optional[float] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        camera_xyz: Optional[str] = None,                 # e.g. "RDF"
        # frames
        t_ns: Optional[Union[Sequence[int], np.ndarray]] = None,
        video_ts_ns: Optional[Union[Sequence[int], np.ndarray]] = None,
        # extras
        label: Optional[str] = None,
    ) -> Camera:
        paths = self._paths_for(cam_id)
        cam_path = paths["cam"]

        # cameras row
        self.cameras = append_rows(
            self.cameras,
            [
                {
                    "entity_path": cam_path,
                    "camera_id": cam_id,
                    "label": label,
                    "stereo_group": None,
                }
            ],
        )

        # pinhole row
        k_arr, w, h = _normalize_intrinsics(K, fx, fy, cx, cy, width, height)
        self.pinhole = append_rows(
            self.pinhole,
            [
                {
                    "entity_path": cam_path if self.layout is EntityLayout.COLOCATED else paths["img"],
                    "image_from_camera": k_arr,
                    "resolution_u": w,
                    "resolution_v": h,
                    "camera_xyz": camera_xyz,
                }
            ],
        )

        # asset row (optional)
        if video_path:
            self.video_assets = append_rows(
                self.video_assets,
                [
                    {
                        "entity_path": cam_path if self.layout is EntityLayout.COLOCATED else paths["asset"],
                        "video_path": video_path,
                        "blob_sha256": None,
                        "media_type": "video/mp4",
                        "fps_hint": None,
                        "duration_ns": None,
                    }
                ],
            )

        # frame refs (optional)
        if t_ns is not None and video_ts_ns is not None:
            t_list = _to_int_list(t_ns)
            v_list = _to_int_list(video_ts_ns)
            if len(t_list) != len(v_list):
                raise ValueError("t_ns and video_ts_ns must have equal length")
            rows = [
                {
                    "entity_path": cam_path if self.layout is EntityLayout.COLOCATED else paths["img"],
                    "t_ns": t_list[i],
                    "video_ts_ns": v_list[i],
                    "source_video_entity_path": None,
                }
                for i in range(len(t_list))
            ]
            self.video_frames = append_rows(self.video_frames, rows)

        return Camera(self, cam_id, cam_path, label=label)

    def add_extrinsics(
        self,
        cam_id: str,
        t_ns: Union[Sequence[int], np.ndarray],
        *,
        translation: Optional[Sequence[Sequence[float]]] = None,
        quaternion_xyzw: Optional[Sequence[Sequence[float]]] = None,
        rotation_mat: Optional[Sequence[Sequence[float]]] = None,
        relation: Optional[Sequence[str]] = None,
    ) -> None:
        paths = self._paths_for(cam_id)
        cam_path = paths["cam"]

        times = _to_int_list(t_ns)
        n = len(times)

        def _maybe_seq(x, expected_len: Optional[int] = None):
            if x is None:
                return [None] * n
            if len(x) != n:
                raise ValueError("component lengths must match t_ns length")
            if expected_len is not None:
                for row in x:
                    if row is not None and len(row) != expected_len:
                        raise ValueError(f"component rows must have length {expected_len}")
            return x

        t_list = _maybe_seq(translation, 3)
        q_list = _maybe_seq(quaternion_xyzw, 4)
        m_list = _maybe_seq(rotation_mat, 9)
        r_list = _maybe_seq(relation, None)

        rows = [
            {
                "entity_path": cam_path,
                "t_ns": times[i],
                "translation": t_list[i],
                "quaternion": q_list[i],
                "rotation_mat": m_list[i],
                "relation": r_list[i],
            }
            for i in range(n)
        ]
        self.extrinsics = append_rows(self.extrinsics, rows)

    def set_video(
        self,
        cam_id: str,
        video_path: str,
        *,
        media_type: Optional[str] = "video/mp4",
        fps_hint: Optional[float] = None,
        duration_ns: Optional[int] = None,
    ) -> None:
        paths = self._paths_for(cam_id)
        cam_path = paths["cam"]
        row = {
            "entity_path": cam_path if self.layout is EntityLayout.COLOCATED else paths["asset"],
            "video_path": video_path,
            "blob_sha256": None,
            "media_type": media_type,
            "fps_hint": fps_hint,
            "duration_ns": duration_ns,
        }
        self.video_assets = append_rows(self.video_assets, [row])

    def add_video_frames(
        self,
        cam_id: str,
        t_ns: Union[Sequence[int], np.ndarray],
        video_ts_ns: Union[Sequence[int], np.ndarray],
        *,
        source_video_entity_path: Optional[str] = None,
    ) -> None:
        paths = self._paths_for(cam_id)
        ep = paths["cam"] if self.layout is EntityLayout.COLOCATED else paths["img"]
        t_list = _to_int_list(t_ns)
        v_list = _to_int_list(video_ts_ns)
        if len(t_list) != len(v_list):
            raise ValueError("t_ns and video_ts_ns must have equal length")
        rows = [
            {
                "entity_path": ep,
                "t_ns": t_list[i],
                "video_ts_ns": v_list[i],
                "source_video_entity_path": source_video_entity_path,
            }
            for i in range(len(t_list))
        ]
        self.video_frames = append_rows(self.video_frames, rows)

    def list_cameras(self) -> List[str]:
        return self.cameras["camera_id"].to_pylist() if self.cameras.num_rows else []

    def get_camera(self, cam_id: str) -> Camera:
        paths = self._paths_for(cam_id)  # raises if unknown
        label = None
        if self.cameras.num_rows:
            ids = self.cameras["camera_id"].to_pylist()
            if cam_id in ids:
                i = ids.index(cam_id)
                label = self.cameras["label"][i].as_py()
        return Camera(self, cam_id, paths["cam"], label=label)

    # Stereo helper (optional; creates two cameras)
    def load_stereo_camera(
        self,
        *,
        left_id: str,
        right_id: str,
        left: dict,
        right: dict,
        rig_label: Optional[str] = None,
    ) -> Tuple[Camera, Camera]:
        camL = self.load_mono_camera(cam_id=left_id, **left)
        camR = self.load_mono_camera(cam_id=right_id, **right)
        # mark stereo group
        self._annotate_stereo_group([left_id, right_id], rig_label or f"{left_id}_{right_id}")
        return camL, camR

    # Persistence
    def save(self, out_rrd: str, *, app_id: str = "sceneio", spawn_viewer: bool = False) -> None:
        self._validate_tables()
        rrd_io.write_rrd(
            out_rrd,
            SceneTables(
                cameras=self.cameras,
                pinhole=self.pinhole,
                video_assets=self.video_assets,
                video_frames=self.video_frames,
                extrinsics=self.extrinsics,
            ),
            app_id=app_id,
            spawn_viewer=spawn_viewer,
        )

    # --------------------------------------------
    # Internal helpers
    # --------------------------------------------

    def _paths_for(self, cam_id: str) -> Dict[str, str]:
        if cam_id in self._cam_index:
            return self._cam_index[cam_id]
        # Create mapping
        cam_path = f"{self.root}/cameras/{cam_id}"
        if self.layout is EntityLayout.COLOCATED:
            paths = {"cam": cam_path, "img": cam_path, "asset": f"{cam_path}/video"}
        else:
            paths = {"cam": cam_path, "img": f"{cam_path}/image", "asset": f"{cam_path}/video"}
        self._cam_index[cam_id] = paths
        return paths

    def _annotate_stereo_group(self, cam_ids: Sequence[str], group_name: str) -> None:
        if not cam_ids or not self.cameras.num_rows:
            return
        
        # Update existing camera rows with stereo group instead of adding new ones
        camera_data = self.cameras.to_pylist()
        for row in camera_data:
            if row["camera_id"] in cam_ids:
                row["stereo_group"] = group_name
        
        # Rebuild table with updated data
        self.cameras = pa.Table.from_pylist(camera_data, schema=self.cameras.schema)

    def _validate_tables(self) -> None:
        ensure_schema(self.pinhole, PINHOLE_SCHEMA, "pinhole")
        ensure_schema(self.video_assets, VIDEO_ASSETS_SCHEMA, "video_assets")
        ensure_schema(self.video_frames, VIDEO_FRAMES_SCHEMA, "video_frames")
        ensure_schema(self.extrinsics, EXTRINSICS_SCHEMA, "extrinsics")
        # Basic referential checks
        cam_paths = set(self.cameras["entity_path"].to_pylist()) if self.cameras.num_rows else set()
        for tbl, name in [(self.pinhole, "pinhole"), (self.video_frames, "video_frames"), (self.extrinsics, "extrinsics")]:
            if not tbl.num_rows:
                continue
            for ep in tbl["entity_path"].to_pylist():
                if ep not in cam_paths:
                    # In SPLIT_IMAGE_NODE mode, allow mismatched img path; skip for now since default is colocated.
                    pass

    def _ingest_tables(self, t: SceneTables) -> None:
        # Replace in-memory tables with loaded ones if present
        if t.pinhole is not None:
            ensure_schema(t.pinhole, PINHOLE_SCHEMA, "pinhole(load)")
            self.pinhole = t.pinhole
        if t.video_assets is not None:
            ensure_schema(t.video_assets, VIDEO_ASSETS_SCHEMA, "video_assets(load)")
            self.video_assets = t.video_assets
        if t.video_frames is not None:
            ensure_schema(t.video_frames, VIDEO_FRAMES_SCHEMA, "video_frames(load)")
            self.video_frames = t.video_frames
        if t.extrinsics is not None:
            ensure_schema(t.extrinsics, EXTRINSICS_SCHEMA, "extrinsics(load)")
            self.extrinsics = t.extrinsics

        # Build cameras table if missing by discovering unique entity paths from pinhole/video/extrinsics
        cam_rows = []
        seen = set()
        for tbl in (self.pinhole, self.video_frames, self.extrinsics):
            if tbl is None or not tbl.num_rows:
                continue
            for ep in tbl["entity_path"].to_pylist():
                if ep not in seen:
                    cam_rows.append({"entity_path": ep, "camera_id": ep.split("/")[-1], "label": None, "stereo_group": None})
                    seen.add(ep)
        if cam_rows:
            self.cameras = append_rows(self.cameras, cam_rows)

    def _rebuild_index(self) -> None:
        self._cam_index.clear()
        if not self.cameras.num_rows:
            return
        for cid, ep in zip(self.cameras["camera_id"].to_pylist(), self.cameras["entity_path"].to_pylist()):
            if self.layout is EntityLayout.COLOCATED:
                paths = {"cam": ep, "img": ep, "asset": f"{ep}/video"}
            else:
                paths = {"cam": ep, "img": f"{ep}/image", "asset": f"{ep}/video"}
            self._cam_index[cid] = paths

    def _discover_entities(self, rrd_path: str) -> List[str]:
        # Delegate discovery to rrd_io if available, else fallback to empty.
        discover = getattr(rrd_io, "discover_camera_entities", None)
        if callable(discover):
            return list(discover(rrd_path))
        return []


# --------------------------------------------------------------------------------------
# Small utils
# --------------------------------------------------------------------------------------

def _normalize_intrinsics(
    K: Optional[Sequence[float]],
    fx: Optional[float],
    fy: Optional[float],
    cx: Optional[float],
    cy: Optional[float],
    width: Optional[int],
    height: Optional[int],
) -> Tuple[List[float], int, int]:
    if K is not None:
        if len(K) != 9:
            raise ValueError("K must have length 9 (row-major 3x3)")
        if width is None or height is None:
            raise ValueError("width and height are required when K is provided")
        return list(map(float, K)), int(width), int(height)
    # Build K from fx,fy,cx,cy
    if any(v is None for v in (fx, fy, cx, cy, width, height)):
        raise ValueError("Provide either K (9) + width/height, or fx,fy,cx,cy,width,height")
    Km = [float(fx), 0.0, float(cx),
          0.0, float(fy), float(cy),
          0.0, 0.0, 1.0]
    return Km, int(width), int(height)

def _to_int_list(x: Union[Sequence[int], np.ndarray]) -> List[int]:
    if isinstance(x, np.ndarray):
        if x.dtype.kind not in ("i", "u"):
            x = x.astype(np.int64)
        return x.tolist()
    return [int(v) for v in x]
