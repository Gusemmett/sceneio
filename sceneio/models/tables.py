# tables.py
# Coupled schemas & containers for SceneIO <-> io_rrd.
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import pyarrow as pa

SCHEMA_VERSION = "0.2.0"

# --------------------------------------------------------------------------------------
# Schemas (timestamp-only policy). All entity_path columns are absolute Rerun paths.
# --------------------------------------------------------------------------------------

CAMERAS_SCHEMA = pa.schema(
    [
        pa.field("entity_path", pa.string(), nullable=False),  # e.g. "/scene/cameras/left"
        pa.field("camera_id", pa.string(), nullable=False),    # stable user-facing ID
        pa.field("label", pa.string(), nullable=True),
        pa.field("stereo_group", pa.string(), nullable=True),
    ],
    metadata={"sceneio:schema": "cameras", "sceneio:version": SCHEMA_VERSION},
)

PINHOLE_SCHEMA = pa.schema(
    [
        pa.field("entity_path", pa.string(), nullable=False),                 # camera node
        pa.field("image_from_camera", pa.list_(pa.float32(), 9), False),      # row-major 3x3
        pa.field("resolution_u", pa.uint32(), nullable=False),                # width
        pa.field("resolution_v", pa.uint32(), nullable=False),                # height
        pa.field("camera_xyz", pa.string(), nullable=True),                   # e.g. "RDF"
    ],
    metadata={"sceneio:schema": "pinhole", "sceneio:version": SCHEMA_VERSION},
)

VIDEO_ASSETS_SCHEMA = pa.schema(
    [
        pa.field("entity_path", pa.string(), nullable=False),
        pa.field("video_path", pa.string(), nullable=True),
        pa.field("blob_sha256", pa.string(), nullable=True),
        pa.field("blob_bytes", pa.large_binary(), nullable=True),  # NEW
        pa.field("media_type", pa.string(), nullable=True),
        pa.field("fps_hint", pa.float32(), nullable=True),
        pa.field("duration_ns", pa.int64(), nullable=True),
    ],
    metadata={"sceneio:schema": "video_assets", "sceneio:version": SCHEMA_VERSION},
)

VIDEO_FRAMES_SCHEMA = pa.schema(
    [
        pa.field("entity_path", pa.string(), nullable=False),  # camera node
        pa.field("t_ns", pa.int64(), nullable=False),          # global timeline (index)
        pa.field("video_ts_ns", pa.int64(), nullable=False),   # media PTS relative to asset start
        pa.field("source_video_entity_path", pa.string(), nullable=True),  # optional override
    ],
    metadata={"sceneio:schema": "video_frames", "sceneio:version": SCHEMA_VERSION},
)

EXTRINSICS_SCHEMA = pa.schema(
    [
        pa.field("entity_path", pa.string(), nullable=False),             # camera node
        pa.field("t_ns", pa.int64(), nullable=False),
        pa.field("translation", pa.list_(pa.float32(), 3), nullable=True),      # [tx,ty,tz] m
        pa.field("quaternion", pa.list_(pa.float32(), 4), nullable=True),  # [qx,qy,qz,qw]
        pa.field("rotation_mat", pa.list_(pa.float32(), 9), nullable=True),     # row-major 3x3
        pa.field("relation", pa.string(), nullable=True),                        # e.g. "ChildFromParent"
    ],
    metadata={"sceneio:schema": "extrinsics", "sceneio:version": SCHEMA_VERSION},
)

# --------------------------------------------------------------------------------------
# Containers
# --------------------------------------------------------------------------------------

@dataclass
class SceneTables:
    """Strongly-typed container for Arrow tables passed between api.py and io_rrd.py."""
    cameras: Optional[pa.Table] = None
    pinhole: Optional[pa.Table] = None
    video_assets: Optional[pa.Table] = None
    video_frames: Optional[pa.Table] = None
    extrinsics: Optional[pa.Table] = None


# --------------------------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------------------------

def _empty_from_schema(schema: pa.Schema) -> pa.Table:
    return pa.table({f.name: pa.array([], type=f.type) for f in schema}, schema=schema)

def new_cameras_table() -> pa.Table:       return _empty_from_schema(CAMERAS_SCHEMA)
def new_pinhole_table() -> pa.Table:       return _empty_from_schema(PINHOLE_SCHEMA)
def new_video_assets_table() -> pa.Table:  return _empty_from_schema(VIDEO_ASSETS_SCHEMA)
def new_video_frames_table() -> pa.Table:  return _empty_from_schema(VIDEO_FRAMES_SCHEMA)
def new_extrinsics_table() -> pa.Table:    return _empty_from_schema(EXTRINSICS_SCHEMA)

# --------------------------------------------------------------------------------------
# Validation and ops
# --------------------------------------------------------------------------------------

def ensure_schema(table: pa.Table, schema: pa.Schema, where: str = "") -> None:
    if table.schema != schema:
        raise TypeError(f"{where} schema mismatch:\nExpected:\n{schema}\nGot:\n{table.schema}")

def concat_like(tables: Iterable[pa.Table], schema: pa.Schema) -> pa.Table:
    mats = [t for t in tables if t is not None and t.num_rows]
    if not mats:
        return _empty_from_schema(schema)
    return pa.concat_tables(mats, promote=True)

def append_rows(table: pa.Table, rows: List[dict]) -> pa.Table:
    if not rows:
        return table
    batch = pa.Table.from_pylist(rows, schema=table.schema)
    return pa.concat_tables([table, batch], promote=True)

__all__ = [
    "SCHEMA_VERSION",
    "CAMERAS_SCHEMA",
    "PINHOLE_SCHEMA",
    "VIDEO_ASSETS_SCHEMA",
    "VIDEO_FRAMES_SCHEMA",
    "EXTRINSICS_SCHEMA",
    "SceneTables",
    "new_cameras_table",
    "new_pinhole_table",
    "new_video_assets_table",
    "new_video_frames_table",
    "new_extrinsics_table",
    "ensure_schema",
    "concat_like",
    "append_rows",
]
