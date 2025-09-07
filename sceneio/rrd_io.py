# io_rrd.py
# Coupled I/O layer. Only this file imports `rerun`.
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import hashlib
import os
import logging
import tempfile

import pyarrow as pa
import pyarrow.compute as pc
import numpy as np
import rerun as rr
from rerun.dataframe import IndexColumnSelector

from .models.tables import (
    PINHOLE_SCHEMA,
    VIDEO_ASSETS_SCHEMA,
    VIDEO_FRAMES_SCHEMA,
    EXTRINSICS_SCHEMA,
    SceneTables,
    ensure_schema,
    concat_like,
    new_pinhole_table,
    new_video_assets_table,
    new_video_frames_table,
    new_extrinsics_table,
)

# Module logger
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------
# Write
# --------------------------------------------------------------------------------------

def write_rrd(
    out_rrd: str,
    tables: SceneTables,
    *,
    app_id: str = "sceneio",
    spawn_viewer: bool = False,
) -> None:
    """Emit an .rrd from columnar tables. Timestamp-only. Colocated entities."""
    logger.debug(
        "write_rrd: start out_rrd=%s app_id=%s spawn_viewer=%s", out_rrd, app_id, spawn_viewer
    )
    rr.init(app_id, spawn=spawn_viewer)
    rr.save(out_rrd)  # open sink before logging

    if tables.video_assets is not None and tables.video_assets.num_rows:
        logger.debug(
            "write_rrd: video_assets rows=%d", tables.video_assets.num_rows
        )
        ensure_schema(tables.video_assets, VIDEO_ASSETS_SCHEMA, "video_assets")
        _log_asset_videos(tables.video_assets)
    else:
        logger.debug("write_rrd: no video_assets to log")

    if tables.pinhole is not None and tables.pinhole.num_rows:
        logger.debug("write_rrd: pinhole rows=%d", tables.pinhole.num_rows)
        ensure_schema(tables.pinhole, PINHOLE_SCHEMA, "pinhole")
        _log_pinholes(tables.pinhole)
    else:
        logger.debug("write_rrd: no pinhole to log")

    if tables.extrinsics is not None and tables.extrinsics.num_rows:
        logger.debug("write_rrd: extrinsics rows=%d", tables.extrinsics.num_rows)
        ensure_schema(tables.extrinsics, EXTRINSICS_SCHEMA, "extrinsics")
        _log_extrinsics(tables.extrinsics)
    else:
        logger.debug("write_rrd: no extrinsics to log")

    if tables.video_frames is not None and tables.video_frames.num_rows:
        logger.debug("write_rrd: video_frames rows=%d", tables.video_frames.num_rows)
        ensure_schema(tables.video_frames, VIDEO_FRAMES_SCHEMA, "video_frames")
        _log_video_frames(tables.video_frames)
    else:
        logger.debug("write_rrd: no video_frames to log")

    logger.debug("write_rrd: done out_rrd=%s", out_rrd)

# --------------------------------------------------------------------------------------
# Read (focused; per-entity helpers + batch over a list of entities)
# --------------------------------------------------------------------------------------

def read_pinhole(rrd_path: str, entity_path: str) -> pa.Table:
    logger.debug("read_pinhole: rrd_path=%s entity_path=%s", rrd_path, entity_path)
    ep = _qpath(entity_path)
    rec = rr.dataframe.load_recording(rrd_path)

    # Static read only.
    view = rec.view(index=None, contents=f"{ep}/**")

    reader = view.select_static(
        f"{ep}:Pinhole:image_from_camera",
        f"{ep}:Pinhole:resolution",
        f"{ep}:Pinhole:camera_xyz",
    )
    tbl = reader.read_all()
    logger.debug("read_pinhole: read rows=%d for %s", tbl.num_rows, ep)
    if tbl.num_rows == 0:
        logger.debug("read_pinhole: empty, returning empty table for %s", ep)
        return new_pinhole_table()

    K   = _unwrap_static(tbl.column(f"{ep}:Pinhole:image_from_camera")[0].as_py())
    res = _unwrap_static(tbl.column(f"{ep}:Pinhole:resolution")[0].as_py())
    xyz_raw = _unwrap_static(tbl.column(f"{ep}:Pinhole:camera_xyz")[0].as_py())
    xyz = _normalize_vc(xyz_raw)  # <-- keep canonical token like "RDF"

    if K is None or res is None:
        return new_pinhole_table()

    return pa.table(
        {
            "entity_path": pa.array([ep]),
            "image_from_camera": pa.array([K], type=pa.list_(pa.float32(), 9)),
            "resolution_u": pa.array([int(res[0])], type=pa.uint32()),
            "resolution_v": pa.array([int(res[1])], type=pa.uint32()),
            "camera_xyz": pa.array([xyz], type=pa.string()),  # may be None
        },
        schema=PINHOLE_SCHEMA,
    )


# read_extrinsics
def read_extrinsics(rrd_path: str, entity_path: str, timeline: str = "time_ns") -> pa.Table:
    logger.debug(
        "read_extrinsics: rrd_path=%s entity_path=%s timeline=%s", rrd_path, entity_path, timeline
    )
    ep = _qpath(entity_path)
    rec = rr.dataframe.load_recording(rrd_path)
    view = rec.view(index=timeline, contents=f"{ep}/**")

    names = set(_schema_component_names(view))
    logger.debug("read_extrinsics: available components=%d for %s", len(names), ep)

    # Prefer modern Transform3D names, fall back to legacy if present.
    cand = []
    for sfx in (":Transform3D:translation", ":Transform3D:rotation", ":Transform3D:mat3x3", ":Transform3D:relation"):
        n = f"{ep}{sfx}"
        if n in names:
            cand.append(n)
    # legacy compatibility
    legacy = []
    for sfx in (":Translation3D", ":RotationQuat", ":TransformMat3x3", ":TransformRelation"):
        n = f"{ep}{sfx}"
        if n in names:
            legacy.append(n)

    if not cand and not legacy:
        logger.debug("read_extrinsics: no transform components found for %s", ep)
        return new_extrinsics_table()

    reader = view.select(IndexColumnSelector(timeline), *(cand or legacy))

    times, t_list, q_list, m_list, rel_list = [], [], [], [], []
    trans_col = next((c for c in cand if c.endswith(":translation")), None) or f"{ep}:Translation3D"
    quat_col  = next((c for c in cand if c.endswith(":rotation")), None)    or f"{ep}:RotationQuat"
    mat_col   = next((c for c in cand if c.endswith(":mat3x3")), None)      or f"{ep}:TransformMat3x3"
    rel_col   = next((c for c in cand if c.endswith(":relation")), None)    or f"{ep}:TransformRelation"
    logger.debug(
        "read_extrinsics: using columns trans=%s quat=%s mat=%s rel=%s",
        trans_col,
        quat_col,
        mat_col,
        rel_col,
    )

    for batch in reader:
        if timeline not in batch.schema.names:
            logger.debug("read_extrinsics: timeline column '%s' missing in batch for %s", timeline, ep)
            return new_extrinsics_table()
        idx = batch.column(timeline)
        if pa.types.is_timestamp(idx.type) or pa.types.is_time(idx.type):
            idx = pc.cast(idx, pa.int64())
        times.extend(idx.to_pylist())

        n = batch.num_rows
        trans_col = ...
        # keep your existing per-row extraction for translation/quat/mat/relation
        for i in range(n):
            t_list.append(batch.column(trans_col)[i].as_py() if trans_col in batch.schema.names else None)
            q_list.append(batch.column(quat_col)[i].as_py() if quat_col in batch.schema.names else None)
            m_list.append(batch.column(mat_col)[i].as_py() if mat_col in batch.schema.names else None)
            rel_list.append(batch.column(rel_col)[i].as_py() if rel_col in batch.schema.names else None)


    if not times:
        logger.debug("read_extrinsics: no samples for %s", ep)
        return new_extrinsics_table()

    out_tbl = pa.table(
        {
            "entity_path": pa.array([ep] * len(times)),
            "t_ns": pa.array(times, type=pa.int64()),
            "translation": pa.array(t_list, type=pa.list_(pa.float32(), 3)),
            "quaternion": pa.array(q_list, type=pa.list_(pa.float32(), 4)),
            "rotation_mat": pa.array(m_list, type=pa.list_(pa.float32(), 9)),
            "relation": pa.array(rel_list, type=pa.string()),
        },
        schema=EXTRINSICS_SCHEMA,
    )
    logger.debug("read_extrinsics: returning rows=%d for %s", out_tbl.num_rows, ep)
    return out_tbl


def read_video_frames(rrd_path: str, entity_path: str, timeline: str = "time_ns") -> pa.Table:
    logger.debug(
        "read_video_frames: rrd_path=%s entity_path=%s timeline=%s", rrd_path, entity_path, timeline
    )
    ep = _qpath(entity_path)
    rec = rr.dataframe.load_recording(rrd_path)
    view = rec.view(index=timeline, contents=f"{ep}/**")

    names = set(_schema_component_names(view))
    ts_col = None
    for cand in (f"{ep}:VideoFrameReference:timestamp", f"{ep}:VideoTimestamp"):
        if cand in names:
            ts_col = cand
            break
    if ts_col is None:
        logger.debug("read_video_frames: no timestamp component for %s", ep)
        return new_video_frames_table()

    ep_col = f"{ep}:EntityPath" if f"{ep}:EntityPath" in names else None

    reader = view.select(IndexColumnSelector(timeline), ts_col, *( [ep_col] if ep_col else [] ))

    times, pts, srcs = [], [], []
    for batch in reader:
        if timeline not in batch.schema.names:
            logger.debug("read_video_frames: timeline column '%s' missing in batch for %s", timeline, ep)
            return new_video_frames_table()

        idx = batch.column(timeline)
        if pa.types.is_timestamp(idx.type) or pa.types.is_time(idx.type):
            idx = pc.cast(idx, pa.int64())
        times.extend(idx.to_pylist())

        n = batch.num_rows
        for i in range(n):
            v = batch.column(ts_col)[i].as_py() if ts_col in batch.schema.names else None
            v = _unwrap1(v)
            pts.append(None if v is None else int(v))

            s = batch.column(ep_col)[i].as_py() if ep_col and ep_col in batch.schema.names else None
            s = _unwrap1(s)
            srcs.append(s if isinstance(s, str) else None)

    out_tbl = pa.table(
        {
            "entity_path": pa.array([ep] * len(times)),
            "t_ns": pa.array(times, type=pa.int64()),
            "video_ts_ns": pa.array(pts, type=pa.int64()),
            "source_video_entity_path": pa.array(srcs, type=pa.string()),
        },
        schema=VIDEO_FRAMES_SCHEMA,
    )
    logger.debug("read_video_frames: returning rows=%d for %s", out_tbl.num_rows, ep)
    return out_tbl


def read_video_asset_meta(rrd_path: str, entity_path: str) -> pa.Table:
    ep = _qpath(entity_path)
    rec = rr.dataframe.load_recording(rrd_path)
    view = rec.view(index=None, contents=f"{ep}/**")
    reader = view.select_static(f"{ep}:AssetVideo:blob", f"{ep}:AssetVideo:media_type")
    tbl = reader.read_all()

    out = new_video_assets_table()
    if tbl.num_rows == 0:
        return out

    blob_col = f"{ep}:AssetVideo:blob"
    type_col = f"{ep}:AssetVideo:media_type"

    for i in range(tbl.num_rows):
        blob = _unwrap_static(tbl.column(blob_col)[i].as_py())
        if blob:
            raw = bytes(blob)
            sha = hashlib.sha256(raw).hexdigest()
            mtype = _unwrap_static(tbl.column(type_col)[i].as_py()) or "video/mp4"
            return pa.table(
                {
                    "entity_path": pa.array([ep]),
                    "video_path": pa.array([None], type=pa.string()),
                    "blob_sha256": pa.array([sha], type=pa.string()),
                    "blob_bytes": pa.array([raw], type=pa.large_binary()),  # NEW
                    "media_type": pa.array([mtype], type=pa.string()),
                    "fps_hint": pa.array([None], type=pa.float32()),
                    "duration_ns": pa.array([None], type=pa.int64()),
                },
                schema=VIDEO_ASSETS_SCHEMA,
            )
    return out


def read_all_for_entities(rrd_path: str, entity_paths: Sequence[str]) -> SceneTables:
    """Batch read for a known set of camera entities."""
    logger.debug(
        "read_all_for_entities: rrd_path=%s num_entities=%d", rrd_path, len(entity_paths)
    )
    pinholes = [read_pinhole(rrd_path, ep) for ep in entity_paths]
    extr = [read_extrinsics(rrd_path, ep) for ep in entity_paths]
    vframes = [read_video_frames(rrd_path, ep) for ep in entity_paths]
    vassets = [read_video_asset_meta(rrd_path, ep) for ep in entity_paths]

    out = SceneTables(
        pinhole=concat_like(pinholes, PINHOLE_SCHEMA),
        extrinsics=concat_like(extr, EXTRINSICS_SCHEMA),
        video_frames=concat_like(vframes, VIDEO_FRAMES_SCHEMA),
        video_assets=concat_like(vassets, VIDEO_ASSETS_SCHEMA),
    )
    logger.debug(
        "read_all_for_entities: combined tables pinhole=%d extrinsics=%d video_frames=%d video_assets=%d",
        0 if out.pinhole is None else out.pinhole.num_rows,
        0 if out.extrinsics is None else out.extrinsics.num_rows,
        0 if out.video_frames is None else out.video_frames.num_rows,
        0 if out.video_assets is None else out.video_assets.num_rows,
    )
    return out

# --------------------------------------------------------------------------------------
# Internal write helpers
# --------------------------------------------------------------------------------------

def _log_asset_videos(tbl: pa.Table) -> None:
    ep     = _to_pylist(tbl, "entity_path", cast=str)
    paths  = _opt_pylist(tbl, "video_path", cast=str)
    mtypes = _opt_pylist(tbl, "media_type", cast=str)
    blobs  = _opt_pylist(tbl, "blob_bytes")  # large_binary -> Python bytes/memoryview

    for i, entity in enumerate(ep):
        mtype = mtypes[i] if mtypes else None

        # 1) Prefer real file path if present.
        p = (paths[i] if paths else None)
        if p:
            rr.log(entity, rr.AssetVideo(path=p, media_type=mtype) if mtype else rr.AssetVideo(path=p), static=True)
            continue

        # 2) Else use embedded bytes.
        raw = blobs[i] if blobs else None
        if raw is None:
            continue
        if isinstance(raw, memoryview):
            raw = raw.tobytes()
        elif isinstance(raw, bytearray):
            raw = bytes(raw)

        av = None
        # Try possible constructor spellings across versions.
        for ctor in (
            lambda: rr.AssetVideo(blob=raw,  media_type=mtype) if mtype else rr.AssetVideo(blob=raw),
            lambda: rr.AssetVideo(bytes=raw, media_type=mtype) if mtype else rr.AssetVideo(bytes=raw),
            lambda: rr.AssetVideo(data=raw,  media_type=mtype) if mtype else rr.AssetVideo(data=raw),
        ):
            try:
                av = ctor()
                break
            except TypeError:
                pass

        # 3) Fallback: temp file then path (ensures blob lands in the RRD).
        if av is None:
            sha = hashlib.sha256(raw).hexdigest()[:12]
            tmp = os.path.join(tempfile.gettempdir(), f"sceneio_{sha}.mp4")
            with open(tmp, "wb") as f:
                f.write(raw)
            av = rr.AssetVideo(path=tmp, media_type=mtype) if mtype else rr.AssetVideo(path=tmp)

        rr.log(entity, av, static=True)

def _log_pinholes(tbl: pa.Table) -> None:
    ep     = _to_pylist(tbl, "entity_path", cast=str)
    K_list = _to_pylist(tbl, "image_from_camera")          # list[9]
    w_list = _to_pylist(tbl, "resolution_u", cast=int)
    h_list = _to_pylist(tbl, "resolution_v", cast=int)
    xyz_l  = _opt_pylist(tbl, "camera_xyz", cast=str)

    for i, entity in enumerate(ep):
        k9 = K_list[i]
        fx, fy, cx, cy = _derive_fx_fy_cx_cy(k9)
        kwargs = dict(focal_length=(fx, fy), principal_point=(cx, cy),
                      width=w_list[i], height=h_list[i])
        token = xyz_l[i] if xyz_l else None
        vc = _vc_constant(token)
        if vc is not None:
            kwargs["camera_xyz"] = vc
        rr.log(entity, rr.Pinhole(**kwargs), static=True)


def _log_extrinsics(tbl: pa.Table) -> None:
    logger.debug("_log_extrinsics: rows=%d", tbl.num_rows)
    ep = _to_pylist(tbl, "entity_path", cast=str)
    tns = _to_pylist(tbl, "t_ns", cast=int)
    trans = _opt_pylist(tbl, "translation")
    quat = _opt_pylist(tbl, "quaternion")
    rmat = _opt_pylist(tbl, "rotation_mat")
    rel = _opt_pylist(tbl, "relation", cast=str)

    # group by entity
    groups: Dict[str, List[int]] = {}
    for i, p in enumerate(ep):
        groups.setdefault(p, []).append(i)

    for entity, idxs in groups.items():
        times = [tns[i] for i in idxs]
        t_list = [trans[i] if trans else None for i in idxs]
        q_list = [quat[i] if quat else None for i in idxs]
        m_list = [rmat[i] if rmat else None for i in idxs]
        r_list = [rel[i] if rel else None for i in idxs]

        times_ns = [tns[i] for i in idxs]
        times_s = _ns_to_seconds_list(times_ns)

        logger.debug(
            "_log_extrinsics: entity=%s rows=%d has_t=%s has_q=%s has_m=%s has_rel=%s",
            entity,
            len(idxs),
            any(v is not None for v in t_list),
            any(v is not None for v in q_list),
            any(v is not None for v in m_list),
            any(v is not None for v in r_list),
        )
        rr.send_columns(
            entity,
            indexes=[rr.TimeColumn("time_ns", timestamp=times_s)],
            columns=rr.Transform3D.columns(
                translation=t_list if any(v is not None for v in t_list) else None,
                quaternion=q_list if any(v is not None for v in q_list) else None,
                mat3x3=m_list if any(v is not None for v in m_list) else None,
                relation=r_list if any(v is not None for v in r_list) else None,
            ),
        )

def _log_video_frames(tbl: pa.Table) -> None:
    logger.debug("_log_video_frames: rows=%d", tbl.num_rows)
    ep = _to_pylist(tbl, "entity_path", cast=str)
    tns = _to_pylist(tbl, "t_ns", cast=int)
    vts = _to_pylist(tbl, "video_ts_ns", cast=int)
    # Note: cross-entity video references are optional; colocated by default.
    src = _opt_pylist(tbl, "source_video_entity_path", cast=str)

    groups: Dict[str, List[int]] = {}
    for i, p in enumerate(ep):
        groups.setdefault(p, []).append(i)

    for entity, idxs in groups.items():
        times = [tns[i] for i in idxs]
        media = [vts[i] for i in idxs]

        times_ns = [tns[i] for i in idxs]
        times_s = _ns_to_seconds_list(times_ns)

        logger.debug(
            "_log_video_frames: entity=%s rows=%d", entity, len(idxs)
        )
        rr.send_columns(
            entity,
            indexes=[rr.TimeColumn("time_ns", timestamp=times_s)],
            columns=rr.VideoFrameReference.columns_nanos(media),
        )

# --------------------------------------------------------------------------------------
# Small utils
# --------------------------------------------------------------------------------------

def discover_camera_entities(rrd_path: str) -> list[str]:
    """
    Heuristically discover camera entity paths in an .rrd.
    Picks entities that have Pinhole, VideoFrameReference, Transform3D, or AssetVideo.
    Normalizes '/.../image' and '/.../video' to the camera parent.
    """
    rec = rr.dataframe.load_recording(rrd_path)
    # Full-schema pass; static view is fastest for discovery.
    view = rec.view(index=None, contents="/**")

    cams: set[str] = set()
    for d in view.schema().component_columns():
        ep = d.entity_path
        if ep.startswith("/__"):
            continue  # skip recording properties

        comp = getattr(d, "component", None) or getattr(d, "component_name", None)
        comp = str(comp) if comp is not None else ""

        # Identify camera-ish components
        if (
            comp.startswith("Pinhole:")
            or comp.startswith("VideoFrameReference:")
            or comp.startswith("Transform3D:")
            or comp in ("Translation3D", "RotationQuat", "TransformMat3x3", "TransformRelation")
            or comp.startswith("AssetVideo:")
        ):
            # Normalize child nodes commonly used for media/image splits
            if ep.endswith("/image") or ep.endswith("/video"):
                ep = ep.rsplit("/", 1)[0]
            cams.add(ep)

    out = sorted(cams)
    logger.debug("discover_camera_entities: found %d entities", len(out))
    return out

def _to_pylist(tbl: pa.Table, name: str, cast=None):
    if name not in tbl.column_names:
        logger.error("_to_pylist: missing required column '%s' (available=%s)", name, list(tbl.column_names))
        raise KeyError(f"Missing required column: {name}")
    col = tbl[name]
    vals = col.to_pylist()
    if cast is None:
        logger.debug("_to_pylist: column=%s len=%d (no cast)", name, len(vals))
        return vals
    out = [None if v is None else cast(v) for v in vals]
    logger.debug("_to_pylist: column=%s len=%d (cast applied)", name, len(out))
    return out

def _opt_pylist(tbl: pa.Table, name: str, cast=None):
    if name not in tbl.column_names:
        logger.debug("_opt_pylist: optional column '%s' not present", name)
        return None
    return _to_pylist(tbl, name, cast=cast)

def _unwrap_static(v):
    if isinstance(v, list) and len(v) == 1:
        return v[0]
    return v

def _nullable_list(batch: pa.RecordBatch, col_name: str, i: int):
    if col_name not in batch.schema.names:
        logger.debug("_nullable_list: column '%s' not in schema (row=%d)", col_name, i)
        return None
    v = batch.column(col_name)[i].as_py()
    return v if v is not None else None

def _ns_to_seconds_list(ns_vals):
    arr = np.asarray(ns_vals, dtype=np.int64)
    out = (arr * 1e-9).tolist()
    logger.debug("_ns_to_seconds_list: converted %d timestamps", len(out))
    return out

def _qpath(ep: str) -> str:
    """Normalize to exactly one leading slash."""
    out = ep if ep.startswith("/") else f"/{ep}"
    if out != ep:
        logger.debug("_qpath: normalized '%s' -> '%s'", ep, out)
    return out

def _schema_component_names(view) -> list[str]:
    sch = view.schema()
    out = []
    for d in sch.component_columns():
        # 0.24.x exposes `component`; older builds may expose `component_name`
        comp = getattr(d, "component", None) or getattr(d, "component_name", None)
        out.append(f"{d.entity_path}:{comp}")
    return out

def _unwrap1(v):
    return None if v is None else (v[0] if isinstance(v, list) and len(v) > 0 else (v if not isinstance(v, list) else None))

# top-level
VC_CODE = {1:"U", 2:"D", 3:"R", 4:"L", 5:"F", 6:"B"}  # docs enum

def _to_mat3x3_rows(k):
    # Accept 9-flat or 3x3. Return nested 3 rows.
    if k is None:
        return None
    if isinstance(k, (list, tuple)) and len(k) == 9:
        return [k[0:3], k[3:6], k[6:9]]
    if isinstance(k, (list, tuple)) and len(k) == 3 and all(isinstance(r, (list, tuple)) and len(r) == 3 for r in k):
        return [list(k[0]), list(k[1]), list(k[2])]
    return None

def _normalize_vc(val):
    # Accept "RDF", ["R","D","F"], [3,2,5], {"x":"R","y":"D","z":"F"}.
    if val is None:
        return None
    if isinstance(val, str):
        s = val.upper()
        return s if s in {"RDF","RUB","RUF","RDB","RBD","RFD","RFU","RUB","RUF"} or len(s)==3 else None
    if isinstance(val, (list, tuple)) and len(val) == 3:
        if all(isinstance(x, int) for x in val):
            s = "".join(VC_CODE.get(x, "?") for x in val)
            return s if "?" not in s else None
        if all(isinstance(x, str) and len(x) >= 1 for x in val):
            s = "".join(x[0].upper() for x in val)  # ["Right","Down","Forward"] -> "RDF"
            return s
    if isinstance(val, dict):
        parts = [val.get("x"), val.get("y"), val.get("z")]
        return _normalize_vc(parts)
    return None

def _vc_constant(token):
    # Try rr.ViewCoordinates.<TOKEN>, else rr.components.ViewCoordinates.<TOKEN>
    import rerun as rr
    if token is None:
        return None
    for ns in (rr.ViewCoordinates, getattr(rr, "components", None) and rr.components.ViewCoordinates):
        if ns is None: 
            continue
        obj = getattr(ns, token, None)
        if obj is not None:
            return obj
    return None

def _derive_fx_fy_cx_cy(k9: list[float]) -> tuple[float,float,float,float]:
    # k9 is image_from_camera flattened; could be row- or column-major.
    # Common K:
    # [ fx, 0, cx,
    #   0, fy, cy,
    #   0,  0,  1 ]
    # Row-major indices: 0,4,2,5. Column-major indices: 0,4,6,7.
    if len(k9) != 9:
        raise ValueError("image_from_camera must have 9 elements")
    row_major = (k9[2] != 0) or (k9[6] == 0)  # heuristic
    fx = float(k9[0])
    fy = float(k9[4])
    cx = float(k9[2] if row_major else k9[6])
    cy = float(k9[5] if row_major else k9[7])
    return fx, fy, cx, cy