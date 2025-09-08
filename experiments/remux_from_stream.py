#!/usr/bin/env python3
"""Remux one VideoStream from a Rerun .rrd recording to MP4 without re-encoding.

This script currently is hardcoded for a particular rrd file. You can change the entitity path at the ENTITY variable

Example:
    python experiments/remux_from_stream.py path/to/recording.rrd -o out.mp4

From pixi environment:
    pixi run python experiments/remux_from_stream.py /Users/angusemmett/Downloads/20250907_142545_t265_slam_rrd_0.24.1.rrd -o right.mp4
"""
import argparse, logging, re, subprocess
from pathlib import Path
from statistics import median

import rerun as rr
import pyarrow as pa

ENTITY = "/t265/right/pinhole/video_stream"
COMP_SAMPLE = f"{ENTITY}:VideoStream:sample"
COMP_CODEC  = f"{ENTITY}:VideoStream:codec"
INDEX_PREFS = ("video_time", "time", "log_time", "frame_nr", "log_tick")

# ----------------------- utils -----------------------

def setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")

def unwrap(v):
    # Unwrap [x] → x repeatedly
    while isinstance(v, list) and len(v) == 1:
        v = v[0]
    return v

def to_bytes(cell) -> bytes:
    v = unwrap(cell)
    if v is None:
        return b""
    if isinstance(v, (bytes, bytearray, memoryview)):
        return bytes(v)
    if isinstance(v, list):
        # If list of ints → bytes; if list of bytes → first element
        if v and isinstance(v[0], int):
            return bytes(v)
        if v and isinstance(v[0], (bytes, bytearray, memoryview)):
            return bytes(v[0])
        # Flatten nested singletons again
        v = unwrap(v)
        if isinstance(v, list) and v and isinstance(v[0], int):
            return bytes(v)
        if isinstance(v, (bytes, bytearray, memoryview)):
            return bytes(v)
        raise TypeError(f"Unexpected list cell type: {type(v)}")
    if isinstance(v, int):
        # Rare: a single byte value. Convert robustly.
        return bytes([v & 0xFF])
    if isinstance(v, str):
        return v.encode("latin1", "ignore")
    try:
        return bytes(v)
    except Exception as e:
        raise TypeError(f"Cannot convert cell of type {type(v)} to bytes: {e}")

def decode_codec(cell):
    v = unwrap(cell)
    if v is None:
        return None
    if isinstance(v, int):
        # FourCC as int → ascii, e.g. 1635148593 -> 'avc1'
        try:
            return int(v).to_bytes(4, "big").decode("ascii", "ignore")
        except Exception:
            return str(v)
    if isinstance(v, (bytes, bytearray, memoryview)):
        try:
            return bytes(v).decode("ascii", "ignore")
        except Exception:
            return str(v)
    return str(v)

def pick_index(rec):
    idx_descs = rec.schema().index_columns()
    names = [getattr(c, "name", str(c)) for c in idx_descs]
    logging.debug("Index columns discovered: %s", names)
    # names look like "Index(timeline:log_time)"
    for cand in INDEX_PREFS:
        target = f"Index(timeline:{cand})"
        if target in names:
            logging.info("Using index timeline: %s", cand)
            return target, cand
    if names:
        # fallback to first
        m = re.match(r"Index\(timeline:(.+)\)", names[0])
        tl = m.group(1) if m else names[0]
        logging.info("Using fallback index timeline: %s", tl)
        return names[0], tl
    raise SystemExit("No index columns in recording")

def get_first_non_null(tbl: pa.Table, col: str):
    if tbl is None or tbl.num_rows == 0 or col not in tbl.column_names:
        return None
    arr = tbl[col]
    for i in range(len(arr)):
        v = arr[i].as_py()
        v = unwrap(v)
        if v is not None:
            return v
    return None

def fps_from_ts(ts_list):
    if len(ts_list) < 2:
        return None
    def to_sec(x):
        if isinstance(x, int):
            # Heuristic: nanoseconds vs frame_nr
            return x / 1e9 if x > 1_000_000_000 else None
        try:
            return float(x)
        except Exception:
            return None
    vals = [to_sec(x) for x in ts_list]
    vals = [v for v in vals if v is not None]
    if len(vals) < 2:
        return None
    deltas = [b - a for a, b in zip(vals, vals[1:]) if b > a]
    if not deltas:
        return None
    dt = median(deltas)
    return 1.0 / dt if dt > 0 else None

def probe_packets_and_fps(view, idx_colname, comp_colname, max_rows=200):
    cached = []
    ts = []
    total_rows = 0
    reader = view.select(idx_colname, comp_colname)
    logging.debug("Probe reader schema: %s", reader.schema)
    for batch_idx, batch in enumerate(reader, 1):
        n = batch.num_rows
        total_rows += n
        logging.debug("Probe batch %d: rows=%d", batch_idx, n)
        idx_col = batch.column(idx_colname)
        data_col = batch.column(comp_colname)
        for i in range(n):
            pkt = to_bytes(data_col[i].as_py())
            cached.append(pkt)
            ts.append(idx_col[i].as_py())
            if len(cached) >= max_rows:
                fps = fps_from_ts(ts)
                logging.info("Probe collected %d samples; fps_est=%s",
                             len(cached), f"{fps:.3f}" if fps else "n/a")
                return cached, fps
    fps = fps_from_ts(ts)
    logging.info("Probe read complete; rows=%d; fps_est=%s",
                 total_rows, f"{fps:.3f}" if fps else "n/a")
    return cached, fps

def open_ffmpeg(out_path: Path, fps: float | None):
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "warning", "-y"]
    if fps and fps > 0:
        cmd += ["-r", f"{fps:.6f}"]  # input rate
    cmd += ["-fflags", "+genpts", "-f", "h264", "-i", "pipe:0",
            "-c", "copy", "-movflags", "+faststart", str(out_path)]
    logging.debug("Spawn ffmpeg: %s", " ".join(cmd))
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)

# ----------------------- main -----------------------

def main():
    ap = argparse.ArgumentParser(description="Remux one VideoStream from Rerun RRD to MP4 without re-encoding.")
    ap.add_argument("rrd", help="Path to .rrd")
    ap.add_argument("-o", "--output", default="out.mp4", help="Output mp4")
    ap.add_argument("--fps", type=float, default=30.0, help="Fallback FPS")
    ap.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    args = ap.parse_args()
    setup_logging(args.verbose)

    logging.info("Loading recording: %s", args.rrd)
    rec = rr.dataframe.load_recording(args.rrd)
    logging.debug("Recording schema: index_cols=%s component_cols=%d",
                  [getattr(c, "name", str(c)) for c in rec.schema().index_columns()],
                  len(rec.schema().component_columns()))

    idx_colname, tl = pick_index(rec)
    view = rec.view(index=tl, contents="/**")
    logging.debug("View ready. Index timeline: %s | Index column: %s", tl, idx_colname)

    # Codec info
    try:
        tbl = view.select(COMP_CODEC).read_all()
        codec_raw = get_first_non_null(tbl, COMP_CODEC)
        codec = decode_codec(codec_raw)
        logging.info("Codec: %r (raw=%r)", codec, codec_raw)
        if codec and ("264" not in codec and "avc" not in codec.lower()):
            logging.warning("Non-H.264 codec seen; remux may fail")
    except Exception as e:
        logging.warning("Codec read failed: %s", e)

    # Probe a few samples to estimate FPS and warm the pipeline
    cached_packets, fps_est = probe_packets_and_fps(view, idx_colname, COMP_SAMPLE, max_rows=200)
    fps_use = fps_est or args.fps
    logging.info("Using FPS: %s (fallback=%s)", f"{fps_use:.3f}", args.fps)

    # Start ffmpeg
    out_path = Path(args.output)
    proc = open_ffmpeg(out_path, fps_use)

    # Send cached first
    sent_total = 0
    for i, pkt in enumerate(cached_packets, 1):
        if pkt:
            proc.stdin.write(pkt)
            sent_total += 1
            if i % 50 == 0:
                logging.debug("Wrote cached packet #%d (total=%d)", i, sent_total)

    # Stream the rest by re-reading and skipping 'sent_total'
    logging.debug("Streaming remainder after %d cached packets…", sent_total)
    skipped = 0
    for batch_idx, batch in enumerate(view.select(idx_colname, COMP_SAMPLE), 1):
        n = batch.num_rows
        data_col = batch.column(COMP_SAMPLE)
        for i in range(n):
            if skipped < sent_total:
                skipped += 1
                continue
            pkt = to_bytes(data_col[i].as_py())
            if pkt:
                proc.stdin.write(pkt)
                sent_total += 1
        if batch_idx % 10 == 0:
            logging.debug("Streamed through batch %d (rows this batch=%d, total packets=%d)",
                          batch_idx, n, sent_total)

    proc.stdin.close()
    ret = proc.wait()
    if ret != 0:
        raise SystemExit(f"ffmpeg exited with code {ret}")
    logging.info("Wrote %s (packets=%d)", out_path, sent_total)

if __name__ == "__main__":
    main()
