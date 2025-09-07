#!/usr/bin/env python3
import sys
from pathlib import Path
import rerun as rr
import logging

RRD = sys.argv[1]
ENTITY = "video"

logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(message)s')
logging.debug(f"RRD input: {RRD}")
logging.debug(f"Entity: {ENTITY}")

rec = rr.dataframe.load_recording(RRD)
# Prefer selecting the archetype in contents; columns are chosen in select().
view = rec.view(index=ENTITY, contents="/**")
logging.debug(view.schema())

# select(...) -> RecordBatchReader
reader = view.select_static("/video:AssetVideo:blob", "/video:AssetVideo:media_type")
logging.debug("Reader schema: %s", reader.schema)

# Iterate to log batch counts, rows, and approximate size
batch_count = 0
total_rows = 0
total_bytes = 0
for batch in reader:
    batch_count += 1
    num_rows = batch.num_rows
    batch_bytes = sum(col.nbytes for col in batch.columns)
    total_rows += num_rows
    total_bytes += batch_bytes
    logging.debug(f"Batch {batch_count}: rows={num_rows}, approx_bytes={batch_bytes} bytes")

logging.info(
    "RecordBatchReader summary: batches=%d, total_rows=%d, approx_bytes=%d bytes (~%.2f MiB)",
    batch_count,
    total_rows,
    total_bytes,
    total_bytes / 1_048_576 if total_bytes else 0.0,
)

# Re-create reader (the previous one was consumed) and read all into a table
reader = view.select_static("/video:AssetVideo:blob", "/video:AssetVideo:media_type")
tbl = reader.read_all()
logging.debug(f"Read table with {tbl.num_rows} rows")

blob_col = "/video:AssetVideo:blob"
type_col = "/video:AssetVideo:media_type"

# First non-null blob
for i in range(tbl.num_rows):
    b = tbl.column(blob_col)[i].as_py()
    # Unwrap potential single-element list from static components
    if isinstance(b, list) and len(b) == 1:
        b = b[0]
    if b:
        mtype = tbl.column(type_col)[i].as_py() or "video/mp4"
        if isinstance(mtype, list) and len(mtype) >= 1:
            mtype = mtype[0]
        ext = "mp4" if mtype == "video/mp4" else mtype.split("/")[-1]
        output_path = Path(f"video.{ext}").resolve()
        output_path.write_bytes(bytes(b))
        logging.info(f"Saved video to {output_path}")
        break
else:
    logging.warning("No non-null video blob found in recording for entity %s", ENTITY)
