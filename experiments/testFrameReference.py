"""Log a video asset using automatically determined frame references.

Also attaches a Pinhole camera and logs smooth, random-like extrinsic poses over time.
"""

import sys
import math

import rerun as rr

if len(sys.argv) < 2:
    # TODO(#7354): Only mp4 is supported for now.
    print(f"Usage: {sys.argv[0]} <path_to_video.[mp4]>")
    sys.exit(1)

rr.init("rerun_example_asset_video_auto_frames", spawn=True)

# Log video asset which is referred to by frame references.
video_asset = rr.AssetVideo(path=sys.argv[1])
rr.log("video", video_asset, static=True)

# Also log a pinhole camera on the same path as the video.
# Try to probe the video's resolution with OpenCV if available; otherwise use a reasonable default.
try:
    import cv2  # type: ignore
    cap = cv2.VideoCapture(sys.argv[1])
    if not cap.isOpened():
        width, height = 1920, 1080
    else:
        width = int(cap.get(getattr(cv2, "CAP_PROP_FRAME_WIDTH", 3)))
        height = int(cap.get(getattr(cv2, "CAP_PROP_FRAME_HEIGHT", 4)))
        cap.release()
        if width <= 0 or height <= 0:
            width, height = 1920, 1080
except Exception:
    width, height = 1920, 1080

fx, fy = float(width), float(height)
rr.log(
    "video",
    rr.Pinhole(
        focal_length=(fx, fy),
        width=width,
        height=height,
    ),
    static=True,
)

# Send automatically determined video frame timestamps.
frame_timestamps_ns = video_asset.read_frame_timestamps_nanos()

# Batch-log both Transform3D and VideoFrameReference using send_columns, aligned by frame index.
if len(frame_timestamps_ns) > 0:
    t0 = frame_timestamps_ns[0]
    num_frames = len(frame_timestamps_ns)
    frame_seq = list(range(num_frames))

    times_s = [(ts - t0) * 1e-9 for ts in frame_timestamps_ns]
    translations = [
        [
            0.5 * math.cos(0.5 * t) + 0.05 * math.sin(1.7 * t),
            0.25 * math.sin(0.3 * t) + 0.05 * math.cos(2.3 * t),
            1.0 + 0.1 * math.sin(0.2 * t),
        ]
        for t in times_s
    ]
    rotations = [
        rr.RotationAxisAngle(axis=[0.0, 1.0, 0.0], angle=0.3 * t)
        for t in times_s
    ]

    rr.send_columns(
        "video",
        indexes=[rr.TimeColumn("frame_nr", sequence=frame_seq)],
        columns=[
            *rr.VideoFrameReference.columns_nanos(frame_timestamps_ns),
            *rr.Transform3D.columns(
                translation=translations,
                rotation_axis_angle=rotations,
            ),
        ],
    )
