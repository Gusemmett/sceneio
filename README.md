sceneio
======

Columnar scene I/O for Rerun recordings (.rrd).

### Installation

```bash
pip install -e .
```

### Quick start: build a scene from a video

```python
import time, numpy as np
import rerun as rr
from sceneio.api import SceneIO

# Derive per-frame timestamps (ns) from the media
av = rr.AssetVideo(path="video.mp4")
vts_ns = av.read_frame_timestamps_nanos()
base = time.time_ns(); v0 = vts_ns[0]
t_ns = [base + (ts - v0) for ts in vts_ns]
video_ts_ns = [ts - v0 for ts in vts_ns]

# Simple intrinsics
width, height = 1920, 1080
fx, fy = float(width), float(height)
cx, cy = width / 2.0, height / 2.0
K = [fx,0,cx,  0,fy,cy,  0,0,1]

scene = SceneIO(root="/scene")
cam = scene.load_mono_camera(
    cam_id="cam0",
    video_path="video.mp4",
    K=K, width=width, height=height, camera_xyz="RDF",
)
cam.add_video_frames(t_ns=np.array(t_ns), video_ts_ns=np.array(video_ts_ns))

scene.save("out.rrd", app_id="sceneio_demo", spawn_viewer=False)
```

### Load and append to an existing .rrd

```python
import numpy as np
from sceneio.api import SceneIO

scene = SceneIO(rrd_path="in.rrd")
print(scene.list_cameras())  # e.g. ['cam0']
cam = scene.get_camera("cam0")

# Use existing video frame times for this camera
vf = scene.video_frames
mask = np.array(vf["entity_path"].to_pylist()) == cam.entity_path
t_ns = np.array(vf["t_ns"].to_pylist())[mask]

# Append poses (example: simple path in meters)
t0 = t_ns[0]
t_s = (t_ns - t0) * 1e-9
translations = np.stack([
    0.25 * np.sin(0.7 * t_s),
    0.0 * t_s,
    1.0 + 0.10 * np.cos(0.3 * t_s),
], axis=1).tolist()

cam.add_extrinsics(t_ns=t_ns.tolist(), translation=translations)
scene.save("out_with_poses.rrd")
```

### API overview (SDK)

- **Scene construction/loading**
  - `SceneIO(rrd_path: Optional[str] = None, *, root="/scene")`
  - `scene.save(out_rrd: str, *, app_id: str = "sceneio", spawn_viewer: bool = False)`
  - `scene.list_cameras() -> list[str]`
  - `scene.get_camera(cam_id: str) -> Camera`

- **Cameras**
  - `scene.load_mono_camera(*, cam_id, video_path=None, K=None | fx,fy,cx,cy, width, height, camera_xyz=None, t_ns=None, video_ts_ns=None, label=None) -> Camera`
  - `scene.load_stereo_camera(*, left_id, right_id, left: dict, right: dict, rig_label=None) -> (Camera, Camera)`

- **Assets and frames**
  - `scene.set_video(cam_id, video_path, *, media_type="video/mp4", fps_hint=None, duration_ns=None)`
  - `scene.add_video_frames(cam_id, t_ns, video_ts_ns, *, source_video_entity_path=None)`

- **Extrinsics (poses)**
  - `scene.add_extrinsics(cam_id, t_ns, *, translation=None, quaternion_xyzw=None, rotation_mat=None, relation=None)`

- **Camera handle conveniences**
  - `Camera.set_video(...)`
  - `Camera.add_video_frames(...)`
  - `Camera.add_extrinsics(...)`

See the scripts in `examples/` for end-to-end usage:
- `examples/smoke_test.py`: create a scene from a video and save.
- `examples/append_test.py`: load a scene, append poses, and save.

