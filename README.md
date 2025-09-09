sceneio
======

Columnar scene I/O for Rerun recordings (.rrd).

### Quick start: build a scene from a video

```python
scene = SceneIO(root="/scene")
cam = scene.load_mono_camera(video_path="video.mp4")
scene.save("out.rrd", app_id="sceneio_demo", spawn_viewer=False)
```

### Load and append to an existing .rrd

```python
scene = SceneIO(rrd_path="in.rrd")
print(scene.list_cameras())  # e.g. ['cam0']
cam = scene.get_camera("cam0")

cam.add_extrinsics(t_ns=..., translation=...)
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