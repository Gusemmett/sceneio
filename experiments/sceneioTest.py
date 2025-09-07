from pathlib import Path

from sceneio import SceneIO, CameraIntrinsics


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    # Prefer existing sample video in repo; keep it simple.
    candidate_paths = [
        project_root / "testData" / "left.mp4",
        project_root / "testData" / "p1.mp4",
    ]

    video_path = next((p for p in candidate_paths if p.exists()), None)
    if video_path is None:
        print("No sample video found in testData/. Aborting simple test.")
        return

    scene = SceneIO()  # or SceneIO(rrd_file="scene1.rrd")
    intr = CameraIntrinsics((1000.0, 1000.0), (960.0, 540.0), 1280, 720)

    entity_path = scene.add_mono_camera("s", video_path,intr)
    entity_path = scene.add_stereo_camera("s", video_path,intr,video_path,intr)
    # info = scene.get_camera_info(entity_path)

    print(f"Added mono camera at entity path: {entity_path}")
    print("Camera info:")
    # print(info)

    scene.save("test.rrd")


if __name__ == "__main__":
    main()



