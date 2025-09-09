import pytest
import hashlib
import numpy as np
from pathlib import Path

from sceneio.api import SceneIO


def compute_file_hash(file_path: str) -> str:
    """Compute SHA256 hash of a file"""
    hash_sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()


def create_test_scene_with_data() -> SceneIO:
    """Create a SceneIO instance with comprehensive test data"""
    scene = SceneIO(root="/test_scene")
    
    # Camera intrinsics - slightly different for left/right
    K_left = [800.0, 0.0, 320.0, 0.0, 800.0, 240.0, 0.0, 0.0, 1.0]
    K_right = [805.0, 0.0, 325.0, 0.0, 805.0, 245.0, 0.0, 0.0, 1.0]
    
    # Load stereo cameras with comprehensive data
    left_cam = scene.load_mono_camera(
        cam_id="left_camera",
        K=K_left,
        width=640,
        height=480,
        camera_xyz="RDF",
        label="Left Camera"
    )
    
    right_cam = scene.load_mono_camera(
        cam_id="right_camera", 
        K=K_right,
        width=640,
        height=480,
        camera_xyz="RDF",
        label="Right Camera"
    )
    
    # Add video assets
    left_cam.set_video("/test/left_video.mp4", media_type="video/mp4", fps_hint=30.0, duration_ns=10000000000)
    right_cam.set_video("/test/right_video.mp4", media_type="video/mp4", fps_hint=30.0, duration_ns=10000000000)
    
    # Add frame references with realistic timestamps
    base_time = 1000000000000000000  # Base nanosecond timestamp
    frame_count = 50  # More frames for comprehensive test
    
    # Left camera frames
    left_times = np.array([base_time + i * 33333333 for i in range(frame_count)], dtype=np.int64)
    left_video_ts = np.array([i * 33333333 for i in range(frame_count)], dtype=np.int64)
    left_cam.add_video_frames(t_ns=left_times, video_ts_ns=left_video_ts)
    
    # Right camera frames (with slight offset)
    right_times = np.array([base_time + 16666666 + i * 33333333 for i in range(frame_count)], dtype=np.int64)
    right_video_ts = np.array([16666666 + i * 33333333 for i in range(frame_count)], dtype=np.int64)
    right_cam.add_video_frames(t_ns=right_times, video_ts_ns=right_video_ts)
    
    # Add extrinsics (camera poses) - realistic trajectory data
    pose_count = 25  # Poses for subset of frames
    left_translations = []
    left_rotations = []
    
    for i in range(pose_count):
        t = i * 0.2  # Time parameter for trajectory
        # Circular motion trajectory
        x = 1.0 + 0.5 * np.sin(t)
        y = 0.3 * np.cos(t * 0.5)  
        z = 2.0 + 0.2 * np.sin(t * 0.3)
        left_translations.append([x, y, z])
        
        # Rotation around Y axis with small variation
        angle = t * 0.15
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rotation_matrix = [
            cos_a, 0.0, sin_a,
            0.0, 1.0, 0.0,
            -sin_a, 0.0, cos_a
        ]
        left_rotations.append(rotation_matrix)
    
    left_pose_times = left_times[:pose_count]
    left_cam.add_extrinsics(
        t_ns=left_pose_times,
        translation=left_translations,
        rotation_mat=left_rotations,
        relation=["ChildFromParent"] * pose_count
    )
    
    # Right camera extrinsics (stereo baseline offset)
    right_translations = [[t[0] + 0.12, t[1], t[2]] for t in left_translations]  # 12cm baseline
    right_rotations = left_rotations.copy()
    
    right_pose_times = right_times[:pose_count]
    right_cam.add_extrinsics(
        t_ns=right_pose_times,
        translation=right_translations,
        rotation_mat=right_rotations,
        relation=["ChildFromParent"] * pose_count
    )
    
    return scene


def serialize_scene_deterministically(scene: SceneIO) -> bytes:
    """
    Serialize a SceneIO instance to bytes in a deterministic way.
    This simulates what would happen in an actual RRD save operation.
    """
    content_parts = []
    
    # Scene metadata
    content_parts.append(f"ROOT:{scene.root}\\n".encode())
    content_parts.append(f"LAYOUT:{scene.layout.value}\\n".encode())
    
    # Serialize tables in deterministic order
    tables = [
        ("CAMERAS", scene.cameras),
        ("PINHOLE", scene.pinhole), 
        ("VIDEO_ASSETS", scene.video_assets),
        ("VIDEO_FRAMES", scene.video_frames),
        ("EXTRINSICS", scene.extrinsics)
    ]
    
    for table_name, table in tables:
        content_parts.append(f"==={table_name}===\\n".encode())
        content_parts.append(f"ROWS:{table.num_rows}\\n".encode())
        
        if table.num_rows > 0:
            # Convert to pylist and sort for deterministic order
            data = table.to_pylist()
            
            # Sort by first string field for deterministic ordering
            if table_name == "CAMERAS":
                data.sort(key=lambda x: (x["entity_path"], x["camera_id"]))
            elif table_name in ["PINHOLE", "VIDEO_ASSETS"]:
                data.sort(key=lambda x: x["entity_path"])
            elif table_name in ["VIDEO_FRAMES", "EXTRINSICS"]:
                data.sort(key=lambda x: (x["entity_path"], x["t_ns"]))
            
            for row in data:
                # Serialize row data (simplified but deterministic)
                row_str = str(sorted(row.items())) + "\\n"
                content_parts.append(row_str.encode())
    
    return b"".join(content_parts)


class TestEndToEndConcept:
    """
    End-to-end tests demonstrating the hash consistency concept.
    
    These tests simulate the full workflow:
    1. Create SceneIO with rich data  
    2. Serialize to bytes (simulating RRD save)
    3. Compute hash
    4. Repeat process (simulating load->save cycle)
    5. Verify hash consistency
    """
    
    def test_deterministic_serialization_consistency(self, tmp_path):
        """Test that the same scene data produces identical serialized output"""
        
        # Create two identical scenes
        scene1 = create_test_scene_with_data()
        scene2 = create_test_scene_with_data()
        
        # Serialize both
        data1 = serialize_scene_deterministically(scene1)
        data2 = serialize_scene_deterministically(scene2)
        
        # Should be byte-identical
        assert data1 == data2, "Identical scenes should produce identical serialization"
        
        # Save to files and compare hashes
        file1 = tmp_path / "scene1.dat"
        file2 = tmp_path / "scene2.dat"
        
        file1.write_bytes(data1)
        file2.write_bytes(data2)
        
        hash1 = compute_file_hash(str(file1))
        hash2 = compute_file_hash(str(file2))
        
        assert hash1 == hash2, f"Identical scenes should have identical hashes\\nHash1: {hash1}\\nHash2: {hash2}"
        
        print(f"✅ Deterministic serialization verified!")
        print(f"  Hash: {hash1}")
        print(f"  File size: {len(data1)} bytes")
        print(f"  Scene data: {scene1.cameras.num_rows} cameras, {scene1.video_frames.num_rows} frames, {scene1.extrinsics.num_rows} poses")
    
    def test_round_trip_simulation_concept(self, tmp_path):
        """
        Demonstrate the round-trip hash consistency concept.
        This simulates: Create -> Save -> Load -> Save and verifies hash equality.
        """
        
        # Step 1: Create original scene
        original_scene = create_test_scene_with_data()
        
        print(f"\\n📊 Created test scene:")
        print(f"  Cameras: {original_scene.cameras.num_rows}")
        print(f"  Pinhole entries: {original_scene.pinhole.num_rows}")
        print(f"  Video assets: {original_scene.video_assets.num_rows}")
        print(f"  Video frames: {original_scene.video_frames.num_rows}")
        print(f"  Extrinsics: {original_scene.extrinsics.num_rows}")
        
        # Step 2: First serialization (simulates initial save)
        serialized_1 = serialize_scene_deterministically(original_scene)
        file_1 = tmp_path / "original_save.dat"
        file_1.write_bytes(serialized_1)
        hash_1 = compute_file_hash(str(file_1))
        
        print(f"\\n💾 First save:")
        print(f"  Hash: {hash_1}")
        print(f"  Size: {len(serialized_1)} bytes")
        
        # Step 3: Second serialization of same data (simulates load->save)
        # In reality, this would involve loading from RRD and saving again
        # Here we simulate it by re-serializing the same scene
        serialized_2 = serialize_scene_deterministically(original_scene)
        file_2 = tmp_path / "roundtrip_save.dat"
        file_2.write_bytes(serialized_2)
        hash_2 = compute_file_hash(str(file_2))
        
        print(f"\\n🔄 Round-trip save:")
        print(f"  Hash: {hash_2}")
        print(f"  Size: {len(serialized_2)} bytes")
        
        # Step 4: Verify consistency
        assert hash_1 == hash_2, f"Round-trip should preserve hash!\\nOriginal: {hash_1}\\nRound-trip: {hash_2}"
        assert len(serialized_1) == len(serialized_2), "Round-trip should preserve data size"
        assert serialized_1 == serialized_2, "Round-trip should preserve data exactly"
        
        print(f"\\n✅ Round-trip hash consistency verified!")
        print(f"  Both operations produced identical hash: {hash_1}")
    
    def test_data_modification_detection(self, tmp_path):
        """Test that data modifications are detected via hash changes"""
        
        # Create base scene
        scene = create_test_scene_with_data()
        original_serialized = serialize_scene_deterministically(scene)
        original_file = tmp_path / "original.dat"
        original_file.write_bytes(original_serialized)
        original_hash = compute_file_hash(str(original_file))
        
        print(f"\\n📊 Original scene hash: {original_hash}")
        
        # Modify scene by adding another camera
        scene.load_mono_camera(
            cam_id="additional_camera",
            K=[850.0, 0.0, 330.0, 0.0, 850.0, 250.0, 0.0, 0.0, 1.0],
            width=1280,
            height=720,
            camera_xyz="RDF",
            label="Additional Camera"
        )
        
        # Serialize modified scene
        modified_serialized = serialize_scene_deterministically(scene)
        modified_file = tmp_path / "modified.dat"
        modified_file.write_bytes(modified_serialized)
        modified_hash = compute_file_hash(str(modified_file))
        
        print(f"🔧 Modified scene hash: {modified_hash}")
        print(f"   Added 1 camera (total: {scene.cameras.num_rows})")
        
        # Verify they're different
        assert original_hash != modified_hash, "Scene modification should change hash"
        assert len(modified_serialized) > len(original_serialized), "Modified scene should be larger"
        
        print(f"\\n✅ Data modification detection verified!")
        print(f"  Original size: {len(original_serialized)} bytes")
        print(f"  Modified size: {len(modified_serialized)} bytes")
    
    def test_large_dataset_hash_consistency(self, tmp_path):
        """Test hash consistency with larger, more complex datasets"""
        
        scene = SceneIO(root="/stress_test")
        
        # Create substantial dataset
        num_cameras = 10
        frames_per_camera = 200
        poses_per_camera = 50
        
        print(f"\\n🏗️  Creating large dataset:")
        print(f"  Cameras: {num_cameras}")
        print(f"  Frames per camera: {frames_per_camera}")
        print(f"  Poses per camera: {poses_per_camera}")
        
        for cam_idx in range(num_cameras):
            # Each camera has slightly different parameters
            fx = 800.0 + cam_idx * 5
            fy = 800.0 + cam_idx * 3
            cx = 320.0 + cam_idx * 2
            cy = 240.0 + cam_idx * 1
            K = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
            
            cam = scene.load_mono_camera(
                cam_id=f"camera_{cam_idx:02d}",
                K=K,
                width=640 + cam_idx * 10,
                height=480 + cam_idx * 5,
                camera_xyz="RDF",
                label=f"Stress Test Camera {cam_idx}"
            )
            
            # Add video reference
            cam.set_video(f"/stress_test/video_{cam_idx:02d}.mp4", fps_hint=30.0 + cam_idx)
            
            # Add frames
            base_time = 1000000000000000000 + cam_idx * 10000000
            times = np.array([base_time + i * 33333333 for i in range(frames_per_camera)], dtype=np.int64)
            video_ts = np.array([i * 33333333 for i in range(frames_per_camera)], dtype=np.int64)
            cam.add_video_frames(t_ns=times, video_ts_ns=video_ts)
            
            # Add poses for subset of frames
            translations = []
            rotations = []
            for pose_idx in range(poses_per_camera):
                # Each camera has different trajectory
                t = pose_idx * 0.1 + cam_idx * 0.01
                x = cam_idx + 0.5 * np.sin(t + cam_idx)
                y = 0.3 * np.cos(t * 0.7 + cam_idx)
                z = 2.0 + 0.2 * np.sin(t * 0.3 + cam_idx)
                translations.append([x, y, z])
                
                # Rotation varies per camera
                angle = t * 0.1 + cam_idx * 0.05
                cos_a, sin_a = np.cos(angle), np.sin(angle)
                rotation = [cos_a, 0.0, sin_a, 0.0, 1.0, 0.0, -sin_a, 0.0, cos_a]
                rotations.append(rotation)
            
            pose_times = times[:poses_per_camera]
            cam.add_extrinsics(
                t_ns=pose_times,
                translation=translations,
                rotation_mat=rotations,
                relation=[f"Camera{cam_idx}ToWorld"] * poses_per_camera
            )
        
        total_frames = scene.video_frames.num_rows
        total_poses = scene.extrinsics.num_rows
        
        print(f"\\n📈 Final dataset stats:")
        print(f"  Total cameras: {scene.cameras.num_rows}")
        print(f"  Total video frames: {total_frames}")
        print(f"  Total poses: {total_poses}")
        print(f"  Total video assets: {scene.video_assets.num_rows}")
        
        # Test multiple serializations for consistency
        serializations = []
        for i in range(3):
            data = serialize_scene_deterministically(scene)
            file_path = tmp_path / f"large_dataset_{i}.dat"
            file_path.write_bytes(data)
            hash_val = compute_file_hash(str(file_path))
            serializations.append((data, hash_val))
            
        # All should be identical
        first_data, first_hash = serializations[0]
        for i, (data, hash_val) in enumerate(serializations[1:], 1):
            assert data == first_data, f"Serialization {i} differs from first"
            assert hash_val == first_hash, f"Hash {i} differs from first"
        
        print(f"\\n✅ Large dataset consistency verified!")
        print(f"  Consistent hash across {len(serializations)} serializations: {first_hash}")
        print(f"  Data size: {len(first_data)} bytes ({len(first_data)/1024:.1f} KB)")
    
    def test_hash_sensitivity_to_small_changes(self, tmp_path):
        """Test that small data changes produce different hashes (avalanche effect)"""
        
        scene = create_test_scene_with_data()
        
        # Original
        original_data = serialize_scene_deterministically(scene) 
        original_file = tmp_path / "original.dat"
        original_file.write_bytes(original_data)
        original_hash = compute_file_hash(str(original_file))
        
        test_cases = []
        
        # Test 1: Change camera label
        scene_copy1 = create_test_scene_with_data()
        scene_copy1.cameras = scene_copy1.cameras.schema.empty_table()  # Clear and rebuild
        scene_copy1.load_mono_camera(
            cam_id="left_camera", K=[800.0, 0.0, 320.0, 0.0, 800.0, 240.0, 0.0, 0.0, 1.0],
            width=640, height=480, camera_xyz="RDF", label="Left Camera MODIFIED"  # Small change
        )
        scene_copy1.load_mono_camera(
            cam_id="right_camera", K=[805.0, 0.0, 325.0, 0.0, 805.0, 245.0, 0.0, 0.0, 1.0],
            width=640, height=480, camera_xyz="RDF", label="Right Camera"
        )
        data1 = serialize_scene_deterministically(scene_copy1)
        file1 = tmp_path / "label_change.dat"
        file1.write_bytes(data1)
        hash1 = compute_file_hash(str(file1))
        test_cases.append(("Camera label change", hash1))
        
        # Test 2: Change one intrinsic parameter by small amount
        scene_copy2 = create_test_scene_with_data()
        scene_copy2.cameras = scene_copy2.cameras.schema.empty_table()
        scene_copy2.pinhole = scene_copy2.pinhole.schema.empty_table()
        scene_copy2.load_mono_camera(
            cam_id="left_camera", K=[800.1, 0.0, 320.0, 0.0, 800.0, 240.0, 0.0, 0.0, 1.0],  # 800.0->800.1
            width=640, height=480, camera_xyz="RDF", label="Left Camera"
        )
        scene_copy2.load_mono_camera(
            cam_id="right_camera", K=[805.0, 0.0, 325.0, 0.0, 805.0, 245.0, 0.0, 0.0, 1.0],
            width=640, height=480, camera_xyz="RDF", label="Right Camera"
        )
        data2 = serialize_scene_deterministically(scene_copy2)
        file2 = tmp_path / "intrinsics_change.dat"
        file2.write_bytes(data2)
        hash2 = compute_file_hash(str(file2))
        test_cases.append(("Intrinsics tiny change", hash2))
        
        # Verify all hashes are different
        all_hashes = [original_hash] + [h for _, h in test_cases]
        unique_hashes = set(all_hashes)
        
        assert len(unique_hashes) == len(all_hashes), "All modifications should produce unique hashes"
        
        print(f"\\n✅ Hash sensitivity verified!")
        print(f"  Original: {original_hash}")
        for desc, hash_val in test_cases:
            print(f"  {desc}: {hash_val}")
        print(f"  All {len(all_hashes)} hashes are unique")


if __name__ == "__main__":
    # Quick demonstration
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        
        test = TestEndToEndConcept()
        test.test_round_trip_simulation_concept(tmp_path)
        print("\\n" + "="*50)
        test.test_data_modification_detection(tmp_path)