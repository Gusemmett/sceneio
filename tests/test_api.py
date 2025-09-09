import pytest
import numpy as np
import pyarrow as pa
from unittest.mock import patch, MagicMock
from sceneio.api import SceneIO, Camera, EntityLayout, _normalize_intrinsics, _to_int_list
from sceneio.models.tables import CAMERAS_SCHEMA, PINHOLE_SCHEMA


class TestEntityLayout:
    """Test EntityLayout enum"""
    
    def test_entity_layout_values(self):
        """Test EntityLayout enum has expected values"""
        assert EntityLayout.COLOCATED.value == 1
        assert EntityLayout.SPLIT_IMAGE_NODE.value == 2


class TestIntrinsicsNormalization:
    """Test intrinsic parameter normalization utilities"""
    
    def test_normalize_intrinsics_with_k_matrix(self, sample_k_matrix):
        """Test normalizing intrinsics from K matrix"""
        width, height = 640, 480
        k_result, w_result, h_result = _normalize_intrinsics(
            sample_k_matrix, None, None, None, None, width, height
        )
        
        assert k_result == sample_k_matrix
        assert w_result == width
        assert h_result == height
        
    def test_normalize_intrinsics_with_focal_params(self, sample_focal_params):
        """Test normalizing intrinsics from focal parameters"""
        k_result, w_result, h_result = _normalize_intrinsics(
            None, 
            sample_focal_params["fx"],
            sample_focal_params["fy"],
            sample_focal_params["cx"], 
            sample_focal_params["cy"],
            sample_focal_params["width"],
            sample_focal_params["height"]
        )
        
        expected_k = [800.0, 0.0, 320.0, 0.0, 800.0, 240.0, 0.0, 0.0, 1.0]
        assert k_result == expected_k
        assert w_result == sample_focal_params["width"]
        assert h_result == sample_focal_params["height"]
        
    def test_normalize_intrinsics_invalid_k_length(self):
        """Test error when K matrix has wrong length"""
        with pytest.raises(ValueError, match="K must have length 9"):
            _normalize_intrinsics([1, 2, 3, 4], None, None, None, None, 640, 480)
            
    def test_normalize_intrinsics_k_missing_dimensions(self, sample_k_matrix):
        """Test error when K provided but dimensions missing"""
        with pytest.raises(ValueError, match="width and height are required"):
            _normalize_intrinsics(sample_k_matrix, None, None, None, None, None, 480)
            
    def test_normalize_intrinsics_focal_params_incomplete(self):
        """Test error when focal parameters are incomplete"""
        with pytest.raises(ValueError, match="Provide either K"):
            _normalize_intrinsics(None, 800.0, None, 320.0, 240.0, 640, 480)


class TestToIntList:
    """Test _to_int_list utility function"""
    
    def test_to_int_list_from_numpy_int(self):
        """Test conversion from numpy integer array"""
        arr = np.array([1, 2, 3, 4], dtype=np.int64)
        result = _to_int_list(arr)
        assert result == [1, 2, 3, 4]
        assert all(isinstance(x, int) for x in result)
        
    def test_to_int_list_from_numpy_float(self):
        """Test conversion from numpy float array"""
        arr = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        result = _to_int_list(arr)
        assert result == [1, 2, 3, 4]
        assert all(isinstance(x, int) for x in result)
        
    def test_to_int_list_from_python_list(self):
        """Test conversion from Python list"""
        lst = [1, 2, 3, 4]
        result = _to_int_list(lst)
        assert result == [1, 2, 3, 4]
        assert all(isinstance(x, int) for x in result)


class TestSceneIOInitialization:
    """Test SceneIO class initialization"""
    
    def test_sceneio_init_empty(self):
        """Test SceneIO initialization with default parameters"""
        scene = SceneIO()
        
        assert scene.root == "/scene"
        assert scene.layout == EntityLayout.COLOCATED
        assert scene.cameras.num_rows == 0
        assert scene.pinhole.num_rows == 0
        assert scene.video_assets.num_rows == 0
        assert scene.video_frames.num_rows == 0
        assert scene.extrinsics.num_rows == 0
        assert len(scene._cam_index) == 0
        
    def test_sceneio_init_custom_params(self):
        """Test SceneIO initialization with custom parameters"""
        scene = SceneIO(root="/custom", layout=EntityLayout.SPLIT_IMAGE_NODE)
        
        assert scene.root == "/custom"
        assert scene.layout == EntityLayout.SPLIT_IMAGE_NODE
        
    def test_sceneio_init_root_normalization(self):
        """Test root path normalization (trailing slash removal)"""
        scene = SceneIO(root="/scene/")
        assert scene.root == "/scene"
        
    @patch('sceneio.api.rrd_io.read_all_for_entities')
    @patch('sceneio.api.SceneIO._discover_entities')
    def test_sceneio_init_from_rrd(self, mock_discover, mock_read_all):
        """Test SceneIO initialization from RRD file"""
        # Setup mocks
        mock_discover.return_value = ["/scene/cameras/cam1"]
        mock_tables = MagicMock()
        mock_read_all.return_value = mock_tables
        
        with patch.object(SceneIO, '_ingest_tables') as mock_ingest, \
             patch.object(SceneIO, '_rebuild_index') as mock_rebuild:
            
            scene = SceneIO(rrd_path="test.rrd")
            
            mock_discover.assert_called_once_with("test.rrd")
            mock_read_all.assert_called_once_with("test.rrd", ["/scene/cameras/cam1"])
            mock_ingest.assert_called_once_with(mock_tables)
            mock_rebuild.assert_called_once()


class TestCameraPathGeneration:
    """Test camera path generation logic"""
    
    def test_paths_for_colocated_layout(self):
        """Test path generation for colocated layout"""
        scene = SceneIO(root="/scene", layout=EntityLayout.COLOCATED)
        paths = scene._paths_for("test_cam")
        
        expected = {
            "cam": "/scene/cameras/test_cam",
            "img": "/scene/cameras/test_cam", 
            "asset": "/scene/cameras/test_cam/video"
        }
        assert paths == expected
        
        # Test caching
        paths2 = scene._paths_for("test_cam")
        assert paths2 is paths  # Should return same object
        
    def test_paths_for_split_layout(self):
        """Test path generation for split image node layout"""
        scene = SceneIO(root="/scene", layout=EntityLayout.SPLIT_IMAGE_NODE)
        paths = scene._paths_for("test_cam")
        
        expected = {
            "cam": "/scene/cameras/test_cam",
            "img": "/scene/cameras/test_cam/image",
            "asset": "/scene/cameras/test_cam/video"
        }
        assert paths == expected


class TestMonoCameraLoading:
    """Test mono camera loading functionality"""
    
    def test_load_mono_camera_with_k_matrix(self, sample_k_matrix):
        """Test loading mono camera with K matrix"""
        scene = SceneIO()
        
        camera = scene.load_mono_camera(
            cam_id="test_cam",
            K=sample_k_matrix,
            width=640,
            height=480,
            label="Test Camera"
        )
        
        # Check camera object
        assert isinstance(camera, Camera)
        assert camera.scene is scene
        assert camera.cam_id == "test_cam"
        assert camera.label == "Test Camera"
        assert camera.entity_path == "/scene/cameras/test_cam"
        
        # Check cameras table
        assert scene.cameras.num_rows == 1
        cameras_data = scene.cameras.to_pylist()[0]
        assert cameras_data["entity_path"] == "/scene/cameras/test_cam"
        assert cameras_data["camera_id"] == "test_cam"
        assert cameras_data["label"] == "Test Camera"
        assert cameras_data["stereo_group"] is None
        
        # Check pinhole table
        assert scene.pinhole.num_rows == 1
        pinhole_data = scene.pinhole.to_pylist()[0]
        assert pinhole_data["entity_path"] == "/scene/cameras/test_cam"
        assert pinhole_data["image_from_camera"] == sample_k_matrix
        assert pinhole_data["resolution_u"] == 640
        assert pinhole_data["resolution_v"] == 480
        
    def test_load_mono_camera_with_focal_params(self, sample_focal_params):
        """Test loading mono camera with focal parameters"""
        scene = SceneIO()
        
        camera = scene.load_mono_camera(
            cam_id="test_cam",
            fx=sample_focal_params["fx"],
            fy=sample_focal_params["fy"],
            cx=sample_focal_params["cx"],
            cy=sample_focal_params["cy"],
            width=sample_focal_params["width"],
            height=sample_focal_params["height"],
            camera_xyz="RDF"
        )
        
        assert isinstance(camera, Camera)
        
        # Check pinhole table has correct derived K matrix
        assert scene.pinhole.num_rows == 1
        pinhole_data = scene.pinhole.to_pylist()[0]
        expected_k = [800.0, 0.0, 320.0, 0.0, 800.0, 240.0, 0.0, 0.0, 1.0]
        assert pinhole_data["image_from_camera"] == expected_k
        assert pinhole_data["camera_xyz"] == "RDF"
        
    def test_load_mono_camera_with_video(self, sample_k_matrix):
        """Test loading mono camera with video path"""
        scene = SceneIO()
        video_path = "/path/to/video.mp4"
        
        camera = scene.load_mono_camera(
            cam_id="test_cam",
            video_path=video_path,
            K=sample_k_matrix,
            width=640,
            height=480
        )
        
        # Check video assets table
        assert scene.video_assets.num_rows == 1
        asset_data = scene.video_assets.to_pylist()[0]
        assert asset_data["entity_path"] == "/scene/cameras/test_cam"
        assert asset_data["video_path"] == video_path
        assert asset_data["media_type"] == "video/mp4"
        
    def test_load_mono_camera_with_frames(self, sample_k_matrix, sample_global_timestamps, sample_video_timestamps):
        """Test loading mono camera with frame references"""
        scene = SceneIO()
        
        camera = scene.load_mono_camera(
            cam_id="test_cam", 
            K=sample_k_matrix,
            width=640,
            height=480,
            t_ns=sample_global_timestamps,
            video_ts_ns=sample_video_timestamps
        )
        
        # Check video frames table
        assert scene.video_frames.num_rows == len(sample_global_timestamps)
        frames_data = scene.video_frames.to_pylist()
        
        for i, frame in enumerate(frames_data):
            assert frame["entity_path"] == "/scene/cameras/test_cam"
            assert frame["t_ns"] == sample_global_timestamps[i]
            assert frame["video_ts_ns"] == sample_video_timestamps[i]
            
    def test_load_mono_camera_mismatched_frame_lengths(self, sample_k_matrix):
        """Test error when t_ns and video_ts_ns have different lengths"""
        scene = SceneIO()
        
        with pytest.raises(ValueError, match="equal length"):
            scene.load_mono_camera(
                cam_id="test_cam",
                K=sample_k_matrix,
                width=640,
                height=480,
                t_ns=[1, 2, 3],
                video_ts_ns=[1, 2]  # Different length
            )


class TestStereoCameraLoading:
    """Test stereo camera loading functionality"""
    
    def test_load_stereo_camera(self, sample_k_matrix):
        """Test loading stereo camera pair"""
        scene = SceneIO()
        
        left_params = {"K": sample_k_matrix, "width": 640, "height": 480, "label": "Left"}
        right_params = {"K": sample_k_matrix, "width": 640, "height": 480, "label": "Right"}
        
        cam_left, cam_right = scene.load_stereo_camera(
            left_id="left",
            right_id="right", 
            left=left_params,
            right=right_params,
            rig_label="test_rig"
        )
        
        # Check camera objects
        assert isinstance(cam_left, Camera)
        assert isinstance(cam_right, Camera)
        assert cam_left.cam_id == "left"
        assert cam_right.cam_id == "right"
        
        # Both cameras should be in tables
        assert scene.cameras.num_rows == 2
        assert scene.pinhole.num_rows == 2
        
        # Check stereo group annotation
        cameras_data = scene.cameras.to_pylist()
        for cam_data in cameras_data:
            assert cam_data["stereo_group"] == "test_rig"


class TestExtrinsicsHandling:
    """Test extrinsics data handling"""
    
    def test_add_extrinsics_translation_only(self, sample_global_timestamps, sample_translation):
        """Test adding extrinsics with translation only"""
        scene = SceneIO()
        
        # Need a camera first
        scene.load_mono_camera(cam_id="test_cam", K=[800,0,320,0,800,240,0,0,1], width=640, height=480)
        
        scene.add_extrinsics(
            "test_cam",
            sample_global_timestamps,
            translation=sample_translation
        )
        
        assert scene.extrinsics.num_rows == len(sample_global_timestamps)
        extrinsics_data = scene.extrinsics.to_pylist()
        
        for i, ext in enumerate(extrinsics_data):
            assert ext["entity_path"] == "/scene/cameras/test_cam"
            assert ext["t_ns"] == sample_global_timestamps[i]
            # Use approximate comparison for float32 precision
            assert len(ext["translation"]) == len(sample_translation[i])
            for j in range(len(ext["translation"])):
                assert abs(ext["translation"][j] - sample_translation[i][j]) < 1e-6
            assert ext["quaternion"] is None
            assert ext["rotation_mat"] is None
            
    def test_add_extrinsics_all_components(self, sample_global_timestamps, sample_translation, 
                                          sample_quaternion_xyzw, sample_rotation_matrices):
        """Test adding extrinsics with all components"""
        scene = SceneIO()
        scene.load_mono_camera(cam_id="test_cam", K=[800,0,320,0,800,240,0,0,1], width=640, height=480)
        
        relations = ["ChildFromParent"] * len(sample_global_timestamps)
        
        scene.add_extrinsics(
            "test_cam",
            sample_global_timestamps,
            translation=sample_translation,
            quaternion_xyzw=sample_quaternion_xyzw,
            rotation_mat=sample_rotation_matrices,
            relation=relations
        )
        
        assert scene.extrinsics.num_rows == len(sample_global_timestamps)
        extrinsics_data = scene.extrinsics.to_pylist()
        
        for i, ext in enumerate(extrinsics_data):
            # Use approximate comparison for float32 precision
            assert len(ext["translation"]) == len(sample_translation[i])
            for j in range(len(ext["translation"])):
                assert abs(ext["translation"][j] - sample_translation[i][j]) < 1e-6
            
            assert len(ext["quaternion"]) == len(sample_quaternion_xyzw[i])
            for j in range(len(ext["quaternion"])):
                assert abs(ext["quaternion"][j] - sample_quaternion_xyzw[i][j]) < 1e-6
                
            assert len(ext["rotation_mat"]) == len(sample_rotation_matrices[i])
            for j in range(len(ext["rotation_mat"])):
                assert abs(ext["rotation_mat"][j] - sample_rotation_matrices[i][j]) < 1e-6
                
            assert ext["relation"] == relations[i]
            
    def test_add_extrinsics_length_mismatch(self):
        """Test error when component lengths don't match timestamp length"""
        scene = SceneIO()
        scene.load_mono_camera(cam_id="test_cam", K=[800,0,320,0,800,240,0,0,1], width=640, height=480)
        
        with pytest.raises(ValueError, match="component lengths must match"):
            scene.add_extrinsics(
                "test_cam",
                [1, 2, 3],  # 3 timestamps
                translation=[[1,2,3], [4,5,6]]  # 2 translations - mismatch
            )


class TestCameraClass:
    """Test Camera class functionality"""
    
    def test_camera_init(self):
        """Test Camera initialization"""
        scene = SceneIO()
        camera = Camera(scene, "test_cam", "/scene/cameras/test_cam", label="Test")
        
        assert camera.scene is scene
        assert camera.cam_id == "test_cam"
        assert camera.entity_path == "/scene/cameras/test_cam"
        assert camera.label == "Test"
        
    def test_camera_add_extrinsics_convenience(self, sample_global_timestamps, sample_translation):
        """Test Camera.add_extrinsics convenience method"""
        scene = SceneIO()
        camera = Camera(scene, "test_cam", "/scene/cameras/test_cam")
        
        # Mock the scene method
        with patch.object(scene, 'add_extrinsics') as mock_add:
            camera.add_extrinsics(
                sample_global_timestamps,
                translation=sample_translation
            )
            
            mock_add.assert_called_once_with(
                "test_cam",
                sample_global_timestamps, 
                translation=sample_translation,
                quaternion_xyzw=None,
                rotation_mat=None,
                relation=None
            )
            
    def test_camera_set_video_convenience(self):
        """Test Camera.set_video convenience method"""
        scene = SceneIO()
        camera = Camera(scene, "test_cam", "/scene/cameras/test_cam")
        
        with patch.object(scene, 'set_video') as mock_set:
            camera.set_video("/path/video.mp4", fps_hint=30.0)
            
            mock_set.assert_called_once_with(
                "test_cam", 
                "/path/video.mp4",
                media_type="video/mp4",
                fps_hint=30.0,
                duration_ns=None
            )


class TestSceneIOQueries:
    """Test SceneIO query methods"""
    
    def test_list_cameras_empty(self):
        """Test listing cameras from empty scene"""
        scene = SceneIO()
        cameras = scene.list_cameras()
        assert cameras == []
        
    def test_list_cameras_populated(self, sample_k_matrix):
        """Test listing cameras from populated scene"""
        scene = SceneIO()
        scene.load_mono_camera(cam_id="cam1", K=sample_k_matrix, width=640, height=480)
        scene.load_mono_camera(cam_id="cam2", K=sample_k_matrix, width=640, height=480)
        
        cameras = scene.list_cameras()
        assert sorted(cameras) == ["cam1", "cam2"]
        
    def test_get_camera_existing(self, sample_k_matrix):
        """Test getting existing camera"""
        scene = SceneIO()
        original_camera = scene.load_mono_camera(
            cam_id="test_cam",
            K=sample_k_matrix, 
            width=640,
            height=480,
            label="Test Label"
        )
        
        retrieved_camera = scene.get_camera("test_cam")
        assert isinstance(retrieved_camera, Camera)
        assert retrieved_camera.cam_id == "test_cam"
        assert retrieved_camera.label == "Test Label"
        assert retrieved_camera.entity_path == original_camera.entity_path
        
    def test_get_camera_unknown(self):
        """Test getting unknown camera creates new camera handle"""
        scene = SceneIO()
        
        # Implementation creates camera handle even for unknown cameras
        camera = scene.get_camera("unknown_cam")
        assert isinstance(camera, Camera)
        assert camera.cam_id == "unknown_cam"
        assert camera.entity_path == "/scene/cameras/unknown_cam"
        assert camera.label is None  # No label since camera doesn't exist in tables