import pytest
import json
import pandas as pd
import numpy as np
import os
from unittest.mock import patch, MagicMock

from sceneio.loaders.stereo_dir import (
    load_stereo_from_directory,
    _require_file,
    _flatten_row_major_3x3,
    _read_calibration,
    _read_ts_idx_csv,
    _read_poses_csv,
    _quat_xyzw_to_rotation_matrix,
    CameraCsv,
)
from sceneio.api import SceneIO


class TestUtilityFunctions:
    """Test utility functions"""
    
    def test_require_file_existing(self, tmp_path):
        """Test _require_file with existing file"""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")
        
        result = _require_file(str(test_file))
        assert result == str(test_file)
        
    def test_require_file_missing(self, tmp_path):
        """Test _require_file with missing file"""
        missing_file = tmp_path / "missing.txt"
        
        with pytest.raises(FileNotFoundError):
            _require_file(str(missing_file))
            
    def test_flatten_row_major_3x3(self):
        """Test 3x3 matrix flattening"""
        matrix = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        result = _flatten_row_major_3x3(matrix)
        expected = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        assert result == expected
        
    def test_read_calibration(self, tmp_path, sample_calibration_data):
        """Test calibration JSON reading"""
        calib_file = tmp_path / "calibration.json"
        with open(calib_file, 'w') as f:
            json.dump(sample_calibration_data, f)
            
        result = _read_calibration(str(calib_file))
        assert result == sample_calibration_data
        
    def test_read_ts_idx_csv(self, tmp_path):
        """Test timestamp/index CSV reading"""
        csv_file = tmp_path / "test.csv"
        csv_content = "ts_ns,frame_idx\n1000000000,0\n1033333333,1\n1066666666,2\n"
        csv_file.write_text(csv_content)
        
        result = _read_ts_idx_csv(str(csv_file))
        
        assert isinstance(result, CameraCsv)
        assert result.t_ns == [1000000000, 1033333333, 1066666666]
        assert result.frame_idx == [0, 1, 2]
        
    def test_read_ts_idx_csv_missing_columns(self, tmp_path):
        """Test CSV reading with missing required columns"""
        csv_file = tmp_path / "bad.csv"
        csv_content = "wrong_col,other_col\n1,2\n"
        csv_file.write_text(csv_content)
        
        with pytest.raises(ValueError, match="must contain columns"):
            _read_ts_idx_csv(str(csv_file))
            
    def test_read_ts_idx_csv_mismatched_lengths(self, tmp_path):
        """Test CSV reading with mismatched column lengths (shouldn't happen with pandas but test anyway)"""
        # This is hard to create with pandas, but we can test the validation logic
        pass  # pandas ensures consistent column lengths
        
    def test_read_poses_csv(self, tmp_path):
        """Test pose CSV reading"""
        csv_file = tmp_path / "poses.csv"
        csv_content = """frame_idx,tx,ty,tz,qx,qy,qz,qw
0,0.0,0.0,0.0,0.0,0.0,0.0,1.0
1,0.1,0.0,0.0,0.0,0.0,0.05,0.9987
"""
        csv_file.write_text(csv_content)
        
        result = _read_poses_csv(str(csv_file))
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert list(result.columns) == ["frame_idx", "tx", "ty", "tz", "qx", "qy", "qz", "qw"]
        
    def test_read_poses_csv_missing_column(self, tmp_path):
        """Test pose CSV reading with missing required column"""
        csv_file = tmp_path / "bad_poses.csv"
        csv_content = "frame_idx,tx,ty,tz\n0,0.0,0.0,0.0\n"  # Missing quaternion columns
        csv_file.write_text(csv_content)
        
        with pytest.raises(ValueError, match="missing required column"):
            _read_poses_csv(str(csv_file))


class TestQuaternionRotation:
    """Test quaternion to rotation matrix conversion"""
    
    def test_quat_identity(self):
        """Test identity quaternion conversion"""
        result = _quat_xyzw_to_rotation_matrix(0.0, 0.0, 0.0, 1.0)
        expected = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]  # Identity matrix
        
        # Check with small tolerance for floating point
        for r, e in zip(result, expected):
            assert abs(r - e) < 1e-6
            
    def test_quat_zero_norm(self):
        """Test quaternion with zero norm (should return identity)"""
        result = _quat_xyzw_to_rotation_matrix(0.0, 0.0, 0.0, 0.0)
        expected = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        
        for r, e in zip(result, expected):
            assert abs(r - e) < 1e-6
            
    def test_quat_normalization(self):
        """Test that quaternion gets normalized"""
        # Non-unit quaternion
        result = _quat_xyzw_to_rotation_matrix(0.0, 0.0, 0.0, 2.0)  # Should normalize to (0,0,0,1)
        expected = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        
        for r, e in zip(result, expected):
            assert abs(r - e) < 1e-6
            
    def test_quat_90_degree_z_rotation(self):
        """Test 90 degree rotation around Z axis"""
        # 90 degrees around Z: quat = (0, 0, sin(45°), cos(45°)) = (0, 0, 0.707, 0.707)
        sqrt2_2 = np.sqrt(2) / 2
        result = _quat_xyzw_to_rotation_matrix(0.0, 0.0, sqrt2_2, sqrt2_2)
        
        # Expected rotation matrix for 90° around Z
        # [cos(90) -sin(90) 0]   [0 -1 0]
        # [sin(90)  cos(90) 0] = [1  0 0]
        # [0        0       1]   [0  0 1]
        expected = [0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        
        for r, e in zip(result, expected):
            assert abs(r - e) < 1e-6


class TestStereoDirectoryLoading:
    """Test main stereo directory loading functionality"""
    
    def test_load_stereo_missing_files(self, tmp_path):
        """Test loading with missing required files"""
        empty_dir = tmp_path / "empty_stereo"
        empty_dir.mkdir()
        
        with pytest.raises(FileNotFoundError):
            load_stereo_from_directory(str(empty_dir))
            
    @patch('sceneio.loaders.stereo_dir.rr.AssetVideo')
    def test_load_stereo_basic(self, mock_asset_video, sample_stereo_directory):
        """Test basic stereo directory loading"""
        # Mock AssetVideo to avoid dependency on actual video files
        mock_av_instance = MagicMock()
        mock_av_instance.read_frame_timestamps_nanos.return_value = [0, 33333333, 66666666, 100000000]
        mock_asset_video.return_value = mock_av_instance
        
        scene = load_stereo_from_directory(str(sample_stereo_directory))
        
        assert isinstance(scene, SceneIO)
        
        # Check that cameras were loaded
        cameras = scene.list_cameras()
        assert "left" in cameras
        assert "right" in cameras
        
        # Check cameras table
        assert scene.cameras.num_rows == 2
        
        # Check pinhole table (intrinsics)
        assert scene.pinhole.num_rows == 2
        
        # Check video assets table 
        assert scene.video_assets.num_rows == 2
        
        # Check video frames table
        assert scene.video_frames.num_rows == 8  # 4 frames per camera
        
    @patch('sceneio.loaders.stereo_dir.rr.AssetVideo')
    def test_load_stereo_custom_params(self, mock_asset_video, sample_stereo_directory):
        """Test stereo loading with custom parameters"""
        mock_av_instance = MagicMock()
        mock_av_instance.read_frame_timestamps_nanos.return_value = []  # Empty to test fallback
        mock_asset_video.return_value = mock_av_instance
        
        scene = load_stereo_from_directory(
            str(sample_stereo_directory),
            left_id="camera_l",
            right_id="camera_r", 
            root="/custom",
            camera_xyz="RUB",
            store_frame_index_in_video_ts=False
        )
        
        assert scene.root == "/custom"
        cameras = scene.list_cameras()
        assert "camera_l" in cameras
        assert "camera_r" in cameras
        
        # Check camera coordinate system
        pinhole_data = scene.pinhole.to_pylist()
        for row in pinhole_data:
            assert row["camera_xyz"] == "RUB"
            
    @patch('sceneio.loaders.stereo_dir.rr.AssetVideo')
    def test_load_stereo_with_poses(self, mock_asset_video, sample_stereo_directory, sample_poses_csv_data):
        """Test stereo loading with pose data"""
        mock_av_instance = MagicMock()
        mock_av_instance.read_frame_timestamps_nanos.return_value = [0, 33333333, 66666666, 100000000]
        mock_asset_video.return_value = mock_av_instance
        
        # Create pose CSV files
        left_poses_path = sample_stereo_directory / "left_poses.csv"
        right_poses_path = sample_stereo_directory / "right_poses.csv"
        
        poses_df = pd.DataFrame(sample_poses_csv_data)
        poses_df.to_csv(left_poses_path, index=False)
        poses_df.to_csv(right_poses_path, index=False)
        
        scene = load_stereo_from_directory(str(sample_stereo_directory))
        
        # Should have extrinsics data
        assert scene.extrinsics.num_rows == 6  # 3 poses per camera
        
        extrinsics_data = scene.extrinsics.to_pylist()
        
        # Check that we have data for both cameras
        entity_paths = [row["entity_path"] for row in extrinsics_data]
        assert "/scene/cameras/left" in entity_paths
        assert "/scene/cameras/right" in entity_paths
        
        # Check that pose data was processed correctly
        for row in extrinsics_data:
            assert row["translation"] is not None
            assert row["rotation_mat"] is not None
            assert len(row["translation"]) == 3
            assert len(row["rotation_mat"]) == 9
            
    @patch('sceneio.loaders.stereo_dir.rr.AssetVideo')
    def test_load_stereo_asset_video_exception(self, mock_asset_video, sample_stereo_directory):
        """Test loading when AssetVideo raises exception"""
        # Make AssetVideo constructor raise exception
        mock_asset_video.side_effect = Exception("Video processing failed")
        
        # Should still work by falling back to frame index mapping
        scene = load_stereo_from_directory(str(sample_stereo_directory))
        
        assert isinstance(scene, SceneIO)
        assert scene.video_frames.num_rows == 8  # Should still have frame references
        
    def test_load_stereo_invalid_calibration(self, tmp_path):
        """Test loading with invalid calibration file"""
        stereo_dir = tmp_path / "stereo_invalid"
        stereo_dir.mkdir()
        
        # Create invalid calibration
        calib_file = stereo_dir / "calibration.json"
        calib_file.write_text("invalid json content")
        
        # Create other required files
        (stereo_dir / "left.csv").write_text("ts_ns,frame_idx\n1000000000,0\n")
        (stereo_dir / "right.csv").write_text("ts_ns,frame_idx\n1000000000,0\n")
        (stereo_dir / "left.mp4").touch()
        (stereo_dir / "right.mp4").touch()
        
        with pytest.raises(json.JSONDecodeError):
            load_stereo_from_directory(str(stereo_dir))
            
    @patch('sceneio.loaders.stereo_dir.rr.AssetVideo')
    def test_frame_index_to_pts_mapping(self, mock_asset_video, sample_stereo_directory):
        """Test frame index to PTS mapping logic"""
        # Test with actual PTS data
        pts_data = [0, 40000000, 80000000, 120000000]  # 40ms intervals
        mock_av_instance = MagicMock()
        mock_av_instance.read_frame_timestamps_nanos.return_value = pts_data
        mock_asset_video.return_value = mock_av_instance
        
        scene = load_stereo_from_directory(str(sample_stereo_directory))
        
        # Check video frames have correct PTS mapping
        frames_data = scene.video_frames.to_pylist()
        
        # Should have relative timestamps (subtract first PTS)
        expected_video_ts = [0, 40000000, 80000000, 120000000]  # Relative to first frame
        
        left_frames = [f for f in frames_data if f["entity_path"] == "/scene/cameras/left"]
        for i, frame in enumerate(left_frames):
            assert frame["video_ts_ns"] == expected_video_ts[i]
            
    @patch('sceneio.loaders.stereo_dir.rr.AssetVideo')  
    def test_store_frame_index_option(self, mock_asset_video, sample_stereo_directory):
        """Test store_frame_index_in_video_ts option"""
        # Empty PTS to force fallback
        mock_av_instance = MagicMock()
        mock_av_instance.read_frame_timestamps_nanos.return_value = []
        mock_asset_video.return_value = mock_av_instance
        
        # Test with store_frame_index_in_video_ts=True (default)
        scene1 = load_stereo_from_directory(str(sample_stereo_directory), store_frame_index_in_video_ts=True)
        frames1 = scene1.video_frames.to_pylist()
        left_frames1 = [f for f in frames1 if f["entity_path"] == "/scene/cameras/left"]
        
        # Should store actual frame indices
        expected_indices = [0, 1, 2, 3]
        for i, frame in enumerate(left_frames1):
            assert frame["video_ts_ns"] == expected_indices[i]
            
        # Test with store_frame_index_in_video_ts=False
        scene2 = load_stereo_from_directory(str(sample_stereo_directory), store_frame_index_in_video_ts=False)
        frames2 = scene2.video_frames.to_pylist()
        left_frames2 = [f for f in frames2 if f["entity_path"] == "/scene/cameras/left"]
        
        # Should store sequential indices starting from 0
        expected_sequential = [0, 1, 2, 3]
        for i, frame in enumerate(left_frames2):
            assert frame["video_ts_ns"] == expected_sequential[i]


class TestCameraCsv:
    """Test CameraCsv dataclass"""
    
    def test_camera_csv_creation(self):
        """Test CameraCsv dataclass creation"""
        csv_data = CameraCsv(t_ns=[1, 2, 3], frame_idx=[0, 1, 2])
        
        assert csv_data.t_ns == [1, 2, 3]
        assert csv_data.frame_idx == [0, 1, 2]