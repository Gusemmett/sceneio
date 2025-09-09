import pytest
import numpy as np
import pyarrow as pa
import tempfile
import json
import os
from pathlib import Path
from typing import List, Dict


@pytest.fixture
def sample_k_matrix():
    """Sample camera intrinsic matrix (3x3 row-major flattened)"""
    return [800.0, 0.0, 320.0, 0.0, 800.0, 240.0, 0.0, 0.0, 1.0]


@pytest.fixture
def sample_focal_params():
    """Sample focal length parameters"""
    return {
        "fx": 800.0,
        "fy": 800.0, 
        "cx": 320.0,
        "cy": 240.0,
        "width": 640,
        "height": 480
    }


@pytest.fixture
def sample_video_timestamps():
    """Sample video timestamp sequence (ns)"""
    return np.array([0, 33333333, 66666666, 100000000], dtype=np.int64)


@pytest.fixture
def sample_global_timestamps():
    """Sample global timeline timestamps (ns)"""
    base = 1000000000000000000  # Some base timestamp
    return np.array([base, base + 33333333, base + 66666666, base + 100000000], dtype=np.int64)


@pytest.fixture
def sample_translation():
    """Sample 3D translation vectors"""
    return [[1.0, 2.0, 3.0], [1.1, 2.1, 3.1], [1.2, 2.2, 3.2], [1.3, 2.3, 3.3]]


@pytest.fixture
def sample_quaternion_xyzw():
    """Sample quaternions in xyzw format"""
    return [[0.0, 0.0, 0.0, 1.0], [0.1, 0.0, 0.0, 0.995], [0.0, 0.1, 0.0, 0.995], [0.0, 0.0, 0.1, 0.995]]


@pytest.fixture
def sample_rotation_matrices():
    """Sample 3x3 rotation matrices (flattened)"""
    return [
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],  # Identity
        [0.995, -0.1, 0.0, 0.1, 0.995, 0.0, 0.0, 0.0, 1.0],  # Small rotation
        [0.98, -0.2, 0.0, 0.2, 0.98, 0.0, 0.0, 0.0, 1.0],     # Larger rotation
        [0.96, -0.28, 0.0, 0.28, 0.96, 0.0, 0.0, 0.0, 1.0]    # Fourth rotation
    ]


@pytest.fixture
def temp_rrd_file(tmp_path):
    """Temporary RRD file path"""
    return tmp_path / "test.rrd"


@pytest.fixture
def sample_calibration_data():
    """Sample stereo calibration data"""
    return {
        "left": {
            "intrinsics": [[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]],
            "width": 640,
            "height": 480,
            "socket": "left_camera"
        },
        "right": {
            "intrinsics": [[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]],
            "width": 640,
            "height": 480,
            "socket": "right_camera"
        }
    }


@pytest.fixture
def sample_stereo_directory(tmp_path, sample_calibration_data):
    """Create a minimal stereo directory structure for testing"""
    stereo_dir = tmp_path / "stereo_data"
    stereo_dir.mkdir()
    
    # Create calibration.json
    calib_file = stereo_dir / "calibration.json"
    with open(calib_file, 'w') as f:
        json.dump(sample_calibration_data, f)
    
    # Create CSV files with timestamp data
    left_csv = stereo_dir / "left.csv"
    right_csv = stereo_dir / "right.csv"
    
    csv_data = "ts_ns,frame_idx\n1000000000,0\n1033333333,1\n1066666666,2\n1100000000,3\n"
    
    with open(left_csv, 'w') as f:
        f.write(csv_data)
    with open(right_csv, 'w') as f:
        f.write(csv_data)
    
    # Create dummy video files (empty for now)
    (stereo_dir / "left.mp4").touch()
    (stereo_dir / "right.mp4").touch()
    
    return stereo_dir


@pytest.fixture
def sample_poses_csv_data():
    """Sample pose data for CSV"""
    return [
        {"frame_idx": 0, "tx": 0.0, "ty": 0.0, "tz": 0.0, "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0},
        {"frame_idx": 1, "tx": 0.1, "ty": 0.0, "tz": 0.0, "qx": 0.0, "qy": 0.0, "qz": 0.05, "qw": 0.9987},
        {"frame_idx": 2, "tx": 0.2, "ty": 0.0, "tz": 0.0, "qx": 0.0, "qy": 0.0, "qz": 0.1, "qw": 0.995},
    ]


@pytest.fixture
def mock_rerun_asset_video(monkeypatch):
    """Mock rerun AssetVideo for testing"""
    class MockAssetVideo:
        def __init__(self, path=None, **kwargs):
            self.path = path
            
        def read_frame_timestamps_nanos(self):
            # Return some dummy timestamps
            return [0, 33333333, 66666666, 100000000]
    
    monkeypatch.setattr("rerun.AssetVideo", MockAssetVideo)
    return MockAssetVideo


@pytest.fixture  
def sample_entity_paths():
    """Sample camera entity paths"""
    return ["/scene/cameras/left", "/scene/cameras/right", "/scene/cameras/center"]