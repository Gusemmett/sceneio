import pytest
import numpy as np
import pyarrow as pa
from unittest.mock import patch, MagicMock, mock_open
import tempfile
import os

from sceneio import rrd_io
from sceneio.models.tables import SceneTables, new_pinhole_table, new_extrinsics_table


class TestWriteRRD:
    """Test RRD writing functionality"""
    
    @patch('sceneio.rrd_io.rr')
    def test_write_rrd_basic(self, mock_rr):
        """Test basic RRD writing"""
        tables = SceneTables()
        
        rrd_io.write_rrd("test.rrd", tables, app_id="test", spawn_viewer=False)
        
        mock_rr.init.assert_called_once_with("test", spawn=False)
        mock_rr.save.assert_called_once_with("test.rrd")
        
    @patch('sceneio.rrd_io.rr')
    def test_write_rrd_with_spawn(self, mock_rr):
        """Test RRD writing with viewer spawn"""
        tables = SceneTables()
        
        rrd_io.write_rrd("test.rrd", tables, spawn_viewer=True)
        
        mock_rr.init.assert_called_once_with("sceneio", spawn=True)
        
    @patch('sceneio.rrd_io._log_asset_videos')
    @patch('sceneio.rrd_io.rr')
    def test_write_rrd_with_video_assets(self, mock_rr, mock_log_videos):
        """Test RRD writing with video assets"""
        from sceneio.models.tables import VIDEO_ASSETS_SCHEMA
        
        # Create table with correct schema types
        video_table = pa.table({
            "entity_path": pa.array(["test"], type=pa.string()),
            "video_path": pa.array(["test.mp4"], type=pa.string()),
            "blob_sha256": pa.array([None], type=pa.string()),
            "blob_bytes": pa.array([None], type=pa.large_binary()),
            "media_type": pa.array(["video/mp4"], type=pa.string()),
            "fps_hint": pa.array([30.0], type=pa.float32()),
            "duration_ns": pa.array([None], type=pa.int64())
        }, schema=VIDEO_ASSETS_SCHEMA)
        
        tables = SceneTables(video_assets=video_table)
        
        rrd_io.write_rrd("test.rrd", tables)
        
        mock_log_videos.assert_called_once_with(video_table)


class TestReadPinhole:
    """Test pinhole data reading"""
    
    @patch('sceneio.rrd_io.rr')
    def test_read_pinhole_valid_entity(self, mock_rr, sample_k_matrix):
        """Test reading pinhole data for valid entity"""
        # Mock the rerun dataframe interface
        mock_recording = MagicMock()
        mock_view = MagicMock()
        mock_reader = MagicMock()
        mock_table = MagicMock()
        
        mock_rr.dataframe.load_recording.return_value = mock_recording
        mock_recording.view.return_value = mock_view
        mock_view.select_static.return_value = mock_reader
        mock_reader.read_all.return_value = mock_table
        
        # Configure mock table data
        mock_table.num_rows = 1
        mock_column_k = MagicMock()
        mock_column_k.__getitem__.return_value.as_py.return_value = [sample_k_matrix]
        mock_column_res = MagicMock()  
        mock_column_res.__getitem__.return_value.as_py.return_value = [[640, 480]]
        mock_column_xyz = MagicMock()
        mock_column_xyz.__getitem__.return_value.as_py.return_value = ["RDF"]
        
        mock_table.column.side_effect = lambda name: {
            "/test:Pinhole:image_from_camera": mock_column_k,
            "/test:Pinhole:resolution": mock_column_res,
            "/test:Pinhole:camera_xyz": mock_column_xyz
        }[name]
        
        result = rrd_io.read_pinhole("test.rrd", "/test")
        
        assert result.num_rows == 1
        data = result.to_pylist()[0]
        assert data["entity_path"] == "/test"
        assert data["image_from_camera"] == sample_k_matrix
        assert data["resolution_u"] == 640
        assert data["resolution_v"] == 480
        assert data["camera_xyz"] == "RDF"
        
    @patch('sceneio.rrd_io.rr')
    def test_read_pinhole_empty_entity(self, mock_rr):
        """Test reading pinhole data for entity with no data"""
        mock_recording = MagicMock()
        mock_view = MagicMock() 
        mock_reader = MagicMock()
        mock_table = MagicMock()
        
        mock_rr.dataframe.load_recording.return_value = mock_recording
        mock_recording.view.return_value = mock_view
        mock_view.select_static.return_value = mock_reader
        mock_reader.read_all.return_value = mock_table
        mock_table.num_rows = 0
        
        result = rrd_io.read_pinhole("test.rrd", "/test")
        
        assert result.num_rows == 0
        assert result.schema == rrd_io.PINHOLE_SCHEMA


class TestDiscoverCameraEntities:
    """Test camera entity discovery"""
    
    @patch('sceneio.rrd_io.rr')
    def test_discover_camera_entities_basic(self, mock_rr):
        """Test basic camera entity discovery"""
        # Mock the recording and view
        mock_recording = MagicMock()
        mock_view = MagicMock()
        mock_schema = MagicMock()
        
        mock_rr.dataframe.load_recording.return_value = mock_recording
        mock_recording.view.return_value = mock_view
        mock_view.schema.return_value = mock_schema
        
        # Mock component columns with camera-related components
        mock_columns = [
            MagicMock(entity_path="/scene/cameras/left", component="Pinhole:image_from_camera"),
            MagicMock(entity_path="/scene/cameras/right", component="Pinhole:image_from_camera"), 
            MagicMock(entity_path="/scene/cameras/left/video", component="AssetVideo:blob"),
            MagicMock(entity_path="/scene/other", component="SomeOtherComponent"),
        ]
        
        mock_schema.component_columns.return_value = mock_columns
        
        result = rrd_io.discover_camera_entities("test.rrd")
        
        # Should find cameras and normalize video paths
        assert "/scene/cameras/left" in result
        assert "/scene/cameras/right" in result
        assert "/scene/other" not in result
        
    @patch('sceneio.rrd_io.rr')
    def test_discover_entities_path_normalization(self, mock_rr):
        """Test entity path normalization during discovery"""
        mock_recording = MagicMock()
        mock_view = MagicMock()
        mock_schema = MagicMock()
        
        mock_rr.dataframe.load_recording.return_value = mock_recording
        mock_recording.view.return_value = mock_view
        mock_view.schema.return_value = mock_schema
        
        # Mock component columns with /image and /video suffixes
        mock_columns = [
            MagicMock(entity_path="/scene/cameras/cam1/image", component="VideoFrameReference:timestamp"),
            MagicMock(entity_path="/scene/cameras/cam1/video", component="AssetVideo:blob"),
            MagicMock(entity_path="/scene/cameras/cam2", component="Pinhole:image_from_camera"),
        ]
        
        mock_schema.component_columns.return_value = mock_columns
        
        result = rrd_io.discover_camera_entities("test.rrd")
        
        # All should normalize to the camera parent paths
        assert "/scene/cameras/cam1" in result
        assert "/scene/cameras/cam2" in result
        assert len([x for x in result if "image" in x or "video" in x]) == 0


class TestUtilityFunctions:
    """Test utility functions"""
    
    def test_qpath_normalization(self):
        """Test entity path normalization"""
        assert rrd_io._qpath("test") == "/test"
        assert rrd_io._qpath("/test") == "/test"
        # Implementation only ensures one leading slash, doesn't normalize multiple
        assert rrd_io._qpath("///test") == "///test"
        
    def test_to_pylist_valid_column(self):
        """Test _to_pylist with valid column"""
        table = pa.table({"test_col": [1, 2, 3]})
        result = rrd_io._to_pylist(table, "test_col")
        assert result == [1, 2, 3]
        
    def test_to_pylist_with_cast(self):
        """Test _to_pylist with type casting"""
        table = pa.table({"test_col": [1.0, 2.0, 3.0]})
        result = rrd_io._to_pylist(table, "test_col", cast=int)
        assert result == [1, 2, 3]
        assert all(isinstance(x, int) for x in result)
        
    def test_to_pylist_missing_column(self):
        """Test _to_pylist with missing column"""
        table = pa.table({"other_col": [1, 2, 3]})
        
        with pytest.raises(KeyError, match="Missing required column"):
            rrd_io._to_pylist(table, "test_col")
            
    def test_opt_pylist_missing_column(self):
        """Test _opt_pylist with missing optional column"""
        table = pa.table({"other_col": [1, 2, 3]})
        result = rrd_io._opt_pylist(table, "test_col")
        assert result is None
        
    def test_opt_pylist_existing_column(self):
        """Test _opt_pylist with existing column"""
        table = pa.table({"test_col": [1, 2, 3]})
        result = rrd_io._opt_pylist(table, "test_col")
        assert result == [1, 2, 3]
        
    def test_unwrap_static(self):
        """Test _unwrap_static utility"""
        assert rrd_io._unwrap_static([42]) == 42
        assert rrd_io._unwrap_static([]) == []
        assert rrd_io._unwrap_static(42) == 42
        
    def test_ns_to_seconds_list(self):
        """Test nanosecond to seconds conversion"""
        ns_vals = [1000000000, 2000000000, 3000000000]  # 1s, 2s, 3s in ns
        result = rrd_io._ns_to_seconds_list(ns_vals)
        expected = [1.0, 2.0, 3.0]
        
        assert len(result) == len(expected)
        for r, e in zip(result, expected):
            assert abs(r - e) < 1e-9


class TestIntrinsicsProcessing:
    """Test camera intrinsics processing utilities"""
    
    def test_derive_fx_fy_cx_cy_row_major(self):
        """Test focal parameter derivation from row-major K matrix"""
        k9 = [800.0, 0.0, 320.0, 0.0, 600.0, 240.0, 0.0, 0.0, 1.0]
        fx, fy, cx, cy = rrd_io._derive_fx_fy_cx_cy(k9)
        
        assert fx == 800.0
        assert fy == 600.0
        assert cx == 320.0
        assert cy == 240.0
        
    def test_derive_fx_fy_cx_cy_invalid_length(self):
        """Test error for invalid K matrix length"""
        with pytest.raises(ValueError, match="9 elements"):
            rrd_io._derive_fx_fy_cx_cy([1, 2, 3])
            
    def test_normalize_vc_string(self):
        """Test view coordinate normalization from string"""
        assert rrd_io._normalize_vc("RDF") == "RDF"
        assert rrd_io._normalize_vc("rdf") == "RDF"  # Should uppercase
        assert rrd_io._normalize_vc("xyz") == "XYZ"
        assert rrd_io._normalize_vc("invalid") is None  # Invalid coordinate
        
    def test_normalize_vc_list_strings(self):
        """Test view coordinate normalization from string list"""
        result = rrd_io._normalize_vc(["Right", "Down", "Forward"])
        assert result == "RDF"
        
    def test_normalize_vc_list_ints(self):
        """Test view coordinate normalization from integer codes"""
        # Based on VC_CODE mapping: R=3, D=2, F=5
        result = rrd_io._normalize_vc([3, 2, 5])  
        assert result == "RDF"
        
    def test_normalize_vc_dict(self):
        """Test view coordinate normalization from dict"""
        vc_dict = {"x": "R", "y": "D", "z": "F"}
        result = rrd_io._normalize_vc(vc_dict)
        assert result == "RDF"
        
    def test_normalize_vc_invalid_input(self):
        """Test view coordinate normalization with invalid input"""
        assert rrd_io._normalize_vc(None) is None
        assert rrd_io._normalize_vc([1, 2]) is None  # Wrong length
        assert rrd_io._normalize_vc([99, 100, 101]) is None  # Invalid codes


class TestReadAllForEntities:
    """Test batch reading for multiple entities"""
    
    @patch('sceneio.rrd_io.read_pinhole')
    @patch('sceneio.rrd_io.read_extrinsics')
    @patch('sceneio.rrd_io.read_video_frames')
    @patch('sceneio.rrd_io.read_video_asset_meta')
    def test_read_all_for_entities(self, mock_read_assets, mock_read_frames, 
                                  mock_read_extr, mock_read_pinhole):
        """Test batch reading for multiple camera entities"""
        entities = ["/scene/cameras/left", "/scene/cameras/right"]
        
        # Mock individual readers to return empty tables
        mock_read_pinhole.return_value = new_pinhole_table()
        mock_read_extr.return_value = new_extrinsics_table()
        mock_read_frames.return_value = pa.table({
            "entity_path": pa.array([], type=pa.string()),
            "t_ns": pa.array([], type=pa.int64()),
            "video_ts_ns": pa.array([], type=pa.int64()),
            "source_video_entity_path": pa.array([], type=pa.string())
        })
        mock_read_assets.return_value = pa.table({
            "entity_path": pa.array([], type=pa.string()),
            "video_path": pa.array([], type=pa.string()),
            "blob_sha256": pa.array([], type=pa.string()),
            "blob_bytes": pa.array([], type=pa.large_binary()),
            "media_type": pa.array([], type=pa.string()),
            "fps_hint": pa.array([], type=pa.float32()),
            "duration_ns": pa.array([], type=pa.int64())
        })
        
        result = rrd_io.read_all_for_entities("test.rrd", entities)
        
        # Should have called each reader for each entity
        assert mock_read_pinhole.call_count == 2
        assert mock_read_extr.call_count == 2
        assert mock_read_frames.call_count == 2
        assert mock_read_assets.call_count == 2
        
        # Check call arguments
        mock_read_pinhole.assert_any_call("test.rrd", "/scene/cameras/left")
        mock_read_pinhole.assert_any_call("test.rrd", "/scene/cameras/right")
        
        # Result should be SceneTables
        assert isinstance(result, SceneTables)
        assert result.pinhole is not None
        assert result.extrinsics is not None
        assert result.video_frames is not None
        assert result.video_assets is not None


class TestAssetVideoLogging:
    """Test video asset logging functionality"""
    
    @patch('sceneio.rrd_io.rr')
    def test_log_asset_videos_with_path(self, mock_rr):
        """Test logging video assets with file paths"""
        table = pa.table({
            "entity_path": ["/scene/cameras/cam1"],
            "video_path": ["/path/to/video.mp4"],
            "blob_bytes": [None],
            "media_type": ["video/mp4"]
        })
        
        rrd_io._log_asset_videos(table)
        
        # Should call rr.log with AssetVideo
        mock_rr.log.assert_called_once()
        args, kwargs = mock_rr.log.call_args
        assert args[0] == "/scene/cameras/cam1"  # entity path
        assert "static" in kwargs and kwargs["static"] is True
        
    @patch('sceneio.rrd_io.rr')
    @patch('tempfile.gettempdir')
    @patch('builtins.open', new_callable=mock_open)
    def test_log_asset_videos_with_blob(self, mock_file, mock_tempdir, mock_rr):
        """Test logging video assets with blob data"""
        mock_tempdir.return_value = "/tmp"
        
        # Mock AssetVideo constructor to fail with blob/bytes, succeed with path
        mock_asset_video = MagicMock()
        
        def asset_video_side_effect(*args, **kwargs):
            if 'blob' in kwargs or 'bytes' in kwargs or 'data' in kwargs:
                raise TypeError("blob not supported")
            return mock_asset_video
            
        mock_rr.AssetVideo.side_effect = asset_video_side_effect
        
        blob_data = b"fake video data"
        table = pa.table({
            "entity_path": ["/scene/cameras/cam1"],
            "video_path": [None],
            "blob_bytes": [blob_data],
            "media_type": ["video/mp4"]
        })
        
        rrd_io._log_asset_videos(table)
        
        # Should have written temp file and used path fallback
        mock_file.assert_called()
        mock_rr.log.assert_called_once()


class TestPinholeLogging:
    """Test pinhole parameter logging"""
    
    @patch('sceneio.rrd_io.rr')
    def test_log_pinholes_basic(self, mock_rr, sample_k_matrix):
        """Test logging pinhole parameters"""
        table = pa.table({
            "entity_path": ["/scene/cameras/cam1"],
            "image_from_camera": [sample_k_matrix],
            "resolution_u": [640],
            "resolution_v": [480], 
            "camera_xyz": ["RDF"]
        })
        
        with patch('sceneio.rrd_io._vc_constant') as mock_vc:
            mock_vc.return_value = MagicMock()  # Mock view coordinate constant
            
            rrd_io._log_pinholes(table)
            
            mock_rr.log.assert_called_once()
            args, kwargs = mock_rr.log.call_args
            assert args[0] == "/scene/cameras/cam1"
            assert "static" in kwargs and kwargs["static"] is True