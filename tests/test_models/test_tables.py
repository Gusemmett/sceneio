import pytest
import pyarrow as pa
import numpy as np
from sceneio.models.tables import (
    SCHEMA_VERSION,
    CAMERAS_SCHEMA,
    PINHOLE_SCHEMA, 
    VIDEO_ASSETS_SCHEMA,
    VIDEO_FRAMES_SCHEMA,
    EXTRINSICS_SCHEMA,
    SceneTables,
    new_cameras_table,
    new_pinhole_table,
    new_video_assets_table,
    new_video_frames_table,
    new_extrinsics_table,
    ensure_schema,
    concat_like,
    append_rows,
)


class TestSchemas:
    """Test schema definitions and their properties"""
    
    def test_schema_version_exists(self):
        """Test that schema version is defined"""
        assert SCHEMA_VERSION is not None
        assert isinstance(SCHEMA_VERSION, str)
        assert "." in SCHEMA_VERSION  # Should be semantic version
    
    def test_cameras_schema_structure(self):
        """Test cameras schema has expected fields"""
        field_names = [f.name for f in CAMERAS_SCHEMA]
        expected_fields = ["entity_path", "camera_id", "label", "stereo_group"]
        
        assert field_names == expected_fields
        assert CAMERAS_SCHEMA.field("entity_path").type == pa.string()
        assert CAMERAS_SCHEMA.field("camera_id").type == pa.string()
        assert CAMERAS_SCHEMA.field("label").type == pa.string()
        assert CAMERAS_SCHEMA.field("stereo_group").type == pa.string()
        
        # Check nullable constraints
        assert not CAMERAS_SCHEMA.field("entity_path").nullable
        assert not CAMERAS_SCHEMA.field("camera_id").nullable
        assert CAMERAS_SCHEMA.field("label").nullable
        assert CAMERAS_SCHEMA.field("stereo_group").nullable
    
    def test_pinhole_schema_structure(self):
        """Test pinhole schema has expected fields"""
        field_names = [f.name for f in PINHOLE_SCHEMA]
        expected_fields = ["entity_path", "image_from_camera", "resolution_u", "resolution_v", "camera_xyz"]
        
        assert field_names == expected_fields
        assert PINHOLE_SCHEMA.field("entity_path").type == pa.string()
        assert PINHOLE_SCHEMA.field("image_from_camera").type == pa.list_(pa.float32(), 9)
        assert PINHOLE_SCHEMA.field("resolution_u").type == pa.uint32()
        assert PINHOLE_SCHEMA.field("resolution_v").type == pa.uint32()
        assert PINHOLE_SCHEMA.field("camera_xyz").type == pa.string()
    
    def test_video_assets_schema_structure(self):
        """Test video assets schema has expected fields"""
        field_names = [f.name for f in VIDEO_ASSETS_SCHEMA]
        expected_fields = ["entity_path", "video_path", "blob_sha256", "blob_bytes", "media_type", "fps_hint", "duration_ns"]
        
        assert field_names == expected_fields
        assert VIDEO_ASSETS_SCHEMA.field("entity_path").type == pa.string()
        assert VIDEO_ASSETS_SCHEMA.field("video_path").type == pa.string()
        assert VIDEO_ASSETS_SCHEMA.field("blob_sha256").type == pa.string()
        assert VIDEO_ASSETS_SCHEMA.field("blob_bytes").type == pa.large_binary()
        assert VIDEO_ASSETS_SCHEMA.field("media_type").type == pa.string()
        assert VIDEO_ASSETS_SCHEMA.field("fps_hint").type == pa.float32()
        assert VIDEO_ASSETS_SCHEMA.field("duration_ns").type == pa.int64()
    
    def test_video_frames_schema_structure(self):
        """Test video frames schema has expected fields"""
        field_names = [f.name for f in VIDEO_FRAMES_SCHEMA]
        expected_fields = ["entity_path", "t_ns", "video_ts_ns", "source_video_entity_path"]
        
        assert field_names == expected_fields
        assert VIDEO_FRAMES_SCHEMA.field("entity_path").type == pa.string()
        assert VIDEO_FRAMES_SCHEMA.field("t_ns").type == pa.int64()
        assert VIDEO_FRAMES_SCHEMA.field("video_ts_ns").type == pa.int64()
        assert VIDEO_FRAMES_SCHEMA.field("source_video_entity_path").type == pa.string()
    
    def test_extrinsics_schema_structure(self):
        """Test extrinsics schema has expected fields"""
        field_names = [f.name for f in EXTRINSICS_SCHEMA]
        expected_fields = ["entity_path", "t_ns", "translation", "quaternion", "rotation_mat", "relation"]
        
        assert field_names == expected_fields
        assert EXTRINSICS_SCHEMA.field("entity_path").type == pa.string()
        assert EXTRINSICS_SCHEMA.field("t_ns").type == pa.int64()
        assert EXTRINSICS_SCHEMA.field("translation").type == pa.list_(pa.float32(), 3)
        assert EXTRINSICS_SCHEMA.field("quaternion").type == pa.list_(pa.float32(), 4)
        assert EXTRINSICS_SCHEMA.field("rotation_mat").type == pa.list_(pa.float32(), 9)
        assert EXTRINSICS_SCHEMA.field("relation").type == pa.string()


class TestTableBuilders:
    """Test empty table creation functions"""
    
    def test_new_cameras_table(self):
        """Test new cameras table creation"""
        table = new_cameras_table()
        assert isinstance(table, pa.Table)
        assert table.num_rows == 0
        assert table.schema == CAMERAS_SCHEMA
        
    def test_new_pinhole_table(self):
        """Test new pinhole table creation"""
        table = new_pinhole_table()
        assert isinstance(table, pa.Table)
        assert table.num_rows == 0
        assert table.schema == PINHOLE_SCHEMA
        
    def test_new_video_assets_table(self):
        """Test new video assets table creation"""
        table = new_video_assets_table()
        assert isinstance(table, pa.Table)
        assert table.num_rows == 0
        assert table.schema == VIDEO_ASSETS_SCHEMA
        
    def test_new_video_frames_table(self):
        """Test new video frames table creation"""
        table = new_video_frames_table()
        assert isinstance(table, pa.Table)
        assert table.num_rows == 0
        assert table.schema == VIDEO_FRAMES_SCHEMA
        
    def test_new_extrinsics_table(self):
        """Test new extrinsics table creation"""
        table = new_extrinsics_table()
        assert isinstance(table, pa.Table)
        assert table.num_rows == 0
        assert table.schema == EXTRINSICS_SCHEMA


class TestSchemaValidation:
    """Test schema validation functions"""
    
    def test_ensure_schema_valid(self):
        """Test ensure_schema with matching schema"""
        table = new_cameras_table()
        # Should not raise
        ensure_schema(table, CAMERAS_SCHEMA, "test")
        
    def test_ensure_schema_invalid(self):
        """Test ensure_schema with mismatched schema"""
        table = new_cameras_table()
        wrong_schema = pa.schema([pa.field("wrong_field", pa.string())])
        
        with pytest.raises(TypeError, match="schema mismatch"):
            ensure_schema(table, wrong_schema, "test")


class TestTableOperations:
    """Test table manipulation operations"""
    
    def test_append_rows_empty_list(self):
        """Test appending empty row list"""
        table = new_cameras_table()
        result = append_rows(table, [])
        assert result.num_rows == 0
        assert result.schema == table.schema
        
    def test_append_rows_valid_data(self, sample_entity_paths):
        """Test appending valid rows to cameras table"""
        table = new_cameras_table()
        rows = [
            {
                "entity_path": sample_entity_paths[0],
                "camera_id": "left",
                "label": "Left Camera", 
                "stereo_group": "main_rig"
            },
            {
                "entity_path": sample_entity_paths[1], 
                "camera_id": "right",
                "label": "Right Camera",
                "stereo_group": "main_rig"
            }
        ]
        
        result = append_rows(table, rows)
        assert result.num_rows == 2
        assert result.schema == CAMERAS_SCHEMA
        
        # Check data integrity
        entity_paths = result["entity_path"].to_pylist()
        camera_ids = result["camera_id"].to_pylist()
        assert entity_paths == [sample_entity_paths[0], sample_entity_paths[1]]
        assert camera_ids == ["left", "right"]
        
    def test_append_rows_pinhole_data(self, sample_k_matrix, sample_entity_paths):
        """Test appending pinhole data"""
        table = new_pinhole_table()
        rows = [
            {
                "entity_path": sample_entity_paths[0],
                "image_from_camera": sample_k_matrix,
                "resolution_u": 640,
                "resolution_v": 480,
                "camera_xyz": "RDF"
            }
        ]
        
        result = append_rows(table, rows)
        assert result.num_rows == 1
        
        # Check K matrix integrity
        k_retrieved = result["image_from_camera"][0].as_py()
        assert k_retrieved == sample_k_matrix
        
    def test_concat_like_empty_tables(self):
        """Test concatenating empty tables"""
        table1 = new_cameras_table()
        table2 = new_cameras_table()
        
        result = concat_like([table1, table2], CAMERAS_SCHEMA)
        assert result.num_rows == 0
        assert result.schema == CAMERAS_SCHEMA
        
    def test_concat_like_mixed_tables(self, sample_entity_paths):
        """Test concatenating mix of empty and populated tables"""
        empty_table = new_cameras_table()
        
        populated_table = append_rows(empty_table, [{
            "entity_path": sample_entity_paths[0],
            "camera_id": "test_cam",
            "label": None,
            "stereo_group": None
        }])
        
        result = concat_like([empty_table, populated_table, empty_table], CAMERAS_SCHEMA)
        assert result.num_rows == 1
        assert result["camera_id"][0].as_py() == "test_cam"
        
    def test_concat_like_multiple_populated_tables(self, sample_entity_paths):
        """Test concatenating multiple populated tables"""
        table1 = append_rows(new_cameras_table(), [{
            "entity_path": sample_entity_paths[0],
            "camera_id": "cam1", 
            "label": None,
            "stereo_group": None
        }])
        
        table2 = append_rows(new_cameras_table(), [{
            "entity_path": sample_entity_paths[1],
            "camera_id": "cam2",
            "label": None, 
            "stereo_group": None
        }])
        
        result = concat_like([table1, table2], CAMERAS_SCHEMA)
        assert result.num_rows == 2
        
        camera_ids = result["camera_id"].to_pylist()
        assert "cam1" in camera_ids
        assert "cam2" in camera_ids


class TestSceneTables:
    """Test SceneTables dataclass container"""
    
    def test_scene_tables_init_empty(self):
        """Test SceneTables initialization with defaults"""
        tables = SceneTables()
        assert tables.cameras is None
        assert tables.pinhole is None
        assert tables.video_assets is None
        assert tables.video_frames is None
        assert tables.extrinsics is None
        
    def test_scene_tables_init_with_data(self):
        """Test SceneTables initialization with table data"""
        cameras = new_cameras_table()
        pinhole = new_pinhole_table()
        
        tables = SceneTables(cameras=cameras, pinhole=pinhole)
        assert tables.cameras is cameras
        assert tables.pinhole is pinhole
        assert tables.video_assets is None
        assert tables.video_frames is None
        assert tables.extrinsics is None