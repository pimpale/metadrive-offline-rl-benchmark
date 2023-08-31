from waymo_protos import keypoint_pb2 as _keypoint_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Label(_message.Message):
    __slots__ = ["box", "metadata", "type", "id", "detection_difficulty_level", "tracking_difficulty_level", "num_lidar_points_in_box", "num_top_lidar_points_in_box", "laser_keypoints", "camera_keypoints", "association", "most_visible_camera_name", "camera_synced_box"]
    class Type(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
        TYPE_UNKNOWN: _ClassVar[Label.Type]
        TYPE_VEHICLE: _ClassVar[Label.Type]
        TYPE_PEDESTRIAN: _ClassVar[Label.Type]
        TYPE_SIGN: _ClassVar[Label.Type]
        TYPE_CYCLIST: _ClassVar[Label.Type]
    TYPE_UNKNOWN: Label.Type
    TYPE_VEHICLE: Label.Type
    TYPE_PEDESTRIAN: Label.Type
    TYPE_SIGN: Label.Type
    TYPE_CYCLIST: Label.Type
    class DifficultyLevel(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
        UNKNOWN: _ClassVar[Label.DifficultyLevel]
        LEVEL_1: _ClassVar[Label.DifficultyLevel]
        LEVEL_2: _ClassVar[Label.DifficultyLevel]
    UNKNOWN: Label.DifficultyLevel
    LEVEL_1: Label.DifficultyLevel
    LEVEL_2: Label.DifficultyLevel
    class Box(_message.Message):
        __slots__ = ["center_x", "center_y", "center_z", "length", "width", "height", "heading"]
        class Type(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
            __slots__ = []
            TYPE_UNKNOWN: _ClassVar[Label.Box.Type]
            TYPE_3D: _ClassVar[Label.Box.Type]
            TYPE_2D: _ClassVar[Label.Box.Type]
            TYPE_AA_2D: _ClassVar[Label.Box.Type]
        TYPE_UNKNOWN: Label.Box.Type
        TYPE_3D: Label.Box.Type
        TYPE_2D: Label.Box.Type
        TYPE_AA_2D: Label.Box.Type
        CENTER_X_FIELD_NUMBER: _ClassVar[int]
        CENTER_Y_FIELD_NUMBER: _ClassVar[int]
        CENTER_Z_FIELD_NUMBER: _ClassVar[int]
        LENGTH_FIELD_NUMBER: _ClassVar[int]
        WIDTH_FIELD_NUMBER: _ClassVar[int]
        HEIGHT_FIELD_NUMBER: _ClassVar[int]
        HEADING_FIELD_NUMBER: _ClassVar[int]
        center_x: float
        center_y: float
        center_z: float
        length: float
        width: float
        height: float
        heading: float
        def __init__(self, center_x: _Optional[float] = ..., center_y: _Optional[float] = ..., center_z: _Optional[float] = ..., length: _Optional[float] = ..., width: _Optional[float] = ..., height: _Optional[float] = ..., heading: _Optional[float] = ...) -> None: ...
    class Metadata(_message.Message):
        __slots__ = ["speed_x", "speed_y", "speed_z", "accel_x", "accel_y", "accel_z"]
        SPEED_X_FIELD_NUMBER: _ClassVar[int]
        SPEED_Y_FIELD_NUMBER: _ClassVar[int]
        SPEED_Z_FIELD_NUMBER: _ClassVar[int]
        ACCEL_X_FIELD_NUMBER: _ClassVar[int]
        ACCEL_Y_FIELD_NUMBER: _ClassVar[int]
        ACCEL_Z_FIELD_NUMBER: _ClassVar[int]
        speed_x: float
        speed_y: float
        speed_z: float
        accel_x: float
        accel_y: float
        accel_z: float
        def __init__(self, speed_x: _Optional[float] = ..., speed_y: _Optional[float] = ..., speed_z: _Optional[float] = ..., accel_x: _Optional[float] = ..., accel_y: _Optional[float] = ..., accel_z: _Optional[float] = ...) -> None: ...
    class Association(_message.Message):
        __slots__ = ["laser_object_id"]
        LASER_OBJECT_ID_FIELD_NUMBER: _ClassVar[int]
        laser_object_id: str
        def __init__(self, laser_object_id: _Optional[str] = ...) -> None: ...
    BOX_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    ID_FIELD_NUMBER: _ClassVar[int]
    DETECTION_DIFFICULTY_LEVEL_FIELD_NUMBER: _ClassVar[int]
    TRACKING_DIFFICULTY_LEVEL_FIELD_NUMBER: _ClassVar[int]
    NUM_LIDAR_POINTS_IN_BOX_FIELD_NUMBER: _ClassVar[int]
    NUM_TOP_LIDAR_POINTS_IN_BOX_FIELD_NUMBER: _ClassVar[int]
    LASER_KEYPOINTS_FIELD_NUMBER: _ClassVar[int]
    CAMERA_KEYPOINTS_FIELD_NUMBER: _ClassVar[int]
    ASSOCIATION_FIELD_NUMBER: _ClassVar[int]
    MOST_VISIBLE_CAMERA_NAME_FIELD_NUMBER: _ClassVar[int]
    CAMERA_SYNCED_BOX_FIELD_NUMBER: _ClassVar[int]
    box: Label.Box
    metadata: Label.Metadata
    type: Label.Type
    id: str
    detection_difficulty_level: Label.DifficultyLevel
    tracking_difficulty_level: Label.DifficultyLevel
    num_lidar_points_in_box: int
    num_top_lidar_points_in_box: int
    laser_keypoints: _keypoint_pb2.LaserKeypoints
    camera_keypoints: _keypoint_pb2.CameraKeypoints
    association: Label.Association
    most_visible_camera_name: str
    camera_synced_box: Label.Box
    def __init__(self, box: _Optional[_Union[Label.Box, _Mapping]] = ..., metadata: _Optional[_Union[Label.Metadata, _Mapping]] = ..., type: _Optional[_Union[Label.Type, str]] = ..., id: _Optional[str] = ..., detection_difficulty_level: _Optional[_Union[Label.DifficultyLevel, str]] = ..., tracking_difficulty_level: _Optional[_Union[Label.DifficultyLevel, str]] = ..., num_lidar_points_in_box: _Optional[int] = ..., num_top_lidar_points_in_box: _Optional[int] = ..., laser_keypoints: _Optional[_Union[_keypoint_pb2.LaserKeypoints, _Mapping]] = ..., camera_keypoints: _Optional[_Union[_keypoint_pb2.CameraKeypoints, _Mapping]] = ..., association: _Optional[_Union[Label.Association, _Mapping]] = ..., most_visible_camera_name: _Optional[str] = ..., camera_synced_box: _Optional[_Union[Label.Box, _Mapping]] = ...) -> None: ...

class Polygon2dProto(_message.Message):
    __slots__ = ["x", "y", "id"]
    X_FIELD_NUMBER: _ClassVar[int]
    Y_FIELD_NUMBER: _ClassVar[int]
    ID_FIELD_NUMBER: _ClassVar[int]
    x: _containers.RepeatedScalarFieldContainer[float]
    y: _containers.RepeatedScalarFieldContainer[float]
    id: str
    def __init__(self, x: _Optional[_Iterable[float]] = ..., y: _Optional[_Iterable[float]] = ..., id: _Optional[str] = ...) -> None: ...
