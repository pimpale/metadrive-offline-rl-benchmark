from waymo_protos import label_pb2 as _label_pb2
from waymo_protos import map_pb2 as _map_pb2
from waymo_protos import vector_pb2 as _vector_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf.internal import python_message as _python_message
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class MatrixShape(_message.Message):
    __slots__ = ["dims"]
    DIMS_FIELD_NUMBER: _ClassVar[int]
    dims: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, dims: _Optional[_Iterable[int]] = ...) -> None: ...

class MatrixFloat(_message.Message):
    __slots__ = ["data", "shape"]
    DATA_FIELD_NUMBER: _ClassVar[int]
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    data: _containers.RepeatedScalarFieldContainer[float]
    shape: MatrixShape
    def __init__(self, data: _Optional[_Iterable[float]] = ..., shape: _Optional[_Union[MatrixShape, _Mapping]] = ...) -> None: ...

class MatrixInt32(_message.Message):
    __slots__ = ["data", "shape"]
    DATA_FIELD_NUMBER: _ClassVar[int]
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    data: _containers.RepeatedScalarFieldContainer[int]
    shape: MatrixShape
    def __init__(self, data: _Optional[_Iterable[int]] = ..., shape: _Optional[_Union[MatrixShape, _Mapping]] = ...) -> None: ...

class CameraName(_message.Message):
    __slots__ = []
    class Name(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
        UNKNOWN: _ClassVar[CameraName.Name]
        FRONT: _ClassVar[CameraName.Name]
        FRONT_LEFT: _ClassVar[CameraName.Name]
        FRONT_RIGHT: _ClassVar[CameraName.Name]
        SIDE_LEFT: _ClassVar[CameraName.Name]
        SIDE_RIGHT: _ClassVar[CameraName.Name]
    UNKNOWN: CameraName.Name
    FRONT: CameraName.Name
    FRONT_LEFT: CameraName.Name
    FRONT_RIGHT: CameraName.Name
    SIDE_LEFT: CameraName.Name
    SIDE_RIGHT: CameraName.Name
    def __init__(self) -> None: ...

class LaserName(_message.Message):
    __slots__ = []
    class Name(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
        UNKNOWN: _ClassVar[LaserName.Name]
        TOP: _ClassVar[LaserName.Name]
        FRONT: _ClassVar[LaserName.Name]
        SIDE_LEFT: _ClassVar[LaserName.Name]
        SIDE_RIGHT: _ClassVar[LaserName.Name]
        REAR: _ClassVar[LaserName.Name]
    UNKNOWN: LaserName.Name
    TOP: LaserName.Name
    FRONT: LaserName.Name
    SIDE_LEFT: LaserName.Name
    SIDE_RIGHT: LaserName.Name
    REAR: LaserName.Name
    def __init__(self) -> None: ...

class Transform(_message.Message):
    __slots__ = ["transform"]
    TRANSFORM_FIELD_NUMBER: _ClassVar[int]
    transform: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, transform: _Optional[_Iterable[float]] = ...) -> None: ...

class Velocity(_message.Message):
    __slots__ = ["v_x", "v_y", "v_z", "w_x", "w_y", "w_z"]
    V_X_FIELD_NUMBER: _ClassVar[int]
    V_Y_FIELD_NUMBER: _ClassVar[int]
    V_Z_FIELD_NUMBER: _ClassVar[int]
    W_X_FIELD_NUMBER: _ClassVar[int]
    W_Y_FIELD_NUMBER: _ClassVar[int]
    W_Z_FIELD_NUMBER: _ClassVar[int]
    v_x: float
    v_y: float
    v_z: float
    w_x: float
    w_y: float
    w_z: float
    def __init__(self, v_x: _Optional[float] = ..., v_y: _Optional[float] = ..., v_z: _Optional[float] = ..., w_x: _Optional[float] = ..., w_y: _Optional[float] = ..., w_z: _Optional[float] = ...) -> None: ...

class CameraCalibration(_message.Message):
    __slots__ = ["name", "intrinsic", "extrinsic", "width", "height", "rolling_shutter_direction"]
    class RollingShutterReadOutDirection(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
        UNKNOWN: _ClassVar[CameraCalibration.RollingShutterReadOutDirection]
        TOP_TO_BOTTOM: _ClassVar[CameraCalibration.RollingShutterReadOutDirection]
        LEFT_TO_RIGHT: _ClassVar[CameraCalibration.RollingShutterReadOutDirection]
        BOTTOM_TO_TOP: _ClassVar[CameraCalibration.RollingShutterReadOutDirection]
        RIGHT_TO_LEFT: _ClassVar[CameraCalibration.RollingShutterReadOutDirection]
        GLOBAL_SHUTTER: _ClassVar[CameraCalibration.RollingShutterReadOutDirection]
    UNKNOWN: CameraCalibration.RollingShutterReadOutDirection
    TOP_TO_BOTTOM: CameraCalibration.RollingShutterReadOutDirection
    LEFT_TO_RIGHT: CameraCalibration.RollingShutterReadOutDirection
    BOTTOM_TO_TOP: CameraCalibration.RollingShutterReadOutDirection
    RIGHT_TO_LEFT: CameraCalibration.RollingShutterReadOutDirection
    GLOBAL_SHUTTER: CameraCalibration.RollingShutterReadOutDirection
    NAME_FIELD_NUMBER: _ClassVar[int]
    INTRINSIC_FIELD_NUMBER: _ClassVar[int]
    EXTRINSIC_FIELD_NUMBER: _ClassVar[int]
    WIDTH_FIELD_NUMBER: _ClassVar[int]
    HEIGHT_FIELD_NUMBER: _ClassVar[int]
    ROLLING_SHUTTER_DIRECTION_FIELD_NUMBER: _ClassVar[int]
    name: CameraName.Name
    intrinsic: _containers.RepeatedScalarFieldContainer[float]
    extrinsic: Transform
    width: int
    height: int
    rolling_shutter_direction: CameraCalibration.RollingShutterReadOutDirection
    def __init__(self, name: _Optional[_Union[CameraName.Name, str]] = ..., intrinsic: _Optional[_Iterable[float]] = ..., extrinsic: _Optional[_Union[Transform, _Mapping]] = ..., width: _Optional[int] = ..., height: _Optional[int] = ..., rolling_shutter_direction: _Optional[_Union[CameraCalibration.RollingShutterReadOutDirection, str]] = ...) -> None: ...

class LaserCalibration(_message.Message):
    __slots__ = ["name", "beam_inclinations", "beam_inclination_min", "beam_inclination_max", "extrinsic"]
    NAME_FIELD_NUMBER: _ClassVar[int]
    BEAM_INCLINATIONS_FIELD_NUMBER: _ClassVar[int]
    BEAM_INCLINATION_MIN_FIELD_NUMBER: _ClassVar[int]
    BEAM_INCLINATION_MAX_FIELD_NUMBER: _ClassVar[int]
    EXTRINSIC_FIELD_NUMBER: _ClassVar[int]
    name: LaserName.Name
    beam_inclinations: _containers.RepeatedScalarFieldContainer[float]
    beam_inclination_min: float
    beam_inclination_max: float
    extrinsic: Transform
    def __init__(self, name: _Optional[_Union[LaserName.Name, str]] = ..., beam_inclinations: _Optional[_Iterable[float]] = ..., beam_inclination_min: _Optional[float] = ..., beam_inclination_max: _Optional[float] = ..., extrinsic: _Optional[_Union[Transform, _Mapping]] = ...) -> None: ...

class Context(_message.Message):
    __slots__ = ["name", "camera_calibrations", "laser_calibrations", "stats"]
    class Stats(_message.Message):
        __slots__ = ["laser_object_counts", "camera_object_counts", "time_of_day", "location", "weather"]
        class ObjectCount(_message.Message):
            __slots__ = ["type", "count"]
            TYPE_FIELD_NUMBER: _ClassVar[int]
            COUNT_FIELD_NUMBER: _ClassVar[int]
            type: _label_pb2.Label.Type
            count: int
            def __init__(self, type: _Optional[_Union[_label_pb2.Label.Type, str]] = ..., count: _Optional[int] = ...) -> None: ...
        LASER_OBJECT_COUNTS_FIELD_NUMBER: _ClassVar[int]
        CAMERA_OBJECT_COUNTS_FIELD_NUMBER: _ClassVar[int]
        TIME_OF_DAY_FIELD_NUMBER: _ClassVar[int]
        LOCATION_FIELD_NUMBER: _ClassVar[int]
        WEATHER_FIELD_NUMBER: _ClassVar[int]
        laser_object_counts: _containers.RepeatedCompositeFieldContainer[Context.Stats.ObjectCount]
        camera_object_counts: _containers.RepeatedCompositeFieldContainer[Context.Stats.ObjectCount]
        time_of_day: str
        location: str
        weather: str
        def __init__(self, laser_object_counts: _Optional[_Iterable[_Union[Context.Stats.ObjectCount, _Mapping]]] = ..., camera_object_counts: _Optional[_Iterable[_Union[Context.Stats.ObjectCount, _Mapping]]] = ..., time_of_day: _Optional[str] = ..., location: _Optional[str] = ..., weather: _Optional[str] = ...) -> None: ...
    NAME_FIELD_NUMBER: _ClassVar[int]
    CAMERA_CALIBRATIONS_FIELD_NUMBER: _ClassVar[int]
    LASER_CALIBRATIONS_FIELD_NUMBER: _ClassVar[int]
    STATS_FIELD_NUMBER: _ClassVar[int]
    name: str
    camera_calibrations: _containers.RepeatedCompositeFieldContainer[CameraCalibration]
    laser_calibrations: _containers.RepeatedCompositeFieldContainer[LaserCalibration]
    stats: Context.Stats
    def __init__(self, name: _Optional[str] = ..., camera_calibrations: _Optional[_Iterable[_Union[CameraCalibration, _Mapping]]] = ..., laser_calibrations: _Optional[_Iterable[_Union[LaserCalibration, _Mapping]]] = ..., stats: _Optional[_Union[Context.Stats, _Mapping]] = ...) -> None: ...

class RangeImage(_message.Message):
    __slots__ = ["range_image_compressed", "camera_projection_compressed", "range_image_pose_compressed", "range_image_flow_compressed", "segmentation_label_compressed", "range_image"]
    RANGE_IMAGE_COMPRESSED_FIELD_NUMBER: _ClassVar[int]
    CAMERA_PROJECTION_COMPRESSED_FIELD_NUMBER: _ClassVar[int]
    RANGE_IMAGE_POSE_COMPRESSED_FIELD_NUMBER: _ClassVar[int]
    RANGE_IMAGE_FLOW_COMPRESSED_FIELD_NUMBER: _ClassVar[int]
    SEGMENTATION_LABEL_COMPRESSED_FIELD_NUMBER: _ClassVar[int]
    RANGE_IMAGE_FIELD_NUMBER: _ClassVar[int]
    range_image_compressed: bytes
    camera_projection_compressed: bytes
    range_image_pose_compressed: bytes
    range_image_flow_compressed: bytes
    segmentation_label_compressed: bytes
    range_image: MatrixFloat
    def __init__(self, range_image_compressed: _Optional[bytes] = ..., camera_projection_compressed: _Optional[bytes] = ..., range_image_pose_compressed: _Optional[bytes] = ..., range_image_flow_compressed: _Optional[bytes] = ..., segmentation_label_compressed: _Optional[bytes] = ..., range_image: _Optional[_Union[MatrixFloat, _Mapping]] = ...) -> None: ...

class CameraSegmentationLabel(_message.Message):
    __slots__ = ["panoptic_label_divisor", "panoptic_label", "instance_id_to_global_id_mapping", "sequence_id", "num_cameras_covered"]
    class InstanceIDToGlobalIDMapping(_message.Message):
        __slots__ = ["local_instance_id", "global_instance_id", "is_tracked"]
        LOCAL_INSTANCE_ID_FIELD_NUMBER: _ClassVar[int]
        GLOBAL_INSTANCE_ID_FIELD_NUMBER: _ClassVar[int]
        IS_TRACKED_FIELD_NUMBER: _ClassVar[int]
        local_instance_id: int
        global_instance_id: int
        is_tracked: bool
        def __init__(self, local_instance_id: _Optional[int] = ..., global_instance_id: _Optional[int] = ..., is_tracked: bool = ...) -> None: ...
    PANOPTIC_LABEL_DIVISOR_FIELD_NUMBER: _ClassVar[int]
    PANOPTIC_LABEL_FIELD_NUMBER: _ClassVar[int]
    INSTANCE_ID_TO_GLOBAL_ID_MAPPING_FIELD_NUMBER: _ClassVar[int]
    SEQUENCE_ID_FIELD_NUMBER: _ClassVar[int]
    NUM_CAMERAS_COVERED_FIELD_NUMBER: _ClassVar[int]
    panoptic_label_divisor: int
    panoptic_label: bytes
    instance_id_to_global_id_mapping: _containers.RepeatedCompositeFieldContainer[CameraSegmentationLabel.InstanceIDToGlobalIDMapping]
    sequence_id: str
    num_cameras_covered: bytes
    def __init__(self, panoptic_label_divisor: _Optional[int] = ..., panoptic_label: _Optional[bytes] = ..., instance_id_to_global_id_mapping: _Optional[_Iterable[_Union[CameraSegmentationLabel.InstanceIDToGlobalIDMapping, _Mapping]]] = ..., sequence_id: _Optional[str] = ..., num_cameras_covered: _Optional[bytes] = ...) -> None: ...

class CameraImage(_message.Message):
    __slots__ = ["name", "image", "pose", "velocity", "pose_timestamp", "shutter", "camera_trigger_time", "camera_readout_done_time", "camera_segmentation_label"]
    NAME_FIELD_NUMBER: _ClassVar[int]
    IMAGE_FIELD_NUMBER: _ClassVar[int]
    POSE_FIELD_NUMBER: _ClassVar[int]
    VELOCITY_FIELD_NUMBER: _ClassVar[int]
    POSE_TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    SHUTTER_FIELD_NUMBER: _ClassVar[int]
    CAMERA_TRIGGER_TIME_FIELD_NUMBER: _ClassVar[int]
    CAMERA_READOUT_DONE_TIME_FIELD_NUMBER: _ClassVar[int]
    CAMERA_SEGMENTATION_LABEL_FIELD_NUMBER: _ClassVar[int]
    name: CameraName.Name
    image: bytes
    pose: Transform
    velocity: Velocity
    pose_timestamp: float
    shutter: float
    camera_trigger_time: float
    camera_readout_done_time: float
    camera_segmentation_label: CameraSegmentationLabel
    def __init__(self, name: _Optional[_Union[CameraName.Name, str]] = ..., image: _Optional[bytes] = ..., pose: _Optional[_Union[Transform, _Mapping]] = ..., velocity: _Optional[_Union[Velocity, _Mapping]] = ..., pose_timestamp: _Optional[float] = ..., shutter: _Optional[float] = ..., camera_trigger_time: _Optional[float] = ..., camera_readout_done_time: _Optional[float] = ..., camera_segmentation_label: _Optional[_Union[CameraSegmentationLabel, _Mapping]] = ...) -> None: ...

class CameraLabels(_message.Message):
    __slots__ = ["name", "labels"]
    NAME_FIELD_NUMBER: _ClassVar[int]
    LABELS_FIELD_NUMBER: _ClassVar[int]
    name: CameraName.Name
    labels: _containers.RepeatedCompositeFieldContainer[_label_pb2.Label]
    def __init__(self, name: _Optional[_Union[CameraName.Name, str]] = ..., labels: _Optional[_Iterable[_Union[_label_pb2.Label, _Mapping]]] = ...) -> None: ...

class Laser(_message.Message):
    __slots__ = ["name", "ri_return1", "ri_return2"]
    NAME_FIELD_NUMBER: _ClassVar[int]
    RI_RETURN1_FIELD_NUMBER: _ClassVar[int]
    RI_RETURN2_FIELD_NUMBER: _ClassVar[int]
    name: LaserName.Name
    ri_return1: RangeImage
    ri_return2: RangeImage
    def __init__(self, name: _Optional[_Union[LaserName.Name, str]] = ..., ri_return1: _Optional[_Union[RangeImage, _Mapping]] = ..., ri_return2: _Optional[_Union[RangeImage, _Mapping]] = ...) -> None: ...

class Frame(_message.Message):
    __slots__ = ["context", "timestamp_micros", "pose", "images", "lasers", "laser_labels", "projected_lidar_labels", "camera_labels", "no_label_zones", "map_features", "map_pose_offset"]
    Extensions: _python_message._ExtensionDict
    CONTEXT_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_MICROS_FIELD_NUMBER: _ClassVar[int]
    POSE_FIELD_NUMBER: _ClassVar[int]
    IMAGES_FIELD_NUMBER: _ClassVar[int]
    LASERS_FIELD_NUMBER: _ClassVar[int]
    LASER_LABELS_FIELD_NUMBER: _ClassVar[int]
    PROJECTED_LIDAR_LABELS_FIELD_NUMBER: _ClassVar[int]
    CAMERA_LABELS_FIELD_NUMBER: _ClassVar[int]
    NO_LABEL_ZONES_FIELD_NUMBER: _ClassVar[int]
    MAP_FEATURES_FIELD_NUMBER: _ClassVar[int]
    MAP_POSE_OFFSET_FIELD_NUMBER: _ClassVar[int]
    context: Context
    timestamp_micros: int
    pose: Transform
    images: _containers.RepeatedCompositeFieldContainer[CameraImage]
    lasers: _containers.RepeatedCompositeFieldContainer[Laser]
    laser_labels: _containers.RepeatedCompositeFieldContainer[_label_pb2.Label]
    projected_lidar_labels: _containers.RepeatedCompositeFieldContainer[CameraLabels]
    camera_labels: _containers.RepeatedCompositeFieldContainer[CameraLabels]
    no_label_zones: _containers.RepeatedCompositeFieldContainer[_label_pb2.Polygon2dProto]
    map_features: _containers.RepeatedCompositeFieldContainer[_map_pb2.MapFeature]
    map_pose_offset: _vector_pb2.Vector3d
    def __init__(self, context: _Optional[_Union[Context, _Mapping]] = ..., timestamp_micros: _Optional[int] = ..., pose: _Optional[_Union[Transform, _Mapping]] = ..., images: _Optional[_Iterable[_Union[CameraImage, _Mapping]]] = ..., lasers: _Optional[_Iterable[_Union[Laser, _Mapping]]] = ..., laser_labels: _Optional[_Iterable[_Union[_label_pb2.Label, _Mapping]]] = ..., projected_lidar_labels: _Optional[_Iterable[_Union[CameraLabels, _Mapping]]] = ..., camera_labels: _Optional[_Iterable[_Union[CameraLabels, _Mapping]]] = ..., no_label_zones: _Optional[_Iterable[_Union[_label_pb2.Polygon2dProto, _Mapping]]] = ..., map_features: _Optional[_Iterable[_Union[_map_pb2.MapFeature, _Mapping]]] = ..., map_pose_offset: _Optional[_Union[_vector_pb2.Vector3d, _Mapping]] = ...) -> None: ...
