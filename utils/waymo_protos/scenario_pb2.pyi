from waymo_protos import compressed_lidar_pb2 as _compressed_lidar_pb2
from waymo_protos import map_pb2 as _map_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ObjectState(_message.Message):
    __slots__ = ["center_x", "center_y", "center_z", "length", "width", "height", "heading", "velocity_x", "velocity_y", "valid"]
    CENTER_X_FIELD_NUMBER: _ClassVar[int]
    CENTER_Y_FIELD_NUMBER: _ClassVar[int]
    CENTER_Z_FIELD_NUMBER: _ClassVar[int]
    LENGTH_FIELD_NUMBER: _ClassVar[int]
    WIDTH_FIELD_NUMBER: _ClassVar[int]
    HEIGHT_FIELD_NUMBER: _ClassVar[int]
    HEADING_FIELD_NUMBER: _ClassVar[int]
    VELOCITY_X_FIELD_NUMBER: _ClassVar[int]
    VELOCITY_Y_FIELD_NUMBER: _ClassVar[int]
    VALID_FIELD_NUMBER: _ClassVar[int]
    center_x: float
    center_y: float
    center_z: float
    length: float
    width: float
    height: float
    heading: float
    velocity_x: float
    velocity_y: float
    valid: bool
    def __init__(self, center_x: _Optional[float] = ..., center_y: _Optional[float] = ..., center_z: _Optional[float] = ..., length: _Optional[float] = ..., width: _Optional[float] = ..., height: _Optional[float] = ..., heading: _Optional[float] = ..., velocity_x: _Optional[float] = ..., velocity_y: _Optional[float] = ..., valid: bool = ...) -> None: ...

class Track(_message.Message):
    __slots__ = ["id", "object_type", "states"]
    class ObjectType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
        TYPE_UNSET: _ClassVar[Track.ObjectType]
        TYPE_VEHICLE: _ClassVar[Track.ObjectType]
        TYPE_PEDESTRIAN: _ClassVar[Track.ObjectType]
        TYPE_CYCLIST: _ClassVar[Track.ObjectType]
        TYPE_OTHER: _ClassVar[Track.ObjectType]
    TYPE_UNSET: Track.ObjectType
    TYPE_VEHICLE: Track.ObjectType
    TYPE_PEDESTRIAN: Track.ObjectType
    TYPE_CYCLIST: Track.ObjectType
    TYPE_OTHER: Track.ObjectType
    ID_FIELD_NUMBER: _ClassVar[int]
    OBJECT_TYPE_FIELD_NUMBER: _ClassVar[int]
    STATES_FIELD_NUMBER: _ClassVar[int]
    id: int
    object_type: Track.ObjectType
    states: _containers.RepeatedCompositeFieldContainer[ObjectState]
    def __init__(self, id: _Optional[int] = ..., object_type: _Optional[_Union[Track.ObjectType, str]] = ..., states: _Optional[_Iterable[_Union[ObjectState, _Mapping]]] = ...) -> None: ...

class DynamicMapState(_message.Message):
    __slots__ = ["lane_states"]
    LANE_STATES_FIELD_NUMBER: _ClassVar[int]
    lane_states: _containers.RepeatedCompositeFieldContainer[_map_pb2.TrafficSignalLaneState]
    def __init__(self, lane_states: _Optional[_Iterable[_Union[_map_pb2.TrafficSignalLaneState, _Mapping]]] = ...) -> None: ...

class RequiredPrediction(_message.Message):
    __slots__ = ["track_index", "difficulty"]
    class DifficultyLevel(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
        NONE: _ClassVar[RequiredPrediction.DifficultyLevel]
        LEVEL_1: _ClassVar[RequiredPrediction.DifficultyLevel]
        LEVEL_2: _ClassVar[RequiredPrediction.DifficultyLevel]
    NONE: RequiredPrediction.DifficultyLevel
    LEVEL_1: RequiredPrediction.DifficultyLevel
    LEVEL_2: RequiredPrediction.DifficultyLevel
    TRACK_INDEX_FIELD_NUMBER: _ClassVar[int]
    DIFFICULTY_FIELD_NUMBER: _ClassVar[int]
    track_index: int
    difficulty: RequiredPrediction.DifficultyLevel
    def __init__(self, track_index: _Optional[int] = ..., difficulty: _Optional[_Union[RequiredPrediction.DifficultyLevel, str]] = ...) -> None: ...

class Scenario(_message.Message):
    __slots__ = ["scenario_id", "timestamps_seconds", "current_time_index", "tracks", "dynamic_map_states", "map_features", "sdc_track_index", "objects_of_interest", "tracks_to_predict", "compressed_frame_laser_data"]
    SCENARIO_ID_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMPS_SECONDS_FIELD_NUMBER: _ClassVar[int]
    CURRENT_TIME_INDEX_FIELD_NUMBER: _ClassVar[int]
    TRACKS_FIELD_NUMBER: _ClassVar[int]
    DYNAMIC_MAP_STATES_FIELD_NUMBER: _ClassVar[int]
    MAP_FEATURES_FIELD_NUMBER: _ClassVar[int]
    SDC_TRACK_INDEX_FIELD_NUMBER: _ClassVar[int]
    OBJECTS_OF_INTEREST_FIELD_NUMBER: _ClassVar[int]
    TRACKS_TO_PREDICT_FIELD_NUMBER: _ClassVar[int]
    COMPRESSED_FRAME_LASER_DATA_FIELD_NUMBER: _ClassVar[int]
    scenario_id: str
    timestamps_seconds: _containers.RepeatedScalarFieldContainer[float]
    current_time_index: int
    tracks: _containers.RepeatedCompositeFieldContainer[Track]
    dynamic_map_states: _containers.RepeatedCompositeFieldContainer[DynamicMapState]
    map_features: _containers.RepeatedCompositeFieldContainer[_map_pb2.MapFeature]
    sdc_track_index: int
    objects_of_interest: _containers.RepeatedScalarFieldContainer[int]
    tracks_to_predict: _containers.RepeatedCompositeFieldContainer[RequiredPrediction]
    compressed_frame_laser_data: _containers.RepeatedCompositeFieldContainer[_compressed_lidar_pb2.CompressedFrameLaserData]
    def __init__(self, scenario_id: _Optional[str] = ..., timestamps_seconds: _Optional[_Iterable[float]] = ..., current_time_index: _Optional[int] = ..., tracks: _Optional[_Iterable[_Union[Track, _Mapping]]] = ..., dynamic_map_states: _Optional[_Iterable[_Union[DynamicMapState, _Mapping]]] = ..., map_features: _Optional[_Iterable[_Union[_map_pb2.MapFeature, _Mapping]]] = ..., sdc_track_index: _Optional[int] = ..., objects_of_interest: _Optional[_Iterable[int]] = ..., tracks_to_predict: _Optional[_Iterable[_Union[RequiredPrediction, _Mapping]]] = ..., compressed_frame_laser_data: _Optional[_Iterable[_Union[_compressed_lidar_pb2.CompressedFrameLaserData, _Mapping]]] = ...) -> None: ...
