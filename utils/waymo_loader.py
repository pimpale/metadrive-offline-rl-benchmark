import os
# tensorflow don't allocate all gpu memory right away
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf
from waymo_protos import scenario_pb2
from waymo_protos import map_pb2
import numpy as np
import numpy.typing as npt
import typing
import scenario
from collections import defaultdict
from dataclasses import dataclass

from scipy.ndimage import gaussian_filter1d

def extract_traffic_signal_state(t: map_pb2.TrafficSignalLaneState.State) -> str:
    match t:
        case map_pb2.TrafficSignalLaneState.State.LANE_STATE_UNKNOWN:
            return 'LANE_STATE_UNKNOWN'
        case map_pb2.TrafficSignalLaneState.State.LANE_STATE_ARROW_STOP:
            return 'LANE_STATE_ARROW_STOP'
        case map_pb2.TrafficSignalLaneState.State.LANE_STATE_ARROW_CAUTION:
            return 'LANE_STATE_ARROW_CAUTION'
        case map_pb2.TrafficSignalLaneState.State.LANE_STATE_ARROW_GO:
            return 'LANE_STATE_ARROW_GO'
        case map_pb2.TrafficSignalLaneState.State.LANE_STATE_STOP:
            return 'LANE_STATE_STOP'
        case map_pb2.TrafficSignalLaneState.State.LANE_STATE_CAUTION:
            return 'LANE_STATE_CAUTION'
        case map_pb2.TrafficSignalLaneState.State.LANE_STATE_GO:
            return 'LANE_STATE_GO'
        case map_pb2.TrafficSignalLaneState.State.LANE_STATE_FLASHING_STOP:
            return 'LANE_STATE_FLASHING_STOP'
        case map_pb2.TrafficSignalLaneState.State.LANE_STATE_FLASHING_CAUTION:
            return 'LANE_STATE_FLASHING_CAUTION'
        case _:
            raise ValueError(f'Unknown waymo traffic signal state: {t}')

def extract_road_line_type(t: map_pb2.RoadLine.RoadLineType) -> str:
    match t:
        case map_pb2.RoadLine.RoadLineType.TYPE_UNKNOWN:
            return 'UNKNOWN'
        case map_pb2.RoadLine.RoadLineType.TYPE_BROKEN_SINGLE_WHITE:
            return 'ROAD_LINE_BROKEN_SINGLE_WHITE'
        case map_pb2.RoadLine.RoadLineType.TYPE_SOLID_SINGLE_WHITE:
            return 'ROAD_LINE_SOLID_SINGLE_WHITE'
        case map_pb2.RoadLine.RoadLineType.TYPE_SOLID_DOUBLE_WHITE:
            return 'ROAD_LINE_SOLID_DOUBLE_WHITE'
        case map_pb2.RoadLine.RoadLineType.TYPE_BROKEN_SINGLE_YELLOW:
            return 'ROAD_LINE_BROKEN_SINGLE_YELLOW'
        case map_pb2.RoadLine.RoadLineType.TYPE_BROKEN_DOUBLE_YELLOW:
            return 'ROAD_LINE_BROKEN_DOUBLE_YELLOW'
        case map_pb2.RoadLine.RoadLineType.TYPE_SOLID_SINGLE_YELLOW:
            return 'ROAD_LINE_SOLID_SINGLE_YELLOW'
        case map_pb2.RoadLine.RoadLineType.TYPE_SOLID_DOUBLE_YELLOW:
            return 'ROAD_LINE_SOLID_DOUBLE_YELLOW'
        case map_pb2.RoadLine.RoadLineType.TYPE_PASSING_DOUBLE_YELLOW:
            return 'ROAD_LINE_PASSING_DOUBLE_YELLOW'
        case _:
            raise ValueError(f'Unknown waymo road line type: {t}')

def extract_road_edge_type(t: map_pb2.RoadEdge.RoadEdgeType) -> str:
    match t:
        case map_pb2.RoadEdge.RoadEdgeType.TYPE_UNKNOWN:
            return 'UNKNOWN'
        case map_pb2.RoadEdge.RoadEdgeType.TYPE_ROAD_EDGE_BOUNDARY:
            return 'ROAD_EDGE_BOUNDARY'
        case map_pb2.RoadEdge.RoadEdgeType.TYPE_ROAD_EDGE_MEDIAN:
            return 'ROAD_EDGE_MEDIAN'
        case _:
            raise ValueError(f'Unknown waymo road edge type: {t}')

def extract_object_type(t: scenario_pb2.Track.ObjectType) -> str:
    match t:
        case scenario_pb2.Track.ObjectType.TYPE_UNSET:
            return 'UNSET'
        case scenario_pb2.Track.ObjectType.TYPE_VEHICLE:
            return 'VEHICLE'
        case scenario_pb2.Track.ObjectType.TYPE_PEDESTRIAN:
            return 'PEDESTRIAN'
        case scenario_pb2.Track.ObjectType.TYPE_CYCLIST:
            return 'CYCLIST'
        case scenario_pb2.Track.ObjectType.TYPE_OTHER:
            return 'OTHER'
        case _:
            raise ValueError(f'Unknown waymo object type: {t}')


def extract_track_state(state: scenario_pb2.ObjectState) -> scenario.AgentState:
    return scenario.AgentState(
        position=np.array([state.center_x, state.center_y], dtype=np.float32),
        velocity=np.array([state.velocity_x, state.velocity_y], dtype=np.float32),
        heading=state.heading,
        valid=state.valid
    )

def extract_track(track: scenario_pb2.Track) -> scenario.AgentTrack:
    lengths = []
    widths = []
    heights = []
    for s in track.states:
        if s.valid:
            lengths.append(s.length)
            widths.append(s.width)
            heights.append(s.height)
     
    return scenario.AgentTrack(
        object_type=extract_object_type(track.object_type),
        length=np.median(lengths).item(),
        width=np.median(widths).item(),
        height=np.median(heights).item(),
        states=[extract_track_state(s) for s in track.states]
    )

def extract_poly(message: typing.Iterable[map_pb2.MapPoint]) -> np.ndarray:
    x = np.array([i.x for i in message], dtype=np.float32)
    y = np.array([i.y for i in message], dtype=np.float32)
    return np.stack((x, y), axis=1)

def extract_lane_center(f: map_pb2.LaneCenter) -> scenario.LaneCenter:
    return scenario.LaneCenter(
        polyline=extract_poly(f.polyline),
    )

def extract_line(f: map_pb2.RoadLine) -> scenario.RoadLine:
    return scenario.RoadLine(
        polyline=extract_poly(f.polyline),
        kind=extract_road_line_type(f.type)
    )

def extract_edge(f: map_pb2.RoadEdge) -> scenario.RoadEdge:
    return scenario.RoadEdge(
        polyline=extract_poly(f.polyline),
        kind=extract_road_edge_type(f.type)
    )

def extract_stop(f: map_pb2.StopSign) -> scenario.StopSign:
    return scenario.StopSign(
        lane=list(f.lane),
        position=np.array([f.position.x, f.position.y], dtype=np.float32)
    )

def extract_crosswalk(f: map_pb2.Crosswalk) -> scenario.Crosswalk:
    return scenario.Crosswalk(
        polygon=extract_poly(f.polygon)
    )

def extract_bump(f: map_pb2.SpeedBump) -> scenario.SpeedBump:
    return scenario.SpeedBump(
        polygon=extract_poly(f.polygon)
    )

def extract_driveway(f: map_pb2.Driveway) -> scenario.Driveway:
    return scenario.Driveway(
        polygon=extract_poly(f.polygon)
    )

def extract_map_features(map_features: typing.Iterable[map_pb2.MapFeature]) -> dict[int, scenario.MapFeature]:
    ret = {}
    for feature in map_features:
        match feature.WhichOneof('feature_data'):
            case 'lane':
                ret[feature.id] = extract_lane_center(feature.lane)
            case 'road_line':
                ret[feature.id] = extract_line(feature.road_line)
            case 'road_edge':
                ret[feature.id] = extract_edge(feature.road_edge)
            case 'stop_sign':
                ret[feature.id] = extract_stop(feature.stop_sign)
            case 'crosswalk':
                ret[feature.id] = extract_crosswalk(feature.crosswalk)
            case 'speed_bump':
                ret[feature.id] = extract_bump(feature.speed_bump)
            case 'driveway':
                ret[feature.id] = extract_driveway(feature.driveway)
    return ret

def extract_dynamic_state(dynamic_state: typing.Iterable[scenario_pb2.DynamicMapState]) -> list[scenario.DynamicState]:
    track_length = len(list(dynamic_state))
    ret: dict[int, scenario.TrafficLight] = defaultdict(
        lambda: scenario.TrafficLight(
            lane=0,
            states=['LANE_STATE_UNKNOWN' for _ in range(track_length)],
            position=np.array([0, 0], dtype=np.float32),
        )
    )
    for i, dynamic_map_state in enumerate(dynamic_state):
        for light_state in dynamic_map_state.lane_states:
            ret[light_state.lane].lane = light_state.lane
            ret[light_state.lane].states[i] = extract_traffic_signal_state(light_state.state)
            ret[light_state.lane].position = (
                np.array([light_state.stop_point.x, light_state.stop_point.y], dtype=np.float32)
            )

    return list(ret.values())

def extract_scenarios_file(file_path: str) -> list[scenario.Scenario]:
    scenarios = []
    scenario_proto = scenario_pb2.Scenario()
    for data in tf.data.TFRecordDataset(file_path, compression_type="").as_numpy_iterator():
        scenario_proto.ParseFromString(bytes(data))
        scenarios.append(extract_scenario(scenario_proto))
    return scenarios

def extract_scenario(s: scenario_pb2.Scenario) -> scenario.Scenario:
    return scenario.Scenario(
        scenario_id=s.scenario_id,
        ego_track_index=s.sdc_track_index,
        tracks=[extract_track(t) for t in s.tracks], 
        map_features=extract_map_features(s.map_features),
        dynamic_state=extract_dynamic_state(s.dynamic_map_states)
    )


from env import State


def extract_trajectory(scenario: scenario_pb2.Scenario) -> list[State]:
    vx = np.array([state.velocity_x for state in scenario.tracks[scenario.sdc_track_index].states if state.valid], dtype=np.float32)
    vy = np.array([state.velocity_y for state in scenario.tracks[scenario.sdc_track_index].states if state.valid], dtype=np.float32)
    heading = np.array([state.heading for state in scenario.tracks[scenario.sdc_track_index].states if state.valid], dtype=np.float32)
    
    # filter
    vx = gaussian_filter1d(vx, sigma=3)
    vy = gaussian_filter1d(vy, sigma=3)
    heading = gaussian_filter1d(heading, sigma=3)
    
    return [
        State(
            heading=heading,
            velocity=np.array([vx, vy], dtype=np.float32),
        )
        for vx, vy, heading in zip(vx, vy, heading)
    ]

def extract_trajectory_file(file_path: str) -> list[list[State]]:
    trajectories = []
    scenario_proto = scenario_pb2.Scenario()
    for data in tf.data.TFRecordDataset(file_path, compression_type="").as_numpy_iterator():
        scenario_proto.ParseFromString(bytes(data))
        trajectories.append(extract_trajectory(scenario_proto))
    return trajectories