import os
# tensorflow don't allocate all gpu memory right away
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf
from waymo_protos import scenario_pb2
import numpy as np
from env import State
from metadrive.scenario import ScenarioDescription
from metadrive.type import MetaDriveType
import typing


TrafficSignal = {
    0: 'LANE_STATE_UNKNOWN',
    1: 'LANE_STATE_ARROW_STOP',
    2: 'LANE_STATE_ARROW_CAUTION',
    3: 'LANE_STATE_ARROW_GO',
    4: 'LANE_STATE_STOP',
    5: 'LANE_STATE_CAUTION',
    6: 'LANE_STATE_GO',
    7: 'LANE_STATE_FLASHING_STOP',
    8: 'LANE_STATE_FLASHING_CAUTION'
}


class WaymoLaneType:
    UNKNOWN = 0
    LANE_FREEWAY = 1
    LANE_SURFACE_STREET = 2
    LANE_BIKE_LANE = 3

    ENUM_TO_STR = {
        UNKNOWN: 'UNKNOWN',
        LANE_FREEWAY: 'LANE_FREEWAY',
        LANE_SURFACE_STREET: 'LANE_SURFACE_STREET',
        LANE_BIKE_LANE: 'LANE_BIKE_LANE'
    }

    @classmethod
    def from_waymo(cls, item):
        return cls.ENUM_TO_STR[item]


class WaymoRoadLineType:
    UNKNOWN = 0
    BROKEN_SINGLE_WHITE = 1
    SOLID_SINGLE_WHITE = 2
    SOLID_DOUBLE_WHITE = 3
    BROKEN_SINGLE_YELLOW = 4
    BROKEN_DOUBLE_YELLOW = 5
    SOLID_SINGLE_YELLOW = 6
    SOLID_DOUBLE_YELLOW = 7
    PASSING_DOUBLE_YELLOW = 8

    ENUM_TO_STR = {
        UNKNOWN: 'UNKNOWN',
        BROKEN_SINGLE_WHITE: 'ROAD_LINE_BROKEN_SINGLE_WHITE',
        SOLID_SINGLE_WHITE: 'ROAD_LINE_SOLID_SINGLE_WHITE',
        SOLID_DOUBLE_WHITE: 'ROAD_LINE_SOLID_DOUBLE_WHITE',
        BROKEN_SINGLE_YELLOW: 'ROAD_LINE_BROKEN_SINGLE_YELLOW',
        BROKEN_DOUBLE_YELLOW: 'ROAD_LINE_BROKEN_DOUBLE_YELLOW',
        SOLID_SINGLE_YELLOW: 'ROAD_LINE_SOLID_SINGLE_YELLOW',
        SOLID_DOUBLE_YELLOW: 'ROAD_LINE_SOLID_DOUBLE_YELLOW',
        PASSING_DOUBLE_YELLOW: 'ROAD_LINE_PASSING_DOUBLE_YELLOW'
    }

    @classmethod
    def from_waymo(cls, item):
        return cls.ENUM_TO_STR[item]


class WaymoRoadEdgeType:
    UNKNOWN = 0
    # Physical road boundary that doesn't have traffic on the other side (e.g., a curb or the k-rail on the right side of a freeway).
    BOUNDARY = 1
    # Physical road boundary that separates the car from other traffic (e.g. a k-rail or an island).
    MEDIAN = 2

    ENUM_TO_STR = {UNKNOWN: 'UNKNOWN', BOUNDARY: 'ROAD_EDGE_BOUNDARY', MEDIAN: 'ROAD_EDGE_MEDIAN'}

    @classmethod
    def from_waymo(cls, item):
        return cls.ENUM_TO_STR[item]


class WaymoAgentType:
    UNSET = 0
    VEHICLE = 1
    PEDESTRIAN = 2
    CYCLIST = 3
    OTHER = 4

    ENUM_TO_STR = {UNSET: 'UNSET', VEHICLE: 'VEHICLE', PEDESTRIAN: 'PEDESTRIAN', CYCLIST: 'CYCLIST', OTHER: 'OTHER'}

    @classmethod
    def from_waymo(cls, item):
        return cls.ENUM_TO_STR[item]



def parse_file(file_path: str) -> list[list[State]]:
    trajectories_in_file = []
    scenario = scenario_pb2.Scenario()
    for data in tf.data.TFRecordDataset(file_path, compression_type="").as_numpy_iterator():
        scenario.ParseFromString(data)
        trajectories_in_file.append(parse_scenario(scenario))
    return trajectories_in_file

def parse_scenario(scenario: scenario_pb2.Scenario) -> ScenarioDescription:
    vx = []
    vy = []
    heading = []
    for s in scenario.tracks[scenario.sdc_track_index].states:
        if s.valid:
            vx.append(s.velocity_x)
            vy.append(s.velocity_y)
            heading.append(s.heading)
    
    states = []
    for row in np.array([vx, vy, heading], dtype=np.float32).T:
        states.append(State(velocity=row[:2], heading=row[2]))
    
    return states





def extract_poly(message):
    x = [i.x for i in message]
    y = [i.y for i in message]
    z = [i.z for i in message]
    coord = np.stack((x, y, z), axis=1).astype("float32")
    return coord


def extract_boundaries(fb):
    b = []
    # b = np.zeros([len(fb), 4], dtype="int64")
    for k in range(len(fb)):
        c = dict()
        c["lane_start_index"] = fb[k].lane_start_index
        c["lane_end_index"] = fb[k].lane_end_index
        c["boundary_type"] = WaymoRoadLineType.from_waymo(fb[k].boundary_type)
        c["boundary_feature_id"] = fb[k].boundary_feature_id
        for key in c:
            c[key] = str(c[key])
        b.append(c)

    return b


def extract_neighbors(fb):
    nbs = []
    for k in range(len(fb)):
        nb = dict()
        nb["feature_id"] = fb[k].feature_id
        nb["self_start_index"] = fb[k].self_start_index
        nb["self_end_index"] = fb[k].self_end_index
        nb["neighbor_start_index"] = fb[k].neighbor_start_index
        nb["neighbor_end_index"] = fb[k].neighbor_end_index
        for key in nb:
            nb[key] = str(nb[key])
        nb["boundaries"] = extract_boundaries(fb[k].boundaries)
        nbs.append(nb)
    return nbs


def extract_center(f):
    center = dict()
    f = f.lane
    center["speed_limit_mph"] = f.speed_limit_mph

    center["type"] = WaymoLaneType.from_waymo(f.type)

    center["polyline"] = extract_poly(f.polyline)

    center["interpolating"] = f.interpolating

    center["entry_lanes"] = [x for x in f.entry_lanes]

    center["exit_lanes"] = [x for x in f.exit_lanes]

    center["left_boundaries"] = extract_boundaries(f.left_boundaries)

    center["right_boundaries"] = extract_boundaries(f.right_boundaries)

    center["left_neighbor"] = extract_neighbors(f.left_neighbors)

    center["right_neighbor"] = extract_neighbors(f.right_neighbors)

    return center


def extract_line(f):
    line = dict()
    f = f.road_line
    line["type"] = WaymoRoadLineType.from_waymo(f.type)
    line["polyline"] = extract_poly(f.polyline)
    return line


def extract_edge(f):
    edge = dict()
    f_ = f.road_edge

    edge["type"] = WaymoRoadEdgeType.from_waymo(f_.type)

    edge["polyline"] = extract_poly(f_.polyline)

    return edge


def extract_stop(f):
    stop = dict()
    f = f.stop_sign
    stop["type"] = MetaDriveType.STOP_SIGN
    stop["lane"] = [x for x in f.lane]
    stop["position"] = np.array([f.position.x, f.position.y, f.position.z], dtype="float32")
    return stop


def extract_crosswalk(f):
    cross_walk = dict()
    f = f.crosswalk
    cross_walk["type"] = MetaDriveType.CROSSWALK
    cross_walk["polygon"] = extract_poly(f.polygon)
    return cross_walk


def extract_bump(f):
    speed_bump_data = dict()
    f = f.speed_bump
    speed_bump_data["type"] = MetaDriveType.SPEED_BUMP
    speed_bump_data["polygon"] = extract_poly(f.polygon)
    return speed_bump_data


def extract_driveway(f):
    driveway_data = dict()
    f = f.driveway
    driveway_data["type"] = MetaDriveType.DRIVEWAY
    driveway_data["polygon"] = extract_poly(f.polygon)
    return driveway_data


def extract_tracks(tracks, sdc_idx, track_length):
    ret = dict()

    class ObjectStateState(typing.TypedDict):
        position: np.ndarray
        length: np.ndarray
        width: np.ndarray
        height: np.ndarray
        heading: np.ndarray
        velocity: np.ndarray
        valid: np.ndarray

    class ObjectState(typing.TypedDict):
        type: typing.Optional[str]
        state: ObjectStateState
        metadata: dict[str, typing.Any]



    def _object_state_template(object_id: str) -> ObjectState:
        return ObjectState(
            type=None,
            state=ObjectStateState(
                # Never add extra dim if the value is scalar.
                position=np.zeros([track_length, 3], dtype=np.float32),
                length=np.zeros([track_length], dtype=np.float32),
                width=np.zeros([track_length], dtype=np.float32),
                height=np.zeros([track_length], dtype=np.float32),
                heading=np.zeros([track_length], dtype=np.float32),
                velocity=np.zeros([track_length, 2], dtype=np.float32),
                valid=np.zeros([track_length], dtype=bool),
            ),
            metadata=dict(track_length=track_length, type=None, object_id=object_id, dataset="waymo")
        )

    for obj in tracks:
        object_id = str(obj.id)

        obj_state = _object_state_template(object_id)

        waymo_string = WaymoAgentType.from_waymo(obj.object_type)  # Load waymo type string
        metadrive_type = MetaDriveType.from_waymo(waymo_string)  # Transform it to Waymo type string
        obj_state["type"] = metadrive_type

        for step_count, state in enumerate(obj.states):

            if step_count >= track_length:
                break

            obj_state["state"]["position"][step_count][0] = state.center_x
            obj_state["state"]["position"][step_count][1] = state.center_y
            obj_state["state"]["position"][step_count][2] = state.center_z

            # l = [state.length for state in obj.states]
            # w = [state.width for state in obj.states]
            # h = [state.height for state in obj.states]
            # obj_state["state"]["size"] = np.stack([l, w, h], 1).astype("float32")
            obj_state["state"]["length"][step_count] = state.length
            obj_state["state"]["width"][step_count] = state.width
            obj_state["state"]["height"][step_count] = state.height

            # heading = [state.heading for state in obj.states]
            obj_state["state"]["heading"][step_count] = state.heading

            obj_state["state"]["velocity"][step_count][0] = state.velocity_x
            obj_state["state"]["velocity"][step_count][1] = state.velocity_y

            obj_state["state"]["valid"][step_count] = state.valid

        obj_state["metadata"]["type"] = metadrive_type

        ret[object_id] = obj_state

    return ret, str(tracks[sdc_idx].id)


def extract_map_features(map_features):
    ret = {}

    for lane_state in map_features:
        lane_id = str(lane_state.id)

        if lane_state.HasField("lane"):
            ret[lane_id] = extract_center(lane_state)

        if lane_state.HasField("road_line"):
            ret[lane_id] = extract_line(lane_state)

        if lane_state.HasField("road_edge"):
            ret[lane_id] = extract_edge(lane_state)

        if lane_state.HasField("stop_sign"):
            ret[lane_id] = extract_stop(lane_state)

        if lane_state.HasField("crosswalk"):
            ret[lane_id] = extract_crosswalk(lane_state)

        if lane_state.HasField("speed_bump"):
            ret[lane_id] = extract_bump(lane_state)

        # Supported only in Waymo dataset 1.2.0
        if lane_state.HasField("driveway"):
            ret[lane_id] = extract_driveway(lane_state)

    return ret


def extract_dynamic_map_states(dynamic_map_states, track_length):
    processed_dynamics_map_states = {}

    def _traffic_light_state_template(object_id):
        return dict(
            type=MetaDriveType.TRAFFIC_LIGHT,
            state=dict(object_state=[None] * track_length),
            lane=None,
            stop_point=np.zeros([
                3,
            ], dtype=np.float32),
            metadata=dict(
                track_length=track_length, type=MetaDriveType.TRAFFIC_LIGHT, object_id=object_id, dataset="waymo"
            )
        )

    for step_count, step_states in enumerate(dynamic_map_states):
        # Each step_states is the state of all objects in one time step
        lane_states = step_states.lane_states

        if step_count >= track_length:
            break

        for object_state in lane_states:
            lane = object_state.lane
            object_id = str(lane)  # Always use string to specify object id

            # We will use lane index to serve as the traffic light index.
            if object_id not in processed_dynamics_map_states:
                processed_dynamics_map_states[object_id] = _traffic_light_state_template(object_id=object_id)

            if processed_dynamics_map_states[object_id]["lane"] is not None:
                assert lane == processed_dynamics_map_states[object_id]["lane"]
            else:
                processed_dynamics_map_states[object_id]["lane"] = lane

            object_state_string = object_state.State.Name(object_state.state)
            processed_dynamics_map_states[object_id]["state"]["object_state"][step_count] = object_state_string

            processed_dynamics_map_states[object_id]["stop_point"][0] = object_state.stop_point.x
            processed_dynamics_map_states[object_id]["stop_point"][1] = object_state.stop_point.y
            processed_dynamics_map_states[object_id]["stop_point"][2] = object_state.stop_point.z

    for obj in processed_dynamics_map_states.values():
        assert len(obj["state"]["object_state"]) == obj["metadata"]["track_length"]

    return processed_dynamics_map_states

# return the nearest point"s index of the line
def nearest_point(point, line):
    dist = np.square(line - point)
    dist = np.sqrt(dist[:, 0] + dist[:, 1])
    return int(np.argmin(dist))

def extract_width(map, polyline, boundary):
    l_width = np.zeros(polyline.shape[0], dtype="float32")
    for b in boundary:
        boundary_int = {k: int(v) if k != "boundary_type" else v for k, v in b.items()}  # All values are int

        b_feat_id = str(boundary_int["boundary_feature_id"])
        lb = map[b_feat_id]
        b_polyline = lb["polyline"][:, :2]

        start_p = polyline[boundary_int["lane_start_index"]]
        start_index = nearest_point(start_p, b_polyline)
        seg_len = boundary_int["lane_end_index"] - boundary_int["lane_start_index"]
        end_index = min(start_index + seg_len, lb["polyline"].shape[0] - 1)
        length = min(end_index - start_index, seg_len) + 1
        self_range = range(boundary_int["lane_start_index"], boundary_int["lane_start_index"] + length)
        bound_range = range(start_index, start_index + length)
        centerLane = polyline[self_range]
        bound = b_polyline[bound_range]
        dist = np.square(centerLane - bound)
        dist = np.sqrt(dist[:, 0] + dist[:, 1])
        l_width[self_range] = dist
    return l_width


def compute_width(map):
    for map_feat_id, lane in map.items():
        if not "LANE" in lane["type"]:
            continue

        width = np.zeros((lane["polyline"].shape[0], 2), dtype="float32")

        width[:, 0] = extract_width(map, lane["polyline"][:, :2], lane["left_boundaries"])
        width[:, 1] = extract_width(map, lane["polyline"][:, :2], lane["right_boundaries"])

        width[width[:, 0] == 0, 0] = width[width[:, 0] == 0, 1]
        width[width[:, 1] == 0, 1] = width[width[:, 1] == 0, 0]

        lane["width"] = width
    return

def convert_waymo_scenario(scenario: scenario_pb2.Scenario) -> ScenarioDescription:
    md_scenario = ScenarioDescription()
    md_scenario[ScenarioDescription.ID] = scenario.scenario_id

    # Please note that SDC track index is not identical to sdc_id.
    # sdc_id is a unique indicator to a track, while sdc_track_index is only the index of the sdc track
    # in the tracks datastructure.

    track_length = len(list(scenario.timestamps_seconds))

    tracks, sdc_id = extract_tracks(scenario.tracks, scenario.sdc_track_index, track_length)

    md_scenario[ScenarioDescription.LENGTH] = track_length

    md_scenario[ScenarioDescription.TRACKS] = tracks

    dynamic_states = extract_dynamic_map_states(scenario.dynamic_map_states, track_length)

    md_scenario[ScenarioDescription.DYNAMIC_MAP_STATES] = dynamic_states

    map_features = extract_map_features(scenario.map_features)
    md_scenario[ScenarioDescription.MAP_FEATURES] = map_features

    compute_width(md_scenario[ScenarioDescription.MAP_FEATURES])

    md_scenario[ScenarioDescription.METADATA] = {}
    md_scenario[ScenarioDescription.METADATA][ScenarioDescription.ID] = md_scenario[ScenarioDescription.ID]
    md_scenario[ScenarioDescription.METADATA][ScenarioDescription.COORDINATE] = MetaDriveType.COORDINATE_WAYMO
    md_scenario[ScenarioDescription.METADATA][ScenarioDescription.TIMESTEP] = np.asarray(list(scenario.timestamps_seconds), dtype=np.float32)
    md_scenario[ScenarioDescription.METADATA][ScenarioDescription.METADRIVE_PROCESSED] = False
    md_scenario[ScenarioDescription.METADATA][ScenarioDescription.SDC_ID] = str(sdc_id)
    md_scenario[ScenarioDescription.METADATA]["dataset"] = "waymo"
    md_scenario[ScenarioDescription.METADATA]["track_length"] = track_length

    return md_scenario