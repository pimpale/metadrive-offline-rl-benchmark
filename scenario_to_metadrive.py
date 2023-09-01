
import scenario
from metadrive.scenario import ScenarioDescription as SD
from metadrive.type import MetaDriveType

SPLIT_KEY = "|"


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

    center["speed_limit_kmh"] = mph_to_kmh(f.speed_limit_mph)

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

    def _object_state_template(object_id):
        return dict(
            type=None,
            state=dict(

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



def convert_scenario(s: scenario.Scenario):
    md_scenario = SD()

    md_scenario[SD.ID] = s.scenario_id
    md_scenario[SD.VERSION] = "0.4"

    track_length = len(s.tracks[s.ego_track_index].states)

    md_scenario[SD.LENGTH] = track_length

    md_scenario[SD.TRACKS] = { str(i): convert_track(t, track_length) for i, t in enumerate(s.tracks)} 

    dynamic_states = extract_dynamic_map_states(scenario.dynamic_map_states, track_length)

    md_scenario[SD.DYNAMIC_MAP_STATES] = dynamic_states

    md_scenario[SD.MAP_FEATURES] = { str(k): convert_map_feature(v) for k,v in s.map_features.items()}

    md_scenario[SD.METADATA] = {}
    md_scenario[SD.METADATA][SD.ID] = md_scenario[SD.ID]
    md_scenario[SD.METADATA][SD.COORDINATE] = MetaDriveType.COORDINATE_WAYMO
    md_scenario[SD.METADATA][SD.TIMESTEP] = np.asarray(list(scenario.timestamps_seconds), dtype=np.float32)
    md_scenario[SD.METADATA][SD.METADRIVE_PROCESSED] = False
    md_scenario[SD.METADATA][SD.SDC_ID] = str(sdc_id)
    md_scenario[SD.METADATA]["dataset"] = "scenariogen"
    md_scenario[SD.METADATA]["scenario_id"] = scenario.scenario_id[:id_end]
    md_scenario[SD.METADATA]["source_file"] = scenario.scenario_id[id_end + 1:]
    md_scenario[SD.METADATA]["track_length"] = track_length

    # obj id
    md_scenario[SD.METADATA]["objects_of_interest"] = [str(obj) for obj in scenario.objects_of_interest]

    track_index = [obj.track_index for obj in scenario.tracks_to_predict]
    track_id = [str(scenario.tracks[ind].id) for ind in track_index]
    track_difficulty = [obj.difficulty for obj in scenario.tracks_to_predict]
    track_obj_type = [tracks[id]["type"] for id in track_id]
    # clean memory
    del scenario
    scenario = None
    return md_scenario

