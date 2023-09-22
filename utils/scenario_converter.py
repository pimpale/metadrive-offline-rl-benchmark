import numpy as np
from metadrive.scenario import ScenarioDescription as SD
from metadrive.type import MetaDriveType

import utils.scenario as scenario
from utils.env import State, normalize_angle
from scipy.ndimage import gaussian_filter1d

def convert_2D_points(t: np.ndarray) -> np.ndarray:
    # convert from (2, N) to (3, N)
    return np.pad(t, ((0, 0), (0, 1)), mode="constant", constant_values=0)

def convert_map_feature(t: scenario.MapFeature, id_val: str) -> dict:
    match t:
        case scenario.LaneCenter():
            return {
                "type": MetaDriveType.LANE_SURFACE_STREET,
                "polyline": convert_2D_points(t.polyline),
            }
        case scenario.RoadLine():
            return {
                "type": t.kind,
                "polyline": convert_2D_points(t.polyline),
            }
        case scenario.RoadEdge():
            return {
                "type": t.kind,
                "polyline": convert_2D_points(t.polyline),
            }
        case scenario.StopSign():
            return {
                "type": MetaDriveType.STOP_SIGN,
                "position": np.append(t.position, 0),
                "lane": t.lane,
            }
        case scenario.Crosswalk():
            return {
                "type": MetaDriveType.CROSSWALK,
                "polygon": convert_2D_points(t.polygon),
            }
        case scenario.SpeedBump():
            return {
                "type": MetaDriveType.SPEED_BUMP,
                "polygon": convert_2D_points(t.polygon),
            }
        case scenario.Driveway():
            return {
                "type": MetaDriveType.DRIVEWAY,
                "polygon": convert_2D_points(t.polygon),
            }
        case _:
            raise ValueError(f"Unknown map feature type: {t}")

def convert_dynamic_map_state(t: scenario.DynamicState, id_val:str, track_length:int) -> dict:
    return {
        "type": MetaDriveType.TRAFFIC_LIGHT,
        "state": {
            "object_state": t.states,
        },
        "lane": t.lane,
        "stop_point": np.append(t.position, 0),
        "metadata": {
            "track_length": track_length,
            "type": MetaDriveType.TRAFFIC_LIGHT,
            "object_id": id_val,
            "dataset": "waymo"
        }
    }

def convert_track(t: scenario.AgentTrack, track_id:str, track_length:int) -> dict:
    return {
        "type": t.object_type,
        "state": {
            "position": convert_2D_points(np.stack([s.position for s in t.states])),
            "length": np.zeros(track_length) + t.length,
            "width": np.zeros(track_length) + t.width,
            "height": np.zeros(track_length) + t.height,
            "heading": np.array([s.heading for s in t.states]),
            "velocity": np.stack([s.velocity for s in t.states]),
            "valid": np.array([s.valid for s in t.states])
        },
        "metadata": {
            "track_length": track_length,
            "type": t.object_type,
            "object_id": track_id,
            "dataset": "waymo"
        }
    }

def convert_scenario(s: scenario.Scenario):
    track_length = len(s.tracks[s.ego_track_index].states)

    md_scenario = SD()

    md_scenario[SD.ID] = s.scenario_id
    md_scenario[SD.VERSION] = "0.4"
    md_scenario[SD.LENGTH] = track_length
    md_scenario[SD.TRACKS] = { str(i): convert_track(t, str(i), track_length) for i, t in enumerate(s.tracks)} 
    md_scenario[SD.DYNAMIC_MAP_STATES] = { str(s.lane): convert_dynamic_map_state(s, str(s.lane), track_length) for s in s.dynamic_state }
    md_scenario[SD.MAP_FEATURES] = { str(k): convert_map_feature(v, str(k)) for k, v in enumerate(s.map_features)}
    md_scenario[SD.METADATA] = {}
    md_scenario[SD.METADATA][SD.ID] = md_scenario[SD.ID]
    md_scenario[SD.METADATA][SD.COORDINATE] = MetaDriveType.COORDINATE_WAYMO
    md_scenario[SD.METADATA][SD.TIMESTEP] = np.arange(track_length)*0.1
    md_scenario[SD.METADATA][SD.METADRIVE_PROCESSED] = False
    md_scenario[SD.METADATA][SD.SDC_ID] = str(s.ego_track_index)
    md_scenario[SD.METADATA]["dataset"] = "scenariogen"
    md_scenario[SD.METADATA]["scenario_id"] = s.scenario_id
    md_scenario[SD.METADATA]["source_file"] = "idk"
    md_scenario[SD.METADATA]["track_length"] = track_length

    return md_scenario


def extract_trajectory(track: scenario.AgentTrack) -> list[State]:
    vx = np.array([state.velocity[0] for state in track.states if state.valid], dtype=np.float32)
    vy = np.array([state.velocity[1] for state in track.states if state.valid], dtype=np.float32)
    heading = np.array([state.heading for state in track.states if state.valid], dtype=np.float32)
    
    # filter
    vx = gaussian_filter1d(vx, sigma=3)
    vy = gaussian_filter1d(vy, sigma=3)
    
    # reconstruct heading before smoothing
    heading_reconstructed = np.cumsum(normalize_angle(np.diff(heading, prepend=0)))
    heading = normalize_angle(gaussian_filter1d(heading_reconstructed, sigma=3))
    
    return [
        State(
            heading=heading,
            velocity=(vx, vy)
        )
        for vx, vy, heading in zip(vx, vy, heading)
    ]