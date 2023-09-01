from dataclasses import dataclass
import numpy as np
import numpy.typing as npt

@dataclass
class AgentState:
    """
    A state of an agent
    """
    position: npt.NDArray[np.float32] # in (2,)
    velocity: npt.NDArray[np.float32] # in (2,)
    heading: float
    valid: bool


@dataclass
class AgentTrack:
    """
    An agent is a vehicle or pedestrian
    Each state is sampled at 10Hz
    """
    object_type: str
    length: float
    width: float
    height: float
    states: list[AgentState]

@dataclass
class LaneCenter:
    polyline: npt.NDArray[np.float32] # in (N, 2``)

@dataclass
class RoadLine:
    kind: str
    polyline: npt.NDArray[np.float32] # in (N, 2)

@dataclass
class RoadEdge:
    kind: str
    polyline: npt.NDArray[np.float32] # in (N, 2)

@dataclass
class StopSign:
    lane: list[int]
    position: npt.NDArray[np.float32] # in (2,)

@dataclass
class Crosswalk:
    polygon: npt.NDArray[np.float32]

@dataclass
class SpeedBump:
    polygon: npt.NDArray[np.float32]

@dataclass
class Driveway:
    polygon: npt.NDArray[np.float32]

MapFeature = LaneCenter | RoadLine | RoadEdge | StopSign | Crosswalk | SpeedBump | Driveway


@dataclass
class TrafficLight:
    lane: int
    position: npt.NDArray[np.float32] # in (2,)
    states: list[str]

DynamicState = TrafficLight

@dataclass
class Scenario:
    # the scenario id
    scenario_id: str
    # the index of the ego track
    ego_track_index: int
    # all tracks must be the same length
    tracks: list[AgentTrack]
    # map features
    map_features: dict[int, MapFeature]
    # dynamic state
    dynamic_state: list[DynamicState]