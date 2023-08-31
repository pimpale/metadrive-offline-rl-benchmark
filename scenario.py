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
    kind: str
    length: float
    width: float
    height: float
    states: list[AgentState]

@dataclass
class LaneCenter:
    kind: str
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
    lane: int
    position: tuple[float, float]

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
class Scenario:
    # all tracks must be the same length
    agents: list[AgentTrack]
    # map features
    features: dict[int, MapFeature]