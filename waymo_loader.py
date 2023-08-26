import os
# tensorflow don't allocate all gpu memory right away
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf
from waymo_protos import scenario_pb2
import numpy as np
from env import State

def parse_scenario(scenario: scenario_pb2.Scenario) -> list[State]:
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

def parse_file(file_path: str) -> list[list[State]]:
    trajectories_in_file = []
    scenario = scenario_pb2.Scenario()
    for data in tf.data.TFRecordDataset(file_path, compression_type="").as_numpy_iterator():
        scenario.ParseFromString(data)
        trajectories_in_file.append(parse_scenario(scenario))
    return trajectories_in_file