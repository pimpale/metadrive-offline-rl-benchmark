import typing
import dataclasses
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
import metadrive
from metadrive import MetaDriveEnv
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class State:
    heading: float
    velocity: tuple[float, float]

@dataclass
class Action:
    steer: float
    throttle: float

@dataclass
class Transition:
    state: State
    action: Action
    next_state: State

    @staticmethod
    def from_json(data: dict):
        return Transition(State(**data['state']), Action(steer=data['action'][0], throttle=data['action'][1]), State(**data['next_state']))

Observation: typing.TypeAlias = tuple[State, State]

def normalize_angle(angle: float) -> float:
    """
    Normalize the angle to [-pi, pi)
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi

def get_metadrive_state(env: MetaDriveEnv) -> State:
    return State(heading=env.vehicle.heading_theta, velocity=env.vehicle.velocity[:2])

def next_state(env: MetaDriveEnv, s: State, a: Action) -> State:
    """
    runs the policy and returns the total reward
    """
    # reset
    env.reset()
    env.vehicle.set_position(env.vehicle.position, height=0.49)

    # allow car to settle
    for _ in range(5):
        env.step([0,0])

    # set the initial state
    env.vehicle.set_velocity(s.velocity)
    env.vehicle.set_heading_theta(s.heading)
    
    # run the simulator
    env.step(a)

    # get the new state
    s_prime = get_metadrive_state(env)

    # allow car to settle (if rendering)
    if env.config.use_render:
        for _ in range(10):
            env.step([0,0])

    return s_prime

def state_batch_to_tensor(states: list[State], device: torch.device) -> torch.Tensor:
    """
    Reshape the state from State to a tensor of shape (batch_size, 4)
    """
    velocities = torch.tensor(np.array([st.velocity for st in states]), dtype=torch.float32, device=device)
    heading = torch.tensor([st.heading for st in states], dtype=torch.float32, device=device)
    return torch.cat([velocities, torch.cos(heading).unsqueeze(1), torch.sin(heading).unsqueeze(1)], dim=1)

def action_batch_to_tensor(actions: list[Action], device: torch.device) -> torch.Tensor:
    """
    Reshape the action from Action to a tensor of shape (batch_size, 2)
    """
    return torch.tensor(np.array([(a.steer, a.throttle) for a in actions]), dtype=torch.float32, device=device)

def obs_batch_to_tensor(obs: list[Observation], device: torch.device) -> torch.Tensor:
    """
    Reshape the observation from tuple[State, State] to a tensor of shape (batch_size, 4, 2)
    """

    observations = []

    for st0, st1 in obs:
        observations.append(np.array([
            [st0.velocity[0], st1.velocity[0]], 
            [st0.velocity[1], st1.velocity[1]],
            [np.cos(st0.heading), np.cos(st1.heading)],
            [np.sin(st0.heading), np.sin(st1.heading)],
        ]))

    return torch.tensor(np.stack(observations), dtype=torch.float32, device=device)


def transition_batch_to_json(transition: list[Transition]) -> list[dict]:
    """
    Convert a batch of transitions to a list of json objects
    """
    return [dataclasses.asdict(t) for t in transition]


def transition_json_to_batch(data: list[dict]) -> list[Transition]:
    """
    Convert a list of json objects to a batch of transitions
    """
    return [Transition.from_json(d) for d in data]