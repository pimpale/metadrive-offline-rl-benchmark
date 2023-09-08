import os
import json 
import numpy as np
from collections import defaultdict
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Slerp
from scipy.ndimage import gaussian_filter1d

from utils.env import State


def read_json(path: str) -> list:
    with open(path, "r") as f:
        return json.load(f)

def load_nuscene_data(path:str):
    # load necessary files
    ego_pose_json = read_json(os.path.join(os.path.expanduser(path), "ego_pose.json"))
    sample_json  = read_json(os.path.join(os.path.expanduser(path), "sample.json"))
    sample_data_json = read_json(os.path.join(os.path.expanduser(path), "sample_data.json"))


    # build scene lookup dict
    scene_token_by_sample_token: dict[str, str] = { sample["token"]: sample["scene_token"] for sample in sample_json }

    # build ego_pose lookup dict
    ego_pose_by_ego_pose_token: dict[str, dict] = { ego_pose["token"]: ego_pose for ego_pose in ego_pose_json }

    # build list of ego_pose_tokens for each scene
    ego_pose_token_by_scene_token: defaultdict[str, set[str]] = defaultdict(set)
    for sample_data in sample_data_json:
        scene_token = scene_token_by_sample_token[sample_data["sample_token"]]
        ego_pose_token_by_scene_token[scene_token].add(sample_data["token"])

    # build trajectories
    trajectories: list[list[State]] = []

    for scene_token, ego_pose_tokens in ego_pose_token_by_scene_token.items():
        # sort ego poses by timestamp and deduplicate
        ego_poses = [ego_pose_by_ego_pose_token[ego_pose_token] for ego_pose_token in ego_pose_tokens]
        ego_poses = sorted(ego_poses, key=lambda x: x["timestamp"])
        ego_poses = [ego_poses[i] for i in range(len(ego_poses)) if i == 0 or ego_poses[i]["timestamp"] - ego_poses[i-1]["timestamp"] > 0.001*1e6]

        # gather data
        time_micros: list[int] = []
        quaternions: list[list[float]] = []
        xs: list[float] = []
        ys: list[float] = []
        zs: list[float] = []

        for ego_pose in ego_poses:
            time_micros.append(ego_pose["timestamp"])

            qw, qx, qy, qz = ego_pose["rotation"] 
            quaternions.append([qx, qy, qz, qw])

            x, y, z = ego_pose["translation"]
            xs.append(x)
            ys.append(y)
            zs.append(z)

        if len(time_micros) < 2:
            print(f"Skipping scene {scene_token} because it has no data")
            continue

        # convert time to seconds
        times = np.array(time_micros, dtype=np.float64) / 1e6

        # sample at 10Hz
        sample_times = np.arange(times[0], times[-1], 0.1)

        # get headings at sampled times
        rotation_interpolator = Slerp(times, Rotation.from_quat(quaternions))
        headings = rotation_interpolator(sample_times).as_euler('xyz')[:, 2]

        # get velocities at sampled times
        tdiff = np.diff(times)
        xvel_smoothed = gaussian_filter1d(np.diff(xs) / tdiff, sigma=32)
        yvel_smoothed = gaussian_filter1d(np.diff(ys) / tdiff, sigma=32)

        x_vel = np.interp(sample_times, times[:-1], xvel_smoothed)
        y_vel = np.interp(sample_times, times[:-1], yvel_smoothed)

        velocities = np.stack([x_vel, y_vel], axis=1)

        # create trajectory
        trajectory = [State(heading=h, velocity=v) for h, v in zip(headings, velocities)]
        trajectories.append(trajectory)
    
    return trajectories