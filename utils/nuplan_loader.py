import numpy as np
import sqlite3
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Slerp
from scipy.ndimage import gaussian_filter1d

from env import State


def parse_file(file_path: str) -> list[list[State]]:
    time_micros = []
    quaternions = []
    xs = []
    ys = []

    # gather headings and positions from sqlite3 database
    with sqlite3.connect(file_path) as conn:
        for (timestamp, qw, qx, qy, qz, x, y) in conn.cursor().execute("SELECT timestamp, qw, qx, qy, qz, x, y FROM ego_pose"):
            time_micros.append(timestamp)
            quaternions.append([qx, qy, qz, qw])
            xs.append(x)
            ys.append(y)

    if len(time_micros) == 0:
        return []
    
    # convert time to seconds
    times = np.array(time_micros, dtype=np.float64) / 1e6

    # sample at 10Hz
    sample_times = np.arange(times[0], times[-1], 0.1)

    # get headings at sampled times
    rotation_interpolator = Slerp(times, Rotation.from_quat(quaternions))
    headings = rotation_interpolator(sample_times).as_euler('xyz')[:, 2]

    # get velocities at sampled times
    xvel_smoothed = gaussian_filter1d(np.diff(xs) / np.diff(times), sigma=32)
    yvel_smoothed = gaussian_filter1d(np.diff(ys) / np.diff(times), sigma=32)

    x_vel = np.interp(sample_times, times[:-1], xvel_smoothed)
    y_vel = np.interp(sample_times, times[:-1], yvel_smoothed)

    velocities = np.stack([x_vel, y_vel], axis=1)

    # create trajectory
    trajectory = [State(heading=h, velocity=v) for h, v in zip(headings, velocities)]
    return [trajectory]
