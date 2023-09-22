from metadrive.envs.base_env import BaseEnv
from metadrive.constants import DEFAULT_AGENT
from metadrive.utils import Config
from metadrive.manager.scenario_map_manager import ScenarioMapManager
from metadrive.manager.base_manager import BaseManager
from metadrive.scenario.scenario_description import ScenarioDescription
from metadrive.manager.scenario_light_manager import ScenarioLightManager
from metadrive.manager.scenario_traffic_manager import ScenarioTrafficManager
from metadrive.component.vehicle_navigation_module.trajectory_navigation import TrajectoryNavigation
from metadrive.scenario import ScenarioDescription as SD
from metadrive.policy.replay_policy import ReplayEgoCarPolicy
from metadrive.obs.top_down_obs_multi_channel import TopDownMultiChannel
from typing import Optional
import utils.scenario as scenario
from utils.scenario_converter import convert_scenario


class InMemoryScenarioDataManager(BaseManager):
    def __init__(self, s:scenario.Scenario):
        super().__init__()
        self.current_scenario = SD(convert_scenario(s))
        self.current_scenario_length = self.current_scenario[SD.LENGTH]
    
    def get_scenario(self, i, should_copy=False):
        return self.current_scenario

class InMemoryScenarioEnv(BaseEnv):
    @classmethod
    def default_config(cls) -> Config:
        config = super().default_config()
        config.update({
            "start_seed": 0,
            "num_scenarios": 1,
            "start_scenario_index": 0,
            "top_down_show_real_size": True,
            "no_map": False,
            "store_map": False,
            "need_lane_localization": True,
            "vehicle_config": dict(
                lidar=dict(num_lasers=120, distance=50),
                lane_line_detector=dict(num_lasers=0, distance=50),
                side_detector=dict(num_lasers=12, distance=50),
                show_dest_mark=True,
                navigation_module=TrajectoryNavigation,
            ),
            "max_lateral_dist": 4,
            # whether or not to base vehicle class purely on size or whether to evenly sample from all vehicle classes
            "even_sample_vehicle_class": False,
            # do show traffic lights
            "no_light": False,
            "skip_missing_light": False,
            "static_traffic_object": True,
            "no_static_vehicles": False,
            # if true, then any vehicle that is overlapping with another vehicle will be filtered 
            "filter_overlapping_car": False,
            # whether to use the default vehicle model
            "default_vehicle_in_traffic":False,
            "reactive_traffic": False,
        })
        return config


    s: Optional[scenario.Scenario] = None

    def __init__(self, config):
        super().__init__(config)

    def done_function(self, vehicle_id: str):
        return False, {}

    def cost_function(self, vehicle_id: str):
        return 0, {}
    
    def reward_function(self, vehicle_id: str):
        return 0, {}

    def _get_observations(self):
        return {DEFAULT_AGENT: self.get_single_observation()}
    

    def set_scenario(self, s: scenario.Scenario):
        self.s = s

    def setup_engine(self):
        assert self.s is not None, "Please set scenario first!"
        self.engine.register_manager("agent_manager", self.agent_manager)
        self.engine.register_manager("map_manager", ScenarioMapManager())
        self.engine.register_manager("scenario_traffic_manager", ScenarioTrafficManager())
        self.engine.register_manager("scenario_light_manager", ScenarioLightManager()) 
        self.engine.register_manager("data_manager", InMemoryScenarioDataManager(self.s))


class TopDownInMemoryScenarioEnv(InMemoryScenarioEnv):
    @classmethod
    def default_config(cls) -> Config:
        config = super().default_config()
        # config["vehicle_config"]["lidar"].update({"num_lasers": 0, "distance": 0})  # Remove lidar
        config.update(
            {
                "frame_skip": 5,
                "frame_stack": 3,
                "post_stack": 5,
                "rgb_clip": True,
                "resolution_size": 84,
                "distance": 30
            }
        )
        return config

    def get_single_observation(self, _=None):
        return TopDownMultiChannel(
            self.config["vehicle_config"],
            onscreen=self.config["use_render"],
            clip_rgb=self.config["rgb_clip"],
            frame_stack=self.config["frame_stack"],
            post_stack=self.config["post_stack"],
            frame_skip=self.config["frame_skip"],
            resolution=(self.config["resolution_size"], self.config["resolution_size"]),
            max_distance=self.config["distance"]
        )
