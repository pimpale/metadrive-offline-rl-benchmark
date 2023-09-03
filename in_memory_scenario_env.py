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

import scenario
from scenario_converter import convert_scenario


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

    def __init__(self, config, s: scenario.Scenario):
        super().__init__(config)
        self.scenario = s

    def done_function(self, vehicle_id: str):
        return False, {}

    def cost_function(self, vehicle_id: str):
        return 0, {}
    
    def reward_function(self, vehicle_id: str):
        return 0, {}

    def _get_observations(self):
        return {DEFAULT_AGENT: self.get_single_observation()}
    
    def setup_engine(self):
        self.engine.register_manager("agent_manager", self.agent_manager)
        self.engine.register_manager("map_manager", ScenarioMapManager())
        self.engine.register_manager("scenario_traffic_manager", ScenarioTrafficManager())
        self.engine.register_manager("scenario_light_manager", ScenarioLightManager()) 
        self.engine.register_manager("data_manager", InMemoryScenarioDataManager(self.scenario))
