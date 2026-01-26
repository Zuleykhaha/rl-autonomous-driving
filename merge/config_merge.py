from typing import Dict

def get_env_config() -> Dict:
    return {
        "observation": {"type": "TimeToCollision", "horizon": 10},
        "action": {"type": "DiscreteMetaAction"},

        "simulation_frequency": 15,
        "policy_frequency": 15,

        "duration": 100,

        "collision_terminal": True,
        "offroad_terminal": True,

        "vehicles_density": 1.5,

        "collision_reward": -1.0,
        "right_lane_reward": 0.1,
        "high_speed_reward": 0.4,
        "reward_speed_range": [20, 30],

        "screen_width": 600,
        "screen_height": 150,
        "centering_position": [0.3, 0.5],
        "scaling": 5.5,
    }
