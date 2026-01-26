from typing import Dict


def get_env_config() -> Dict:
    return {
        "vehicles_count": 70,
        "lanes_count": 4,
        "duration": 80,

        "simulation_frequency": 15,
        "policy_frequency": 15, 

        "collision_terminal": True,
        "offroad_terminal": True,

        "collision_reward": -2.0,
        "high_speed_reward": 1.5,
        "lane_change_reward": 0.0,
        "reward_speed_range": [30, 45],

        "observation": {"type": "TimeToCollision"},
        "action": {"type": "DiscreteMetaAction"},
    }