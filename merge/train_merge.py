import argparse
import os
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
import highway_env  # noqa: F401

from config_merge import get_env_config

ENV_ID = "merge-v0"


def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist"""
    os.makedirs(path, exist_ok=True)


def train(out: str, timesteps: int, load: str | None = None, seed: int = 42):

    # Ensure output directory exists
    ensure_dir(os.path.dirname(out))
    
    # Ensure logs directory exists
    ensure_dir("merge/logs")
    
    # Create environment
    env = gym.make(
        ENV_ID,
        render_mode=None,
        config=get_env_config()
    )
    
    # Wrap with Monitor for logging (needed for plot_rewards.py)
    env = Monitor(env, filename="merge/logs/monitor.csv")
    
    if load:
        print(f" Loading model for fine-tuning: {load}")
        model = PPO.load(load, env=env, device="cpu")
    else:
        print("Training new model from scratch")
        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            clip_range=0.2,
            verbose=1,
            device="cpu",
            tensorboard_log="tensorboard/merge",
            seed=seed,
        )
    
    print(f" Starting training for {timesteps} timesteps...")
    model.learn(total_timesteps=timesteps)
    
    # Save model
    model.save(out)
    print(f" Saved model: {out}.zip")
    
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO on Merge-v0")
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output model path (without .zip), e.g., 'merge/models/ppo_half'"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        required=True,
        help="Total training timesteps, e.g., 15000"
    )
    parser.add_argument(
        "--load",
        type=str,
        default=None,
        help="Optional: path to existing model for fine-tuning"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    
    args = parser.parse_args()
    
    train(
        out=args.out,
        timesteps=args.timesteps,
        load=args.load,
        seed=args.seed,
    )