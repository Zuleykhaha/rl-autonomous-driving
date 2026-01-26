import argparse
import os
from typing import Optional

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import highway_env  # noqa: F401

from config_highway import get_env_config


ENV_ID = "highway-fast-v0"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def train(
    model_out: str,
    timesteps: int,
    n_envs: int,
    seed: int = 42,
    load_path: Optional[str] = None,
) -> None:
    """
    Trains PPO on highway-fast-v0 WITHOUT rendering and saves the model.

    model_out: path WITHOUT .zip, e.g. "models/ppo_half"
    load_path: optional path to an existing .zip model to resume training (fine-tuning)
    """
    ensure_dir(os.path.dirname(model_out) or ".")

    env = make_vec_env(
        ENV_ID,
        n_envs=n_envs,
        env_kwargs={"config": get_env_config()},
        monitor_dir="./logs/",
        seed=seed,
    )

    if load_path:
        # Fine-tuning: resume from a pretrained policy
        print(f"Loading model for fine-tuning: {load_path}")
        model = PPO.load(load_path, env=env, device="cpu")
    else:
        # Train from scratch
        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            device="cpu",
            verbose=1,
            tensorboard_log="./tensorboard/",
            seed=seed,
        )

    model.learn(total_timesteps=timesteps)
    model.save(model_out)

    env.close()
    print(f"Saved model: {model_out}.zip")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="models/ppo_half", help="Output model path (without .zip)")
    parser.add_argument("--timesteps", type=int, default=20_000)
    parser.add_argument("--n_envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--load",
        type=str,
        default=None,
        help="Optional path to an existing .zip model to resume training (fine-tuning)",
    )
    args = parser.parse_args()

    train(
        model_out=args.out,
        timesteps=args.timesteps,
        n_envs=args.n_envs,
        seed=args.seed,
        load_path=args.load,
    )
