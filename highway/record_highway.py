import os
import time
from typing import Optional, List, Tuple

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO
import highway_env  # noqa: F401
from datetime import datetime

from config_highway import get_env_config

ENV_ID = "highway-fast-v0"

# Unique run id so videos are never overwritten
RUN_ID = time.strftime("%Y%m%d-%H%M%S")

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def latest_mp4_in(folder: str) -> str:
    mp4s = [f for f in os.listdir(folder) if f.lower().endswith(".mp4")]
    if not mp4s:
        raise RuntimeError(f"No MP4 found in {folder}")
    mp4s.sort(key=lambda f: os.path.getmtime(os.path.join(folder, f)))
    return os.path.join(folder, mp4s[-1])


def try_episode(model: PPO, seed: int = 42, max_steps: int = 2000) -> Tuple[float, int, bool]:
   
    env = gym.make(ENV_ID, render_mode=None, config=get_env_config())
    obs, _ = env.reset(seed=seed)

    total_reward = 0.0
    steps = 0
    done = False
    crashed = False

    while not done and steps < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += float(reward)
        done = terminated or truncated
        steps += 1

        # Highway-env info often contains "crashed"
        if isinstance(info, dict) and info.get("crashed", False):
            crashed = True

    env.close()
    return total_reward, steps, crashed


def pick_best_seed(model_path: str, seeds: List[int], max_steps: int = 2000) -> int:
   
    model = PPO.load(model_path)

    best_seed = seeds[0]
    best_key = (-1, -1, float("-inf"))

    for s in seeds:
        score, steps, crashed = try_episode(model, seed=s, max_steps=max_steps)
        safe_flag = 0 if crashed else 1
        key = (safe_flag, steps, score)

        if key > best_key:
            best_key = key
            best_seed = s

    return best_seed


def record_one_episode(
    video_dir: str,
    name_prefix: str,
    model_path: Optional[str],
    seed: int = 42,
    max_steps: int = 1500,
) -> str:
   
    ensure_dir(video_dir)

    env = gym.make(ENV_ID, render_mode="rgb_array", config=get_env_config())

    # Stabilize video timing (reduces frame-timing artifacts)
    env.metadata["render_fps"] = 30

    env = RecordVideo(
        env,
        video_folder=video_dir,
        name_prefix=name_prefix,
        episode_trigger=lambda episode_id: episode_id == 0,
        disable_logger=True,
    )

    model = PPO.load(model_path) if model_path else None

    obs, _ = env.reset(seed=seed)
    done = False
    steps = 0

    while not done and steps < max_steps:
        if model is None:
            action = env.action_space.sample()
        else:
            action, _ = model.predict(obs, deterministic=True)

        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        steps += 1

    env.close()

    mp4_path = latest_mp4_in(video_dir)
    print(f"Saved video: {mp4_path}")
    return mp4_path


def stitch_videos(video_paths: List[str], out_path: str) -> None:
    """
    Concatenate videos into one evolution.mp4
    """
    from moviepy import VideoFileClip, concatenate_videoclips

    clips = [VideoFileClip(v) for v in video_paths]
    final = concatenate_videoclips(clips, method="compose")

    ensure_dir(os.path.dirname(out_path) or ".")
    final.write_videofile(
        out_path,
        audio=False,
        fps=30,
        codec="libx264",
        preset="medium",
        ffmpeg_params=["-movflags", "+faststart"],
    )

    final.close()
    for c in clips:
        c.close()

    print(f"Saved stitched evolution video: {out_path}")


if __name__ == "__main__":
    # Base folders
    ensure_dir("videos")
    ensure_dir(f"videos/raw/{RUN_ID}")

    # 1) Untrained (random policy)
    v_untrained = record_one_episode(
        video_dir=f"videos/raw/{RUN_ID}/untrained",
        name_prefix="untrained",
        model_path=None,
        seed=42,
        max_steps=800,  
    )

    # 2) Half-trained
    v_half = record_one_episode(
        video_dir=f"videos/raw/{RUN_ID}/half",
        name_prefix="half",
        model_path="models/ppo_half_new.zip",
        seed=42,
        max_steps=1500,
    )

    # 3) Fully-trained (fine-tuned) 
    full_model_path = "models/ppo_full_new.zip"
    
    best_seed = pick_best_seed(full_model_path, seeds=[10, 20, 30, 40, 50], max_steps=2000)
    print(f"Selected best seed for FULL video: {best_seed}")

    v_full = record_one_episode(
        video_dir=f"videos/raw/{RUN_ID}/full",
        name_prefix="full",
        model_path=full_model_path,
        seed=best_seed,
        max_steps=2000,
    )

    # Stitch all into one evolution video
    stitch_videos(
        [v_untrained, v_half, v_full],
        out_path="videos/evolution.mp4",
    )
