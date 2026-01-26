import os
import time
from typing import Optional, List

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO
import highway_env  # noqa: F401

from config_merge import get_env_config

ENV_ID = "merge-v0"

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


def record_one_episode(
    video_dir: str,
    name_prefix: str,
    model_path: Optional[str],
    seed: int = 42,
    max_steps: int = 1500,
) -> str:
    
    ensure_dir(video_dir)

    env = gym.make(ENV_ID, render_mode="rgb_array", config=get_env_config())
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
    time.sleep(0.3)  

    mp4_path = latest_mp4_in(video_dir)
    print(f"Saved video: {mp4_path} ({steps} steps)")
    return mp4_path


def stitch_videos(video_paths: List[str], out_path: str) -> None:
    """
    Concatenate videos into one file.
    """
    from moviepy import VideoFileClip, concatenate_videoclips

    clips = [VideoFileClip(v) for v in video_paths]
    final = concatenate_videoclips(clips, method="compose")

    ensure_dir(os.path.dirname(out_path) or ".")
    final.write_videofile(out_path, audio=False, fps=30)

    final.close()
    for c in clips:
        c.close()

    print(f"Saved stitched video: {out_path}")


if __name__ == "__main__":
    print("Starting MERGE environment video recording...")

    # Base folders 
    ensure_dir("videos")
    ensure_dir(f"videos/raw/{RUN_ID}")

   
    HALF_MODEL = "models/ppo_merge_half.zip"
    FULL_MODEL = "models/ppo_merge_full.zip"

    # 1) Untrained
    print("\nRecording UNTRAINED agent...")
    v_untrained = record_one_episode(
        video_dir=f"videos/raw/{RUN_ID}/untrained",
        name_prefix="untrained",
        model_path=None,
        seed=42,
        max_steps=800,
    )

    # 2) Half-trained
    print("\nRecording HALF-TRAINED agent...")
    v_half = record_one_episode(
        video_dir=f"videos/raw/{RUN_ID}/half",
        name_prefix="half",
        model_path=HALF_MODEL,
        seed=42,
        max_steps=1500,
    )

    # 3) Full-trained
    print("\nRecording FULL-TRAINED agent...")
    v_full = record_one_episode(
        video_dir=f"videos/raw/{RUN_ID}/full",
        name_prefix="full",
        model_path=FULL_MODEL,
        seed=42,
        max_steps=2000,
    )

    # Stitch
    print("\nStitching videos...")
    stitch_videos(
        [v_untrained, v_half, v_full],
        out_path="videos/evolution_merge.mp4",
    )

    print("\nDONE! Check videos/evolution_merge.mp4")
