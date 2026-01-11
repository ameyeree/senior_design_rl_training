# eval_dobot_dqn_poses.py

import time
import numpy as np
import pybullet as p
from stable_baselines3 import DQN
from dobot_ik_env import DobotIKEnv


def set_target_pose(env, target_xyz):
    """
    Set a new target position in the existing env without calling reset().
    Updates:
      - env.target (used for distance/reward/obs)
      - target marker position in PyBullet
      - observation (call this, then env._get_obs() in the main loop)
    """
    target_xyz = np.array(target_xyz, dtype=np.float32)
    env.target = target_xyz

    # Move the magenta target sphere if it exists
    if getattr(env, "target_visual_id", None) is not None:
        p.resetBasePositionAndOrientation(
            env.target_visual_id,
            target_xyz.tolist(),
            [0, 0, 0, 1],
            physicsClientId=env.client_id,
        )


def main():
    # GUI=True so you can see the robot in PyBullet
    env = DobotIKEnv(gui=True)

    model_path = "models/dobot_dqn_ik"
    print(f"Loading model from {model_path}.zip")
    model = DQN.load(model_path, env=env)

    # ---------------------------------------------------------
    # Define five poses (x, y, z) in meters
    poses = [
        [0.100, -0.050,  0.080],
        [0.120,  0.030,  0.100],
        [0.150, -0.020, -0.060],  # keep as you had it, just fixed comma
        [0.090,  0.060,  0.090],
        [0.110, -0.070,  0.070],
    ]
    # ---------------------------------------------------------

    pose_index = 0

    # One reset at the beginning to initialize joints & create target sphere
    obs, info = env.reset()

    # Override initial target with our first pose
    set_target_pose(env, poses[pose_index])
    obs = env._get_obs()

    print("Starting evaluation through 5 poses (no resets between poses)... (Ctrl+C to stop)")

    try:
        while True:
            # deterministic=True -> greedy actions (no exploration)
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            # Slow animation for PyBullet
            time.sleep(1.0 / 60.0)

            if terminated or truncated:
                print(
                    f"Pose {pose_index + 1} finished. "
                    f"Final distance: {info['distance']:.4f}"
                )
                pose_index = (pose_index + 1) % len(poses)
                set_target_pose(env, poses[pose_index])
                env.step_count = 0
                obs = env._get_obs()

    except KeyboardInterrupt:
        print("Stopping evaluation.")

    env.close()


if __name__ == "__main__":
    main()
