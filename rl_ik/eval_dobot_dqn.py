# eval_dobot_dqn.py

import time

from stable_baselines3 import DQN
from dobot_ik_env import DobotIKEnv


def main():
    # GUI=True so you can see the robot in PyBullet
    env = DobotIKEnv(gui=True)

    model_path = "models/dobot_dqn_ik"
    print(f"Loading model from {model_path}.zip")
    model = DQN.load(model_path, env=env)

    obs, info = env.reset()
    print("Starting evaluation... (Ctrl+C to stop)")

    try:
        while True:
            # deterministic=True -> greedy policy (no exploration)
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            # slow it down so you can see
            time.sleep(1.0 / 60.0)

            if terminated or truncated:
                print(f"Episode finished. Final distance: {info['distance']:.4f}")
                obs, info = env.reset()
    except KeyboardInterrupt:
        print("Stopping evaluation.")

    env.close()


if __name__ == "__main__":
    main()
