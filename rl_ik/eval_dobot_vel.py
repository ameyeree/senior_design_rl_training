# eval_dobot_td3_vel.py  (or eval_dobot_sac_vel.py)

import time
from stable_baselines3 import TD3, SAC  # or SAC
from dobot_ik_vel_env import DobotIKVelEnv


# model = "dobot_td3_vel"
model = "dobot_sac_vel"
def main():
    env = DobotIKVelEnv(gui=True)

    # model_path = f"models/dobot_td3_vel"  # or "models/dobot_sac_vel"
    model_path = "models/dobot_sac_vel"
    # print(f"Loading model from dobot_td3_vel.zip")
    # model = TD3.load(model_path, env=env)  # or SAC.load(...)
    model = SAC.load(model_path, env=env)

    obs, info = env.reset()
    print("Starting evaluation... (Ctrl+C to stop)")

    try:
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            time.sleep(1.0 / 60.0)

            if terminated or truncated:
                print(f"Episode finished. Final distance: {info['distance']:.4f}")
                time.sleep(1)
                obs, info = env.reset()
    except KeyboardInterrupt:
        print("Stopping evaluation.")

    env.close()


if __name__ == "__main__":
    main()
