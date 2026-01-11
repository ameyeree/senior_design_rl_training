# train_dobot_sac_vel.py

import os

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from dobot_ik_vel_env import DobotIKVelEnv


def main():
    env = DobotIKVelEnv(gui=False)
    env = Monitor(env)

    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=200_000,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        train_freq=(1, "step"),
        gradient_steps=1,
        verbose=1,
        tensorboard_log="./logs_dobot_sac_vel/",
        device="auto"
    )

    model.learn(total_timesteps=300_000)

    os.makedirs("models", exist_ok=True)
    model.save("models/dobot_sac_vel")

    env.close()
    print("Training finished. Model saved to models/dobot_sac_vel.zip")


if __name__ == "__main__":
    main()
