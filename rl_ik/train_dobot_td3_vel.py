# train_dobot_td3_vel.py

import os

from stable_baselines3 import TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np

from dobot_ik_vel_env import DobotIKVelEnv


def main():
    env = DobotIKVelEnv(gui=False)
    env = Monitor(env)

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.1 * np.ones(n_actions),
    )

    model = TD3(
        "MlpPolicy",
        env,
        action_noise=action_noise,
        learning_rate=3e-4,
        buffer_size=200_000,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        train_freq=(1, "step"),
        gradient_steps=1,
        verbose=1,
        tensorboard_log="./logs_dobot_td3_vel/",
    )

    model.learn(total_timesteps=300_000)

    os.makedirs("models", exist_ok=True)
    model.save("models/dobot_td3_vel")

    env.close()
    print("Training finished. Model saved to models/dobot_td3_vel.zip")


if __name__ == "__main__":
    main()
