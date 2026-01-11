# train_dobot_dqn.py

import os

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

from dobot_ik_env import DobotIKEnv


def main():
    # Create env (no GUI during training)
    env = DobotIKEnv(gui=False)
    env = Monitor(env)  # logs episode stats (length, reward, etc.)

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-3,
        buffer_size=50_000,
        batch_size=64,
        gamma=0.99,
        exploration_fraction=0.3,
        exploration_final_eps=0.05,
        target_update_interval=1_000,
        train_freq=4,
        gradient_steps=1,
        verbose=1,
        tensorboard_log="./logs_dobot_dqn/",
    )

    # Train the agent
    model.learn(total_timesteps=50_000)

    # Save the trained model
    os.makedirs("models", exist_ok=True)
    model.save("models/dobot_dqn_ik")

    env.close()
    print("Training finished. Model saved to models/dobot_dqn_ik.zip")


if __name__ == "__main__":
    main()
