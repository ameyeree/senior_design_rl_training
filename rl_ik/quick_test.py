import time
import numpy as np
import pybullet as p
from dobot_ik_env import DobotIKEnv


def set_neutral_pose(env):
    """
    Set a safe, reasonable neutral pose for the Dobot Magician
    in conceptual joint space (theta: [base, shoulder, elbow] in radians).
    """
    # A decent-looking pose within theta_low/theta_high
    theta_neutral = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    env.theta = theta_neutral.copy()
    q_urdf = env._concept_to_urdf_joints(env.theta)
    env._apply_urdf_joints(q_urdf)

    p.stepSimulation(physicsClientId=env.client_id)
    ee_pos = env._get_ee_pos()
    print("Neutral theta:", env.theta)
    print("Neutral q_urdf:", q_urdf)
    print("Neutral EE position:", ee_pos)


def setup_camera():
    """
    Put the camera in a nice spot looking at the base of the robot.
    On Windows 11, use right-click drag / ALT+drag to rotate if needed.
    """
    p.resetDebugVisualizerCamera(
        cameraDistance=0.6,
        cameraYaw=45,
        cameraPitch=-35,
        cameraTargetPosition=[0, 0, 0.1],
    )


def main():
    # Start environment in GUI mode so you can see the robot
    env = DobotIKEnv(gui=True)

    print("DobotIKEnv initialized.")
    print("Conceptual joints (theta indices): 0=base, 1=shoulder, 2=elbow")
    print("Controls:")
    print("  1 / 2 : base + / -")
    print("  3 / 4 : shoulder + / -")
    print("  5 / 6 : elbow + / -")
    print("  q     : quit")

    # Reset once and then override to a neutral pose
    env.reset()
    set_neutral_pose(env)
    setup_camera()

    # Key mapping: key -> (concept_joint_idx, direction)
    # direction: +1 (increment), -1 (decrement)
    keymap = {
        ord('1'): (0, +1),
        ord('2'): (0, -1),
        ord('3'): (1, +1),
        ord('4'): (1, -1),
        ord('5'): (2, +1),
        ord('6'): (2, -1),
    }

    while True:
        keys = p.getKeyboardEvents()

        # Quit on 'q'
        if ord('q') in keys and keys[ord('q')] & p.KEY_WAS_TRIGGERED:
            print("Exiting.")
            break

        for k, v in keys.items():
            if v & p.KEY_WAS_TRIGGERED and k in keymap:
                joint_idx, direction = keymap[k]

                # Convert to an action index used by DobotIKEnv:
                # actions:
                #   0: joint0 +
                #   1: joint0 -
                #   2: joint1 +
                #   3: joint1 -
                #   4: joint2 +
                #   5: joint2 -
                action = joint_idx * 2 + (0 if direction > 0 else 1)

                obs, reward, done, info = env.step(action)

                print(
                    f"Joint {joint_idx} "
                    f"{'+' if direction > 0 else '-'} step"
                )
                print("  theta:", info["theta"])
                print("  q_urdf:", info["q_urdf"])
                print("  EE pos:", info["ee_pos"])
                print("  target:", info["target"])
                print("  distance:", info["distance"])
                print("-" * 40)

        p.stepSimulation(physicsClientId=env.client_id)
        time.sleep(1.0 / 240.0)

    env.close()


if __name__ == "__main__":
    main()
