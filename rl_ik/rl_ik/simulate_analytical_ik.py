# eval_dobot_analytical_ik_smooth.py

import time
import numpy as np
import pybullet as p

from dobot_ik_env import DobotIKEnv
from inverse_kinematics import calc_inv_kin


# Five poses in millimeters + degrees: (x_mm, y_mm, z_mm, r_deg)
POSES_MM = [
    (100,  -50,  80,   0),   # Pose 1
    (120,   30, 100,  45),   # Pose 2
    (150,  -20,  60, -30),   # Pose 3
    ( 90,   60,  90,  90),   # Pose 4
    (110,  -70,  70, -90),   # Pose 5
]


def move_to_pose_analytic(env, pose_mm, smooth_steps=200, dt=1.0/240.0, hold_time=1.0):
    """
    Use the analytical IK to move the Dobot smoothly to the given pose.

    pose_mm: (x_mm, y_mm, z_mm, r_deg)
    smooth_steps: number of interpolation steps in joint space
    dt: simulation timestep delay
    hold_time: how long to hold at the final pose (seconds)
    """
    x_mm, y_mm, z_mm, r_deg = pose_mm

    # Compute IK in Dobot convention (deg)
    dobot_angles = calc_inv_kin(x_mm, y_mm, z_mm, r_deg)
    if not dobot_angles:
        print(f"IK failed for pose (mm): {pose_mm}")
        return

    base_deg, shoulder_deg, elbow_plus_shoulder_deg, wrist_deg = dobot_angles

    # print("\nAnalytical IK target (mm): "
    #       f"x={x_mm}, y={y_mm}, z={z_mm}, r={r_deg}")
    # print("Analytical Dobot joint angles (deg): "
    #       f"base={base_deg:.3f}, shoulder={shoulder_deg:.3f}, "
    #       f"elbow+shoulder={elbow_plus_shoulder_deg:.3f}, wrist={wrist_deg:.3f}")

    # Convert Dobot convention to your conceptual joints θ0, θ1, θ2
    # θ0 = base, θ1 = shoulder, elbow_plus_shoulder = θ1 + θ2
    theta0 = np.radians(base_deg)
    theta1 = np.radians(shoulder_deg)
    theta2 = np.radians(elbow_plus_shoulder_deg - shoulder_deg)
    theta_target = np.array([theta0, theta1, theta2], dtype=np.float32)

    # Starting conceptual joints (from current env state)
    theta_start = env.theta.copy()

    # Create a visual marker at the target
    x_m, y_m, z_m = x_mm / 1000.0, y_mm / 1000.0, z_mm / 1000.0
    target_visual_shape = p.createVisualShape(
        shapeType=p.GEOM_SPHERE,
        radius=0.01,
        rgbaColor=[1, 0.4, 0, 1],
        physicsClientId=env.client_id,
    )
    target_body_id = p.createMultiBody(
        baseVisualShapeIndex=target_visual_shape,
        basePosition=[x_m, y_m, z_m],
        physicsClientId=env.client_id,
    )

    # Smoothly interpolate in joint space
    for i in range(smooth_steps + 1):
        alpha = i / smooth_steps  # 0 → 1
        theta_interp = (1.0 - alpha) * theta_start + alpha * theta_target

        q_urdf = env._concept_to_urdf_joints(theta_interp)
        env._apply_urdf_joints(q_urdf)
        env.theta = theta_interp  # keep env state consistent

        p.stepSimulation(physicsClientId=env.client_id)
        time.sleep(dt)

    # Hold at final pose
    hold_steps = int(hold_time / dt)
    for _ in range(hold_steps):
        p.stepSimulation(physicsClientId=env.client_id)
        time.sleep(dt)

    # Remove the target marker
    p.removeBody(target_body_id, physicsClientId=env.client_id)

    # Report final EE position and error
    ee_pos = env._get_ee_pos()  # meters
    ee_pos_mm = ee_pos * 1000.0
    err = np.linalg.norm(ee_pos - np.array([x_m, y_m, z_m]))
    # print(f"End-effector actual position (mm): {ee_pos_mm}")
    print(f"Final Distance (m): {err:.6f}")


def main():
    env = DobotIKEnv(gui=True)

    # Initialize env once so env.theta and joints are in a valid state
    obs, info = env.reset()
    print("Starting ANALYTICAL IK demo (smooth motion, no RL). Ctrl+C to stop.")

    pose_index = 0
    try:
        while True:
            pose_mm = POSES_MM[pose_index]
            print(f"Pose {pose_index + 1}.")
            move_to_pose_analytic(env, pose_mm,
                                  smooth_steps=200,
                                  dt=1.0/240.0,
                                  hold_time=1.0)

            pose_index = (pose_index + 1) % len(POSES_MM)

    except KeyboardInterrupt:
        print("Stopping analytical IK demo.")

    env.close()


if __name__ == "__main__":
    main()
