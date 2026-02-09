import os
import numpy as np

# Repo paths
NN_IK_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(NN_IK_DIR)

# Use the same URDF as rl_ik/DobotIKVelEnv
URDF_PATH = os.path.join(REPO_ROOT, "rl_ik", "gripper.urdf")

# URDF joint indices used by the environment
CONTROL_JOINT_INDICES = [1, 2, 3, 4]
EE_LINK_INDEX = 6

# Conceptual joint limits (radians) from rl_ik/dobot_ik_vel_env.py
THETA_LOW = np.array([-2.1, -0.05, -0.3], dtype=np.float32)
THETA_HIGH = np.array([2.1, 1.55, 1.0], dtype=np.float32)

# Percentile-based outlier trimming (matches env approach)
PCTL_LOW = 2
PCTL_HIGH = 98


def concept_to_urdf(theta: np.ndarray) -> np.ndarray:
    """
    Convert conceptual joints [θ0, θ1, θ2] into the 4 URDF joints.

        q0 = θ0
        q1 = θ1
        q2 = θ2 - θ1
        q3 = -(q1 + q2)
    """
    q = np.zeros(4, dtype=np.float32)
    q[0] = theta[0]
    q[1] = theta[1]
    q[2] = theta[2] - theta[1]
    q[3] = -(q[1] + q[2])
    return q


def urdf_to_concept(q: np.ndarray) -> np.ndarray:
    """
    Inverse mapping: URDF joints -> conceptual joints [θ0, θ1, θ2].
    """
    theta = np.zeros(3, dtype=np.float32)
    theta[0] = q[0]
    theta[1] = q[1]
    theta[2] = q[1] + q[2]
    return theta


def reachable_points_path() -> str:
    return os.path.join(REPO_ROOT, "rl_ik", "reachable_tcp_points.npy")
