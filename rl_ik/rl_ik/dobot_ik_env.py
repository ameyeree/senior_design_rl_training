import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data


class DobotIKEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, urdf_path=None, gui=False):
        super().__init__()

        # ==== 3 CONCEPTUAL JOINTS ============================================
        # θ0 = base
        # θ1 = shoulder
        # θ2 = elbow
        #
        # URDF joints 1–4:
        #   q0 = θ0
        #   q1 = θ1
        #   q2 = θ2 - θ1
        #   q3 = -(q1 + q2)
        #
        # This matches your collision.py kinematic constraint and keeps the
        # tool always facing downward.
        # ====================================================================
        self.num_concept_joints = 3

        # Each conceptual joint has +Δ or -Δ => 2 actions per joint
        self.action_space = spaces.Discrete(self.num_concept_joints * 2)

        # Observation:
        # [target_x, target_y, target_z, θ0, θ1, θ2, distance]
        #   3       +   3       +  1  = 7
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
        )

        # Connect to PyBullet
        if gui:
            self.client_id = p.connect(p.GUI)  # or p.connect(p.GUI, options="--opengl2")
        else:
            self.client_id = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Resolve URDF path
        if urdf_path is None:
            # assuming gripper.urdf is in the same folder as this file
            base_dir = os.path.dirname(os.path.abspath(__file__))
            urdf_path = os.path.join(base_dir, "gripper.urdf")

        self.robot_id = p.loadURDF(
            urdf_path, [0, 0, 0], useFixedBase=True, physicsClientId=self.client_id
        )

        # URDF joints we control: indices 1..4
        self.control_joint_indices = [1, 2, 3, 4]

        # End-effector link index (end_effector_part)
        self.ee_link_index = 5

        # Conceptual joint angles θ = [θ0, θ1, θ2] in radians
        self.theta = np.zeros(3, dtype=np.float32)

        # Joint step size (radians) in conceptual space
        self.delta = 0.05

        # Conceptual joint limits (tuned to be inside URDF limits)
        # You can tweak these as needed
        self.theta_low = np.array([-2.1, -0.05, -0.3], dtype=np.float32)
        self.theta_high = np.array([2.1,  1.55,  1.0], dtype=np.float32)

        # Episode settings
        self.max_steps = 100
        self.step_count = 0

        # Target placeholder
        self.target = np.zeros(3, dtype=np.float32)

        self.target_visual_id = None


    # --------------------------------------------------------------------- #
    # KINEMATIC MAPPING: conceptual joints -> URDF joints
    # --------------------------------------------------------------------- #
    def _concept_to_urdf_joints(self, theta):
        """
        Convert conceptual joints [θ0, θ1, θ2] into the 4 URDF joints.
        Matches the logic from your collision code:

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

    def _apply_urdf_joints(self, q):
        for i, j in enumerate(self.control_joint_indices):
            p.resetJointState(self.robot_id, j, q[i], physicsClientId=self.client_id)

    # --------------------------------------------------------------------- #
    # Helper functions
    # --------------------------------------------------------------------- #
    def _get_ee_pos(self):
        state = p.getLinkState(
            self.robot_id, self.ee_link_index, physicsClientId=self.client_id
        )
        pos = state[0]
        return np.array(pos, dtype=np.float32)

    def _get_obs(self):
        ee_pos = self._get_ee_pos()
        distance = np.linalg.norm(ee_pos - self.target)
        obs = np.concatenate([self.target, self.theta, [distance]]).astype(np.float32)
        return obs

    # --------------------------------------------------------------------- #
    # Gymnasium API: reset -> (obs, info)
    # --------------------------------------------------------------------- #
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0

        # Sample conceptual joint configuration θ within limits
        self.theta = np.random.uniform(self.theta_low, self.theta_high).astype(np.float32)

        # Map to URDF joints and apply
        q = self._concept_to_urdf_joints(self.theta)
        self._apply_urdf_joints(q)

        # Sample random target in front of the robot (tune these)
        self.target = np.array([
            np.random.uniform(0.12, 0.22),   # x forward from base
            np.random.uniform(-0.10, 0.10),  # y sideways
            np.random.uniform(0.03, 0.15)    # z up
        ], dtype=np.float32)

        # Remove old target marker if exists
        if self.target_visual_id is not None:
            p.removeBody(self.target_visual_id)

        # Create a small sphere to mark the target
        self.target_visual_id = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=0.01,
            rgbaColor=[1, 0, 1, 1],  # bright magenta
        )

        self.target_visual_id = p.createMultiBody(
            baseVisualShapeIndex=self.target_visual_id,
            basePosition=self.target.tolist()
)

        p.stepSimulation(physicsClientId=self.client_id)
        obs = self._get_obs()
        info = {}
        return obs, info

    # --------------------------------------------------------------------- #
    # Gymnasium API: step -> (obs, reward, terminated, truncated, info)
    # --------------------------------------------------------------------- #
    def step(self, action):
        self.step_count += 1

        # Decode action in conceptual joint space:
        # actions: 0,1 -> θ0 +/- ; 2,3 -> θ1 +/- ; 4,5 -> θ2 +/-
        joint_idx = action // 2            # 0..2
        direction = 1 if (action % 2) == 0 else -1

        # Update conceptual joint
        self.theta[joint_idx] += direction * self.delta
        self.theta = np.clip(self.theta, self.theta_low, self.theta_high)

        # Map to URDF joints and apply
        q = self._concept_to_urdf_joints(self.theta)
        self._apply_urdf_joints(q)

        p.stepSimulation(physicsClientId=self.client_id)

        ee_pos = self._get_ee_pos()
        distance = float(np.linalg.norm(ee_pos - self.target))

        # reward: encourage close distance, penalize steps
        reward = -distance**2 - 0.01

        # Gymnasium split: terminated (task success/fail), truncated (time limit)
        terminated = False
        truncated = False

        # success condition (tune threshold)
        if distance < 0.01:
            reward += 100.0
            terminated = True

        # time-out
        if self.step_count >= self.max_steps:
            truncated = True

        obs = self._get_obs()
        info = {
            "distance": distance,
            "ee_pos": ee_pos.copy(),
            "target": self.target.copy(),
            "theta": self.theta.copy(),
            "q_urdf": q.copy(),
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        # If you want GUI, create env with gui=True and PyBullet takes care of rendering
        pass

    def close(self):
        if p.isConnected(self.client_id):
            p.disconnect(self.client_id)
