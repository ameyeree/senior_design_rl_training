import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data


class DobotIKVelEnv(gym.Env):
    """
    Continuous-action, velocity-aware version of DobotIKEnv.

    State:
        [target_x, target_y, target_z,
         theta0, theta1, theta2,
         theta_dot0, theta_dot1, theta_dot2,
         prev_action0, prev_action1, prev_action2,
         distance]

        => 3 + 3 + 3 + 3 + 1 = 13 dims

    Action:
        Δθ = [Δθ0, Δθ1, Δθ2]  (radians per step)
    """

    # Advertises supported render modes *GUI is the human render*
    metadata = {"render_modes": ["human"]}

    def __init__(self, urdf_path=None, gui=False):
        super().__init__()

        # Conceptual joints
        self.num_concept_joints = 3

        # For end-effector debug
        self.ee_marker_id = None

        # Continuous joint deltas (radians per step)
        self.max_delta = 0.05
        self.action_space = spaces.Box(
            low=-self.max_delta,
            high=self.max_delta,
            shape=(self.num_concept_joints,),
            dtype=np.float32,
        )

        # State: target(3) + theta(3) + theta_dot(3) + prev_action(3) + distance(1)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32
        )

        # Connect to PyBullet, use GUI or headless
        if gui:
            self.client_id = p.connect(p.GUI)
        else:
            self.client_id = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Resolve URDF path
        base_dir = os.path.dirname(os.path.abspath(__file__))
        if urdf_path is None:
            urdf_path = os.path.join(base_dir, "gripper.urdf")

        reach_path = os.path.join(base_dir, "reachable_tcp_points.npy")
        self.reachable_pts = np.load(reach_path)

        # remove outliers
        lo = np.percentile(self.reachable_pts, 2, axis=0)
        hi = np.percentile(self.reachable_pts, 98, axis=0)
        mask = np.all((self.reachable_pts >= lo) & (self.reachable_pts <= hi), axis=1)
        self.reachable_pts = self.reachable_pts[mask]

        # useFixedBase means the robot is bolted the world
        self.robot_id = p.loadURDF(
            urdf_path, [0, 0, 0], useFixedBase=True, physicsClientId=self.client_id
        )

        # URDF joints we control: indices 1..4 (same as your original env)
        self.control_joint_indices = [1, 2, 3, 4]
        self.ee_link_index = 6

        # Conceptual joint angles θ = [θ0, θ1, θ2]
        self.theta = np.zeros(3, dtype=np.float32)
        self.theta_prev = np.zeros(3, dtype=np.float32)

        # Conceptual joint velocities θ_dot (we’ll compute as (θ - θ_prev) / dt)
        self.theta_dot = np.zeros(3, dtype=np.float32)

        # Previous action (for smoothness term)
        self.prev_action = np.zeros(3, dtype=np.float32)

        # Below is used for computing velocities
        self.dt = 1.0 / 60.0

        # Conceptual joint limits
        # TODO check these
        self.theta_low = np.array([-2.1, -0.05, -0.3], dtype=np.float32)
        self.theta_high = np.array([2.1,  1.55,  1.0], dtype=np.float32)

        # Optional "safe" velocity limits for penalty (rad/s)
        self.vel_limits = np.array([1.0, 1.0, 1.0], dtype=np.float32)

        # Episode settings
        self.max_steps = 100
        self.step_count = 0

        # Target & marker
        self.target = np.zeros(3, dtype=np.float32)
        self.target_visual_id = None

        # Reward weights
        # TODO tune these
        self.w_dist = 1.0
        self.w_action = 0.01
        self.w_smooth = 0.01
        self.w_joint_limit = 0.1
        self.w_vel = 0.01
        self.step_penalty = 0.001

    # ------------------------------------------------------------------ #
    # KINEMATIC MAPPING: conceptual joints -> URDF joints
    # ------------------------------------------------------------------ #
    def _concept_to_urdf_joints(self, theta):
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

    def _apply_urdf_joints(self, q):
        # Still using resetJointState for simplicity (kinematic control)
        for i, j in enumerate(self.control_joint_indices):
            p.resetJointState(self.robot_id, j, q[i], physicsClientId=self.client_id)

    # ------------------------------------------------------------------ #
    # Helper functions
    # ------------------------------------------------------------------ #
    def _get_ee_pos(self):
        state = p.getLinkState(
            self.robot_id, self.ee_link_index, physicsClientId=self.client_id
        )
        pos = state[0]
        return np.array(pos, dtype=np.float32)

    def _get_obs(self):
        """
        Docstring for _get_obs
        Gets the EE position, computed the Euclidena distance to target.
        Returns self.target, self.theta, self.theta_dot, self.prev_action, 
        Euclidean distance.
        
        :param self: Description
        """
        ee_pos = self._get_ee_pos()
        distance = np.linalg.norm(ee_pos - self.target)

        obs = np.concatenate(
            [
                self.target,          # 3
                self.theta,           # 3
                self.theta_dot,       # 3
                self.prev_action,     # 3
                np.array([distance], dtype=np.float32),  # 1
            ]
        ).astype(np.float32)
        return obs

    # ------------------------------------------------------------------ #
    # Reward components
    # ------------------------------------------------------------------ #
    def _compute_reward(self, action):
        """
        Docstring for _compute_reward
        Calculated the reward. 
        :param self: Description
        :param action: Description
        """
        ee_pos = self._get_ee_pos()
        distance = float(np.linalg.norm(ee_pos - self.target))

        # Distance penalty
        r_dist = - self.w_dist * (distance ** 2)

        # Large action penalty
        r_action = - self.w_action * float(np.sum(np.square(action)))

        # Smoothness penalty
        r_smooth = - self.w_smooth * float(np.sum(np.square(action - self.prev_action)))

        # Soft joint limit penalty: penalize being near bounds
        r_joint_limit = 0.0
        margin = 0.1  # rad from limit where penalty ramps up
        for i in range(self.num_concept_joints):
            low = self.theta_low[i]
            high = self.theta_high[i]
            th = self.theta[i]

            if th < (low + margin):
                r_joint_limit -= self.w_joint_limit * ((low + margin - th) ** 2)
            elif th > (high - margin):
                r_joint_limit -= self.w_joint_limit * ((th - (high - margin)) ** 2)

        # Velocity limit penalty
        r_vel = 0.0
        for i in range(self.num_concept_joints):
            v = abs(self.theta_dot[i])
            if v > self.vel_limits[i]:
                r_vel -= self.w_vel * ((v - self.vel_limits[i]) ** 2)

        # Small step penalty (will encourage shorter solutions)
        r_step = - self.step_penalty

        reward = r_dist + r_action + r_smooth + r_joint_limit + r_vel + r_step

        # Success bonus
        # TODO: Needs to be more accurate than 10 cms, need to get closer to < 1 mm.
        if distance < 0.01:
            reward += 100.0

        return reward, distance


    def reset(self, *, seed=None, options=None):
        """
        Docstring for reset
        Resets the arm into a random position, creates a new random target,
        creates the target marker. Performs one simulation step, and returns the
        observation.
        :param self: Description
        :param seed: Description
        :param options: Description
        """
        super().reset(seed=seed)
        self.step_count = 0

        # Sample θ within limits
        self.theta = np.random.uniform(self.theta_low, self.theta_high).astype(
            np.float32
        )
        self.theta_prev = self.theta.copy()
        self.theta_dot = np.zeros_like(self.theta, dtype=np.float32)
        self.prev_action = np.zeros(self.num_concept_joints, dtype=np.float32)

        # Apply to URDF
        q = self._concept_to_urdf_joints(self.theta)
        self._apply_urdf_joints(q)

        # Random target in front of the robot (same ranges as before)
        # Will take from reachable points


        self.target = self.reachable_pts[np.random.randint(len(self.reachable_pts))].copy()
        # self.target = np.array(
        #     [
        #         np.random.uniform(0.12, 0.22),
        #         np.random.uniform(-0.10, 0.10),
        #         np.random.uniform(0.03, 0.15),
        #     ],
        #     dtype=np.float32,
        # )

        # Remove old target marker if exists
        if self.target_visual_id is not None:
            p.removeBody(self.target_visual_id)

        # Create a small sphere to mark the target
        sphere_shape = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=0.01,
            rgbaColor=[1, 0, 1, 1],
        )
        self.target_visual_id = p.createMultiBody(
            baseVisualShapeIndex=sphere_shape,
            basePosition=self.target.tolist(),
        )

        p.stepSimulation(physicsClientId=self.client_id)
        obs = self._get_obs()
        info = {}
        return obs, info


    def step(self, action):
        """
        Docstring for step
        Performs a gymnasium step. Increments the step count, clips
        action to the bounding box, updates conceptual joints, saves
        previous values for calculations, calculates conceptual velocity,
        takes a step in the simulation, updates prev_action,
        computes the reward and distance. 

        Things that can be set:
        If distrance < x, terminate
        If step_count >= max_steps (would need to change max steps)

        Returns a dictionary of debugging information: distance, theta, 
        theta_dot, action, q_urdf in tuple.
        :param self: Description
        :param action: Description
        """
        self.step_count += 1

        # Clip action to allowed delta range
        action = np.clip(action, self.action_space.low, self.action_space.high).astype(
            np.float32
        )

        # Update joints in conceptual space
        self.theta_prev = self.theta.copy()
        new_theta = self.theta + action
        # Hard clip to conceptual joint limits
        self.theta = np.clip(new_theta, self.theta_low, self.theta_high)

        # Compute conceptual velocities (simple finite difference)
        self.theta_dot = (self.theta - self.theta_prev) / self.dt

        # Map to URDF and apply
        q = self._concept_to_urdf_joints(self.theta)
        self._apply_urdf_joints(q)

        p.stepSimulation(physicsClientId=self.client_id)

        # Debug EE Location (GUI Only)
        # if p.isConnected(self.client_id):
        #     ee = self._get_ee_pos()

        #     print("EE:", ee)

        #     if self.ee_marker_id is None:
        #         shape = p.createVisualShape(
        #             p.GEOM_SPHERE, radius=0.01, rgbaColor=[0, 1, 0, 1],
        #             physicsClientId=self.client_id
        #         )
        #         self.ee_marker_id = p.createMultiBody(
        #             baseVisualShapeIndex=shape,
        #             basePosition=ee.tolist(),
        #             physicsClientId=self.client_id
        #         )
        #     else:
        #         p.resetBasePositionAndOrientation(
        #             self.ee_marker_id, ee.tolist(), [0, 0, 0, 1],
        #             physicsClientId=self.client_id
        #         )

        # Reward
        reward, distance = self._compute_reward(action)

        # Termination & truncation
        terminated = False
        truncated = False

        if distance < 0.01:
            terminated = True

        if self.step_count >= self.max_steps:
            truncated = True

        obs = self._get_obs()
        info = {
            "distance": distance,
            "theta": self.theta.copy(),
            "theta_dot": self.theta_dot.copy(),
            "action": action.copy(),
            "q_urdf": q.copy(),
        }

        # Update prev_action after computing reward
        self.prev_action = action.copy()

        return obs, reward, terminated, truncated, info

    def render(self):
        # GUI is handled by PyBullet when created with gui=True
        pass

    def close(self):
        if p.isConnected(self.client_id):
            p.disconnect(self.client_id)
