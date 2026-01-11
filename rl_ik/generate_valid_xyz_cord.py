import numpy as np
import pybullet as p
import pybullet_data
import os

# --- load robot like your env does ---
p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

base_dir = os.path.dirname(os.path.abspath(__file__))
urdf_path = os.path.join(base_dir, "gripper.urdf")

robot_id = p.loadURDF(urdf_path, [0,0,0], useFixedBase=True)

# Your env uses these URDF joint indices to set angles:
control_joint_indices = [1, 2, 3, 4]   # if that's still true in your env
tcp_link_index = 6

# Your conceptual joint limits (radians) - from your env
theta_low  = np.array([-2.1, -0.05, -0.3], dtype=np.float32)
theta_high = np.array([ 2.1,  1.55,  1.0], dtype=np.float32)

def concept_to_urdf(theta):
    q = np.zeros(4, dtype=np.float32)
    q[0] = theta[0]
    q[1] = theta[1]
    q[2] = theta[2] - theta[1]
    q[3] = -(q[1] + q[2])  # = -theta[2]
    return q

def apply_q(q):
    for i, j in enumerate(control_joint_indices):
        p.resetJointState(robot_id, j, float(q[i]))

def tcp_pos():
    ls = p.getLinkState(robot_id, tcp_link_index, computeForwardKinematics=True)
    return np.array(ls[0], dtype=np.float32)

# --- sample workspace ---
N = 100000
pts = np.zeros((N, 3), dtype=np.float32)

for k in range(N):
    theta = np.random.uniform(theta_low, theta_high).astype(np.float32)
    q = concept_to_urdf(theta)
    apply_q(q)
    p.stepSimulation()
    pts[k] = tcp_pos()

mins = pts.min(axis=0)
maxs = pts.max(axis=0)

print("TCP workspace bounds (meters):")
print("x:", mins[0], "to", maxs[0])
print("y:", mins[1], "to", maxs[1])
print("z:", mins[2], "to", maxs[2])

np.save(os.path.join(base_dir, "reachable_tcp_points.npy"), pts)
print("Saved reachable points to reachable_tcp_points.npy")

p.disconnect()
