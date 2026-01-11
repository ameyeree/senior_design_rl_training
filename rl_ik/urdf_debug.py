import pybullet as p
import pybullet_data
import os

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
base_dir = os.path.dirname(os.path.abspath(__file__))
urdf_path = os.path.join(base_dir, "gripper.urdf")
robot_id = p.loadURDF(urdf_path, [0,0,0], useFixedBase=True)

for i in range(p.getNumJoints(robot_id)):
    info = p.getJointInfo(robot_id, i)
    print(i, info[1].decode(), info[12].decode())
