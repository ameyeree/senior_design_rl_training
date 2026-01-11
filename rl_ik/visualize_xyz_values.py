import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os

# Assuming you loaded reachable_tcp_points.npy
base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir, "reachable_tcp_points.npy")
reachable_points = np.load(file_path)

# Plotting 3D points
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(reachable_points[:, 0], reachable_points[:, 1], reachable_points[:, 2], s=1, c='g')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()
