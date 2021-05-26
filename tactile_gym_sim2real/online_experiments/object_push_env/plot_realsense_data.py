import os
import matplotlib.pyplot as plt
from robopush.utils import Namespace


rs_save_file = os.path.join(
    'collected_data',
    'rs_data',
    'rs_data.pkl'
)

rs_data = Namespace()
rs_data.load(rs_save_file)


# Plot and save ArUco marker centroid trajectory
plt.figure()
plt.scatter(rs_data.base_centroids[:, 0], rs_data.base_centroids[:, 1], marker='.')
plt.title("ArUco marker centroid trajectory")
plt.xlabel("x")
plt.ylabel("y")
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
