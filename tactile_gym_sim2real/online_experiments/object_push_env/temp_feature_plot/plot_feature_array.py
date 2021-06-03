import os
import matplotlib.pyplot as plt
import numpy as np


sim_action_array = np.load(
    os.path.join(
        os.path.dirname(__file__),
        'sim_action_array.npy'
    )
)

sim_feature_array = np.load(
    os.path.join(
        os.path.dirname(__file__),
        'sim_feature_array.npy'
    )
)

real_action_array = np.load(
    os.path.join(
        os.path.dirname(__file__),
        'real_action_array.npy'
    )
)


real_feature_array = np.load(
    os.path.join(
        os.path.dirname(__file__),
        'real_feature_array.npy'
    )
)

x = np.array(range(len(sim_feature_array)))

fig, ax = plt.subplots(nrows=5, ncols=3)
ax = ax.flatten()

for i in range(14):
    ax[i].scatter(x, real_feature_array[:, i], color='r', alpha=0.5)
    ax[i].scatter(x, sim_feature_array[:, i],  color='b', alpha=0.5)

fig, ax = plt.subplots(nrows=5, ncols=3)
ax = ax.flatten()

for i in range(14):
    ax[i].scatter(x, real_action_array[:, i], color='r', alpha=0.5)
    ax[i].scatter(x, sim_action_array[:, i],  color='b', alpha=0.5)

plt.show()
