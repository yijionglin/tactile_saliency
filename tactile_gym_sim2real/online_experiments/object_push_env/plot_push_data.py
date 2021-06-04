import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits import mplot3d
import json
import os

from cri import transforms
import pybullet as p
import glob

from robopush.utils import Namespace

# data_dir = 'red_cube_straight'
# data_dir = 'red_cube_curve'
# data_dir = 'red_cube_sin'

# data_dir = 'cylinder_straight'
# data_dir = 'cylinder_curve'
# data_dir = 'cylinder_sin'

# data_dir = 'triangle_straight'
# data_dir = 'triangle_curve'
data_dir = 'triangle_sin'

traj_files = glob.glob(
    os.path.join(
        os.path.dirname(__file__),
        'collected_data',
        data_dir+"/*.npy"
        )
    )

num_traj = int(len(traj_files)/2)

# setup figure
fig, ax = plt.subplots()

def basepos_to_workpos(pos):
    """
    Transforms a vector in world frame to a vector in work frame.
    """
    work_frame = np.array([-200.0, -420.0, 55, -180, 0, 0])
    transformation_matrix = transforms.euler2mat(work_frame, axes='rxyz')
    inv_transformation_matrix = np.linalg.inv(transformation_matrix)

    pos_mat = transforms.euler2mat([*pos,0,0,0])

    workframe_pos_mat = np.dot(inv_transformation_matrix, pos_mat)
    workframe_pos = transforms.mat2euler(workframe_pos_mat)

    # workframe_pos = pos + [350, -520, 0]
    # workframe_pos = pos + [0, 0, 0]

    # workframe_pos += np.array([20, 0, 0, 0, 0, 0])

    return np.array(workframe_pos)

def get_tip_direction_workframe(current_tip_pose):
    """
    Warning, deadline research code (specific to current workframe)
    """
    # angle for perp and par vectors
    par_ang  = -( current_tip_pose[5] ) * np.pi/180
    perp_ang = -( current_tip_pose[5] - 90 ) * np.pi/180

    # create vectors (directly in workframe) pointing in perp and par directions of current sensor
    workframe_par_tip_direction  = np.array([np.cos(par_ang),  np.sin(par_ang), 0]) # vec pointing outwards from tip
    workframe_perp_tip_direction = np.array([np.cos(perp_ang), np.sin(perp_ang),0]) # vec pointing perp to tip

    return workframe_par_tip_direction, workframe_perp_tip_direction

# plot target trajectory
def plot_traj():
    for i in range(num_traj):
        traj_pos = np.load(os.path.join('collected_data', data_dir, 'traj_pos_{}.npy'.format(i))) * 1000
        traj_rpy = np.load(os.path.join('collected_data', data_dir, 'traj_rpy_{}.npy'.format(i))) * 180 / np.pi

        for pos, rpy in zip(traj_pos, traj_rpy):
            pose = np.concatenate([pos, rpy], axis=0)
            par_dir, perp_dir  = get_tip_direction_workframe(pose)

            goal_x, goal_y = pos[0], pos[1]
            par_x, perp_x = par_dir[0], perp_dir[0]
            par_y, perp_y = par_dir[1], perp_dir[1]

            ax.scatter(goal_x, goal_y, color='b', marker='.', alpha=1.0)

            ax.quiver(goal_x, goal_y,
                      par_x, par_y,
                      color='g', alpha=1.0, scale=25.0, angles='uv',
                      width=0.0025, headwidth=2.5, headlength=5.0)

            ax.quiver(goal_x, goal_y,
                      perp_x, perp_y,
                      color='r', alpha=1.0, scale=25.0, angles='uv',
                      width=0.0025, headwidth=2.5, headlength=5.0)

def plot_realsense_data():

    rs_save_file = os.path.join(
        'collected_data',
        data_dir,
        'rs_data',
        'rs_data.pkl'
    )

    rs_data = Namespace()
    rs_data.load(rs_save_file)

    # convert centroids to workframe
    trans_centroids = np.array([basepos_to_workpos(pos) for pos in rs_data.base_centroids])

    # Plot and save ArUco marker centroid trajectory
    ax.scatter(
        trans_centroids[:, 0],
        trans_centroids[:, 1],
        marker='.',
        color='r',
        alpha=0.15
    )

    # ax.scatter(rs_data.base_poses[:, 0, 3], rs_data.base_poses[:, 1, 3], marker='.')



plot_traj()
plot_realsense_data()

# format plot

# ax.set_xlim(np.min(px), np.max(px))
# ax.set_ylim(np.min(py), np.max(py))

ax.invert_yaxis()
ax.axis('equal')

# fig.savefig(
#     os.path.join(
#         'collected_data',
#         data_dir,
#         'traj_scatter.png'
#     ),
#     dpi=320,
#     pad_inches=0.01,
#     bbox_inches='tight'
# )

plt.show()
