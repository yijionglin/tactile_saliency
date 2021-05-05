import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits import mplot3d
import json
import os
from scipy import interpolate

from cri import transforms
import pybullet as p
import seaborn as sns
sns.set(style="darkgrid")

list_converter = lambda x: np.array(x.strip("[]").replace("'","").replace(","," ").split()).astype(np.float32)
array_converter = lambda x: np.array(x.strip("[]").replace("'","").split()).astype(np.float32)

# data dir
# data_dir = 'cube_straight'
# data_dir = 'cylinder_straight'
# data_dir = 'hexagonal_prism_straight'
# data_dir = 'mug_straight'
#
# data_dir = 'cube_curve'
# data_dir = 'cylinder_curve'
# data_dir = 'hexagonal_prism_curve'
# data_dir = 'mug_curve'
#
data_dir = 'cube_sin'
# data_dir = 'cylinder_sin'
# data_dir = 'hexagonal_prism_sin'
# data_dir = 'mug_sin'

df = pd.read_csv(os.path.join('collected_data', data_dir, 'evaluation_movement_data.csv'),
                 converters={"action":   list_converter,
                 "tcp_pose": array_converter,
                 "tcp_vel":  array_converter})


action_array   = np.stack(df['action'].to_numpy())
tcp_pose_array = np.stack(df['tcp_pose'].to_numpy())
tcp_vel_array  = np.stack(df['tcp_vel'].to_numpy())

# pull step data
steps = np.stack(df['step'].to_numpy())

def get_tip_direction_workframe(current_tip_pose):
    """
    Warning, deadline research code (specific to current workframe)
    """
    # angle for perp and par vectors
    # par_ang  = -( current_tip_pose[5] + 45 ) * np.pi/180
    # perp_ang = -( current_tip_pose[5] + 45 - 90 ) * np.pi/180
    par_ang  = -( current_tip_pose[5] ) * np.pi/180
    perp_ang = -( current_tip_pose[5] - 90 ) * np.pi/180

    # create vectors (directly in workframe) pointing in perp and par directions of current sensor
    workframe_par_tip_direction  = np.array([np.cos(par_ang),  np.sin(par_ang), 0]) # vec pointing outwards from tip
    workframe_perp_tip_direction = np.array([np.cos(perp_ang), np.sin(perp_ang),0]) # vec pointing perp to tip

    return workframe_par_tip_direction, workframe_perp_tip_direction

# get direction of tip by transforming rpy into vector
par_tip_direction = []
perp_tip_direction = []
for pose in tcp_pose_array:

    workframe_par_tip_direction, workframe_perp_tip_direction = get_tip_direction_workframe(pose)

    par_tip_direction.append(workframe_par_tip_direction)
    perp_tip_direction.append(workframe_perp_tip_direction)

par_tip_direction = np.stack(par_tip_direction)
perp_tip_direction = np.stack(perp_tip_direction)


# pul pose data
px  = tcp_pose_array[:,0]
py  = tcp_pose_array[:,1]

par_x, perp_x = par_tip_direction[:,0], perp_tip_direction[:,0]
par_y, perp_y = par_tip_direction[:,1], perp_tip_direction[:,1]

# plot figure
fig, ax = plt.subplots()

# plot trajectory taken
scat_step_size = 1
# ax.scatter(px[::scat_step_size], py[::scat_step_size], color='b', marker='.', alpha=0.35)

#  plot normals
quiv_step_size = 3
# ax.quiver(px[::quiv_step_size],  py[::quiv_step_size],
#           par_x[::quiv_step_size], par_y[::quiv_step_size],
#           color='b', alpha=1.0, scale=25.0, angles='uv',
#           width=0.0025, headwidth=2.5, headlength=5.0)

# ax.quiver(px[::quiv_step_size],  py[::quiv_step_size],
#           perp_x[::quiv_step_size], perp_y[::quiv_step_size],
#           color='g', alpha=1.0, scale=25.0, angles='uv',
#           width=0.0025, headwidth=2.5, headlength=5.0)

# plot target trajectory
# load trajectory data
num_traj = 3
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

# ax.set_xlim(np.min(px), np.max(px))
# ax.set_ylim(np.min(py), np.max(py))

ax.invert_yaxis()
ax.axis('equal')

fig.savefig(os.path.join('collected_data', data_dir, 'traj_scatter.png'), dpi=320, pad_inches=0.01, bbox_inches='tight')
plt.show()
