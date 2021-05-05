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
# sns.set(style="darkgrid")

list_converter = lambda x: np.array(x.strip("[]").replace("'","").replace(","," ").split()).astype(np.float32)
array_converter = lambda x: np.array(x.strip("[]").replace("'","").split()).astype(np.float32)

data_dir = 'collected_data/square/'
csv_filename = os.path.join(data_dir, '3d_surface_movement_data.csv')

df = pd.read_csv(csv_filename, converters={"action":   list_converter,
                                           "tcp_pose": array_converter,
                                           "tcp_vel":  array_converter})


action_array   = np.stack(df['action'].to_numpy())
tcp_pose_array = np.stack(df['tcp_pose'].to_numpy())
tcp_vel_array  = np.stack(df['tcp_vel'].to_numpy())

# pull step data
steps = np.stack(df['step'].to_numpy())

# transform poses into base frame
def workframe_to_baseframe_vels(vels):
    """
    takes a 6 dim vector of velocities [dx, dy, dz, dRx, dRy, dRz] in the
    coord frame and converts them to the base frame for velocity control.
    """
    work_frame = [0.0, -450.0, 150, -180, 0, 0]
    transformation_matrix = transforms.euler2mat(work_frame, axes='rxyz')
    rotation_matrix = transformation_matrix[:3,:3]

    trans_xyz_vels = np.dot(vels[:3], rotation_matrix)
    trans_rpy_vels = np.dot(vels[3:], rotation_matrix)
    trans_vels = np.concatenate([trans_xyz_vels, trans_rpy_vels])

    return trans_vels

trans_pose_array = []
for pose in tcp_pose_array:
    trans_pose = workframe_to_baseframe_vels(pose)
    trans_pose_array.append(trans_pose)
trans_pose_array = np.stack(trans_pose_array)


# transform rpy into vector
vector_array = []
extended_points_array = []
for pose in trans_pose_array:
    cur_tip_rpy = pose[3:] * np.pi /  180
    cur_tip_orn = p.getQuaternionFromEuler(cur_tip_rpy)
    rot_matrix = p.getMatrixFromQuaternion(cur_tip_orn)
    rot_matrix = np.array(rot_matrix).reshape(3, 3)

    init_vector = np.array([0,0,1]) * 10
    rot_vector = rot_matrix.dot(init_vector)
    vector_array.append(rot_vector)

    extended_points = pose[:3]+rot_vector
    extended_points_array.append(extended_points)

vector_array = np.stack(vector_array)
extended_points_array = np.stack(extended_points_array)



# pul pos data
px  = trans_pose_array[:,0]
py  = trans_pose_array[:,1]
pz  = trans_pose_array[:,2]
pRx = vector_array[:,0]
pRy = vector_array[:,1]
pRz = vector_array[:,2]

# pRx += 180
# pRy += 180
# pRz += 180

# pRx *= np.pi /  180
# pRy *= np.pi /  180
# pRz *= np.pi /  180

# fit a Surface to the data
nx, ny = len(px),len(py)
nx, ny = 128,128
x = np.linspace(px.min(), px.max(), nx)
y = np.linspace(px.min(), px.max(), ny)
xv, yv = np.meshgrid(x, y, sparse=False, indexing='ij')

points = np.array((px, py)).T
values = pz
surf = interpolate.griddata(points, values, (xv, yv), method='linear')
# surf[np.isnan(surf)] = 0
# np.save(os.path.join(data_dir,'real_hieghtfield_data.npy'), surf*0.001)

# plot figure
fig = plt.figure()
ax = plt.axes(projection='3d')

ax.plot_surface(xv, yv, surf, cmap=cm.coolwarm, vmin=np.nanmin(surf), vmax=np.nanmax(surf), alpha=0.5)
# ax.plot_trisurf(xv.ravel(), yv.ravel(), surf.ravel(), cmap=cm.coolwarm, vmin=np.nanmin(surf), vmax=np.nanmax(surf))

scat_step_size = 2
quiv_step_size = 2
ax.scatter(px[::scat_step_size], py[::scat_step_size], pz[::scat_step_size], color='b', marker='.', alpha=0.2)

#  plot normals
# ax.quiver(px[::quiv_step_size],  py[::quiv_step_size],  pz[::quiv_step_size],
#           pRx[::quiv_step_size], pRy[::quiv_step_size], pRz[::quiv_step_size],
#           color='r', length=1.0, normalize=False, alpha=0.25, arrow_length_ratio=0.0)

# setup plot

# remove ticks
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

# make the panes transparent
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

# make the grid lines transparent
ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)

# make the spines transparent
ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

# set view for surf
ax.view_init(azim=-70, elev=62)
ax.dist = 7

fig.savefig(os.path.join(data_dir, 'surf_view.png'), dpi=320, pad_inches=0.01, bbox_inches='tight')
plt.show()

#  plot normals
# ax.clear()
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

ax.quiver(px[::quiv_step_size],  py[::quiv_step_size],  pz[::quiv_step_size],
          pRx[::quiv_step_size], pRy[::quiv_step_size], pRz[::quiv_step_size],
          color='r', length=4.0, normalize=False, alpha=0.5, arrow_length_ratio=0.0)

# set view for normals
ax.view_init(azim=-90, elev=90)
ax.dist = 6.5

fig.savefig(os.path.join(data_dir, 'norm_view.png'), dpi=320, pad_inches=0.01, bbox_inches='tight', transparent=True)
plt.show()
