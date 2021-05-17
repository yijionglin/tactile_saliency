import itertools
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
# import seaborn as sns
# sns.set(style="darkgrid")

list_converter = lambda x: np.array(x.strip("[]").replace("'","").replace(","," ").split()).astype(np.float32)
array_converter = lambda x: np.array(x.strip("[]").replace("'","").split()).astype(np.float32)

# real data dir
# object = 'square'
# object = 'circle'
# object = 'clover'
# object = 'foil'
objects = ['square', 'circle', 'clover', 'foil']

# gan_data = 'tap'
# gan_data = 'shear'
gan_datas = ['tap', 'shear']

# vel = '10mmps'
# vel = '5mmps'
vels = ['5mmps', '10mmps']

for [object, gan_data, vel] in list(itertools.product(objects, gan_datas, vels)):

    data_dir = os.path.join(
        'collected_data',
        object,
        gan_data,
        vel
    )


    fig, ax = plt.subplots()

    df = pd.read_csv(
        os.path.join(data_dir, 'eval_data.csv'),
        converters={
            "action": list_converter,
            "tcp_pose": array_converter,
            # "tcp_vel": array_converter
        }
    )


    action_array   = np.stack(df['action'].to_numpy())
    tcp_pose_array = np.stack(df['tcp_pose'].to_numpy())
    # tcp_vel_array  = np.stack(df['tcp_vel'].to_numpy())

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

    # plot figure
    scat_step_size = 1
    end_point = 10000 # radial
    ax.scatter(px[:end_point:scat_step_size], py[:end_point:scat_step_size], color='r', marker='.', s=50.0, alpha=1.0)
    # ax.scatter(px[:end_point:scat_step_size], py[:end_point:scat_step_size], color='r', marker='x', s=100.0, alpha=1.0)

    ax.set_xlim(np.min(px), np.max(px))
    ax.set_ylim(np.min(py), np.max(py))

    # remove ticks
    # ax.set_xticks([])
    # ax.set_yticks([])

    # ax.axis('off')

    # plt.legend(labels=['Real', 'Sim'], bbox_to_anchor=(-0.325, 0.525, 0.5, 0.5), fontsize=10)
    ax.axis('equal')

    fig.savefig(os.path.join(data_dir, 'edge_scatter.png'), dpi=320, pad_inches=0.01, bbox_inches='tight', transparent=False)
    plt.show()
