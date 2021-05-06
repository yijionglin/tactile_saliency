import os
import time
import numpy as np
import pandas as pd
import json

from tactile_gym.utils.general_utils import check_dir
from tactile_gym_sim2real.data_collection.sim.collect_data import make_target_df_csv

def make_target_df_rand(poses_rng, moves_rng, num_poses, obj_poses, shuffle_data=False):
    # generate random poses
    np.random.seed()

    # generate poses in the normal way
    poses = np.random.uniform(low=poses_rng[0], high=poses_rng[1], size=(num_poses, 6))

    # overwrite x, y with spherical random pose generation
    ang = np.random.uniform(0, 2*np.pi, size=num_poses)
    rad = np.random.uniform(0, 1, size=num_poses)

    # set the limit of the radius
    rad_lim = 15

    # use sqrt r to get uniform disk spread
    x = np.sqrt(rad)*np.cos(ang) * rad_lim
    y = np.sqrt(rad)*np.sin(ang) * rad_lim
    poses[:, 0], poses[:, 1] = x, y

    # for now don't do shear moves as care needs to be taken
    moves = np.zeros(shape=(num_poses, 6))

    # generate and save target data
    target_df = pd.DataFrame(columns=['sensor_image', 'obj_id', 'obj_pose', 'pose_id',
                                      'pose_1', 'pose_2', 'pose_3', 'pose_4', 'pose_5', 'pose_6',
                                      'move_1', 'move_2', 'move_3', 'move_4', 'move_5', 'move_6'])

    # populate dateframe
    for i in range(num_poses * len(obj_poses)):
        image_file = 'image_{:d}.png'.format(i + 1)
        i_pose, i_obj = (int(i % num_poses), int(i / num_poses))
        pose = poses[i_pose, :]
        move = moves[i_pose, :]
        target_df.loc[i] = np.hstack(((image_file, i_obj+1, obj_poses[i_obj], i_pose+1), pose, move))

    if shuffle_data:
        target_df = target_df.sample(frac=1).reset_index(drop=True) # shuffle randomly

    return target_df

def setup_collect_dir(
        num_samples=100,
        apply_shear=True,
        shuffle_data=False,
        og_collect_dir=None,
        collect_dir_name=None
    ):

    # experiment metadata
    home_dir = os.path.join(
        os.path.dirname(__file__),
        '../data/spherical_probe',
        'shear' if apply_shear else 'tap'
    )
    if collect_dir_name is None:
        if og_collect_dir is None:
            collect_dir_name = 'collect_tap_rand_' + time.strftime('%m%d%H%M')
        else:
            collect_dir_name = os.path.basename(os.path.normpath((og_collect_dir)))

    collect_dir = os.path.join(home_dir, collect_dir_name)
    image_dir = os.path.join(collect_dir, 'images')
    target_file = os.path.join(collect_dir, 'targets.csv')

    # set the work frame of the robot
    hover_dist = 0.002 # 2mm above stim
    workframe_pos = [0.6,    0.0, 0.045+hover_dist]   # relative to world frame
    workframe_rpy = [-np.pi, 0.0, np.pi/2]  # relative to world frame

    # Random data collection
    if og_collect_dir is None:

        obj_poses = [[ 60,  60, 0, 0, 0, 0],
                     [ 0,   60, 0, 0, 0, 0],
                     [-60,  60, 0, 0, 0, 0],
                     [ 60,  0,  0, 0, 0, 0],
                     [ 0,   0,  0, 0, 0, 0],
                     [-60,  0,  0, 0, 0, 0],
                     [ 60, -60, 0, 0, 0, 0],
                     [ 0,  -60, 0, 0, 0, 0],
                     [-60, -60, 0, 0, 0, 0]]

        poses_rng = [[0, 0, 4.5, 0, 0, 0], [0, 0, 5.5, 0, 0, 0]]

        if apply_shear:
            moves_rng = [[-5, -5, 0, -5, -5, -5], [5, 5, 0, 5, 5, 5]]
        else:
            moves_rng = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]

        target_df = make_target_df_rand(poses_rng, moves_rng, num_samples, obj_poses, shuffle_data)

    # collect from values in csv
    else:
        og_target_file = os.path.join(og_collect_dir, 'targets.csv')
        target_df = make_target_df_csv(og_target_file, shuffle_data)

    # check save dir exists
    check_dir(collect_dir)

    # create dirs
    os.makedirs(collect_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)

    # save metadata (remove unneccesary non json serializable stuff)
    meta = locals().copy()
    del meta['collect_dir_name'], meta['target_df']
    with open(os.path.join(collect_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f)

    # save target csv
    target_df.to_csv(target_file, index=False)

    return target_df, image_dir, workframe_pos, workframe_rpy
