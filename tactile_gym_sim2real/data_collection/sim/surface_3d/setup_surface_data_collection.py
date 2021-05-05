import os
import time
import numpy as np
import pandas as pd
import json

from pybullet_sims.utils.general_utils import check_dir

from pybullet_real2sim.data_collection.sim.collect_data import make_target_df_csv, make_target_df_rand

def setup_collect_dir(num_samples=100, apply_shear=True, shuffle_data=False, og_collect_dir=None, collect_dir_name=None):

    # experiment metadata
    home_dir = os.path.join('../data/surface_3d', 'shear' if apply_shear else 'tap')

    if collect_dir_name is None:
        if og_collect_dir is None:
            collect_dir_name = 'collect_tap_rand_' + time.strftime('%m%d%H%M')
        else:
            collect_dir_name = os.path.basename(os.path.normpath((og_collect_dir)))

    collect_dir       = os.path.join(home_dir, collect_dir_name)
    image_dir         = os.path.join(collect_dir, 'images')
    target_file       = os.path.join(collect_dir, 'targets.csv')

    # set the work frame of the robot
    workframe_pos = [0.6,    0.0, 0.0635]   # relative to world frame
    workframe_rpy = [-np.pi, 0.0, np.pi/2]  # relative to world frame

    # Random data collection
    if og_collect_dir is None:

        obj_poses = [[0, 0, 0, 0, 0, 0]]
        poses_rng = [[0, 0, 3.5, -15, -15, 0], [0, 0, 5.5, 15, 15, 0]]

        if apply_shear:
            moves_rng = [[-5, -5, 0, -5, -5, -5], [5, 5, 0, 5, 5, 5]]
        else:
            moves_rng = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]

        target_df = make_target_df_rand(poses_rng, moves_rng, num_samples, obj_poses, target_file, shuffle_data)

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
