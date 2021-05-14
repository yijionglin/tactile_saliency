# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 13:01:54 2019

@author: John
"""

import os
import time
import json

import numpy as np
import pandas as pd
import cv2

from tactile_gym_sim2real.data_collection.real.collect_data import collect_data

def make_target_df(poses_rng, moves_rng, num_poses, obj_poses, target_file, shuffle_data=False):
    # generate random poses
    np.random.seed()
    poses = np.random.uniform(low=poses_rng[0], high=poses_rng[1], size=(num_poses, 6))
    poses = poses[np.lexsort((poses[:,1], poses[:,5]))]
    moves = np.random.uniform(low=moves_rng[0], high=moves_rng[1], size=(num_poses, 6))

    # generate and save target data
    target_df = pd.DataFrame(columns=['sensor_video', 'sensor_image', 'obj_id', 'obj_pose', 'pose_id',
                                      'pose_1', 'pose_2', 'pose_3', 'pose_4', 'pose_5', 'pose_6',
                                      'move_1', 'move_2', 'move_3', 'move_4', 'move_5', 'move_6'])

    # populate dateframe
    for i in range(num_poses * len(obj_poses)):
        video_file = 'video_{:d}.mp4'.format(i + 1)
        image_file = 'image_{:d}.png'.format(i + 1)
        i_pose, i_obj = (int(i % num_poses), int(i / num_poses))
        pose = poses[i_pose, :]
        move = moves[i_pose, :]
        target_df.loc[i] = np.hstack(((video_file, image_file, i_obj+1, obj_poses[i_obj], i_pose+1), pose, move))

    if shuffle_data:
        target_df = target_df.sample(frac=1).reset_index(drop=True) # shuffle randomly

    return target_df

def main(shuffle_data=False):

    # ====== data collection setup ========
    # mode = 'tap'
    mode = 'shear'
    num_samples = 5000

    # set the work frame of the robot
    robot_tcp  = [0, 0, 101.0, 0, 0, 0] # change to 101.0
    base_frame = [0, 0, 0, 0, 0, 0]
    home_pose  = [0, -451.0, 300, -180, 0, 0]
    work_frame = [0, -451.0, 54.0, -180, 0, 0] # edge height = 52.0mm
    sensor_offset_ang = -48 # align camera with axis
    tap_depth = -5.0

    # create a csv with poses to collect
    poses_rng = [[0, -6, 3.0, 0, 0, -179], [0, 6, 5.0, 0, 0, 180]]

    if mode == 'shear':
        moves_rng = [[-5, -5, 0, -5, -5, -5], [5, 5, 0, 5, 5, 5]]
    elif mode == 'tap':
        moves_rng = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]

    obj_poses = [[0, 0, 0, 0, 0, sensor_offset_ang]]

    # experiment metadata
    home_dir          = os.path.join(os.path.dirname(__file__), '../data/edge_2d', mode)
    collect_dir_name  = 'collect_rand_' + time.strftime('%m%d%H%M')
    collect_dir = os.path.join(home_dir, collect_dir_name)
    target_file = os.path.join(collect_dir, 'targets.csv')

    target_df = make_target_df(poses_rng, moves_rng, num_samples, obj_poses, target_file, shuffle_data)

    collect_data(
        target_df,
        collect_dir,
        target_file,
        robot_tcp,
        base_frame,
        home_pose,
        work_frame,
        tap_depth,
    )

if __name__=="__main__":
    main()
