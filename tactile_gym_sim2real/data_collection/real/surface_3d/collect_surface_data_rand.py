# -*- coding: utf-8 -*-
import os
import time
import json

import numpy as np
import pandas as pd
import cv2

from cri.robot import SyncRobot, AsyncRobot
from cri.controller import RTDEController

from vsp.video_stream import CvVideoDisplay, CvVideoOutputFile, CvVideoCamera
from vsp.processor import CameraStreamProcessorMT, AsyncProcessor

from pybullet_sims.utils.general_utils import str2bool, save_json_obj, empty_dir

def make_robot():
    return AsyncRobot(SyncRobot(RTDEController(ip='192.11.72.10')))
    # return AsyncRobot(SyncRobot(RTDEController(ip='127.0.0.1')))

def make_sensor():
    return AsyncProcessor(CameraStreamProcessorMT(
            camera=CvVideoCamera(source=0,
                                 frame_size=(640, 480),
                                 is_color=True),
            display=CvVideoDisplay(name='preview'),
            writer=CvVideoOutputFile(is_color=True),
        ))

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

    # experiment metadata
    home_dir          = os.path.join(os.path.dirname(__file__), '../data/surface_3d', mode)
    collect_dir_name  = 'collect_rand_' + time.strftime('%m%d%H%M')
    collect_dir = os.path.join(home_dir, collect_dir_name)
    video_dir = os.path.join(collect_dir, 'videos')
    image_dir = os.path.join(collect_dir, 'images')
    target_file = os.path.join(collect_dir, 'targets.csv')

    # set the work frame of the robot
    robot_tcp  = [0, 0, 108.0, 0, 0, 0]
    base_frame = [0, 0, 0, 0, 0, 0]
    home_pose  = [0, -451.0, 300, -180, 0, 0]
    work_frame = [0, -500.0, 47.5, -180, 0, 0] # 47.5
    sensor_offset_ang = -48 # align camera with axis

    num_samples = 5000
    poses_rng = [[0, 0, 2, -15, -15, 0], [0, 0, 5, 15, 15, 0]]

    if mode == 'shear':
        moves_rng = [[-5, -5, 0, -5, -5, -5], [5, 5, 0, 5, 5, 5]]
    elif mode == 'tap':
        moves_rng = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]

    obj_poses = [[0, 0, 0, 0, 0, sensor_offset_ang]]

    target_df = make_target_df(poses_rng, moves_rng, num_samples, obj_poses, target_file, shuffle_data)

    # check save dir exists
    if os.path.isdir(collect_dir):
        input_str = input('Save directories already exists, would you like to continue (y,n)? ')
        if not str2bool(input_str):
            exit()
        else:
            empty_dir(collect_dir) # clear out existing files)

    # create dirs
    os.makedirs(collect_dir, exist_ok=True)
    os.makedirs(video_dir,   exist_ok=True)
    os.makedirs(image_dir,   exist_ok=True)

    # save metadata (remove unneccesary non json serializable stuff) and save params
    meta = locals().copy()
    del meta['collect_dir_name'], meta['target_df']
    with open(os.path.join(collect_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f)

    target_df.to_csv(target_file, index=False)

    def unwind_robot(robot):
        wrist_joint_3 = robot.joint_angles[5]

        if wrist_joint_3 > 350:
            unwind = True
            direction = -1
        elif wrist_joint_3 < -350:
            unwind = True
            direction = 1
        else:
            unwind = False

        if unwind == True:
            print('wrist joint 3 too close to robot limts: {}, unwinding...'.format(wrist_joint_3))
            robot.linear_speed = 50
            for i in range(4):
                robot.coord_frame = base_frame
                robot.coord_frame = robot.pose
                robot.move_linear([0,0,0,0,0,direction*90])
            robot.linear_speed = 100

        robot.coord_frame = work_frame

    # ==== data collection loop ====

    with make_robot() as robot, make_sensor() as sensor:
        # configure robot
        robot.tcp = robot_tcp

        # grab initial frames from sensor
        sensor.process(num_frames=2)

        # move to home position
        print("Moving to home position ...")
        robot.coord_frame = base_frame
        robot.linear_speed = 30
        robot.move_linear(home_pose)

        # move to origin of work frame
        print("Moving to work frame origin ...")
        robot.coord_frame = work_frame
        robot.move_linear([0, 0, 0, 0, 0, 0])

        # iterate over objects and poses
        robot.linear_speed = 100

        for index, row in target_df.iterrows():

            # pull neccesary info from data frame
            i_obj, i_pose = (int(row.loc['obj_id']), int(row.loc['pose_id']))
            obj_pose = row.loc['obj_pose']
            pose = row.loc['pose_1' : 'pose_6'].values.astype(np.float)
            move = row.loc['move_1' : 'move_6'].values.astype(np.float)
            sensor_image = row.loc['sensor_image']
            sensor_video = row.loc['sensor_video']

            # new pose relative to object pose
            new_pose = obj_pose + pose

            # print current info
            with np.printoptions(precision=2, suppress=True):
                print(f'Collecting data for object {i_obj}, pose {i_pose}: ...')
                # print(f'pose = {pose}'.replace('. ',''))
                # print(f'joints = {robot.joint_angles}'.replace('. ',''))

            # move to slightly above new pose (avoid changing pose in contact with object)
            robot.move_linear(new_pose - move + [0, 0, -5, 0, 0, 0])

            # move towards offset position
            robot.move_linear(new_pose - move)

            # move to target positon inducing shear effects
            robot.move_linear(new_pose)

            # process frames and save
            frames = sensor.process(num_frames=2, outfile=os.path.join(video_dir, sensor_video))

            # save image (use second image to avoid buffer issues)
            cv2.imwrite(os.path.join(image_dir, sensor_image), frames[1,:,:,:])

            # raise tip before next move
            robot.move_linear(new_pose + [0, 0, -5, 0, 0, 0])

            # check if robot is neearly tangled, if yes then unwind (becareful when making large jumps in angle)
            unwind_robot(robot)


        # move to home position
        print("Moving to home position ...")
        robot.coord_frame = base_frame
        robot.linear_speed = 30
        robot.move_linear(home_pose)


if __name__=="__main__":
    main()
