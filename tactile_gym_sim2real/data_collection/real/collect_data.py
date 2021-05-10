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

from tactile_gym.utils.general_utils import str2bool, save_json_obj, empty_dir

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

def unwind_robot(robot, work_frame, base_frame):
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

def collect_data(
        target_df,
        collect_dir,
        target_file,
        robot_tcp,
        base_frame,
        home_pose,
        work_frame,
        tap_depth,
    ):

    # check save dir exists already
    if os.path.isdir(collect_dir):
        input_str = input('Save directories already exists, would you like to continue (y,n)? ')
        if not str2bool(input_str):
            exit()
        else:
            empty_dir(collect_dir) # clear out existing files)

    # define dirs
    video_dir = os.path.join(collect_dir, 'videos')
    image_dir = os.path.join(collect_dir, 'images')

    # create dirs
    os.makedirs(collect_dir, exist_ok=True)
    os.makedirs(video_dir,   exist_ok=True)
    os.makedirs(image_dir,   exist_ok=True)

    # save metadata (remove unneccesary non json serializable stuff) and save params
    meta = locals().copy()
    del meta['target_df']
    with open(os.path.join(collect_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f)

    target_df.to_csv(target_file, index=False)

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

            # move to slightly above new pose (avoid changing pose in contact with object)
            robot.move_linear(new_pose - move + [0, 0, tap_depth, 0, 0, 0])

            # move towards offset position
            robot.move_linear(new_pose - move)

            # move to target positon inducing shear effects
            robot.move_linear(new_pose)

            # process frames and save
            frames = sensor.process(num_frames=2, outfile=os.path.join(video_dir, sensor_video))

            # save image (use second image to avoid buffer issues)
            cv2.imwrite(os.path.join(image_dir, sensor_image), frames[1,:,:,:])

            # raise tip before next move
            robot.move_linear(new_pose + [0, 0, tap_depth, 0, 0, 0])

            # check if robot is nearly tangled, if yes then unwind (be careful when making large jumps in angle)
            unwind_robot(robot, work_frame, base_frame)

        # move to home position
        print("Moving to home position ...")
        robot.coord_frame = base_frame
        robot.linear_speed = 30
        robot.move_linear(home_pose)


if __name__=="__main__":
    pass
