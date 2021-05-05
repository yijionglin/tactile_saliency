# -*- coding: utf-8 -*-

import os
import time
import json
import copy
import numpy as np
import pandas as pd
import cv2

import cri
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
                                 is_color=True,
                                 ),
            qsize=1,
            writer=CvVideoOutputFile(is_color=True),
        ))

class UR5_TacTip:

    def __init__(self, workframe, TCP_lims, sensor_offset_ang, action_lims):

        # set the work frame of the robot
        self.robot_tcp  = [0, 0, 108.0, 0, 0, 0]
        self.base_frame = [0, 0, 0, 0, 0, 0]
        self.home_pose  = [0, -451.0, 300, -180, 0, 0]
        self.work_frame = workframe
        self.sensor_offset_ang = sensor_offset_ang # align camera with axis

        # self.control_mode = 'sync'
        self.control_mode = 'async'
        self.control_timestep = 1.0/240 # sleep time before observation is pulled from camera

        # limit the region the TCP can go
        self.TCP_lims = TCP_lims

        # make the robot
        self.robot = make_robot()

        # make the sensor
        self.sensor = make_sensor()
        self.sensor.process(num_frames=2)

        # configure robot
        self.robot.tcp = self.robot_tcp

        # initialise variables
        self.init_joints([0.0, 0.0, 0.0], [0.0, 0.0, self.sensor_offset_ang])

        # move to home position
        print("Moving to home position ...")
        self.robot.linear_speed = 30
        self.robot.coord_frame = self.base_frame
        self.robot.move_linear(self.home_pose)

        # set the coord frame for the robot as the workframe
        self.set_coord_frame(self.work_frame)

        # set image dims for easy access
        temp_obs = self.process_sensor()
        self.image_dims = temp_obs.shape

    def init_joints(self, TCP_pos, TCP_rpy):
        # initial EE positions
        self.rel_TCP_pos = TCP_pos.copy()
        self.rel_TCP_rpy = TCP_rpy.copy()
        self.rel_TCP_pose = [*self.rel_TCP_pos , *self.rel_TCP_rpy]

    def set_coord_frame(self, frame):
        self.work_frame = frame
        self.robot.coord_frame = self.work_frame

    def reset(self):

        # if in async mode block last move before reset
        if self.control_mode == 'async':
            self.block_async_finished()

        # in order to test variation we could add a random orientation to the sensor here
        self.init_joints([0.0, 0.0, 0.0], [0.0, 0.0, self.sensor_offset_ang])

        # move to origin of work frame
        print("Moving to work frame origin ...")
        self.robot.linear_speed = 30
        self.robot.move_linear([0, 0, -10, 0, 0, self.sensor_offset_ang]) # move sensor to just above workframe
        self.robot.move_linear(self.rel_TCP_pose)

        # set robot speed for movement
        self.robot.linear_speed = 50

    def close(self):
        if self.robot is not None:
            # move to home position
            print("Moving to home position...")
            self.robot.coord_frame = self.base_frame
            self.robot.move_linear(self.home_pose)

            self.robot.close()
            self.robot = None
            print('Robot Closed')

        if self.sensor is not None:
            self.sensor.close()
            self.sensor = None
            print('Sensor Closed')

    def process_sensor(self):
        # pull 2 frames from buffer (of size 1) and use second frame
        # this ensures we are not one step delayed
        frames = self.sensor.process(num_frames=1)
        img = frames[0]
        return img

    def raise_tip(self, dist):
        print("Raising tip...")
        self.apply_action([0, 0, -dist, 0, 0, 0])
        # if in async mode block last move before reset
        if self.control_mode == 'async':
            self.block_async_finished()

    def block_async_finished(self):
        try:
            self.robot.async_result()
        except cri.robot.AsyncNotBusy:
            # Async not busy, possible duplicate calls to robot.async_result()
            # or call after move linear
            pass


    def apply_action(self, motor_commands):

        # add actions to current positions
        self.rel_TCP_pos[0] = self.rel_TCP_pos[0] + motor_commands[0]
        self.rel_TCP_pos[1] = self.rel_TCP_pos[1] + motor_commands[1]
        self.rel_TCP_pos[2] = self.rel_TCP_pos[2] + motor_commands[2]
        self.rel_TCP_rpy[0] = self.rel_TCP_rpy[0] + motor_commands[3]
        self.rel_TCP_rpy[1] = self.rel_TCP_rpy[1] + motor_commands[4]
        self.rel_TCP_rpy[2] = self.rel_TCP_rpy[2] + motor_commands[5]

        # limit actions to safe ranges
        self.check_lims()

        # change relative pose
        self.rel_TCP_pose = [*self.rel_TCP_pos , *self.rel_TCP_rpy]

        # ==== Arm TCP control ====
        # sync control
        if self.control_mode == 'sync':
            self.robot.move_linear(self.rel_TCP_pose)

        # async control
        elif self.control_mode == 'async':
            self.block_async_finished()
            self.robot.async_move_linear(self.rel_TCP_pose)
            time.sleep(self.control_timestep) # move for x amount of time before pulling obs

    def check_lims(self):
        self.rel_TCP_pos = np.clip(self.rel_TCP_pos, self.TCP_lims[:3,0], self.TCP_lims[:3,1])
        self.rel_TCP_rpy = np.clip(self.rel_TCP_rpy, self.TCP_lims[3:,0], self.TCP_lims[3:,1])


    def get_observation(self):
        observation = self.process_sensor()
        return observation
