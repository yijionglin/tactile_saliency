# -*- coding: utf-8 -*-

import os
import time
import json
import copy
import numpy as np
import pandas as pd
import cv2
import threading

import cri
from cri.robot import SyncRobot, AsyncRobot
from cri.controller import RTDEController

from vsp.video_stream import CvVideoDisplay, CvVideoOutputFile, CvVideoCamera
from vsp.processor import CameraStreamProcessorMT, AsyncProcessor

from tactile_gym.utils.general_utils import str2bool, save_json_obj, empty_dir

np.set_printoptions(precision=4, suppress=True)

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

    def __init__(self, control_mode, workframe, TCP_lims, sensor_offset_ang, action_lims, tactip_type):

        # set the work frame of the robot
        self.control_mode = control_mode

        if tactip_type == 'standard':
            self.robot_tcp  = [0, 0, 101.0, 0, 0, 0]
        elif tactip_type == 'flat':
            self.robot_tcp  = [0, 0, 88.5, 0, 0, 0]
        elif tactip_type == 'right_angle':
            self.robot_tcp  = [56.56, -56.56, 85.0, 0, 0, 0] # 80.0 mm at 45deg off
        else:
            sys.exit('Incorrect TacTip type specified {}'.format(tactip_type))

        self.base_frame = [0, 0, 0, 0, 0, 0]
        self.home_pose  = [0, -451.0, 300, -180, 0, 0]
        self.work_frame = workframe
        self.sensor_offset_ang = sensor_offset_ang # align camera with axis

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
        self.init_TCP([0.0, 0.0, 0.0], [0.0, 0.0, self.sensor_offset_ang])

        # initialise current pose
        self.current_TCP_pose = self.robot.pose

        # move to home position
        print("Moving to home position ...")
        self.robot.linear_speed = 30
        self.robot.angular_speed = 5
        self.robot.coord_frame = self.base_frame
        self.robot.move_linear(self.home_pose)

        # set the coord frame for the robot as the workframe
        self.set_coord_frame(self.work_frame)

        # set image dims for easy access
        temp_obs = self.process_sensor()
        self.image_dims = temp_obs.shape

        # for changing reset on subsequent runs
        # self.first_run = True

        # set the velocity control thread info
        if self.control_mode == 'TCP_velocity_control':
            self.setup_vel_control_thread()
            self.start_vel_thread()

    def setup_vel_control_thread(self):
        """
        set variables for running a background thread doing consistent,
        velocity moves with small timestep.
        This allows for smooth movement and fast interrupt when using velocity
        control.
        """
        self.rtde_client = self.robot.sync_robot.controller._client
        self.running = False
        self.vel_thread_lock = threading.Lock()
        self.velocity = (0, 0, 0, 0, 0, 0)
        self.acceleration = 2500    # mm/s rad/s?
        self.vel_ret_time = 1./20. # timestep sent to ur5 for vel commands

    def _vel_control_worker(self):
        """
        Worker thread sending velocity commmands to the robot. Updating the
        shared variable will update the command sent to the robot.
        """
        while self.running:
            with self.vel_thread_lock:
                worker_velocity = self.velocity
                worker_accel = self.acceleration
                worker_ret_time = self.vel_ret_time

            self.rtde_client.move_linear_speed(worker_velocity, worker_accel, worker_ret_time)

    def start_vel_thread(self):
        """
        Create and start a worker thread.
        """
        self.thread = threading.Thread(target=self._vel_control_worker, args=[], kwargs={})
        self.running = True
        self.thread.start()

    def stop_vel_thread(self):
        """
        Set the current velocity to 0 to stop the robot, break the update loop,
        then wait for the thread to finish executing.
        """
        # set velocity to 0
        with self.vel_thread_lock:
            self.velocity = [0,0,0,0,0,0]

        # finish the thread
        self.running = False
        self.thread.join()

    def init_TCP(self, TCP_pos, TCP_rpy):
        # initial EE positions
        self.rel_TCP_pos = TCP_pos.copy()
        self.rel_TCP_rpy = TCP_rpy.copy()
        self.rel_TCP_pose = [*self.rel_TCP_pos , *self.rel_TCP_rpy]

    def set_coord_frame(self, frame):
        self.work_frame = frame
        self.robot.coord_frame = self.work_frame

    def reset(self, reset_to_origin=True):

        # if velocity move still executing then stop
        if self.control_mode == 'TCP_velocity_control':
            self.stop_vel_thread()

        # in order to test variation we could add a random orientation to the sensor here
        self.init_TCP([0.0, 0.0, 0.0], [0.0, 0.0, self.sensor_offset_ang])

        # initialise current pose
        self.current_TCP_pose = self.robot.pose

        # move to origin of work frame
        if reset_to_origin:
            print("Moving to work frame origin ...")
            self.robot.linear_speed = 30
            self.robot.move_linear([0, 0, -10, 0, 0, self.sensor_offset_ang]) # move sensor to just above workframe

            self.robot.linear_speed = 10
            self.robot.move_linear(self.rel_TCP_pose)

            # set robot speed for movement
            self.robot.linear_speed = 30

        # start the velocity worker
        if self.control_mode == 'TCP_velocity_control':
            self.start_vel_thread()

    def close(self):

        if self.robot is not None:
            # try close the vel thread
            if self.control_mode == 'TCP_velocity_control':
                self.stop_vel_thread()

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

        # try close the vel thread
        if self.control_mode == 'TCP_velocity_control':
            self.stop_vel_thread()

        print("Raising tip...")

        # move away from surface using tcp orientation
        self.robot.coord_frame = self.base_frame
        self.robot.coord_frame = self.robot.pose
        self.robot.move_linear([0, 0, -dist, 0, 0, 0])
        self.robot.coord_frame = self.work_frame

        # move above workframe at high distance
        # self.robot.move_linear([0, 0, -150, 0, 0, self.sensor_offset_ang]) # move sensor to just above workframe

        # start the velocity worker
        if self.control_mode == 'TCP_velocity_control':
            self.start_vel_thread()

    def stop_robot(self):
        self.rtde_client.stop_linear(self.acceleration)

    def workframe_to_baseframe_vels(self, vels):
        """
        takes a 6 dim vector of velocities [dx, dy, dz, dRx, dRy, dRz] in the
        coord frame and converts them to the base frame for velocity control.
        """
        transformation_matrix = cri.transforms.euler2mat(self.work_frame, axes='rxyz')
        rotation_matrix = transformation_matrix[:3,:3]

        trans_xyz_vels = np.dot(vels[:3], rotation_matrix)
        trans_rpy_vels = np.dot(vels[3:], rotation_matrix)
        trans_vels = np.concatenate([trans_xyz_vels, trans_rpy_vels])

        return trans_vels

    def baseframe_to_workframe_vels(self, vels):
        """
        takes a 6 dim vector of velocities [dx, dy, dz, dRx, dRy, dRz] in the
        coord frame and converts them to the base frame for velocity control.
        """
        transformation_matrix = cri.transforms.euler2mat(self.work_frame, axes='rxyz')
        rotation_matrix = transformation_matrix[:3,:3]

        # A^-1 = A^T for orthonormal (rotation) matrices
        inv_rotation_matrix = rotation_matrix.T

        trans_xyz_vels = np.dot(vels[:3], inv_rotation_matrix)
        trans_rpy_vels = np.dot(vels[3:], inv_rotation_matrix)
        trans_vels = np.concatenate([trans_xyz_vels, trans_rpy_vels])

        return trans_vels

    def basevec_to_workvec(self, worldframe_vec):
        """
        Transforms a vector in world frame to a vector in work frame.
        """
        worldframe_vec = np.array(worldframe_vec)

        transformation_matrix = cri.transforms.euler2mat(self.work_frame, axes='rxyz')
        rotation_matrix = transformation_matrix[:3,:3]

        # A^-1 = A^T for orthonormal (rotation) matrices
        inv_rotation_matrix = rotation_matrix.T

        workframe_vec = inv_rotation_matrix.dot(worldframe_vec)

        return np.array(workframe_vec)

    def apply_position_action(self, actions):

        # add actions to current positions
        self.rel_TCP_pos[0] = self.rel_TCP_pos[0] + actions[0]
        self.rel_TCP_pos[1] = self.rel_TCP_pos[1] + actions[1]
        self.rel_TCP_pos[2] = self.rel_TCP_pos[2] + actions[2]
        self.rel_TCP_rpy[0] = self.rel_TCP_rpy[0] + actions[3]
        self.rel_TCP_rpy[1] = self.rel_TCP_rpy[1] + actions[4]
        self.rel_TCP_rpy[2] = self.rel_TCP_rpy[2] + actions[5]

        # limit actions to safe ranges
        self.check_pos_lims()

        # change relative pose
        self.rel_TCP_pose = [*self.rel_TCP_pos , *self.rel_TCP_rpy]

        # blocking move of the arm until target pose reached
        self.robot.move_linear(self.rel_TCP_pose)

    def apply_velocity_action(self, actions):

        # get current tcp pose
        self.current_TCP_pose = self.robot.pose

        # reduce velocities to 0 if we are currently at the TCP limits
        vels = self.check_vel_lims(actions)

        # convert from base frame to coord frame
        vels = self.workframe_to_baseframe_vels(vels)

        # update the velocity variable that will be continuously sent to
        # the robot via the velocity worker thread
        with self.vel_thread_lock:
            self.velocity = list(vels)

    def check_pos_lims(self):
        self.rel_TCP_pos = np.clip(self.rel_TCP_pos, self.TCP_lims[:3,0], self.TCP_lims[:3,1])
        self.rel_TCP_rpy = np.clip(self.rel_TCP_rpy, self.TCP_lims[3:,0], self.TCP_lims[3:,1])

    def check_vel_lims(self, vels):

        # get bool arrays for if limits are exceeded and if velocity is in
        # the direction that's exceeded
        exceed_pos_llims = np.logical_and(self.current_TCP_pose[:3] < self.TCP_lims[:3,0], vels[:3] < 0)
        exceed_pos_ulims = np.logical_and(self.current_TCP_pose[:3] > self.TCP_lims[:3,1], vels[:3] > 0)
        exceed_rpy_llims = np.logical_and(self.current_TCP_pose[3:] < self.TCP_lims[3:,0], vels[3:] < 0)
        exceed_rpy_ulims = np.logical_and(self.current_TCP_pose[3:] > self.TCP_lims[3:,1], vels[3:] > 0)

        # combine all bool arrays into one
        exceeded_pos = np.logical_or(exceed_pos_llims, exceed_pos_ulims)
        exceeded_rpy = np.logical_or(exceed_rpy_llims, exceed_rpy_ulims)
        exceeded = np.concatenate([exceeded_pos, exceeded_rpy])

        # cap the velocities at 0 if limits are exceeded
        capped_vels = np.array(vels)
        capped_vels[np.array(exceeded)] = 0

        # print('')
        # print(current_TCP_pose)
        # print(exceeded)
        # print(capped_vels)

        return capped_vels

    def get_observation(self):
        observation = self.process_sensor()
        return observation
