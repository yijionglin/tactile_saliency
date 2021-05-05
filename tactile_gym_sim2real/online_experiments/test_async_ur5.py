import os
import time
import numpy as np

import cri
from cri.robot import SyncRobot, AsyncRobot
from cri.controller import RTDEController

from pybullet_sims.utils.general_utils import str2bool, save_json_obj, empty_dir

def main():

    base_frame = (0, 0, 0, 0, 0, 0)
#    work_frame = (109.1, -487.0, 341.3, 180, 0, -90)   # base frame: x->right, y->back, z->up
    work_frame = (0, -451.0, 300, -180, 0, 0)

    with AsyncRobot(SyncRobot(RTDEController(ip='192.11.72.10'))) as robot:
    # with AsyncRobot(SyncRobot(RTDEController(ip='127.0.0.1'))) as robot:
        robot.tcp = (0, 0, 108, 0, 0, 0)
        robot.linear_speed = 50
        robot.angular_speed = 5
        robot.coord_frame = work_frame

        # Display robot info
        print("Robot info: {}".format(robot.info))

        # Display initial joint angles
        print("Initial joint angles: {}".format(robot.joint_angles))

        # Display initial pose in work frame
        print("Initial pose in work frame: {}".format(robot.pose))

        # Move to origin of work frame
        print("Moving to origin of work frame ...")
        robot.move_linear((0, 0, 0, 0, 0, 0))

        # Move backward and forward (async)
        print("Moving backward and forward (async) ...")
        robot.async_move_linear((20, 0, 0, 0, 0, 0))
        print("Getting on with something else while command completes ...")
        robot.async_result()
        robot.async_move_linear((0, 0, 0, 0, 0, 0))
        print("Getting on with something else while command completes ...")
        robot.async_result()

        # Move right and left
        print("Moving right and left (async) ...")
        robot.async_move_linear((0, 20, 0, 0, 0, 0))
        print("Getting on with something else while command completes ...")
        robot.async_result()
        robot.async_move_linear((0, 0, 0, 0, 0, 0))
        print("Getting on with something else while command completes ...")
        robot.async_result()

        # Move down and up (async)
        print("Moving down and up (async) ...")
        robot.async_move_linear((0, 0, 20, 0, 0, 0))
        print("Getting on with something else while command completes ...")
        robot.async_result()
        robot.async_move_linear((0, 0, 0, 0, 0, 0))
        print("Getting on with something else while command completes ...")
        robot.async_result()

        # Roll right and left (async)
        print("Rolling right and left (async) ...")
        robot.async_move_linear((0, 0, 0, 20, 0, 0))
        print("Getting on with something else while command completes ...")
        robot.async_result()
        robot.async_move_linear((0, 0, 0, 0, 0, 0))
        print("Getting on with something else while command completes ...")
        robot.async_result()

        # Roll forward and backward (async)
        print("Rolling forward and backward (async) ...")
        robot.async_move_linear((0, 0, 0, 0, 20, 0))
        print("Getting on with something else while command completes ...")
        robot.async_result()
        robot.async_move_linear((0, 0, 0, 0, 0, 0))
        print("Getting on with something else while command completes ...")
        robot.async_result()

        # Turn clockwise and anticlockwise around work frame z-axis (async)
        print("Turning clockwise and anticlockwise around work frame z-axis (async) ...")
        robot.async_move_linear((0, 0, 0, 0, 0, 20))
        print("Getting on with something else while command completes ...")
        robot.async_result()
        robot.async_move_linear((0, 0, 0, 0, 0, 0))
        print("Getting on with something else while command completes ...")
        robot.async_result()


if __name__ == '__main__':
    main()
