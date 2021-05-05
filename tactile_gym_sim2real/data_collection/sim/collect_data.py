import os
import pybullet as p
import pybullet_utils.bullet_client as bc
import time
import numpy as np
import argparse
import matplotlib.pyplot as plt
import pkgutil
import json
import pandas as pd
import cv2

from pybullet_sims.cri_envs.cri_ur5_tactip import CRI_UR5_TacTip

from pybullet_sims.utils.serialization_utils import convert_json
from pybullet_sims.utils.general_utils import str2bool, save_json_obj, empty_dir, check_dir
from pybullet_sims import models

def make_target_df_rand(poses_rng, moves_rng, num_poses, obj_poses, target_file, shuffle_data=False):
    # generate random poses
    np.random.seed()
    poses = np.random.uniform(low=poses_rng[0], high=poses_rng[1], size=(num_poses, 6))
    poses = poses[np.lexsort((poses[:,1], poses[:,5]))]
    moves = np.random.uniform(low=moves_rng[0], high=moves_rng[1], size=(num_poses, 6))

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

def make_target_df_csv(og_target_file, shuffle_data):

    # convert list saved as string to float array
    list_converter = lambda x: np.array(x.strip("[]").replace("'","").replace(","," ").split()).astype(np.float32)

    # read target data
    target_df = pd.read_csv(og_target_file, converters={"obj_pose": list_converter})

    # change filenames from video -> image
    target_df['sensor_image'] = target_df['sensor_video'].str.replace('video', 'image')
    target_df['sensor_image'] = target_df['sensor_image'].str.replace('mp4', 'png')

    # sort by angle (can make collection more accurate/faster due to smaller moves)
    # target_df = target_df.sort_values('pose_6')

    # shuffle randomly to remove any bias induced by ordered movement
    if shuffle_data:
        target_df = target_df.sample(frac=1).reset_index(drop=True) # shuffle randomly

    return target_df

def setup_pybullet_env(stim_path, tactip_params, stimulus_pos, stimulus_rpy, workframe_pos, workframe_rpy, show_gui, show_tactile):

    # ========= environment set up ===========
    time_step = 1./240 # low for small objects

    if show_gui:
        pb = bc.BulletClient(connection_mode=p.GUI)
    else:
        pb = bc.BulletClient(connection_mode=p.DIRECT)
        egl = pkgutil.get_loader('eglRenderer')
        if (egl):
          p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
        else:
          p.loadPlugin("eglRendererPlugin")

    pb.setGravity(0,0,-10)
    pb.setPhysicsEngineParameter(fixedTimeStep=time_step,
                                 numSolverIterations=300,
                                 numSubSteps=1,
                                 contactBreakingThreshold=0.0005,
                                 erp=0.05,
                                 contactERP=0.05,
                                 frictionERP=0.2, # need to enable friction anchors (something to experiment with)
                                 solverResidualThreshold=1e-7,
                                 contactSlop=0.001,
                                 globalCFM=0.0001)

    plane_id = pb.loadURDF(os.path.join(models.getDataPath(), "environment_objects/plane/plane.urdf"))

    if show_gui:
        # set debug camera position
        cam_dist = 1.0
        cam_yaw = 90
        cam_pitch = -25
        cam_pos = [0.65, 0, 0.025]
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW,0)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW,0)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW,0)
        p.resetDebugVisualizerCamera(cam_dist, cam_yaw, cam_pitch, cam_pos)

    # add stimulus
    stimulus_id  = p.loadURDF(stim_path, stimulus_pos, p.getQuaternionFromEuler(stimulus_rpy), useFixedBase=True)

    # define tactip type and image size
    tactip_type       = tactip_params['type']
    tactip_core       = tactip_params['core']
    tactip_image_size = tactip_params['image_size']
    tactip_border     = tactip_params['border']

    robot = CRI_UR5_TacTip(pb,
                           workframe_pos=workframe_pos,
                           workframe_rpy=workframe_rpy,
                           image_size=tactip_image_size,
                           add_border=tactip_border,
                           base_collisions_on=False,
                           tactip_type=tactip_type,
                           tactip_core=tactip_core,
                           show_gui=show_gui,
                           show_tactile=show_tactile)

    return robot


def collect_data(target_df, image_dir, stim_path, stimulus_pos, stimulus_rpy, workframe_pos, workframe_rpy, tactip_params, show_gui=True, show_tactile=True):

    # setup robot data collection env
    robot = setup_pybullet_env(stim_path, tactip_params, stimulus_pos, stimulus_rpy, workframe_pos, workframe_rpy, show_gui, show_tactile)
    max_iter_steps = 2000 # max num iterations for IK solver and movement
    tap_depth = -0.005  # 5mm

    # ==== data collection loop ====
    # move to work frame
    robot.move_linear([0,0,0],[0,0,0], max_steps=2000, gain=0.5)

    for index, row in target_df.iterrows():
        i_obj  = int(row.loc['obj_id'])
        i_pose = int(row.loc['pose_id'])
        pose = row.loc['pose_1' : 'pose_6'].values.astype(np.float)
        move = row.loc['move_1' : 'move_6'].values.astype(np.float)
        obj_pose = row.loc['obj_pose']
        sensor_image = row.loc['sensor_image']

        # define the new pos and rpy
        obj_pose_array = np.array([float(i) for i in obj_pose])
        obj_pose_array[5] = 0.0 # remove offset angle (TODO: sort this)

        pose_array     = np.array([float(i) for i in pose])
        move_array     = np.array([float(i) for i in move])

        # combine relative pose and object pose
        new_pose = obj_pose_array + pose_array
        # new_pose = pose_array

        # convery to pybullet form
        final_pos = new_pose[:3] * 0.001     # to mm
        final_rpy = new_pose[3:] * np.pi/180 # to rad
        move_pos  = move_array[:3] * 0.001     # to mm
        move_rpy  = move_array[3:] * np.pi/180 # to rad

        with np.printoptions(precision=2, suppress=True):
            print(f'Collecting data for object {i_obj}, pose {i_pose}: ...')

        # move to slightly above new pose (avoid changing pose in contact with object)
        robot.move_linear(final_pos - move_pos + [0, 0, tap_depth], final_rpy - move_rpy, max_steps=max_iter_steps, gain=0.5)
        # time.sleep(1)

        # move down to offset position
        robot.move_linear(final_pos - move_pos, final_rpy - move_rpy, max_steps=max_iter_steps, gain=0.5)
        # time.sleep(1)

        # move to target positon inducing shear effects
        robot.move_linear(final_pos, final_rpy, max_steps=max_iter_steps, gain=0.5)
        # time.sleep(1)

        # process frames and save
        img = robot.process_sensor()

        # raise tip before next move
        robot.move_linear(final_pos + [0, 0, tap_depth], final_rpy)
        # time.sleep(1)

        # save tap img
        image_outfile = os.path.join(image_dir, sensor_image)
        cv2.imwrite(image_outfile, img)

        # debugging
        robot.UR5.draw_workframe()

    # move to home position
    print("Moving to home position ...")
    robot.move_home_pos()


if __name__=="__main__":
    pass
