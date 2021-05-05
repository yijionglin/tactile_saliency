import os, inspect
import math
import gym
import numpy as np
import time
import random
import pkgutil
from pkg_resources import parse_version
import cv2
import imageio
import matplotlib.pyplot as plt
import cri
import pybullet as pb

from pybullet_real2sim.online_experiments.ur5_tactip import UR5_TacTip
from pybullet_real2sim.online_experiments.gan_net import pix2pix_GAN

from pybullet_sims.rl_envs.ur5_envs.tactip_reference_images import *

class ObjectPushEnv(gym.Env):

    def __init__(self,
                 env_modes,
                 gan_model_dir,
                 max_steps=1000,
                 image_size=[64,64],
                 add_border=False,
                 show_plot=True):

        self._observation = []
        self._env_step_counter = 0
        self._max_steps = max_steps
        self.image_size = image_size
        self.add_border = add_border
        self.show_plot = show_plot
        self.first_run = True
        self.tactip_type = 'right_angle'

        # for incrementing workframe each episode
        self.reset_counter = -1

        self.record_video = True
        if self.record_video:
            self.video_frames = []

        # define the movement mode used in the saved model
        self.movement_mode = env_modes['movement_mode']
        self.control_mode  = env_modes['control_mode']

        # what traj to generate
        self.traj_type = 'straight'

        # set the workframe for the tool center point origin
        # self.work_frame = [0.0, -420.0, 200, -180, 0, 0] # safe
        self.work_frame = [-200.0, -420.0, 55, -180, 0, 0] # x on blue mat

        # add rotation to yaw in order to allign camera without changing workframe
        self.sensor_offset_ang = 45

        # set limits for the tool center point (rel to workframe)
        self.TCP_lims = np.zeros(shape=(6,2))
        self.TCP_lims[0,0], self.TCP_lims[0,1] = -50.0, 400.0  # x lims
        self.TCP_lims[1,0], self.TCP_lims[1,1] = -100.0, 100.0  # y lims
        self.TCP_lims[2,0], self.TCP_lims[2,1] = 0.0, 0.0     # z lims
        self.TCP_lims[3,0], self.TCP_lims[3,1] = 0.0, 0.0     # roll lims
        self.TCP_lims[4,0], self.TCP_lims[4,1] = 0.0, 0.0     # pitch lims
        self.TCP_lims[5,0], self.TCP_lims[5,1] = self.sensor_offset_ang - 45, self.sensor_offset_ang + 45    # yaw lims

        # setup action space to match sim
        self.setup_action_space()

        # load the trained pix2pix GAN network
        self.GAN = pix2pix_GAN(gan_model_dir=gan_model_dir, rl_image_size=self.image_size)

        # load saved border image files
        border_gray_savefile = os.path.join( getBorderImagesPath(), 'standard', str(self.image_size[0]) + 'x' + str(self.image_size[0]), 'border_gray.npy')
        border_mask_savefile = os.path.join( getBorderImagesPath(), 'standard', str(self.image_size[0]) + 'x' + str(self.image_size[0]), 'border_mask.npy')
        self.border_gray = np.load(border_gray_savefile)
        self.border_mask = np.load(border_mask_savefile)

        # set up plot for generated image
        if self.show_plot:
            plt.ion()
            plot_data = np.random.rand(self.image_size[0], self.image_size[1])
            self._fig, self._ax = plt.subplots(1,2, figsize=(10,5))
            self._real_image_window = self._ax[0].imshow(plot_data, interpolation='none', animated=True, label="tactip_view", vmin=0, vmax=255, cmap='gray')
            self._gen_image_window  = self._ax[1].imshow(plot_data, interpolation='none', animated=True, label="tactip_view", vmin=0, vmax=255, cmap='gray')
            self._ax[0].set_title('Processed Real Image')
            self._ax[1].set_title('Generated Sim Image')
            plt.tight_layout()
            plt.pause(0.001)

        # setup the UR5
        self._UR5 = UR5_TacTip(control_mode=self.control_mode,
                               workframe=self.work_frame,
                               TCP_lims=self.TCP_lims,
                               sensor_offset_ang=self.sensor_offset_ang,
                               action_lims=[self.min_action, self.max_action],
                               tactip_type=self.tactip_type)


        # this is needed to set some variables used for initial observation/obs_dim()
        self.reset()

        # set the observation space
        self.setup_observation_space()

        self.seed()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        # save recorded video
        if self.record_video:
            video_file = os.path.join('collected_data', 'tactile_video.mp4')
            imageio.mimwrite(video_file, np.stack(self.video_frames), fps=7)

        self._UR5.close()

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def setup_action_space(self):

        # these are used for bounds on the action space in SAC and clipping
        # range for PPO
        self.min_action, self.max_action  = -0.25,  0.25

        # define action ranges per act dim to rescale output of policy
        if self.control_mode == 'TCP_position_control':

            max_pos_change = 1
            max_ang_change = 1 * (np.pi/180) # degree per step

            self.x_act_min, self.x_act_max = -max_pos_change, max_pos_change
            self.y_act_min, self.y_act_max = -max_pos_change, max_pos_change
            self.z_act_min, self.z_act_max = 0, 0
            self.roll_act_min,  self.roll_act_max  = 0, 0
            self.pitch_act_min, self.pitch_act_max = 0, 0
            self.yaw_act_min,   self.yaw_act_max   = -max_ang_change, max_ang_change

        elif self.control_mode == 'TCP_velocity_control':

            # approx sim_vel / 1.6
            max_pos_vel = 10                # mm/s
            max_ang_vel = 5  * (np.pi/180) # rad/s

            self.x_act_min, self.x_act_max = -max_pos_vel, max_pos_vel
            self.y_act_min, self.y_act_max = -max_pos_vel, max_pos_vel
            self.z_act_min, self.z_act_max = 0, 0
            self.roll_act_min,  self.roll_act_max  = 0, 0
            self.pitch_act_min, self.pitch_act_max = 0, 0
            self.yaw_act_min,   self.yaw_act_max   = -max_ang_vel, max_ang_vel

        # setup action space
        self.act_dim = self.get_act_dim()
        self.action_space = gym.spaces.Box(low=self.min_action,
                                           high=self.max_action,
                                           shape=(self.act_dim,),
                                           dtype=np.float32)

    def setup_observation_space(self):

        # image dimensions for sensor
        self.obs_dim = self.get_obs_dim()
        self.observation_space = gym.spaces.Box(low=-np.inf,
                                                high=np.inf,
                                                shape=self.obs_dim,
                                                dtype=np.float32)

    def setup_traj(self):
        self.traj_n_points = 20
        self.traj_spacing = 0.0125
        self.traj_max_perturb = 0.05

        self.goal_update_rate = int(self._max_steps / self.traj_n_points)

        # setup traj arrays
        self.targ_traj_list_id = -1
        self.traj_pos_workframe = np.zeros(shape=(self.traj_n_points,3))
        self.traj_rpy_workframe = np.zeros(shape=(self.traj_n_points,3))

        if self.traj_type == 'simplex':
            self.load_trajectory_simplex()
        elif self.traj_type == 'straight':
            self.load_trajectory_straight()
        elif self.traj_type == 'curve':
            self.load_trajectory_curve()
        elif self.traj_type == 'sin':
            self.load_trajectory_sin()
        else:
            sys.exit('Incorrect traj_type specified: {}'.format(self.traj_type))

        traj_idx = int(self.reset_counter / 2)
        np.save('collected_data/traj_pos_{}.npy'.format(traj_idx), self.traj_pos_workframe)
        np.save('collected_data/traj_rpy_{}.npy'.format(traj_idx), self.traj_rpy_workframe)

        self.update_goal()

    def load_trajectory_straight(self):

        # randomly pick traj direction
        # traj_ang = np.random.uniform(-np.pi/8, np.pi/8)
        traj_angs = [-np.pi/8, np.pi/8, 0.0]
        # traj_angs = [0.0, 0.0, 0.0, 0.0]
        traj_idx = int(self.reset_counter / 2)
        init_offset = 0.04 + self.traj_spacing

        for i in range(int(self.traj_n_points)):

            traj_ang = traj_angs[traj_idx]

            dir_x = np.cos(traj_ang)
            dir_y  = np.sin(traj_ang)
            dist = (i*self.traj_spacing)

            x = init_offset + dist*dir_x
            y = dist*dir_y
            z = 0.0
            self.traj_pos_workframe[i] = [x, y, z]

        # calc orientation to place object at
        self.traj_rpy_workframe[:,2] = np.gradient(self.traj_pos_workframe[:,1], self.traj_spacing)

    def load_trajectory_curve(self):

        # pick traj direction
        traj_idx = int(self.reset_counter / 2)
        curve_dir = -1 if traj_idx % 2 == 0 else +1

        def curve_func(x):
            y = curve_dir*x**2
            return y

        init_offset = 0.04 + self.traj_spacing

        for i in range(int(self.traj_n_points)):
            dist = (i*self.traj_spacing)
            x = init_offset + dist
            y = curve_func(x)
            z = 0.0
            self.traj_pos_workframe[i] = [x, y, z]

        # calc orientation to place object at
        self.traj_rpy_workframe[:,2] = np.gradient(self.traj_pos_workframe[:,1], self.traj_spacing)

    def load_trajectory_sin(self):

        #  pick traj direction
        traj_idx = int(self.reset_counter / 2)
        curve_dir = -1 if traj_idx % 2 == 0 else +1
        init_offset = 0.04 + self.traj_spacing

        def curve_func(x):
            y = curve_dir*0.025*np.sin(20*(x-init_offset))
            return y


        for i in range(int(self.traj_n_points)):
            dist = (i*self.traj_spacing)
            x = init_offset + dist
            y = curve_func(x)
            z = 0.0
            self.traj_pos_workframe[i] = [x, y, z]

        # calc orientation to place object at
        self.traj_rpy_workframe[:,2] = np.gradient(self.traj_pos_workframe[:,1], self.traj_spacing)


    def update_goal(self):

        # increment targ list
        self.targ_traj_list_id += 1

        if self.targ_traj_list_id >= self.traj_n_points-1:
            self.targ_traj_list_id = self.traj_n_points-1

        # create variables for goal pose in workframe to use later
        self.goal_pos_workframe = self.traj_pos_workframe[self.targ_traj_list_id]
        self.goal_rpy_workframe = self.traj_rpy_workframe[self.targ_traj_list_id]


    def reset(self):

        self._env_step_counter = 0

        # increment reset counter for iterating through directions
        self.reset_counter += 1

        # reset the ur5 arm
        self._UR5.reset()

        # reset the goal
        self.setup_traj()

        # get the starting observation
        self._observation = self.get_extended_observation()

        # use to avoid doing things on first call to reset
        self.first_run = False

        return self._observation

    def get_tip_direction_workframe(self):
        """
        Warning, deadline research code (specific to current workframe)
        """
        # get rotation from current tip orientation
        current_tip_pose = self._UR5.current_TCP_pose

        # angle for perp and par vectors
        par_ang  = ( current_tip_pose[5]  ) * np.pi/180
        perp_ang = ( current_tip_pose[5] - 90 ) * np.pi/180

        # create vectors (directly in workframe) pointing in perp and par directions of current sensor
        workframe_par_tip_direction  = np.array([np.cos(par_ang),  np.sin(par_ang), 0]) # vec pointing outwards from tip
        workframe_perp_tip_direction = np.array([np.cos(perp_ang), np.sin(perp_ang),0]) # vec pointing perp to tip

        return workframe_par_tip_direction, workframe_perp_tip_direction

    def encode_TCP_frame_actions(self, actions):
        """
        Warning, deadline research code (specific to current workframe)
        """

        encoded_actions = np.zeros(6)

        workframe_par_tip_direction, workframe_perp_tip_direction = self.get_tip_direction_workframe()

        if self.movement_mode == 'TyRz':

            # translate the direction
            perp_scale = actions[0]
            print(perp_scale)
            perp_action = np.dot(workframe_perp_tip_direction, perp_scale)

            # auto move in the dir tip is pointing
            # par_scale = 1.0 # always at max
            par_scale = 1.0*self.max_action
            par_action = np.dot(workframe_par_tip_direction, par_scale)

            encoded_actions[0] += perp_action[0] + par_action[0]
            encoded_actions[1] += perp_action[1] + par_action[1]
            encoded_actions[5] += actions[1]

        elif self.movement_mode == 'TxTyRz':

            # translate the direction
            perp_scale = actions[1]
            perp_action = np.dot(workframe_perp_tip_direction, perp_scale)

            par_scale = actions[0]
            par_action = np.dot(workframe_par_tip_direction, par_scale)

            encoded_actions[0] += perp_action[0] + par_action[0]
            encoded_actions[1] += perp_action[1] + par_action[1]
            encoded_actions[5] += actions[2]

        return encoded_actions

    def encode_work_frame_actions(self, actions):
        """
        Return actions as np.array in correct places for sending to ur5.
        """

        encoded_actions = np.zeros(6)

        if self.movement_mode == 'y':
            encoded_actions[0] = self.max_action
            encoded_actions[1] = actions[0]

        if self.movement_mode == 'yRz':
            encoded_actions[0] = self.max_action
            encoded_actions[1] = actions[0]
            encoded_actions[5] = actions[1]

        elif self.movement_mode == 'xyRz':
            encoded_actions[0] = actions[0]
            encoded_actions[1] = actions[1]
            encoded_actions[5] = actions[2]

        return encoded_actions

    def scale_actions(self, actions):

        # would prefer to enforce action bounds on algorithm side, but this is ok for now
        actions = np.clip(actions, self.min_action, self.max_action)

        input_range = (self.max_action - self.min_action)

        new_x_range = (self.x_act_max - self.x_act_min)
        new_y_range = (self.y_act_max - self.y_act_min)
        new_z_range = (self.z_act_max - self.z_act_min)
        new_roll_range  = (self.roll_act_max  - self.roll_act_min)
        new_pitch_range = (self.pitch_act_max - self.pitch_act_min)
        new_yaw_range   = (self.yaw_act_max   - self.yaw_act_min)

        scaled_actions = [
            (((actions[0] - self.min_action) * new_x_range) / input_range) + self.x_act_min,
            (((actions[1] - self.min_action) * new_y_range) / input_range) + self.y_act_min,
            (((actions[2] - self.min_action) * new_z_range) / input_range) + self.z_act_min,
            (((actions[3] - self.min_action) * new_roll_range)  / input_range) + self.roll_act_min,
            (((actions[4] - self.min_action) * new_pitch_range) / input_range) + self.pitch_act_min,
            (((actions[5] - self.min_action) * new_yaw_range)   / input_range) + self.yaw_act_min,
        ] # 6 dim when sending to ur5

        return np.array(scaled_actions)

    def step(self, action):

        start_time = time.time()

        # scale and embed actions appropriately
        if self.movement_mode in ['y', 'yRz', 'xyRz']:
            encoded_actions = self.encode_work_frame_actions(action)
        elif self.movement_mode in ['TyRz', 'TxTyRz']:
            encoded_actions = self.encode_TCP_frame_actions(action)

        scaled_actions  = self.scale_actions(encoded_actions)

        self._env_step_counter += 1

        # send action to ur5
        if self.control_mode == 'TCP_position_control':
            self._UR5.apply_position_action(scaled_actions)

        elif self.control_mode == 'TCP_velocity_control':
            self._UR5.apply_velocity_action(scaled_actions)

        # pull info after step
        done = self.termination()
        reward = self.reward()
        self._observation = np.array(self.get_extended_observation())

        if self._env_step_counter % self.goal_update_rate == 0:
            self.update_goal()

        return self._observation, reward, done, {}

    def termination(self):
        # terminate when max ep len reached
        if self._env_step_counter >= self._max_steps:
            return True
        return False

    def reward(self):
        return 0


    def get_extended_observation(self):
        # get image from sensor
        observation = self._UR5.get_observation()

        # process with gan here
        generated_sim_image, processed_real_image = self.GAN.gen_sim_image(observation)

        # add border to image
        if self.add_border:
            generated_sim_image[self.border_mask==1] = self.border_gray[self.border_mask==1]

        # add a channel axis at end
        generated_sim_image = generated_sim_image[..., np.newaxis]

        # plot data
        if self.show_plot:

            self._real_image_window.set_data(processed_real_image)
            self._gen_image_window.set_data(generated_sim_image)
            self._ax[0].plot([0])
            self._ax[1].plot([0])
            plt.pause(0.001)

        if self.record_video:
            # resize to 256, 256 for video
            resized_sim_image = cv2.resize(generated_sim_image,
                                           (256,256),
                                           interpolation=cv2.INTER_NEAREST)
            resized_real_image = cv2.resize(processed_real_image,
                                           (256,256),
                                           interpolation=cv2.INTER_NEAREST)

            frame = np.hstack([resized_real_image, resized_sim_image])
            self.video_frames.append(frame)

        ## ============ Extend observation with features ================
        # get pose in workframe
        robot_pose = self._UR5.current_TCP_pose
        robot_pose[5] -= self.sensor_offset_ang

        # Hacky and shouldnt really work but it does
        tcp_pos_workframe = robot_pose[:3] * 0.001
        tcp_rpy_workframe = robot_pose[3:] * np.pi/180

        feature_array = np.array([*tcp_pos_workframe,  *tcp_rpy_workframe,
                                  *self.goal_pos_workframe, *self.goal_rpy_workframe
                                  ])

        # print('')
        # print(tcp_pos_workframe)
        # print(tcp_rpy_workframe)
        # print(self.goal_pos_workframe)
        # print(self.goal_rpy_workframe)

        num_features = len(feature_array)
        padded_feature_array = np.zeros(self.image_size)
        padded_feature_array[0, :num_features] = feature_array

        extended_observation = np.dstack([generated_sim_image/255.0, padded_feature_array])

        return extended_observation

    def get_act_dim(self):
        if self.movement_mode == 'y':
            return 1
        elif self.movement_mode == 'yRz':
            return 2
        elif self.movement_mode == 'xyRz':
            return 3
        if self.movement_mode == 'TyRz':
            return 2
        if self.movement_mode == 'TxTyRz':
            return 3
        else:
            sys.exit('Incorrect movement mode specified: {}'.format(self.movement_mode))

    def get_obs_dim(self):
        return self.get_extended_observation().shape
