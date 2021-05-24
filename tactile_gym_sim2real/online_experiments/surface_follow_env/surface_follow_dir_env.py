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

from tactile_gym_sim2real.online_experiments.ur5_tactip import UR5_TacTip
from tactile_gym_sim2real.online_experiments.gan_net import pix2pix_GAN
from tactile_gym.assets import get_assets_path, add_assets_path

class SurfaceFollowDirEnv(gym.Env):

    def __init__(self,
                 env_modes,
                 gan_model_dir,
                 GanGenerator,
                 max_steps=1000,
                 rl_image_size=[64,64],
                 show_plot=True):

        self._observation = []
        self._env_step_counter = 0
        self._max_steps = max_steps
        self.rl_image_size = rl_image_size
        self.show_plot = show_plot
        self.first_run = True
        self.tactip_type = 'standard'

        # for incrementing direction each episode
        self.reset_counter = -1

        self.record_video = True
        if self.record_video:
            self.video_frames = []

        # define the movement mode used in the saved model
        self.movement_mode = env_modes['movement_mode']
        self.control_mode  = env_modes['control_mode']

        # set which task we're testing for setting workframe/limits
        self.task = 'taichi'
        # self.task = 'perspex'
        # self.task = 'ball'
        # self.task = '3d_surface'

        if self.task == 'taichi':
            # set the workframe for the tool center point origin
            self.work_frame = [0.0, -510.0, 150, -180, 0, 0]

            # set limits for the tool center point (rel to workframe)
            self.TCP_lims = np.zeros(shape=(6,2))
            self.TCP_lims[0,0], self.TCP_lims[0,1] = -50.0, 50.0  # x lims
            self.TCP_lims[1,0], self.TCP_lims[1,1] = -50.0, 50.0  # y lims
            self.TCP_lims[2,0], self.TCP_lims[2,1] = -50.0, 50.0  # z lims
            self.TCP_lims[3,0], self.TCP_lims[3,1] = -45, 45  # roll lims
            self.TCP_lims[4,0], self.TCP_lims[4,1] = -45, 45  # pitch lims
            self.TCP_lims[5,0], self.TCP_lims[5,1] = -179, 180    # yaw lims

        if self.task == 'perspex':
            # set the workframe for the tool center point origin
            self.work_frame = [-60.0, -510.0, 117, -180, 0, 0]

            # set limits for the tool center point (rel to workframe)
            self.TCP_lims = np.zeros(shape=(6,2))
            self.TCP_lims[0,0], self.TCP_lims[0,1] = -100.0, 200.0  # x lims
            self.TCP_lims[1,0], self.TCP_lims[1,1] = -100.0, 100.0  # y lims
            self.TCP_lims[2,0], self.TCP_lims[2,1] = -55.0, 65.0  # z lims
            self.TCP_lims[3,0], self.TCP_lims[3,1] = -45, 45  # roll lims
            self.TCP_lims[4,0], self.TCP_lims[4,1] = -45, 45  # pitch lims
            self.TCP_lims[5,0], self.TCP_lims[5,1] = -179, 180    # yaw lims

        if self.task == 'ball':
            # set the workframe for the tool center point origin
            self.work_frame = [109, -502.5, 205, -180, 0, 0]

            # set limits for the tool center point (rel to workframe)
            self.TCP_lims = np.zeros(shape=(6,2))
            self.TCP_lims[0,0], self.TCP_lims[0,1] = -50.0, 50.0  # x lims
            self.TCP_lims[1,0], self.TCP_lims[1,1] = -50.0, 50.0  # y lims
            self.TCP_lims[2,0], self.TCP_lims[2,1] = -50.0, 50.0  # z lims
            self.TCP_lims[3,0], self.TCP_lims[3,1] = -45, 45  # roll lims
            self.TCP_lims[4,0], self.TCP_lims[4,1] = -45, 45  # pitch lims
            self.TCP_lims[5,0], self.TCP_lims[5,1] = -179, 180    # yaw lims

        if self.task == '3d_surface':
            # set the workframe for the tool center point origin
            self.work_frame = [0.0, -420.0, 18, -180, 0, 0]

            # set limits for the tool center point (rel to workframe)
            self.TCP_lims = np.zeros(shape=(6,2))
            self.TCP_lims[0,0], self.TCP_lims[0,1] = -130.0, 130.0  # x lims
            self.TCP_lims[1,0], self.TCP_lims[1,1] = -130.0, 130.0  # y lims
            self.TCP_lims[2,0], self.TCP_lims[2,1] = -60.0, 0.0  # z lims
            self.TCP_lims[3,0], self.TCP_lims[3,1] = -45, 45  # roll lims
            self.TCP_lims[4,0], self.TCP_lims[4,1] = -45, 45  # pitch lims
            self.TCP_lims[5,0], self.TCP_lims[5,1] = -179, 180    # yaw lims

        # add rotation to yaw in order to allign camera without changing workframe
        self.sensor_offset_ang = -48

        # set up the action space
        self.setup_action_space()

        # load the trained pix2pix GAN network
        self.GAN = pix2pix_GAN(gan_model_dir=gan_model_dir, Generator=GanGenerator, rl_image_size=self.rl_image_size)

        # load saved border image files
        ref_images_path = add_assets_path(
            os.path.join('robot_assets', 'tactip', 'tactip_reference_images', 'standard')
        )

        border_gray_savefile = os.path.join( ref_images_path, str(self.rl_image_size[0]) + 'x' + str(self.rl_image_size[0]), 'nodef_gray.npy')
        border_mask_savefile = os.path.join( ref_images_path, str(self.rl_image_size[0]) + 'x' + str(self.rl_image_size[0]), 'border_mask.npy')
        self.border_gray = np.load(border_gray_savefile)
        self.border_mask = np.load(border_mask_savefile)

        # setup plot for rendering
        if self.show_plot:
            cv2.namedWindow('real_vs_generated')
            self._render_closed = False
        else:
            self._render_closed = True

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
            imageio.mimwrite(video_file, np.stack(self.video_frames), fps=5)

        # raise arm to avoid moving directly to workframe pos potentially hitting objects
        self._UR5.raise_tip(dist=40)
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
            max_ang_change = 1 # degree per step

            self.x_act_min, self.x_act_max = -max_pos_change, max_pos_change
            self.y_act_min, self.y_act_max = -max_pos_change, max_pos_change
            self.z_act_min, self.z_act_max = -max_pos_change, max_pos_change
            self.roll_act_min,  self.roll_act_max  = -max_ang_change, max_ang_change
            self.pitch_act_min, self.pitch_act_max = -max_ang_change, max_ang_change
            self.yaw_act_min,   self.yaw_act_max   = 0, 0

        elif self.control_mode == 'TCP_velocity_control':

            # approx sim_vel / 1.6
            max_pos_vel = 2.5              # mm/s
            max_ang_vel = 5  * (np.pi/180) # rad/s

            self.x_act_min, self.x_act_max = -max_pos_vel, max_pos_vel
            self.y_act_min, self.y_act_max = -max_pos_vel, max_pos_vel
            self.z_act_min, self.z_act_max = -max_pos_vel, max_pos_vel
            self.roll_act_min,  self.roll_act_max  = -max_ang_vel, max_ang_vel
            self.pitch_act_min, self.pitch_act_max = -max_ang_vel, max_ang_vel
            self.yaw_act_min,   self.yaw_act_max   = 0, 0

        # setup action space
        self.act_dim = self.get_act_dim()
        self.action_space = gym.spaces.Box(low=self.min_action,
                                           high=self.max_action,
                                           shape=(self.act_dim,),
                                           dtype=np.float32)

    def setup_observation_space(self):

        # image dimensions for sensor
        self.tactile_obs_dim = self.get_tactile_obs().shape

        self.observation_space = gym.spaces.Dict({
            'tactile': gym.spaces.Box(
                low=0, high=255, shape=self.tactile_obs_dim, dtype=np.uint8
            )
        })

    def reset(self):

        self._env_step_counter = 0

        # make a "goal" by setting a direction
        self.set_direction()

        # increment reset counter for iterating through directions
        self.reset_counter += 1

        # raise arm to avoid moving directly to workframe pos potentially hitting objects
        if not self.first_run:
            self._UR5.raise_tip(dist=40)

        # reset the ur5 arm
        self._UR5.reset()

        # get the starting observation
        self._observation = self.get_observation()

        # use to avoid doing things on first call to reset
        self.first_run = False

        return self._observation

    def set_direction(self):

        # default
        self.coord_directions = [0, 0, 0]

        if self.task == 'taichi':
            self.coord_directions = [0, 0, 0]

        elif self.task == 'perspex':
            self.coord_directions = [1, 0, 0]

        elif self.task in ['ball', '3d_surface']:
            # ang = np.random.uniform(-np.pi, np.pi)
            # ang = 3*np.pi/4
            ang = (self.reset_counter/2) * np.pi/4 # half for double reset

            self.coord_directions[0] = np.cos(ang)
            self.coord_directions[1] = np.sin(ang)

    def encode_actions(self, actions):
        """
        return actions as np.array in correct places for sending to ur5
        """

        encoded_actions = np.zeros(6)

        if self.movement_mode == 'yz':
            encoded_actions[2] = actions[0]
        if self.movement_mode == 'xyz':
            encoded_actions[2] = actions[0]
        if self.movement_mode == 'yzRx':
            encoded_actions[2] = actions[0]
            encoded_actions[3] = actions[1]
        if self.movement_mode == 'xyzRxRy':
            encoded_actions[2] = actions[0]
            encoded_actions[3] = actions[1]
            encoded_actions[4] = actions[2]

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

        # scale and embed actions appropriately
        encoded_actions = self.encode_actions(action)
        scaled_actions  = self.scale_actions(encoded_actions)

        # set actions for automatic movement to goal
        scaled_actions[0] = self.coord_directions[0] * self.x_act_max/2
        scaled_actions[1] = self.coord_directions[1] * self.y_act_max/2

        self._env_step_counter += 1

        # send action to ur5
        if self.control_mode == 'TCP_position_control':
            self._UR5.apply_position_action(scaled_actions)

        elif self.control_mode == 'TCP_velocity_control':
            self._UR5.apply_velocity_action(scaled_actions)

        # pull info after step
        done = self.termination()
        reward = self.reward()
        self._observation = self.get_observation()

        return self._observation, reward, done, {}

    def termination(self):

        # terminate when max ep len reached
        if self._env_step_counter >= self._max_steps:
            return True

        return False

    def reward(self):
        return 0

    def get_tactile_obs(self):
        # get image from sensor
        observation = self._UR5.get_observation()

        # process with gan here
        generated_sim_image, processed_real_image = self.GAN.gen_sim_image(observation)

        # add border to image
        generated_sim_image[self.border_mask==1] = self.border_gray[self.border_mask==1]

        # add a channel axis at end
        generated_sim_image = generated_sim_image[..., np.newaxis]

        # plot data
        if not self._render_closed:
            # resize to 256, 256 for video
            resized_real_image = cv2.resize(processed_real_image,
                                           (256,256),
                                           interpolation=cv2.INTER_NEAREST)
            resized_sim_image = cv2.resize(generated_sim_image,
                                           (256,256),
                                           interpolation=cv2.INTER_NEAREST)
            # show the images
            frame = np.hstack([resized_real_image, resized_sim_image])
            cv2.imshow('real_vs_generated', frame)
            if cv2.waitKey(1) & 0xFF == 27:
                cv2.destroyWindow('real_vs_generated')
                self._render_closed = True

            # save images in buffer
            if self.record_video:
                self.video_frames.append(frame)


        return generated_sim_image

    def get_observation(self):
        """
        Returns the observation dependent on which mode is set.
        """
        # init obs dict
        observation = {}
        observation['tactile'] = self.get_tactile_obs()
        return observation

    def get_act_dim(self):
        if self.movement_mode == 'yz':
            return 1
        if self.movement_mode == 'xyz':
            return 1
        if self.movement_mode == 'yzRx':
            return 2
        if self.movement_mode == 'xyzRxRy':
            return 3
