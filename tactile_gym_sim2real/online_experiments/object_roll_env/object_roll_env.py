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

class ObjectRollEnv(gym.Env):

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
        self.tactip_type = 'flat'

        self.record_video = True
        if self.record_video:
            self.video_frames = []

        self.save_traj = True
        if self.save_traj:
            self.pixel_traj = []
            self.goal_traj = []

        # define the movement mode used in the saved model
        self.movement_mode = env_modes['movement_mode']
        self.control_mode  = env_modes['control_mode']

        # task mode, use hand for testing
        # self.task = 'from_hand'
        self.task = 'from_pad'
        self.randomise_goal = True
        self.reset_to_origin = False
        self.reset_counter = -1

        # set the workframe for the tool center point origin
        if self.task == 'from_hand':
            self.work_frame = [0.0, -450.0, 200, -180, 0, 0]
        elif self.task == 'from_pad':
            # be careful as close to base
            self.work_frame = [0.0, -450.0, 19.5, -180, 0, 0] # 8mm
            # self.work_frame = [0.0, -450.0, 18.5, -180, 0, 0] # 6mm
            # self.work_frame = [0.0, -450.0, 17.0, -180, 0, 0] # 4mm
            # self.work_frame = [0.0, -450.0, 15.0, -180, 0, 0] # 2mm

        # set limits for the tool center point (rel to workframe)
        self.TCP_lims = np.zeros(shape=(6,2))
        self.TCP_lims[0,0], self.TCP_lims[0,1] = -50.0, 50.0  # x lims
        self.TCP_lims[1,0], self.TCP_lims[1,1] = -50.0, 50.0  # y lims
        self.TCP_lims[2,0], self.TCP_lims[2,1] = 0.0, 0.0     # z lims
        self.TCP_lims[3,0], self.TCP_lims[3,1] = 0.0, 0.0     # roll lims
        self.TCP_lims[4,0], self.TCP_lims[4,1] = 0.0, 0.0     # pitch lims
        self.TCP_lims[5,0], self.TCP_lims[5,1] = -179, 180    # yaw lims

        # add rotation to yaw in order to allign camera without changing workframe
        self.sensor_offset_ang = -48

        # setup action space to match sim
        self.setup_action_space()

        # load the trained pix2pix GAN network
        self.GAN = pix2pix_GAN(gan_model_dir=gan_model_dir, Generator=GanGenerator, rl_image_size=self.rl_image_size)

        # load saved border image files
        ref_images_path = add_assets_path(
            os.path.join('robot_assets', 'tactip', 'tactip_reference_images', 'flat')
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

        # Set up the detector with default parameters.
        self.setup_blob_detection()

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
            bgr_frames = np.stack(self.video_frames)
            rgb_frames = bgr_frames[:,:,:,[2, 1, 0]]
            imageio.mimwrite(video_file, rgb_frames, fps=10)

        if self.save_traj:
            pixel_traj_file = os.path.join('collected_data', 'pixel_traj.csv')
            goal_traj_file = os.path.join('collected_data', 'goal_traj.csv')
            np.savetxt(pixel_traj_file, self.pixel_traj, delimiter=",")
            np.savetxt(goal_traj_file, self.goal_traj, delimiter=",")

        # raise arm to avoid moving directly to workframe pos potentially hitting objects
        self._UR5.raise_tip(dist=10)
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
            self.yaw_act_min,   self.yaw_act_max   = 0, 0

        elif self.control_mode == 'TCP_velocity_control':

            # approx sim_vel / 1.6
            # max_pos_vel = 2.5                # mm/s
            max_pos_vel = 5                # mm/s
            max_ang_vel = 0  * (np.pi/180) # rad/s

            self.x_act_min, self.x_act_max = -max_pos_vel, max_pos_vel
            self.y_act_min, self.y_act_max = -max_pos_vel, max_pos_vel
            self.z_act_min, self.z_act_max = 0, 0
            self.roll_act_min,  self.roll_act_max  = 0, 0
            self.pitch_act_min, self.pitch_act_max = 0, 0
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
        self.feature_obs_dim = self.get_feature_obs().shape

        self.observation_space = gym.spaces.Dict({
            'tactile': gym.spaces.Box(
                low=0, high=255, shape=self.tactile_obs_dim, dtype=np.uint8
            ),
            'extended_feature': gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=self.feature_obs_dim, dtype=np.float32
            )
        })

    def setup_goal(self):
        # for now just place in center
        self.goal_pos_tcp = np.array([0.0,0.0,0.0])
        self.pixel_distance_thresh = 5
        goal_far_enough = False

        # repeatedly generate random goal until one is far enough
        while not goal_far_enough:
            if self.randomise_goal:
                # overwrite x, y with spherical random pose generation within
                # 12.5mm disk
                ang = np.random.uniform(0, 2*np.pi)
                rad = np.random.uniform(0, 1)

                # set the limit of the radius
                rad_lim = 0.01

                # use sqrt r to get uniform disk spread
                x = np.sqrt(rad)*np.cos(ang) * rad_lim
                y = np.sqrt(rad)*np.sin(ang) * rad_lim
                self.goal_pos_tcp[0], self.goal_pos_tcp[1] = x, y

            # get the coords of the goal in image space
            # min/max from 20mm radius tip + extra for border
            min, max = -0.021, 0.021
            norm_tcp_pos_x = (self.goal_pos_tcp[0] - min) / (max - min)
            norm_tcp_pos_y = (self.goal_pos_tcp[1] - min) / (max - min)

            self.goal_pixel_coords = (
                int(norm_tcp_pos_x*self.rl_image_size[0]),
                int(norm_tcp_pos_y*self.rl_image_size[1])
            )

            # break when goal > 2 * distance thresh away
            if self.latest_obj_pixel_coords is not None:
                goal_far_enough = not self.check_pixel_dist(2*self.pixel_distance_thresh)
            else:
                goal_far_enough = True


    def setup_blob_detection(self):
        params = cv2.SimpleBlobDetector_Params()
        params.minThreshold = 0
        params.maxThreshold = 255
        params.filterByArea = True
        params.minArea = 20
        params.filterByCircularity = False
        params.filterByConvexity = False
        params.filterByInertia = False
        self.blob_detector = cv2.SimpleBlobDetector_create(params)

        self.latest_obj_pixel_coords = None


    def reset(self):

        self._env_step_counter = 0
        self.reset_counter += 1

        # avoid double reset
        if self.reset_counter % 2 == 0:

            # raise arm to avoid moving directly to workframe pos potentially hitting objects
            if not self.first_run and self.reset_to_origin:
                self._UR5.raise_tip(dist=50)

            # reset the ur5 arm
            if self.first_run:
                self._UR5.reset(reset_to_origin=True)
            else:
                self._UR5.reset(reset_to_origin=self.reset_to_origin)

            # reset the goal
            self.setup_goal()

        # get the starting observation
        self._observation = self.get_observation()

        # use to avoid doing things on first call to reset
        self.first_run = False

        return self._observation

    def encode_actions(self, actions):
        """
        return actions as np.array in correct places for sending to ur5
        """

        encoded_actions = np.zeros(6)

        if self.movement_mode == 'xy':
            encoded_actions[0] = actions[0]
            encoded_actions[1] = actions[1]
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
        encoded_actions = self.encode_actions(action)
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
        self._observation = self.get_observation()

        return self._observation, reward, done, {}

    def check_pixel_dist(self, dist_thresh=5):
        pixel_distance = np.linalg.norm(
            np.array(self.latest_obj_pixel_coords) - np.array(self.goal_pixel_coords)
        )
        return pixel_distance < dist_thresh

    def termination(self):
        # terminate when max ep len reached
        if self._env_step_counter >= self._max_steps:
            return True

        # terminate when pixel distance below thresh
        if self.latest_obj_pixel_coords is not None:
            if self.check_pixel_dist(self.pixel_distance_thresh):
                return True

        return False

    def reward(self):
        return 0

    def overlay_goal_on_image(self, tactile_image):
        """
        Overlay a circle onto the observation in roughly the position of the goal
        """

        tactile_image = cv2.cvtColor(tactile_image, cv2.COLOR_GRAY2BGR)

        # Draw a circle at the goal
        circle_rad = int(self.rl_image_size[0] / 32)
        # overlay_img = cv2.circle(tactile_image, goal_coordinates, radius=circle_rad, color=(255,255,255), thickness=-1)
        overlay_img = cv2.drawMarker(tactile_image,
                                     self.goal_pixel_coords,
                                     (0, 0, 255),
                                     markerType=cv2.MARKER_CROSS,
                                     markerSize=10,
                                     thickness=1,
                                     line_type=cv2.LINE_AA)
        # overlay_img = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2GRAY)
        # overlay_img = overlay_img[..., np.newaxis]

        return overlay_img

    def track_blob(self, image):
        # detect the object as blob
        inverted_img = cv2.bitwise_not(image[..., np.newaxis])
        self.keypoints = self.blob_detector.detect(inverted_img)

        if self.keypoints != []:
            self.latest_obj_pixel_coords = self.keypoints[0].pt

            if self.save_traj:
                self.pixel_traj.append(self.latest_obj_pixel_coords)
                self.goal_traj.append(self.goal_pixel_coords)


    def get_tactile_obs(self):
        # get image from sensor
        observation = self._UR5.get_observation()

        # process with gan here
        generated_sim_image, processed_real_image = self.GAN.gen_sim_image(observation)

        # track the pose of the object
        self.track_blob(generated_sim_image)

        # add border to image
        generated_sim_image[self.border_mask==1] = self.border_gray[self.border_mask==1]

        # add a channel axis at end
        generated_sim_image = generated_sim_image[..., np.newaxis]

        # plot data
        if not self._render_closed:

            # get image with target in approximate position
            overlay_sim_image = self.overlay_goal_on_image(generated_sim_image)

            # draw detected blob on image
            overlay_sim_image = cv2.drawKeypoints(
                overlay_sim_image,
                self.keypoints,
                None,
                color=(0,255,0),
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
            )

            # resize to 256, 256 for video
            resized_real_image = cv2.resize(processed_real_image,
                                           (256,256),
                                           interpolation=cv2.INTER_NEAREST)

            resized_sim_image = cv2.resize(overlay_sim_image,
                                           (256,256),
                                           interpolation=cv2.INTER_NEAREST)

            # convert bgr to match drawn on gen image
            resized_real_image = cv2.cvtColor(resized_real_image, cv2.COLOR_GRAY2BGR)

            frame = np.hstack([resized_real_image, resized_sim_image])
            cv2.imshow('real_vs_generated', frame)
            if cv2.waitKey(1) & 0xFF == 27:
                cv2.destroyWindow('real_vs_generated')
                self._render_closed = True

            if self.record_video:
                self.video_frames.append(frame)

        return generated_sim_image

    def get_feature_obs(self):
        """
        Get feature to extend current observations.
        """
        return np.array(self.goal_pos_tcp)

    def get_observation(self):
        """
        Returns the observation
        """
        # init obs dict
        observation = {}
        observation['tactile'] = self.get_tactile_obs()
        observation['extended_feature'] = self.get_feature_obs()
        return observation

    def get_act_dim(self):
        if self.movement_mode == 'xy':
            return 2
        else:
            sys.exit('Incorrect movement mode specified: {}'.format(self.movement_mode))
