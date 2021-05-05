import random
import time
import os
from tactile_gym_sim2real.online_experiments.ur5_test_env.ur5_test_env import UR5TestEnv
import pandas as pd
import numpy as np

import cri


def main():

    control_hz = 10.0
    control_rate = 1./control_hz
    change_every = 20
    max_steps = 50*change_every
    gan_model_dir = os.path.join(os.path.dirname(__file__), '../trained_gans/[surface_3d]/256x256_[shear]_500epochs/')
    env_modes = {"noise_mode": None,
                 "movement_mode": "xyzRxRyRz",
                 "observation_mode": "full_image",
                 "reward_mode": "dense"}



    env = UR5TestEnv(env_modes=env_modes,
                     gan_model_dir=gan_model_dir,
                     max_steps=max_steps,
                     image_size=[128,128],
                     add_border=True,
                     show_plot=False)

    with env:

        # set seeding (still not perfectly deterministic)
        seed = 1
        random.seed(seed)
        env.seed(seed)
        env.action_space.np_random.seed(seed)

        # setup for transforms
        rtde_client = env._UR5.robot.sync_robot.controller._client
        workframe_mat = cri.transforms.euler2mat(env._UR5.work_frame)
        inv_workframe_mat = np.linalg.inv(workframe_mat)

        # setup for data collection
        column_names = ['step', 'action', 'tcp_pose', 'tcp_vel', 'time']
        target_df = pd.DataFrame(columns=column_names)

        # get observation features
        obs_dim = env.observation_space.shape
        act_dim = env.action_space.shape[0]
        act_low  = env.action_space.low[0]
        act_high = env.action_space.high[0]

        o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

        start_time = time.time()
        step = 0
        a = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        stop     = [0.0,   0.0,  0.0,  0.0,  0.0,  0.0]
        plus_x   = [0.01,  0.0,  0.0,  0.0,  0.0,  0.0]
        minus_x  = [-0.01, 0.0,  0.0,  0.0,  0.0,  0.0]
        plus_y   = [0.0,   0.01, 0.0,  0.0,  0.0,  0.0]
        minus_y  = [0.0,  -0.01, 0.0,  0.0,  0.0,  0.0]
        plus_z   = [0.0,   0.0,  0.01, 0.0,  0.0,  0.0]
        minus_z  = [0.0,   0.0, -0.01, 0.0,  0.0,  0.0]
        plus_Rx  = [0.0,   0.0,  0.0,  0.01, 0.0,  0.0]
        minus_Rx = [0.0,   0.0,  0.0, -0.01, 0.0,  0.0]
        plus_Ry  = [0.0,   0.0,  0.0,  0.0,  0.01, 0.0]
        minus_Ry = [0.0,   0.0,  0.0,  0.0, -0.01, 0.0]
        plus_Rz  = [0.0,   0.0,  0.0,  0.0,  0.0,  0.01]
        minus_Rz = [0.0,   0.0,  0.0,  0.0,  0.0, -0.01]

        fps_start_time = time.clock()
        while (not d):

            fps_next_time = fps_start_time + control_rate

            # +x
            if step == 0*change_every:
                a = plus_x
            # -x
            elif step == 1*change_every:
                a = minus_x
            # -x
            elif step == 2*change_every:
                a = minus_x
            # +x
            elif step == 3*change_every:
                a = plus_x
            # stop
            elif step == 4*change_every:
                a = stop

            # +y
            elif step == 5*change_every:
                a = plus_y
            # -y
            elif step == 6*change_every:
                a = minus_y
            # -y
            elif step == 7*change_every:
                a = minus_y
            # +y
            elif step == 8*change_every:
                a = plus_y
            # stop
            elif step == 9*change_every:
                a = stop

            # +z
            elif step == 10*change_every:
                a = plus_z
            # -z
            elif step == 11*change_every:
                a = minus_z
            # -z
            elif step == 12*change_every:
                a = minus_z
            # +z
            elif step == 13*change_every:
                a = plus_z
            # stop
            elif step == 14*change_every:
                a = stop

            # +Rx
            elif step == 15*change_every:
                a = plus_Rx
            # -Rx
            elif step == 16*change_every:
                a = minus_Rx
            # -Rx
            elif step == 17*change_every:
                a = minus_Rx
            # +Rx
            elif step == 18*change_every:
                a = plus_Rx
            # stop
            elif step == 19*change_every:
                a = stop

            # +Ry
            elif step == 20*change_every:
                a = plus_Ry
            # -Ry
            elif step == 21*change_every:
                a = minus_Ry
            # -Ry
            elif step == 22*change_every:
                a = minus_Ry
            # +Ry
            elif step == 23*change_every:
                a = plus_Ry
            # stop
            elif step == 24*change_every:
                a = stop

            # +Rz
            elif step == 25*change_every:
                a = plus_Rz
            # -Rz
            elif step == 26*change_every:
                a = minus_Rz
            # -Rz
            elif step == 27*change_every:
                a = minus_Rz
            # +Rz
            elif step == 28*change_every:
                a = plus_Rz
            # stop
            elif step == 29*change_every:
                a = stop

            # move in square
            elif step == 30*change_every:
                a = plus_x
            # -Rz
            elif step == 31*change_every:
                a = plus_y
            # -Rz
            elif step == 32*change_every:
                a = minus_x
            # +Rz
            elif step == 33*change_every:
                a = minus_y
            # stop
            elif step == 34*change_every:
                a = stop

            # rotate in square
            elif step == 35*change_every:
                a = plus_Rx
            # -Rz
            elif step == 36*change_every:
                a = plus_Ry
            # -Rz
            elif step == 37*change_every:
                a = minus_Rx
            # +Rz
            elif step == 38*change_every:
                a = minus_Ry
            # stop
            elif step == 39*change_every:
                a = stop

            elif step >= 40*change_every:
                a = env.action_space.sample()

            # ============== Collect Data =================
            # TODO:
            # check rpy pose is being converted correctly
            # check vels are in deg/s

            # get pose in workframe
            robot_pose = env._UR5.robot.pose
            robot_pose[5] -= env._UR5.sensor_offset_ang

            # convert vel from baseframe to workframe
            current_vels = rtde_client.get_linear_speed()
            current_vels[3:] *= 180/np.pi # convert to deg
            transformed_vels = env._UR5.baseframe_to_workframe_vels(current_vels)
            # transformed_vels[3:] *= 180/np.pi # convert to deg

            print('')
            print('Step:       {}'.format(step))
            print('Action:     {}'.format(a))
            print('TCP Pose:   {}'.format(robot_pose))
            print('TCP Speed:  {}'.format(transformed_vels))
            print('Time:       {}'.format(time.time() - start_time))

            target_df.loc[step] = [step, a, robot_pose, transformed_vels, time.time() - start_time]

            # step the environment
            o, r, d, info = env.step(a)

            # increment step
            step+=1

            # enforce accurate control rate
            while time.clock() < fps_next_time:
                pass
            print("FPS: ", 1.0 / (time.clock() - fps_start_time))
            fps_start_time = fps_next_time

        print(target_df)
        csv_file = 'collected_data/tempname.csv'
        target_df.to_csv(csv_file)

if __name__=="__main__":
    main()
