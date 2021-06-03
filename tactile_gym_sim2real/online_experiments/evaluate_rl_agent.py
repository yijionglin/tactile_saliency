import gym
import os, sys
import time
import numpy as np
import torch
import pandas as pd

import stable_baselines3 as sb3
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecFrameStack

from tactile_gym.utils.general_utils import load_json_obj


def make_eval_env(
        EnvClass,
        gan_model_dir,
        GanGenerator,
        rl_params,
        n_steps=100,
        show_plot=True
    ):
    """
    Make a single environment with visualisation specified.
    """
    eval_env = EnvClass(env_modes=rl_params['env_modes'],
                        gan_model_dir=gan_model_dir,
                        GanGenerator=GanGenerator,
                        max_steps=n_steps,
                        rl_image_size=rl_params['image_size'],
                        show_plot=show_plot)

    # dummy vec env generally faster than SubprocVecEnv for small networks
    eval_env = DummyVecEnv([lambda:eval_env])

    # stack observations
    eval_env = VecFrameStack(eval_env, n_stack=rl_params['n_stack'])

    # transpose images for pytorch channel first format
    eval_env = VecTransposeImage(eval_env)

    return eval_env

def final_evaluation(
        EnvClass,
        rl_model_dir,
        gan_model_dir,
        GanGenerator,
        n_eval_episodes,
        n_steps=100,
        show_plot=True,
        save_data=False,
        ):

    rl_params  = load_json_obj(os.path.join(rl_model_dir, 'rl_params'))
    ppo_params = load_json_obj(os.path.join(rl_model_dir, 'algo_params'))

    # create the evaluation environment
    # have to vectorize/frame stack here
    eval_env =  make_eval_env(
        EnvClass,
        gan_model_dir,
        GanGenerator,
        rl_params,
        n_steps,
        show_plot
    )

    # load the trained model
    model_path = os.path.join(rl_model_dir, 'trained_models', 'best_model.zip')
    # model_path = os.path.join(rl_model_dir, 'trained_models', 'final_model.zip')

    if 'ppo' in rl_model_dir:
        model = sb3.PPO.load(model_path)
    elif 'sac' in rl_model_dir:
        model = sb3.SAC.load(model_path)
    else:
        sys.exit('Incorrect saved model dir specified.')

    def eval_model(model, env, n_eval_episodes=10, deterministic=True):

        if save_data:
            column_names = ['step', 'action', 'tcp_pose', 'time']
            target_df = pd.DataFrame(columns=column_names)
            UR5 = env.envs[0]._UR5
            csv_id = 0

        episode_rewards, episode_lengths = [], []
        for _ in range(n_eval_episodes):
            obs = env.reset()
            done, state = False, None
            episode_reward = 0.0
            episode_length = 0

            # for keeping accurate control rate
            control_hz = 10.0
            control_rate = 1./control_hz
            fps_start_time = time.perf_counter()
            start_time = time.time()

            while not done:
                fps_next_time = fps_start_time + control_rate

                action, state = model.predict(obs, state=state, deterministic=deterministic)
                obs, reward, done, _info = env.step(action)

                print('')
                print('Step:       {}'.format(episode_length))
                print('Act:        {}'.format(action))
                print("Obs:  ")
                for key, value in obs.items():
                    if value is None:
                        print('  ', key, ':', value)
                    else:
                        print('  ', key, ':', value.shape)
                print('Rew:        {}'.format(reward))
                print('Done:       {}'.format(done))

                if save_data:
                    # get pose in workframe
                    robot_pose = UR5.current_TCP_pose
                    robot_pose[5] -= UR5.sensor_offset_ang

                    # get vel in workframe (takes too long)
                    # current_vels = rtde_client.get_linear_speed()
                    # current_vels[3:] *= 180/np.pi # convert to deg
                    # transformed_vels = UR5.baseframe_to_workframe_vels(current_vels)

                    print('TCP Pose:   {}'.format(robot_pose))
                    # print('TCP Speed:  {}'.format(transformed_vels))
                    print('Time:       {}'.format(time.time() - start_time))

                    # add to csv
                    # target_df.loc[csv_id] = [episode_length, action, robot_pose, transformed_vels, time.time() - start_time]
                    target_df.loc[csv_id] = [episode_length, action, robot_pose, time.time() - start_time]

                    csv_id += 1

                episode_reward += reward
                episode_length += 1

                # enforce accurate control rate
                while time.perf_counter() < fps_next_time:
                    pass

                print("FPS: ", 1.0 / (time.perf_counter() - fps_start_time))
                fps_start_time = fps_next_time

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

        if save_data:
            csv_file = os.path.join('collected_data', 'eval_data.csv')
            target_df.to_csv(csv_file)

        return episode_rewards, episode_lengths


    # evaluate the trained agent
    try:
        init_time = time.time()
        episode_rewards, episode_lengths = eval_model(model, eval_env,
                                                      n_eval_episodes=n_eval_episodes,
                                                      deterministic=True)

        print('Avg Ep Rew: {}, Avg Ep Len: {}'.format(np.mean(episode_rewards), np.mean(episode_lengths)))
        print('Time Taken: {}'.format(time.time()-init_time))

    finally:
        eval_env.close()

if __name__ == '__main__':
    pass
