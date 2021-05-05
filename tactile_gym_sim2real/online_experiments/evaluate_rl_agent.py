import gym
import os, sys
import time
import numpy as np
import torch
import pandas as pd

import stable_baselines3 as sb3
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecTransposeImage, VecFrameStack
from tactile_gym.rl_algos.stable_baselines.custom.custom_vec_transpose import NoAssertVecTransposeImage

from tactile_gym.utils.general_utils import load_json_obj

IMAGE_FEATURE_EXTRACTORS_NAMES = ['MixObsNatureCNN', 'ImageAugMixObsNatureCNN']

def make_eval_env(EnvClass, gan_model_dir, rl_params, features_extractor_class_name, n_steps=100, show_plot=True):
    """
    Make a single environment with visualisation specified.
    """
    eval_env = EnvClass(env_modes=rl_params['env_modes'],
                        gan_model_dir=gan_model_dir,
                        max_steps=n_steps,
                        image_size=rl_params['image_size'],
                        add_border=rl_params['add_border'],
                        show_plot=show_plot)

    # dummy vec env generally faster than SubprocVecEnv for small networks
    eval_env = DummyVecEnv([lambda:eval_env])

    # stack observations
    eval_env = VecFrameStack(eval_env, n_stack=rl_params['n_stack'])

    # transpose images for pytorch channel first format
    if sb3.common.preprocessing.is_image_space(eval_env.observation_space):
        eval_env = VecTransposeImage(eval_env)

    # custom cnn policy nets that expect image of type float, this will fail the
    # is_image_space function but we still want to apply image transpose for pytorch
    elif features_extractor_class_name in IMAGE_FEATURE_EXTRACTORS_NAMES:
        eval_env = NoAssertVecTransposeImage(eval_env)

    return eval_env

def final_evaluation(EnvClass, rl_model_dir, gan_model_dir, n_eval_episodes, n_steps=100, show_plot=True):

    rl_params  = load_json_obj(os.path.join(rl_model_dir, 'rl_params'))
    ppo_params = load_json_obj(os.path.join(rl_model_dir, 'algo_params'))

    # class name is alread str after being saved as json
    features_extractor_class_name = ppo_params['policy_kwargs']['features_extractor_class']

    # create the evaluation environment
    # have to vectorize/frame stack here
    eval_env =  make_eval_env(EnvClass, gan_model_dir, rl_params, features_extractor_class_name, n_steps, show_plot)

    # load the trained model
    model_path = os.path.join(rl_model_dir, 'trained_models', 'best_model.zip')
    # model_path = os.path.join(rl_model_dir, 'trained_models', 'final_model.zip')
    if 'ppo' in rl_model_dir:
        model = sb3.PPO.load(model_path)
    elif 'sac' in rl_model_dir:
        model = sb3.SAC.load(model_path)
    else:
        sys.exit('Incorrect saved model dir specified.')

    # turn off augmentation of loaded model
    # even if no augmentation used this shouldnt cause error
    model.policy.features_extractor.apply_augmentation = False


    def eval_model(model, env, n_eval_episodes=10, deterministic=True):

        save_movement_data = True
        if save_movement_data:
            column_names = ['step', 'action', 'tcp_pose', 'tcp_vel', 'time']
            target_df = pd.DataFrame(columns=column_names)
            UR5 = env.envs[0]._UR5
            rtde_client = UR5.robot.sync_robot.controller._client
            csv_id = 0

        episode_rewards, episode_lengths = [], []
        for _ in range(n_eval_episodes):
            obs = env.reset()
            done, state = False, None
            episode_reward = 0.0
            episode_length = 0

            # for keeping accurate control rate
            control_hz = 7.0
            control_rate = 1./control_hz
            fps_start_time = time.clock()
            start_time = time.time()

            while not done:
                fps_next_time = fps_start_time + control_rate

                action, state = model.predict(obs, state=state, deterministic=deterministic)
                obs, reward, done, _info = env.step(action)

                print('')
                print('Step:       {}'.format(episode_length))
                print('Act:        {}'.format(action))
                print('Obs:        {}'.format(obs.shape))
                print('Rew:        {}'.format(reward))
                print('Done:       {}'.format(done))

                if save_movement_data:
                    # get pose in workframe
                    robot_pose = UR5.current_TCP_pose
                    robot_pose[5] -= UR5.sensor_offset_ang

                    # get vel in workframe
                    current_vels = rtde_client.get_linear_speed()
                    current_vels[3:] *= 180/np.pi # convert to deg
                    transformed_vels = UR5.baseframe_to_workframe_vels(current_vels)

                    print('TCP Pose:   {}'.format(robot_pose))
                    print('TCP Speed:  {}'.format(transformed_vels))
                    print('Time:       {}'.format(time.time() - start_time))

                    # add to csv
                    target_df.loc[csv_id] = [episode_length, action, robot_pose, transformed_vels, time.time() - start_time]

                    csv_id += 1

                episode_reward += reward
                episode_length += 1

                # enforce accurate control rate
                while time.clock() < fps_next_time:
                    pass

                print("FPS: ", 1.0 / (time.clock() - fps_start_time))
                fps_start_time = fps_next_time

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

        if save_movement_data:
            csv_file = os.path.join('collected_data', 'evaluation_movement_data.csv')
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
