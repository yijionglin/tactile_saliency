import random
import time
import os
from tactile_gym_sim2real.online_experiments.object_roll_env.object_roll_env import ObjectRollEnv

def main():

    num_iter = 4
    max_steps = 10
    gan_model_dir = os.path.join(os.path.dirname(__file__), '../trained_gans/[spherical_probe]/256x256_[tap]_250epochs_thresh/')

    env_modes = {"movement_mode": "xy",
                 "control_mode": "TCP_velocity_control",
                 # "control_mode": "TCP_position_control",

                 # used in training, not real robot
                 "rand_init_obj_pos": False,
                 "rand_obj_size": False,
                 "rand_embed_dist": False,

                 "observation_mode": "tactile_image_and_feature",
                 "reward_mode": "dense"}

    env = ObjectRollEnv(env_modes=env_modes,
                        gan_model_dir=gan_model_dir,
                        max_steps=max_steps,
                        image_size=[128,128],
                        add_border=True,
                        show_plot=True)
    with env:
        # set seeding (still not perfectly deterministic)
        seed = 1
        random.seed(seed)
        env.seed(seed)
        env.action_space.np_random.seed(seed)

        # get observation features
        obs_dim = env.observation_space.shape
        act_dim = env.action_space.shape[0]
        act_low  = env.action_space.low[0]
        act_high = env.action_space.high[0]

        for i in range(num_iter):
            r_sum = 0
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

            step = 0
            while (not d):

                a = []

                a = env.action_space.sample()
                # a = [0.01, 0.0]

                t = time.time()

                # step the environment
                o, r, d, info = env.step(a)

                print('')
                print('Step:   ', step)
                print('Act:  ', a)
                print('Obs:  ', o.shape)
                print('Rew:  ', r)
                print('Done: ', d)

                r_sum += r
                step+=1

            print('Total Reward: ', r_sum)


if __name__=="__main__":
    main()
