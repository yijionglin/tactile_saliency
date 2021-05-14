import random
import time
import os
from tactile_gym_sim2real.online_experiments.surface_follow_env.surface_follow_dir_env import SurfaceFollowDirEnv

def main():

    num_iter = 1
    max_steps = 100
    env_modes = {
                 # "movement_mode": "yzRx",
                 "movement_mode": "xyzRxRy",

                 "control_mode": "TCP_velocity_control",

                 "noise_mode": "simplex",
                 "observation_mode": "full_image",
                 "reward_mode": "dense"}

    # select which gan
    dataset = 'surface_3d'

    data_type = 'tap'
    # data_type = 'shear'

    # gan_image_size = [64, 64]
    gan_image_size = [128, 128]
    # gan_image_size = [256, 256]
    gan_image_size_str = str(gan_image_size[0]) + 'x' + str(gan_image_size[1])

    # get the dir for the saved gan
    gan_model_dir = os.path.join(
        os.path.dirname(__file__),
        '../trained_gans/[' + dataset + ']/' + gan_image_size_str + '_[' + data_type + ']_250epochs/'
    )

    # import the correct sized generator
    if gan_image_size == [64,64]:
        from tactile_gym_sim2real.pix2pix.gan_models.models_64 import GeneratorUNet
    if gan_image_size == [128,128]:
        from tactile_gym_sim2real.pix2pix.gan_models.models_128 import GeneratorUNet
    if gan_image_size == [256,256]:
        from tactile_gym_sim2real.pix2pix.gan_models.models_256 import GeneratorUNet

    env = SurfaceFollowDirEnv(
        env_modes=env_modes,
        gan_model_dir=gan_model_dir,
        GanGenerator=GeneratorUNet,
        max_steps=max_steps,
        rl_image_size=[128,128],
        show_plot=True
    )

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

                t = time.time()

                # step the environment
                o, r, d, info = env.step(a)

                print('')
                print('Step:   ', step)
                print('Act:  ', a)
                print("Obs:  ")
                for key, value in o.items():
                    if value is None:
                        print('  ', key, ':', value)
                    else:
                        print('  ', key, ':', value.shape)
                print('Rew:  ', r)
                print('Done: ', d)

                r_sum += r
                step+=1

            print('Total Reward: ', r_sum)

if __name__=="__main__":
    main()
