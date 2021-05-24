import random
import time
import os
from tactile_gym_sim2real.online_experiments.object_push_env.object_push_env import ObjectPushEnv

def main():

    num_iter = 1
    max_steps = 150
    env_modes = {
                 # "movement_mode": "y",
                 # "movement_mode": "yRz",
                 # "movement_mode": "xyRz",
                 # "movement_mode": "TyRz",
                 "movement_mode": "TxTyRz",
                 "control_mode": "TCP_velocity_control",
                 # "control_mode": "TCP_position_control",

                 # used in training, not real robot
                 'rand_init_orn':False,
                 # 'traj_type':'simplex',
                 'traj_type':'straight',

                 "observation_mode": "tactile_image_and_feature",
                 "reward_mode": "dense"}

    # select which gan
    dataset = 'surface_3d'

    # data_type = 'tap'
    data_type = 'shear'

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


    env = ObjectPushEnv(
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
                # a = [0.0, 0.0, -0.25]

                # if step < 50:
                #     a = [0.25, 0.0, 0.0]
                # elif step < 100:
                #     a = [0.0, 0.0, -0.25]
                # else:
                #     a = [0.0, 0.25, 0.0]

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
