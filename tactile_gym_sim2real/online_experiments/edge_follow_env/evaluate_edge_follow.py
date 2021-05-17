import os

from tactile_gym_sim2real.online_experiments.edge_follow_env.edge_follow_env import EdgeFollowEnv
from tactile_gym_sim2real.online_experiments.evaluate_rl_agent import final_evaluation

# evaluate params
n_eval_episodes = 1
n_steps = 1000
show_plot = True
save_data = True

# rl models
rl_model_dir = os.path.join(
    os.path.dirname(__file__),
    "../trained_rl_networks",
    'edge_follow-v0',
    'rad_ppo',
    'tactile'
)

# select which gan
dataset = 'edge_2d'

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

# run the evaluation
final_evaluation(
    EdgeFollowEnv,
    rl_model_dir,
    gan_model_dir,
    GeneratorUNet,
    n_eval_episodes,
    n_steps,
    show_plot,
    save_data
)
