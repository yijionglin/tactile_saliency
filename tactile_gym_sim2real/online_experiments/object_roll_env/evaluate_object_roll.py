import os

from tactile_gym_sim2real.online_experiments.object_roll_env.object_roll_env import ObjectRollEnv
from tactile_gym_sim2real.online_experiments.evaluate_rl_agent import final_evaluation

# evaluate params
n_eval_episodes = 5
n_steps = 100
show_plot = False

# rl models
rl_model_dir = os.path.join(os.path.dirname(__file__), '../trained_rl_networks/object_roll/rad_ppo/tactile/')

# gan models
# image_size_str = '64x64'
# image_size_str = '128x128'
image_size_str = '256x256'

# data_type = 'tap'
data_type = 'shear'

# gan models
gan_model_dir = os.path.join(
    os.path.dirname(__file__),
    'trained_gans/[spherical_probe]/' + image_size_str + '_[' + data_type + ']_250epochs/'
)

# run the evaluation
final_evaluation(ObjectRollEnv, rl_model_dir, gan_model_dir, n_eval_episodes, n_steps, show_plot)
