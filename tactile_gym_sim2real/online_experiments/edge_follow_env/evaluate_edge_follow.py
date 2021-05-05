import os

from tactile_gym_sim2real.online_experiments.edge_follow_env.edge_follow_env import EdgeFollowEnv
from tactile_gym_sim2real.online_experiments.evaluate_rl_agent import final_evaluation

# evaluate params
n_eval_episodes = 1
n_steps = 600
show_plot = False

# rl models
# prev best
# rl_model_dir = os.path.join(os.path.dirname(__file__), '../trained_rl_networks/edge_follow/ppo/update/128_xy_rand/')

# updated w velocity control
# rl_model_dir = os.path.join(os.path.dirname(__file__), '../trained_rl_networks/edge_follow/ppo/keep/128_rand_pos/')
# rl_model_dir = os.path.join(os.path.dirname(__file__), '../trained_rl_networks/edge_follow/ppo/keep/128_rand_vel/')
rl_model_dir = os.path.join(os.path.dirname(__file__), '../trained_rl_networks/edge_follow/ppo/keep/128_rand_vel_actlim_0.25/')

# gan models
# gan_model_dir = os.path.join(os.path.dirname(__file__), '../trained_gans/[edge_2d]/outdated/shear/256x256_[shear]_500epochs/')
# gan_model_dir = os.path.join(os.path.dirname(__file__), '../trained_gans/[edge_2d]/256x256_[tap]_500epochs/')
gan_model_dir = os.path.join(os.path.dirname(__file__), '../trained_gans/[edge_2d]/256x256_[shear]_500epochs/')
# gan_model_dir = os.path.join(os.path.dirname(__file__), '../trained_gans/[edge_2d,surface_3d]/256x256_[shear]_250epochs/')

# run the evaluation
final_evaluation(EdgeFollowEnv, rl_model_dir, gan_model_dir, n_eval_episodes, n_steps, show_plot)
