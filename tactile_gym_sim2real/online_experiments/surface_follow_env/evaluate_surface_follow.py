import os

from tactile_gym_sim2real.online_experiments.surface_follow_env.surface_follow_dir_env import SurfaceFollowDirEnv
from tactile_gym_sim2real.online_experiments.evaluate_rl_agent import final_evaluation

# evaluate params
n_eval_episodes = 1
n_steps = 400
show_plot = False

# rl models
# prev best
# rl_model_dir = os.path.join(os.path.dirname(__file__), '../trained_rl_networks/surface_follow_dir/ppo/update/128_yzr/')
# rl_model_dir = os.path.join(os.path.dirname(__file__), '../trained_rl_networks/surface_follow_dir/ppo/update/128_xyzrp/')

# rl_model_dir = os.path.join(os.path.dirname(__file__), '../trained_rl_networks/surface_follow_dir/ppo/keep/128_xyzRxRy_pos/')
rl_model_dir = os.path.join(os.path.dirname(__file__), '../trained_rl_networks/surface_follow_dir/ppo/keep/128_xyzRxRy_vel/')
# rl_model_dir = os.path.join(os.path.dirname(__file__), '../trained_rl_networks/surface_follow_dir/ppo/keep/128_xyzRxRy_vel_actlim_0.25/')

# gan models
# gan_model_dir = os.path.join(os.path.dirname(__file__), '../trained_gans/[surface_3d]/256x256_[tap]_500epochs/')
gan_model_dir = os.path.join(os.path.dirname(__file__), '../trained_gans/[surface_3d]/256x256_[shear]_500epochs/')
# gan_model_dir = os.path.join(os.path.dirname(__file__), '../trained_gans/[edge_2d,surface_3d]/256x256_[shear]_250epochs/')

# run the evaluation
final_evaluation(SurfaceFollowDirEnv, rl_model_dir, gan_model_dir, n_eval_episodes, n_steps, show_plot)
