import os

from tactile_gym_sim2real.online_experiments.object_push_env.object_push_env import ObjectPushEnv
from tactile_gym_sim2real.online_experiments.evaluate_rl_agent import final_evaluation

# evaluate params
n_eval_episodes = 2
n_steps = 175
show_plot = False

# rl models
# rl_model_dir = os.path.join(os.path.dirname(__file__), '../trained_rl_networks/object_push/ppo/keep/aug/tactile_TyRz_straight_norand/')
# rl_model_dir = os.path.join(os.path.dirname(__file__), '../trained_rl_networks/object_push/ppo/keep/aug/tactile_TyRz_simplex_norand/')
rl_model_dir = os.path.join(os.path.dirname(__file__), '../trained_rl_networks/object_push/ppo/keep/aug/tactile_yRz_straight_norand/')
# rl_model_dir = os.path.join(os.path.dirname(__file__), '../trained_rl_networks/object_push/ppo/keep/aug/tactile_yRz_simplex_norand/')

# gan models
gan_model_dir = os.path.join(os.path.dirname(__file__), '../trained_gans/[surface_3d]/256x256_[shear]_500epochs/')


# run the evaluation
final_evaluation(ObjectPushEnv, rl_model_dir, gan_model_dir, n_eval_episodes, n_steps, show_plot)
