import os

from tactile_gym_sim2real.online_experiments.object_roll_env.object_roll_env import ObjectRollEnv
from tactile_gym_sim2real.online_experiments.evaluate_rl_agent import final_evaluation

# evaluate params
n_eval_episodes = 5
n_steps = 100
show_plot = False

# rl models
# rl_model_dir = os.path.join(os.path.dirname(__file__), '../trained_rl_networks/object_roll/ppo/keep/tactile_fullrand_pos/')
# rl_model_dir = os.path.join(os.path.dirname(__file__), '../trained_rl_networks/object_roll/ppo/keep/tactile_fullrand_vel/')
rl_model_dir = os.path.join(os.path.dirname(__file__), '../trained_rl_networks/object_roll/ppo/keep/tactile_fullrand_vel_actlim_0.25/')

# gan models
gan_model_dir = os.path.join(os.path.dirname(__file__), '../trained_gans/[spherical_probe]/256x256_[tap]_250epochs_thresh/')


# run the evaluation
final_evaluation(ObjectRollEnv, rl_model_dir, gan_model_dir, n_eval_episodes, n_steps, show_plot)
