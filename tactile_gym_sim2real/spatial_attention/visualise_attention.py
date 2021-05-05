import os
import stable_baselines3 as sb3
import torch
from torch.autograd import Variable
import torch.nn as nn
import torchvision.transforms as T
import cv2
import numpy as np
import time
import gym
import matplotlib.pyplot as plt

from stable_baselines3.common.preprocessing import get_action_dim, is_image_space, preprocess_obs
from stable_baselines3.common.torch_layers import NatureCNN

from pybullet_real2sim.supervised_learning.image_generator import DataGenerator
from pybullet_sims.utils.general_utils import str2bool, load_json_obj, save_json_obj, empty_dir
from pybullet_real2sim.image_transforms import convert_image_uint8


cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# define params
batch_size = 4
save_flag = True

# Load a trained RL model
# mode = 'edge_follow'
mode = 'surface_follow'
# mode = 'object_roll'


# edge
if mode == 'edge_follow':
    rl_model_dir = '/home/alex/Documents/pybullet_sims/pybullet_sims/rl_algos/stable_baselines/saved_models/edge_follow/ppo/keep/128_rand_pos/'
    # rl_model_dir = '/home/alex/Documents/pybullet_sims/pybullet_sims/rl_algos/stable_baselines/saved_models/edge_follow/ppo/keep/128_rand_vel/'
    # rl_model_dir = '/home/alex/Documents/pybullet_sims/pybullet_sims/rl_algos/stable_baselines/saved_models/edge_follow/ppo/keep/128_rand_pos_noaug/'
    # rl_model_dir = '/home/alex/Documents/pybullet_sims/pybullet_sims/rl_algos/stable_baselines/saved_models/edge_follow/ppo/keep/128_rand_vel_noaug/'

# surface
if mode == 'surface_follow':
    rl_model_dir = '/home/alex/Documents/pybullet_sims/pybullet_sims/rl_algos/stable_baselines/saved_models/surface_follow_dir/ppo/keep/128_xyzrp_pos/'
    # rl_model_dir = '/home/alex/Documents/pybullet_sims/pybullet_sims/rl_algos/stable_baselines/saved_models/surface_follow_dir/ppo/keep/128_xyzrp_vel/'
    # rl_model_dir = '/home/alex/Documents/pybullet_sims/pybullet_sims/rl_algos/stable_baselines/saved_models/surface_follow_dir/ppo/keep/128_xyzrp_pos_noaug/'
    # rl_model_dir = '/home/alex/Documents/pybullet_sims/pybullet_sims/rl_algos/stable_baselines/saved_models/surface_follow_dir/ppo/keep/128_xyzrp_vel_noaug/'

if mode == 'object_roll':
    # rl_model_dir = '/home/alex/Documents/pybullet_sims/pybullet_sims/rl_algos/stable_baselines/saved_models/object_roll/ppo/keep/tactile_fullrand_pos/'
    rl_model_dir = '/home/alex/Documents/pybullet_sims/pybullet_sims/rl_algos/stable_baselines/saved_models/object_roll/ppo/keep/tactile_fullrand_vel/'


rl_params  = load_json_obj(os.path.join(rl_model_dir, 'rl_params'))
algo_params  = load_json_obj(os.path.join(rl_model_dir, 'algo_params'))
rl_model_path = os.path.join(rl_model_dir, 'trained_models', 'best_model.zip')

if 'ppo' in rl_model_dir:
    rl_model = sb3.PPO.load(rl_model_path)
elif 'sac' in rl_model_dir:
    rl_model = sb3.SAC.load(rl_model_path)
else:
    sys.exit('Incorrect saved model dir specified.')

# turn off augmentation of loaded RL model for more consistency
# even if no augmentation used this shouldnt cause error
rl_model.policy.features_extractor.apply_augmentation = False


env_name   = rl_params['env_name']
image_size = rl_params['image_size']
border     = rl_params['add_border']
# apply_aug  = algo_params['policy_kwargs']['features_extractor_kwargs']['apply_augmentation']

# make rl deterministic
rl_model.policy.eval()
apply_aug  = False

# get strs for loading / saving
image_size_str = str(image_size[0]) + 'x' + str(image_size[1])
border_str = 'border' if border else 'noborder'
aug_str = 'shift' if apply_aug else 'noaug'

# get latent rl
def rl_get_conv(obs):
    """
    forward pass each layer of the convolutional feature extractor used during training

    return outputs of each layer as list
    """

    x = preprocess_obs(obs, rl_model.policy.observation_space, normalize_images=True)
    cnn_out = rl_model.policy.features_extractor.cnn(x)

    layer_list =list( rl_model.policy.features_extractor.cnn.modules() )

    layer_outs = []
    for l in layer_list[1:]:
        x = l(x)
        layer_outs.append(x)

    return layer_outs

# create image generator
if env_name in ['edge_follow']:
    data_dir = os.path.join('../data_collection/sim/data/edge_2d/tap/', border_str, image_size_str, 'rand_val')

elif env_name in ['surface_follow_dir', 'surface_follow_goal']:
    data_dir = os.path.join('../data_collection/sim/data/surface_3d/tap/', border_str, image_size_str, 'rand_val')

elif env_name in ['object_roll']:
    data_dir = os.path.join('../data_collection/sim/data/spherical_probe/tap/', border_str, image_size_str, 'rand_val')

image_dir = os.path.join(data_dir, 'images')
target_file = os.path.join(data_dir, 'targets.csv')

generator = DataGenerator(target_file=target_file,
                          data_dir=image_dir,
                          dim=image_size,
                          bbox=None,
                          stdiz=False,
                          normlz=True if mode == 'object_roll' else False,
                          thresh=False,
                          rshift=None,
                          rzoom=None,
                          brightlims=None,
                          noise_var=None)

loader = torch.utils.data.DataLoader(generator,
                                     batch_size=batch_size,
                                     shuffle=True,
                                     num_workers=1)

sampled_batch = next(iter(loader))

# set up figure
fig, axs = plt.subplots(batch_size, 4)
plt.rcParams.update({'axes.titlesize': 'small'})

for batch_id in range(batch_size):

    # plot the current image
    np_image = sampled_batch['images'][batch_id].numpy().squeeze()
    axs[batch_id,0].imshow(np_image, interpolation='none')
    axs[batch_id,0].set_title('input_image')

    # get the image as a torch tensor
    torch_image = Variable(sampled_batch['images'][batch_id].type(Tensor).unsqueeze(0))

    # repeat for n_frames here
    if rl_params['n_stack'] > 1:
        torch_image = torch.repeat_interleave(torch_image, rl_params['n_stack'], dim=1)

    # get conv outputs
    with torch.no_grad():
        layer_outs = rl_get_conv(torch_image)

        spatial_maps = []
        for l in layer_outs[1::2]:
            spatial_map = l.abs().mean(1, keepdim=True)
            spatial_map = nn.Softmax(2)(spatial_map.view(*spatial_map.size()[:2], -1)).view_as(spatial_map)
            spatial_maps.append(spatial_map.cpu().detach().numpy())

    # visualise the spatial maps
    for i, g in enumerate(spatial_maps):
        g = g.squeeze()
        axs[batch_id,i+1].imshow(g, interpolation='none')
        axs[batch_id,i+1].set_title('layer_{}'.format(i))


if save_flag:
    save_image_name = border_str + '_' + image_size_str + '_' + aug_str + '.png'
    plt.savefig(os.path.join('saved_images',save_image_name))

plt.show()
