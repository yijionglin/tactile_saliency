import argparse
import os
import numpy as np
import math
import itertools
import time
import datetime
import sys
from skimage.metrics import structural_similarity

from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

from pybullet_real2sim.common_utils import *
from pybullet_real2sim.image_transforms import *

from pybullet_sims.utils.general_utils import str2bool, load_json_obj, save_json_obj, empty_dir
from pybullet_sims.rl_envs.ur5_envs.tactip_reference_images import *

save_flag = False
take_snapshot = False

# mode = 'edge_2d'
# mode = 'surface_3d'
mode = 'spherical_probe'

# directory where model is stored
if mode == 'edge_2d':
    save_dir_name = 'saved_models/[edge_2d]/256x256_[tap]_500epochs'
elif mode == 'surface_3d':
    save_dir_name = 'saved_models/[surface_3d]/256x256_[tap]_500epochs'
elif mode == 'spherical_probe':
    save_dir_name = 'saved_models/[spherical_probe]/256x256_[tap]_250epochs_thresh'

augmentation_params = load_json_obj(os.path.join(save_dir_name, 'augmentation_params'))

# overide some augmentation params as we dont want them when generating new data
augmentation_params['rshift'] = None
augmentation_params['rzoom'] = None
augmentation_params['brightlims'] = None
augmentation_params['noise_var'] = None

# import the correct GAN models
if list(augmentation_params['dims']) == [256,256]:
    if mode in ['edge_2d', 'surface_3d']:
        from pybullet_real2sim.pix2pix.gan_models.models_256_auxrl import *
    elif mode in ['spherical_probe']:
        from pybullet_real2sim.pix2pix.gan_models.models_256 import *

elif list(augmentation_params['dims']) == [128,128]:
    if mode in ['edge_2d', 'surface_3d']:
        from pybullet_real2sim.pix2pix.gan_models.models_256_auxrl import *
    elif mode in ['spherical_probe']:
        from pybullet_real2sim.pix2pix.gan_models.models_256 import *

elif list(augmentation_params['dims']) == [64,64]:
    if mode in ['edge_2d', 'surface_3d']:
        from pybullet_real2sim.pix2pix.gan_models.models_64_auxrl import *
    elif mode in ['spherical_probe']:
        from pybullet_real2sim.pix2pix.gan_models.models_64 import *
else:
    sys.exit('Incorrect dims specified')

# for selecting simulated data dirs with images already at the specified size
image_size_str = str(augmentation_params['dims'][0]) + 'x' + str(augmentation_params['dims'][1])

# directory where data is stored
real_data_dir = os.path.join('../data_collection/real/data', mode, 'tap', 'csv_val')
real_target_file = os.path.join(real_data_dir, 'targets.csv')
real_image_dir = os.path.join(real_data_dir, 'images')

sim_data_dir  = os.path.join('../data_collection/sim/data', mode, 'tap', 'noborder', image_size_str, 'csv_val')
sim_target_file = os.path.join(sim_data_dir, 'targets.csv')
sim_image_dir = os.path.join(sim_data_dir, 'images')

# load csv file
real_label_df = pd.read_csv(real_target_file)
sim_label_df = pd.read_csv(sim_target_file)

# where to save the data
output_dir = os.path.join(os.path.abspath(os.getcwd()), '../data_collection/generated_data', real_data_dir.split('/')[-2], real_data_dir.split('/')[-1], save_dir_name.split('/')[-1])
output_image_dir = os.path.join(output_dir, 'images')

if save_flag:
    check_dir(output_dir)

    # make dirs
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_image_dir, exist_ok=True)
    real_label_df.to_csv(os.path.join(output_dir, 'targets.csv'), index=False) # save in new dir

# Initialize generator and discriminator
generator = GeneratorUNet(in_channels=1, out_channels=1)

if mode in ['edge_2d', 'surface_3d']:
    discriminator = Discriminator(in_channels=1, act_dim=2)
elif mode in ['spherical_probe']:
    discriminator = Discriminator(in_channels=1)

# configure gpu use
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
if cuda:
    generator = generator.cuda()
    discriminator = discriminator.cuda()

add_border = True
if add_border:
    border_gray_savefile = os.path.join( getBorderImagesPath(), 'standard', image_size_str, 'border_gray.npy')
    border_mask_savefile = os.path.join( getBorderImagesPath(), 'standard', image_size_str, 'border_mask.npy')
    border_gray = np.load(border_gray_savefile)
    border_mask = np.load(border_mask_savefile)

# Load pretrained models
generator.load_state_dict(torch.load(os.path.join(save_dir_name, 'checkpoints/best_generator.pth')))
discriminator.load_state_dict(torch.load(os.path.join(save_dir_name, 'checkpoints/best_discriminator.pth')))

score_list = []

# switch to eval mode
generator.eval()
discriminator.eval()

for index, row in real_label_df.iterrows():

    real_image_filename = os.path.join(real_image_dir, real_label_df.iloc[index]['sensor_image'])
    sim_image_filename = os.path.join(sim_image_dir, sim_label_df.iloc[index]['sensor_image'])
    real_image = cv2.imread(real_image_filename)
    sim_image = cv2.imread(sim_image_filename)

    # preprocess/augment image
    processed_real_image = process_image(real_image, gray=True, bbox=[80,25,530,475], dims=augmentation_params['dims'], stdiz=augmentation_params['stdiz'], normlz=augmentation_params['normlz'],
                                         rshift=augmentation_params['rshift'], rzoom=augmentation_params['rzoom'], thresh=augmentation_params['thresh'],
                                         add_axis=False, brightlims=augmentation_params['brightlims'], noise_var=augmentation_params['noise_var'])

    processed_sim_image = process_image(sim_image, gray=True, bbox=None, dims=augmentation_params['dims'], stdiz=augmentation_params['stdiz'], normlz=augmentation_params['normlz'],
                                         rshift=None, rzoom=None, thresh=False,
                                         add_axis=False, brightlims=None, noise_var=None)

    # put the channel into first axis because pytorch
    processed_real_image = np.rollaxis(processed_real_image, 2, 0)

    # add an axis to make a batches
    processed_real_image = processed_real_image[np.newaxis, ...]

    # convert to torch tensor
    processed_real_image = Variable(torch.from_numpy(processed_real_image).type(Tensor))

    # generate an image
    gen_sim_image = generator(processed_real_image)

    # convert to numpy, image format
    processed_real_image = (processed_real_image[0,...].detach().cpu().numpy()*255).astype(np.uint8)
    processed_sim_image = (processed_sim_image*255).astype(np.uint8)
    gen_sim_image  = (np.clip(gen_sim_image[0,...].detach().cpu().numpy(), 0, 1)*255).astype(np.uint8)


    # revert back to channel last for displaying/saving
    processed_real_image = np.swapaxes(np.swapaxes(processed_real_image, 0, 2), 0, 1)
    gen_sim_image = np.swapaxes(np.swapaxes(gen_sim_image, 0, 2), 0, 1)

    if add_border:
        gen_sim_image[border_mask==1,:] = border_gray[border_mask==1][...,np.newaxis]
        processed_sim_image[border_mask==1,:] = border_gray[border_mask==1][...,np.newaxis]

    # check the difference between the generated and actual sim image
    (score, diff_image) = structural_similarity(processed_sim_image.squeeze(), gen_sim_image.squeeze(), full=True)
    diff_image = (diff_image * 255).astype(np.uint8)
    diff_image = diff_image[..., np.newaxis]

    # absolute difference image
    # diff_image = np.abs(processed_sim_image.squeeze().astype(np.float32) - gen_sim_image.squeeze().astype(np.float32))
    # diff_image = (diff_image).astype(np.uint8)
    # diff_image = diff_image[..., np.newaxis]

    # save generated image
    image_outfile = os.path.join(output_image_dir, real_label_df.iloc[index]['sensor_image'])

    if save_flag:
        cv2.imwrite(image_outfile, gen_sim_image)

    # concatenate for displayinf in one window
    x0, y0, x1, y1 = [80,25,530,475]
    unprocessed_real_image = real_image[y0:y1, x0:x1]
    unprocessed_real_image = cv2.cvtColor(cv2.resize(unprocessed_real_image, tuple(augmentation_params['dims']), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)[...,np.newaxis]
    pad = np.ones(shape=(256,5,1), dtype=np.uint8)*255
    disp_image = np.concatenate([pad, unprocessed_real_image, pad,
                                 pad, processed_real_image, pad,
                                 pad, gen_sim_image, pad,
                                 pad, processed_sim_image, pad,
                                 pad, diff_image, pad], axis=1)

    # record scores to array for stats after
    score_list.append(score)
    print('ID: {}, Diff_Score: {}'.format(index, score))


    # show image
    cv2.imshow("images", disp_image)

    k = cv2.waitKey(5)
    if k==27:    # Esc key to stop
        break

    if take_snapshot:
        # edge_id    = best = 1736, worst = 778
        # surface_id = best = 365, worst = 473
        # probe_id   : best = 262, worst = 153

        if index == 153:
            static_image_outfile = "example_videos/image_comparison.png"
            cv2.imwrite(static_image_outfile, disp_image)
            break

# display stats on image difference scroes
score_array = np.asarray(score_list)
print('Mean Score: ', np.mean(score_array))
print('Min Score:  ', np.min(score_array), np.argmin(score_array))
print('Max Score:  ', np.max(score_array), np.argmax(score_array))
