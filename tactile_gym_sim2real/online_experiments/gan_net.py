import os
import numpy as np
import time

import torch.nn as nn
import torch.nn.functional as F
import torch

# TODO: do this but better
# from tactile_gym_sim2real.pix2pix.gan_models.models_64_auxrl import *
# from tactile_gym_sim2real.pix2pix.gan_models.models_128_auxrl import *
# from tactile_gym_sim2real.pix2pix.gan_models.models_256_auxrl import *
from tactile_gym_sim2real.pix2pix.gan_models.models_256 import *

from tactile_gym_sim2real.common_utils import *
from tactile_gym_sim2real.image_transforms import *

from tactile_gym.utils.general_utils import load_json_obj

class pix2pix_GAN():

    def __init__(self, gan_model_dir, rl_image_size=[64,64]):

        self.rl_image_size = rl_image_size
        self.params = load_json_obj(os.path.join(gan_model_dir, 'augmentation_params'))

        # overide some augmentation params as we dont want them when generating new data
        self.params['rshift'] = None
        self.params['rzoom'] = None
        self.params['brightlims'] = None
        self.params['noise_var'] = None

        # Initialize generator and discriminator
        generator = GeneratorUNet(in_channels=1, out_channels=1)

        # configure gpu use
        cuda = True if torch.cuda.is_available() else False
        self.Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        if cuda:
            self.generator = generator.cuda()
        else:
            self.generator = generator

        # Load pretrained models
        self.generator.load_state_dict(torch.load(os.path.join(gan_model_dir, 'checkpoints/final_generator.pth')))
        # self.generator.load_state_dict(torch.load(os.path.join(gan_model_dir, 'checkpoints/best_generator.pth')))

        # put in eval mode to disable dropout etc
        self.generator.eval()

    def gen_sim_image(self, real_image):

        # preprocess/augment image
        processed_real_image = process_image(real_image, gray=True, bbox=[80,25,530,475], dims=self.params['dims'],
                                             stdiz=self.params['stdiz'], normlz=self.params['normlz'],
                                             rshift=self.params['rshift'], rzoom=self.params['rzoom'],
                                             thresh=self.params['thresh'], add_axis=False,
                                             brightlims=self.params['brightlims'], noise_var=self.params['noise_var'])
        # setup the processed image for plotting
        processed_real_image_plot = (np.clip(processed_real_image, 0, 1)*255).astype(np.uint8) # convert to image format

        # put the channel into first axis because pytorch
        processed_real_image_pt = np.rollaxis(processed_real_image, 2, 0)

        # add an axis to make a batch
        processed_real_image_pt = processed_real_image_pt[np.newaxis, ...]

        # convert to torch tensor
        processed_real_image_pt = torch.from_numpy(processed_real_image_pt).type(self.Tensor)

        # generate an image
        gen_sim_image = self.generator(processed_real_image_pt)

        # convert to numpy, image format, size expected by rl agent
        gen_sim_image = gen_sim_image[0,0,...].detach().cpu().numpy() # pytorch batch -> numpy image
        gen_sim_image = (np.clip(gen_sim_image, 0, 1)*255).astype(np.uint8) # convert to image format

        if self.params['dims'] != self.rl_image_size:
            gen_sim_image = cv2.resize(gen_sim_image, tuple(self.rl_image_size), interpolation=cv2.INTER_NEAREST) # resize to RL expected

        return gen_sim_image, processed_real_image_plot
