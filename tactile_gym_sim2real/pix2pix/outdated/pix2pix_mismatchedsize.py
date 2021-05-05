import argparse
import os
import numpy as np
import math
import itertools
import time
import datetime
import sys

from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

# from tactile_gym_sim2real.pix2pix.models.models_64 import *
# from tactile_gym_sim2real.pix2pix.models.models_128 import *
# from tactile_gym_sim2real.pix2pix.models.models_64_specnorm import *
# from tactile_gym_sim2real.pix2pix.models.models_128_specnorm import *
# from tactile_gym_sim2real.pix2pix.models.models_128_altactivation import *
# from tactile_gym_sim2real.pix2pix.models.models_256 import *
from tactile_gym_sim2real.pix2pix.models.models_256to64_specnorm import *

from tactile_gym_sim2real.pix2pix.image_generator import DataGenerator
from tactile_gym_sim2real.common_utils import *

from tactile_gym.utils.general_utils import str2bool, save_json_obj, empty_dir

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--shuffle", type=str2bool, default=True, help="shuffle the generated image data")
parser.add_argument("--sample_interval", type=int, default=500, help="interval between sampling of images from generators")
parser.add_argument("--checkpoint_interval", type=int, default=10, help="interval between model checkpoints")
opt = parser.parse_args()

# Parameters
augmentation_params = {
          'dims':        (256,256),
          'rshift':      (0.02, 0.02),
          'rzoom':       (0.98, 1),
          'thresh':      True,
          'brightlims':  None,  #[0.3,1.0,-50,50], # alpha limits for contrast, beta limits for brightness
          'noise_var':   None,  # 0.001,
          'stdiz':       False,
          'normlz':      True
          }

# data dir real -> sim
training_real_data_dir   = '../data_collection/real/data/edge2dTap/square_360_-6_6_train'
validation_real_data_dir = '../data_collection/real/data/edge2dTap/square_360_-6_6_val'
training_sim_data_dir   = '../data_collection/sim/data/edge2dTap/rigid/256x256/square_360_-6_6_train'
validation_sim_data_dir = '../data_collection/sim/data/edge2dTap/rigid/256x256/square_360_-6_6_val'

save_dir_name = os.path.join('saved_models', 'trial_' + time.strftime('%m%d%H%M'))
image_dir = os.path.join(save_dir_name, 'images')
checkpoint_dir = os.path.join(save_dir_name, 'checkpoints')

# check save dir exists
if os.path.isdir(save_dir_name):
    input_str = input('Save directories already exists, would you like to continue (y,n)? ')
    if not str2bool(input_str):
        exit()
    else:
        empty_dir(save_dir_name) # clear out existing files)

# make the dirs
os.makedirs(image_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

# save params
save_json_obj(augmentation_params, os.path.join(save_dir_name, 'augmentation_params'))
save_json_obj(vars(opt), os.path.join(save_dir_name, 'training_params'))

cuda = True if torch.cuda.is_available() else False

# Loss functions
criterion_GAN = torch.nn.MSELoss()
criterion_pixelwise = torch.nn.L1Loss()

# Loss weight of L1 pixel-wise loss between translated image and real image
lambda_pixel = 100.0
lambda_gan = 1.0

# Calculate output of image discriminator (PatchGAN)
# patch = (1, augmentation_params['dims'][0] // 2 ** 4, augmentation_params['dims'][1] // 2 ** 4)
patch = (1, 64 // 2 ** 4, 64 // 2 ** 4)

# Initialize generator and discriminator
generator = GeneratorUNet(in_channels=opt.channels, out_channels=opt.channels)
discriminator = Discriminator(in_channels=opt.channels)

if cuda:
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    criterion_GAN.cuda()
    criterion_pixelwise.cuda()

if opt.epoch != 0:
    # Load pretrained models
    generator.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'generator_{}.pth'.format(opt.epoch))))
    discriminator.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'discriminator_{}.pth'.format(opt.epoch))))
else:
    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# Configure dataloaders
training_generator = DataGenerator(real_data_dir=training_real_data_dir,
                                   sim_data_dir=training_sim_data_dir,
                                   dim=augmentation_params['dims'],
                                   stdiz=augmentation_params['stdiz'],
                                   normlz=augmentation_params['normlz'],
                                   thresh=augmentation_params['thresh'],
                                   rshift=augmentation_params['rshift'],
                                   rzoom=augmentation_params['rzoom'],
                                   brightlims=augmentation_params['brightlims'],
                                   noise_var=augmentation_params['noise_var'])

val_generator = DataGenerator(real_data_dir=validation_real_data_dir,
                              sim_data_dir=validation_sim_data_dir,
                              dim=augmentation_params['dims'],
                              stdiz=augmentation_params['stdiz'],
                              normlz=augmentation_params['normlz'],
                              thresh=augmentation_params['thresh'],
                              rshift=None,
                              rzoom=None,
                              brightlims=None,
                              noise_var=None)

training_loader = torch.utils.data.DataLoader(training_generator,
                                              batch_size=opt.batch_size,
                                              shuffle=opt.shuffle,
                                              num_workers=opt.n_cpu)

val_loader = torch.utils.data.DataLoader(val_generator,
                                         batch_size=opt.batch_size,
                                         shuffle=opt.shuffle,
                                         num_workers=opt.n_cpu)


# Tensor type
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

n_save_images = np.min([opt.batch_size, 8])
def sample_images(batches_done):
    """Saves a generated sample from the validation set"""
    imgs = next(iter(val_loader))
    real_imgs = Variable(imgs["real"].type(Tensor))
    small_real_imgs = Variable(imgs["small_real"].type(Tensor))
    sim_imgs = Variable(imgs["sim"].type(Tensor))
    gen_sim_imgs = generator(real_imgs)
    img_sample = torch.cat((small_real_imgs.data[:n_save_images,:,:,:],
                            gen_sim_imgs.data[:n_save_images,:,:,:],
                            sim_imgs.data[:n_save_images,:,:,:]), -2)
    save_image(img_sample, os.path.join(image_dir, '{}.png'.format(batches_done)), nrow=4, normalize=False)


# ----------
#  Training
# ----------

prev_time = time.time()

for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(training_loader):

        # Model inputs
        tip_images = Variable(batch['real'].type(Tensor))
        small_tip_images = Variable(batch['small_real'].type(Tensor))
        sim_images = Variable(batch['sim'].type(Tensor))

        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((small_tip_images.size(0), *patch))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((small_tip_images.size(0), *patch))), requires_grad=False)

        # ------------------
        #  Train Generators
        # ------------------

        optimizer_G.zero_grad()

        # GAN loss
        gen_sim_images = generator(tip_images)
        pred_gen = discriminator(gen_sim_images, small_tip_images)
        loss_GAN = criterion_GAN(pred_gen, valid)

        # Pixel-wise loss
        loss_pixel = criterion_pixelwise(gen_sim_images, sim_images)

        # Total loss
        loss_G = (lambda_gan*loss_GAN) + (lambda_pixel*loss_pixel)

        loss_G.backward()

        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Real loss
        pred_real = discriminator(sim_images, small_tip_images)
        loss_real = criterion_GAN(pred_real, valid)

        # Fake loss
        pred_fake = discriminator(gen_sim_images.detach(), small_tip_images)
        loss_fake = criterion_GAN(pred_fake, fake)

        # Total loss
        loss_D = 0.5 * (loss_real + loss_fake)

        loss_D.backward()
        optimizer_D.step()

        # --------------
        #  Log Progress
        # --------------

        # Determine approximate time left
        batches_done = epoch * len(training_loader) + i
        batches_left = opt.n_epochs * len(training_loader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # Print log
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, pixel: %f, adv: %f] ETA: %s"
            % (
                epoch,
                opt.n_epochs,
                i,
                len(training_loader),
                loss_D.item(),
                loss_G.item(),
                loss_pixel.item(),
                loss_GAN.item(),
                time_left,
            )
        )

        # If at sample interval save image
        if batches_done % opt.sample_interval == 0:
            sample_images(batches_done)

    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        print('\nSaving Model {}'.format(epoch))
        # Save model checkpoints
        torch.save(generator.state_dict(), os.path.join(checkpoint_dir, 'generator_{}.pth'.format(opt.epoch)))
        torch.save(discriminator.state_dict(), os.path.join(checkpoint_dir, 'discriminator_{}.pth'.format(opt.epoch)))
