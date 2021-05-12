import argparse
import os
import numpy as np
import itertools
import time
import datetime
import sys
import pandas as pd

import torch
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable

from tactile_gym.utils.general_utils import str2bool, save_json_obj, check_dir
from tactile_gym_sim2real.pix2pix.image_generator import DataGenerator
from tactile_gym_sim2real.pix2pix.plot_tools import plot_dataframe

def main(opt, augmentation_params, weights, task_dirs, data_dirs):

    # for selecting simulated data dirs with images already at the specified size
    image_size_str = str(augmentation_params['dims'][0]) + 'x' + str(augmentation_params['dims'][1])

    # combine the data directories
    combined_dirs = list(itertools.product(task_dirs, data_dirs))
    combined_paths = [os.path.join(*i) for i in combined_dirs]

    # data dir real -> sim
    training_real_data_dirs = [os.path.join('../data_collection/real/data/', data_path, 'csv_train') for data_path in combined_paths]
    validation_real_data_dirs = [os.path.join('../data_collection/real/data/', data_path, 'csv_val') for data_path in combined_paths]
    training_sim_data_dirs = [os.path.join('../data_collection/sim/data/',  data_path, image_size_str, 'csv_train') for data_path in combined_paths]
    validation_sim_data_dirs = [os.path.join('../data_collection/sim/data/',  data_path, image_size_str, 'csv_val') for data_path in combined_paths]

    # Create a save directory
    task_str = "[" + ",".join(task_dirs) + "]"
    dir_str = "[" + ",".join(data_dirs) + "]"
    save_dir_name = os.path.join('saved_models', task_str, image_size_str + "_" + dir_str + '_' + str(opt.n_epochs) + 'epochs' )
    image_dir = os.path.join(save_dir_name, 'images')
    checkpoint_dir = os.path.join(save_dir_name, 'checkpoints')

    # check save dir exists
    check_dir(save_dir_name)

    # make the dirs
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # save params
    save_json_obj(augmentation_params, os.path.join(save_dir_name, 'augmentation_params'))
    save_json_obj(weights, os.path.join(save_dir_name, 'weights'))
    save_json_obj(vars(opt), os.path.join(save_dir_name, 'training_params'))

    # enable gpu
    cuda = True if torch.cuda.is_available() else False

    # Loss functions
    criterion_GAN = torch.nn.MSELoss()
    criterion_pixelwise = torch.nn.L1Loss()

    # Calculate output of image discriminator (PatchGAN)
    patch = (1, augmentation_params['dims'][0] // 2 ** 4, augmentation_params['dims'][1] // 2 ** 4)

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
        generator.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'final_generator.pth')))
        discriminator.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'final_discriminator_{}.pth')))
    else:
        # Initialize weights
        generator.apply(weights_init_normal)
        discriminator.apply(weights_init_normal)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(),     lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    # Configure dataloaders
    training_generator = DataGenerator(real_data_dirs=training_real_data_dirs,
                                       sim_data_dirs=training_sim_data_dirs,
                                       dim=augmentation_params['dims'],
                                       stdiz=augmentation_params['stdiz'],
                                       normlz=augmentation_params['normlz'],
                                       thresh=augmentation_params['thresh'],
                                       rshift=augmentation_params['rshift'],
                                       rzoom=augmentation_params['rzoom'],
                                       brightlims=augmentation_params['brightlims'],
                                       noise_var=augmentation_params['noise_var'],
                                       joint_aug=augmentation_params['joint_aug'])

    val_generator = DataGenerator(real_data_dirs=validation_real_data_dirs,
                                  sim_data_dirs=validation_sim_data_dirs,
                                  dim=augmentation_params['dims'],
                                  stdiz=augmentation_params['stdiz'],
                                  normlz=augmentation_params['normlz'],
                                  thresh=augmentation_params['thresh'],
                                  rshift=None,
                                  rzoom=None,
                                  brightlims=None,
                                  noise_var=None,
                                  joint_aug=False)

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
        sim_imgs = Variable(imgs["sim"].type(Tensor))
        gen_sim_imgs = torch.clamp(generator(real_imgs), 0, 1)
        img_sample = torch.cat((real_imgs.data[:n_save_images,:,:,:],
                                gen_sim_imgs.data[:n_save_images,:,:,:],
                                sim_imgs.data[:n_save_images,:,:,:]), -2)
        save_image(img_sample, os.path.join(image_dir, '{}.png'.format(batches_done)), nrow=4, normalize=False)

    # ----------
    # -------------------------------- Training --------------------------------
    # ----------

    # save an image with no training
    sample_images('no_training')

    # create dataframe for storing tracked data
    loss_df = pd.DataFrame(columns=['Epoch', 'D_Loss', 'Real_Loss', 'Fake_Loss', 'G_Loss', 'GAN_Loss', 'Pixel_Loss'])

    # initialise tracking vars
    running_losses = np.zeros(loss_df.shape[1]-1)
    sample_batch_count = 0
    row_id = 0
    prev_time = time.time()

    for epoch in range(opt.epoch, opt.n_epochs+1):
        for i, batch in enumerate(training_loader):

            # Model inputs
            tip_images = Variable(batch['real'].type(Tensor))
            sim_images = Variable(batch['sim'].type(Tensor))

            # Adversarial ground truths
            valid = Variable(Tensor(np.ones((tip_images.size(0), *patch))), requires_grad=False)
            fake = Variable(Tensor(np.zeros((tip_images.size(0), *patch))), requires_grad=False)

            # ------------------
            #  Train Generators
            # ------------------

            optimizer_G.zero_grad()

            # GAN loss
            gen_sim_images = generator(tip_images)

            pred_gen = discriminator(gen_sim_images, tip_images)

            loss_GAN = criterion_GAN(pred_gen, valid)

            # Pixel-wise loss
            loss_pixel = criterion_pixelwise(gen_sim_images, sim_images)

            # Total loss
            loss_G = (weights['W_gan']*loss_GAN) + (weights['W_pixel']*loss_pixel)

            loss_G.backward()

            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Real loss
            pred_real = discriminator(sim_images, tip_images)
            loss_real = criterion_GAN(pred_real, valid)

            # Fake loss
            pred_fake = discriminator(gen_sim_images.detach(), tip_images)
            loss_fake = criterion_GAN(pred_fake, fake)

            # Total loss
            loss_disc = 0.5 * (loss_real + loss_fake)
            loss_D = loss_disc

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
                "\r[Epoch {}/{}] [Batch {}/{}] [D_loss: {:.5f}, real_loss: {:.5f}, fake_loss: {:.5f}] [G_loss: {:.5f}, GAN_loss: {:.5f}, pix_loss: {:.5f}] ETA: {}".format(
                    epoch,
                    opt.n_epochs,
                    i,
                    len(training_loader),
                    loss_D.item(),
                    loss_real.item(),
                    loss_fake.item(),
                    loss_G.item(),
                    loss_GAN.item(),
                    loss_pixel.item(),
                    time_left,
                )
            )

            running_losses[0] += loss_D.item()
            running_losses[1] += loss_real.item()
            running_losses[2] += loss_fake.item()
            running_losses[3] += loss_G.item()
            running_losses[4] += loss_GAN.item()
            running_losses[5] += loss_pixel.item()
            sample_batch_count += 1

        # If at sample interval save image
        if (epoch % opt.sample_interval == 0) or ((epoch) % opt.n_epochs == 0):

            sample_images('epoch_{}'.format(epoch))

            # average the running losses over the number of batches done
            running_losses = running_losses / sample_batch_count

            # append to df
            loss_df.loc[row_id] = [epoch, *running_losses]

            # print
            print('')
            print('')
            print(loss_df.loc[row_id])

            # check if this has the lowest running pixel loss
            if loss_df.loc[row_id]['Pixel_Loss'] == min(loss_df['Pixel_Loss']):
                best_model_flag = True
            else:
                best_model_flag = False

            # save the dataframe as csv
            loss_df.to_csv(os.path.join(save_dir_name, 'training_losses.csv'))

            # plot the dataframe
            plot_dataframe(loss_df, save_file=os.path.join(save_dir_name, 'training_curves.png'))

            # update tracking vars
            running_losses = np.zeros(loss_df.shape[1]-1)
            sample_batch_count = 0
            row_id += 1

            # Save latest model checkpoints
            print('')
            print('Saving Model {}'.format(epoch))
            print('')
            torch.save(generator.state_dict(), os.path.join(checkpoint_dir, 'final_generator.pth'))
            torch.save(discriminator.state_dict(), os.path.join(checkpoint_dir, 'final_discriminator.pth'))

            # Save best model checkpoints
            if best_model_flag:
                print('Saving Best Model {}'.format(epoch))
                print('')
                torch.save(generator.state_dict(), os.path.join(checkpoint_dir, 'best_generator.pth'))
                torch.save(discriminator.state_dict(), os.path.join(checkpoint_dir, 'best_discriminator.pth'))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
    parser.add_argument("--n_epochs", type=int, default=250, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
    parser.add_argument("--n_cpu", type=int, default=12, help="number of cpu threads to use during batch generation")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--shuffle", type=str2bool, default=True, help="shuffle the generated image data")
    parser.add_argument("--sample_interval", type=int, default=5, help="interval between sampling of images from generators")
    opt = parser.parse_args()

    # Parameters
    augmentation_params = {
              'dims':        (256, 256),
              'rshift':      (0.025, 0.025),
              'rzoom':       None,  # (0.98, 1),
              'thresh':      True,
              'brightlims':  None,  # [0.3, 1.0, -50, 50], # alpha limits for contrast, beta limits for brightness
              'noise_var':   None,  # 0.001,
              'stdiz':       False,
              'normlz':      True,
              'joint_aug':   False
              }

    # weighting for loss functions
    weights = {
        'W_gan': 1.0,
        'W_pixel': 100.0,
    }

    # data collected for task
    # task_dirs = ['edge_2d']
    # task_dirs = ['surface_3d']
    # task_dirs = ['spherical_probe']
    # task_dirs = ['edge_2d', 'surface_3d']

    # for GAN data
    data_dirs = ['tap']
    # data_dirs = ['shear']
    # data_dirs = ['tap', 'shear']

    # import the correct GAN models
    if list(augmentation_params['dims']) == [256, 256]:
        from tactile_gym_sim2real.pix2pix.gan_models.models_256 import GeneratorUNet, Discriminator, weights_init_normal
    elif list(augmentation_params['dims']) == [128, 128]:
        from tactile_gym_sim2real.pix2pix.gan_models.models_128 import GeneratorUNet, Discriminator, weights_init_normal
    elif list(augmentation_params['dims']) == [64, 64]:
        from tactile_gym_sim2real.pix2pix.gan_models.models_64 import GeneratorUNet, Discriminator, weights_init_normal
    else:
        sys.exit('Incorrect dims specified')

    for task_dirs in [['edge_2d'], ['surface_3d'], ['spherical_probe']]:
        main(opt, augmentation_params, weights, task_dirs, data_dirs)