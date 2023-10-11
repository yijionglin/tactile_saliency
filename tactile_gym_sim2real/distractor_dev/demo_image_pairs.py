import os
import itertools
import argparse
from torch.utils.data import DataLoader
import torch

from tactile_gym.assets import add_assets_path
from tactile_gym.utils.general_utils import str2bool
from tactile_gym_sim2real.distractor_dev import get_distractor_folder_path
# from tactile_gym_sim2real.pix2pix.image_generator import DataGenerator
from tactile_gym_sim2real.distractor_dev.noise_images_generator  import DataGenerator

from tactile_gym_sim2real.image_transforms import *


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--shuffle", type=str2bool, default=False, help="shuffle the generated image data")
opt = parser.parse_args()

# Parameters
augmentation_params = {
          'dims':        (128, 128),

          }

add_border = False
image_size_str = str(augmentation_params['dims'][0]) + 'x' + str(augmentation_params['dims'][1])

if add_border:

    border_images_path = add_assets_path(
        os.path.join('robot_assets', 'tactip', 'tactip_reference_images')
    )

    saved_file_dir = os.path.join(
        border_images_path,
        'standard',  # TODO: automate flat/standard/right_angle choice
        image_size_str,
    )

    border_gray = torch.FloatTensor(np.load(os.path.join(saved_file_dir, "nodef_gray.npy")))
    border_mask = torch.FloatTensor(np.load(os.path.join(saved_file_dir, "border_mask.npy")))

# data collected for task
task_dirs = 'edge_2d'
home_dirs = get_distractor_folder_path()

tar_train_data_dirs = os.path.join(home_dirs,'data', task_dirs,'target', image_size_str, 'csv_train/images') 
tar_val_data_dirs = os.path.join(home_dirs, 'data', task_dirs,'target', image_size_str, 'csv_val/images') 

noise_train_data_dir = os.path.join(home_dirs,'data', task_dirs, 'noise_images',image_size_str, 'csv_train/images') 
noise_val_data_dir = os.path.join(home_dirs,'data', task_dirs, 'noise_images',image_size_str, 'csv_val/images') 



# Configure dataloaders
training_generator = DataGenerator(tar_data_dirs=tar_train_data_dirs,
                                   noise_data_dirs=noise_train_data_dir,)

val_generator = DataGenerator(tar_data_dirs=tar_val_data_dirs,
                              noise_data_dirs=noise_val_data_dir,)

training_loader = torch.utils.data.DataLoader(training_generator,
                                              batch_size=opt.batch_size,
                                              shuffle=opt.shuffle,
                                              num_workers=opt.n_cpu)

val_loader = torch.utils.data.DataLoader(val_generator,
                                         batch_size=opt.batch_size,
                                         shuffle=opt.shuffle,
                                         num_workers=opt.n_cpu)

for (i_batch, sample_batched) in enumerate(val_loader,0):
    
    tar_images = sample_batched['tar']
    noise_images = sample_batched['noise']
    
    if add_border:
        noise_images[:, :, border_mask == 1] = border_gray[border_mask == 1] / 255.0

    cv2.namedWindow("training_images")
        
    for i in range(opt.batch_size):

        # convert image to opencv format, not pytorch
        real_image = np.swapaxes(np.swapaxes(tar_images[i].numpy(), 0, 2), 0, 1)
        sim_image  = np.swapaxes(np.swapaxes(noise_images[i].numpy(), 0, 2), 0, 1)
        overlay_image = cv2.addWeighted(real_image, 1, sim_image, 1, 0)[...,np.newaxis]

        
        disp_image = np.concatenate([real_image, sim_image, overlay_image], axis=1)


    cv2.imshow("training_images", disp_image)

    k = cv2.waitKey(1000)

    if k == 27:    # Esc key to stop
        break
