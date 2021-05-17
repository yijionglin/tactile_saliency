import os
import itertools
import argparse
from torch.utils.data import DataLoader
import torch

from tactile_gym.assets import add_assets_path
from tactile_gym.utils.general_utils import str2bool

from tactile_gym_sim2real.pix2pix.image_generator import DataGenerator
from tactile_gym_sim2real.image_transforms import *


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--shuffle", type=str2bool, default=True, help="shuffle the generated image data")
opt = parser.parse_args()

# Parameters
augmentation_params = {
          'dims':        (256, 256),
          'rshift':      None,  # (0.025, 0.025),
          'rzoom':       None,  # (0.98, 1),
          'thresh':      True,
          'brightlims':  None,  # [0.3,1.0,-50,50], # alpha limits for contrast, beta limits for brightness
          'noise_var':   None,  # 0.001,
          'stdiz':       False,
          'normlz':      True,
          'joint_aug':   False
          }

add_border = True
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
# task_dirs = ['edge_2d']
task_dirs = ['surface_3d']
# task_dirs = ['spherical_probe']
# task_dirs = ['edge_2d', 'surface_3d']

# for GAN data
# data_dirs = ['tap']
data_dirs = ['shear']
# data_dirs = ['tap', 'shear']

# combine the data directories
combined_dirs = list(itertools.product(task_dirs, data_dirs))
combined_paths = [os.path.join(*i) for i in combined_dirs]

# data dir real -> sim
training_real_data_dirs = [os.path.join('../data_collection/real/data/', data_path, 'csv_train') for data_path in combined_paths]
validation_real_data_dirs = [os.path.join('../data_collection/real/data/', data_path, 'csv_val') for data_path in combined_paths]
training_sim_data_dirs = [os.path.join('../data_collection/sim/data/', data_path, image_size_str, 'csv_train') for data_path in combined_paths]
validation_sim_data_dirs = [os.path.join('../data_collection/sim/data/', data_path, image_size_str, 'csv_val') for data_path in combined_paths]

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

for (i_batch, sample_batched) in enumerate(training_loader,0):

    real_images = sample_batched['real']
    sim_images = sample_batched['sim']

    if add_border:
        sim_images[:, :, border_mask == 1] = border_gray[border_mask == 1] / 255.0

    cv2.namedWindow("training_images")

    for i in range(opt.batch_size):

        # convert image to opencv format, not pytorch
        real_image = np.swapaxes(np.swapaxes(real_images[i].numpy(), 0, 2), 0, 1)
        sim_image  = np.swapaxes(np.swapaxes(sim_images[i].numpy(), 0, 2), 0, 1)
        overlay_image = cv2.addWeighted(real_image, 0.25, sim_image, 0.9, 0)[...,np.newaxis]

        disp_image = np.concatenate([real_image, sim_image, overlay_image], axis=1)


    # show image
    cv2.imshow("training_images", disp_image)
    k = cv2.waitKey(500)
    if k == 27:    # Esc key to stop
        break
