import numpy as np
import os
import cv2
import pandas as pd
import torch

from tactile_gym_sim2real.image_transforms import process_image
from ipdb import set_trace
def get_gray_img(img_dir):
    # set_trace()
    img = cv2.imread(img_dir)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Add channel axis
    img = img[np.newaxis, ...]

    img = img.astype(np.float32) / 255.0

    return img

class DataGenerator(torch.utils.data.Dataset):

    def __init__(self, noise_data_dirs, tar_data_dirs):

        # check if data dirs are lists
        assert isinstance(noise_data_dirs, str), "Real data dirs should be a str!"
        assert isinstance(tar_data_dirs, str),  "Sim data dirs should be a str!"


        # load csv file
        self.tar_data_dirs = tar_data_dirs
        self.noise_data_dirs  = noise_data_dirs

    def __len__(self):
        'Denotes the number of batches per epoch'
        path, dirs, files = next(os.walk(self.noise_data_dirs))
        return int(len(files))



    def __getitem__(self, index):
        'Generate one batch of data'

        # Generate data
        img_index = 'image_' + str(index+1) + '.png'
        tar_image_filename = os.path.join(self.tar_data_dirs, img_index)
        noise_image_filename  = os.path.join(self.noise_data_dirs, img_index)
        tar_image = get_gray_img(tar_image_filename)
        noise_sim_image = get_gray_img(noise_image_filename)

        # set_trace()
        return {"tar": tar_image, "noise": noise_sim_image}
