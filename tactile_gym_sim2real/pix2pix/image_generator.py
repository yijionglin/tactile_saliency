import numpy as np
import os
import time
import cv2
import pandas as pd
import torch

from pybullet_real2sim.image_transforms import process_image

class DataGenerator(torch.utils.data.Dataset):

    def __init__(self, real_data_dirs, sim_data_dirs,
                 dim=(100,100), stdiz=False, normlz=False, thresh=None,
                 rshift=None, rzoom=None, brightlims=None, noise_var=None,
                 joint_aug=False):

        # check if data dirs are lists
        assert isinstance(real_data_dirs, list), "Real data dirs should be a list!"
        assert isinstance(sim_data_dirs, list),  "Sim data dirs should be a list!"

        self.dim = dim
        self.bbox = [80,25,530,475] # crop physical images with this
        self._stdiz = stdiz
        self._normlz = normlz
        self._thresh = thresh
        self._rshift = rshift
        self._rzoom = rzoom
        self._brightlims = brightlims
        self._noise_var = noise_var
        self._joint_aug = joint_aug

        # load csv file
        self.real_label_df = self.load_data_dirs(real_data_dirs)
        self.sim_label_df  = self.load_data_dirs(sim_data_dirs)

    def load_data_dirs(self, data_dirs):

        # add collumn for which dir data is stored in
        df_list = []
        for data_dir in data_dirs:
            df = pd.read_csv(os.path.join(data_dir, 'targets.csv'))
            df['image_dir'] = os.path.join(data_dir, 'images')
            df_list.append(df)

        # concat all df
        full_df = pd.concat(df_list)
        return full_df

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.real_label_df)))

    def __getitem__(self, index):
        'Generate one batch of data'

        # Generate data
        real_image_filename = os.path.join(self.real_label_df.iloc[index]['image_dir'], self.real_label_df.iloc[index]['sensor_image'])
        sim_image_filename  = os.path.join(self.sim_label_df.iloc[index]['image_dir'], self.sim_label_df.iloc[index]['sensor_image'])

        raw_real_image = cv2.imread(real_image_filename)
        raw_sim_image = cv2.imread(sim_image_filename)

        # preprocess/augment images separetly
        if not self._joint_aug:
            processed_real_image = process_image(raw_real_image, gray=True, bbox=self.bbox, dims=self.dim, stdiz=self._stdiz, normlz=self._normlz,
                                                 rshift=self._rshift, rzoom=self._rzoom, thresh=self._thresh,
                                                 add_axis=False, brightlims=self._brightlims, noise_var=self._noise_var)

            processed_sim_image = process_image(raw_sim_image, gray=True, bbox=None, dims=None, stdiz=self._stdiz, normlz=self._normlz,
                                                 rshift=None, rzoom=None, thresh=None,
                                                 add_axis=False, brightlims=None, noise_var=None)

            # put the channel into first axis because pytorch
            processed_real_image = np.rollaxis(processed_real_image, 2, 0)
            processed_sim_image = np.rollaxis(processed_sim_image, 2, 0)

        elif self._joint_aug:
            # apply some processing to the real image only
            processed_real_image = process_image(raw_real_image, gray=True, bbox=self.bbox, dims=self.dim, stdiz=self._stdiz, normlz=self._normlz,
                                                 rshift=None, rzoom=None, thresh=self._thresh,
                                                 add_axis=False, brightlims=self._brightlims, noise_var=self._noise_var)

            processed_sim_image = process_image(raw_sim_image, gray=True, bbox=None, dims=None, stdiz=self._stdiz, normlz=self._normlz,
                                                 rshift=None, rzoom=None, thresh=None,
                                                 add_axis=False, brightlims=None, noise_var=None)

            # stack images to apply the same data augmentations to both
            stacked_image = np.concatenate([processed_real_image, processed_sim_image], axis=2)

            # apply shift/zoom augs
            augmented_images = process_image(stacked_image, gray=False, bbox=None, dims=None, stdiz=False, normlz=False,
                                             rshift=self._rshift, rzoom=self._rzoom, thresh=None,
                                             add_axis=False, brightlims=None, noise_var=None)

            # unstack the images
            processed_real_image = augmented_images[...,0]
            processed_sim_image = augmented_images[...,1]

            # put the channel into first axis because pytorch
            processed_real_image = processed_real_image[np.newaxis,...]
            processed_sim_image = processed_sim_image[np.newaxis,...]

        return {"real": processed_real_image, "sim": processed_sim_image}
