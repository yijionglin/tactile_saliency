import numpy as np
import os
import time
import cv2
import pandas as pd
import torch

from pybullet_real2sim.image_transforms import process_image


class DataGenerator(torch.utils.data.Dataset):

    def __init__(self, target_file, data_dir,
                 dim=(100,100), bbox=None, stdiz=False, normlz=False, thresh=None,
                 rshift=None, rzoom=None, brightlims=None, noise_var=None):

        self.data_dir = data_dir
        self.dim = dim
        self.bbox = bbox
        self._stdiz = stdiz
        self._normlz = normlz
        self._thresh = thresh
        self._rshift = rshift
        self._rzoom = rzoom
        self._brightlims = brightlims
        self._noise_var = noise_var

        # load csv file
        self.label_df = pd.read_csv(target_file)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.label_df)))

    def __getitem__(self, index):
        'Generate one batch of data'

        # Generate data
        image_filename = os.path.join(self.data_dir, self.label_df.iloc[index]['sensor_image'])
        raw_image = cv2.imread(image_filename)
        
        # preprocess/augment image
        processed_image = process_image(raw_image, gray=True, bbox=self.bbox, dims=self.dim, stdiz=self._stdiz, normlz=self._normlz,
                                        rshift=self._rshift, rzoom=self._rzoom, thresh=self._thresh,
                                        add_axis=False, brightlims=self._brightlims, noise_var=self._noise_var)

        # put the channel into first axis because pytorch
        processed_image = np.rollaxis(processed_image, 2, 0)

        # get label
        target = np.array([self.label_df.iloc[index]['pose_2'], self.label_df.iloc[index]['pose_6']])

        sample = {'images': processed_image, 'labels': target}

        return sample
