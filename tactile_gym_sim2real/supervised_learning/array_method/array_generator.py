import numpy as np
import os
import time
import pandas as pd
import torch


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
        array_filename = os.path.join(self.data_dir, self.label_df.iloc[index]['sensor_array'])
        raw_array = np.load(array_filename)

        # augmentation goes here
        processed_array = raw_array[np.newaxis, ...].astype(np.float32)

        # get label
        target = np.array([self.label_df.iloc[index]['pose_1'], self.label_df.iloc[index]['pose_6']])

        sample = {'arrays': processed_array, 'labels': target}

        return sample
