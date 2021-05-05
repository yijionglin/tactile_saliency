import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import json
import os
import cv2
from skimage.metrics import structural_similarity

from pybullet_real2sim.image_transforms import load_video_frames

import seaborn as sns
sns.set(style="darkgrid")

# load data
# data_dirs = ['cube_straight']
# data_dirs = ['cylinder_curve']
data_dirs = ['mug_sin']
n_traj = 2

for traj_i in range(1,n_traj+1):
    for data_dir in data_dirs:

        full_data_dir = os.path.join('collected_data', data_dir, 'trajectories')
        cap = cv2.VideoCapture(os.path.join(full_data_dir, 'traj_{}.mp4'.format(traj_i)))

        # pull first frame
        ret, prev = cap.read()

        frame_counter = 0
        idx = 1
        frame_increment = 500

        while(cap.isOpened()):
            # pull frame
            ret, img = cap.read()

            if not ret:
                break

            if frame_counter % frame_increment == 0:

                if idx == 1:
                    first_img = img
                else:
                    second_img = img
                    second_weight = 1/(idx+1)
                    first_weight = 1 - second_weight
                    first_img = cv2.addWeighted(first_img, first_weight, second_img, second_weight, 0)

                # show frame
                cv2.imshow('overlay',first_img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                idx += 1

            frame_counter += 1

        # save image
        save_file = os.path.join(full_data_dir, 'overlay_image_{}.png'.format(traj_i))
        cv2.imwrite(save_file, first_img)

        cap.release()
        cv2.destroyAllWindows()
