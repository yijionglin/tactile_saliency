import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import json
import os
import cv2
from skimage.metrics import structural_similarity

from tactile_gym_sim2real.image_transforms import load_video_frames

import seaborn as sns
sns.set(style="darkgrid")

# load data
# data_dirs = ['2mm_5hz', '4mm_5hz', '6mm_5hz', '8mm_5hz', '8mm_8hz']
data_dirs = ['2mm', '4mm', '6mm', '8mm']

for data_dir in data_dirs:

    full_data_dir = os.path.join('collected_data', 'data', data_dir)
    video_frames = load_video_frames(os.path.join(full_data_dir, 'tactile_video.mp4'))
    overlay_increment = 2
    offset = 10
    # steps_per_goal = 50
    steps_per_goal = 100
    n_goals = 5

    # create block of frames for each goal
    frame_blocks = []
    for i in range(n_goals):

        first_id = (i*steps_per_goal) + offset
        last_id = ((i+1)*steps_per_goal)

        frame_block = video_frames[first_id:last_id:overlay_increment, :, :, :]

        frame_blocks.append(frame_block)

    # overlay image method
    for i, frame_block in enumerate(frame_blocks):

        prev = frame_block[0]

        for frame in frame_block:

            # ssim diff image
            # (score, diff_image) = structural_similarity(frame, prev, full=True, multichannel=True)
            # diff_image = (diff_image * 255).astype(np.uint8)
            # print(diff_image.shape)

            # abs diff image
            diff_image = np.abs(frame.astype(np.float32) - prev.astype(np.float32))
            diff_image = (diff_image).astype(np.uint8)

            overlay_image = cv2.addWeighted(prev, 0.99, frame, 0.5, gamma=0.0)

            cv2.imshow("overlay", diff_image)
            k = cv2.waitKey(50)
            if k==27:    # Esc key to stop
                break
            prev = overlay_image

        os.makedirs(os.path.join(full_data_dir, 'difference_images'), exist_ok=True)
        save_file = os.path.join(full_data_dir, 'difference_images', 'difference_image_{}.png'.format(i))
        cv2.imwrite(save_file, diff_image)

        os.makedirs(os.path.join(full_data_dir, 'overlay_images'), exist_ok=True)
        save_file = os.path.join(full_data_dir, 'overlay_images', 'overlay_image_{}.png'.format(i))
        cv2.imwrite(save_file, overlay_image)


    # optic flow method
    # for frame_block in frame_blocks:
    #
    #     prev = np.zeros(shape=video_frames[0].shape, dtype=np.uint8)
    #     prev = cv2.cvtColor(prev,cv2.COLOR_BGR2GRAY)
    #
    #     hsv = np.zeros(shape=video_frames[0].shape, dtype=np.uint8)
    #     hsv[...,1] = 255
    #
    #     for frame in frame_block:
    #         next = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #         flow = cv2.calcOpticalFlowFarneback(prev,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    #
    #         mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    #         hsv[...,0] = ang*180/np.pi/2
    #         hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    #         rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    #
    #         cv2.imshow('opt_flow',rgb)
    #         k = cv2.waitKey(30)
    #         if k == 27:
    #             break
    #
    #         prev = next
