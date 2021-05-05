import os
import pandas as pd
import numpy as np
import cv2
from pybullet_sims.utils.general_utils import str2bool, load_video_frames, empty_dir

# experiment metadata
target_home_dir   = '/home/alex/Documents/pybullet_real2sim/pybullet_real2sim/data_collection/real/data/edge2dTap' # set this to stored data
# target_dir_name   = 'sqaure_-180_180_0_-6_train'
target_dir_name   = 'sqaure_-180_180_0_-6_val'

# original directory
og_collect_dir    = os.path.join(target_home_dir, target_dir_name)
og_video_dir      = os.path.join(og_collect_dir, 'videos')
og_target_file    = os.path.join(og_collect_dir, 'targets_video.csv')

# new directory
image_dir         = os.path.join(og_collect_dir, 'images')
target_file       = os.path.join(og_collect_dir, 'targets.csv')

# check save dir exists
if os.path.isdir(image_dir):
    input_str = input('Save directories already exists, would you like to continue (y,n)? ')
    if not str2bool(input_str):
        exit()
    else:
        empty_dir(image_dir) # clear out existing files)

# create dirs
os.makedirs(image_dir, exist_ok=True)

# read target data
target_df = pd.read_csv(og_target_file)

# change filenames from video -> image
target_df['sensor_image'] = target_df['sensor_video'].str.replace('video', 'image')
target_df['sensor_image'] = target_df['sensor_image'].str.replace('mp4', 'png')

# save the new csv
target_df.to_csv(target_file, index=False) # save in new dir

for index, row in target_df.iterrows():
    i_obj        = int(row.loc['obj_id'])
    sensor_video = row.loc['sensor_video']
    sensor_image = row.loc['sensor_image']

    # load video
    video = load_video_frames(os.path.join(og_video_dir, sensor_video))

    # take second frame as image
    image = video[1,:,:,:]

    # save the image
    image_outfile = os.path.join(image_dir, sensor_image)
    cv2.imwrite(image_outfile, image)

    print('Saving Image: ', sensor_image)
