import os
import itertools
import argparse
from torch.utils.data import DataLoader
import torch

from tactile_gym.assets import add_assets_path
from tactile_gym.utils.general_utils import str2bool

from tactile_gym_sim2real.pix2pix.image_generator import DataGenerator
from tactile_gym_sim2real.image_transforms import *
import random

from tactile_gym_sim2real.distractor_dev.image_augmentation import augment_img, rotate_img, rotate_img_angle
from tactile_gym_sim2real.distractor_dev import get_distractor_folder_path

def generate_overlay_images():
  augmentation_params = {
          #   'dims':        (680, 480),
          #   'dims':        (256, 256),
            'dims':        (128, 128),
            }

  home_dirs = get_distractor_folder_path()
  # data collected for task
  task_dirs = 'edge_2d'




  image_size_str = str(augmentation_params['dims'][0]) + 'x' + str(augmentation_params['dims'][1])


  tar_train_data_dirs = os.path.join(home_dirs,'data', task_dirs,'target', image_size_str, 'csv_train/images') 
  tar_val_data_dirs = os.path.join(home_dirs,'data', task_dirs,'target', image_size_str, 'csv_val/images') 

  dis_train_data_dirs = os.path.join(home_dirs,'data', task_dirs, 'distractor',image_size_str, 'csv_train/images') 
  dis_val_data_dirs = os.path.join(home_dirs,'data', task_dirs, 'distractor',image_size_str, 'csv_val/images') 

  aug_tar_train_data_dirs = os.path.join(home_dirs,'data', task_dirs,'aug_target', image_size_str, 'csv_train/images') 
  aug_val_data_dirs = os.path.join(home_dirs,'data', task_dirs,'aug_target', image_size_str, 'csv_val/images') 

  noise_train_data_dir = os.path.join(home_dirs,'data', task_dirs, 'noise_images',image_size_str, 'csv_train/images') 
  noise_val_data_dir = os.path.join(home_dirs,'data', task_dirs, 'noise_images',image_size_str, 'csv_val/images') 



  _, _, tar_train_imgs_files = next(os.walk(tar_train_data_dirs))
  train_set_len = len(tar_train_imgs_files)

  _, _, tar_val_imgs_files = next(os.walk(tar_val_data_dirs))
  val_set_len = len(tar_val_imgs_files)


  train_dis_index_list = list(range(1,train_set_len+1)) # list of integers from 1 to 99
                                # adjust this boundaries to fit your needs
  random.shuffle(train_dis_index_list)


  val_dis_index_list = list(range(1,val_set_len+1)) # list of integers from 1 to 99
                                # adjust this boundaries to fit your needs
  random.shuffle(val_dis_index_list)


  # generate train noise images
  for i in range(train_set_len):
    img_indx = "image_" + str(i+1) + ".png"
    dis_img_indx = train_dis_index_list.pop()
    dis_img_indx = "image_" + str(dis_img_indx) + ".png"
    
    tar_train_img_dir = os.path.join(tar_train_data_dirs,img_indx)
    # tar_train_img = augment_img(tar_train_img_dir, zoom_size_range = None, trans_range = [-5,5], if_rdn_rot = True, contrast_range = [0.8, 1])
    tar_train_img = augment_img(tar_train_img_dir, zoom_size_range = None, trans_range =None, if_rdn_rot = None, if_rdn_rot_range = 1, contrast_range = [0.8, 1])

    dis_train_img_dir = os.path.join(dis_train_data_dirs,dis_img_indx)
    # dis_train_img = augment_img(dis_train_img_dir, zoom_size_range = [1,120],trans_range = [-30,30], if_rdn_rot = True, contrast_range = [0.5, 2.0])
    dis_train_img = augment_img(dis_train_img_dir, zoom_size_range = [0,10],trans_range = None, trans_ver_range =  [-20,0], if_rdn_rot = None, if_rdn_rot_range = 75, contrast_range = [0.5, 2.0])

    if np.random.uniform(0,1)>0.01: 
      dis_ratio = 1
    else:
      dis_ratio = 0

    overlay_image = cv2.addWeighted(tar_train_img, 1, dis_train_img, dis_ratio, 0)[...,np.newaxis]
    

    angle = np.random.uniform(-180,180)
    tar_train_img = rotate_img_angle(tar_train_img, angle)[...,np.newaxis]
    overlay_image = rotate_img_angle(overlay_image, angle)[...,np.newaxis]
    noise_img_save_file_name = os.path.join(noise_train_data_dir, img_indx)
    cv2.imwrite(noise_img_save_file_name, overlay_image)
    aug_tar_img_save_file_name = os.path.join(aug_tar_train_data_dirs, img_indx)
    cv2.imwrite(aug_tar_img_save_file_name, tar_train_img)

  # generate val noise images
  for i in range(val_set_len):
    img_indx = "image_" + str(i+1) + ".png"
    dis_img_indx = val_dis_index_list.pop()
    dis_img_indx = "image_" + str(dis_img_indx) + ".png"

    tar_val_img_dir = os.path.join(tar_val_data_dirs,img_indx)
    tar_val_img = augment_img(tar_val_img_dir, zoom_size_range = None, trans_range = None, if_rdn_rot = None, if_rdn_rot_range = 1,contrast_range = [1, 1.2])

    dis_val_img_dir = os.path.join(dis_val_data_dirs,dis_img_indx)
    dis_val_img = augment_img(dis_val_img_dir, zoom_size_range = [0,10], trans_range = None,trans_ver_range = [-20,0], if_rdn_rot = None, if_rdn_rot_range = 75,contrast_range = [0.5, 2])
    if np.random.uniform(0,1)>0.01: 
        dis_ratio = 1
    else:
      dis_ratio = 0
    overlay_image = cv2.addWeighted(tar_val_img, 1, dis_val_img, dis_ratio, 0)[...,np.newaxis]
    
    angle = np.random.uniform(-180,180)
    tar_val_img = rotate_img_angle(tar_val_img, angle)[...,np.newaxis]
    overlay_image = rotate_img_angle(overlay_image, angle)[...,np.newaxis]

    noise_img_save_file_name = os.path.join(noise_val_data_dir, img_indx)
    cv2.imwrite(noise_img_save_file_name, overlay_image)

    aug_tar_img_save_file_name = os.path.join(aug_val_data_dirs, img_indx)
    cv2.imwrite(aug_tar_img_save_file_name, tar_val_img)

if __name__=="__main__":
    generate_overlay_images()
