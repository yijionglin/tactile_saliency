import os
import numpy as np
from pybullet_sims import models
from pybullet_real2sim.data_collection.sim.probe.setup_probe_data_collection import setup_collect_dir
from pybullet_real2sim.data_collection.sim.collect_data import collect_data


"""
Random collect
Train: 5000
Val: 2000

CSV collect
Dir: '/home/alex/Documents/pybullet_real2sim/pybullet_real2sim/data_collection/real/data/' # set this to stored data
Train: 'csv_train'
Val:   'csv_val'
"""

# Specify TacTip params
tactip_params = {
    'type':'flat',
    'core':'no_core',
}

# setup stimulus
stimulus_pos = [0.6,0.0,0.0]
stimulus_rpy = [0,0,np.pi/2]
stim_path = os.path.join(models.getDataPath(),"tactile_stimuli/probe_stimuli/spherical_probe/spherical_probe.urdf")

# shared params
show_gui     = False
show_tactile = False
shuffle_data = False
apply_shear  = False

if apply_shear:
    target_home_dir   = '/home/alex/Documents/pybullet_real2sim/pybullet_real2sim/data_collection/real/data/spherical_probe/shear'
else:
    target_home_dir   = '/home/alex/Documents/pybullet_real2sim/pybullet_real2sim/data_collection/real/data/spherical_probe/tap'

image_sizes = [[64,64], [128,128], [256,256]]
border_types = [True, False]

for border_type in border_types:
    for image_size in image_sizes:

        tactip_params['image_size'] = image_size
        tactip_params['border'] = border_type

        image_size_str = str(tactip_params['image_size'][0]) + 'x' + str(tactip_params['image_size'][1])
        border_str = 'border' if tactip_params['border'] else 'noborder'
        base_dir_name = os.path.join(border_str, image_size_str)

        # Random Collect
        # train
        collect_dir_name = os.path.join(base_dir_name, 'rand_train')
        num_samples = 500
        target_df, image_dir, workframe_pos, workframe_rpy = setup_collect_dir(num_samples=num_samples,
                                                                               apply_shear=apply_shear,
                                                                               shuffle_data=shuffle_data,
                                                                               og_collect_dir=None,
                                                                               collect_dir_name=collect_dir_name)
        collect_data(target_df, image_dir, stim_path, stimulus_pos, stimulus_rpy, workframe_pos, workframe_rpy, tactip_params, show_gui=show_gui, show_tactile=show_tactile)


        # val
        collect_dir_name = os.path.join(base_dir_name, 'rand_val')
        num_samples = 200
        target_df, image_dir, workframe_pos, workframe_rpy = setup_collect_dir(num_samples=num_samples,
                                                                               apply_shear=apply_shear,
                                                                               shuffle_data=shuffle_data,
                                                                               og_collect_dir=None,
                                                                               collect_dir_name=collect_dir_name)
        collect_data(target_df, image_dir, stim_path, stimulus_pos, stimulus_rpy, workframe_pos, workframe_rpy, tactip_params, show_gui=show_gui, show_tactile=show_tactile)


        # CSV Collect
        target_dir_name  = 'csv_train'
        og_collect_dir   = os.path.join(target_home_dir, target_dir_name)
        collect_dir_name = os.path.join(base_dir_name, target_dir_name)
        target_df, image_dir, workframe_pos, workframe_rpy = setup_collect_dir(num_samples=None,
                                                                               apply_shear=apply_shear,
                                                                               shuffle_data=shuffle_data,
                                                                               og_collect_dir=og_collect_dir,
                                                                               collect_dir_name=collect_dir_name)
        collect_data(target_df, image_dir, stim_path, stimulus_pos, stimulus_rpy, workframe_pos, workframe_rpy, tactip_params, show_gui=show_gui, show_tactile=show_tactile)

        target_dir_name   = 'csv_val'
        og_collect_dir   = os.path.join(target_home_dir, target_dir_name)
        collect_dir_name = os.path.join(base_dir_name, target_dir_name)
        target_df, image_dir, workframe_pos, workframe_rpy = setup_collect_dir(num_samples=None,
                                                                               apply_shear=apply_shear,
                                                                               shuffle_data=shuffle_data,
                                                                               og_collect_dir=og_collect_dir,
                                                                               collect_dir_name=collect_dir_name)
        collect_data(target_df, image_dir, stim_path, stimulus_pos, stimulus_rpy, workframe_pos, workframe_rpy, tactip_params, show_gui=show_gui, show_tactile=show_tactile)
