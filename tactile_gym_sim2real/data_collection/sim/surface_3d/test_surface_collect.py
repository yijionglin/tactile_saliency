import os
import numpy as np

from tactile_gym import models

from tactile_gym_sim2real.data_collection.sim.surface_3d.setup_surface_data_collection import setup_collect_dir
from tactile_gym_sim2real.data_collection.sim.collect_data import collect_data

tactip_params = {
    'type':'standard',
    'core':'no_core',
    'image_size':[256,256],
    'border':True
}

show_gui = True
show_tactile = True
num_samples = 100 # ignored if using csv
apply_shear = True
collect_dir_name = 'temp'

# use csv or not
# og_collect_dir = '/home/alex/Documents/tactile_gym_sim2real/tactile_gym_sim2real/data_collection/real/data/surface_3d/tap/csv_val' # set this to stored data
og_collect_dir = None

# setup stimulus
stimulus_pos = [0.6,0.0,0.025]
stimulus_rpy = [0,0,np.pi/2]
stim_path = os.path.join(models.getDataPath(),"tactile_stimuli/edge_stimuli/square/square.urdf")

target_df, image_dir, workframe_pos, workframe_rpy = setup_collect_dir(num_samples=num_samples,
                                                                       apply_shear=apply_shear,
                                                                       shuffle_data=False,
                                                                       og_collect_dir=og_collect_dir,
                                                                       collect_dir_name=collect_dir_name)

collect_data(target_df, image_dir, stim_path, stimulus_pos, stimulus_rpy, workframe_pos, workframe_rpy, tactip_params, show_gui=show_gui, show_tactile=show_tactile)
