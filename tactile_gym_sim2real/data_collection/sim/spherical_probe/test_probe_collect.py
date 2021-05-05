import os
import numpy as np
from pybullet_sims import models

from pybullet_real2sim.data_collection.sim.spherical_probe.setup_probe_data_collection import setup_collect_dir
from pybullet_real2sim.data_collection.sim.collect_data import collect_data

tactip_params = {
    'type':'flat',
    'core':'no_core',
    'image_size':[256,256],
    'border':True
}

show_gui = True
show_tactile = True
num_samples = 10 # ignored if using csv
apply_shear = False
collect_dir_name = 'temp'

# use csv or not
# og_collect_dir = '/home/alex/Documents/pybullet_real2sim/pybullet_real2sim/data_collection/real/data/spherical_probe/tap/csv_val' # set this to stored data
og_collect_dir = None

# setup stimulus
stimulus_pos = [0.6,0.0,0.0]
stimulus_rpy = [0,0,np.pi/2]
stim_path = os.path.join(models.getDataPath(),"tactile_stimuli/probe_stimuli/spherical_probe/spherical_probe.urdf")

target_df, image_dir, workframe_pos, workframe_rpy = setup_collect_dir(num_samples=num_samples,
                                                                       apply_shear=apply_shear,
                                                                       shuffle_data=False,
                                                                       og_collect_dir=og_collect_dir,
                                                                       collect_dir_name=collect_dir_name)

collect_data(target_df, image_dir, stim_path, stimulus_pos, stimulus_rpy, workframe_pos, workframe_rpy, tactip_params, show_gui=show_gui, show_tactile=show_tactile)
