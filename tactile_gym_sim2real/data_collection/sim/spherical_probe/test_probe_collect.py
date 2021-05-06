import os
import numpy as np
from tactile_gym.assets import add_assets_path
from tactile_gym_sim2real.data_collection.sim.spherical_probe.setup_probe_data_collection import setup_collect_dir
from tactile_gym_sim2real.data_collection.sim.collect_data import collect_data

tactip_params = {
    'type': 'flat',
    'core': 'no_core',
    'dynamics': {},
    'image_size': [256, 256],
    'turn_off_border': True,
}

show_gui = True
show_tactile = True
num_samples = 10 # ignored if using csv
apply_shear = False
collect_dir_name = 'temp'

# use csv or not
# og_collect_dir = os.path.join(
#     os.path.dirname(__file__),
#     '../../real/data/spherical_probe/tap/csv_val'
# )
og_collect_dir = None

# setup stimulus
stimulus_pos = [0.6, 0.0, 0.0]
stimulus_rpy = [0, 0, np.pi/2]
stim_path = os.path.join(
    os.path.dirname(__file__),
    "../stimuli/probe_stimuli/spherical_probe/spherical_probe.urdf"
)

target_df, image_dir, workframe_pos, workframe_rpy = setup_collect_dir(num_samples=num_samples,
                                                                       apply_shear=apply_shear,
                                                                       shuffle_data=False,
                                                                       og_collect_dir=og_collect_dir,
                                                                       collect_dir_name=collect_dir_name)

collect_data(
    target_df,
    image_dir,
    stim_path,
    stimulus_pos,
    stimulus_rpy,
    workframe_pos,
    workframe_rpy,
    tactip_params,
    show_gui=show_gui,
    show_tactile=show_tactile
)
