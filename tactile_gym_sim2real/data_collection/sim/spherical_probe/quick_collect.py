import os
import numpy as np
from tactile_gym_sim2real.data_collection.sim.spherical_probe.setup_probe_data_collection import setup_collect_dir
from tactile_gym_sim2real.data_collection.sim.collect_data import quick_collect_csv

# define dir where real data is stored
target_dir = os.path.join(
    os.path.dirname(__file__),
    '../../real/data/spherical_probe/'
)

# Specify TacTip params
tactip_params = {
    'type': 'flat',
    'core': 'no_core',
    'dynamics': {},
    'turn_off_border': True,
}

# setup stimulus
stimulus_pos = [0.6, 0.0, 0.0]
stimulus_rpy = [0, 0, np.pi/2]
stim_path = os.path.join(
    os.path.dirname(__file__),
    "../stimuli/probe_stimuli/spherical_probe/spherical_probe.urdf"
)

# define parameters to iterate over in collection loops
image_sizes = [[64, 64], [128, 128], [256, 256]]
shear_types = [False]

# run the collection
quick_collect_csv(
    target_dir,
    tactip_params,
    stimulus_pos,
    stimulus_rpy,
    stim_path,
    setup_collect_dir,
    image_sizes,
    shear_types,
)