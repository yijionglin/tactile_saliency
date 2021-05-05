import torch
import time
import os
import cv2
import numpy as np
from pybullet_real2sim.supervised_learning.array_generator import DataGenerator
from pybullet_real2sim.image_transforms import convert_image_uint8

data_par_dir = '/home/alexc/fast_data/edge_sim_to_real/rigid_sim/' # rigid sim image data

# Parameters
params = {'batch_size':  32,
          'epochs':      30,
          'lr':          1e-5,
          'dropout':     0.25,
          'dim':         (64,64),
          'bbox':        None,
          'shuffle':     True,
          'rshift':      (0.05, 0.02),
          'rzoom':       (0.95, 1),
          'thresh':      False,
          'brightlims':  None,  #[0.3,1.0,-50,50], # alpha limits for contrast, beta limits for brightness
          'noise_var':   None,  # 0.001,
          'stdiz':       False,
          'normlz':      True,
          'train_data_dir':    os.path.join(data_par_dir, 'training', 'np_arrays'),
          'train_target_file': os.path.join(data_par_dir, 'training', 'targets.csv'),
          'val_data_dir':      os.path.join(data_par_dir, 'validation',  'np_arrays'),
          'val_target_file':   os.path.join(data_par_dir, 'validation',  'targets.csv')
          }


training_generator = DataGenerator(target_file=params['train_target_file'],
                                   data_dir=params['train_data_dir'],
                                   dim=params['dim'],
                                   bbox=params['bbox'],
                                   stdiz=params['stdiz'],
                                   normlz=params['normlz'],
                                   thresh=params['thresh'],
                                   rshift=params['rshift'],
                                   rzoom=params['rzoom'],
                                   brightlims=params['brightlims'],
                                   noise_var=params['noise_var'])

training_loader = torch.utils.data.DataLoader(training_generator,
                                              batch_size=params['batch_size'],
                                              shuffle=params['shuffle'],
                                              num_workers=1)


for (i_batch, sample_batched) in enumerate(training_loader,0):

    labels_np = sample_batched['labels'].numpy()
    # when generator returns merged X
    cv2.namedWindow("training_images");
    for i in range(params['batch_size']):

        r = labels_np[i,0]
        theta = labels_np[i,1]
        title = 'r={:.2f}, theta={:.2f}'.format(r,theta)
        print(title)

        # convert image to opencv format, not pytorch
        disp_image = convert_image_uint8(sample_batched['arrays'][i].numpy())

        # remove channel axis (needed for pytorch)
        disp_image = np.squeeze(disp_image)

        # show image
        cv2.imshow("training_images", disp_image)
        k = cv2.waitKey(50)
        if k==27:    # Esc key to stop
            break


        time.sleep(0.5)
