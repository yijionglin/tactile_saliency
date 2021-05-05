import torch
from pybullet_real2sim.supervised_learning.image_generator import DataGenerator
from pybullet_real2sim.image_transforms import *


# train_data_dir = '../data_collection/real/data/edge2dTap/square_360_-6_6_train'
# train_data_dir = '../data_collection/generated_data/edge2dTap/square_360_-6_6_val'

# generated dataset
# train_data_dir = '../data_collection/generated_data/edge2dTap/square_360_-6_6_val/64_250epoch_[-6,6]_specnorm'
# train_data_dir = '../data_collection/generated_data/edge2dTap/square_360_-6_6_val/128_250epoch_[-6,6]_specnorm'
train_data_dir = '../data_collection/generated_data/edge2dTap/square_360_-6_6_val/256_250epoch_[-6,6]_specnorm'

validation_data_dir = ''

# Parameters
params = {'batch_size':  32,
          'epochs':      30,
          'lr':          1e-5,
          'lr_factor':   0.5,
          'lr_patience': 8,
          'dropout':     0.25,
          'dim':         (128,128),
          'bbox':        [70,0,550,480] if 'real' in train_data_dir else None,
          'shuffle':     True,
          'rshift':      None, #(0.05, 0.02),
          'rzoom':       None, #(0.95, 1),
          'thresh':      True if 'real' in train_data_dir else False,
          'brightlims':  None, #[0.3,1.0,-50,50], # alpha limits for contrast, beta limits for brightness
          'noise_var':   None, #0.001,
          'stdiz':       False,
          'normlz':      True,
          'train_data_dir':    os.path.join(train_data_dir, 'images'),
          'train_target_file': os.path.join(train_data_dir, 'targets.csv'),
          'val_data_dir':      os.path.join(validation_data_dir,  'images'),
          'val_target_file':   os.path.join(validation_data_dir,  'targets.csv')
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

        # print(np.max(sample_batched['images'][i].numpy()))
        # convert image to opencv format, not pytorch
        disp_image = convert_image_uint8(sample_batched['images'][i].numpy())
        disp_image = np.swapaxes(disp_image, 0, 2) # revert to channel last
        disp_image = np.swapaxes(disp_image, 0, 1) # revert to channel last

        # show image
        cv2.imshow("training_images", disp_image)
        k = cv2.waitKey(50)
        if k==27:    # Esc key to stop
            break


        time.sleep(0.5)
