import os
import cv2
import imageio
import numpy as np

from vsp.video_stream import CvVideoDisplay, CvVideoOutputFile, CvVideoCamera
from vsp.processor import CameraStreamProcessorMT, AsyncProcessor

from pybullet_real2sim.online_experiments.gan_net import pix2pix_GAN
from pybullet_real2sim.image_transforms import *

from pybullet_sims.rl_envs.ur5_envs.tactip_reference_images import *

def make_sensor():
    return AsyncProcessor(CameraStreamProcessorMT(
            camera=CvVideoCamera(source=0,
                                 frame_size=(640, 480),
                                 is_color=True)
        ))

record_video = False
if record_video:
    video_frames = []

# gan models
# gan_model_dir = os.path.join(os.path.dirname(__file__), 'trained_gans/[edge_2d]/256x256_[tap]_500epochs/')
gan_model_dir = os.path.join(os.path.dirname(__file__), 'trained_gans/[edge_2d]/256x256_[shear]_500epochs/')
# gan_model_dir = os.path.join(os.path.dirname(__file__), 'trained_gans/[surface_3d]/256x256_[tap]_500epochs/')
# gan_model_dir = os.path.join(os.path.dirname(__file__), 'trained_gans/[surface_3d]/256x256_[shear]_500epochs/')

# gan_model_dir = os.path.join(os.path.dirname(__file__), 'trained_gans/[edge_2d,surface_3d]/256x256_[tap]_250epochs/')
# gan_model_dir = os.path.join(os.path.dirname(__file__), 'trained_gans/[edge_2d,surface_3d]/256x256_[shear]_250epochs/')

# gan_model_dir = os.path.join(os.path.dirname(__file__), 'trained_gans/[spherical_probe]/256x256_[tap]_250epochs_thresh/')

# home pc
# gan_model_dir = os.path.join(os.path.dirname(__file__), '../pix2pix/saved_models/[surface_3d]/256x256_[shear]_500epochs/')

# load the gan params and overide some augmentation params as we dont want them when generating new data
gan_params = load_json_obj(os.path.join(gan_model_dir, 'augmentation_params'))
gan_params['rshift'] = None
gan_params['rzoom'] = None
gan_params['brightlims'] = None
gan_params['noise_var'] = None

# load the trained pix2pix GAN network
image_size = [256,256]
add_border = True
GAN = pix2pix_GAN(gan_model_dir=gan_model_dir, rl_image_size=image_size)

# load saved border image files
border_gray_savefile = os.path.join( getBorderImagesPath(), 'standard', str(image_size[0]) + 'x' + str(image_size[0]), 'border_gray.npy')
border_mask_savefile = os.path.join( getBorderImagesPath(), 'standard', str(image_size[0]) + 'x' + str(image_size[0]), 'border_mask.npy')
border_gray = np.load(border_gray_savefile)
border_mask = np.load(border_mask_savefile)

# init the sensor
sensor = make_sensor()

cv2.namedWindow("GAN_display")
while True:

    raw_real_frames = sensor.process(num_frames=1)
    raw_real_image = raw_real_frames[0]

    # process with gan here (proccessing applied in GAN class)
    generated_sim_image, processed_real_image = GAN.gen_sim_image(raw_real_image)

    # add a border to the generated image
    if add_border:
        generated_sim_image[border_mask==1] = border_gray[border_mask==1]

    # add axis to generated sim image
    generated_sim_image = generated_sim_image[...,np.newaxis]

    # convert the generated image to float format to match processed real image
    # (which is usually normalised before inputting into network)
    processed_real_image = (processed_real_image/255).astype(np.float32) # convert to image format
    generated_sim_image  = (generated_sim_image/255).astype(np.float32) # convert to image format


    # create an overlay of the two images
    overlay_image = cv2.addWeighted(processed_real_image, 0.25, generated_sim_image, 0.9, 0)[...,np.newaxis]

    # concat all images to display at once
    disp_image = np.concatenate([processed_real_image, generated_sim_image, overlay_image], axis=1)

    if record_video:
        video_frames.append(disp_image)

    # show image
    cv2.imshow("GAN_display", disp_image)
    k = cv2.waitKey(10)
    if k==27:    # Esc key to stop

        if record_video:
            video_file = os.path.join('gan_videos', 'tactile_video.mp4')
            imageio.mimwrite(video_file, np.stack(video_frames), fps=20)

        break
