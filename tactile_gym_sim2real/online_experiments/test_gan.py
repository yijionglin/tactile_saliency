import time
import os
import cv2
import imageio
import numpy as np

from vsp.video_stream import CvVideoDisplay, CvVideoOutputFile, CvVideoCamera
from vsp.processor import CameraStreamProcessorMT, AsyncProcessor

from tactile_gym_sim2real.online_experiments.gan_net import pix2pix_GAN
from tactile_gym_sim2real.image_transforms import *

from tactile_gym.assets import get_assets_path, add_assets_path

def make_sensor():
    return AsyncProcessor(CameraStreamProcessorMT(
            camera=CvVideoCamera(source=0,
                                 frame_size=(640, 480),
                                 is_color=True)
        ))

record_video = False
if record_video:
    video_frames = []


# image_size = [64,64]
# image_size = [128,128]
image_size = [256,256]

# set which gan
# dataset = 'edge_2d'
dataset = 'surface_3d'
# dataset = 'spherical_probe'

data_type = 'tap'
# data_type = 'shear'

image_size_str = str(image_size[0]) + 'x' + str(image_size[1])

# gan models
gan_model_dir = os.path.join(
    os.path.dirname(__file__),
    'trained_gans/[' + dataset + ']/' + image_size_str + '_[' + data_type + ']_250epochs/'
)

# load the gan params and overide some augmentation params as we dont want them when generating new data
gan_params = load_json_obj(os.path.join(gan_model_dir, 'augmentation_params'))
gan_params['rshift'] = None
gan_params['rzoom'] = None
gan_params['brightlims'] = None
gan_params['noise_var'] = None

# import the correct sized generator
if image_size == [64,64]:
    from tactile_gym_sim2real.pix2pix.gan_models.models_64 import GeneratorUNet
if image_size == [128,128]:
    from tactile_gym_sim2real.pix2pix.gan_models.models_128 import GeneratorUNet
if image_size == [256,256]:
    from tactile_gym_sim2real.pix2pix.gan_models.models_256 import GeneratorUNet

# load saved border image files
add_border = True
if add_border:
    ref_images_path = add_assets_path(
        os.path.join('robot_assets','tactip','tactip_reference_images')
    )

    border_gray_savefile = os.path.join( ref_images_path, 'standard', str(image_size[0]) + 'x' + str(image_size[0]), 'nodef_gray.npy')
    border_mask_savefile = os.path.join( ref_images_path, 'standard', str(image_size[0]) + 'x' + str(image_size[0]), 'border_mask.npy')
    border_gray = np.load(border_gray_savefile)
    border_mask = np.load(border_mask_savefile)

# init the GAN
GAN = pix2pix_GAN(gan_model_dir=gan_model_dir, Generator=GeneratorUNet, rl_image_size=image_size)

# init the sensor
sensor = make_sensor()

cv2.namedWindow("GAN_display")
while True:
    # pull raw frames from camera
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
