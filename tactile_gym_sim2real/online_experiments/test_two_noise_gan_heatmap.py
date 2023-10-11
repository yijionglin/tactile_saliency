import time
import os
import cv2
import imageio
import numpy as np

from vsp.video_stream import CvVideoDisplay, CvVideoOutputFile, CvVideoCamera
from vsp.processor import CameraStreamProcessorMT, AsyncProcessor

from tactile_gym_sim2real.online_experiments.gan_net import pix2pix_GAN
from tactile_gym_sim2real.online_experiments.noise_gan_net import noise_pix2pix_GAN
from tactile_gym_sim2real.image_transforms import *

from tactile_gym.assets import get_assets_path, add_assets_path

def make_sensor():
    return AsyncProcessor(CameraStreamProcessorMT(
            camera=CvVideoCamera(source=0,#TODO: change the source to 1.
                                 frame_size=(640, 480),
                                 is_color=True)
        ))

record_video = True
if record_video:
    video_frames = []

# useful for rolling task
track_indent = False

image_size = [128,128]
display_size = tuple([512,512])
# set which gan
dataset = 'edge_2d'


# data_type = 'tap'
data_type = 'shear'
show_attention = True
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

# noise gan models
noise_gan_model_dir = os.path.join(
    os.path.dirname(__file__),
    'trained_noise_gans/[' + dataset + ']/' + image_size_str + '_[' + data_type + ']_250epochs/'
)

# load the gan params and overide some augmentation params as we dont want them when generating new data
noise_gan_params = load_json_obj(os.path.join(noise_gan_model_dir, 'augmentation_params'))
noise_gan_params['rshift'] = None
noise_gan_params['rzoom'] = None
noise_gan_params['brightlims'] = None
noise_gan_params['noise_var'] = None

# noise gan models
gau_noise_gan_model_dir = os.path.join(
    os.path.dirname(__file__),
    'trained_noise_gans/[' + dataset + ']/' + image_size_str + '_[' + data_type + ']_250epochs_gaussian/'
)

# load the gan params and overide some augmentation params as we dont want them when generating new data
gau_noise_gan_params = load_json_obj(os.path.join(noise_gan_model_dir, 'augmentation_params'))
gau_noise_gan_params['rshift'] = None
gau_noise_gan_params['rzoom'] = None
gau_noise_gan_params['brightlims'] = None
gau_noise_gan_params['noise_var'] = None


# import the correct sized generator
if image_size == [64,64]:
    from tactile_gym_sim2real.pix2pix.gan_models.models_64 import GeneratorUNet
if image_size == [128,128]:
    from tactile_gym_sim2real.pix2pix.gan_models.models_128 import GeneratorUNet
if image_size == [256,256]:
    from tactile_gym_sim2real.pix2pix.gan_models.models_256 import GeneratorUNet

# Setup SimpleBlobDetector parameters.
if track_indent:
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 0
    params.maxThreshold = 255
    params.filterByArea = True
    params.minArea = 20
    params.filterByCircularity = False
    params.filterByConvexity = False
    params.filterByInertia = False
    detector = cv2.SimpleBlobDetector_create(params)

# load saved border image files
add_border = False
if add_border:
    ref_images_path = add_assets_path(
        os.path.join('robot_assets','tactip','tactip_reference_images')
    )

    border_gray_savefile = os.path.join( ref_images_path, 'standard', str(image_size[0]) + 'x' + str(image_size[0]), 'nodef_gray.npy')
    border_mask_savefile = os.path.join( ref_images_path, 'standard', str(image_size[0]) + 'x' + str(image_size[0]), 'border_mask.npy')
    border_gray = np.load(border_gray_savefile)
    border_mask = np.load(border_mask_savefile)

# init the GAN
GAN = pix2pix_GAN(gan_model_dir=gan_model_dir, Generator=GeneratorUNet, rl_image_size=image_size, bbox = [170,90,458,378])
NOISE_GAN = noise_pix2pix_GAN(gan_model_dir=noise_gan_model_dir, Generator=GeneratorUNet, rl_image_size=image_size)
GAU_NOISE_GAN = noise_pix2pix_GAN(gan_model_dir=gau_noise_gan_model_dir, Generator=GeneratorUNet, rl_image_size=image_size)

# init the sensor
sensor = make_sensor()

cv2.namedWindow("GAN_display")
while True:
    # pull raw frames from camera
    raw_real_frames = sensor.process(num_frames=1)
    raw_real_image = raw_real_frames[0]

    # process with gan here (proccessing applied in GAN class)
    generated_sim_image, processed_real_image = GAN.gen_sim_image(raw_real_image)
    gen_denoise_sim_image, _ = NOISE_GAN.gen_denoise_sim_image(generated_sim_image)
    gau_gen_denoise_sim_image, _ = GAU_NOISE_GAN.gen_denoise_sim_image(generated_sim_image)

    # resize just for recording videos
    x0, y0, x1, y1 =  [149, 64, 478, 394]
    raw_real_image = raw_real_image[y0:y1, x0:x1]
    raw_real_image = cv2.resize(raw_real_image, display_size, interpolation=cv2.INTER_AREA)
    # 58 is the mask diameter for training
    raw_real_image = apply_circle_mask (raw_real_image,58 * 4)
    # add axis to generated sim image
    generated_sim_image = generated_sim_image[...,np.newaxis]
    gen_denoise_sim_image = gen_denoise_sim_image[...,np.newaxis]
    gau_gen_denoise_sim_image = gau_gen_denoise_sim_image[...,np.newaxis]

    # convert the generated image to float format to match processed real image
    # (which is usually normalised before inputting into network)
    processed_real_image = cv2.resize(processed_real_image, display_size)


    # backup an RGB format of the grey scale images
    generated_sim_image_grey_to_rgb = cv2.cvtColor(generated_sim_image,cv2.COLOR_GRAY2RGB)
    generated_sim_image_grey_to_rgb = cv2.resize(generated_sim_image_grey_to_rgb, display_size)
    gen_denoise_sim_image_grey_to_rgb = cv2.cvtColor(gen_denoise_sim_image,cv2.COLOR_GRAY2RGB)
    gen_denoise_sim_image_grey_to_rgb = cv2.resize(gen_denoise_sim_image_grey_to_rgb, display_size)

    gau_gen_denoise_sim_image_grey_to_rgb = cv2.cvtColor(gau_gen_denoise_sim_image,cv2.COLOR_GRAY2RGB)
    gau_gen_denoise_sim_image_grey_to_rgb = cv2.resize(gau_gen_denoise_sim_image, display_size)
    
    if show_attention:
        # heatmap
        if generated_sim_image.max()>5:
            # set_trace()
            generated_sim_image = generated_sim_image/int(generated_sim_image.max())
            generated_sim_image = (generated_sim_image*255).astype(np.uint8)
        generated_sim_image = cv2.applyColorMap(generated_sim_image, cv2.COLORMAP_JET)
        generated_sim_image = cv2.resize(generated_sim_image, display_size)
        

        if gen_denoise_sim_image.max()>5:
            # set_trace()
            gen_denoise_sim_image = gen_denoise_sim_image/int(gen_denoise_sim_image.max())
            gen_denoise_sim_image = (gen_denoise_sim_image*255).astype(np.uint8)
        gen_denoise_sim_image = cv2.applyColorMap(gen_denoise_sim_image, cv2.COLORMAP_JET)
        gen_denoise_sim_image = cv2.resize(gen_denoise_sim_image, display_size)
        
        if gau_gen_denoise_sim_image.max()>5:
            # set_trace()
            gau_gen_denoise_sim_image = gau_gen_denoise_sim_image/int(gau_gen_denoise_sim_image.max())
            gau_gen_denoise_sim_image = (gau_gen_denoise_sim_image*255).astype(np.uint8)
        gau_gen_denoise_sim_image = cv2.applyColorMap(gau_gen_denoise_sim_image, cv2.COLORMAP_JET)
        gau_gen_denoise_sim_image = cv2.resize(gau_gen_denoise_sim_image, display_size)

        processed_real_image = cv2.cvtColor(processed_real_image, cv2.COLOR_GRAY2RGB)


    # concat all images to display at once
    overlay_raw_image_w_sal = cv2.addWeighted(raw_real_image, 0.75, gen_denoise_sim_image, 0.65, 0)
    overlay_raw_image_w_sal = cv2.resize(overlay_raw_image_w_sal, display_size)

    gau_overlay_raw_image_w_sal = cv2.addWeighted(raw_real_image, 0.75, gau_gen_denoise_sim_image, 0.65, 0)
    gau_overlay_raw_image_w_sal = cv2.resize(gau_overlay_raw_image_w_sal, display_size)


    overlay_depth_image_with_sal = cv2.addWeighted(generated_sim_image_grey_to_rgb, 0.65, gen_denoise_sim_image, 0.35, 0)
    overlay_depth_image_with_sal = cv2.resize(overlay_depth_image_with_sal, display_size)

    overlay_raw_image_w_dep = cv2.addWeighted( generated_sim_image_grey_to_rgb, 1.5, raw_real_image, 0.75, 0)
    overlay_raw_image_w_dep = cv2.resize(overlay_raw_image_w_dep, display_size)

    denoise_img = np.concatenate([raw_real_image, overlay_raw_image_w_dep,overlay_raw_image_w_sal, gau_overlay_raw_image_w_sal ], axis=1)

    cv2.imshow("denoise_img", denoise_img)
    
    k = cv2.waitKey(10)
    if k==27:    # Esc key to stop

        if record_video:
            video_file = os.path.join('gan_videos', 'tactile_video.mp4')
            imageio.mimwrite(video_file, np.stack(video_frames), fps=20)

        break