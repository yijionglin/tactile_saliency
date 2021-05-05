import os
import re
import cv2
import imageio

directory = '/home/alex/Documents/tactile_gym_sim2real/tactile_gym_sim2real/pix2pix/saved_models/[edge_2d,surface_3d]/256x256_[tap]_250epochs/images'

dir_list = os.path.normpath(directory).split(os.sep)
model_name = dir_list[-3] + '_' + dir_list[-2]

image_filenames = []
for filename in os.listdir(directory):
    image_filenames.append(os.path.join(directory, filename))


# natural sorting of names
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

image_filenames.sort(key=natural_keys)

# flip last filename (no training) to first
image_filenames.insert(0, image_filenames.pop())

# cycle through and load each image
images = []
for filename in image_filenames:
    image = cv2.imread(filename)
    images.append(image)


duration = [1]*len(images)
duration[-1] = 20 # pause on last image for 20sec
imageio.mimwrite(os.path.join('example_videos', '{}_GAN_training.gif'.format(model_name)), images, duration=duration)
