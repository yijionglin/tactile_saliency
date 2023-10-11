
from tactile_gym.assets import add_assets_path
from tactile_gym.utils.general_utils import str2bool
import cv2
import numpy as np
import random

def get_gray_img(img_dir):
  img = cv2.imread(img_dir)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  # Add channel axis
  img = img[..., np.newaxis]
  return img

def smaller_img(img, zoom_size_range):

    # print("printing raw")
    # cv2.imshow("img", img)
    # k = cv2.waitKey(1000)
    row,col = img.shape[:2]
    bottom = img[row-2:row,0:col]
    mean = cv2.mean(bottom)[0]
    
    # bordersize = int(np.random.uniform(0,100))
    bordersize = int(np.random.uniform(zoom_size_range[0],zoom_size_range[1]))
    border = cv2.copyMakeBorder(
        img,
        top = bordersize,
        bottom = bordersize,
        left = bordersize,
        right = bordersize,
        borderType = cv2.BORDER_CONSTANT,
        value = [mean,mean,mean]
    )
    
    resize_img = cv2.resize(border,(row,col))
    # print("printing augmented")
    # cv2.imshow("resize_img", resize_img)
    # k = cv2.waitKey(1000)

    # resize_img = resize_img[...,np.newaxis]
    if (np.random.uniform(0,1) >0.5) and (zoom_size_range[0] == 0):
        img = cv2.resize(img, (row,col))
        return img

    else:
        return resize_img

def image_translate(img, trans_range):
    # trans_h,trans_w = np.random.uniform(-20,20,2)
    # print("printing raw")
    # cv2.imshow("img", img)
    # k = cv2.waitKey(1000)

    trans_h,trans_w = np.random.uniform(trans_range[0],trans_range[1],2)
    T_m = np.float32([[1,0,trans_h],[0,1,trans_w]])
    #
    trans_img = cv2.warpAffine(img,T_m,(img.shape[0],img.shape[1]))
    trans_img = trans_img[...,np.newaxis]
    trans_img = cv2.resize(trans_img, (img.shape[0],img.shape[1]))

    # print("printing augmented")
    # cv2.imshow("trans_img", trans_img)
    # k = cv2.waitKey(1000)

    return trans_img

def image_translate_vertical(img, trans_range):
    # trans_h,trans_w = np.random.uniform(-20,20,2)
    # print("printing raw")
    # cv2.imshow("img", img)
    # k = cv2.waitKey(1000)

    trans_h = np.random.uniform(trans_range[0],trans_range[1])
    T_m = np.float32([[1,0,0],[0,1,trans_h]])
    #
    trans_img = cv2.warpAffine(img,T_m,(img.shape[0],img.shape[1]))
    trans_img = trans_img[...,np.newaxis]
    trans_img = cv2.resize(trans_img, (img.shape[0],img.shape[1]))

    # print("printing augmented")
    # cv2.imshow("trans_img", trans_img)
    # k = cv2.waitKey(1000)

    return trans_img


def rotate_img_range(img, rot_range):
    # print("printing raw")
    # cv2.imshow("img", img)
    # k = cv2.waitKey(1000)
    h,w,c = img.shape
    center = (h//2,w//2)
    angle = np.random.uniform(-rot_range,rot_range)
    rot_m = cv2.getRotationMatrix2D(center,int(angle),1)
    rot_img = cv2.warpAffine(img, rot_m, (h,w),)
    # print("printing augmented")
    # cv2.imshow("img", rot_img)
    # k = cv2.waitKey(1000)
    return rot_img

def rotate_img(img):
    h,w,c = img.shape
    center = (h//2,w//2)
    angle = np.random.uniform(-180,180)
    rot_m = cv2.getRotationMatrix2D(center,int(angle),1)
    rot_img = cv2.warpAffine(img, rot_m, (h,w),)
    return rot_img

def rotate_img_angle(img, angle):
    # print("printing raw")
    # cv2.imshow("img", img)
    # k = cv2.waitKey(1000)

    h,w,c = img.shape
    center = (h//2,w//2)
    rot_m = cv2.getRotationMatrix2D(center,int(angle),1)
    rot_img = cv2.warpAffine(img, rot_m, (h,w),)

    # print("printing augmented")
    # cv2.imshow("img", rot_img)
    # k = cv2.waitKey(1000)

    return rot_img

def img_contrast(img, contrast_range):

    if img.dtype != np.uint8:
        raise ValueError('This random brightness should only be applied to uint8 images on a 0-255 scale')

    a1,a2 = contrast_range
    alpha = np.random.uniform(a1,a2)  # Simple contrast control
    new_image = np.clip(alpha*img, 0, 255).astype(np.uint8)

    return new_image


def augment_img(img_dir, zoom_size_range = None, trans_range = None, trans_ver_range = None, if_rdn_rot = None,  if_rdn_rot_range = None, contrast_range = None ):
    # get img dir and output gray image with shape 128x128x1
    img = get_gray_img(img_dir)
    # p_img = img.copy()
    if if_rdn_rot is not None:
        img = rotate_img(img)
    if if_rdn_rot_range is not None:
        img = rotate_img_range(img, if_rdn_rot_range)
    if contrast_range is not None:
        img = img_contrast(img, contrast_range)
    if trans_range is not None:
        img = image_translate(img, trans_range)
    if trans_ver_range is not None:
        img = image_translate_vertical(img, trans_ver_range)
    if zoom_size_range is not None:
        img = smaller_img(img, zoom_size_range)


    img = img[...,np.newaxis]
    return img