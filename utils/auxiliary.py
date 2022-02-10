import numpy as np
import os
import torch
from math import atan, pi
from PIL import Image
from sklearn.preprocessing import minmax_scale
from scipy import ndimage


# get the computation device
def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device


# make the `images` directory
def make_dir():
    image_dir = './results/images'
    model_dir = './results/models'
    score_dir = './results/scores'
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(score_dir):
        os.makedirs(score_dir)


def img_to_obs(img_left, img_right):
    observation = np.zeros([2, 32, 32])
    observation[0, :, :] = img_left
    observation[1, :, :] = img_right
    return observation


def dist_to_angle(texture_dist, eye_dist_to_center=3.4):
    return atan(eye_dist_to_center/texture_dist)*180/pi


def get_cropped_image(img):
    fine, coarse = 128, 256
    resize_scale = 32
    pil_img = Image.fromarray(img)
    pil_bw = pil_img.convert('L')  # greyscale
    width, height = pil_bw.size
    area_fine = (width/2-(fine/2-1), height/2-(fine/2-1), width/2+(fine/2+1), height/2+(fine/2+1))
    area_coarse = (width/2-(coarse/2-1), height/2-(coarse/2-1), width/2+(coarse/2+1), height/2+(coarse/2+1))
    crop_img_fine = pil_bw.crop(area_fine)
    resize_img_fine = crop_img_fine.resize([resize_scale, resize_scale])
    img_fine = minmax_scale(resize_img_fine, feature_range=(-1, 1))
    crop_img_coarse = pil_bw.crop(area_coarse)
    resize_img_coarse = crop_img_coarse.resize([resize_scale, resize_scale])
    img_coarse = minmax_scale(resize_img_coarse, feature_range=(-1, 1))
    return img_fine, img_coarse


def add_gaussian_blur(img_left, img_right, blur):  # Edit gaussian blur whitening factor
    img_left = ndimage.gaussian_filter(img_left, blur)
    img_right = ndimage.gaussian_filter(img_right, blur)
    return img_left, img_right


def filt2D():
    window_size = 32
    nx = np.linspace(-window_size / 2, window_size / 2, 32)
    fx, fy = np.meshgrid(nx, nx)
    rho = np.sqrt(fx * fx + fy * fy)
    f_0 = 0.4 * window_size
    phi = 4
    a = 1.24
    return (rho ** a) * np.exp(-(rho / f_0) ** phi)
