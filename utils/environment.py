import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from scipy import ndimage

import os
import sys
file_path = os.path.abspath(os.path.dirname(__file__))
main_path = os.path.abspath(os.path.join(file_path,'..'))
if main_path != sys.path[0]:
    sys.path.insert(0, main_path)
    os.chdir(main_path)
    del main_path, file_path

from utils.framework import Framework
from utils.auxiliary import get_cropped_image, dist_to_angle, add_gaussian_blur
from utils.plots import plot_observations


class Environment():
    
    def __init__(self, action_to_angle=None, action_to_blur=None, min_texture_dist=50, max_texture_dist=500,
                 min_camera_angle=0, max_camera_angle=4):
        self.framework = Framework(width=256, height=256, fov=0.25)
        self.min_texture_dist = min_texture_dist
        self.max_texture_dist = max_texture_dist
        self.min_camera_angle = min_camera_angle
        self.max_camera_angle = max_camera_angle
        if action_to_angle:
            self.action_to_angle = action_to_angle
        else:
            self.action_to_angle = [-1, -.5, -.25, -.125, 0, .125, .25, .5, 1]
        if action_to_blur:
            self.action_to_blur = action_to_blur
        else:
            #self.action_to_blur = [-.1, -.2, -.3, -.4, -.5, -.6, -.7, -.8, -.9]
            self.action_to_blur = [-.5, -1, -1.5, -2, -2.5, -3, -3.5, -4, -4.5]

    def reset_camera(self, initial_camera_angle=0):
        self.framework.reset_camera(initial_camera_angle)

    def new_background(self, background_file):
        self.framework.new_background(background_file)

    def new_episode(self, texture_dist=None, texture_file=None):
        if texture_dist is None:
            texture_dist = self.min_texture_dist + (self.max_texture_dist-self.min_texture_dist)*np.random.random()
        if self.framework.texture_node:
            self.framework.remove_texture()
        if texture_file:
            self.framework.change_texture(texture_file)
        self.framework.add_texture(texture_dist)

    def perform_action_angle(self, action):
        angle = self.action_to_angle[action]
        self.framework.move_camera(angle,
                                   min_camera_angle=self.min_camera_angle,
                                   max_camera_angle=self.max_camera_angle)

    def perform_action_blur(self, img_left_fine, img_left_coarse, img_right_fine, img_right_coarse, action):  # Edit gaussian blur
        blur = self.action_to_blur[action]  # Edit gaussian blur
        img_left_fine, img_left_coarse = add_gaussian_blur(img_left_fine, img_left_coarse, blur)  # Edit gaussian blur
        img_right_fine, img_right_coarse = add_gaussian_blur(img_right_fine, img_right_coarse, blur)  # Edit gaussian blur
        return img_left_fine, img_left_coarse, img_right_fine, img_right_coarse

    def get_observations(self):
        img_left, img_right = self.framework.render_scene()
        img_left_fine, img_left_coarse = get_cropped_image(img_left)
        img_right_fine, img_right_coarse = get_cropped_image(img_right)
        #plot_observations(img_left_coarse, img_right_coarse)  # Edit verg plotten lassen
        return img_left_fine, img_left_coarse, img_right_fine, img_right_coarse


if __name__ == '__main__':

    texture_dist = 500
    environment = Environment()
    environment.new_episode(texture_dist=texture_dist, texture_file='data/texture0.jpg')

    img_left_fine, img_left_coarse, img_right_coarse, img_right_coarse = environment.get_observations()
    mse = mean_squared_error(img_left_coarse, img_right_coarse)
    #img_left_fine, img_left_coarse, img_right_coarse, img_right_coarse = environment.perform_action_blur(img_left_fine, img_left_coarse, img_right_coarse, img_right_coarse, 1)
    img_left_coarse, img_right_coarse = add_gaussian_blur(img_left_coarse, img_right_coarse, 1)
    #img_left = ndimage.gaussian_filter(img_left_coarse, 1)
    plot_observations(img_left_coarse, img_right_coarse, texture_dist=texture_dist,
                      camera_angle=environment.framework.camera_angle, mse=mse)
    img_left_coarse, img_right_coarse = add_gaussian_blur(img_left_coarse, img_right_coarse, -1)
    plot_observations(img_left_coarse, img_right_coarse, texture_dist=texture_dist,
                      camera_angle=environment.framework.camera_angle, mse=mse)
    #img_left = ndimage.gaussian_filter(img_left_coarse, -1)
    #plot_observations(img_left, img_right_coarse, texture_dist=texture_dist,
    #                  camera_angle=environment.framework.camera_angle, mse=mse)
'''
    camera_angle = dist_to_angle(texture_dist=texture_dist)
    environment.framework.move_camera(camera_angle+1)
    _, img_left_coarse, _, img_right_coarse = environment.get_observations()
    mse = mean_squared_error(img_left_coarse, img_right_coarse)
    plot_observations(img_left_coarse, img_right_coarse, texture_dist=texture_dist,
                      camera_angle=environment.framework.camera_angle, mse=mse)
'''