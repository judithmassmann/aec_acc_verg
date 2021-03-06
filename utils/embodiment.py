import time
import random
import torch
import numpy as np
from sklearn.metrics import mean_squared_error
from multiprocessing import Pool
from scipy import ndimage
import matplotlib.pyplot as plt

import os
import sys
file_path = os.path.abspath(os.path.dirname(__file__))
main_path = os.path.abspath(os.path.join(file_path, '..'))
if main_path != sys.path[0]:
    sys.path.insert(0, main_path)
    os.chdir(main_path)
    del main_path, file_path

from utils.agent import Agent
from utils.environment import Environment
from utils.buffer import Buffer
from utils.auxiliary import img_to_obs, dist_to_angle, add_gaussian_blur, filt2D


def run_embodiment(name, textures, autoencoder_params, actor_critic_params, n_episodes):
    embodiment = Embodiment(name, textures, autoencoder_params, actor_critic_params, n_episodes=n_episodes)
    buffer = embodiment.run()
    return buffer


class Embodiment():
    def __init__(self, name, textures, autoencoder_params=None, actor_critic_params=None, epsilon=1.0,
                 n_actions=9, n_episodes=5, n_timesteps=10):
        super(Embodiment, self).__init__()

        self.name = 'proc-%02d' % name
        self.environment = Environment()
        self.buffer = Buffer()
        self.agent = Agent()
        self.n_timesteps = n_timesteps
        self.n_episodes = n_episodes
        if autoencoder_params:
            self.agent.autoencoder.load_state_dict(autoencoder_params)
        if actor_critic_params:
            self.agent.actor_critic.load_state_dict(actor_critic_params)
        if textures:
            self.textures = random.sample(textures, len(textures))
        self.reward_c = 1
        self.min_texture_dist = 50
        self.max_texture_dist = 500

    def run(self, print_time=False):
        
        score = 0
        self.buffer.clear()

        for episode_idx in range(self.n_episodes):
            start = time.time()
            running_reward = 0
            
            texture_file = self.textures[episode_idx % len(self.textures)]
            texture_dist = self.min_texture_dist + (self.max_texture_dist-self.min_texture_dist)*1  # Edit verg off *np.random.random()
            initial_camera_angle = dist_to_angle(texture_dist)  # Edit verg off (...)- 1 + 2*np.random.random()
            self.environment.reset_camera(initial_camera_angle)
            self.environment.new_episode(texture_dist=texture_dist, texture_file=texture_file)

            img_left_fine, img_left_coarse, img_right_fine, img_right_coarse = self.environment.get_observations()

            blur = 1#np.random.random()  # Edit gaussian blur
            img_left_fine, img_left_coarse = add_gaussian_blur(
                img_left_fine, img_left_coarse, blur)  # Edit gaussian blur
            img_right_fine, img_right_coarse = add_gaussian_blur(
                img_right_fine, img_right_coarse, blur)  # Edit gaussian blur

            observation_fine = img_to_obs(img_left_fine, img_right_fine)
            observation_coarse = img_to_obs(img_left_coarse, img_right_coarse)
            
            for _ in range(self.n_timesteps):
                encoding_fine, reconstruction_loss_fine = self.agent.get_encoding(observation_fine)
                encoding_coarse, reconstruction_loss_coarse = self.agent.get_encoding(observation_coarse)
                action = self.agent.choose_action(encoding_fine, encoding_coarse)
                #self.environment.perform_action_angle(action)  # Edit verg off

                img_left_fine, img_left_coarse, img_right_fine, img_right_coarse = self.environment.get_observations()

                new_img_left_fine, new_img_left_coarse, new_img_right_fine, new_img_right_coarse = self.environment.perform_action_blur(
                    img_left_fine, img_left_coarse, img_right_fine, img_right_coarse, action)  # Edit action gaussian blur
                new_observation_fine = img_to_obs(new_img_left_fine, new_img_right_fine)
                new_observation_coarse = img_to_obs(new_img_left_coarse, new_img_right_coarse)
                new_encoding_fine, new_reconstruction_loss_fine = self.agent.get_encoding(new_observation_fine)
                new_encoding_coarse, new_reconstruction_loss_coarse = self.agent.get_encoding(new_observation_coarse)

                if _ == 0:
                    plt.imshow(img_left_fine, cmap='gray', vmin=-1, vmax=1)
                    plt.show()
                    plt.imshow(new_img_left_fine, cmap='gray', vmin=-1, vmax=1)
                    plt.show()

                # Reward as improvement in reconstruction
                # :
                #reward = self.reward_c * (reconstruction_loss_fine + reconstruction_loss_coarse - (
                #                     new_reconstruction_loss_fine + new_reconstruction_loss_coarse))
                
                # Reward as reconstruction loss:
                #reward = - self.reward_c * (new_reconstruction_loss_fine + new_reconstruction_loss_coarse)
# Edit gaussian blur, old reward for vergence:
#                # Reward as improvement in MSE:
#                mse_fine = mean_squared_error(observation_fine[0, :, :], observation_fine[1, :, :])
#                mse_coarse = mean_squared_error(observation_coarse[0, :, :], observation_coarse[1, :, :])
#                mse = (mse_
                #
                #                fine+mse_coarse)/2
#                new_mse_fine = mean_squared_error(new_observation_fine[0, :, :], new_observation_fine[1, :, :])
#                new_mse_coarse = mean_squared_error(new_observation_coarse[0, :, :], new_observation_coarse[1, :, :])
#                new_mse = (new_mse_fine+new_mse_coarse)/2
#                #reward = mse - new_mse
#
#                # Reward as MSE:
#                reward = - new_mse

                # Edit gaussian blur new reward see eckmann et al.:

                im_l_f = np.fft.fft2(observation_fine[0, :, :])
                im_r_f = np.fft.fft2(observation_fine[1, :, :])
                im_l_c = np.fft.fft2(observation_coarse[0, :, :])
                im_r_c = np.fft.fft2(observation_coarse[1, :, :])
                imagew_l_f = np.real(np.fft.ifft2(np.multiply(im_l_f, np.fft.fftshift(filt2D()))))
                imagew_r_f = np.real(np.fft.ifft2(np.multiply(im_r_f, np.fft.fftshift(filt2D()))))
                imagew_l_c = np.real(np.fft.ifft2(np.multiply(im_l_c, np.fft.fftshift(filt2D()))))
                imagew_r_c = np.real(np.fft.ifft2(np.multiply(im_r_c, np.fft.fftshift(filt2D()))))
                reward = np.mean([np.mean(imagew_l_f), np.mean(imagew_r_f), np.mean(imagew_l_c), np.mean(imagew_r_c)])
                #print(reward)  # Edit gaussian blur

                running_reward += reward
                self.buffer.store(observation_fine, observation_coarse, new_observation_fine, new_observation_coarse,
                                  encoding_fine, encoding_coarse, new_encoding_fine, new_encoding_coarse,
                                  action, reward)
                
                observation_fine = new_observation_fine
                observation_coarse = new_observation_coarse

            score += running_reward / self.n_timesteps

            end = time.time()
            #print('Episode', episode_idx,'of', self.name, 'completed in %.1f seconds' %(end-start),
            #        '??? AE loss: %.2e, Score: %.2e' % (new_reconstruction_loss_coarse, running_reward/self.n_timesteps))

        return self.buffer, score / self.n_episodes


if __name__ == '__main__':
    
    texture = ['data/texture0.jpg']
    embodiment = Embodiment(name=0,
                            textures=texture,
                            n_episodes=5,
                            n_timesteps=10,
                            )
    embodiment.run()

