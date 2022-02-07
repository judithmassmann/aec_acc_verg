import numpy as np
import matplotlib.pyplot as plt
from math import atan, pi
from utils.auxiliary import dist_to_angle

plt.rcParams.update({'font.size': 18})


def plot_observations(img_left, img_right, texture_dist=None, camera_angle=None, mse=None):
    img_left_3d = np.reshape(img_left, img_left.shape + (1, ))
    img_right_3d = np.reshape(img_right, img_right.shape + (1, ))
    img_center_3d = (img_left_3d + img_right_3d) / 2.0
    img_stereo = np.concatenate((img_left_3d, img_center_3d, img_right_3d), axis=2)
    img_stereo = (img_stereo+1)/2
    fig, ax = plt.subplot_mosaic([
        ['original_left', 'original_stereo', 'original_right'],
    ], figsize=(10, 6), constrained_layout=True)
    ax['original_left'].imshow(img_left, 
                               cmap='gray', vmin=-1, vmax=1)
    ax['original_left'].set_title('left')
    ax['original_left'].axis('off')
    ax['original_stereo'].imshow(img_stereo)
    ax['original_stereo'].set_title('stereo')
    ax['original_stereo'].axis('off')
    ax['original_right'].imshow(img_right, 
                                cmap='gray', vmin=-1, vmax=1)
    ax['original_right'].set_title('right')
    ax['original_right'].axis('off')
    
    #fig.suptitle('Distance: %.1f  MSE: %.2e \nAngle: %.2e  Expected angle: %.2e' % (
    #                            texture_dist, mse, camera_angle,
    #                            dist_to_angle(texture_dist)))
    plt.show()
    plt.close()


# for saving the original and reconstructed images
def save_decoded_image(original, reconstruction, name, texture_dist=None, camera_angle=None, mse=None):
    img_left, img_right = original[0, :, :], original[1, :, :]
    img_left_3d = np.reshape(img_left, img_left.shape + (1, ))
    img_right_3d = np.reshape(img_right, img_right.shape + (1, ))
    img_center_3d = (img_left_3d + img_right_3d) / 2.0
    img_stereo = np.concatenate((img_left_3d, img_center_3d, img_right_3d), axis=2)
    original_stereo = (img_stereo+1)/2

    img_left, img_right = reconstruction[0, :, :], reconstruction[1, :, :]
    img_left_3d = np.reshape(img_left, img_left.shape + (1, ))
    img_right_3d = np.reshape(img_right, img_right.shape + (1, ))
    img_center_3d = (img_left_3d + img_right_3d) / 2.0
    img_stereo = np.concatenate((img_left_3d, img_center_3d, img_right_3d), axis=2)
    reconstruction_stereo = (img_stereo+1)/2

    fig, ax = plt.subplot_mosaic([
        ['original_left', 'original_stereo', 'original_right'],
        ['reconstruction_left', 'reconstruction_stereo', 'reconstruction_right']
    ], figsize=(10,10), constrained_layout=True)
    ax['original_left'].imshow(original[0, :, :],
                               cmap='gray', vmin=-1, vmax=1)
    ax['original_left'].set_title('original left')
    ax['original_left'].axis('off')
    ax['original_stereo'].imshow(original_stereo)
    ax['original_stereo'].set_title('stereo')
    ax['original_stereo'].axis('off')
    ax['original_right'].imshow(original[1, :, :],
                                cmap='gray', vmin=-1, vmax=1)
    ax['original_right'].set_title('original right')
    ax['original_right'].axis('off')
    ax['reconstruction_left'].imshow(reconstruction[0, :, :],
                                     cmap='gray', vmin=-1, vmax=1)
    ax['reconstruction_left'].set_title('reconstruction left')
    ax['reconstruction_left'].axis('off')
    ax['reconstruction_stereo'].imshow(reconstruction_stereo)
    ax['reconstruction_stereo'].set_title('stereo')
    ax['reconstruction_stereo'].axis('off')
    ax['reconstruction_right'].imshow(reconstruction[1, :, :],
                                      cmap='gray', vmin=-1, vmax=1)
    ax['reconstruction_right'].set_title('reconstruction right')
    ax['reconstruction_right'].axis('off')
    if texture_dist and camera_angle and mse:
        fig.suptitle('Distance: %.1f  MSE: %.2e \nAngle: %.2e  Expected angle: %.2e' % (
                                texture_dist, mse, camera_angle, dist_to_angle(texture_dist)))
    plt.savefig(name)
    plt.close()


# loss plots
def plot_ae_loss(loss):
    plt.figure(figsize=(10, 7))
    plt.plot(loss, color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Reconstruction loss')
    plt.title('Autoencoder')
    plt.savefig('./results/autoencoder_loss.png')
    plt.close()


def plot_ac_loss(loss):
    plt.figure(figsize=(10, 7))
    plt.plot(loss, color='darkgreen')
    plt.xlabel('Epochs')
    plt.ylabel('ActorCritic loss')
    plt.title('Reinforcement learner')
    plt.savefig('./results/ac_loss.png')
    plt.close()


def plot_rewards(scores):
    plt.figure(figsize=(10, 7))
    plt.plot(scores, color='purple')
    plt.xlabel('Epochs')
    plt.ylabel('Reward')
    plt.title('Reinforcement learner')
    plt.savefig('./results/ac_reward.png')
    plt.close()
