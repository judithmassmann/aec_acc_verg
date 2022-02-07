from math import ceil
import time
import copy
import random
import argparse
import torch
import numpy as np
import concurrent.futures
from multiprocessing import Process, Queue, Pool, Manager
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

import os
import sys
file_path = os.path.abspath(os.path.dirname(__file__))
main_path = os.path.abspath(os.path.join(file_path, '..'))
if main_path != sys.path[0]:
    sys.path.insert(0, main_path)
    os.chdir(main_path)
    del main_path, file_path

from utils.environment import Environment
from utils.agent import Agent
from utils.embodiment import Embodiment, run_embodiment
from utils.buffer import Buffer
from utils.plots import save_decoded_image, plot_ae_loss, plot_ac_loss, plot_rewards
from utils.auxiliary import make_dir, img_to_obs


def main():

    n_epochs = 5  # 100000
    n_processes = 1
    n_episodes_per_epoch = 1
    n_timesteps = 10
    encoding_dims = 16*7*7
    ae_sparse_lambda = 0  # 1e-3
    ae_learning_rate = 1e-3
    reward_c = 1 # reward = reward_c * (Lt - Lt+1)
    min_texture_dist = 50   # cm
    max_texture_dist = 500  # cm
    min_camera_angle = 0
    max_camera_angle = 6
    actions_to_angles = [-1, -.5, -.25, -.125, 0, .125, .25, .5, 1]
    n_actions = len(actions_to_angles)

    parser = argparse.ArgumentParser(description='Optional app description')
    parser.add_argument('--checkpoint', type=str,
                        help='Path to file to load pytorch checkpoint')
    args = parser.parse_args()
    
    texture_files = [
        'data/texture0.jpg',
        'data/texture1.jpg',
        'data/texture2.jpg',
        'data/texture3.jpg',
        'data/texture4.jpg',
        'data/texture5.jpg',
        'data/texture6.jpg',
        'data/texture7.jpg',
        'data/texture8.jpg',
        'data/texture9.jpg'
    ]

    train_ae_loss = []
    train_ac_loss = []
    scores = []
    epoch = 0

    make_dir()
    global_agent = Agent(ae_learning_rate=ae_learning_rate, ae_sparse_lambda=ae_sparse_lambda,
                         encoding_dims=encoding_dims, n_actions=n_actions)
    global_buffer = Buffer()
       
    if args.checkpoint:
        print('Checkpoint loaded')
        checkpoint = torch.load(args.checkpoint)
        epoch = checkpoint['epoch']
        global_agent.autoencoder.load_state_dict(checkpoint['ae_state_dict'])
        global_agent.ae_optimizer.load_state_dict(checkpoint['ae_optimizer_state_dict'])
        global_agent.actor_critic.load_state_dict(checkpoint['ac_state_dict'])
        global_agent.ac_optimizer.load_state_dict(checkpoint['ac_optimizer_state_dict'])

    start = time.time()
    while epoch <= n_epochs:
        epoch += 1
        start_epoch = time.time()

        autoencoder_params = global_agent.autoencoder.state_dict()
        actor_critic_params = global_agent.actor_critic.state_dict()
        if n_processes == 1:
            embodiment = Embodiment(name=0,
                                    textures=texture_files,
                                    autoencoder_params=autoencoder_params,
                                    actor_critic_params=actor_critic_params,
                                    n_episodes=n_episodes_per_epoch,
                                    )
            process_buffer, score = embodiment.run()
            train_ac_epoch_loss = global_agent.train_actor_critic(process_buffer, epoch=epoch)
            train_ac_loss.append(train_ac_epoch_loss)
            global_buffer.concat(process_buffer)
            scores.append(score)
            del embodiment
        else:
            with concurrent.futures.ProcessPoolExecutor() as executor:
                processes = []
                process_scores = []
                process_train_ae_loss = []
                process_train_ac_loss = []
                names = np.arange(n_processes)

                for idx in range(n_processes):
                    processes.append(executor.submit(run_embodiment,
                                                     names[idx],
                                                     texture_files,
                                                     autoencoder_params,
                                                     actor_critic_params,
                                                     n_episodes_per_epoch))

                for process in concurrent.futures.as_completed(processes):
                    process_buffer, score = process.result()
                    train_ac_epoch_loss = global_agent.train_actor_critic(process_buffer, epoch=epoch)
                    process_train_ac_loss.append(train_ac_epoch_loss)
                    global_buffer.concat(process_buffer)
                    process_scores.append(score)
                scores.append(sum(process_scores)/n_processes)
                del processes

        # Train autoencoder from a buffer of experiences
        train_ae_epoch_loss = global_agent.train_autoencoder(global_buffer, epoch=epoch)
        train_ae_loss.append(train_ae_epoch_loss)

        # Outputs:
        if epoch <= 100 and epoch % 10 == 0 or epoch <= 1000 and epoch % 100 == 0 or epoch % 1000 == 0:
            plot_ae_loss(train_ae_loss)
            plot_ac_loss(train_ac_loss)
            plot_rewards(scores)

        end_epoch = time.time()
        print(
            '\nEpoch %d completed in %.1f seconds; %d seconds since beginning of experiment' % (
                                            epoch, end_epoch-start_epoch, end_epoch-start),
            '\nAE loss: %.2e' % train_ae_epoch_loss if train_ae_epoch_loss else '\nAE loss: None',
            '  AC loss: %.2e' % train_ac_epoch_loss if train_ac_epoch_loss else '  AC loss: None',
            '  Reward: %.2e' % scores[-1],
        )

        # Save the trained models
        if epoch % 10000 == 0:
            torch.save({
                'epoch': epoch+1,
                'ae_state_dict': global_agent.autoencoder.state_dict(),
                'ae_optimizer_state_dict': global_agent.ae_optimizer.state_dict(),
                'ae_loss': train_ae_epoch_loss,
                'ac_state_dict': global_agent.actor_critic.state_dict(),
                'ac_optimizer_state_dict': global_agent.ac_optimizer.state_dict(),
                'ac_loss': train_ac_epoch_loss,
                }, f'./results/models/model{epoch+1}.pt')
            with open("./results/scores/ae_loss.txt", "w") as f:
                for s in train_ae_loss:
                    f.write(str(s) + "\n")
            with open("./results/scores/ac_loss.txt", "w") as f:
                for s in train_ac_loss:
                    f.write(str(s) + "\n")
            with open("./results/scores/rewards.txt", "w") as f:
                for s in scores:
                    f.write(str(s) + "\n")

    end = time.time()
    print("\nCompleted experiment in %d seconds" % (end-start))


if __name__ == '__main__':
    main()
