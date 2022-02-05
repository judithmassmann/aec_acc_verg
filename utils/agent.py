import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.models import Autoencoder, sparse_loss, ActorCritic
from utils.buffer import Buffer
from utils.plots import save_decoded_image
from utils.auxiliary import get_device


class Agent():
    def __init__(self, img_size=32, n_actions=9,
                ae_learning_rate=1e-3, ae_sparse_lambda=0,
                gamma=0.6, encoding_dims=512,
                learning_rate=1e-3, weight_decay=5e-06,
                minibatch_size=40, batch_size=200, buffer_size=1000,
                epsilon=0.01, epsilon_min=0.01, epsilon_dec=1e-3
    ):
        self.device = get_device()
        # autoencoder
        self.autoencoder = Autoencoder().float()
        self.ae_sparse_lambda = ae_sparse_lambda
        self.ae_learning_rate = ae_learning_rate
        self.ae_criterion = nn.MSELoss()
        self.ae_optimizer = optim.Adam(self.autoencoder.parameters(), lr=self.ae_learning_rate)
        # reinforcement learner
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_dec = epsilon_dec
        self.learning_rate = learning_rate
        self.action_space = [i for i in range(n_actions)]
        self.minibatch_size = minibatch_size
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.buffer = Buffer()
        self.buffer_counter = 0
        self.actor_critic = ActorCritic(n_actions)
        self.ac_optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.learning_rate)
        


    # --- Autoencoder ---

    # get one encoding for both scales concatenated:
    def get_encoding(self, observation):
        observation = np.reshape(observation, (1, 2, 32, 32))
        observation = torch.from_numpy(observation).float()
        encoding, reconstruction = self.autoencoder(observation)
        encoding = torch.reshape(encoding, (16, 7, 7))
        loss = self.ae_criterion(reconstruction, observation).item()
        return encoding, loss


    def choose_action(self, encoding_fine, encoding_coarse):
        if np.random.random() > self.epsilon:
            encoding_fine = torch.reshape(encoding_fine, (1, 16, 7, 7))
            encoding_coarse = torch.reshape(encoding_coarse, (1, 16, 7, 7))
            policy, _ = self.actor_critic.forward(encoding_fine, encoding_coarse)
            probabilities = torch.softmax(policy, dim=1)
            distribution = Categorical(probabilities)
            action = distribution.sample().numpy()[0]
        else:
            action = np.random.choice(self.action_space)
        return action

    def train_autoencoder(self, buffer, epoch):
        #buffer_size = len(buffer.reward)
        #if buffer_size < self.batch_size:
        #    return None
           
        self.autoencoder.train()
                
        observations_fine = torch.from_numpy(buffer.observation_fine)
        observations_fine = np.reshape(observations_fine, (observations_fine.size(0), 2, 32, 32)).float()
        encodings_fine, reconstructions_fine = self.autoencoder(observations_fine)
        mse_loss_fine = self.ae_criterion(reconstructions_fine, observations_fine)
        l1_loss_fine = sparse_loss(encodings_fine)

        observations_coarse = torch.from_numpy(buffer.observation_coarse)
        observations_coarse = np.reshape(observations_coarse, (observations_coarse.size(0), 2, 32, 32)).float()
        encodings_coarse, reconstructions_coarse = self.autoencoder(observations_coarse)
        mse_loss_coarse = self.ae_criterion(reconstructions_coarse, observations_coarse)
        l1_loss_coarse = sparse_loss(encodings_coarse)

        self.ae_optimizer.zero_grad()
        loss = (mse_loss_fine + mse_loss_coarse).mean() + self.ae_sparse_lambda*(l1_loss_fine+l1_loss_coarse).mean()
        loss.backward()
        self.ae_optimizer.step()

#        if epoch <= 100 or (epoch+1)%100 == 0:
#            save_decoded_image(observations_coarse[-1, :, :, :].detach().numpy(),
#                               reconstructions_coarse[-1, :, :, :].detach().numpy(),
#                               f"./results/images/{epoch}.png", 500)
        epoch_loss = loss.item()
        return epoch_loss, observations_coarse, reconstructions_coarse  # Edit verg nur ersteres

    def train_actor_critic(self, buffer, epoch):
                  
        self.autoencoder.train()

        encodings_fine = torch.from_numpy(buffer.encoding_fine)
        encodings_coarse = torch.from_numpy(buffer.encoding_coarse)
        actions = torch.from_numpy(buffer.action)
        rewards = torch.from_numpy(buffer.reward)

        policies, values = self.actor_critic.forward(encodings_fine, encodings_coarse)
        values = values.squeeze()

        ret = 0
        returns = []
        for idx in range(len(rewards)):
            r = rewards[-(idx+1)]
            ret = r + self.gamma * ret
            returns.insert(0, ret)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-6)
        
        critic_loss = ((returns - values)**2).mean()

        probabilities = torch.softmax(policies, dim=1)
        distribution = Categorical(probabilities)
        log_probabilities = distribution.log_prob(actions)
        actor_loss = (- log_probabilities * returns-values).mean()

        self.ac_optimizer.zero_grad()
        loss = actor_loss + 0.5*critic_loss
        loss.backward()
        self.ac_optimizer.step()

        if (self.epsilon - self.epsilon_dec) > self.epsilon_min:
            self.epsilon = self.epsilon - self.epsilon_dec
        else:
            self.epsilon = self.epsilon_min
        
        epoch_loss = loss.item()
        return epoch_loss
