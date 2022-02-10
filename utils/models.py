import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import namedtuple, deque

def init_weights(layer):
    if isinstance(layer, nn.Linear):
        nn.init.xavier_normal_(layer.weight)
    elif isinstance(layer, nn.Conv2d):
        nn.init.xavier_normal_(layer.weight)
    elif isinstance(layer, nn.ConvTranspose2d):
        nn.init.xavier_normal_(layer.weight)


# --- Autoencoder ---

class Autoencoder(nn.Module):

    """
    def __init__(self):
        super().__init__()

        ### Encoder
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=(3,3)),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=(3,3)),
            nn.ReLU(),
        )
        self.encoder_cnn.apply(init_weights)

        ### Linear section (encoder)
        self.encoder_lin = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(16 * 28 * 28, 256),
            nn.ReLU(True),
        )
        self.encoder_lin.apply(init_weights)
        
        ### Linear section (decoder)
        self.decoder_lin = nn.Sequential(
            nn.Linear(256, 16 * 28 * 28),
            nn.ReLU(True),
            nn.Unflatten(dim=1, unflattened_size=(16, 28, 28))
        )
        self.decoder_lin.apply(init_weights)

        ### Decoder
        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size=(3,3)),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 2, kernel_size=(3,3)),
            nn.Tanh()
        )
        self.decoder_cnn.apply(init_weights)

    def forward(self, x):
        enc = self.encoder_cnn(x)
        enc = self.encoder_lin(enc)
        dec = self.decoder_lin(enc)
        dec = self.decoder_cnn(dec)
        return enc, dec
    """

    def __init__(self):
        super().__init__()

        ### Encoder
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=(8, 8), stride=4),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=(1, 1)),
            nn.ReLU(),
        )
        self.encoder_cnn.apply(init_weights)

        ### Decoder
        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size=(3, 3), stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 2, kernel_size=(4, 4), stride=2),
            nn.Tanh()
        )
        self.decoder_cnn.apply(init_weights)

    def forward(self, x):
        enc = self.encoder_cnn(x)
        dec = self.decoder_cnn(enc)
        return enc, dec


# define the sparse loss function on encoded images
def sparse_loss(encoding):
    loss = torch.mean(torch.abs(encoding))
    return loss


# --- Actor Critic ---


class ActorCritic(nn.Module):
    def __init__(self, n_actions):
        super(ActorCritic, self).__init__()
        
        self.convpool = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=(2, 2), stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        )
        #self.convpool.apply(init_weights)

        self.flatten = nn.Flatten(start_dim=1)

        self.fc = nn.Sequential(
            nn.Linear(144, 200),
            nn.ReLU()
        )
        #self.fc.apply(init_weights)

        self.policy_head = nn.Sequential(
            nn.Linear(200, n_actions)
        )
        #self.policy_head.apply(init_weights)

        self.critic_head = nn.Sequential(
            nn.Linear(200, 1)
        )
        #self.critic_head.apply(init_weights)

    def forward(self, x1, x2):

        x1 = self.convpool(x1)
        x2 = self.convpool(x2)
        x1 = self.flatten(x1)
        x2 = self.flatten(x2)
        x = torch.cat([x1, x2], dim=1)
        
        policy = self.fc(x)
        policy = self.policy_head(policy)

        critic = self.fc(x)
        critic = self.critic_head(critic)

        return policy, critic
