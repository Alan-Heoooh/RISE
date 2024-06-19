import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms

from policy.tokenizer import Sparse3DEncoder
from policy.transformer import Transformer
from policy.diffusion import DiffusionUNetPolicy, DiffusionUNetLowdimPolicy


class RISE(nn.Module):
    def __init__(
        self, 
        num_action = 20,
        input_dim = 6,
        obs_feature_dim = 512, 
        action_dim = 10, 
        hidden_dim = 512,
        nheads = 8, 
        num_encoder_layers = 4, 
        num_decoder_layers = 1, 
        dim_feedforward = 2048, 
        dropout = 0.1
    ):
        super().__init__()
        num_obs = 1
        self.sparse_encoder = Sparse3DEncoder(input_dim, obs_feature_dim)
        self.transformer = Transformer(hidden_dim, nheads, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout)
        self.action_decoder = DiffusionUNetPolicy(action_dim, num_action, num_obs, obs_feature_dim)
        self.readout_embed = nn.Embedding(1, hidden_dim)

    def forward(self, cloud, actions = None, batch_size = 24):
        src, pos, src_padding_mask = self.sparse_encoder(cloud, batch_size=batch_size)
        readout = self.transformer(src, src_padding_mask, self.readout_embed.weight, pos)[-1]
        readout = readout[:, 0]
        if actions is not None:
            loss = self.action_decoder.compute_loss(readout, actions)
            return loss
        else:
            with torch.no_grad():
                action_pred = self.action_decoder.predict_action(readout)
            return action_pred
        

class MLPDiffusion(nn.Module):
    def __init__(
        self,
        num_action = 20,
        hidden_dim = [16, 32],
        obs_dim = 6,
        obs_feature_dim = 64,
        action_dim = 10,
        dropout = 0.1
    ):
        super().__init__()
        num_obs = 1
        self.fc1 = nn.Linear(obs_dim, hidden_dim[0])  # input layer (6) -> hidden layer (16)
        self.fc2 = nn.Linear(hidden_dim[0], hidden_dim[1])  # hidden layer (16) -> hidden layer (32)
        self.fc3 = nn.Linear(hidden_dim[1], obs_feature_dim)  # hidden layer (32) -> output layer (64)
        self.action_decoder = DiffusionUNetLowdimPolicy(action_dim, num_action, num_obs, obs_feature_dim)

    def forward(self, force, actions = None):
        # force: [batch_size, 6]
        force = torch.relu(self.fc1(force))  # activation function for hidden layer
        force = torch.relu(self.fc2(force))  # activation function for hidden layer
        force = self.fc3(force)
        # force: [batch_size, obs_feature_dim]
        if actions is not None:
            loss = self.action_decoder.compute_loss(force, actions)
            return loss
        else:
            with torch.no_grad():
                action_pred = self.action_decoder.predict_action(force)
            return action_pred