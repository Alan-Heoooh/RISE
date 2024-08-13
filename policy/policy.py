import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms
import numpy as np

from policy.tokenizer import Sparse3DEncoder, ForceEncoder
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
        num_obs = 100,
        obs_dim = 6,
        obs_feature_dim = 256,
        action_dim = 8,
        dropout = 0.1
    ):
        super().__init__()
        self.mlp = nn.Sequential(
                    nn.Linear(obs_dim, 64), nn.LayerNorm(64), nn.ReLU(),
                    nn.Linear(64, 128), nn.LayerNorm(128), nn.ReLU(),
                    nn.Linear(128, obs_feature_dim), nn.LayerNorm(obs_feature_dim), nn.ReLU())
        self.action_decoder = DiffusionUNetLowdimPolicy(action_dim, num_action, num_obs, obs_feature_dim)

    def forward(self, force, actions = None):
        if actions is not None:
            loss = self.action_decoder.compute_loss(force, actions)
            return loss
        else:
            with torch.no_grad():
                action_pred = self.action_decoder.predict_action(force)
            return action_pred

# take force and torque as transformer input tokens
class ForceRISE(nn.Module):
    '''
    ForceRISE model
    '''
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
        dropout = 0.1,
        num_obs_force = 100
    ):
        super().__init__()
        num_obs = 1
        self.num_obs_force = num_obs_force
        self.sparse_encoder = Sparse3DEncoder(input_dim, obs_feature_dim)
        self.force_encoder = ForceEncoder(num_obs_force, input_dim, obs_feature_dim)
        self.transformer = Transformer(hidden_dim, nheads, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout)
        self.action_decoder = DiffusionUNetPolicy(action_dim, num_action, num_obs, obs_feature_dim)
        self.readout_embed = nn.Embedding(1, hidden_dim)

    def forward(self, force_torque, cloud, actions = None, batch_size = 24):
        src, pos, src_padding_mask = self.sparse_encoder(cloud, batch_size=batch_size)
        force_torque_feature, force_torque_pos, force_torque_padding_mask = self.force_encoder(force_torque, batch_size=batch_size)
    
        combined_src = torch.cat([src, force_torque_feature], dim=1)
        combined_pos = torch.cat([pos, force_torque_pos], dim=1)
        combined_padding_mask = torch.cat([src_padding_mask, force_torque_padding_mask], dim=1)

        readout = self.transformer(combined_src, combined_padding_mask, self.readout_embed.weight, combined_pos)[-1]
        readout = readout[:, 0]
        if actions is not None:
            loss = self.action_decoder.compute_loss(readout, actions)
            return loss
        else:
            with torch.no_grad():
                action_pred = self.action_decoder.predict_action(readout)
            return action_pred

# two Transfromer layers, one for force, one for Sparse 3D
class ForceRISE2(nn.Module):
    '''
    ForceRISE2 model
    Two Transformer layers, one for force, one for Sparse 3D.
    Concat readout from two Transformer layers, then feed into action decoder (diffusion model).
    '''
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
            dropout = 0.1,
            num_obs_force = 100
            ):
        super().__init__()
        num_obs = 1
        self.num_obs_force = num_obs_force
        self.sparse_encoder = Sparse3DEncoder(input_dim, obs_feature_dim)
        self.force_encoder = ForceEncoder(num_obs_force, input_dim, obs_feature_dim)
        self.sparse_transformer = Transformer(hidden_dim, nheads, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout)
        self.force_transformer = Transformer(hidden_dim, nheads, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout)
        self.action_decoder = DiffusionUNetPolicy(action_dim, num_action, num_obs, obs_feature_dim * 2) # concat readout from two transformer layers
        self.sparse_readout_embed = nn.Embedding(1, hidden_dim)
        self.force_readout_embed = nn.Embedding(1, hidden_dim)

    def forward(self, force_torque, cloud, actions = None, batch_size = 24):
        src, pos, src_padding_mask = self.sparse_encoder(cloud, batch_size=batch_size)
        force_torque_feature, force_torque_pos, force_torque_padding_mask = self.force_encoder(force_torque, batch_size=batch_size)

        sparse_readout = self.sparse_transformer(src, src_padding_mask, self.sparse_readout_embed.weight, pos)[-1]
        sparse_readout = sparse_readout[:, 0]
        # print(sparse_readout.shape)
        force_readout = self.force_transformer(force_torque_feature, force_torque_padding_mask, self.force_readout_embed.weight, force_torque_pos)[-1]
        force_readout = force_readout[:, 0]
        # print(force_readout.shape)

        combined_readout = torch.cat([sparse_readout, force_readout], dim=1)

        if actions is not None:
            loss = self.action_decoder.compute_loss(combined_readout, actions)
            return loss
        else:
            with torch.no_grad():
                action_pred = self.action_decoder.predict_action(combined_readout)
            return action_pred


# two Transfromer layers, one for force, one for Sparse 3D. When force data is small, apply another embedding directly to the diffusion policy.
class ForceRISE3(nn.Module):
    '''
    ForceRISE2 model
    Two Transformer layers, one for force, one for Sparse 3D.
    Concat readout from two Transformer layers, then feed into action decoder (diffusion model).
    '''
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
            dropout = 0.1,
            num_obs_force = 200
            ):
        super().__init__()
        num_obs = 1
        self.num_obs_force = num_obs_force
        self.sparse_encoder = Sparse3DEncoder(input_dim, obs_feature_dim)
        self.force_encoder = ForceEncoder(num_obs_force, input_dim, obs_feature_dim)
        self.sparse_transformer = Transformer(hidden_dim, nheads, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout)
        self.force_transformer = Transformer(hidden_dim, nheads, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout)
        self.action_decoder = DiffusionUNetPolicy(action_dim, num_action, num_obs, obs_feature_dim * 2) # concat readout from two transformer layers
        self.sparse_readout_embed = nn.Embedding(1, hidden_dim)
        self.force_readout_embed = nn.Embedding(1, hidden_dim)
        self.flat_embed = nn.Embedding(1, hidden_dim)

    def forward(self, force_torque, force_torque_window_std, cloud, actions = None, batch_size = 24):
        src, pos, src_padding_mask = self.sparse_encoder(cloud, batch_size=batch_size)
        force_torque_feature, force_torque_pos, force_torque_padding_mask = self.force_encoder(force_torque, num_obs_feature=self.num_obs_force, batch_size=batch_size)

        sparse_readout = self.sparse_transformer(src, src_padding_mask, self.sparse_readout_embed.weight, pos)[-1]
        sparse_readout = sparse_readout[:, 0]
        if np.max(force_torque_window_std) > 3:
            force_readout = self.force_transformer(force_torque_feature, force_torque_padding_mask, self.force_readout_embed.weight, force_torque_pos)[-1]
            force_readout = force_readout[:, 0]
        else:
            force_readout = self.flat_embed.weight
        # print(force_readout.shape)

        combined_readout = torch.cat([sparse_readout, force_readout], dim=1)

        if actions is not None:
            loss = self.action_decoder.compute_loss(combined_readout, actions)
            return loss
        else:
            with torch.no_grad():
                action_pred = self.action_decoder.predict_action(combined_readout)
            return action_pred