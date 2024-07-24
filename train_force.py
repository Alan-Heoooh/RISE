import os
import torch
import numpy as np
import pickle
import argparse
import matplotlib.pyplot as plt
import torch.nn as nn
from copy import deepcopy
from tqdm import tqdm
from einops import rearrange
from easydict import EasyDict as edict
import time

from torch.utils.data import DataLoader
from dataset.force import ForceDataset, collate_fn
from utils.training import set_seed, plot_history, sync_loss
from policy.policy import MLPDiffusion

default_arg = edict({
    'ckpt_dir': '/data/zihao/ckpt/test',
    'dataset_root': '/data/RH20T',
    'policy_class': 'Diffusion',
    'task_name': 'task_0215',
    'batch_size': 256,
    'seed': 1,
    'num_epochs': 1000,
    'save_epochs': 50,
    'lr': 1e-5,
    'resume_ckpt': None,
    'resume_epoch': -1,
    'num_action': 20,
    'num_obs': 100,
})

def main(args_override):
    args = deepcopy(default_arg)
    for k, v in args_override.items():
        args[k] = v

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # dataset and dataloader
    selected_tasks=[args.task_name]

    train_dataset = ForceDataset(
        root=args.dataset_root,
        offset_path=args.offset_path,
        split='train',
        num_obs=args.num_obs,
        horizon=args.num_obs + args.num_action,
        selected_tasks=selected_tasks,
        frame_sample_step=1,
        selected_cam_ids=['750612070851', '035622060973'],
    )
    # val_dataset = RH20TDataset(
    #     root=args.dataset_root,
    #     task_config_list=task_config_list,
    #     split='val',
    #     num_input=1,
    #     horizon=1+args.num_action,
    #     selected_tasks=selected_tasks
    # )
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=20, collate_fn=collate_fn)
    # val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=100, collate_fn=collate_fn)

    policy = MLPDiffusion(
        num_action = args.num_action,
        num_obs = args.num_obs,
        obs_dim = 6,
        obs_feature_dim = args.obs_feature_dim,
        action_dim = 10,
        dropout = 0.1
    ).to(device)

    n_parameters = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print("Number of parameters: {:.2f}M".format(n_parameters / 1e6))

    if args.resume_ckpt is not None:
        policy.module.load_state_dict(torch.load(args.resume_ckpt, map_location = device), strict = False)
        print("Checkpoint {} loaded.".format(args.resume_ckpt))

    # ckpt path
    if  not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)

    # optimizer and lr scheduler
    print("Loading optimizer and scheduler ...")
    optimizer = torch.optim.AdamW(policy.parameters(), lr = args.lr, betas = [0.95, 0.999], weight_decay = 1e-6)

    # training loop
    train_history = []

    policy.train()
    for epoch in range(args.resume_epoch + 1, args.num_epochs):
        print("Epoch {}".format(epoch))
        optimizer.zero_grad()
        num_steps = len(train_dataloader)
        pbar = tqdm(train_dataloader) 
        avg_loss = 0

        for data in pbar:
            force_data = data['input_frame_list'].to(device)
            # action_data = data['target_frame_tcp_normalized'].to(device)
            offset_data = data['target_offset_list_normalized'].to(device)
            # offset_data = data['target_offset_list'].to(device)

            # print(force_data.shape)
            # forward
            loss = policy(force_data, offset_data)

            # backward
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            avg_loss += loss.item()

        avg_loss /= num_steps
        sync_loss(avg_loss, device)
        train_history.append(avg_loss)
        print("Train loss: {:.6f}".format(avg_loss))
        if (epoch + 1) % args.save_epochs == 0:
            torch.save(
                # policy.module.state_dict(),
                policy.state_dict(),
                os.path.join(args.ckpt_dir, "policy_epoch_{}_seed_{}.ckpt".format(epoch + 1, args.seed))
            )
            plot_history(train_history, epoch, args.ckpt_dir, args.seed)

    torch.save(
        # policy.module.state_dict(),
        policy.state_dict(),
        os.path.join(args.ckpt_dir, "policy_last.ckpt")
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--dataset_root', action='store', type=str, help='dataset_root', required=True)
    parser.add_argument('--offset_path', action='store', type=str, help='offset_path', required=True)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', required=True)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=True)
    parser.add_argument('--save_epochs', action='store', type=int, help='save frequency (epoch)', default=10, required=False)
    parser.add_argument('--lr', action='store', type=float, help='lr', required=True)
    parser.add_argument('--resume_ckpt', action='store', type=str, help='checkpoint to resume training', default=None, required=False)
    parser.add_argument('--resume_epoch', action = 'store', type = int, help = 'resume from which epoch', required = False, default = -1)
    parser.add_argument('--num_action', action='store', type=int, help='num_action', required=False)
    parser.add_argument('--num_obs', action='store', type=int, help='num_obs', required=False)
    parser.add_argument('--obs_feature_dim', action='store', type=int, help='obs_feature_dim', required=False)

    
    main(vars(parser.parse_args()))