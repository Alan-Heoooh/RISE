import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from torch import nn, optim
from tqdm import tqdm

import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from dataset.realworld import RealWorldDataset, collate_fn
from utils.training import set_seed, plot_history, sync_loss

class ResNetBinaryClassifier(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super(ResNetBinaryClassifier, self).__init__()
        self.resnet = resnet18(pretrained=pretrained)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.resnet(x)
        x = self.sigmoid(x)
        return x

def main(args):
    ckpt_dir = args['ckpt_dir']
    dataset_root = args['dataset_root']
    batch_size = args['batch_size']
    seed = args['seed']
    num_epochs = args['num_epochs']
    save_epoch = args['save_epoch']
    lr = args['lr']
    resume_ckpt = args['resume_ckpt']

    # set seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load dataset
    dataset = RealWorldDataset(
        path=dataset_root,
        split='train',
        num_obs=1,
    )
    train_loader = DataLoader(dataset, batch_size=batch_size, collate_fn= collate_fn, shuffle=True, num_workers=1)


    # load model
    model = ResNetBinaryClassifier()
    model = model.cuda()

    # load optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # load loss function
    criterion = nn.BCEWithLogitsLoss()

    # load checkpoint
    start_epoch = 0
    if resume_ckpt is not None:
        checkpoint = torch.load(resume_ckpt)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']

    # train
    train_history = []
    model.train()
    for epoch in range(start_epoch, num_epochs):
        # train
        print(f'\nEpoch {epoch}')
        optimizer.zero_grad()
        num_steps = len(train_loader)
        pbar = tqdm(train_loader)
        avg_loss = 0
        for data in pbar:
            input_frame_list_normalized = data['input_frame_list_normalized']
            is_cut = data['is_cut']
            input_frame_list_normalized, is_cut = input_frame_list_normalized.to(device), is_cut.to(device)

            # forward
            output = model(input_frame_list_normalized)
            loss = criterion(output, is_cut)

            # backward
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            avg_loss += loss.item()

        avg_loss /= num_steps
        sync_loss(avg_loss, device)
        train_history.append(avg_loss)

        print("Train loss: {:.6f}".format(avg_loss))
        if (epoch + 1) % save_epoch == 0:
            torch.save(
                model.state_dict(),
                os.path.join(ckpt_dir, f'ckpt_epoch_{epoch + 1}.ckpt'))
            plot_history(train_history, epoch, ckpt_dir, seed)
    
    # save final model
    torch.save(
        model.state_dict(),
        os.path.join(ckpt_dir, f'ckpt_last.ckpt')
    )
    

def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    # save training curves
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f'train_val_{key}_seed_{seed}.png')
        plt.figure()
        train_values = [summary[key].item() for summary in train_history]
        val_values = [summary[key].item() for summary in validation_history]
        plt.plot(np.linspace(0, num_epochs-1, len(train_history)), train_values, label='train')
        plt.plot(np.linspace(0, num_epochs-1, len(validation_history)), val_values, label='validation')
        # plt.ylim([-0.1, 1])
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
    print(f'Saved plots to {ckpt_dir}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--dataset_root', action='store', type=str, help='dataset_root', required=True)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', required=True)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=True)
    parser.add_argument('--save_epoch', action='store', type=int, help='save frequency (epoch)', default=10, required=False)
    parser.add_argument('--lr', action='store', type=float, help='lr', required=True)
    parser.add_argument('--resume_ckpt', action='store', type=str, help='checkpoint to resume training', default=None, required=False)

    main(vars(parser.parse_args()))
