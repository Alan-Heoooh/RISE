import os
import json
import torch
import numpy as np
import open3d as o3d
import torchvision.transforms as T
import collections.abc as container_abcs

from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset

from dataset.constants import *

from dataset.projector import Projector
from utils.transformation import rot_trans_mat, apply_mat_to_pose, apply_mat_to_pcd, xyz_rot_transform

class ClassifierDataset(Dataset):
    """
    Real-world Dataset.
    """
    def __init__(
        self, 
        path, 
        split = 'train', 
        num_obs = 1,
        num_action = 20, 
        cam_ids = ['750612070851', '035622060973'],
    ):
        assert split in ['train', 'val', 'all']

        self.path = path
        self.split = split
        # self.data_path = os.path.join(path, split)
        self.data_path = os.path.join(path, 'train')

        self.calib_path = os.path.join(path, "calib")
        self.num_obs = num_obs
        self.num_action = num_action
        
        self.all_demos = sorted(os.listdir(self.data_path))
        self.num_demos = len(self.all_demos)

        self.task_names = []
        self.data_paths = []
        self.cam_ids = []
        self.calib_timestamp = []
        self.obs_frame_ids = []
        self.action_frame_ids = []
        self.force_torque_raw_list = []
        self.tcp_base_list = []
        self.projectors = {}
        if split == 'train':
            cur_task_ids = [tid for tid in self.all_demos if 'scene_0009' not in tid]
        elif split == 'val':
            cur_task_ids = [tid for tid in self.all_demos if 'scene_0009' in tid]
        
        self.time_label = json.load(open('/home/zihao/RISE_2/dataset/label.json', 'r'))

        self.num_demos = len(cur_task_ids)
        for i in range(self.num_demos):
            # demo_path = os.path.join(self.data_path, self.all_demos[i])
            demo_path = os.path.join(self.data_path, cur_task_ids[i])
            for cam_id in cam_ids:
                # path
                cam_path = os.path.join(demo_path, "cam_{}".format(cam_id))
                if not os.path.exists(cam_path):
                    continue
                # metadata
                with open(os.path.join(demo_path, "metadata.json"), "r") as f:
                    meta = json.load(f)
                # get frame ids
                frame_ids = [
                    int(os.path.splitext(x)[0]) 
                    for x in sorted(os.listdir(os.path.join(cam_path, "color"))) 
                    if int(os.path.splitext(x)[0]) <= meta["finish_time"]
                ]
                # get calib timestamps
                with open(os.path.join(demo_path, "timestamp.txt"), "r") as f:
                    calib_timestamp = f.readline().rstrip()
                # get samples according to num_obs and num_action
                obs_frame_ids_list = []
                action_frame_ids_list = []
                padding_mask_list = []

                for cur_idx in range(len(frame_ids) - 1):
                    obs_pad_before = max(0, num_obs - cur_idx - 1)
                    action_pad_after = max(0, num_action - (len(frame_ids) - 1 - cur_idx))
                    frame_begin = max(0, cur_idx - num_obs + 1)
                    frame_end = min(len(frame_ids), cur_idx + num_action + 1)
                    obs_frame_ids = frame_ids[:1] * obs_pad_before + frame_ids[frame_begin: cur_idx + 1]
                    action_frame_ids = frame_ids[cur_idx + 1: frame_end] + frame_ids[-1:] * action_pad_after
                    obs_frame_ids_list.append(obs_frame_ids)
                    action_frame_ids_list.append(action_frame_ids)

                self.task_names += [self.all_demos[i]] * len(obs_frame_ids_list)
                self.data_paths += [demo_path] * len(obs_frame_ids_list)
                self.cam_ids += [cam_id] * len(obs_frame_ids_list)
                self.calib_timestamp += [calib_timestamp] * len(obs_frame_ids_list)
                self.obs_frame_ids += obs_frame_ids_list
                self.action_frame_ids += action_frame_ids_list
 
    def __len__(self):
        return len(self.obs_frame_ids)

    def __getitem__(self, index):
        data_path = self.data_paths[index]
        cam_id = self.cam_ids[index]
        calib_timestamp = self.calib_timestamp[index]
        obs_frame_ids = self.obs_frame_ids[index]
        action_frame_ids = self.action_frame_ids[index]

        # directories
        color_dir = os.path.join(data_path, "cam_{}".format(cam_id), 'color')
        depth_dir = os.path.join(data_path, "cam_{}".format(cam_id), 'depth')


        # load colors and depths
        colors_list = []
        depths_list = []
        for frame_id in obs_frame_ids:
            colors = Image.open(os.path.join(color_dir, "{}.png".format(frame_id)))
            colors_list.append(colors)
            depths_list.append(
                np.array(Image.open(os.path.join(depth_dir, "{}.png".format(frame_id))), dtype = np.float32)
            )
        colors_list = np.stack(colors_list, axis = 0)
        depths_list = np.stack(depths_list, axis = 0)


        if self.num_obs == 1:
            colors_list = colors_list[0]
        img_process = T.Compose([
            T.ToTensor(),
            T.Resize((224, 224), antialias = True),
            T.Normalize(mean = IMG_MEAN, std = IMG_STD)
        ])
        color_list_normalized = img_process(colors_list)

        curr_frame_id = obs_frame_ids[-1]
        task_name = self.task_names[index]
        is_cut = curr_frame_id > self.time_label[task_name]['begin'] and curr_frame_id < self.time_label[task_name]['end']
        is_cut = torch.tensor(is_cut, dtype=torch.float)
        
        ret_dict = {
            'input_frame_list': colors_list, # (..., 720, 1280, 3)
            'input_frame_list_normalized': color_list_normalized, # (..., 3, 224, 224)
            'is_cut': is_cut,

        }

        return ret_dict
    
def decode_gripper_width(gripper_width):
    return gripper_width / 1000. * 0.095