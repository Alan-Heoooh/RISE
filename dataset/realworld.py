import os
import json
import torch
import numpy as np
import open3d as o3d
import MinkowskiEngine as ME
import torchvision.transforms as T
import collections.abc as container_abcs

from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset

from dataset.constants import *

from dataset.projector import Projector
from utils.transformation import rot_trans_mat, apply_mat_to_pose, apply_mat_to_pcd, xyz_rot_transform

class RH20T_RealWorldDataset(Dataset):
    """
    Real-world Dataset.
    """
    def __init__(
        self, 
        path, 
        split = 'train', 
        num_obs = 1,
        num_obs_force = 100,
        num_action = 20, 
        voxel_size = 0.005,
        cam_ids = ['750612070851'],
        aug = False,
        aug_trans_min = [-0.2, -0.2, -0.2],
        aug_trans_max = [0.2, 0.2, 0.2],
        aug_rot_min = [-30, -30, -30],
        aug_rot_max = [30, 30, 30],
        aug_jitter = False,
        aug_jitter_params = [0.4, 0.4, 0.2, 0.1],
        aug_jitter_prob = 0.2,
        with_cloud = False,
        selected_tasks = ['task_0215']
    ):
        assert split in ['train', 'val', 'all']

        self.path = path
        self.split = split
        # self.data_path = os.path.join(path, split)
        self.data_path = path
        self.calib_path = os.path.join(path, "calib")
        self.num_obs = num_obs
        self.num_action = num_action
        self.voxel_size = voxel_size
        self.aug = aug
        self.aug_trans_min = np.array(aug_trans_min)
        self.aug_trans_max = np.array(aug_trans_max)
        self.aug_rot_min = np.array(aug_rot_min)
        self.aug_rot_max = np.array(aug_rot_max)
        self.aug_jitter = aug_jitter
        self.aug_jitter_params = np.array(aug_jitter_params)
        self.aug_jitter_prob = aug_jitter_prob
        self.with_cloud = with_cloud
        
        self.all_demos = sorted(os.listdir(self.data_path))
        if selected_tasks is not None:
            self.all_demos = [x for x in self.all_demos if x[:9] in selected_tasks and len(x) == 39]
        else:
            raise ValueError("Please specify selected tasks.")
        if split == 'train':
            cur_task_ids = [tid for tid in self.all_demos if 'scene_0010' not in tid]
        elif split == 'val':
            cur_task_ids = [tid for tid in self.all_demos if 'scene_0010' in tid]
        task_ids = []
        for task_id in cur_task_ids:
            demo_path = os.path.join(self.data_path, task_id)
            with open(os.path.join(demo_path, "metadata.json"), "r") as f:
                meta = json.load(f)
            if 'rating' not in meta or meta['rating'] <= 1:
                continue
            task_ids.append(task_id)
        self.all_demos = task_ids

        self.num_demos = len(self.all_demos)
        print("Number of demos: {}".format(self.num_demos))
        # print(self.all_demos)

        self.task_names = []
        self.data_paths = []
        self.cam_ids = []
        self.calib_timestamp = []
        self.obs_frame_ids = []
        self.action_frame_ids = []
        self.force_torque_list = []
        self.projectors = {}
        
        for i in range(self.num_demos):
            demo_path = os.path.join(self.data_path, self.all_demos[i])
            # metadata
            with open(os.path.join(demo_path, "metadata.json"), "r") as f:
                meta = json.load(f)
            if 'rating' not in meta or meta['rating'] <= 1:
                continue
            for cam_id in cam_ids:
                # cam path
                cam_path = os.path.join(demo_path, "cam_{}".format(cam_id))
                if not os.path.exists(cam_path):
                    continue
                # get frame ids
                frame_ids = [
                    int(os.path.splitext(x)[0]) 
                    for x in sorted(os.listdir(os.path.join(cam_path, "color"))) 
                    if int(os.path.splitext(x)[0]) <= meta["finish_time"]
                ]
                # get calib timestamps
                calib_timestamp = meta["calib"]
                # get samples according to num_obs and num_action
                obs_frame_ids_list = []
                action_frame_ids_list = []
                padding_mask_list = []
                force_torque_list = []
                force_data = np.load(os.path.join(demo_path, 'transformed', 'high_freq_data.npy'), allow_pickle=True)[()][cam_id] # TODO: change to high_freq data
                force_timestamp = [x['timestamp'] for x in force_data]
                force_data = [x['zeroed'] for x in force_data]

                for cur_idx in range(len(frame_ids) - 1):
                    obs_pad_before = max(0, num_obs - cur_idx - 1)
                    action_pad_after = max(0, num_action - (len(frame_ids) - 1 - cur_idx))
                    frame_begin = max(0, cur_idx - num_obs + 1)
                    frame_end = min(len(frame_ids), cur_idx + num_action + 1)
                    obs_frame_ids = frame_ids[:1] * obs_pad_before + frame_ids[frame_begin: cur_idx + 1]
                    action_frame_ids = frame_ids[cur_idx + 1: frame_end] + frame_ids[-1:] * action_pad_after
                    obs_frame_ids_list.append(obs_frame_ids)
                    action_frame_ids_list.append(action_frame_ids)
                    # get force data
                    cur_idx_force = np.argmin(np.abs(np.array(force_timestamp) - frame_ids[cur_idx]))
                    frame_begin_force = max(0, cur_idx_force - num_obs_force + 1)
                    force_torque = force_data[frame_begin_force: cur_idx_force + 1]
                    force_torque = np.array(force_torque).astype(np.float32)
                    force_torque_list.append(force_torque)
                
                self.task_names += [self.all_demos[i]] * len(obs_frame_ids_list)
                self.data_paths += [demo_path] * len(obs_frame_ids_list)
                self.cam_ids += [cam_id] * len(obs_frame_ids_list)
                self.calib_timestamp += [calib_timestamp] * len(obs_frame_ids_list)
                self.obs_frame_ids += obs_frame_ids_list
                self.action_frame_ids += action_frame_ids_list
                self.force_torque_list += force_torque_list

        
    def __len__(self):
        return len(self.obs_frame_ids)

    def _augmentation(self, clouds, tcps):
        translation_offsets = np.random.rand(3) * (self.aug_trans_max - self.aug_trans_min) + self.aug_trans_min
        rotation_angles = np.random.rand(3) * (self.aug_rot_max - self.aug_rot_min) + self.aug_rot_min
        rotation_angles = rotation_angles / 180 * np.pi  # tranform from degree to radius
        aug_mat = rot_trans_mat(translation_offsets, rotation_angles)
        for cloud in clouds:
            cloud = apply_mat_to_pcd(cloud, aug_mat)
        tcps = apply_mat_to_pose(tcps, aug_mat, rotation_rep = "quaternion")
        return clouds, tcps

    def _normalize_tcp(self, tcp_list):
        ''' tcp_list: [T, 3(trans) + 6(rot) + 1(width)]'''
        tcp_list[:, :3] = (tcp_list[:, :3] - TRANS_MIN) / (TRANS_MAX - TRANS_MIN) * 2 - 1
        tcp_list[:, -1] = tcp_list[:, -1] / MAX_GRIPPER_WIDTH * 2 - 1
        return tcp_list

    def load_point_cloud(self, colors, depths, cam_id, calib_timestamp):
        h, w = depths.shape
        INTRINSICS = np.load(os.path.join(self.calib_path, str(calib_timestamp), 'intrinsics.npy'), allow_pickle=True).item()
        fx, fy = INTRINSICS[cam_id][0, 0], INTRINSICS[cam_id][1, 1]
        cx, cy = INTRINSICS[cam_id][0, 2], INTRINSICS[cam_id][1, 2]
        scale = 1000. if 'f' not in cam_id else 4000.
        colors = o3d.geometry.Image(colors.astype(np.uint8))
        depths = o3d.geometry.Image(depths.astype(np.float32))
        camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
            width = w, height = h, fx = fx, fy = fy, cx = cx, cy = cy
        )
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            colors, depths, scale, convert_rgb_to_intensity = False
        )
        cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, camera_intrinsics)
        cloud = cloud.voxel_down_sample(self.voxel_size)
        points = np.array(cloud.points)
        colors = np.array(cloud.colors)
        return points.astype(np.float32), colors.astype(np.float32)

    def __getitem__(self, index):
        data_path = self.data_paths[index]
        cam_id = self.cam_ids[index]
        calib_timestamp = self.calib_timestamp[index]
        obs_frame_ids = self.obs_frame_ids[index]
        action_frame_ids = self.action_frame_ids[index]

        # directories
        color_dir = os.path.join(data_path, "cam_{}".format(cam_id), 'color')
        depth_dir = os.path.join(data_path, "cam_{}".format(cam_id), 'depth')
        tcp_dir = os.path.join(data_path, "cam_{}".format(cam_id), 'tcp')
        gripper_dir = os.path.join(data_path, "cam_{}".format(cam_id), 'gripper_command')

        # load camera projector by calib timestamp
        with open(os.path.join(data_path, "metadata.json"), "r") as f:
            meta = json.load(f)
        timestamp = meta["calib"]

        if timestamp not in self.projectors:
            # create projector cache
            self.projectors[timestamp] = Projector(os.path.join(self.calib_path, str(timestamp)))
        projector = self.projectors[timestamp]

        # create color jitter
        if self.split == 'train' and self.aug_jitter:
            jitter = T.ColorJitter(
                brightness = self.aug_jitter_params[0],
                contrast = self.aug_jitter_params[1],
                saturation = self.aug_jitter_params[2],
                hue = self.aug_jitter_params[3]
            )
            jitter = T.RandomApply([jitter], p = self.aug_jitter_prob)

        # load colors and depths
        colors_list = []
        depths_list = []
        for frame_id in obs_frame_ids:
            colors = Image.open(os.path.join(color_dir, "{}.jpg".format(frame_id)))
            if self.split == 'train' and self.aug_jitter:
                colors = jitter(colors)
            colors_list.append(colors)
            depths_list.append(
                np.array(Image.open(os.path.join(depth_dir, "{}.png".format(frame_id))), dtype = np.float32)
            )
        colors_list = np.stack(colors_list, axis = 0)
        depths_list = np.stack(depths_list, axis = 0)

        # point clouds
        clouds = []
        for i, frame_id in enumerate(obs_frame_ids):
            points, colors = self.load_point_cloud(colors_list[i], depths_list[i], cam_id, calib_timestamp)
            x_mask = ((points[:, 0] >= WORKSPACE_MIN[0]) & (points[:, 0] <= WORKSPACE_MAX[0]))
            y_mask = ((points[:, 1] >= WORKSPACE_MIN[1]) & (points[:, 1] <= WORKSPACE_MAX[1]))
            z_mask = ((points[:, 2] >= WORKSPACE_MIN[2]) & (points[:, 2] <= WORKSPACE_MAX[2]))
            mask = (x_mask & y_mask & z_mask)
            points = points[mask]
            colors = colors[mask]
            # apply imagenet normalization
            colors = (colors - IMG_MEAN) / IMG_STD
            cloud = np.concatenate([points, colors], axis = -1)
            clouds.append(cloud)

        # load force
        force_list = self.force_torque_list[index]

        # actions
        action_tcps = []
        action_grippers = []
        for frame_id in action_frame_ids:
            tcp = np.load(os.path.join(tcp_dir, "{}.npy".format(frame_id)))[:7].astype(np.float32)
            projected_tcp = projector.project_tcp_to_camera_coord(tcp, cam_id)
            gripper_width = decode_gripper_width(np.load(os.path.join(gripper_dir, "{}.npy".format(frame_id)))[0])
            action_tcps.append(projected_tcp)
            action_grippers.append(gripper_width)
        action_tcps = np.stack(action_tcps)
        action_grippers = np.stack(action_grippers)

        # point augmentations
        if self.split == 'train' and self.aug:
            clouds, action_tcps = self._augmentation(clouds, action_tcps)
    
        # rotation transformation (to 6d)
        action_tcps = xyz_rot_transform(action_tcps, from_rep = "quaternion", to_rep = "rotation_6d")
        actions = np.concatenate((action_tcps, action_grippers[..., np.newaxis]), axis = -1)

        # normalization
        actions_normalized = self._normalize_tcp(actions.copy())

        # make voxel input
        input_coords_list = []
        input_feats_list = []
        for cloud in clouds:
            # Upd Note. Make coords contiguous.
            coords = np.ascontiguousarray(cloud[:, :3] / self.voxel_size, dtype = np.int32)
            # Upd Note. API change.
            input_coords_list.append(coords)
            input_feats_list.append(cloud.astype(np.float32))

        # convert to torch
        actions = torch.from_numpy(actions).float()
        actions_normalized = torch.from_numpy(actions_normalized).float()

        ret_dict = {
            'input_coords_list': input_coords_list,
            'input_feats_list': input_feats_list,
            'input_force_list': force_list,
            'action': actions,
            'action_normalized': actions_normalized,

            'task_name': self.task_names[index],
            'data_path': data_path,
            'obs_frame_ids': obs_frame_ids,
        }
        
        if self.with_cloud:  # warning: this may significantly slow down the training process.
            ret_dict["clouds_list"] = clouds

        return ret_dict
        

class RealWorldDataset(Dataset):
    """
    Real-world Dataset.
    """
    def __init__(
        self, 
        path, 
        split = 'train', 
        num_obs = 1,
        num_obs_force = 100,
        num_action = 20, 
        voxel_size = 0.005,
        cam_ids = ['750612070851', '035622060973'],
        aug = False,
        aug_trans_min = [-0.2, -0.2, -0.2],
        aug_trans_max = [0.2, 0.2, 0.2],
        aug_rot_min = [-30, -30, -30],
        aug_rot_max = [30, 30, 30],
        aug_jitter = False,
        aug_jitter_params = [0.4, 0.4, 0.2, 0.1],
        aug_jitter_prob = 0.2,
        with_cloud = False,
    ):
        assert split in ['train', 'val', 'all']

        self.path = path
        self.split = split
        # self.data_path = os.path.join(path, split)
        self.data_path = os.path.join(path, 'train')

        self.calib_path = os.path.join(path, "calib")
        self.num_obs = num_obs
        self.num_action = num_action
        self.voxel_size = voxel_size
        self.aug = aug
        self.aug_trans_min = np.array(aug_trans_min)
        self.aug_trans_max = np.array(aug_trans_max)
        self.aug_rot_min = np.array(aug_rot_min)
        self.aug_rot_max = np.array(aug_rot_max)
        self.aug_jitter = aug_jitter
        self.aug_jitter_params = np.array(aug_jitter_params)
        self.aug_jitter_prob = aug_jitter_prob
        self.with_cloud = with_cloud
        
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
                force_torque_raw_list = []
                tcp_base_list = []
                force_torque_tcp_joint_timestamp = np.load(os.path.join(demo_path, 'high_freq_data', 'force_torque_tcp_joint_timestamp.npy'))
                high_freq_timestamp = force_torque_tcp_joint_timestamp[:, -1]
                force_torque_raw_data = force_torque_tcp_joint_timestamp[:, :6]
                tcp_base_data = force_torque_tcp_joint_timestamp[:, 6:13]


                for cur_idx in range(len(frame_ids) - 1):
                    obs_pad_before = max(0, num_obs - cur_idx - 1)
                    action_pad_after = max(0, num_action - (len(frame_ids) - 1 - cur_idx))
                    frame_begin = max(0, cur_idx - num_obs + 1)
                    frame_end = min(len(frame_ids), cur_idx + num_action + 1)
                    obs_frame_ids = frame_ids[:1] * obs_pad_before + frame_ids[frame_begin: cur_idx + 1]
                    action_frame_ids = frame_ids[cur_idx + 1: frame_end] + frame_ids[-1:] * action_pad_after
                    obs_frame_ids_list.append(obs_frame_ids)
                    action_frame_ids_list.append(action_frame_ids)
                    # get force data
                    cur_idx_force = np.argmin(np.abs(high_freq_timestamp - frame_ids[cur_idx]))
                    frame_begin_force = max(0, cur_idx_force - num_obs_force + 1)
                    force_torque = force_torque_raw_data[frame_begin_force: cur_idx_force + 1]
                    tcp_base = tcp_base_data[frame_begin_force: cur_idx_force + 1]
                    force_torque = np.array(force_torque).astype(np.float32)
                    tcp_base = np.array(tcp_base).astype(np.float32)
                    force_torque_raw_list.append(force_torque)
                    tcp_base_list.append(tcp_base)
                self.task_names += [self.all_demos[i]] * len(obs_frame_ids_list)
                self.data_paths += [demo_path] * len(obs_frame_ids_list)
                self.cam_ids += [cam_id] * len(obs_frame_ids_list)
                self.calib_timestamp += [calib_timestamp] * len(obs_frame_ids_list)
                self.obs_frame_ids += obs_frame_ids_list
                self.action_frame_ids += action_frame_ids_list
                self.force_torque_raw_list += force_torque_raw_list
                self.tcp_base_list += tcp_base_list
        
    def __len__(self):
        return len(self.obs_frame_ids)

    def _augmentation(self, clouds, tcps):
        translation_offsets = np.random.rand(3) * (self.aug_trans_max - self.aug_trans_min) + self.aug_trans_min
        rotation_angles = np.random.rand(3) * (self.aug_rot_max - self.aug_rot_min) + self.aug_rot_min
        rotation_angles = rotation_angles / 180 * np.pi  # tranform from degree to radius
        aug_mat = rot_trans_mat(translation_offsets, rotation_angles)
        for cloud in clouds:
            cloud = apply_mat_to_pcd(cloud, aug_mat)
        tcps = apply_mat_to_pose(tcps, aug_mat, rotation_rep = "quaternion")
        return clouds, tcps

    def _normalize_tcp(self, tcp_list):
        ''' tcp_list: [T, 3(trans) + 6(rot) + 1(width)]'''
        tcp_list[:, :3] = (tcp_list[:, :3] - TRANS_MIN) / (TRANS_MAX - TRANS_MIN) * 2 - 1
        tcp_list[:, -1] = tcp_list[:, -1] / MAX_GRIPPER_WIDTH * 2 - 1
        return tcp_list

    def _normalize_force(self, force_list):
        ''' force_list: [T, 6]'''
        force_list = (force_list - FORCE_MIN) / (FORCE_MAX - FORCE_MIN) * 2 - 1
        return force_list

    def load_point_cloud(self, colors, depths, cam_id):
        h, w = depths.shape
        fx, fy = INTRINSICS[cam_id][0, 0], INTRINSICS[cam_id][1, 1]
        cx, cy = INTRINSICS[cam_id][0, 2], INTRINSICS[cam_id][1, 2]
        scale = 1000. if 'f' not in cam_id else 4000.
        colors = o3d.geometry.Image(colors.astype(np.uint8))
        depths = o3d.geometry.Image(depths.astype(np.float32))
        camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
            width = w, height = h, fx = fx, fy = fy, cx = cx, cy = cy
        )
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            colors, depths, scale, convert_rgb_to_intensity = False
        )
        cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, camera_intrinsics)
        cloud = cloud.voxel_down_sample(self.voxel_size)
        points = np.array(cloud.points)
        colors = np.array(cloud.colors)
        return points.astype(np.float32), colors.astype(np.float32)

    def __getitem__(self, index):
        data_path = self.data_paths[index]
        cam_id = self.cam_ids[index]
        calib_timestamp = self.calib_timestamp[index]
        obs_frame_ids = self.obs_frame_ids[index]
        action_frame_ids = self.action_frame_ids[index]

        # directories
        color_dir = os.path.join(data_path, "cam_{}".format(cam_id), 'color')
        depth_dir = os.path.join(data_path, "cam_{}".format(cam_id), 'depth')
        tcp_dir = os.path.join(data_path, "cam_{}".format(cam_id), 'tcp')
        gripper_dir = os.path.join(data_path, "cam_{}".format(cam_id), 'gripper_command')

        # load camera projector by calib timestamp
        timestamp_path = os.path.join(data_path, 'timestamp.txt')
        with open(timestamp_path, 'r') as f:
            timestamp = f.readline().rstrip()
        if timestamp not in self.projectors:
            # create projector cache
            self.projectors[timestamp] = Projector(os.path.join(self.calib_path, timestamp))
        projector = self.projectors[timestamp]

        # create color jitter
        if self.split == 'train' and self.aug_jitter:
            jitter = T.ColorJitter(
                brightness = self.aug_jitter_params[0],
                contrast = self.aug_jitter_params[1],
                saturation = self.aug_jitter_params[2],
                hue = self.aug_jitter_params[3]
            )
            jitter = T.RandomApply([jitter], p = self.aug_jitter_prob)

        # load colors and depths
        colors_list = []
        depths_list = []
        for frame_id in obs_frame_ids:
            colors = Image.open(os.path.join(color_dir, "{}.png".format(frame_id)))
            if self.split == 'train' and self.aug_jitter:
                colors = jitter(colors)
            colors_list.append(colors)
            depths_list.append(
                np.array(Image.open(os.path.join(depth_dir, "{}.png".format(frame_id))), dtype = np.float32)
            )
        colors_list = np.stack(colors_list, axis = 0)
        depths_list = np.stack(depths_list, axis = 0)

        # point clouds
        clouds = []
        for i, frame_id in enumerate(obs_frame_ids):
            points, colors = self.load_point_cloud(colors_list[i], depths_list[i], cam_id)
            x_mask = ((points[:, 0] >= WORKSPACE_MIN[0]) & (points[:, 0] <= WORKSPACE_MAX[0]))
            y_mask = ((points[:, 1] >= WORKSPACE_MIN[1]) & (points[:, 1] <= WORKSPACE_MAX[1]))
            z_mask = ((points[:, 2] >= WORKSPACE_MIN[2]) & (points[:, 2] <= WORKSPACE_MAX[2]))
            mask = (x_mask & y_mask & z_mask)
            points = points[mask]
            colors = colors[mask]
            # apply imagenet normalization
            colors = (colors - IMG_MEAN) / IMG_STD
            cloud = np.concatenate([points, colors], axis = -1)
            clouds.append(cloud)

        # actions
        action_tcps = []
        action_grippers = []
        for frame_id in action_frame_ids:
            tcp = np.load(os.path.join(tcp_dir, "{}.npy".format(frame_id)))[:7].astype(np.float32)
            projected_tcp = projector.project_tcp_to_camera_coord(tcp, cam_id)
            gripper_width = decode_gripper_width(np.load(os.path.join(gripper_dir, "{}.npy".format(frame_id)))[0])
            action_tcps.append(projected_tcp)
            action_grippers.append(gripper_width)
        action_tcps = np.stack(action_tcps)
        action_grippers = np.stack(action_grippers)

        # point augmentations
        if self.split == 'train' and self.aug:
            clouds, action_tcps = self._augmentation(clouds, action_tcps)
        
        # rotation transformation (to 6d)
        action_tcps = xyz_rot_transform(action_tcps, from_rep = "quaternion", to_rep = "rotation_6d")
        actions = np.concatenate((action_tcps, action_grippers[..., np.newaxis]), axis = -1)

        # normalization
        actions_normalized = self._normalize_tcp(actions.copy())

        # make voxel input
        input_coords_list = []
        input_feats_list = []
        for cloud in clouds:
            # Upd Note. Make coords contiguous.
            coords = np.ascontiguousarray(cloud[:, :3] / self.voxel_size, dtype = np.int32)
            # Upd Note. API change.
            input_coords_list.append(coords)
            input_feats_list.append(cloud.astype(np.float32))

        # force torque data
        force_torque_list = self.force_torque_raw_list[index].astype(np.float32)
        tcp_base_list = self.tcp_base_list[index].astype(np.float32)
        force_torque_base_list = [projector.project_force_to_base_coord(tcp, force) for tcp, force in zip(tcp_base_list, force_torque_list)]
        force_torque_cam_list = [projector.project_force_to_camera_coord(tcp, force, cam_id) for tcp, force in zip(tcp_base_list, force_torque_list)]
        force_torque_base_list = np.array(force_torque_base_list).astype(np.float32)
        force_torque_cam_list = np.array(force_torque_cam_list).astype(np.float32)
        force_torque_std = np.std(force_torque_cam_list, axis = 0)

        # normalize force torque
        force_torque_cam_normalized = self._normalize_force(force_torque_cam_list.copy())
        force_torque_base_normalized = self._normalize_force(force_torque_base_list.copy())

        # convert to torch
        actions = torch.from_numpy(actions).float()
        actions_normalized = torch.from_numpy(actions_normalized).float()
        
        force_torque_cam_list = torch.from_numpy(force_torque_cam_list).float()
        force_torque_cam_normalized = torch.from_numpy(force_torque_cam_normalized).float()


        ret_dict = {
            'input_coords_list': input_coords_list,
            'input_feats_list': input_feats_list,
            'input_force_list': force_torque_cam_normalized,
            'input_force_list_std': force_torque_std,
            'input_force_list_normalized': force_torque_cam_normalized,
            'action': actions,
            'action_normalized': actions_normalized,

            'task_name': self.task_names[index],
            'data_path': data_path,
            'action_frame_ids': action_frame_ids,
            'cam_id': cam_id,
        }
        
        if self.with_cloud:  # warning: this may significantly slow down the training process.
            ret_dict["clouds_list"] = clouds

        return ret_dict
        

def collate_fn(batch):
    if type(batch[0]).__module__ == 'numpy':
        return torch.stack([torch.from_numpy(b) for b in batch], 0)
    elif torch.is_tensor(batch[0]):
        return torch.stack(batch, 0)
    elif isinstance(batch[0], container_abcs.Mapping):
        ret_dict = {}
        for key in batch[0]:
            if key in TO_TENSOR_KEYS:
                ret_dict[key] = collate_fn([d[key] for d in batch])
            else:
                ret_dict[key] = [d[key] for d in batch]
        coords_batch = ret_dict['input_coords_list']
        feats_batch = ret_dict['input_feats_list']
        coords_batch, feats_batch = ME.utils.sparse_collate(coords_batch, feats_batch)
        ret_dict['input_coords_list'] = coords_batch
        ret_dict['input_feats_list'] = feats_batch
        return ret_dict
    elif isinstance(batch[0], container_abcs.Sequence):
        return [sample for b in batch for sample in b]
    
    raise TypeError("batch must contain tensors, dicts or lists; found {}".format(type(batch[0])))


def decode_gripper_width(gripper_width):
    return gripper_width / 1000. * 0.095

if __name__ == '__main__':
    dataset = RealWorldDataset(
        path = '/aidata/zihao/data/realdata_sampled_20240713')
    print(len(dataset))
    force_std = []
    for i in tqdm(range(len(dataset))):
        ret_dict = dataset[i]
        std_value = ret_dict['input_force_list']
        force_std.append(std_value)
    count = np.sum(np.max(force_std, axis=1) > 3)
    print('count:', count)
    # print(dataset[0]['input_coords_list'].shape)
