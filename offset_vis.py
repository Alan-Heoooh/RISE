import os
import time
import json
import torch
import argparse
import numpy as np
import open3d as o3d
import torch.nn as nn
import MinkowskiEngine as ME
import matplotlib.pyplot as plt
import torch.distributed as dist
from PIL import Image

from tqdm import tqdm
from copy import deepcopy
from easydict import EasyDict as edict
from diffusers.optimization import get_cosine_schedule_with_warmup

from dataset.realworld import RealWorldDataset, collate_fn, RH20T_RealWorldDataset
from policy import RISE, MLPDiffusion, ForceRISE, ForceRISE2, ForceRISE3
# from eval_agent import Agent
from utils.constants import *
from utils.training import set_seed
from dataset.projector import Projector
from utils.ensemble import EnsembleBuffer
from utils.transformation import rotation_transform


default_args = edict({
    "ckpt": None,
    "calib": "calib/",
    "num_action": 20,
    "num_inference_step": 20,
    "voxel_size": 0.005,
    "obs_feature_dim": 512,
    "hidden_dim": 512,
    "nheads": 8,
    "num_encoder_layers": 4,
    "num_decoder_layers": 1,
    "dim_feedforward": 2048,
    "dropout": 0.1,
    "max_steps": 300,
    "seed": 233,
    "vis": True,
    "discretize_rotation": True,
    "ensemble_mode": "new"
})


def create_point_cloud(colors, depths, cam_intrinsics, voxel_size = 0.005):
    """
    color, depth => point cloud
    """
    h, w = depths.shape
    fx, fy = cam_intrinsics[0, 0], cam_intrinsics[1, 1]
    cx, cy = cam_intrinsics[0, 2], cam_intrinsics[1, 2]

    colors = o3d.geometry.Image(colors.astype(np.uint8))
    depths = o3d.geometry.Image(depths.astype(np.float32))

    camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
        width = w, height = h, fx = fx, fy = fy, cx = cx, cy = cy
    )
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        colors, depths, depth_scale = 1.0, convert_rgb_to_intensity = False
    )
    cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, camera_intrinsics)
    cloud = cloud.voxel_down_sample(voxel_size)
    points = np.array(cloud.points).astype(np.float32)
    colors = np.array(cloud.colors).astype(np.float32)

    x_mask = ((points[:, 0] >= WORKSPACE_MIN[0]) & (points[:, 0] <= WORKSPACE_MAX[0]))
    y_mask = ((points[:, 1] >= WORKSPACE_MIN[1]) & (points[:, 1] <= WORKSPACE_MAX[1]))
    z_mask = ((points[:, 2] >= WORKSPACE_MIN[2]) & (points[:, 2] <= WORKSPACE_MAX[2]))
    mask = (x_mask & y_mask & z_mask)
    points = points[mask]
    colors = colors[mask]
    # imagenet normalization
    colors = (colors - IMG_MEAN) / IMG_STD
    # final cloud
    cloud_final = np.concatenate([points, colors], axis = -1).astype(np.float32)
    return cloud_final

def create_batch(coords, feats):
    """
    coords, feats => batch coords, batch feats (batch size = 1)
    """
    coords_batch = [coords]
    feats_batch = [feats]
    coords_batch, feats_batch = ME.utils.sparse_collate(coords_batch, feats_batch)
    return coords_batch, feats_batch

def create_input(colors, depths, cam_intrinsics, voxel_size = 0.005):
    """
    colors, depths => batch coords, batch feats
    """
    cloud = create_point_cloud(colors, depths, cam_intrinsics, voxel_size = voxel_size)
    coords = np.ascontiguousarray(cloud[:, :3] / voxel_size, dtype = np.int32)
    coords_batch, feats_batch = create_batch(coords, cloud)
    return coords_batch, feats_batch, cloud

def unnormalize_action(action):
    action[..., :3] = (action[..., :3] + 1) / 2.0 * (TRANS_MAX - TRANS_MIN) + TRANS_MIN
    action[..., -1] = (action[..., -1] + 1) / 2.0 * MAX_GRIPPER_WIDTH
    return action

def unnormalize_offset_action(offset_action):
    trans_min = np.array([-0.15, -0.15, -0.10])
    trans_max = np.array([0.15, 0.15, 0.10])
    max_gripper_width = 0.11 # meter
    offset_action[..., :3] = (offset_action[..., :3] + 1) / 2.0 * (trans_max - trans_min) + trans_min
    offset_action[..., -1] = (offset_action[..., -1] + 1) / 2.0 * max_gripper_width
    return offset_action

def get_offset(args_override):
    # load default arguments
    args = deepcopy(default_args)
    for key, value in args_override.items():
        args[key] = value

    # set up device
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # policy
    print("Loading RISE policy ...")
    policy = RISE(
        num_action = args.num_action,
        input_dim = 6,
        obs_feature_dim = args.obs_feature_dim,
        action_dim = 10,
        hidden_dim = args.hidden_dim,
        nheads = args.nheads,
        num_encoder_layers = args.num_encoder_layers,
        num_decoder_layers = args.num_decoder_layers,
        dropout = args.dropout
    ).to(device)
    # n_parameters = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    # print("Number of parameters: {:.2f}M".format(n_parameters / 1e6))

    # load checkpoint
    assert args.ckpt is not None, "Please provide the checkpoint to evaluate."
    policy.load_state_dict(torch.load(args.ckpt, map_location = device), strict = False)
    print("Checkpoint {} loaded.".format(args.ckpt))

    # offset policy
    print("Loading ForceRISE policy ...")
    if args.policy == 'ForceRISE2':
        offset_policy = ForceRISE2(
            num_action = args.num_action,
            input_dim = 6,
            obs_feature_dim = args.obs_feature_dim,
            action_dim = 10,
            hidden_dim = args.hidden_dim,
            nheads = args.nheads,
            num_encoder_layers = args.num_encoder_layers,
            num_decoder_layers = args.num_decoder_layers,
            dropout = args.dropout
        ).to(device)
    elif args.policy == 'ForceRISE3':
        offset_policy = ForceRISE3(
            num_action= args.num_action,
            input_dim = 6,
            obs_feature_dim = args.obs_feature_dim,
            action_dim = 10,
            hidden_dim = args.hidden_dim,
            nheads = args.nheads,
            num_encoder_layers = args.num_encoder_layers,
            num_decoder_layers = args.num_decoder_layers,
            dropout = args.dropout
        ).to(device)
    else:
        raise NotImplementedError("Policy {} not implemented.".format(args.policy))

    # load offset checkpoint
    assert args.offset_ckpt is not None, "Please provide the offset checkpoint to evaluate."
    offset_policy.load_state_dict(torch.load(args.offset_ckpt, map_location = device), strict = False)
    print("Offset checkpoint {} loaded.".format(args.offset_ckpt))

    # projector = Projector(os.path.join(args.calib, str(calib_timestamp)))
    ensemble_buffer = EnsembleBuffer(mode = args.ensemble_mode)

    dataset = RealWorldDataset(
        path = args.data_path,
        split = 'val',
        num_obs = 1,
        num_obs_force= 100,
        num_action = args.num_action,
        voxel_size = args.voxel_size,
        # aug = True,
        # aug_jitter = True, 
        with_cloud = True,
    )
    print(len(dataset))
    mean_pos_error_total = 0
    mean_rise_pos_error_total = 0 
    with torch.inference_mode():
        policy.eval()
        offset_policy.eval()
        cam_id = '750612070851'
        actions = []
        start_step = 15
        is_force_vary = False
        for i in range(start_step, start_step+args.max_steps):
            ret_dict = dataset[i]
            if np.max(ret_dict['input_force_list_std']) > 3:
                is_force_vary = True
            if i % args.num_action == 0:
                feats = torch.tensor(ret_dict['input_feats_list'][0])
                coords = torch.tensor(ret_dict['input_coords_list'][0])
                cloud = ret_dict['clouds_list'][0]
                feats, coords = feats.to(device), coords.to(device)
                coords, feats = create_batch(coords, feats)
                cloud_data = ME.SparseTensor(feats, coords)
                pred_raw_action = policy(cloud_data, actions = None, batch_size = 1).squeeze(0).cpu().numpy()
                rise_action = unnormalize_action(pred_raw_action) # cam coordinate 
                # load offset action
                force_torque = ret_dict['input_force_list'].unsqueeze(0)
                force_torque = force_torque.to(device)
                force_torque_std = ret_dict['input_force_list_std']
                print("force_torque_std: ", force_torque_std)
                print("is_force_vary: ", is_force_vary)
                is_force_vary = False
                print("max_force_torque_std: ", np.max(force_torque_std))
                # forcerise action
                if args.policy == 'ForceRISE2':
                    pred_raw_force_action = offset_policy(force_torque, cloud_data, actions = None, batch_size = 1).squeeze(0).cpu().numpy()
                elif args.policy == 'ForceRISE3':
                    pred_raw_force_action = offset_policy(force_torque, force_torque_std, cloud_data, actions = None, batch_size = 1).squeeze(0).cpu().numpy()
                else:
                    raise NotImplementedError("Policy {} not implemented.".format(args.policy))
                force_action = unnormalize_action(pred_raw_force_action)
                # final action
                action = force_action
                gt_action = ret_dict['action'].squeeze(0).cpu().numpy()
                if args.vis:
                    print("Show cloud ...")
                    import open3d as o3d
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(cloud[:, :3])
                    pcd.colors = o3d.utility.Vector3dVector(cloud[:, 3:] * IMG_STD + IMG_MEAN)
                    tcp_vis_list = []
                    for raw_tcp in action:
                        tcp_vis = o3d.geometry.TriangleMesh.create_sphere(0.005).translate(raw_tcp[:3])
                        tcp_vis.paint_uniform_color([0.0, 1.0, 0.0])  # set color to green
                        tcp_vis_list.append(tcp_vis)
                    tcp_vis_rise_list = []
                    for raw_tcp in rise_action:
                        tcp_vis_rise = o3d.geometry.TriangleMesh.create_sphere(0.005).translate(raw_tcp[:3])
                        tcp_vis_rise.paint_uniform_color([1.0, 0.0, 0.0]) # set color to red
                        tcp_vis_rise_list.append(tcp_vis_rise) 
                    tcp_vis_gt_list = []
                    for raw_tcp in gt_action:
                        tcp_vis_gt = o3d.geometry.TriangleMesh.create_sphere(0.005).translate(raw_tcp[:3])
                        tcp_vis_gt.paint_uniform_color([0.0, 0.0, 1.0]) # set color to blue
                        tcp_vis_gt_list.append(tcp_vis_gt)
                    # Mean position error of action and gt_action
                    # mean_pos_error = np.sum(np.linalg.norm(action[:, :3] - gt_action[:, :3], axis = 1))
                    # mean_rise_pos_error = np.sum(np.linalg.norm(rise_action[:, :3] - gt_action[:, :3], axis = 1))
                    # mean_angle_error = np.sum(np.linalg.norm(action[:, 3:10] - gt_action[:, 3:10], axis = 1))
                    # mean_rise_angle_error = np.sum(np.linalg.norm(rise_action[:, 3:10] - gt_action[:, 3:10], axis = 1))
                    # print("Mean position error: ", mean_pos_error)
                    # print("Mean rise position error: ", mean_rise_pos_error)
                    # print("Mean angle error: ", mean_angle_error)
                    # print("Mean rise angle error: ", mean_rise_angle_error)
                    # mean_pos_error_total += mean_pos_error
                    # mean_rise_pos_error_total += mean_rise_pos_error

                    o3d.visualization.draw_geometries([pcd, *tcp_vis_list, *tcp_vis_rise_list, *tcp_vis_gt_list])
                ensemble_buffer.add_action(action, i)

            # get step action from ensemble buffer
            step_action = ensemble_buffer.get_action()
            if step_action is None:   # no action in the buffer => no movement.
                continue
            actions.append(step_action)

        print("Mean position error total: ", mean_pos_error_total / args.max_steps)
        print("Mean rise position error total: ", mean_rise_pos_error_total / args.max_steps)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', action = 'store', type = str, help = 'checkpoint path', required = True)
    parser.add_argument('--offset_ckpt', action = 'store', type = str, help = 'checkpoint path', required = True)
    parser.add_argument('--policy', action = 'store', type = str, help='type of policy', required=True)
    parser.add_argument('--calib', action = 'store', type = str, help = 'calibration path', required = True)
    parser.add_argument('--data_path', action = 'store', type = str, help = 'data path', required = True)
    
    parser.add_argument('--num_action', action = 'store', type = int, help = 'number of action steps', required = False, default = 20)
    parser.add_argument('--force_feature_dim', action = 'store', type = int, help = 'observation feature dimension', required = False, default = 64)
    parser.add_argument('--num_inference_step', action = 'store', type = int, help = 'number of inference query steps', required = False, default = 20)
    parser.add_argument('--voxel_size', action = 'store', type = float, help = 'voxel size', required = False, default = 0.005)
    parser.add_argument('--obs_feature_dim', action = 'store', type = int, help = 'observation feature dimension', required = False, default = 512)
    parser.add_argument('--hidden_dim', action = 'store', type = int, help = 'hidden dimension', required = False, default = 512)
    parser.add_argument('--nheads', action = 'store', type = int, help = 'number of heads', required = False, default = 8)
    parser.add_argument('--num_encoder_layers', action = 'store', type = int, help = 'number of encoder layers', required = False, default = 4)
    parser.add_argument('--num_decoder_layers', action = 'store', type = int, help = 'number of decoder layers', required = False, default = 1)
    parser.add_argument('--dim_feedforward', action = 'store', type = int, help = 'feedforward dimension', required = False, default = 2048)
    parser.add_argument('--dropout', action = 'store', type = float, help = 'dropout ratio', required = False, default = 0.1)
    parser.add_argument('--max_steps', action = 'store', type = int, help = 'max steps for evaluation', required = False, default = 300)
    parser.add_argument('--seed', action = 'store', type = int, help = 'seed', required = False, default = 233)
    parser.add_argument('--vis', action = 'store_true', help = 'add visualization during evaluation')
    parser.add_argument('--discretize_rotation', action = 'store_true', help = 'whether to discretize rotation process.')
    parser.add_argument('--ensemble_mode', action = 'store', type = str, help = 'temporal ensemble mode', required = False, default = 'new')

    get_offset(vars(parser.parse_args()))