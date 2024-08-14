import os
import matplotlib.pyplot as plt
import numpy as np
from transforms3d.quaternions import quat2mat
from dataset.projector import Projector
# from dataset.force import ForceDataset
from dataset.force_new import ForceDataset
from tqdm import tqdm

def plot_scene_force():
    data = np.load("/aidata/zihao/data/realdata_sampled_20240713/train/task_0230_user_0099_scene_0002_cfg_0001/high_freq_data/force_torque_tcp_joint_timestamp.npy", allow_pickle=True)
    # print key of force_data
    force_data = [x[:6] for x in data] # raw force data
    tcp_base = [x[6:13] for x in data]
    calib_path = '/aidata/zihao/data/realdata_sampled_20240713/calib/1720840800151'
    projector = Projector(calib_path)
    cam = '750612070851'
    # base force_data
    force_data_base = [projector.project_force_to_base_coord(tcp, force) for tcp, force in zip(tcp_base, force_data)]
    force_data_cam = [projector.project_force_to_camera_coord(tcp, force, cam) for tcp, force in zip(tcp_base, force_data)]

    force_data = force_data_base
    force_x = [x[0] for x in data]
    force_y = [x[1] for x in data]
    force_z = [x[2] for x in data]
    frame_id_list = [x[-1] for x in data]

    plt.figure()
    plt.plot(frame_id_list, force_x, label='Force X', color='red')
    plt.plot(frame_id_list, force_y, label='Force Y', color='green')
    plt.plot(frame_id_list, force_z, label='Force Z', color='blue')
    plt.title('Force Components vs Time')
    plt.xlabel('Time')
    plt.ylabel('Force Components')
    plt.legend()
    plt.show()

def plot_window_force_variation():
    dataset = ForceDataset(
        root='/aidata/zihao/data/realdata_sampled_20240713', 
        split='val',
        num_obs=200, 
        horizon=220, 
        selected_tasks=['task_0230'], 
        frame_sample_step=1, 
        selected_cam_ids = ['035622060973'])
    force_std = []
    count = 0
    for i in tqdm(range(len(dataset))):
        ret_dict = dataset[i]
        std_value = ret_dict['input_frame_list_std']
        force_std.append(std_value)
    force_std = np.array(force_std)
    count = np.sum(np.max(force_std, axis=1) > 3)
    print('count:', count)
    # plot force std
    force_std_x = force_std[:, 0]
    force_std_y = force_std[:, 1]
    force_std_z = force_std[:, 2]
    plt.figure()
    plt.plot(force_std_x, label='Force Std X', color='red')
    plt.plot(force_std_y, label='Force Std Y', color='green')
    plt.plot(force_std_z, label='Force Std Z', color='blue')
    # draw a horizontal line at y = 3
    plt.axhline(y=3, color='black', linestyle='--')
    plt.axhline(y=2, color='black', linestyle='--')

    plt.title('Force Std Components vs Time')
    plt.xlabel('Time')
    plt.ylabel('Force Std Components')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    plot_scene_force()