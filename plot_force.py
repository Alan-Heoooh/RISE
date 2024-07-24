import os
import matplotlib.pyplot as plt
import numpy as np
from transforms3d.quaternions import quat2mat
from dataset.projector import Projector

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