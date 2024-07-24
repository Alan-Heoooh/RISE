import numpy as np
from dataset.projector import Projector
import os
import json
# Load the .npy file
# data = np.load('/aidata/RH100T_cfg1/task_0215_user_0014_scene_0001_cfg_0001/cam_750612070851/force_torque/1635770079514.npy', allow_pickle=True)

# data_path = '/aidata/RH100T_cfg1'
# task_id = 'task_0215_user_0014_scene_0001_cfg_0001'
# calib_path = '/aidata/RH100T_cfg1/calib'
# cam_id = '750612070851'
# with open(os.path.join(data_path, task_id, "metadata.json"), "r") as f:
#     meta = json.load(f)
# calib_timestamp = meta["calib"]
# projector = Projector(os.path.join(calib_path, str(calib_timestamp)))
# force_torque_base = np.load(os.path.join(data_path, task_id, "transformed", "force_torque_base.npy"), allow_pickle = True)[()][cam_id][0]
# force_torque_cam = np.load(os.path.join(data_path, task_id, "transformed", "force_torque.npy"), allow_pickle = True)[()][cam_id][0]
# tcp_base = np.load(os.path.join(data_path, task_id, "transformed", "tcp_base.npy"), allow_pickle = True)[()][cam_id][0]
# print(force_torque_base)
# print(force_torque_cam)
# print(tcp_base)


# test_force_cam = projector.project_force_to_camera_coord(tcp_base['tcp'], force_torque_base['raw'], cam_id)
# print(test_force_cam)

data_path = '/aidata/zihao/data/realdata_sampled_20240713'
cam_id = '750612070851'
calib_timestamp = "1720840800151"
calib_path = '/aidata/zihao/data/realdata_sampled_20240713/calib'
projector = Projector(os.path.join(calib_path, str(calib_timestamp)))


force_torque_tcp_joint_raw = np.load('/aidata/zihao/data/realdata_sampled_20240713/train/task_0230_user_0099_scene_0001_cfg_0001/high_freq_data/force_torque_tcp_joint_timestamp.npy', allow_pickle=True)
tcp_base = force_torque_tcp_joint_raw[300][6:13]
print(tcp_base)
# print(force_torque_raw[656])
force_torque_raw = force_torque_tcp_joint_raw[300][:6]
print(force_torque_raw)
test_base = projector.project_force_to_base_coord(tcp_base, force_torque_raw)
print(test_base)
test_cam = projector.project_force_to_camera_coord(tcp_base, force_torque_raw, cam_id)
print(test_cam)
# print(projector.project_force_to_base_coord(tcp_base['tcp'], force_torque_raw[650][:6]))
