import os
import matplotlib.pyplot as plt
import numpy as np
from transforms3d.quaternions import quat2mat

offset_path = '/aidata/zihao/data/offset/offset_8_0727.npy'
offsets = np.load(offset_path, allow_pickle=True).item()
cam = '750612070851'
task_name = 'task_0230_user_0099_scene_0002_cfg_0001'
offsets = offsets[task_name]
offset_data = [x for x in offsets if x['cam_id'] == cam]
offset = [x['offset'] for x in offset_data]
offset_x = [x[0] for x in offset]
offset_y = [x[1] for x in offset]
offset_z = [x[2] for x in offset]
frame_id_list = [x['timestamp'] for x in offset_data]

plt.figure()
plt.plot(frame_id_list, offset_x, label='Offset X', color='red')
plt.plot(frame_id_list, offset_y, label='Offset Y', color='green')
plt.plot(frame_id_list, offset_z, label='Offset Z', color='blue')
plt.title('Offset Components vs Time')
plt.xlabel('Time')
plt.ylabel('Offset Components')
# magnify the y-axis numbers
plt.legend()
plt.show()

# import os
# import matplotlib.pyplot as plt
# import numpy as np
# from transforms3d.quaternions import quat2mat
# from dataset.projector import Projector

# # Load data
# data = np.load("/aidata/zihao/data/realdata_sampled_20240713/train/task_0230_user_0099_scene_0002_cfg_0001/high_freq_data/force_torque_tcp_joint_timestamp.npy", allow_pickle=True)
# force_data = [x[:6] for x in data] # raw force data
# tcp_base = [x[6:13] for x in data]
# calib_path = '/aidata/zihao/data/realdata_sampled_20240713/calib/1720840800151'
# projector = Projector(calib_path)
# cam = '750612070851'

# # Project force data
# force_data_base = [projector.project_force_to_base_coord(tcp, force) for tcp, force in zip(tcp_base, force_data)]
# force_data_cam = [projector.project_force_to_camera_coord(tcp, force, cam) for tcp, force in zip(tcp_base, force_data)]

# force_data = force_data_base
# force_x = [x[0] for x in force_data]
# force_y = [x[1] for x in force_data]
# force_z = [x[2] for x in force_data]
# frame_id_list_force = [x[-1] for x in data]

# # Load offset data
# offset_path = '/aidata/zihao/data/offset/offset_8_0727.npy'
# offsets = np.load(offset_path, allow_pickle=True).item()
# task_name = 'task_0230_user_0099_scene_0002_cfg_0001'
# offset_data = [x for x in offsets[task_name] if x['cam_id'] == cam]
# offset = [x['offset'] for x in offset_data]
# offset_x = [x[0] for x in offset]
# offset_y = [x[1] for x in offset]
# offset_z = [x[2] for x in offset]
# frame_id_list_offset = [x['timestamp'] for x in offset_data]

# # Plot force and offset data on the same plot
# plt.figure()

# plt.subplot(3, 1, 1)
# plt.plot(frame_id_list_force, force_x, label='Force X', color='red')
# plt.plot(frame_id_list_offset, offset_x, label='Offset X', color='blue')
# plt.title('Force and Offset X vs Time')
# plt.xlabel('Time')
# plt.ylabel('X Component')
# plt.legend()

# plt.subplot(3, 1, 2)
# plt.plot(frame_id_list_force, force_y, label='Force Y', color='green')
# plt.plot(frame_id_list_offset, offset_y, label='Offset Y', color='blue')
# plt.title('Force and Offset Y vs Time')
# plt.xlabel('Time')
# plt.ylabel('Y Component')
# plt.legend()

# plt.subplot(3, 1, 3)
# plt.plot(frame_id_list_force, force_z, label='Force Z', color='purple')
# plt.plot(frame_id_list_offset, offset_z, label='Offset Z', color='blue')
# plt.title('Force and Offset Z vs Time')
# plt.xlabel('Time')
# plt.ylabel('Z Component')
# plt.legend()

# plt.tight_layout()
# plt.show()
