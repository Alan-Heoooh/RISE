import numpy as np

# Load the .npy file
data = np.load('/aidata/RH100T_cfg1/task_0215_user_0014_scene_0001_cfg_0001/cam_750612070851/force_torque/1635770079514.npy', allow_pickle=True)
data = np.load('/aidata/RH100T_cfg1/task_0215_user_0014_scene_0001_cfg_0001/transformed/force_torque.npy', allow_pickle=True)


# Save the data to a .txt file with a specific format
# np.savetxt('file.txt', data, fmt='%f')
# print(data)
with open('file.txt', 'w') as f:
    f.write(str(data[()]['750612070851']))

# print(data)