import os
import json
import numpy as np

from tqdm import tqdm
import torch
import collections.abc as container_abcs
from torch.utils.data import Dataset
from dataset.projector import Projector

TO_TENSOR_KEYS = ['input_frame_list', 'input_frame_tcp_normalized', 'target_frame_tcp_normalized', 'target_offset_list', 'target_offset_list_normalized', 'padding_mask']


class ForceDataset(Dataset):
    def __init__(
            self,
            root, 
            offset_path,
            split='train', 
            num_obs=1,
            horizon=1+20, 
            frame_sample_step=1, 
            top_down_view=True, 
            selected_tasks=None, 
            selected_cam_ids = ['750612070851', '035622060973']):
        if not os.path.exists(root):
            raise AttributeError("Dataset not found")
        assert split in ['train', 'val', 'all']
        self.root = root
        self.split = split
        self.num_obs = num_obs
        self.horizon = horizon
        self.top_down_view = top_down_view
        # self.data_path = os.path.join(root, split)
        self.data_path = os.path.join(root, 'train')
        self.calib_path = os.path.join(root, 'calib')

        self.input_task_ids = []
        self.input_cam_ids = []
        self.target_frame_ids = []
        self.target_force_list = []
        self.target_tcp_list = []
        self.target_offset_list = []
        self.gripper_list = []
        self.padding_mask_list = []

        self.offset_path = offset_path
        self.all_demos = sorted(os.listdir(self.data_path))
        self.num_demos = len(self.all_demos)

        # load all tasks
        assert split in ['train', 'val', 'all']
        task_ids = []
        cam_ids = []
        config_ids = []

        # def _get_scene_meta(scene_dir):
        #     meta_path = os.path.join(scene_dir, 'metadata.json')
        #     metadata = json.load(open(meta_path))
        #     return metadata
        
        if split == 'train':
            cur_task_ids = [tid for tid in self.all_demos if 'scene_0009' not in tid]
        elif split == 'val':
            cur_task_ids = [tid for tid in self.all_demos if 'scene_0009' in tid]

        for task_id in cur_task_ids:
            if selected_tasks is not None and task_id[:9] not in selected_tasks:
                continue
            # print(task_id)
            # scene_dir = os.path.join(root, task_config, task_id)
            for cam_id in selected_cam_ids:
                task_ids.append(task_id)
                cam_ids.append(cam_id)

        self.task_ids = task_ids
        self.cam_ids = cam_ids

        num_tasks = len(self.task_ids)
        print('#tasks:', num_tasks)
        # get all frame_ids(timestamp) under certain task and cam
        for i in tqdm(range(num_tasks), desc='loading data samples...'):
            # task_id, cam_id, task_config = self.task_ids[i], self.cam_ids[i], self.task_configs[i]
            task_id, cam_id = self.task_ids[i], self.cam_ids[i]

            # meta_path = os.path.join(self.root, task_config, task_id, 'metadata.json')
            scene_path = os.path.join(self.data_path, task_id)
            target_frame_ids, target_force_list, target_tcp_list, target_offset_list, padding_mask_list = self._get_input_output_lists(self.offset_path, scene_path, cam_id, task_id, num_obs=num_obs, horizon=horizon, frame_sample_step=frame_sample_step)
            self.target_frame_ids += target_frame_ids
            self.target_force_list += target_force_list
            self.target_tcp_list += target_tcp_list
            # self.gripper_list += target_gripper_list
            self.target_offset_list += target_offset_list
            self.padding_mask_list += padding_mask_list
            self.input_task_ids += [task_id] * len(target_frame_ids)
            self.input_cam_ids += [cam_id] * len(target_frame_ids)
            
            
    def __len__(self):
        return len(self.target_frame_ids)
    
    def _get_input_output_lists(self, offset_path, scene_path, cam, task_id, num_obs=1, horizon=1+20, frame_sample_step=1):
        high_freq_data_path = os.path.join(scene_path, 'high_freq_data', 'force_torque_tcp_joint_timestamp.npy')
        high_freq_data = np.load(high_freq_data_path, allow_pickle=True)
        calib_timestamp = np.loadtxt(os.path.join(scene_path, 'timestamp.txt'), dtype=str)
        projector = Projector(os.path.join(self.calib_path, str(calib_timestamp))) 
        # print('calib_path:', os.path.join(self.calib_path, str(calib_timestamp)))
        # metadata = json.load(open(meta_path))
        # label data offset
        # offset_label_data = json.load(open("offset_timestamp.json"))
        # offset_begin = offset_label_data[task_id]["begin"]
        # offset_end = offset_label_data[task_id]["end"]
        # high_freq_data = [x for x in high_freq_data if x['timestamp'] >= offset_begin and x['timestamp'] <= offset_end and x['timestamp'] <= metadata['finish_time']]
        
        frame_id_list = [x[-1] for x in high_freq_data]
        force_raw_list = [x[:6] for x in high_freq_data] # raw data, sensor coordinate
        tcp_list = [x[6:13] for x in high_freq_data] # raw data, base coordinate 
        force_list = [projector.project_force_to_camera_coord(tcp, force, cam) for tcp, force in zip(tcp_list, force_raw_list)]
        force_base_list = [projector.project_force_to_base_coord(tcp, force) for tcp, force in zip(tcp_list, force_raw_list)]
        target_frame_ids = []
        target_force_list = []
        target_tcp_list = []
        # target_gripper_list = []
        target_offset_list = []
        padding_mask_list = []

        if len(frame_id_list) < horizon:
            # padding
            frame_id_list = frame_id_list + frame_id_list[-1:] * (horizon-len(frame_id_list))
            force_list = force_list + force_list[-1:] * (horizon - len(frame_id_list))
            tcp_list = tcp_list + tcp_list[-1:] * (horizon - len(frame_id_list))

        # padding for the first (num_obs-1) frames
        frame_id_list = frame_id_list[0:1] * (num_obs-1) * frame_sample_step + frame_id_list
        force_list = force_list[0:1] * (num_obs-1) * frame_sample_step + force_list
        tcp_list = tcp_list[0:1] * (num_obs-1) * frame_sample_step + tcp_list

        # gripper_data = np.load(gripper_path, allow_pickle=True)[()][cam_id]
        # gripper_timestamp = np.array(list(gripper_data.keys()))

        offset_data = np.load(offset_path, allow_pickle=True)[()][task_id]
        # pick the item with the cam_id equals to cam
        offset_data = [x for x in offset_data if x['cam_id'] == cam]
        offset_timestamp = np.array([x['timestamp'] for x in offset_data])

        
        for i in range(len(frame_id_list)-int(num_obs*frame_sample_step)):
            cur_target_frame_ids = frame_id_list[i:i+horizon*frame_sample_step:frame_sample_step]
            cur_force_list = force_list[i:i+horizon*frame_sample_step:frame_sample_step]
            cur_tcp_list = tcp_list[i:i+horizon*frame_sample_step:frame_sample_step]
            # cur_gripper_list = []
            cur_offset_list = []

            padding_mask = np.zeros(horizon, dtype=bool)
            if len(cur_target_frame_ids) < horizon:
                cur_target_frame_ids += [frame_id_list[-1]] * (horizon - len(cur_target_frame_ids))
                cur_force_list += [force_list[-1]] * (horizon - len(cur_force_list))
                cur_tcp_list += [tcp_list[-1]] * (horizon - len(cur_tcp_list))
                padding_mask[len(cur_target_frame_ids):] = 1

            for i,fid in enumerate(cur_target_frame_ids):
                # nearest_frame = None
                # diffs = np.abs(gripper_timestamp - fid)
                # nearest_frame = gripper_timestamp[np.argmin(diffs)]
                # if i < self.num_obs:
                #     gripper_command = gripper_data[nearest_frame]['gripper_info']
                # else:
                #     gripper_command = gripper_data[nearest_frame]['gripper_command']
                # gripper_width = gripper_command[0] / 1000. # transform mm into m
                # cur_gripper_list.append(gripper_width)

                nearest_offset_frame = None
                diffs = np.abs(offset_timestamp - fid)
                nearest_offset_frame = np.argmin(diffs)
                cur_offset_list.append(offset_data[nearest_offset_frame]['offset'])

            # if len(cur_gripper_list) < horizon:
            #     cur_gripper_list += [cur_gripper_list[-1]] * (horizon - len(cur_gripper_list))
            if len(cur_offset_list) < horizon:
                cur_offset_list += [cur_offset_list[-1]] * (horizon - len(cur_offset_list))

            target_frame_ids.append(cur_target_frame_ids)
            target_force_list.append(cur_force_list)
            target_tcp_list.append(cur_tcp_list)
            # target_gripper_list.append(cur_gripper_list)
            target_offset_list.append(cur_offset_list)
            padding_mask_list.append(padding_mask)

        # return target_frame_ids, target_force_list, target_tcp_list, target_gripper_list, target_offset_list, padding_mask_list
        return target_frame_ids, target_force_list, target_tcp_list, target_offset_list, padding_mask_list
            
        
    def _augmentation(self, force_list):
        ''' force_list: [T, 6]'''
        force_list = force_list + np.random.normal(0, 1, force_list.shape)
        force_list = force_list.astype(np.float32)
        return force_list

    def _clip_tcp(self, tcp_list):
        ''' tcp_list: [T, 8]'''
        tcp_list[:,0] = np.clip(tcp_list[:,0], -0.64, 0.64)
        tcp_list[:,1] = np.clip(tcp_list[:,1], -0.64, 0.64)
        tcp_list[:,2] = np.clip(tcp_list[:,2], 0, 1.28)
        tcp_list[:,7] = np.clip(tcp_list[:,7], 0, 0.11)
        return tcp_list

    def _normalize_tcp(self, tcp_list):
        ''' tcp_list: [T, 8]'''
        if self.top_down_view:
            trans_min, trans_max = np.array([-0.35, -0.35, 0]), np.array([0.35, 0.35, 0.7])
        else:
            trans_min, trans_max = np.array([-0.64, -0.64, 0]), np.array([0.64, 0.64, 1.28])
        max_gripper_width = 0.11 # meter
        tcp_list[:,:3] = (tcp_list[:,:3] - trans_min) / (trans_max - trans_min) * 2 - 1
        tcp_list[:,7] = tcp_list[:,7] / max_gripper_width * 2 - 1
        return tcp_list

    def _normalize_offset(self, offset_list):
        ''' tcp_list: [T, 3(trans) + 6(rot) + 1(width)]'''
        trans_min = np.array([-0.15, -0.15, -0.10])
        trans_max = np.array([0.15, 0.15, 0.10])
        max_gripper_width = 0.11 # meter
        offset_list[:, :3] = (offset_list[:, :3] - trans_min) / (trans_max - trans_min) * 2 - 1
        offset_list[:, -1] = offset_list[:, -1] / max_gripper_width * 2 - 1
        return offset_list
    
    def __getitem__(self, index):
        task_id = self.input_task_ids[index]
        target_frame_ids = self.target_frame_ids[index]
        target_force_list = self.target_force_list[index]
        # target_tcp_list = self.target_tcp_list[index]
        # target_gripper_width_list = self.gripper_list[index]
        target_offset_list = self.target_offset_list[index]
        padding_mask = self.padding_mask_list[index]
        cam_id = self.input_cam_ids[index]

        # load input force-torque data
        input_frame_list = []
        input_frame_list = np.array(target_force_list[:self.num_obs], dtype=np.float32)
                
        # load input and target tcp pose and gripper width
        # target_frame_tcp_list = []
        # target_frame_tcp_list = np.array(target_tcp_list, dtype=np.float32)
        # target_gripper_width_list = np.array(target_gripper_width_list, dtype=np.float32)[:,np.newaxis]

        # load action offset
        target_offset_list = np.array(target_offset_list, dtype=np.float32)

        # put tcp and gripper width together. action: [tcp, width]
        # target_frame_tcp_list = np.concatenate([target_frame_tcp_list, target_gripper_width_list], axis=-1)

        # get normalized tcp
        # target_frame_tcp_list = np.array(target_frame_tcp_list, dtype=np.float32)
        # target_frame_tcp_list = self._clip_tcp(target_frame_tcp_list)
        # target_frame_tcp_normalized = self._normalize_tcp(target_frame_tcp_list.copy())

        # split data
        input_offset_list = target_offset_list[:self.num_obs] # [num_obs, 10]
        target_offset_list = target_offset_list[self.num_obs:] # [horizon, 10]
        # input_frame_tcp_list = target_frame_tcp_list[:self.num_obs] # [num_obs, 8]
        # target_frame_tcp_list = target_frame_tcp_list[self.num_obs:] # [horizon, 8]
        # input_frame_tcp_normalized = target_frame_tcp_normalized[:self.num_obs] # [num_obs, 8]
        # target_frame_tcp_normalized = target_frame_tcp_normalized[self.num_obs:] # [horizon, 8]
        padding_mask = padding_mask[self.num_obs:]

        # make input
        input_frame_list = np.stack(input_frame_list, axis=0) # [num_obs, 6]

        # force torque augmemtation
        input_frame_list = self._augmentation(input_frame_list)

        # force torque standard deviation
        input_frame_list_std = np.std(input_frame_list, axis=0)
        
        vis = False
        if vis:
            import matplotlib.pyplot as plt
            force_x = [x[0] for x in input_frame_list]
            force_y = [x[1] for x in input_frame_list]
            force_z = [x[2] for x in input_frame_list]
            plt.figure()
            plt.plot(target_frame_ids[:self.num_obs], force_x, label='Force X', color='red')  
            plt.plot(target_frame_ids[:self.num_obs], force_y, label='Force Y', color='green') 
            plt.plot(target_frame_ids[:self.num_obs], force_z, label='Force Z', color='blue')
            plt.title('Force Components vs Target Frame IDs')
            plt.xlabel('Target Frame ID')
            plt.ylabel('Force Components')

            plt.ylim(-10, 10)  # 设置y轴的坐标范围为-10到10

            plt.legend()
            plt.show()


        if self.num_obs == 1:
            input_frame_list = input_frame_list[0]
            # input_frame_tcp_list = input_frame_tcp_list[0]
            # input_frame_tcp_normalized = input_frame_tcp_normalized[0]
            input_offset_list = input_offset_list[0]

        target_offset_list_normalized = self._normalize_offset(target_offset_list.copy())
        
        # target_offset_list = torch.from_numpy(target_offset_list).float()
        # target_offset_list_normalized = torch.from_numpy(target_offset_list_normalized).float()

        ret_dict = {'input_frame_list': input_frame_list, # force
                    'input_frame_list_std': input_frame_list_std, # force std
                    # 'input_frame_tcp_list': input_frame_tcp_list,
                    # 'input_frame_tcp_normalized': input_frame_tcp_normalized,
                    'input_offset_list': input_offset_list, # offset
                    # 'target_frame_tcp_list': target_frame_tcp_list,
                    # 'target_frame_tcp_normalized': target_frame_tcp_normalized,
                    'target_offset_list': target_offset_list,
                    'target_offset_list_normalized': target_offset_list_normalized,
                    'padding_mask': padding_mask,
                    'task_id': task_id,
                    'target_frame_ids': target_frame_ids,
                    'cam_id': cam_id}

        return ret_dict

def collate_fn(batch):
    if type(batch[0]).__module__ == 'numpy':
        return torch.stack([torch.from_numpy(b) for b in batch], 0)
    elif isinstance(batch[0], container_abcs.Mapping):
        ret_dict = {}
        for key in batch[0]:
            if key in TO_TENSOR_KEYS:
                ret_dict[key] = collate_fn([d[key] for d in batch])
            else:
                ret_dict[key] = [d[key] for d in batch]
        return ret_dict
    elif isinstance(batch[0], container_abcs.Sequence):
        return [[torch.from_numpy(sample) for sample in b] for b in batch]
    
    raise TypeError("batch must contain tensors, dicts or lists; found {}".format(type(batch[0])))

if __name__ == '__main__':
    dataset = ForceDataset('/aidata/zihao/data/realdata_sampled_20240713', '/aidata/zihao/data/offset/offset_4_0714.npy', num_obs=100, horizon=120 ,selected_tasks=['task_0230'], frame_sample_step=1, selected_cam_ids = ['750612070851']) # when frame_sample_step = 5 -> around 20Hz (initially 100Hz)

    print(len(dataset))
    print(dataset[0]['target_offset_list'])
    print(dataset[1000]['target_offset_list'])
    print(dataset[2000]['target_offset_list'])
    # for idx in range(10000,10005):
    #     print(f"{idx}:")
    #     print(dataset[idx]["input_offset_list"])