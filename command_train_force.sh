#!/bin/bash
source /home/zihao/anaconda3/etc/profile.d/conda.sh
conda activate rise
python train_force.py --ckpt_dir /data/zihao/ckpts/offset_diffusion_6 --dataset_root /aidata/zihao/data/realdata_sampled_20240713 --offset_path /aidata/zihao/data/offset/offset_5_0714.npy --policy_class Diffusion --task_name task_0230 --batch_size 256 --seed 1 --num_epochs 1000 --save_epochs 50 --lr 1e-5  --num_action 20 --num_obs 100 --obs_feature_dim 64