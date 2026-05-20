#!/bin/bash
# apt update
# apt install libegl-dev xvfb libgl1-mesa-dri libgl1-mesa-dev libgl1-mesa-glx libstdc++6 -y

lr=5e-6
wd=0.1
dropout=0.05
z_loss_weight=1e-5

data_config_train=../configs/lerobot_3_tasks/his_1_wrist_all_img_state_ck_20_abs_awm_256.yaml
data_config_val_ind=../configs/lerobot_3_tasks/his_1_wrist_all_img_state_ck_20_abs_awm_256.yaml
data_config_val_ood=../configs/lerobot_3_tasks/his_1_wrist_all_img_state_ck_20_abs_awm_256.yaml

base_exp_name=eval_lerobot
base_output_dir=../eval_outputs/"$base_exp_name"
mkdir -p "$base_output_dir"

python ../eval_solver_lerobot_action_head_state.py \
    --device 0 \
    --task_suite_name 'grab blocks' \
    --his 1h_1a_img_only_state \
    --action_dim 6 \
    --time_horizon 20 \
    --no_auto_resume \
    --resume_path "your checkpoint path" \
    --eval_only True \
    --model_size 7B \
    --batch_size 4 \
    --accum_iter 1 \
    --epochs 10 \
    --warmup_epochs 0.01 \
    --lr ${lr} \
    --min_lr ${lr} \
    --wd ${wd} \
    --clip_grad 4 \
    --data_config_train $data_config_train \
    --data_config_val_ind $data_config_val_ind \
    --data_config_val_ood $data_config_val_ood \
    --num_workers 0 \
    --output_dir "$base_output_dir" \
    --checkpointing \
    --max_seq_len 4096 \
    --unmask_image_logits \
    --dropout ${dropout} \
    --z_loss_weight ${z_loss_weight} \
    --ckpt_max_keep 0
    # 2>&1 | tee -a "$base_output_dir"/output.log &