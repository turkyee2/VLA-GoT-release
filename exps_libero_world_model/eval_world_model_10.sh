#!/bin/bash
apt update
apt install libegl-dev xvfb libgl1-mesa-dri libgl1-mesa-dev libgl1-mesa-glx libstdc++6 -y
# apt install ffmpeg libsm6 libxext6 libgl1
export LIBGL_DRIVERS_PATH=/usr/lib/x86_64-linux-gnu/dri/
# ln -sf /usr/lib/x86_64-linux-gnu/libstdc++.so.6 /mnt/workspace/workgroup/cenjun.cj/conda/lumina_libero_a800/lib/libstdc++.so.6
ln -sf /usr/lib/x86_64-linux-gnu/libstdc++.so.6 /mnt/nas_jianchong/cenjun.cj/grasping/conda_env/lumina_libero_h20/lib/libstdc++.so.6

lr=1e-5
wd=0.1
dropout=0.05
z_loss_weight=1e-5

data_config_train=../configs/libero_10/his_2_third_view_wrist_w_state_10_256_pretokenize.yaml
data_config_val_ind=../configs/libero_10/his_2_third_view_wrist_w_state_10_256_pretokenize.yaml
data_config_val_ood=../configs/libero_10/his_2_third_view_wrist_w_state_10_256_pretokenize.yaml

base_exp_name=world_model_results
base_output_dir=../eval_outputs/"$base_exp_name"
mkdir -p "$base_output_dir"

for i in {0..7}
do
    torchrun --nnodes=1 --nproc_per_node=1 --master_port=$((17500+i+5)) ../eval_solver_libero_g_video_512_third_wrist.py \
    --device $((i)) \
    --half $((i+1)) \
    --task_suite_name 10 \
    --no_auto_resume \
    --resume_path "your ckpt path" \
    --eval_only True \
    --his 1a2i \
    --disable_length_clustering \
    --ablation 1 \
    --model_size 7B \
    --batch_size 8 \
    --accum_iter 1 \
    --epochs 50 \
    --warmup_epochs 0.01 \
    --lr ${lr} \
    --min_lr ${lr} \
    --wd ${wd} \
    --clip_grad 4 \
    --data_config_train $data_config_train \
    --data_config_val_ind $data_config_val_ind \
    --data_config_val_ood $data_config_val_ood \
    --cache_ann_on_disk \
    --num_workers 8 \
    --output_dir "$base_output_dir" \
    --checkpointing \
    --max_seq_len 8192 \
    --unmask_image_logits \
    --dropout ${dropout} \
    --z_loss_weight ${z_loss_weight} \
    --ckpt_max_keep 0 \
    2>&1 | tee -a "$base_output_dir"/output.log &
done

wait

echo "All experiments completed"
