import pickle
from typing import List, Tuple

from accelerate import init_empty_weights
import torch
import numpy as np

from model import ChameleonXLLMXConfig, ChameleonXLLMXForConditionalGeneration
from xllmx.solvers.pretrain import PretrainSolverBase

import tqdm

from PIL import Image
import imageio


import json


# Append current directory so that interpreter can find experiments.robot
# sys.path.append("../..")

from libero_util.robot_utils import (
    DATE_TIME,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)

from libero_util.Chameleon_utils import get_action_Chameleon_dis_awm_g_video_wrist, reconstruct_img
from data.pre_tokenize_action import ItemProcessor
import time
import xllmx.util as util
from pathlib import Path
import os
from torch.utils.tensorboard import SummaryWriter

def save_rollout_video(rollout_dir, rollout_images, idx, success, task_description):
    """Saves an MP4 replay of an episode."""
    os.makedirs(rollout_dir, exist_ok=True)
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    mp4_path = f"{rollout_dir}/{DATE_TIME}--episode={idx}--success={success}--task={processed_task_description}.mp4"
    video_writer = imageio.get_writer(mp4_path, fps=30)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")
    return mp4_path

class Solver(PretrainSolverBase):
    def __init__(self, args):
        self.args = args
        util.dist.init_distributed_mode(args)
        self.logger = self.configure_logger()
        self.logger.info(args)

        if args.output_dir:
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)

        self.logger.info("work dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
        self.logger.info("{}".format(self.args).replace(", ", ",\n"))

        (Path(args.output_dir) / "tensorboard").mkdir(parents=True, exist_ok=True)
        self.log_writer = SummaryWriter(log_dir=str(Path(args.output_dir) / "tensorboard"))


    @classmethod
    def get_args_parser(cls):
        parser = super().get_args_parser()
        # task-specific parameters
        parser.add_argument("--max_seq_len", default=4096, type=int, help="max token length")
        parser.add_argument("--mask_image_logits", default=True)
        parser.add_argument("--unmask_image_logits", action="store_false", dest="mask_image_logits")
        parser.add_argument("--dropout", type=float, default=0.0)
        parser.add_argument("--z_loss_weight", type=float, default=0.0)
        parser.add_argument("--model_size", type=str, default="7B", choices=["7B", "34B"])
        parser.add_argument("--task_suite_name", type=str, default="spatial", choices=["spatial", "object", "goal", "10",])
        parser.add_argument("--device", default=0, type=int, help="gpu device")
        parser.add_argument("--head", type=str, default="dis", choices=["dis", "ct"])
        parser.add_argument("--his", type=str, default="1h_1a", choices=["1h_1a", "2h_1a", "4h_1a", "2h_2a", "4h_4a", "1a2i"])
        parser.add_argument("--action_steps", default=1, type=int, help="actions to be excuted when multiple actions are generated")
        parser.add_argument("--half", default=0, type=int, help="which part of test set will be evaluated")
        return parser

    def _model_func(
        self,
        init_from: str,
    ) -> (ChameleonXLLMXForConditionalGeneration, None):

        # Only instantiate the model on rank0
        # Other ranks will receive the model weights from rank0 during FSDP wrapping (through `sync_module_states`)
        # See https://github.com/pytorch/pytorch/issues/105840
        model = ChameleonXLLMXForConditionalGeneration.from_pretrained(
            init_from,
            max_position_embeddings=self.args.max_seq_len,
            mask_image_logits=self.args.mask_image_logits,
            dropout=self.args.dropout,
            z_loss_weight=self.args.z_loss_weight,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
        )

        return model, None

    def _item_processor_func(self) -> ItemProcessor:
        return ItemProcessor(target_size=288)

    def _make_and_save_starting_point(self, save_path: str) -> None:

        pretrained_name = {
            "7B": "Alpha-VLLM/Chameleon_7B_mGPT",
            "34B": "Alpha-VLLM/Chameleon_34B_mGPT",
        }[self.args.model_size]

        model = ChameleonXLLMXForConditionalGeneration.from_pretrained(
            pretrained_name,
            max_position_embeddings=self.args.max_seq_len,
            mask_image_logits=self.args.mask_image_logits,
            dropout=self.args.dropout,
            z_loss_weight=self.args.z_loss_weight,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
        )

        image_tokens = model.model.vocabulary_mapping.image_tokens
        model.lm_head.weight.data[image_tokens] = torch.zeros_like(model.lm_head.weight.data[image_tokens])

        model.save_pretrained(save_path, max_shard_size="10GB")
        
    def unnorm_min_max(self, action):
        min_values = np.array([-0.9375, -0.9375, -0.9375, -0.32571429, -0.375, -0.375, -1.0])
        max_values = np.array([0.9375, 0.9375, 0.9375, 0.375, 0.375, 0.375, 1.0])
        if action.shape[0] > 7:
            action = action[:7]
            
        unnorm_action = (action + 1) / 2 * (max_values - min_values + 1e-8) + min_values
        
        return unnorm_action
    
    def val_libero(self,):
        self.model, _ = self._model_func(self.args.resume_path)
        DEVICE = torch.device(f"cuda:{self.args.device}")
        self.model = self.model.to(DEVICE)
        self.model.eval()
        item_processor = ItemProcessor(target_size=512)
        
        file_path = f'{args.task_suite_name}_val_ind_trajectory_paths.json'
        
        with open(file_path, 'r') as f:
            data = json.load(f)

        num_available_gpus = torch.cuda.device_count()
        partition_size = len(data) // num_available_gpus
        if args.half == 0:
            # If half is 0, use the full range
            episode_range = range(len(data))
        elif 1 <= args.half <= num_available_gpus:
            # For half values 1 to num_available_gpus, calculate the corresponding partition
            start_index = (args.half - 1) * partition_size
            end_index = args.half * partition_size if args.half < num_available_gpus else len(data)
            episode_range = range(start_index, end_index)
        else:
            # Handle invalid values of args.half
            raise ValueError("args.half must be an integer between 0 and num_available_gpus.")

        total_episodes = 0
        
        for item in episode_range:
            task_name = ''
            trj_path = data[item]
            # trj_path = trj_path.replace("/mnt/PLNAS/cenjun/libero", "/public/hz_oss/cenjun/libero_data")
            
            action_path = os.path.join(trj_path, 'action')
            imgs_path = os.path.join(trj_path, 'imgs_third_view')
            imgs_path_wrist = os.path.join(trj_path, 'imgs_wrist')
            
            # action_files = sorted([os.path.join(action_path, f) for f in os.listdir(action_path) if f.startswith('action_') and f.endswith('.npy')])
            # img_files = sorted([os.path.join(imgs_path, f) for f in os.listdir(imgs_path) if f.startswith('image_') and f.endswith('.png')])
            action_files = []
            img_files = []
            img_files_wrist = []
            
            for j in range(len(os.listdir(action_path))):
                action_file = os.path.join(action_path, f"action_{j}.npy")
                img_file = os.path.join(imgs_path, f"image_{j}.png")
                img_file_wrist = os.path.join(imgs_path_wrist, f"image_{j}.png")
                img_files.append(img_file)
                img_files_wrist.append(img_file_wrist)
                action_files.append(action_file)
                
            # import pdb; pdb.set_trace()

            print(action_files)
            print(img_files)
            
            his_img = []
            his_action = []
            replay_images = []
            replay_images_gt = []
            replay_images_rt = []
            replay_images_2 = []
            his_img_gt = []
            
            his_img.append(Image.open(img_files[0]))
            his_img_gt.append(Image.open(img_files[0]))
            replay_images.append(np.array(Image.open(img_files[0])))
            replay_images_2.append(np.array(Image.open(img_files[0])))
            replay_images_gt.append(np.array(Image.open(img_files[0])))
            rt_img = reconstruct_img(item_processor, Image.open(img_files[0]))
            replay_images_rt.append(np.array(rt_img))

            his_img_wrist = []
            replay_images_wrist = []
            replay_images_gt_wrist = []
            replay_images_rt_wrist = []
            replay_images_2_wrist = []
            his_img_gt_wrist = []
            
            his_img_wrist.append(Image.open(img_files_wrist[0]))
            his_img_gt_wrist.append(Image.open(img_files_wrist[0]))
            replay_images_wrist.append(np.array(Image.open(img_files_wrist[0])))
            replay_images_2_wrist.append(np.array(Image.open(img_files_wrist[0])))
            replay_images_gt_wrist.append(np.array(Image.open(img_files_wrist[0])))
            rt_img_wrist = reconstruct_img(item_processor, Image.open(img_files_wrist[0]))
            replay_images_rt_wrist.append(np.array(rt_img_wrist))
            
            for i in range(len(action_files)-1):
                try:
                # if 1:
                    his_action.append(action_files[i])

                
                    g_image, g_image_wrist = get_action_Chameleon_dis_awm_g_video_wrist(
                        self.model,
                        task_name,
                        item_processor,
                        his_img,
                        his_img_wrist,
                        his_action,
                        self.args.his,
                    )
                    
                    his_img.append(g_image)
                    # his_img.append(Image.open(img_files[i+1]))
                    replay_images.append(np.array(g_image))
                    replay_images_gt.append(np.array(Image.open(img_files[i+1])))
                    his_img_gt.append(Image.open(img_files[i+1]))

                    rt_img = reconstruct_img(item_processor, Image.open(img_files[i+1]))
                    replay_images_rt.append(np.array(rt_img))

                    task_name = 'front'
                    
                    save_rollout_video(
                        self.args.output_dir, replay_images_gt, item, success='gt', task_description=task_name,
                    )

                    save_rollout_video(
                        self.args.output_dir, replay_images_rt, item, success='gt_recons', task_description=task_name,
                    )
                    
                    save_rollout_video(
                        self.args.output_dir, replay_images, item, success='inf', task_description=task_name,
                    )

                    his_img_wrist.append(g_image_wrist)
                    # his_img.append(Image.open(img_files[i+1]))
                    replay_images_wrist.append(np.array(g_image_wrist))
                    replay_images_gt_wrist.append(np.array(Image.open(img_files_wrist[i+1])))
                    his_img_gt_wrist.append(Image.open(img_files_wrist[i+1]))

                    rt_img_wrist = reconstruct_img(item_processor, Image.open(img_files_wrist[i+1]))
                    replay_images_rt_wrist.append(np.array(rt_img_wrist))

                    task_name = 'wrist'
                    
                    save_rollout_video(
                        self.args.output_dir, replay_images_gt_wrist, item, success='gt', task_description=task_name,
                    )

                    save_rollout_video(
                        self.args.output_dir, replay_images_rt_wrist, item, success='gt_recons', task_description=task_name,
                    )
                    
                    save_rollout_video(
                        self.args.output_dir, replay_images_wrist, item, success='inf', task_description=task_name,
                    )
                    
                except Exception as e:
                    print(f"Caught exception: {e}")
                    break
        
            total_episodes += 1


if __name__ == "__main__":
    args = Solver.get_args_parser().parse_args()
    solver = Solver(args)
    # solver.run()
    solver.val_libero()
    # solver.