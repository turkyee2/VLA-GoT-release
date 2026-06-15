"""
eval_solver_libero_got_v2.py
────────────────────────────
GoT-VLA 평가 스크립트. LIBERO 벤치마크에서 GoT 추론 파이프라인을 평가한다.

실행 예시:
  # baseline
  python eval_solver_libero_got_v2.py \\
      --resume_path /path/to/ckpts/VLA_model_256/libero_spatial \\
      --tokenizer_path /path/to/ckpts/models--Alpha-VLLM--Lumina-mGPT-7B-768 \\
      --mode baseline --output_dir ./results/baseline

  # GoT + Forward Dynamics Score (권장)
  python eval_solver_libero_got_v2.py \\
      --resume_path /path/to/ckpts/VLA_model_256/libero_spatial \\
      --tokenizer_path /path/to/ckpts/models--Alpha-VLLM--Lumina-mGPT-7B-768 \\
      --mode got --score_fn forward_dynamics \\
      --n_segments 3 --segment_len 4 --k_candidates 3 --beam_width 2 \\
      --output_dir ./results/got_fd

  # GoT + World Model Score
  python eval_solver_libero_got_v2.py \\
      --resume_path /path/to/ckpts/VLA_model_256/libero_spatial \\
      --tokenizer_path /path/to/ckpts/models--Alpha-VLLM--Lumina-mGPT-7B-768 \\
      --mode got --score_fn world_model \\
      --unmask_image_logits \\
      --output_dir ./results/got_wm
"""

import csv
import os
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import tqdm
from PIL import Image

from libero.libero import benchmark
from torch.utils.tensorboard import SummaryWriter

from model import ChameleonXLLMXForConditionalGeneration_ck_action_head
from xllmx.solvers.pretrain import PretrainSolverBase
import xllmx.util as util

from libero_util.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video,
)
from libero_util.robot_utils import set_seed_everywhere
from libero_util.Chameleon_utils import get_action_Chameleon_dis_awm_ck_discrete_action
from data.pre_tokenize_action_state import ItemProcessor

from got_vla_v2.got_pipeline import GoTConfig, GoTVLAPipeline


class GoTSolverV2(PretrainSolverBase):

    def __init__(self, args):
        self.args = args
        util.dist.init_distributed_mode(args)
        self.logger = self.configure_logger()
        self.logger.info(args)

        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        self.logger.info("work dir: {}".format(
            os.path.dirname(os.path.realpath(__file__))))

        (Path(args.output_dir) / "tensorboard").mkdir(parents=True, exist_ok=True)
        self.log_writer = SummaryWriter(
            log_dir=str(Path(args.output_dir) / "tensorboard"))
        self.csv_path = str(Path(args.output_dir) / "results.csv")

    def _make_and_save_starting_point(self, save_path: str) -> None:
        pass

    def _item_processor_func(self):
        return ItemProcessor(target_size=288, tokenizer=self.args.tokenizer_path)

    @classmethod
    def get_args_parser(cls):
        parser = super().get_args_parser()
        parser.add_argument("--max_seq_len", default=4096, type=int)
        parser.add_argument("--mask_image_logits", default=True)
        parser.add_argument("--unmask_image_logits", action="store_false",
                            dest="mask_image_logits")
        parser.add_argument("--dropout", type=float, default=0.0)
        parser.add_argument("--z_loss_weight", type=float, default=0.0)
        parser.add_argument("--model_size", type=str, default="7B")
        parser.add_argument("--task_suite_name", type=str,
                            default="libero_spatial",
                            choices=["libero_spatial", "libero_object",
                                     "libero_goal", "libero_10"])
        parser.add_argument("--device", default=0, type=int)
        parser.add_argument("--his", type=str,
                            default="his_2_third_view_wrist_w_state")
        parser.add_argument("--action_steps", default=12, type=int)
        parser.add_argument("--half", default=0, type=int)
        parser.add_argument("--resolution", default=256, type=int)
        parser.add_argument("--load_in_4bit", action="store_true", default=False)
        parser.add_argument("--tokenizer_path", type=str, required=True)
        # GoT 인자
        parser.add_argument("--mode", type=str, default="got",
                            choices=["got", "baseline"])
        parser.add_argument("--score_fn", type=str, default="forward_dynamics",
                            choices=["heuristic", "forward_dynamics", "world_model"])
        parser.add_argument("--n_segments", type=int, default=3)
        parser.add_argument("--segment_len", type=int, default=4)
        parser.add_argument("--k_candidates", type=int, default=3)
        parser.add_argument("--beam_width", type=int, default=2)
        parser.add_argument("--fd_n_lookahead", type=int, default=2)
        parser.add_argument("--num_trials_per_task", type=int, default=4)
        return parser

    def _model_func(self, init_from: str):
        if getattr(self.args, "load_in_4bit", False):
            from transformers import BitsAndBytesConfig
            model = ChameleonXLLMXForConditionalGeneration_ck_action_head.from_pretrained(
                init_from,
                max_position_embeddings=self.args.max_seq_len,
                mask_image_logits=self.args.mask_image_logits,
                dropout=self.args.dropout,
                z_loss_weight=self.args.z_loss_weight,
                quantization_config=BitsAndBytesConfig(load_in_4bit=True),
                device_map="cpu",
            )
        else:
            model = ChameleonXLLMXForConditionalGeneration_ck_action_head.from_pretrained(
                init_from,
                max_position_embeddings=self.args.max_seq_len,
                mask_image_logits=self.args.mask_image_logits,
                dropout=self.args.dropout,
                z_loss_weight=self.args.z_loss_weight,
                torch_dtype=torch.bfloat16,
                device_map="cpu",
            )
        return model, None

    def unnorm_action_min_max(self, action: np.ndarray) -> np.ndarray:
        min_values = np.array([-0.9375, -0.9375, -0.9375,
                                -0.24214286, -0.375, -0.36428571, -1.0])
        max_values = np.array([0.9375, 0.9375, 0.9375,
                                0.34821429, 0.375, 0.375, 1.0])
        if action.shape[0] > 7:
            action = action[:7]
        return (action + 1) / 2 * (max_values - min_values + 1e-8) + min_values

    def norm_state_min_max(self, state: np.ndarray) -> np.ndarray:
        min_values = np.array([-0.4827807, -0.3309336, 0.00812818, 1.00279467,
                                -3.63125079, -1.84273835, -0.00545302, -0.04201502])
        max_values = np.array([2.10313803e-01, 3.90426440e-01, 1.47277813e+00,
                                3.72486417e+00, 3.56188956e+00, 1.38632160e+00,
                                4.23214189e-02, 1.31260958e-03])
        norm = 2 * (state - min_values) / (max_values - min_values + 1e-8) - 1
        return np.clip(norm, -1, 1)

    def _init_csv(self):
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["task_id", "task_description", "episode_idx",
                             "success", "n_steps", "mode", "score_fn",
                             "n_segments", "segment_len", "k_candidates",
                             "beam_width", "fd_n_lookahead"])

    def _append_csv(self, task_id, task_desc, ep_idx, success, n_steps):
        args = self.args
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([task_id, task_desc, ep_idx, int(success), n_steps,
                             args.mode, args.score_fn,
                             args.n_segments, args.segment_len,
                             args.k_candidates, args.beam_width,
                             args.fd_n_lookahead])

    def val_libero(self):
        args = self.args
        DEVICE = torch.device(f"cuda:{args.device}")

        # 모델 로드
        self.model, _ = self._model_func(args.resume_path)
        self.model = (self.model if getattr(args, "load_in_4bit", False)
                      else self.model.to(DEVICE))
        self.model.eval()

        item_processor = ItemProcessor(
            target_size=args.resolution,
            tokenizer=args.tokenizer_path,
        )

        # GoT 파이프라인 설정
        got_cfg = GoTConfig(
            n_segments=args.n_segments,
            segment_len=args.segment_len,
            k_candidates=args.k_candidates,
            beam_width=args.beam_width,
            action_steps=args.action_steps,
            his_type=args.his,
            score_fn=args.score_fn,
            fd_n_lookahead=args.fd_n_lookahead,
            verbose=True,
        )

        got_pipeline = None
        if args.mode == "got":
            # WM Score: WorldVLA는 Action Model과 World Model이 단일 모델
            world_model = self.model if args.score_fn == "world_model" else None
            got_pipeline = GoTVLAPipeline(
                model=self.model,
                item_processor=item_processor,
                cfg=got_cfg,
                device=DEVICE,
                world_model=world_model,
            )

        # LIBERO 벤치마크 설정
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[args.task_suite_name]()
        num_tasks = task_suite.n_tasks

        print(f"\n{'='*60}")
        print(f"[GoT-VLA] 실험 시작")
        print(f"  mode:       {args.mode}")
        print(f"  score_fn:   {args.score_fn}")
        print(f"  suite:      {args.task_suite_name}")
        print(f"  trials:     {args.num_trials_per_task}")
        print(f"{'='*60}\n")

        self._init_csv()

        if args.half == 1:
            task_range = range(0, num_tasks // 2)
        elif args.half == 2:
            task_range = range(num_tasks // 2, num_tasks)
        else:
            task_range = range(num_tasks)

        total_episodes, total_successes = 0, 0

        for task_id in tqdm.tqdm(task_range):
            task = task_suite.get_task(task_id)
            initial_states = task_suite.get_task_init_states(task_id)
            env, task_description = get_libero_env(task, resolution=args.resolution)

            task_episodes, task_successes = 0, 0

            for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
                print(f"\n[GoT-VLA] Task: {task_description}")
                env.reset()
                obs = env.set_init_state(initial_states[episode_idx])

                t = 0
                done = False
                replay_images = []
                max_steps = {
                    "libero_spatial": 220,
                    "libero_object": 280,
                    "libero_goal": 300,
                    "libero_10": 520,
                }.get(args.task_suite_name, 300)

                his_img: List[Image.Image] = []
                his_wrist_img: List[Image.Image] = []
                his_action: List[np.ndarray] = []
                action_queue: List[np.ndarray] = []

                while t < max_steps + 10:
                    try:
                        # 워밍업 (10스텝)
                        if t < 10:
                            obs, reward, done, info = env.step(
                                get_libero_dummy_action())
                            t += 1
                            continue

                        # 이미지 및 상태 획득
                        img_arr = get_libero_image(obs, args.resolution)
                        wrist_arr = get_libero_image(
                            obs, args.resolution, "robot0_eye_in_hand_image")
                        cur_img = Image.fromarray(img_arr)
                        cur_wrist_img = Image.fromarray(wrist_arr)
                        replay_images.append(img_arr)

                        state_raw = np.concatenate((
                            obs["robot0_eef_pos"],
                            quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"],
                        ))
                        state_normed = self.norm_state_min_max(state_raw)

                        # 액션 계획
                        if len(action_queue) == 0:
                            t_plan = time.time()

                            if args.mode == "got":
                                def get_img_fn(o):
                                    fi = Image.fromarray(
                                        get_libero_image(o, args.resolution))
                                    wi = Image.fromarray(
                                        get_libero_image(
                                            o, args.resolution,
                                            "robot0_eye_in_hand_image"))
                                    return fi, wi

                                def norm_state_fn(o):
                                    raw = np.concatenate((
                                        o["robot0_eef_pos"],
                                        quat2axisangle(o["robot0_eef_quat"]),
                                        o["robot0_gripper_qpos"],
                                    ))
                                    return self.norm_state_min_max(raw)

                                seg_done = got_pipeline.plan_and_execute(
                                    cur_img=cur_img,
                                    wrist_img=cur_wrist_img,
                                    task_description=task_description,
                                    his_img=his_img,
                                    his_wrist_img=his_wrist_img,
                                    cur_state=state_normed,
                                    his_action=his_action,
                                    env=env, obs=obs,
                                    unnorm_fn=self.unnorm_action_min_max,
                                    get_img_fn=get_img_fn,
                                    norm_state_fn=norm_state_fn,
                                    his_img_ref=his_img,
                                    his_wrist_ref=his_wrist_img,
                                    his_action_ref=his_action,
                                    replay_images_ref=replay_images,
                                )
                                obs = got_pipeline.last_obs
                                t += got_cfg.n_segments * got_cfg.segment_len
                                if seg_done:
                                    task_successes += 1
                                    total_successes += 1
                                    done = True
                                    break

                            else:  # baseline
                                raw = get_action_Chameleon_dis_awm_ck_discrete_action(
                                    self.model, cur_img, cur_wrist_img,
                                    task_description, item_processor,
                                    his_img, his_wrist_img,
                                    state_normed, his_action,
                                    args.his, args.action_steps,
                                )
                                for a in raw:
                                    if a.shape[0] == 7:
                                        action_queue.append(
                                            a.cpu().float().detach().numpy())

                            print(f"[GoT-VLA] 계획 {time.time()-t_plan:.2f}초")

                        if args.mode == "got":
                            continue

                        if not action_queue:
                            t += 1
                            continue

                        # 액션 실행 (baseline)
                        raw_action = action_queue.pop(0)
                        action_unnorm = self.unnorm_action_min_max(raw_action)
                        obs, reward, done, info = env.step(action_unnorm.tolist())

                        his_img = (his_img + [cur_img])[-3:]
                        his_wrist_img = (his_wrist_img + [cur_wrist_img])[-3:]
                        his_action = (his_action + [raw_action])[-3:]

                        if done:
                            task_successes += 1
                            total_successes += 1
                            break
                        t += 1

                    except Exception as e:
                        print(f"[GoT-VLA] 예외: {e}")
                        import traceback
                        traceback.print_exc()
                        break

                task_episodes += 1
                total_episodes += 1

                save_rollout_video(
                    args.output_dir, replay_images, total_episodes,
                    success=done, task_description=task_description,
                )
                self._append_csv(task_id, task_description, episode_idx, done, t)
                self.log_writer.add_scalar(
                    f"success/{task_description}", float(done), total_episodes)

                print(f"[GoT-VLA] 성공: {done} | "
                      f"총 {total_successes}/{total_episodes} "
                      f"({total_successes/total_episodes*100:.1f}%)")

            task_sr = float(task_successes) / float(task_episodes)
            print(f"\n[GoT-VLA] 태스크 SR: {task_sr:.3f}")
            self.log_writer.add_scalar(
                f"task_sr/{args.task_suite_name}", task_sr, task_id)

        total_sr = (float(total_successes) / float(total_episodes)
                    if total_episodes > 0 else 0.0)

        print(f"\n{'='*60}")
        print(f"[GoT-VLA] 최종 결과")
        print(f"  mode={args.mode}  score_fn={args.score_fn}")
        print(f"  SR: {total_sr:.4f} ({total_successes}/{total_episodes})")
        print(f"  결과: {self.csv_path}")
        print(f"{'='*60}")

        self.log_writer.add_scalar("final/success_rate", total_sr, 0)
        self.log_writer.close()


if __name__ == "__main__":
    args = GoTSolverV2.get_args_parser().parse_args()
    set_seed_everywhere(7)
    solver = GoTSolverV2(args)
    solver.val_libero()
