"""
eval_solver_libero_got.py
─────────────────────────────────────────────────────────────────────────────
GoT-VLA evaluation script for LIBERO benchmarks.

Based on: rynnvla-002/eval_solver_libero_discrete_w_state.py
Changes:
  1. Replaced single get_action_Chameleon_dis_awm_ck_discrete_action call
     with GoTVLAPipeline.plan() (GoT full pipeline).
  2. Added --mode flag: got | bon | baseline
     - got      : GoT temporal decomposition (proposed method)
     - bon      : Best-of-N over full trajectory (ablation, CoT-SC equivalent)
     - baseline : Original single-pass inference (no GoT)
  3. Added GoT hyper-params as CLI args.
  4. Results logged per-segment and overall to tensorboard + CSV.

Run (example):
  python rynnvla-002/eval_solver_libero_got.py \\
      --resume_path /path/to/checkpoint \\
      --tokenizer_path /path/to/tokenizer \\
      --task_suite_name libero_spatial \\
      --mode got \\
      --n_segments 3 \\
      --segment_len 4 \\
      --k_candidates 3 \\
      --output_dir ./results/got_libero_spatial \\
      --device 0

For baseline (no GoT):
  python rynnvla-002/eval_solver_libero_got.py \\
      --resume_path /path/to/checkpoint \\
      --tokenizer_path /path/to/tokenizer \\
      --task_suite_name libero_spatial \\
      --mode baseline \\
      --output_dir ./results/baseline_libero_spatial \\
      --device 0
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

# ── Original RynnVLA imports (unchanged) ──────────────────────────────────
from model import (
    ChameleonXLLMXConfig,
    ChameleonXLLMXForConditionalGeneration_ck_action_head,
)
from xllmx.solvers.pretrain import PretrainSolverBase
import xllmx.util as util

from libero_util.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video,
)
from libero_util.robot_utils import (
    DATE_TIME,
    set_seed_everywhere,
)
from libero_util.Chameleon_utils import get_action_Chameleon_dis_awm_ck_discrete_action
from data.pre_tokenize_action_state import ItemProcessor

# ── GoT-VLA imports ────────────────────────────────────────────────────────
from got_vla import GoTConfig, GoTVLAPipeline


# ─────────────────────────────────────────────────────────────────────────
# Solver class
# ─────────────────────────────────────────────────────────────────────────

class GoTSolver(PretrainSolverBase):
    """Extended solver with GoT-VLA planning pipeline."""

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

        # CSV log path
        self.csv_path = str(Path(args.output_dir) / "results.csv")

    # ── Boilerplate from original solver ──────────────────────────────────

    @classmethod
    def get_args_parser(cls):
        parser = super().get_args_parser()
        # ── original args ──
        parser.add_argument("--max_seq_len", default=4096, type=int)
        parser.add_argument("--mask_image_logits", default=True)
        parser.add_argument("--unmask_image_logits", action="store_false", dest="mask_image_logits")
        parser.add_argument("--dropout", type=float, default=0.0)
        parser.add_argument("--z_loss_weight", type=float, default=0.0)
        parser.add_argument("--model_size", type=str, default="7B", choices=["7B", "34B"])
        parser.add_argument(
            "--task_suite_name", type=str, default="libero_spatial",
            choices=["libero_spatial", "libero_object", "libero_goal", "libero_10"],
        )
        parser.add_argument("--device", default=0, type=int)
        parser.add_argument("--his", type=str, default="his_1_front_wrist_w_state")
        parser.add_argument("--action_steps", default=25, type=int)
        parser.add_argument("--half", default=0, type=int)
        parser.add_argument("--resolution", default=256, type=int)
        parser.add_argument(
            "--tokenizer_path", type=str,
            default="../ckpts/models--Alpha-VLLM--Lumina-mGPT-7B-768/snapshots/"
                    "9624463a82ea5ce814af9b561dcd08a31082c3af",
        )
        # ── GoT-specific args ──────────────────────────────────────────────
        parser.add_argument(
            "--mode", type=str, default="got",
            choices=["got", "bon", "baseline"],
            help=(
                "got      = GoT temporal decomposition (proposed method)\n"
                "bon      = Best-of-N over full trajectory (ablation)\n"
                "baseline = Original single-pass (no GoT)"
            ),
        )
        parser.add_argument("--n_segments", type=int, default=3,
                            help="Number of trajectory segments (n in GoO)")
        parser.add_argument("--segment_len", type=int, default=4,
                            help="Actions per segment")
        parser.add_argument("--k_candidates", type=int, default=3,
                            help="Candidates per segment (k in Generate(k))")
        parser.add_argument("--w_collision", type=float, default=0.6)
        parser.add_argument("--w_consistency", type=float, default=0.4)
        parser.add_argument("--no_world_model_scoring", action="store_true",
                            help="Disable World Model scoring (use heuristic only)")
        parser.add_argument("--num_trials_per_task", type=int, default=50)
        return parser

    def _model_func(self, init_from: str):
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

    def _make_and_save_starting_point(self, save_path: str) -> None:
        """Required by PretrainSolverBase. Not used in evaluation."""
        pass

    def _item_processor_func(self) -> ItemProcessor:
        return ItemProcessor(target_size=288, tokenizer=self.args.tokenizer_path)

    # ── Normalisation (unchanged from original) ───────────────────────────

    def unnorm_action_min_max(self, action: np.ndarray) -> np.ndarray:
        min_values = np.array([-0.9375, -0.9375, -0.9375, -0.24214286, -0.375, -0.36428571, -1.0])
        max_values = np.array([0.9375, 0.9375, 0.9375, 0.34821429, 0.375, 0.375, 1.0])
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

    # ── CSV helper ────────────────────────────────────────────────────────

    def _init_csv(self):
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "task_id", "task_description", "episode_idx",
                "success", "n_steps", "mode",
                "n_segments", "segment_len", "k_candidates",
            ])

    def _append_csv(self, task_id, task_desc, ep_idx, success, n_steps):
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                task_id, task_desc, ep_idx,
                int(success), n_steps, self.args.mode,
                self.args.n_segments, self.args.segment_len, self.args.k_candidates,
            ])

    # ── Main evaluation ───────────────────────────────────────────────────

    def val_libero(self):
        args = self.args
        DEVICE = torch.device(f"cuda:{args.device}")

        # Load model
        self.model, _ = self._model_func(args.resume_path)
        self.model = self.model.to(DEVICE)
        self.model.eval()

        item_processor = ItemProcessor(
            target_size=args.resolution,
            tokenizer=args.tokenizer_path,
        )

        # Build GoT config
        got_cfg = GoTConfig(
            n_segments=args.n_segments,
            segment_len=args.segment_len,
            k_candidates=args.k_candidates,
            action_steps=args.action_steps,
            his_type=args.his,
            w_collision=args.w_collision,
            w_consistency=args.w_consistency,
            use_world_model_scoring=not args.no_world_model_scoring,
            verbose=True,
        )

        # Build GoT pipeline (only used when mode != baseline)
        if args.mode in ("got", "bon"):
            got_pipeline = GoTVLAPipeline(
                model=self.model,
                item_processor=item_processor,
                cfg=got_cfg,
                device=DEVICE,
            )
        else:
            got_pipeline = None

        # LIBERO setup
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[args.task_suite_name]()
        num_tasks = task_suite.n_tasks
        print(f"[GoT-VLA] Task suite: {args.task_suite_name}  |  mode: {args.mode}")
        print(f"[GoT-VLA] GoT config: {got_cfg}")

        self._init_csv()

        # Task range
        if args.half == 0:
            task_range = range(num_tasks)
        elif args.half == 1:
            task_range = range(0, num_tasks // 2)
        else:
            task_range = range(num_tasks // 2, num_tasks)

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
                    "libero_90": 400,
                }.get(args.task_suite_name, 300)

                # History buffers
                his_img: List[Image.Image] = []
                his_wrist_img: List[Image.Image] = []
                his_action: List[np.ndarray] = []
                action_queue: List[np.ndarray] = []  # pre-planned actions to execute

                print(f"[GoT-VLA] Starting episode {task_episodes + 1}...")
                while t < max_steps + 10:
                    try:
                        # Warm-up: stabilise simulator
                        if t < 10:
                            obs, reward, done, info = env.step(get_libero_dummy_action())
                            t += 1
                            continue

                        # Get images
                        img_arr = get_libero_image(obs, args.resolution)
                        wrist_arr = get_libero_image(obs, args.resolution, "robot0_eye_in_hand_image")
                        cur_img = Image.fromarray(img_arr)
                        cur_wrist_img = Image.fromarray(wrist_arr)
                        replay_images.append(img_arr)

                        # State
                        state = np.concatenate((
                            obs["robot0_eef_pos"],
                            quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"],
                        ))

                        # ── ACTION PLANNING ────────────────────────────────
                        if len(action_queue) == 0:
                            t_plan = time.time()

                            if args.mode == "got":
                                # ▶ GoT: temporal decomposition
                                raw_traj = got_pipeline.plan(
                                    cur_img=cur_img,
                                    wrist_img=cur_wrist_img,
                                    task_description=task_description,
                                    his_img=his_img,
                                    his_wrist_img=his_wrist_img,
                                    cur_state=state,
                                    his_action=his_action,
                                )
                                action_queue.extend(raw_traj)

                            elif args.mode == "bon":
                                # ▶ Best-of-N (ablation)
                                raw_traj = got_pipeline.plan_baseline_bon(
                                    cur_img=cur_img,
                                    wrist_img=cur_wrist_img,
                                    task_description=task_description,
                                    his_img=his_img,
                                    his_wrist_img=his_wrist_img,
                                    cur_state=state,
                                    his_action=his_action,
                                    k=got_cfg.k_candidates,
                                )
                                action_queue.extend(raw_traj)

                            else:
                                # ▶ Baseline: original single-pass (no GoT)
                                raw = get_action_Chameleon_dis_awm_ck_discrete_action(
                                    self.model, cur_img, cur_wrist_img,
                                    task_description, item_processor,
                                    his_img, his_wrist_img, state, his_action,
                                    args.his, args.action_steps,
                                )
                                for a in raw:
                                    if a.shape[0] == 7:
                                        action_queue.append(
                                            a.cpu().float().detach().numpy()
                                        )

                            print(f"[GoT-VLA] Planning took {time.time()-t_plan:.2f}s, "
                                  f"queue={len(action_queue)}")

                        if not action_queue:
                            print("[GoT-VLA] Empty action queue — skipping step.")
                            t += 1
                            continue

                        # Pop and execute one action
                        raw_action = action_queue.pop(0)
                        action_unnorm = self.unnorm_action_min_max(raw_action)
                        print(f"  action: {action_unnorm}")

                        obs, reward, done, info = env.step(action_unnorm.tolist())

                        # Update history
                        his_img.append(cur_img)
                        his_wrist_img.append(cur_wrist_img)
                        his_action.append(raw_action)
                        # Keep only last 3
                        his_img = his_img[-3:]
                        his_wrist_img = his_wrist_img[-3:]
                        his_action = his_action[-3:]

                        if done:
                            task_successes += 1
                            total_successes += 1
                            break
                        t += 1

                    except Exception as e:
                        print(f"[GoT-VLA] Exception: {e}")
                        import traceback; traceback.print_exc()
                        break

                task_episodes += 1
                total_episodes += 1

                # Save video
                save_rollout_video(
                    args.output_dir, replay_images, total_episodes,
                    success=done, task_description=task_description,
                )

                # Log
                self._append_csv(task_id, task_description, episode_idx, done, t)
                self.log_writer.add_scalar(
                    f"success/{task_description}", float(done), total_episodes
                )

                print(f"[GoT-VLA] Success: {done}")
                print(f"[GoT-VLA] Episodes: {total_episodes}  "
                      f"Successes: {total_successes} "
                      f"({total_successes/total_episodes*100:.1f}%)")

            # Task summary
            task_sr = float(task_successes) / float(task_episodes)
            print(f"\n[GoT-VLA] Task '{task_description}' SR: {task_sr:.3f}")
            self.log_writer.add_scalar(
                f"task_success_rate/{args.task_suite_name}",
                task_sr, task_id,
            )

        total_sr = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0.0
        print(f"\n{'='*60}")
        print(f"[GoT-VLA] FINAL  mode={args.mode}  suite={args.task_suite_name}")
        print(f"[GoT-VLA] Total SR: {total_sr:.4f}  ({total_successes}/{total_episodes})")
        print(f"[GoT-VLA] Results saved to: {self.csv_path}")
        print(f"{'='*60}")

        self.log_writer.add_scalar("final/success_rate", total_sr, 0)
        self.log_writer.close()


# ─────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = GoTSolver.get_args_parser().parse_args()
    set_seed_everywhere(7)
    solver = GoTSolver(args)
    solver.val_libero()
