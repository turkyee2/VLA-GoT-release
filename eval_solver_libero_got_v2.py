"""
eval_solver_libero_got_v2.py
─────────────────────────────────────────────────────────────────────────────
GoT-VLA v2 평가 스크립트.

변경점 (v1 대비):
  - Score 함수 모듈화 (--score_fn 플래그로 선택)
  - Forward Dynamics Score 추가 (환경 미리 실행)
  - env 객체를 pipeline에 전달
  - 버전별 결과 디렉토리 자동 구분

실행 예시:

  # 버전 A: baseline
  python eval_solver_libero_got_v2.py --mode baseline ...

  # 버전 B: GoT + Forward Dynamics Score (권장)
  python eval_solver_libero_got_v2.py --mode got --score_fn forward_dynamics ...

  # 버전 B-2: BoN + Forward Dynamics Score
  python eval_solver_libero_got_v2.py --mode bon --score_fn forward_dynamics ...

  # 버전 C (서버용): GoT + World Model Score
  python eval_solver_libero_got_v2.py --mode got --score_fn world_model ...
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
        # 기존 인자
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
        parser.add_argument("--action_steps", default=10, type=int)
        parser.add_argument("--half", default=0, type=int)
        parser.add_argument("--resolution", default=256, type=int)
        parser.add_argument("--load_in_4bit", action="store_true", default=False)
        parser.add_argument(
            "--tokenizer_path", type=str,
            default="../ckpts/models--Alpha-VLLM--Lumina-mGPT-7B-768/snapshots/"
                    "9624463a82ea5ce814af9b561dcd08a31082c3af")

        # GoT 인자
        parser.add_argument("--mode", type=str, default="got",
                            choices=["got", "bon", "baseline"])
        parser.add_argument("--score_fn", type=str, default="forward_dynamics",
                            choices=["heuristic", "forward_dynamics", "world_model"])
        parser.add_argument("--got_version", type=str, default="v1",
                            choices=["v1", "v2"],
                            help="v1: ToT 방식, v2: GoT beam search 방식")
        parser.add_argument("--beam_width", type=int, default=2,
                            help="GoT v2: 유지할 경로 수")
        parser.add_argument("--n_segments", type=int, default=3)
        parser.add_argument("--segment_len", type=int, default=4)
        parser.add_argument("--k_candidates", type=int, default=3)
        parser.add_argument("--fd_n_lookahead", type=int, default=2,
                            help="Forward Dynamics: 미리 실행할 스텝 수")
        parser.add_argument("--w_collision", type=float, default=0.6)
        parser.add_argument("--w_consistency", type=float, default=0.4)
        parser.add_argument("--num_trials_per_task", type=int, default=50)
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
                device_map='cpu',
            )
        else:
            model = ChameleonXLLMXForConditionalGeneration_ck_action_head.from_pretrained(
                init_from,
                max_position_embeddings=self.args.max_seq_len,
                mask_image_logits=self.args.mask_image_logits,
                dropout=self.args.dropout,
                z_loss_weight=self.args.z_loss_weight,
                torch_dtype=torch.bfloat16,
                device_map='cpu',
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
                             "fd_n_lookahead"])

    def _append_csv(self, task_id, task_desc, ep_idx, success, n_steps):
        args = self.args
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([task_id, task_desc, ep_idx, int(success), n_steps,
                             args.mode, args.score_fn,
                             args.n_segments, args.segment_len, args.k_candidates,
                             args.fd_n_lookahead])

    def val_libero(self):
        args = self.args
        DEVICE = torch.device(f"cuda:{args.device}")

        # 모델 로드
        self.model, _ = self._model_func(args.resume_path)
        self.model = self.model if getattr(self.args, "load_in_4bit", False) else self.model.to(DEVICE)
        self.model.eval()

        # World Model = VLA 모델 자체 (WorldVLA는 단일 모델)
        self.world_model = None
        if args.score_fn == "world_model":
            self.world_model = self.model
            print(f"[GoT-VLA] World Model = VLA 모델 (WorldVLA 단일 모델)")

        item_processor = ItemProcessor(
            target_size=args.resolution,
            tokenizer=args.tokenizer_path,
        )

        # GoT 설정
        got_cfg = GoTConfig(
            n_segments=args.n_segments,
            segment_len=args.segment_len,
            k_candidates=args.k_candidates,
            action_steps=args.action_steps,
            his_type=args.his,
            score_fn=args.score_fn,
            fd_n_lookahead=args.fd_n_lookahead,
            w_collision=args.w_collision,
            w_consistency=args.w_consistency,
            verbose=True,
        )

        if args.mode in ("got", "bon"):
            # WM Score용 256x256 item_processor 별도 생성
            wm_item_processor = ItemProcessor(
                target_size=256,
                tokenizer=args.tokenizer_path,
            ) if args.score_fn == "world_model" else item_processor

            got_pipeline = GoTVLAPipeline(
                model=self.model,
                item_processor=item_processor,
                cfg=got_cfg,
                world_model=self.world_model,
                device=DEVICE,
                wm_item_processor=wm_item_processor,
            )
        else:
            got_pipeline = None

        # LIBERO 설정
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[args.task_suite_name]()
        num_tasks = task_suite.n_tasks

        print(f"\n{'='*60}")
        print(f"[GoT-VLA v2] 실험 시작")
        print(f"  mode:     {args.mode}")
        print(f"  score_fn: {args.score_fn}")
        print(f"  suite:    {args.task_suite_name}")
        print(f"  trials:   {args.num_trials_per_task}")
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
                print(f"\n[GoT-VLA v2] Task: {task_description}")
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

                print(f"[GoT-VLA v2] 에피소드 {task_episodes+1} 시작...")

                while t < max_steps + 10:
                    try:
                        # 워밍업
                        if t < 10:
                            obs, reward, done, info = env.step(
                                get_libero_dummy_action())
                            t += 1
                            continue

                        # 이미지 획득
                        img_arr = get_libero_image(obs, args.resolution)
                        wrist_arr = get_libero_image(
                            obs, args.resolution, "robot0_eye_in_hand_image")
                        cur_img = Image.fromarray(img_arr)
                        cur_wrist_img = Image.fromarray(wrist_arr)
                        replay_images.append(img_arr)

                        # 상태 획득
                        state_raw = np.concatenate((
                            obs["robot0_eef_pos"],
                            quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"],
                        ))
                        state_normed = self.norm_state_min_max(state_raw)

                        # ── 액션 계획 ─────────────────────────────
                        if len(action_queue) == 0:
                            t_plan = time.time()

                            if args.mode == "got":
                                def get_img_fn(o):
                                    fi = Image.fromarray(get_libero_image(o, args.resolution))
                                    wi = Image.fromarray(get_libero_image(o, args.resolution, "robot0_eye_in_hand_image"))
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
                                    env=env,
                                    obs=obs,
                                    unnorm_fn=self.unnorm_action_min_max,
                                    get_img_fn=get_img_fn,
                                    norm_state_fn=norm_state_fn,
                                    his_img_ref=his_img,
                                    his_wrist_ref=his_wrist_img,
                                    his_action_ref=his_action,
                                    replay_images_ref=replay_images,
                                )
                                obs = got_pipeline.last_obs
                                # plan_and_execute가 실행한 스텝 수 반영
                                t += got_cfg.n_segments * got_cfg.segment_len
                                if seg_done:
                                    task_successes += 1
                                    total_successes += 1
                                    done = True
                                    break

                            elif args.mode == "bon":
                                raw_traj = got_pipeline.plan_baseline_bon(
                                    cur_img=cur_img,
                                    wrist_img=cur_wrist_img,
                                    task_description=task_description,
                                    his_img=his_img,
                                    his_wrist_img=his_wrist_img,
                                    cur_state=state_normed,
                                    his_action=his_action,
                                    env=env,
                                    current_obs=obs,
                                    unnorm_fn=self.unnorm_action_min_max,
                                    k=got_cfg.k_candidates,
                                )
                                action_queue.extend(raw_traj)

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

                            print(f"[GoT-VLA v2] 계획 {time.time()-t_plan:.2f}초, "
                                  f"큐={len(action_queue)}")

                        # got 모드는 plan_and_execute가 직접 실행하므로 큐 체크 건너뜀 (t 이중누적 방지)
                        if args.mode == "got":
                            continue



                        # got 모드는 plan_and_execute가 직접 실행하므로 큐 체크 건너뜀 (t 이중누적 방지)
                        if args.mode == "got":
                            continue

                        if not action_queue:
                            t += 1
                            continue

                        # 액션 실행
                        raw_action = action_queue.pop(0)
                        action_unnorm = self.unnorm_action_min_max(raw_action)
                        print(f"  action: {action_unnorm}")

                        obs, reward, done, info = env.step(action_unnorm.tolist())

                        # 히스토리 업데이트
                        his_img = (his_img + [cur_img])[-3:]
                        his_wrist_img = (his_wrist_img + [cur_wrist_img])[-3:]
                        his_action = (his_action + [raw_action])[-3:]

                        if done:
                            task_successes += 1
                            total_successes += 1
                            break
                        t += 1

                    except Exception as e:
                        print(f"[GoT-VLA v2] 예외: {e}")
                        import traceback
                        traceback.print_exc()
                        break

                task_episodes += 1
                total_episodes += 1

                save_rollout_video(
                    args.output_dir, replay_images, total_episodes,
                    success=done, task_description=task_description,
                )
                self._append_csv(task_id, task_description,
                                 episode_idx, done, t)
                self.log_writer.add_scalar(
                    f"success/{task_description}", float(done), total_episodes)

                print(f"[GoT-VLA v2] 성공: {done} | "
                      f"총 {total_successes}/{total_episodes} "
                      f"({total_successes/total_episodes*100:.1f}%)")

            task_sr = float(task_successes) / float(task_episodes)
            print(f"\n[GoT-VLA v2] 태스크 SR: {task_sr:.3f}")
            self.log_writer.add_scalar(
                f"task_sr/{args.task_suite_name}", task_sr, task_id)

        total_sr = (float(total_successes) / float(total_episodes)
                    if total_episodes > 0 else 0.0)

        print(f"\n{'='*60}")
        print(f"[GoT-VLA v2] 최종 결과")
        print(f"  mode={args.mode}  score_fn={args.score_fn}")
        print(f"  suite={args.task_suite_name}")
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
