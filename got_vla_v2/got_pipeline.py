from __future__ import annotations
import time
from dataclasses import dataclass
from typing import List, Optional, Callable
import numpy as np
import torch
from PIL import Image


@dataclass
class ActionCandidate:
    actions: List[np.ndarray]
    score: float = -float("inf")
    score_collision: float = 0.0
    score_consistency: float = 0.0


@dataclass
class GoTConfig:
    n_segments: int = 3
    segment_len: int = 4
    k_candidates: int = 3
    action_steps: int = 10
    his_type: str = "his_2_third_view_wrist_w_state"
    score_fn: str = "forward_dynamics"
    fd_n_lookahead: int = 2
    w_collision: float = 0.6
    w_consistency: float = 0.4
    verbose: bool = True


def got_generate(model, cur_img, wrist_img, task_description, item_processor,
                 his_img, his_wrist_img, cur_state, his_action, cfg, device, k):
    from got_vla_v2.chameleon_got_utils import get_action_for_got
    candidates = []
    for i in range(k):
        try:
            do_sample = (i > 0)
            raw_actions = get_action_for_got(
                model=model, cur_img=cur_img, cur_wrist_img=wrist_img,
                task_description=task_description, item_processor=item_processor,
                his_img=his_img, his_wrist_img=his_wrist_img,
                cur_state=cur_state, his_action=his_action,
                his_type=cfg.his_type, action_steps=cfg.segment_len,
                do_sample=do_sample, temperature=1.2 if do_sample else 1.0,
                device=device,
            )
            if raw_actions:
                valid = [a.cpu().float().detach().numpy()
                         for a in raw_actions if a.shape[0] == 7]
                if valid:
                    candidates.append(ActionCandidate(actions=valid))
                    if cfg.verbose:
                        print(f"  [Generate] 후보 {i+1}/{k}: {len(valid)}개 액션")
        except Exception as e:
            print(f"  [Generate] 후보 {i+1}/{k} 실패: {e}")
    return candidates


def got_score(candidates, cfg, env=None, current_obs=None,
              task_description="", unnorm_fn=None, current_img=None,
              wrist_img=None, item_processor=None, device=None, lpips_fn=None, world_model=None,
              his_img=None, his_wrist_img=None):
    if not candidates:
        return candidates
    if cfg.score_fn == "forward_dynamics" and env is not None and current_obs is not None:
        from got_vla_v2.scoring.score_forward_dynamics import forward_dynamics_score
        return forward_dynamics_score(
            env=env, candidates=candidates, current_obs=current_obs,
            task_description=task_description, n_lookahead=cfg.fd_n_lookahead,
            unnorm_fn=unnorm_fn, verbose=cfg.verbose,
        )
    if cfg.score_fn == "heuristic":
        from got_vla_v2.scoring.score_heuristic import heuristic_score
        return heuristic_score(candidates, current_img)
    if cfg.score_fn == "world_model" and world_model is not None:
        from got_vla_v2.scoring.score_world_model import world_model_score
        return world_model_score(
            candidates=candidates,
            current_img=current_img,
            wrist_img=wrist_img,
            world_model=world_model,
            item_processor=item_processor,
            device=device,
            lpips_fn=lpips_fn,
            verbose=cfg.verbose,
            his_img=his_img,
            his_wrist_img=his_wrist_img,
        )
    return candidates


def got_keep_best(candidates):
    if not candidates:
        return None
    return max(candidates, key=lambda c: c.score)


class GoTVLAPipeline:
    def __init__(self, model, item_processor, cfg: GoTConfig, device: torch.device, world_model=None, wm_item_processor=None):
        self.model = model
        self.item_processor = item_processor
        self.wm_item_processor = wm_item_processor if wm_item_processor is not None else item_processor
        self.cfg = cfg
        self.device = device
        self.last_obs = None
        self.world_model = world_model
        self.lpips_fn = None
        if cfg.score_fn == "world_model":
            try:
                import lpips as lpips_lib
                self.lpips_fn = lpips_lib.LPIPS(net="vgg").to(device).eval()
                print("[GoT-VLA] LPIPS 로드 완료")
            except Exception as e:
                print(f"[GoT-VLA] LPIPS 없음: {e}")
        print(f"[GoT-VLA] Score 함수: {cfg.score_fn}")

    def plan_and_execute(
        self,
        cur_img: Image.Image,
        wrist_img: Optional[Image.Image],
        task_description: str,
        his_img: List[Image.Image],
        his_wrist_img: List[Image.Image],
        cur_state: np.ndarray,
        his_action: List[np.ndarray],
        env,
        obs: dict,
        unnorm_fn: Callable,
        get_img_fn: Callable,
        norm_state_fn: Callable,
        his_img_ref: List,
        his_wrist_ref: List,
        his_action_ref: List,
        replay_images_ref: Optional[list] = None,
    ) -> bool:
        cfg = self.cfg
        t0 = time.time()

        seg_cur_img = cur_img
        seg_wrist_img = wrist_img
        seg_his_img = list(his_img)
        seg_his_wrist_img = list(his_wrist_img)
        seg_his_action = list(his_action)
        seg_obs = obs
        seg_cur_state = cur_state  # 버그1: 구간마다 갱신

        if cfg.verbose:
            print(f"\n[GoT-VLA] {cfg.n_segments}구간 × {cfg.segment_len}스텝, "
                  f"k={cfg.k_candidates}, score={cfg.score_fn}")

        for seg_idx in range(cfg.n_segments):
            if cfg.verbose:
                print(f"\n[GoT-VLA] ── 구간 {seg_idx+1}/{cfg.n_segments} ──")

            candidates = got_generate(
                model=self.model, cur_img=seg_cur_img, wrist_img=seg_wrist_img,
                task_description=task_description, item_processor=self.item_processor,
                his_img=seg_his_img, his_wrist_img=seg_his_wrist_img,
                cur_state=seg_cur_state,
                his_action=seg_his_action,
                cfg=cfg, device=self.device, k=cfg.k_candidates,
            )
            if not candidates:
                if cfg.verbose:
                    print(f"  후보 없음, 구간 건너뜀")
                continue

            candidates = got_score(
                candidates=candidates, cfg=cfg,
                env=env, current_obs=seg_obs,
                task_description=task_description, unnorm_fn=unnorm_fn,
                current_img=seg_cur_img, wrist_img=seg_wrist_img,
                item_processor=self.wm_item_processor, device=self.device,
                world_model=self.world_model,
                lpips_fn=self.lpips_fn,
                his_img=seg_his_img,
                his_wrist_img=seg_his_wrist_img,
            )

            best = got_keep_best(candidates)
            if best is None or not best.actions:
                continue

            if cfg.verbose:
                print(f"  최선 점수: {best.score:.4f}")

            # 실제 실행: 매 스텝마다 이미지 캡처 (버그2 수정)
            done = False
            for raw_action in best.actions:
                action_unnorm = unnorm_fn(raw_action)
                seg_obs, reward, done, info = env.step(action_unnorm.tolist())

                # 매 스텝 후 실제 이미지 캡처
                step_img, step_wrist = get_img_fn(seg_obs)
                if replay_images_ref is not None:
                    replay_images_ref.append(np.array(step_img))

                his_img_ref.append(step_img)
                if len(his_img_ref) > 3:
                    his_img_ref.pop(0)
                his_wrist_ref.append(step_wrist)
                if len(his_wrist_ref) > 3:
                    his_wrist_ref.pop(0)
                his_action_ref.append(raw_action)
                if len(his_action_ref) > 3:
                    his_action_ref.pop(0)

                if done:
                    if cfg.verbose:
                        print(f"  [GoT-VLA] 태스크 성공!")
                    self.last_obs = seg_obs
                    return True

            # 다음 구간 준비
            seg_cur_img, seg_wrist_img = get_img_fn(seg_obs)
            seg_his_img = list(his_img_ref)
            seg_his_wrist_img = list(his_wrist_ref)
            seg_his_action = list(his_action_ref)

            # 버그1 수정: cur_state 갱신
            try:
                seg_cur_state = norm_state_fn(seg_obs)
            except Exception:
                pass

        elapsed = time.time() - t0
        if cfg.verbose:
            print(f"\n[GoT-VLA] 계획+실행 완료: {elapsed:.2f}초")
        self.last_obs = seg_obs
        return False

    def plan_baseline_bon(self, cur_img, wrist_img, task_description,
                          his_img, his_wrist_img, cur_state, his_action,
                          env=None, current_obs=None, unnorm_fn=None, k=3):
        from got_vla_v2.chameleon_got_utils import get_action_for_got
        candidates = []
        for i in range(k):
            raw = get_action_for_got(
                model=self.model, cur_img=cur_img, cur_wrist_img=wrist_img,
                task_description=task_description, item_processor=self.item_processor,
                his_img=his_img, his_wrist_img=his_wrist_img,
                cur_state=cur_state, his_action=his_action,
                his_type=self.cfg.his_type,
                action_steps=self.cfg.n_segments * self.cfg.segment_len,
                do_sample=(i > 0), temperature=1.2 if i > 0 else 1.0,
                device=self.device,
            )
            if raw:
                valid = [a.cpu().float().detach().numpy()
                         for a in raw if a.shape[0] == 7]
                if valid:
                    candidates.append(ActionCandidate(actions=valid))
        scored = got_score(
            candidates=candidates, cfg=self.cfg,
            env=env, current_obs=current_obs,
            task_description=task_description, unnorm_fn=unnorm_fn,
            current_img=cur_img, wrist_img=wrist_img,
            item_processor=self.item_processor, device=self.device,
        )
        best = got_keep_best(scored)
        return best.actions if best else []
