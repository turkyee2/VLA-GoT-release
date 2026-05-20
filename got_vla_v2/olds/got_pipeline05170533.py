"""
got_pipeline.py  (v2)
─────────────────────────────────────────────────────────────────────────────
GoT-VLA 추론 파이프라인.

Score 함수를 모듈로 분리하여 환경에 따라 교체 가능:
  --score_fn heuristic        : 픽셀 분산 (버전 A, 구분 불가)
  --score_fn forward_dynamics : 시뮬레이터 미리 실행 (버전 B, 권장)
  --score_fn world_model      : World Model 이미지 생성 (버전 C, 서버용)

GoT 구조 (GoT 논문 Listing 5와 1:1 대응):
  Generate(k=1, split)  → 궤적을 n구간으로 분할
  Generate(k=3)         → 각 구간에서 k개 후보 생성
  Score                 → 시뮬레이터로 미래 평가
  KeepBest(1)           → 최선 후보 선택
  Aggregate(concat)     → 시간순 연결 (모달리티 붕괴 없음)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from transformers import GenerationConfig


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ActionCandidate:
    """하나의 궤적 구간에 대한 액션 후보."""
    actions: List[np.ndarray]
    score: float = -float("inf")
    score_collision: float = 0.0
    score_consistency: float = 0.0
    predicted_next_obs: Optional[Image.Image] = None


@dataclass
class GoTConfig:
    """GoT-VLA 파이프라인 하이퍼파라미터."""
    # 궤적 분할
    n_segments: int = 3
    segment_len: int = 4

    # 후보 생성
    k_candidates: int = 3
    action_steps: int = 10
    his_type: str = "his_2_third_view_wrist_w_state"

    # Score 함수 선택
    score_fn: str = "forward_dynamics"  # heuristic | forward_dynamics | world_model

    # Forward Dynamics 파라미터
    fd_n_lookahead: int = 2    # 몇 스텝 미리 실행할지

    # World Model 파라미터 (서버용)
    w_collision: float = 0.6
    w_consistency: float = 0.4

    verbose: bool = True


# ─────────────────────────────────────────────────────────────────────────────
# GoT Operations
# ─────────────────────────────────────────────────────────────────────────────

def got_generate(
    model,
    cur_img: Image.Image,
    wrist_img: Optional[Image.Image],
    task_description: str,
    item_processor,
    his_img: List[Image.Image],
    his_wrist_img: List[Image.Image],
    cur_state: np.ndarray,
    his_action: List[np.ndarray],
    cfg: GoTConfig,
    device: torch.device,
    k: int,
) -> List[ActionCandidate]:
    """GoT Generate(k): 구간별 k개 액션 후보 생성."""
    from got_vla_v2.chameleon_got_utils import get_action_for_got

    candidates: List[ActionCandidate] = []

    for i in range(k):
        try:
            do_sample = (i > 0)
            temperature = 1.2 if do_sample else 1.0

            raw_actions = get_action_for_got(
                model=model,
                cur_img=cur_img,
                cur_wrist_img=wrist_img,
                task_description=task_description,
                item_processor=item_processor,
                his_img=his_img,
                his_wrist_img=his_wrist_img,
                cur_state=cur_state,
                his_action=his_action,
                his_type=cfg.his_type,
                action_steps=cfg.segment_len,
                do_sample=do_sample,
                temperature=temperature,
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


def got_score(
    candidates: List[ActionCandidate],
    cfg: GoTConfig,
    # forward_dynamics용
    env=None,
    current_obs: dict = None,
    task_description: str = "",
    unnorm_fn=None,
    # heuristic용
    current_img: Optional[Image.Image] = None,
    # world_model용 (서버)
    world_model=None,
    item_processor=None,
    device=None,
    lpips_fn=None,
    wrist_img: Optional[Image.Image] = None,
) -> List[ActionCandidate]:
    """GoT Score: Score 함수에 따라 후보 평가."""

    if not candidates:
        return candidates

    if cfg.score_fn == "forward_dynamics":
        if env is None or current_obs is None:
            print("[Score] forward_dynamics 선택했지만 env/obs 없음 → heuristic으로 대체")
            cfg.score_fn = "heuristic"
        else:
            from got_vla_v2.scoring.score_forward_dynamics import forward_dynamics_score
            candidates = forward_dynamics_score(
                env=env,
                candidates=candidates,
                current_obs=current_obs,
                task_description=task_description,
                n_lookahead=cfg.fd_n_lookahead,
                unnorm_fn=unnorm_fn,
                verbose=cfg.verbose,
            )
            return candidates

    if cfg.score_fn == "heuristic":
        from got_vla_v2.scoring.score_heuristic import heuristic_score
        candidates = heuristic_score(candidates, current_img)
        return candidates

    if cfg.score_fn == "world_model":
        from got_vla_v2.scoring.score_world_model import world_model_score
        candidates = world_model_score(
            candidates=candidates,
            current_img=current_img,
            wrist_img=wrist_img,
            world_model=world_model,
            item_processor=item_processor,
            device=device,
            lpips_fn=lpips_fn,
            w_collision=cfg.w_collision,
            w_consistency=cfg.w_consistency,
        )
        return candidates

    return candidates


def got_keep_best(candidates: List[ActionCandidate]) -> Optional[ActionCandidate]:
    """GoT KeepBest(1): 최고 점수 후보 반환."""
    if not candidates:
        return None
    return max(candidates, key=lambda c: c.score)


def got_aggregate(segment_results: List[ActionCandidate]) -> List[np.ndarray]:
    """
    GoT Aggregate: 시간축 연결 (concatenation).

    핵심: 값의 평균이 아닌 순서대로 이어붙이기.
    GoT sorting의 merge sort 병합과 동일한 원리.
    """
    full_trajectory: List[np.ndarray] = []
    for seg in segment_results:
        if seg is not None and seg.actions:
            full_trajectory.extend(seg.actions)
    return full_trajectory


# ─────────────────────────────────────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class GoTVLAPipeline:
    """
    GoT-VLA 메인 파이프라인.

    Score 함수를 cfg.score_fn으로 선택:
        forward_dynamics : 시뮬레이터로 미리 실행 (권장)
        heuristic        : 픽셀 분산 (빠르지만 구분 불가)
        world_model      : 서버 환경에서만 사용
    """

    def __init__(self, model, item_processor, cfg: GoTConfig, device: torch.device):
        self.model = model
        self.item_processor = item_processor
        self.cfg = cfg
        self.device = device
        self._lpips_fn = None

        if cfg.score_fn == "world_model":
            self._lpips_fn = self._init_lpips()

        print(f"[GoT-VLA] Score 함수: {cfg.score_fn}")

    def _init_lpips(self):
        try:
            import lpips as lpips_lib
            fn = lpips_lib.LPIPS(net="vgg").to(self.device).eval()
            print("[GoT-VLA] LPIPS 로드 완료")
            return fn
        except Exception as e:
            print(f"[GoT-VLA] LPIPS 없음: {e}")
            return None

    def plan(
        self,
        cur_img: Image.Image,
        wrist_img: Optional[Image.Image],
        task_description: str,
        his_img: List[Image.Image],
        his_wrist_img: List[Image.Image],
        cur_state: np.ndarray,
        his_action: List[np.ndarray],
        # forward_dynamics용 추가 파라미터
        env=None,
        current_obs: dict = None,
        unnorm_fn=None,
    ) -> List[np.ndarray]:
        """
        GoT 전체 계획 실행.

        Graph of Operations:
          foreach segment:
            Generate(k=3) → Score → KeepBest(1)
          Aggregate(concatenate)
        """
        cfg = self.cfg
        t0 = time.time()

        if cfg.verbose:
            print(f"\n[GoT-VLA] {cfg.n_segments}구간 × {cfg.segment_len}스텝, "
                  f"k={cfg.k_candidates}, score={cfg.score_fn}")

        segment_results: List[ActionCandidate] = []

        seg_cur_img = cur_img
        seg_wrist_img = wrist_img
        seg_his_img = list(his_img)
        seg_his_wrist_img = list(his_wrist_img)
        seg_his_action = list(his_action)
        seg_obs = current_obs

        for seg_idx in range(cfg.n_segments):
            if cfg.verbose:
                print(f"\n[GoT-VLA] ── 구간 {seg_idx+1}/{cfg.n_segments} ──")

            # Generate(k=3)
            candidates = got_generate(
                model=self.model,
                cur_img=seg_cur_img,
                wrist_img=seg_wrist_img,
                task_description=task_description,
                item_processor=self.item_processor,
                his_img=seg_his_img,
                his_wrist_img=seg_his_wrist_img,
                cur_state=cur_state,
                his_action=seg_his_action,
                cfg=cfg,
                device=self.device,
                k=cfg.k_candidates,
            )

            if not candidates:
                if cfg.verbose:
                    print(f"  후보 없음, 구간 건너뜀")
                break

            # Score
            candidates = got_score(
                candidates=candidates,
                cfg=cfg,
                env=env,
                current_obs=seg_obs,
                task_description=task_description,
                unnorm_fn=unnorm_fn,
                current_img=seg_cur_img,
                wrist_img=seg_wrist_img,
                item_processor=self.item_processor,
                device=self.device,
                lpips_fn=self._lpips_fn,
            )

            # KeepBest(1)
            best = got_keep_best(candidates)
            if best is None:
                break

            if cfg.verbose:
                print(f"  최선 점수: {best.score:.4f}")

            segment_results.append(best)

            # 다음 구간을 위한 히스토리 업데이트
            if seg_idx < cfg.n_segments - 1:
                seg_his_img = (seg_his_img + [seg_cur_img])[-3:]
                if seg_wrist_img is not None:
                    seg_his_wrist_img = (seg_his_wrist_img + [seg_wrist_img])[-3:]
                if best.actions:
                    seg_his_action = (seg_his_action + best.actions)[-3:]

        # Aggregate(concatenate)
        full_trajectory = got_aggregate(segment_results)

        elapsed = time.time() - t0
        if cfg.verbose:
            print(f"\n[GoT-VLA] 계획 완료: {len(full_trajectory)}개 액션, {elapsed:.2f}초")

        return full_trajectory

    def plan_baseline_bon(
        self,
        cur_img, wrist_img, task_description,
        his_img, his_wrist_img, cur_state, his_action,
        env=None, current_obs=None, unnorm_fn=None,
        k: int = 3,
    ) -> List[np.ndarray]:
        """
        Best-of-N (bon): 분할 없이 전체 시퀀스 k개 생성 후 Score → 최선 선택.
        GoT와의 차이: 시간축 분할 없음.
        """
        from got_vla_v2.chameleon_got_utils import get_action_for_got

        candidates = []
        for i in range(k):
            raw = get_action_for_got(
                model=self.model,
                cur_img=cur_img,
                cur_wrist_img=wrist_img,
                task_description=task_description,
                item_processor=self.item_processor,
                his_img=his_img,
                his_wrist_img=his_wrist_img,
                cur_state=cur_state,
                his_action=his_action,
                his_type=self.cfg.his_type,
                action_steps=self.cfg.n_segments * self.cfg.segment_len,
                do_sample=(i > 0),
                temperature=1.2 if i > 0 else 1.0,
                device=self.device,
            )
            if raw:
                valid = [a.cpu().float().detach().numpy()
                         for a in raw if a.shape[0] == 7]
                if valid:
                    candidates.append(ActionCandidate(actions=valid))

        scored = got_score(
            candidates=candidates,
            cfg=self.cfg,
            env=env,
            current_obs=current_obs,
            task_description=task_description,
            unnorm_fn=unnorm_fn,
            current_img=cur_img,
            wrist_img=wrist_img,
            item_processor=self.item_processor,
            device=self.device,
        )
        best = got_keep_best(scored)
        return best.actions if best else []
