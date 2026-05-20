"""
got_pipeline.py
---------------
GoT-VLA: Graph of Thoughts applied to WorldVLA for robot manipulation.

Core idea (from the proposal):
  GoT Sorting             ↔  GoT-VLA (robot trajectory)
  ─────────────────────────────────────────────────────
  Split 64 nums → 4×16    ↔  Split T-step traj → n segments
  Sort each 16 k=5 times  ↔  Generate k action candidates / segment
  Score (error count)     ↔  Score via World Model future prediction
  KeepBest(1)             ↔  KeepBest(1) per segment
  Aggregate (merge sort)  ↔  Aggregate (temporal concatenation)
  Improve                 ↔  (optional) Refine full trajectory

Key design decisions:
  - Aggregate = temporal CONCATENATION (not averaging) → no modality collapse
  - Score = World Model generates future frame → LPIPS consistency score
  - Segment boundary: World Model predicts next obs → feeds into next segment
  - No model retraining required; only inference pipeline changes
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import GenerationConfig

# ─────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────

@dataclass
class ActionCandidate:
    """One action-chunk candidate for a single trajectory segment."""
    actions: List[np.ndarray]          # list of 7-dim action arrays for this segment
    score: float = -float("inf")       # higher = better
    score_collision: float = 0.0
    score_consistency: float = 0.0
    predicted_next_obs: Optional[Image.Image] = None   # World Model prediction


@dataclass
class GoTConfig:
    """All hyper-parameters for the GoT pipeline."""

    # ── Trajectory decomposition ──────────────────────────
    n_segments: int = 3          # n: number of trajectory segments
    segment_len: int = 4         # steps per segment (total horizon ≈ n_segments × segment_len)

    # ── Generation ────────────────────────────────────────
    k_candidates: int = 3        # k: action candidates per segment (GoT Generate(k=3))
    action_steps: int = 25       # action chunk size passed to the model
    his_type: str = "his_1_front_wrist_w_state"   # history type for Chameleon_utils

    # ── Scoring ───────────────────────────────────────────
    w_collision: float = 0.6     # weight for collision-safety score
    w_consistency: float = 0.4   # weight for LPIPS consistency score
    lpips_threshold: float = 0.8 # if LPIPS > this → very inconsistent → penalise

    # ── Misc ──────────────────────────────────────────────
    use_world_model_scoring: bool = True   # set False to fall back to random scoring (debug)
    verbose: bool = True


# ─────────────────────────────────────────────────────────
# Scoring helpers
# ─────────────────────────────────────────────────────────

def _pil_to_tensor(img: Image.Image, device: torch.device) -> torch.Tensor:
    """PIL Image → float32 tensor in [0, 1] with shape (1, C, H, W)."""
    arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    return t


def compute_lpips_score(
    current_img: Image.Image,
    predicted_img: Image.Image,
    lpips_fn,
    device: torch.device,
) -> float:
    """
    Returns a consistency score in [0, 1].
    LPIPS is a perceptual distance (lower = more similar).
    We convert: consistency = max(0, 1 - lpips_dist).

    If lpips_fn is None (lpips not installed) we return 0.5 as neutral.
    """
    if lpips_fn is None or predicted_img is None:
        return 0.5

    try:
        # Resize to same shape if needed
        if current_img.size != predicted_img.size:
            predicted_img = predicted_img.resize(current_img.size, Image.BILINEAR)

        t_cur = _pil_to_tensor(current_img, device) * 2.0 - 1.0   # lpips expects [-1, 1]
        t_pred = _pil_to_tensor(predicted_img, device) * 2.0 - 1.0

        with torch.no_grad():
            dist = lpips_fn(t_cur, t_pred).item()

        consistency = max(0.0, 1.0 - dist)
        return float(consistency)

    except Exception as e:
        print(f"[GoT-VLA] LPIPS scoring failed: {e}")
        return 0.5


def compute_collision_score(
    predicted_img: Image.Image,
) -> float:
    """
    Lightweight collision-safety proxy using pixel-space heuristics.

    Strategy: Check whether the centre region of the predicted frame is
    dominated by a single object (possible collision / occlusion).
    A simple variance-based proxy — low variance in centre crop → likely blocked.

    Range: [0, 1], higher = safer.

    Note: A proper implementation would use the Vision Encoder feature map
    to estimate ego–object proximity. That requires access to model internals
    not exposed by the current API. This heuristic is a drop-in placeholder
    that produces *sensible relative rankings* among candidates without
    any external model.
    """
    if predicted_img is None:
        return 0.5

    try:
        img_arr = np.array(predicted_img.convert("RGB")).astype(np.float32) / 255.0
        H, W = img_arr.shape[:2]
        # Centre crop (middle 40%)
        h0, h1 = int(H * 0.3), int(H * 0.7)
        w0, w1 = int(W * 0.3), int(W * 0.7)
        centre = img_arr[h0:h1, w0:w1]

        # High variance → diverse scene → safer
        variance = float(np.var(centre))
        # Empirically normalise: variance ~0.01 → blocked, ~0.05+ → clear
        score = min(1.0, variance / 0.05)
        return score

    except Exception as e:
        print(f"[GoT-VLA] Collision scoring failed: {e}")
        return 0.5


# ─────────────────────────────────────────────────────────
# World Model interface
# ─────────────────────────────────────────────────────────

def generate_world_model_prediction(
    model,
    current_img: Image.Image,
    wrist_img: Optional[Image.Image],
    action: np.ndarray,
    item_processor,
    device: torch.device,
) -> Optional[Image.Image]:
    """
    WorldVLA World Model: image + action → next image 예측
    Chameleon_utils.py의 get_action_Chameleon_dis_awm_g_video_wrist 방식 그대로 사용
    """
    try:
        img_list = [current_img]
        if wrist_img is not None:
            img_list = [current_img, wrist_img]
        n_imgs = len(img_list)
        img_tags = "<|image|>" * n_imgs

        conv = {
            "conversations": [
                {
                    "from": "human",
                    "value": f"Generate the image based on the current image and the action.{img_tags}<|action|>",
                },
            ],
            "image": img_list,
            "action": [action],
        }

        tokens = item_processor.process_item(conv, training_mode=False)

        generation_config = GenerationConfig(
            max_new_tokens=3000,
            max_length=model.config.max_position_embeddings,
            temperature=1,
            top_k=None,
            do_sample=False,
            eos_token_id=[8710],
        )

        input_ids = torch.tensor(tokens, dtype=torch.int64, device=device).unsqueeze(0)

        with torch.no_grad():
            g_image_tokens = model.generate_img(input_ids, generation_config)

        # 원본 방식: IMG_START/END 토큰으로 첫 번째 이미지 추출
        IMG_START_TOKEN = 8197
        IMG_END_TOKEN = 8196

        tokens_seq = g_image_tokens[0]
        start_indices = torch.where(tokens_seq == IMG_START_TOKEN)[0]
        end_indices = torch.where(tokens_seq == IMG_END_TOKEN)[0]

        if len(start_indices) >= 1 and len(end_indices) >= 1:
            # 첫 번째 이미지 토큰 추출
            img_tokens = tokens_seq[start_indices[0]: end_indices[0] + 1]
            predicted_img = item_processor.decode_image(img_tokens.cpu().tolist())
            return predicted_img
        else:
            print(f"[GoT-VLA] World model: 이미지 토큰 쌍 부족 "
                  f"(start={len(start_indices)}, end={len(end_indices)})")
            return None

    except Exception as e:
        print(f"[GoT-VLA] World model prediction failed: {e}")
        return None


# ─────────────────────────────────────────────────────────
# GoT Operations (Generate / Score / KeepBest / Aggregate)
# ─────────────────────────────────────────────────────────

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
    """
    GoT Generate(k): produce k action-chunk candidates for the current segment.

    Uses temperature sampling (do_sample=True) for candidates 2..k to get
    diversity. Candidate 1 is always greedy (deterministic).
    """
    # Import here to avoid circular imports when this file is used standalone
    from got_vla.chameleon_got_utils import get_action_for_got

    candidates: List[ActionCandidate] = []

    for i in range(k):
        try:
            # Candidate 0: greedy; candidates 1+: sampled for diversity
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

            if raw_actions and len(raw_actions) > 0:
                valid = [a.cpu().float().detach().numpy()
                         for a in raw_actions if a.shape[0] == 7]
                if valid:
                    candidates.append(ActionCandidate(actions=valid))
                    if cfg.verbose:
                        print(f"  [Generate] candidate {i+1}/{k}: {len(valid)} actions")

        except Exception as e:
            print(f"  [Generate] candidate {i+1}/{k} failed: {e}")

    return candidates


def got_score(
    candidates: List[ActionCandidate],
    current_img: Image.Image,
    wrist_img: Optional[Image.Image],
    model,
    item_processor,
    device: torch.device,
    cfg: GoTConfig,
    lpips_fn=None,
) -> List[ActionCandidate]:
    """
    GoT Score: evaluate each candidate using the World Model.

    Score(a) = w1 * S_collision(a) + w2 * S_consistency(a)

    The World Model generates the predicted next frame for each candidate's
    first action, then we compute perceptual metrics on that frame.
    """
    for cand in candidates:
        if not cand.actions:
            cand.score = -float("inf")
            continue

        if not cfg.use_world_model_scoring:
            # Debug fallback: random score
            cand.score = float(np.random.rand())
            continue

        # Use the first action of this candidate to query the World Model
        first_action = cand.actions[0]

        predicted_img = generate_world_model_prediction(
            model=model,
            current_img=current_img,
            wrist_img=wrist_img,
            action=first_action,
            item_processor=item_processor,
            device=device,
        )
        cand.predicted_next_obs = predicted_img

        # S_collision: heuristic on predicted frame
        s_col = compute_collision_score(predicted_img)

        # S_consistency: perceptual similarity current → predicted
        s_con = compute_lpips_score(current_img, predicted_img, lpips_fn, device)

        cand.score_collision = s_col
        cand.score_consistency = s_con
        cand.score = cfg.w_collision * s_col + cfg.w_consistency * s_con

        if cfg.verbose:
            print(f"  [Score] collision={s_col:.3f}  consistency={s_con:.3f}  "
                  f"total={cand.score:.3f}")

    return candidates


def got_keep_best(candidates: List[ActionCandidate]) -> Optional[ActionCandidate]:
    """GoT KeepBest(1): return the highest-scoring candidate."""
    if not candidates:
        return None
    return max(candidates, key=lambda c: c.score)


def got_aggregate(segment_results: List[ActionCandidate]) -> List[np.ndarray]:
    """
    GoT Aggregate: temporal concatenation of segment-optimal action lists.

    This is the KEY operation that differentiates GoT from Best-of-N / ToT.
    We do NOT average actions (which would cause modality collapse).
    Instead we concatenate them in time order, exactly like merge-sort
    concatenates sorted sub-arrays.

    Returns the full trajectory as a flat list of 7-dim numpy arrays.
    """
    full_trajectory: List[np.ndarray] = []
    for seg in segment_results:
        if seg is not None and seg.actions:
            full_trajectory.extend(seg.actions)
    return full_trajectory


# ─────────────────────────────────────────────────────────
# Main GoT-VLA inference loop (per robot step)
# ─────────────────────────────────────────────────────────

class GoTVLAPipeline:
    """
    Drop-in replacement for the single-call action generation in
    eval_solver_libero_discrete_w_state.py.

    Usage:
        pipeline = GoTVLAPipeline(model, item_processor, cfg, device)

        # Each time the action queue is empty:
        trajectory = pipeline.plan(
            cur_img, wrist_img, task_description,
            his_img, his_wrist_img, cur_state, his_action
        )
        # trajectory is a list of np.ndarray (7-dim each)
        # pop from the front and execute one per env step
    """

    def __init__(self, model, item_processor, cfg: GoTConfig, device: torch.device):
        self.model = model
        self.item_processor = item_processor
        self.cfg = cfg
        self.device = device
        self._lpips_fn = self._init_lpips()

    def _init_lpips(self):
        try:
            import lpips as lpips_lib
            fn = lpips_lib.LPIPS(net="vgg").to(self.device).eval()
            print("[GoT-VLA] LPIPS scorer loaded (VGG).")
            return fn
        except Exception as e:
            print(f"[GoT-VLA] LPIPS not available ({e}). Falling back to consistency=0.5.")
            return None

    # ── Graph of Operations (mirrors GoT Listing 5 / Listing 6 structure) ──

    def plan(
        self,
        cur_img: Image.Image,
        wrist_img: Optional[Image.Image],
        task_description: str,
        his_img: List[Image.Image],
        his_wrist_img: List[Image.Image],
        cur_state: np.ndarray,
        his_action: List[np.ndarray],
    ) -> List[np.ndarray]:
        """
        Full GoT planning for one decision horizon.

        GoO (Graph of Operations):
          Step 1  Generate(k=1) — conceptual split (handled by segment loop)
          Step 2  foreach segment:
                    Generate(k=cfg.k_candidates)   ← candidates
                    Score(k=1)                      ← world model evaluation
                    KeepBest(N=1)                   ← best segment action
          Step 3  Aggregate(concatenate)            ← full trajectory
          Step 4  (optional) global Score + log

        Returns list of 7-dim action arrays (the full planned trajectory).
        """
        cfg = self.cfg
        t0 = time.time()

        if cfg.verbose:
            print(f"\n[GoT-VLA] Planning {cfg.n_segments} segments × "
                  f"{cfg.segment_len} steps, k={cfg.k_candidates} candidates each")

        segment_results: List[ActionCandidate] = []

        # Running state for the World-Model bridge between segments
        seg_cur_img = cur_img
        seg_wrist_img = wrist_img
        seg_his_img = list(his_img)
        seg_his_wrist_img = list(his_wrist_img)
        seg_his_action = list(his_action)

        # ── Segment loop: Generate → Score → KeepBest ──────────────────────
        for seg_idx in range(cfg.n_segments):
            if cfg.verbose:
                print(f"\n[GoT-VLA] ── Segment {seg_idx+1}/{cfg.n_segments} ──")

            # Step 2a: Generate k candidates for this segment
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
                    print(f"  [GoT-VLA] Segment {seg_idx+1}: no candidates generated, skipping.")
                break

            # Step 2b: Score each candidate via World Model
            candidates = got_score(
                candidates=candidates,
                current_img=seg_cur_img,
                wrist_img=seg_wrist_img,
                model=self.model,
                item_processor=self.item_processor,
                device=self.device,
                cfg=cfg,
                lpips_fn=self._lpips_fn,
            )

            # Step 2c: KeepBest(1)
            best = got_keep_best(candidates)
            if best is None:
                if cfg.verbose:
                    print(f"  [GoT-VLA] Segment {seg_idx+1}: KeepBest returned None, stopping.")
                break

            if cfg.verbose:
                print(f"  [GoT-VLA] Segment {seg_idx+1} best score: {best.score:.4f}")

            segment_results.append(best)

            # ── World-Model bridge: predict next observation ────────────────
            # This is the causal connection between segments:
            # the best candidate's predicted future frame becomes the
            # "current image" for the next segment.
            # (Mirrors GoT sorting: front-segment output feeds into next merge.)
            if seg_idx < cfg.n_segments - 1:
                if best.predicted_next_obs is not None:
                    seg_cur_img = best.predicted_next_obs
                    if cfg.verbose:
                        print(f"  [GoT-VLA] Bridge: using World Model prediction as next obs.")
                else:
                    # World model scoring was off or failed; re-generate prediction
                    if best.actions:
                        pred = generate_world_model_prediction(
                            model=self.model,
                            current_img=seg_cur_img,
                            wrist_img=seg_wrist_img,
                            action=best.actions[0],
                            item_processor=self.item_processor,
                            device=self.device,
                        )
                        if pred is not None:
                            seg_cur_img = pred
                            if cfg.verbose:
                                print(f"  [GoT-VLA] Bridge: generated World Model prediction.")

                # Update history for next segment
                seg_his_img = (seg_his_img + [seg_cur_img])[-3:]
                if seg_wrist_img is not None:
                    seg_his_wrist_img = (seg_his_wrist_img + [seg_wrist_img])[-3:]
                if best.actions:
                    seg_his_action = (seg_his_action + best.actions)[-3:]

        # ── Step 3: Aggregate (temporal concatenation) ─────────────────────
        full_trajectory = got_aggregate(segment_results)

        elapsed = time.time() - t0
        if cfg.verbose:
            print(f"\n[GoT-VLA] Planning complete: {len(full_trajectory)} actions "
                  f"in {elapsed:.2f}s")

        return full_trajectory

    def plan_baseline_bon(
        self,
        cur_img: Image.Image,
        wrist_img: Optional[Image.Image],
        task_description: str,
        his_img: List[Image.Image],
        his_wrist_img: List[Image.Image],
        cur_state: np.ndarray,
        his_action: List[np.ndarray],
        k: int = 3,
    ) -> List[np.ndarray]:
        """
        Baseline: Best-of-N over the FULL trajectory (no temporal decomposition).
        Equivalent to CoT-SC. Used for ablation comparison (RQ1).
        """
        from got_vla.chameleon_got_utils import get_action_for_got

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
            current_img=cur_img,
            wrist_img=wrist_img,
            model=self.model,
            item_processor=self.item_processor,
            device=self.device,
            cfg=self.cfg,
            lpips_fn=self._lpips_fn,
        )
        best = got_keep_best(scored)
        return best.actions if best else []
