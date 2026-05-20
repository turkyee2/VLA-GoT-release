"""
got_pipeline_v2.py
──────────────────
GoT-VLA: Graph-of-Thought 구조 파이프라인

ToT(기존)와의 핵심 차이:
  ToT: 구간마다 독립 선택 → KeepBest(1) → 즉시 실행
  GoT: 경로 누적 score → 가지치기 → 병합 → 한번에 실행

GoT 핵심 연산:
  Generate : 각 경로에서 k개 후보 생성 (분기)
  Score    : segment_score 계산
  Aggregate: path_score = parent.path_score + segment_score
  Prune    : 상위 beam_width개만 유지 (가지치기)
  Merge    : 최선 경로 시간순 연결 후 한번에 실행 (병합)
"""
from __future__ import annotations
import time
from dataclasses import dataclass
from typing import List, Optional, Callable
import numpy as np
import torch
from PIL import Image


@dataclass
class GoTConfig:
    n_segments: int = 3
    segment_len: int = 4
    k_candidates: int = 3
    beam_width: int = 2
    action_steps: int = 12
    his_type: str = "his_2_third_view_wrist_w_state"
    score_fn: str = "forward_dynamics"
    fd_n_lookahead: int = 2
    verbose: bool = True


@dataclass
class PathNode:
    """GoT 노드: 한 구간의 액션 + 경로 누적 score + 부모 포인터"""
    actions: List[np.ndarray]
    segment_score: float = 0.0
    path_score: float = 0.0
    parent: Optional['PathNode'] = None
    depth: int = 0


def collect_path_actions(node: PathNode) -> List[np.ndarray]:
    """역추적으로 전체 경로 액션 수집 (Merge 연산)"""
    segments = []
    cur = node
    while cur is not None and cur.depth >= 0:
        segments.append(cur.actions)
        cur = cur.parent
    segments.reverse()
    all_actions = []
    for seg in segments:
        all_actions.extend(seg)
    return all_actions


class GoTVLAPipelineV2:

    def __init__(self, model, item_processor, cfg: GoTConfig,
                 device: torch.device, world_model=None, wm_item_processor=None):
        self.model = model
        self.item_processor = item_processor
        self.cfg = cfg
        self.device = device
        self.world_model = world_model
        self.wm_item_processor = wm_item_processor or item_processor
        self.last_obs = None
        print(f"[GoT-VLA v2] score={cfg.score_fn}, "
              f"n_seg={cfg.n_segments}, k={cfg.k_candidates}, beam={cfg.beam_width}")

    def _generate(self, ctx: dict, k: int) -> List[List[np.ndarray]]:
        """Generate: k개 후보 생성 (분기)"""
        from got_vla_v2.chameleon_got_utils import get_action_for_got
        results = []
        for i in range(k):
            try:
                # 후보별 temperature 차등 적용 (0~1.5 범위)
                # i=0: greedy(1.0), i=1: 1.2, i=2: 1.4
                temps = [1.0, 1.2, 1.4]
                temp = temps[i] if i < len(temps) else 1.4
                raw = get_action_for_got(
                    model=self.model,
                    cur_img=ctx['cur_img'],
                    cur_wrist_img=ctx['wrist_img'],
                    task_description=ctx['task_description'],
                    item_processor=self.item_processor,
                    his_img=ctx['his_img'],
                    his_wrist_img=ctx['his_wrist_img'],
                    cur_state=ctx['cur_state'],
                    his_action=ctx['his_action'],
                    his_type=self.cfg.his_type,
                    action_steps=self.cfg.segment_len,
                    do_sample=(i > 0),
                    temperature=temp,
                    device=self.device,
                )
                if raw:
                    valid = [a.cpu().float().detach().numpy()
                             for a in raw if a.shape[0] == 7]
                    if valid:
                        results.append(valid)
                        if self.cfg.verbose:
                            print(f"    [Generate] 후보{i+1}/{k}: {len(valid)}개")
            except Exception as e:
                if self.cfg.verbose:
                    print(f"    [Generate] 후보{i+1} 실패: {e}")
        return results

    def _score(self, actions: List[np.ndarray], ctx: dict,
               env, unnorm_fn) -> float:
        """Score: segment_score 계산"""
        from got_vla_v2.got_pipeline import ActionCandidate
        cand = ActionCandidate(actions=actions)

        if self.cfg.score_fn == "forward_dynamics" and env is not None:
            from got_vla_v2.scoring.score_forward_dynamics import forward_dynamics_score
            scored = forward_dynamics_score(
                env=env, candidates=[cand], current_obs=ctx['obs'],
                task_description=ctx['task_description'],
                n_lookahead=self.cfg.fd_n_lookahead,
                unnorm_fn=unnorm_fn, verbose=False,
            )
            return scored[0].score if scored else -float("inf")

        elif self.cfg.score_fn == "world_model" and self.world_model is not None:
            from got_vla_v2.scoring.score_world_model import world_model_score
            scored = world_model_score(
                candidates=[cand], current_img=ctx['cur_img'],
                wrist_img=ctx['wrist_img'], world_model=self.world_model,
                item_processor=self.wm_item_processor,
                device=self.device, verbose=False,
            )
            return scored[0].score if scored else -float("inf")

        elif self.cfg.score_fn == "heuristic":
            from got_vla_v2.scoring.score_heuristic import heuristic_score
            scored = heuristic_score([cand], ctx['cur_img'])
            return scored[0].score if scored else 0.0

        return 0.0

    def plan_and_execute(
        self,
        cur_img, wrist_img, task_description,
        his_img, his_wrist_img, cur_state, his_action,
        env, obs, unnorm_fn, get_img_fn, norm_state_fn,
        his_img_ref, his_wrist_ref, his_action_ref,
        replay_images_ref=None,
    ) -> bool:
        cfg = self.cfg
        t0 = time.time()

        if cfg.verbose:
            print(f"\n[GoT-VLA v2] {cfg.n_segments}구간 × {cfg.segment_len}스텝, "
                  f"k={cfg.k_candidates}, beam={cfg.beam_width}, score={cfg.score_fn}")

        init_ctx = {
            'cur_img': cur_img, 'wrist_img': wrist_img,
            'task_description': task_description,
            'his_img': list(his_img), 'his_wrist_img': list(his_wrist_img),
            'cur_state': cur_state, 'his_action': list(his_action),
            'obs': obs,
        }

        # Phase 1: Planning
        root = PathNode(actions=[], path_score=0.0, depth=-1)
        beam: List[PathNode] = [root]
        beam_ctxs = [init_ctx]

        for seg_idx in range(cfg.n_segments):
            if cfg.verbose:
                print(f"\n[GoT-VLA v2] ── 구간 {seg_idx+1}/{cfg.n_segments} "
                      f"(활성 경로 {len(beam)}개) ──")

            new_nodes: List[PathNode] = []

            for parent_node, ctx in zip(beam, beam_ctxs):
                candidates = self._generate(ctx, cfg.k_candidates)

                for cand_actions in candidates:
                    seg_score = self._score(cand_actions, ctx, env, unnorm_fn)
                    path_score = parent_node.path_score + seg_score
                    new_nodes.append(PathNode(
                        actions=cand_actions,
                        segment_score=seg_score,
                        path_score=path_score,
                        parent=parent_node,
                        depth=seg_idx,
                    ))

            if not new_nodes:
                if cfg.verbose:
                    print("  후보 없음, 조기 종료")
                break

            # Prune: 상위 beam_width개 유지
            new_nodes.sort(key=lambda n: n.path_score, reverse=True)
            beam = new_nodes[:cfg.beam_width]
            beam_ctxs = [init_ctx] * len(beam)

            if cfg.verbose:
                for i, n in enumerate(beam):
                    print(f"  [Prune] 경로{i+1}: "
                          f"seg={n.segment_score:.4f}, path={n.path_score:.4f}")

        # Phase 2: Execution (Merge + 실행)
        if not beam or beam[0].depth < 0:
            self.last_obs = obs
            return False

        best_node = beam[0]
        all_actions = collect_path_actions(best_node)  # Merge

        if cfg.verbose:
            print(f"\n[GoT-VLA v2] 최선 경로 실행: "
                  f"{len(all_actions)}개 액션, path_score={best_node.path_score:.4f}")

        seg_obs = obs
        for raw_action in all_actions:
            action_unnorm = unnorm_fn(raw_action)
            seg_obs, reward, done, info = env.step(action_unnorm.tolist())

            step_img, step_wrist = get_img_fn(seg_obs)
            if replay_images_ref is not None:
                replay_images_ref.append(np.array(step_img))

            his_img_ref.append(step_img)
            if len(his_img_ref) > 3: his_img_ref.pop(0)
            his_wrist_ref.append(step_wrist)
            if len(his_wrist_ref) > 3: his_wrist_ref.pop(0)
            his_action_ref.append(raw_action)
            if len(his_action_ref) > 3: his_action_ref.pop(0)

            if done:
                if cfg.verbose:
                    print(f"  [GoT-VLA v2] 태스크 성공!")
                self.last_obs = seg_obs
                return True

        elapsed = time.time() - t0
        if cfg.verbose:
            print(f"\n[GoT-VLA v2] 완료: {elapsed:.2f}초")
        self.last_obs = seg_obs
        return False
