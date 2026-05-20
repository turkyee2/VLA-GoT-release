"""
got_pipeline_v2.py
─────────────────
GoT-VLA v2: 경로 누적 score + 가지치기 구조
ToT와의 차별점:
  - 구간별 독립 선택(ToT) → 경로 전체 누적 score로 선택(GoT)
  - KeepBest(1) → 상위 beam_width개 경로 유지 (분기)
  - 마지막에 최선 경로 1개 선택 후 실행 (병합)
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import List, Optional, Callable
import numpy as np
import torch
from PIL import Image


@dataclass
class PathNode:
    """GoT의 노드: 한 구간의 후보 액션 + 경로 누적 score"""
    actions: List[np.ndarray]          # 이 구간의 액션
    segment_score: float = 0.0         # 이 구간의 단독 score
    path_score: float = 0.0            # 루트부터 이 노드까지 누적 score
    parent: Optional['PathNode'] = None # 부모 노드 (이전 구간)
    depth: int = 0                     # 구간 인덱스


def collect_path_actions(node: PathNode) -> List[np.ndarray]:
    """노드에서 루트까지 역추적해서 전체 경로 액션 수집"""
    path = []
    cur = node
    while cur is not None:
        path.append(cur.actions)
        cur = cur.parent
    path.reverse()
    # 구간별 액션을 시간순으로 연결 (GoT의 Aggregate)
    all_actions = []
    for seg_actions in path:
        all_actions.extend(seg_actions)
    return all_actions


class GoTVLAPipelineV2:
    """
    GoT 구조가 명확한 파이프라인:
    1. Generate: 각 경로에서 k개 후보 생성 (분기)
    2. Score: 각 후보의 segment_score 계산
    3. Aggregate: path_score = parent.path_score + segment_score (누적)
    4. Prune: 상위 beam_width개만 유지 (가지치기)
    5. 최종: 최선 경로를 시간순 연결 후 실행 (병합)
    """

    def __init__(self, model, item_processor, cfg, device,
                 world_model=None, wm_item_processor=None):
        self.model = model
        self.item_processor = item_processor
        self.cfg = cfg
        self.device = device
        self.world_model = world_model
        self.wm_item_processor = wm_item_processor or item_processor
        self.last_obs = None
        self.lpips_fn = None

        if cfg.score_fn == "world_model":
            try:
                import lpips as lpips_lib
                self.lpips_fn = lpips_lib.LPIPS(net="vgg").to(device).eval()
            except Exception:
                pass

        # beam_width: 유지할 경로 수 (GoT의 가지치기 파라미터)
        self.beam_width = getattr(cfg, 'beam_width', 2)
        print(f"[GoT-VLA v2] score={cfg.score_fn}, beam_width={self.beam_width}")

    def _generate_candidates(self, cur_img, wrist_img, task_description,
                              his_img, his_wrist_img, cur_state, his_action, k):
        from got_vla_v2.chameleon_got_utils import get_action_for_got
        candidates = []
        for i in range(k):
            try:
                raw = get_action_for_got(
                    model=self.model, cur_img=cur_img, cur_wrist_img=wrist_img,
                    task_description=task_description,
                    item_processor=self.item_processor,
                    his_img=his_img, his_wrist_img=his_wrist_img,
                    cur_state=cur_state, his_action=his_action,
                    his_type=self.cfg.his_type,
                    action_steps=self.cfg.segment_len,
                    do_sample=(i > 0),
                    temperature=1.2 if i > 0 else 1.0,
                    device=self.device,
                )
                if raw:
                    valid = [a.cpu().float().detach().numpy()
                             for a in raw if a.shape[0] == 7]
                    if valid:
                        candidates.append(valid)
            except Exception as e:
                if self.cfg.verbose:
                    print(f"  [Generate] 후보 {i+1} 실패: {e}")
        return candidates

    def _score_actions(self, actions, env, obs, unnorm_fn, cur_img, wrist_img):
        """단일 후보 액션의 score 계산"""
        from got_vla_v2.got_pipeline import ActionCandidate, got_score
        cand = ActionCandidate(actions=actions)
        scored = got_score(
            candidates=[cand], cfg=self.cfg,
            env=env, current_obs=obs,
            unnorm_fn=unnorm_fn,
            current_img=cur_img, wrist_img=wrist_img,
            item_processor=self.wm_item_processor,
            device=self.device,
            world_model=self.world_model,
            lpips_fn=self.lpips_fn,
        )
        return scored[0].score if scored else -float("inf")

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
            print(f"\n[GoT-VLA] {cfg.n_segments}구간 × {cfg.segment_len}스텝, "
                  f"k={cfg.k_candidates}, beam={self.beam_width}, score={cfg.score_fn}")

        # ── Phase 1: Planning (실행 없이 최선 경로 탐색) ──────────────────
        # 루트 노드 (더미)
        root = PathNode(actions=[], segment_score=0.0, path_score=0.0, depth=-1)
        
        # 현재 유지 중인 경로들 (beam)
        active_paths: List[PathNode] = [root]
        
        # 구간별 컨텍스트 (경로마다 독립)
        path_contexts = [{
            'cur_img': cur_img,
            'wrist_img': wrist_img,
            'his_img': list(his_img),
            'his_wrist_img': list(his_wrist_img),
            'cur_state': cur_state,
            'his_action': list(his_action),
            'obs': obs,
        }]

        for seg_idx in range(cfg.n_segments):
            if cfg.verbose:
                print(f"\n[GoT-VLA] ── 구간 {seg_idx+1}/{cfg.n_segments} "
                      f"(활성 경로: {len(active_paths)}개) ──")

            new_nodes: List[PathNode] = []

            for path_idx, parent_node in enumerate(active_paths):
                ctx = path_contexts[path_idx]

                # Generate: 이 경로에서 k개 후보 생성 (분기)
                candidates = self._generate_candidates(
                    cur_img=ctx['cur_img'],
                    wrist_img=ctx['wrist_img'],
                    task_description=task_description,
                    his_img=ctx['his_img'],
                    his_wrist_img=ctx['his_wrist_img'],
                    cur_state=ctx['cur_state'],
                    his_action=ctx['his_action'],
                    k=cfg.k_candidates,
                )

                for cand_actions in candidates:
                    # Score: 이 후보의 단독 score
                    seg_score = self._score_actions(
                        actions=cand_actions,
                        env=ctx['obs'] and env,
                        obs=ctx['obs'],
                        unnorm_fn=unnorm_fn,
                        cur_img=ctx['cur_img'],
                        wrist_img=ctx['wrist_img'],
                    )

                    # Aggregate: 경로 누적 score (GoT의 핵심)
                    path_score = parent_node.path_score + seg_score

                    node = PathNode(
                        actions=cand_actions,
                        segment_score=seg_score,
                        path_score=path_score,
                        parent=parent_node,
                        depth=seg_idx,
                    )
                    new_nodes.append(node)

            if not new_nodes:
                if cfg.verbose:
                    print("  후보 없음, 조기 종료")
                break

            # Prune: 누적 score 기준 상위 beam_width개만 유지 (가지치기)
            new_nodes.sort(key=lambda n: n.path_score, reverse=True)
            active_paths = new_nodes[:self.beam_width]

            if cfg.verbose:
                for i, n in enumerate(active_paths):
                    print(f"  경로{i+1}: seg_score={n.segment_score:.4f}, "
                          f"path_score={n.path_score:.4f}")

            # 다음 구간을 위한 컨텍스트는 WM Score의 예측 이미지 기반
            # (실행 없이 계획만 하므로 현재 컨텍스트 유지)
            path_contexts = [path_contexts[0]] * len(active_paths)

        # ── Phase 2: 최선 경로 선택 및 실행 (병합) ────────────────────────
        if not active_paths or active_paths[0].depth < 0:
            self.last_obs = obs
            return False

        best_path = active_paths[0]  # 누적 score 최선 경로
        all_actions = collect_path_actions(best_path)

        if cfg.verbose:
            print(f"\n[GoT-VLA] 최선 경로 실행: {len(all_actions)}개 액션, "
                  f"총 path_score={best_path.path_score:.4f}")

        # 최선 경로 실행
        seg_obs = obs
        for raw_action in all_actions:
            action_unnorm = unnorm_fn(raw_action)
            seg_obs, reward, done, info = env.step(action_unnorm.tolist())

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

        elapsed = time.time() - t0
        if cfg.verbose:
            print(f"\n[GoT-VLA] 완료: {elapsed:.2f}초")
        self.last_obs = seg_obs
        return False
