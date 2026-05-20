from __future__ import annotations
import numpy as np


def get_env_state(env):
    if hasattr(env, "get_sim_state"):
        return env.get_sim_state()
    if hasattr(env, "sim") and hasattr(env.sim, "get_state"):
        return env.sim.get_state().flatten()
    raise AttributeError("[FD Score] 상태 저장 불가")


def set_env_state(env, state):
    if hasattr(env, "set_state"):
        env.set_state(state)
        return
    if hasattr(env, "sim") and hasattr(env.sim, "set_state_from_flattened"):
        env.sim.set_state_from_flattened(state)
        env.sim.forward()
        return
    raise AttributeError("[FD Score] 상태 복원 불가")


def compute_task_progress(obs: dict, task_description: str) -> float:
    try:
        eef_pos = obs.get("robot0_eef_pos", np.zeros(3))
        gripper_qpos = obs.get("robot0_gripper_qpos", np.zeros(2))
        gripper_closed = float(np.mean(gripper_qpos) < 0.035)
        eef_height = float(eef_pos[2]) if len(eef_pos) > 2 else 0.0
        height_score = min(1.0, max(0.0, (eef_height - 0.82) / 0.25))
        return 0.5 * gripper_closed + 0.5 * height_score
    except Exception:
        return 0.5


def forward_dynamics_score(
    env,
    candidates: list,
    current_obs: dict,
    task_description: str,
    n_lookahead: int = 2,
    unnorm_fn=None,
    verbose: bool = False,
) -> list:
    if not candidates:
        return candidates

    try:
        saved_state = get_env_state(env)
        if verbose:
            print(f"  [FD Score] 상태 저장 완료, {len(candidates)}개 후보 평가 시작")

        for i, cand in enumerate(candidates):
            if not cand.actions:
                cand.score = -float("inf")
                continue

            reward = 0.0
            try:
                score_sum = 0.0
                actual_lookahead = min(n_lookahead, len(cand.actions))

                for step_idx in range(actual_lookahead):
                    raw_action = cand.actions[step_idx]
                    action = unnorm_fn(raw_action) if unnorm_fn else raw_action
                    obs, reward, done, info = env.step(action.tolist())

                    step_weight = (step_idx + 1) / actual_lookahead
                    progress = compute_task_progress(obs, task_description)
                    score_sum += step_weight * (reward * 0.7 + progress * 0.3)

                    if done:
                        score_sum = 10.0
                        break

                cand.score = score_sum / actual_lookahead
                cand.score_collision = float(reward)
                cand.score_consistency = compute_task_progress(obs, task_description)

                if verbose:
                    print(f"  [FD Score] 후보 {i+1}: score={cand.score:.4f} "
                          f"(reward={reward:.3f}, progress={cand.score_consistency:.3f})")

            except Exception as e:
                if verbose:
                    print(f"  [FD Score] 후보 {i+1} 실패: {e}")
                cand.score = -float("inf")

            finally:
                try:
                    set_env_state(env, saved_state)
                except Exception as e:
                    if verbose:
                        print(f"  [FD Score] 복원 실패: {e}")

        if verbose:
            scores = [f"{c.score:.4f}" for c in candidates]
            print(f"  [FD Score] 점수: {scores}")

        return candidates

    except Exception as e:
        print(f"[FD Score] 전체 실패: {e}")
        return candidates
