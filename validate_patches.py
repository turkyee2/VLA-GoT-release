import os
import re

GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"

fd_path = "/workspace/rynnvla-002/got_vla_v2/scoring/score_forward_dynamics.py"
pipe_path = "/workspace/rynnvla-002/got_vla_v2/got_pipeline.py"
solver_path = "/workspace/rynnvla-002/eval_solver_libero_got_v2.py"

print("\n🔍 패치 상태 최종 스캔 중...")

# 1. score_forward_dynamics 검증
with open(fd_path, "r") as f:
    fd_code = f.read()
fd_ok = "get_sim_state" in fd_code and "set_state_from_flattened" in fd_code
print(f" - [FD_SCORE] 직렬화 복원 구조: {GREEN}PASS{RESET}" if fd_ok else f" - [FD_SCORE] 직렬화 복원 구조: {RED}FAIL{RESET}")

# 2. got_pipeline 검증
with open(pipe_path, "r") as f:
    pipe_code = f.read()
pipe_ok = "norm_state_fn" in pipe_code and "step_img" in pipe_code
print(f" - [PIPELINE] 상태/이미지 실시간 갱신: {GREEN}PASS{RESET}" if pipe_ok else f" - [PIPELINE] 상태/이미지 실시간 갱신: {RED}FAIL{RESET}")

# 3. eval_solver 검증
with open(solver_path, "r") as f:
    solver_code = f.read()
solver_ok = "load_in_4bit" in solver_code and "load_kwargs" in solver_code
print(f" - [SOLVER] 4-bit 양자화 옵션 지원: {GREEN}PASS{RESET}" if solver_ok else f" - [SOLVER] 4-bit 양자화 옵션 지원: {RED}FAIL{RESET}")

print("\n🎉 모든 항목이 PASS이면 실행하셔도 무방합니다!\n")
