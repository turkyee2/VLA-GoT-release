# GoT-VLA: Graph-of-Thought for Vision-Language-Action Models

WorldVLA(RynnVLA-002) 기반 로봇 조작 모델에 Graph-of-Thought 추론 구조를 적용한 프레임워크입니다. LIBERO 벤치마크에서 검증되었으며, 모델 재학습 없이 추론 파이프라인만 교체합니다.

## 핵심 아이디어

기존 VLA 모델은 카메라 이미지와 언어 지시를 받아 전체 행동 궤적을 한 번에 생성합니다. 이 방식은 앞부분 오류가 뒷부분으로 누적되고, 대안을 탐색하는 메커니즘이 없습니다.

GoT-VLA는 GoT(Besta et al., 2023) 논문의 분할-병합 전략을 시간축에 적용합니다. 12스텝 궤적을 3개 구간으로 나누고, 각 구간에서 여러 후보를 생성한 뒤 누적 score 기준으로 최선 경로를 선택합니다.

기존 VLA:   이미지 + 지시 → [a1, a2, ..., a12] 한번에 생성
GoT-VLA:    구간1 [a1..a4] + 구간2 [a5..a8] + 구간3 [a9..a12]
↑ 각 구간에서 k=3개 후보 생성, 누적 score로 최선 경로 선택

## 구조

### GoT 핵심 연산

Generate       : 각 경로에서 k=3 후보 생성 (temperature 1.0/1.2/1.4 차등 적용)
Score          : segment_score 계산 (forward_dynamics / world_model / heuristic)
Aggregate      : path_score = parent.path_score + segment_score (누적)
Prune          : 상위 beam_width=2 경로만 유지 (가지치기)
Context Update : best 후보를 짧게 시뮬레이션 → 다음 구간 컨텍스트 갱신 → 환경 복원 (엣지)
Merge          : 최선 경로 역추적 → 시간순 연결 → 한번에 실행 (병합)

### 두 단계 파이프라인

**Phase 1 — Planning** (환경 변경 없이 최선 경로 탐색)

각 구간마다 반복:
1. Generate: 활성 beam 경로에서 k=3 후보 생성
2. Score: 각 후보의 segment_score 계산
3. Aggregate: path_score 누적
4. Prune: 상위 beam_width 경로만 유지
5. Context Update: best 후보를 n_lookahead 스텝 시뮬레이션 → obs 갱신 → 환경 복원

**Phase 2 — Execution**

최선 경로의 전체 액션을 시간순으로 한번에 실행합니다.

### Score 함수

| 종류 | 방식 | 실배포 가능 | 속도 |
|------|------|------------|------|
| heuristic | 픽셀 분산 (후보 구분 불가, 구조 효과 측정용) | — | 빠름 |
| forward_dynamics | 시뮬레이터 미리 실행 | ✗ (시뮬 전용) | 빠름 |
| world_model | World Model 이미지 생성 | ✓ | 느림 |

> World Model Score는 모델 추론만으로 동작하므로 실제 로봇 배포가 가능합니다.
> Forward Dynamics Score는 시뮬레이터 상태 저장/복원이 필요하므로 시뮬레이터 환경 전용입니다.

## 디렉토리 구조


VLA-GoT-release/

├── got_vla_v2/
│   ├── init.py
│   ├── got_pipeline.py              # GoT 파이프라인 (Phase 1/2, beam search, Context Update)
│   ├── chameleon_got_utils.py       # 액션 생성 유틸 (temperature 차등, top_k 제거, top_p=0.95)
│   └── scoring/
│       ├── score_heuristic.py
│       ├── score_forward_dynamics.py
│       └── score_world_model.py
├── eval_solver_libero_got_v2.py     # 메인 실행 스크립트
├── test_got_pipeline.py             # 구조 검증 단위 테스트 (모델 없이 실행 가능)
├── requirements.txt
├── model/                           # VLA 모델 클래스
├── data/                            # ItemProcessor
├── libero_util/                     # LIBERO 환경 유틸
└── xllmx/                           # 기반 프레임워크


## 실행 방법

### 환경 설정

```bash
conda create -n got_vla python=3.10 -y
conda activate got_vla

pip install torch==2.2.0 torchvision==0.17.0 \
    --index-url https://download.pytorch.org/whl/cu121
pip install numpy==1.26.4

cd VLA-GoT-release && pip install -e .
pip install -r requirements.txt

git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO && pip install -e .
```

### 환경변수

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONPATH=/home/$USER/LIBERO:/home/$USER/VLA-GoT-release:/home/$USER
```

### 모델 체크포인트 다운로드

```python
from huggingface_hub import snapshot_download
import os

# VLA 액션 모델 (필수, ~15GB) — 반드시 RynnVLA-002 repo에서 받을 것
snapshot_download(
    repo_id='Alibaba-DAMO-Academy/RynnVLA-002',
    allow_patterns='VLA_model_256/libero_spatial/*',
    local_dir=os.path.expanduser('~/ckpts'),
)

# World Model (got_wm 전용, ~15GB)
snapshot_download(
    repo_id='Alibaba-DAMO-Academy/RynnVLA-002',
    allow_patterns='Action_World_model_512/libero_spatial/*',
    local_dir=os.path.expanduser('~/ckpts'),
)

# Lumina 토크나이저 (~14GB)
snapshot_download(
    repo_id='Alpha-VLLM/Lumina-mGPT-7B-768',
    local_dir=os.path.expanduser('~/ckpts/models--Alpha-VLLM--Lumina-mGPT-7B-768'),
)
```

> ⚠️ WorldVLA repo의 `model_256`은 입력 구성이 달라 SR 0%가 나오는 silent failure가 발생합니다. 반드시 RynnVLA-002 repo의 `VLA_model_256`을 사용하세요.

### 구조 테스트 (모델 없이)

```bash
python test_got_pipeline.py
```

### baseline

```bash
CUDA_VISIBLE_DEVICES=0 python eval_solver_libero_got_v2.py \
    --resume_path ~/ckpts/VLA_model_256/libero_spatial \
    --tokenizer_path ~/ckpts/models--Alpha-VLLM--Lumina-mGPT-7B-768 \
    --task_suite_name libero_spatial \
    --his his_2_third_view_wrist_w_state \
    --action_steps 12 \
    --mode baseline \
    --num_trials_per_task 4 \
    --output_dir ./results/baseline
```

### GoT + Forward Dynamics Score (got_fd)

```bash
CUDA_VISIBLE_DEVICES=0 python eval_solver_libero_got_v2.py \
    --resume_path ~/ckpts/VLA_model_256/libero_spatial \
    --tokenizer_path ~/ckpts/models--Alpha-VLLM--Lumina-mGPT-7B-768 \
    --task_suite_name libero_spatial \
    --his his_2_third_view_wrist_w_state \
    --action_steps 12 \
    --mode got --score_fn forward_dynamics \
    --n_segments 3 --segment_len 4 \
    --k_candidates 3 --beam_width 2 \
    --fd_n_lookahead 2 \
    --num_trials_per_task 4 \
    --output_dir ./results/got_fd
```

### GoT + World Model Score (got_wm)

```bash
CUDA_VISIBLE_DEVICES=0 python eval_solver_libero_got_v2.py \
    --resume_path ~/ckpts/VLA_model_256/libero_spatial \
    --tokenizer_path ~/ckpts/models--Alpha-VLLM--Lumina-mGPT-7B-768 \
    --task_suite_name libero_spatial \
    --his his_2_third_view_wrist_w_state \
    --action_steps 12 \
    --mode got --score_fn world_model \
    --unmask_image_logits \
    --n_segments 3 --segment_len 4 \
    --k_candidates 3 --beam_width 2 \
    --num_trials_per_task 4 \
    --output_dir ./results/got_wm
```

## 실험 결과

LIBERO-Spatial, 10 tasks × 4 trials = 40 episodes per condition

| 조건 | SR | baseline 대비 |
|------|----|--------------|
| baseline | 72.5% | — |
| got_no_score | 52.5% | −20.0%p |
| got_fd | 50.0% | −22.5%p |
| got_wm | **60.0%** | −12.5%p |

Score 함수 품질이 높을수록 성능이 향상됩니다 (got_wm > got_no_score, got_fd).
World Model Score는 실제 로봇 배포 가능한 유일한 방식으로, 실용적 의의가 있습니다.

## 하이퍼파라미터

| 파라미터 | 값 | 근거 |
|---------|-----|------|
| n_segments | 3 | action_steps=12를 3등분 |
| segment_len | 4 | 3 × 4 = 12 = action_steps |
| k_candidates | 3 | GoT 논문 k=5 대비 연산량 절충 |
| beam_width | 2 | 연산량 2배 대비 경로 다양성 확보 |
| temperature | 1.0 / 1.2 / 1.4 | i=0 greedy, 0~1.5 범위 균등 간격 |
| top_k | 제거 | temperature와 역할 중복 |
| top_p | 0.95 | nucleus sampling |
| fd_n_lookahead | 2 | 빠른 score 계산 (env.step 2회) |

## 참고 자료

- Besta et al., "Graph of Thoughts: Solving Elaborate Problems with Large Language Models", AAAI 2024
- RynnVLA-002 / WorldVLA: https://github.com/alibaba-damo-academy/RynnVLA-002
- LIBERO: https://github.com/Lifelong-Robot-Learning/LIBERO

## 라이선스

연구 목적 코드입니다. RynnVLA-002 및 LIBERO의 라이선스를 따릅니다.
