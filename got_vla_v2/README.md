# GoT-VLA v2: Forward Dynamics Score

## 버전별 파일 구조

```
got_vla/
├── got_vla/
│   ├── __init__.py
│   ├── got_pipeline.py           ← 공통 파이프라인
│   ├── chameleon_got_utils.py    ← 액션 생성 유틸
│   └── scoring/
│       ├── __init__.py
│       ├── score_heuristic.py        ← 버전 A: 픽셀 분산
│       ├── score_forward_dynamics.py ← 버전 B: 시뮬레이터 미리 실행 (권장)
│       └── score_world_model.py      ← 버전 C: World Model (서버용)
├── eval_solver_libero_got_v2.py  ← 메인 실행 스크립트
└── README.md
```

## Score 함수 비교

| 버전 | score_fn | 원리 | 후보 구분 | 속도 | VRAM |
|---|---|---|---|---|---|
| A | heuristic | 픽셀 분산 | ❌ 불가 | 빠름 | 8GB |
| B | forward_dynamics | 시뮬레이터 미리 실행 | ✅ 가능 | 빠름 | 8GB |
| C | world_model | World Model 이미지 생성 | ✅✅ 최선 | 느림 | 24GB+ |

## 실행 방법

### baseline (비교 기준)
```bash
mkdir -p ./results/baseline && \
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \
PYTHONPATH=/workspace/LIBERO:/workspace/rynnvla-002 \
python eval_solver_libero_got_v2.py \
    --resume_path ./ckpts/VLA_model_256/libero_spatial \
    --tokenizer_path ./ckpts/models--Alpha-VLLM--Lumina-mGPT-7B-768/snapshots/9624463a82ea5ce814af9b561dcd08a31082c3af \
    --task_suite_name libero_spatial \
    --his his_2_third_view_wrist_w_state \
    --action_steps 10 \
    --mode baseline \
    --num_trials_per_task 1 \
    --output_dir ./results/baseline \
    --device 0
```

### GoT + Forward Dynamics Score (권장)
```bash
mkdir -p ./results/got_fd && \
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \
PYTHONPATH=/workspace/LIBERO:/workspace/rynnvla-002 \
python eval_solver_libero_got_v2.py \
    --resume_path ./ckpts/VLA_model_256/libero_spatial \
    --tokenizer_path ./ckpts/models--Alpha-VLLM--Lumina-mGPT-7B-768/snapshots/9624463a82ea5ce814af9b561dcd08a31082c3af \
    --task_suite_name libero_spatial \
    --his his_2_third_view_wrist_w_state \
    --action_steps 10 \
    --mode got \
    --score_fn forward_dynamics \
    --n_segments 3 \
    --segment_len 4 \
    --k_candidates 3 \
    --fd_n_lookahead 2 \
    --num_trials_per_task 1 \
    --output_dir ./results/got_fd \
    --device 0
```

### BoN + Forward Dynamics Score
```bash
mkdir -p ./results/bon_fd && \
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \
PYTHONPATH=/workspace/LIBERO:/workspace/rynnvla-002 \
python eval_solver_libero_got_v2.py \
    --resume_path ./ckpts/VLA_model_256/libero_spatial \
    --tokenizer_path ./ckpts/models--Alpha-VLLM--Lumina-mGPT-7B-768/snapshots/9624463a82ea5ce814af9b561dcd08a31082c3af \
    --task_suite_name libero_spatial \
    --his his_2_third_view_wrist_w_state \
    --action_steps 10 \
    --mode bon \
    --score_fn forward_dynamics \
    --k_candidates 3 \
    --fd_n_lookahead 2 \
    --num_trials_per_task 1 \
    --output_dir ./results/bon_fd \
    --device 0
```

## Forward Dynamics Score 동작 원리

```
현재 상태 저장 (세이브포인트)
    ↓
후보 A: 2스텝 실행 → reward + 목표달성도 → Score_A
환경 복원
    ↓
후보 B: 2스텝 실행 → reward + 목표달성도 → Score_B
환경 복원
    ↓
후보 C: 2스텝 실행 → reward + 목표달성도 → Score_C
환경 복원
    ↓
KeepBest: Score가 가장 높은 후보 선택
```

추가 계산 시간: 약 2초 (env.step 6회)
→ 에피소드 전체 추가 시간 거의 없음

## 서버 환경 확장 (World Model Score)

24GB+ VRAM 서버 확보 시:
```bash
--score_fn world_model \
--world_model_path ./ckpts/Action_World_model_512/libero_spatial
```
코드 변경 없이 플래그만 바꾸면 됨.
