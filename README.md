# GoT-VLA: Graph-of-Thought for Vision-Language-Action Models

WorldVLA 기반 로봇 조작 모델에 Graph-of-Thought 추론 구조를 적용하여 성능을 향상시키는 프레임워크입니다. LIBERO 벤치마크에서 검증되었으며, 모델 재학습 없이 추론 파이프라인만 교체합니다.

## 핵심 아이디어

기존 VLA 모델은 카메라 이미지와 언어 지시를 받아 전체 행동 궤적을 한 번에 생성합니다. 이 방식은 앞부분 오류가 뒷부분으로 누적되고, 대안을 탐색하는 메커니즘이 없습니다.

GoT-VLA는 GoT(Besta et al., 2023) 논문의 분할-병합 전략을 시간축에 적용합니다. 12스텝 궤적을 3개 구간으로 나누고, 각 구간에서 여러 후보를 생성한 뒤 누적 score 기준으로 최선 경로를 선택합니다.

```
기존 VLA:    이미지 + 지시 → [a1, a2, ..., a12] 한번에 생성
GoT-VLA:     구간1 [a1..a4]  + 구간2 [a5..a8]  + 구간3 [a9..a12]
              ↑ 각 구간에서 k=3개 후보 생성, 누적 score로 선택
```

## 구조

### GoT 핵심 연산

```
Generate  : 각 경로에서 k=3 후보 생성 (do_sample, temperature 1.0/1.2/1.4)
Score     : segment_score 계산 (forward_dynamics / world_model / heuristic)
Aggregate : path_score = parent.path_score + segment_score
Prune     : 상위 beam_width=2 경로만 유지 (가지치기)
Merge     : 최선 경로 역추적 → 시간순 연결 → 한번에 실행 (병합)
```

### 두 단계 파이프라인

**Phase 1 — Planning** (실행 없이 최선 경로 탐색)

각 구간마다 다음을 반복합니다:
1. Generate: 활성 beam 경로에서 k=3 후보 생성
2. Score: 각 후보의 segment_score 계산
3. Aggregate: 부모 경로의 path_score와 합산
4. Prune: 상위 beam_width 경로만 유지
5. Context update: best 후보를 n_lookahead 스텝 실행해 다음 구간 컨텍스트 갱신 후 환경 복원

**Phase 2 — Execution**

최선 경로의 전체 액션(15개)을 시간순으로 한번에 실행합니다.

### Score 함수

| 종류 | 방식 | VRAM | 속도 |
|------|------|------|------|
| heuristic | 픽셀 분산 | 8GB | 빠름 |
| forward_dynamics | 시뮬레이터 미리 실행 (권장) | 8GB | 빠름 |
| world_model | World Model 이미지 생성 | 24GB+ | 느림 |

## 디렉토리 구조

```
rynnvla-002/
├── got_vla_v2/
│   ├── __init__.py
│   ├── got_pipeline.py             # v1: ToT 구조 (구간마다 즉시 실행)
│   ├── got_pipeline_v2.py          # v2: GoT 구조 (beam search + 일괄 실행)
│   ├── chameleon_got_utils.py      # 액션 생성 유틸 (temperature 차등 적용)
│   └── scoring/
│       ├── score_heuristic.py
│       ├── score_forward_dynamics.py
│       └── score_world_model.py
├── eval_solver_libero_got_v2.py    # 메인 실행 스크립트
├── test_got_v2_structure.py        # 구조 검증 단위 테스트
└── README.md
```

## 실행 방법

### 환경 설정

```bash
# Conda 환경
conda create -n got_vla python=3.10
conda activate got_vla

# 의존성
pip install -r requirements.txt

# LIBERO 설치
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO && pip install -e .
```

### 모델 체크포인트

RynnVLA-002 (VLA_model_256/libero_spatial)을 `./ckpts/`에 다운로드합니다.

### 실험 조건별 실행

baseline (기준선):
```bash
python eval_solver_libero_got_v2.py \
    --resume_path ./ckpts/VLA_model_256/libero_spatial \
    --tokenizer_path ./ckpts/models--Alpha-VLLM--Lumina-mGPT-7B-768/snapshots/... \
    --task_suite_name libero_spatial \
    --his his_2_third_view_wrist_w_state \
    --action_steps 12 \
    --mode baseline \
    --num_trials_per_task 4 \
    --output_dir ./results/baseline
```

bon (Best-of-N 비교 기준):
```bash
python eval_solver_libero_got_v2.py [공통 인자] \
    --mode bon \
    --score_fn forward_dynamics \
    --k_candidates 3 \
    --output_dir ./results/bon
```

GoT v2 (제안 방법):
```bash
python eval_solver_libero_got_v2.py [공통 인자] \
    --mode got --got_version v2 \
    --score_fn forward_dynamics \
    --n_segments 3 --segment_len 4 \
    --k_candidates 3 --beam_width 2 \
    --fd_n_lookahead 2 \
    --output_dir ./results/got_fd
```

GoT v2 + World Model Score (24GB+ VRAM):
```bash
python eval_solver_libero_got_v2.py [공통 인자] \
    --mode got --got_version v2 \
    --score_fn world_model \
    --unmask_image_logits \
    --output_dir ./results/got_wm
```

### 8GB VRAM 환경 (4bit 양자화)

```bash
python eval_solver_libero_got_v2.py [인자들] \
    --load_in_4bit
```

## 실험 설계

LIBERO-Spatial 10 tasks × 4 trials = 40 episodes per condition

| 조건 | 분할 | Score | 답하는 질문 |
|------|------|-------|------------|
| baseline | ❌ | 없음 | 기준선 |
| bon | ❌ | FD | k개 생성 + 선택의 효과 |
| got_no_score | ✅ | heuristic | GoT 구조 자체의 효과 |
| got_fd | ✅ | FD | GoT + 실제 평가의 시너지 |
| got_wm | ✅ | WM | WM Score의 효과 |

### 주요 비교

- baseline → bon: 다중 후보 생성의 효과
- baseline → got_no_score: GoT 분할-병합 구조의 효과
- bon → got_fd: 분할이 전체 선택보다 나은가
- got_no_score → got_fd: Score 함수의 중요성
- got_fd → got_wm: WM Score vs FD Score

## 하이퍼파라미터

| 파라미터 | 값 | 근거 |
|---------|-----|------|
| n_segments | 3 | action_steps=12를 3등분 |
| segment_len | 4 | n_segments × segment_len = action_steps |
| k_candidates | 3 | GoT 논문 k=5 대비 연산량 절충 |
| beam_width | 2 | 연산량 2배 대비 경로 다양성 확보 |
| temperature | [1.0, 1.2, 1.4] | 0~1.5 범위에서 균등 간격 (i=0 greedy) |
| top_p | 0.95 | temperature와 중복인 top_k 대신 사용 |
| fd_n_lookahead | 2 | 빠른 score 계산 |

## 연구적 기여

1. **GoT 추론을 VLA에 최초 적용**: LLM 텍스트 추론에 쓰이던 GoT를 로봇 행동 생성으로 확장
2. **시간축 분할-병합 전략**: Aggregate를 시간순 연결로 정의 (모달리티 붕괴 방지)
3. **Forward Dynamics를 Score 함수로 활용**: World Model 없이도 의미있는 후보 선택 가능
4. **모델 재학습 불필요**: 추론 파이프라인만 교체

## 참고 자료

- Besta et al., "Graph of Thoughts: Solving Elaborate Problems with Large Language Models", AAAI 2024
- RynnVLA-002 / WorldVLA: Vision-Language-Action 모델 기반
- LIBERO 벤치마크: https://github.com/Lifelong-Robot-Learning/LIBERO

## 라이선스

연구 목적의 코드입니다. WorldVLA / RynnVLA-002 및 LIBERO의 라이선스를 따릅니다.
