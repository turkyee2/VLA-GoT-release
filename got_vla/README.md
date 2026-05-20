# GoT-VLA: Graph of Thoughts for Robot Manipulation

WorldVLA 추론 파이프라인에 Graph of Thoughts(GoT)의 **시간축 분할-병합** 전략을 적용한 구현입니다.

---

## 핵심 아이디어 요약

| GoT Sorting (64개 숫자) | GoT-VLA (T 스텝 궤적) |
|---|---|
| 64개 → 4×16으로 분할 | T 스텝 → n개 구간으로 분할 |
| 각 16개를 k=5회 정렬 | 각 구간에서 k=3개 행동 후보 생성 |
| 정렬 오류 수로 점수 | World Model 미래 예측으로 점수 |
| KeepBest(1) | KeepBest(1) |
| merge sort로 병합 | **시간순 연결** (concatenate) |

**핵심**: Aggregate = 값의 평균이 아닌 시간축 연결 → 모달리티 붕괴 없음

---

## 디렉터리 구조

```
RynnVLA-002-main/              ← 기존 코드 (수정 없음)
│
├── rynnvla-002/
│   ├── eval_solver_libero_discrete_w_state.py   (원본 — 건드리지 않음)
│   ├── libero_util/Chameleon_utils.py           (원본 — 건드리지 않음)
│   └── ...
│
got_vla/                       ← 새로 추가하는 폴더 (여기만 수정)
├── got_vla/
│   ├── __init__.py
│   ├── got_pipeline.py          ← GoT 핵심 파이프라인
│   └── chameleon_got_utils.py   ← 기존 Chameleon_utils 어댑터
├── eval_solver_libero_got.py    ← 새 실행 스크립트
├── run_ablation.py              ← Ablation 자동화
└── test_got_pipeline.py         ← 단위 테스트
```

### 기존 파일 수정 여부

| 파일 | 상태 |
|---|---|
| `eval_solver_libero_discrete_w_state.py` | **수정 없음** |
| `libero_util/Chameleon_utils.py` | **수정 없음** |
| `model/modeling_xllmx_*.py` | **수정 없음** |
| `got_vla/got_pipeline.py` | **신규 추가** |
| `got_vla/chameleon_got_utils.py` | **신규 추가** |
| `eval_solver_libero_got.py` | **신규 추가** |

---

## 설치

```bash
# 기존 RynnVLA-002 환경이 이미 세팅되어 있다고 가정
cd RynnVLA-002-main

# got_vla 폴더를 rynnvla-002/ 안에 복사
cp -r /path/to/got_vla/got_vla  rynnvla-002/got_vla
cp /path/to/got_vla/eval_solver_libero_got.py  rynnvla-002/eval_solver_libero_got.py
cp /path/to/got_vla/run_ablation.py             rynnvla-002/run_ablation.py
cp /path/to/got_vla/test_got_pipeline.py        rynnvla-002/test_got_pipeline.py

# lpips (Score 함수용, requirements.txt에 이미 있음)
pip install lpips --break-system-packages
```

---

## 실행 방법

### 1) 단위 테스트 (GPU 없이 확인)

```bash
cd rynnvla-002
python test_got_pipeline.py
# → 16 tests, OK
```

### 2) GoT 평가 실행 (제안 방법)

```bash
cd rynnvla-002
python eval_solver_libero_got.py \
    --resume_path /path/to/checkpoint \
    --tokenizer_path /path/to/tokenizer \
    --task_suite_name libero_spatial \
    --mode got \
    --n_segments 3 \
    --segment_len 4 \
    --k_candidates 3 \
    --output_dir ./results/got_spatial \
    --device 0
```

### 3) Baseline (원본과 동일한 단일 추론)

```bash
python eval_solver_libero_got.py \
    --resume_path /path/to/checkpoint \
    --tokenizer_path /path/to/tokenizer \
    --task_suite_name libero_spatial \
    --mode baseline \
    --output_dir ./results/baseline_spatial \
    --device 0
```

### 4) Best-of-N (Ablation, CoT-SC 상당)

```bash
python eval_solver_libero_got.py \
    --mode bon --k_candidates 3 \
    --resume_path ... --tokenizer_path ... \
    --task_suite_name libero_spatial \
    --output_dir ./results/bon_spatial --device 0
```

### 5) 4-Way Ablation 자동 실행

```bash
python run_ablation.py \
    --resume_path /path/to/checkpoint \
    --tokenizer_path /path/to/tokenizer \
    --task_suite_name libero_spatial \
    --output_base ./results/ablation \
    --device 0 \
    --num_trials 10       # 빠른 테스트: 10 / 최종 결과: 50
```

출력 예시:
```
=================================================================
GoT-VLA Ablation Results
=================================================================
Condition            SR (%)       Episodes     Successes
-----------------------------------------------------------------
baseline             72.0         50           36
bon                  76.0         50           38
got_no_wm            78.0         50           39
got                  82.0         50           41
=================================================================

Improvement over baseline:
  bon               : +5.6%
  got_no_wm         : +8.3%
  got               : +13.9%
```

---

## 주요 파라미터

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `--mode` | `got` | `got` / `bon` / `baseline` |
| `--n_segments` | `3` | 궤적 분할 수 (n) |
| `--segment_len` | `4` | 구간당 액션 수 |
| `--k_candidates` | `3` | 구간당 후보 수 (k) |
| `--w_collision` | `0.6` | 충돌 안전성 가중치 |
| `--w_consistency` | `0.4` | LPIPS 일관성 가중치 |
| `--no_world_model_scoring` | off | World Model 스코어 비활성화 |

### RQ3 실험용 파라미터 조합 예시

```bash
# n 변화 실험 (segment_len × n = 총 12 스텝 고정)
for n in 2 3 4 6; do
    python eval_solver_libero_got.py \
        --mode got --n_segments $n --segment_len $((12/n)) \
        --output_dir ./results/rq3_n${n} ...
done

# k 변화 실험
for k in 1 2 3 5; do
    python eval_solver_libero_got.py \
        --mode got --k_candidates $k \
        --output_dir ./results/rq3_k${k} ...
done
```

---

## 코드 흐름 상세

### `got_pipeline.py` 핵심 함수

```
GoTVLAPipeline.plan()
│
├── for seg in n_segments:
│   ├── got_generate(k=3)          # Generate(k=3): 후보 생성
│   │   └── chameleon_got_utils.get_action_for_got()
│   │       └── model.generate_dis_ma()   ← 기존 모델 API 그대로 사용
│   │
│   ├── got_score()                # Score: World Model으로 평가
│   │   ├── generate_world_model_prediction()
│   │   │   └── model.generate_img()      ← 기존 모델 API 그대로 사용
│   │   ├── compute_collision_score()     # S_collision
│   │   └── compute_lpips_score()        # S_consistency
│   │
│   ├── got_keep_best()            # KeepBest(1)
│   │
│   └── World-Model Bridge         # 다음 구간의 obs = 예측된 미래 프레임
│
└── got_aggregate()                # Aggregate: 시간축 연결
```

### Scoring 함수

```
Score(a) = w1 × S_collision(a) + w2 × S_consistency(a)
         = 0.6 × S_collision  + 0.4 × S_consistency

S_collision:   predicted_img 중앙 영역의 픽셀 분산 (높을수록 안전)
S_consistency: LPIPS(current_img, predicted_img) 기반 (낮을수록 일관적 → 점수 높음)
```

---

## 자율주행으로의 확장

현재 구현은 LIBERO 로봇 제어를 기준으로 하지만, 구조적으로 자율주행으로 확장 가능합니다.

| 로봇 제어 (현재) | 자율주행 (확장) |
|---|---|
| LIBERO benchmark | nuScenes dataset |
| `get_libero_image()` | 6-camera view processing |
| 7-dim end-effector action | steering + acceleration |
| `S_collision`: 픽셀 분산 | `S_collision`: Bounding Box IoU |
| `S_consistency`: LPIPS | `S_consistency`: FID / LPIPS |

확장 시 `got_pipeline.py`의 `compute_collision_score()` 함수만 교체하면 됩니다.

---

## References

- WorldVLA: [arXiv:2506.21539](https://arxiv.org/abs/2506.21539)
- Graph of Thoughts: [arXiv:2308.09687](https://arxiv.org/abs/2308.09687)
- LIBERO: [Liu et al., NeurIPS 2023](https://libero-project.github.io)
