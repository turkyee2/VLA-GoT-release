"""
score_heuristic.py
──────────────────
버전 A: 픽셀 분산 기반 휴리스틱 Score.
현재 이미지만 보므로 후보 간 구분 불가. 기준선용.
"""

from __future__ import annotations
from typing import Optional
import numpy as np
from PIL import Image


def heuristic_score(candidates: list, current_img: Image.Image) -> list:
    """모든 후보에 동일한 픽셀 분산 점수 부여."""
    if current_img is None:
        for c in candidates:
            c.score = 0.5
        return candidates

    arr = np.array(current_img.convert("RGB")).astype(np.float32) / 255.0
    H, W = arr.shape[:2]
    h0, h1 = int(H * 0.3), int(H * 0.7)
    w0, w1 = int(W * 0.3), int(W * 0.7)
    variance = float(np.var(arr[h0:h1, w0:w1]))
    score = min(1.0, variance / 0.05)

    for c in candidates:
        c.score = score  # 모두 동일 → 사실상 첫 번째 선택
        c.score_collision = score
        c.score_consistency = 0.5

    return candidates
