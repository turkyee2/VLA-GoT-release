"""
test_got_pipeline.py
─────────────────────────────────────────────────────────────────────────────
Unit tests for GoT-VLA pipeline components.
Runs WITHOUT a GPU or the actual model — uses mock objects.

Run:
    python -m pytest test_got_pipeline.py -v
    # or
    python test_got_pipeline.py
"""

import sys
import types
import unittest
from typing import List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
# Minimal stubs so we can import got_pipeline without torch/transformers
# ─────────────────────────────────────────────────────────────────────────────

def _make_stub_module(name: str, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod

# torch stub
torch_stub = _make_stub_module("torch")
torch_stub.device = lambda x: x
torch_stub.no_grad = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))
torch_stub.from_numpy = MagicMock(side_effect=lambda x: x)

# transformers stub
_make_stub_module("transformers", GenerationConfig=MagicMock)

# Now import the module under test
from got_vla_v2.got_pipeline import (
    ActionCandidate,
    GoTConfig,
    GoTVLAPipeline,
    compute_collision_score,
    compute_lpips_score,
    got_aggregate,
    got_keep_best,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_action(val: float = 0.0) -> np.ndarray:
    return np.full(7, val, dtype=np.float32)


def _make_image(colour=(128, 64, 200)) -> Image.Image:
    return Image.new("RGB", (64, 64), colour)


def _make_candidate(n_actions: int = 4, score: float = 0.5) -> ActionCandidate:
    c = ActionCandidate(actions=[_make_action(float(i)) for i in range(n_actions)])
    c.score = score
    return c


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestGoTAggregate(unittest.TestCase):
    """GoT Aggregate = temporal concatenation (NOT averaging)."""

    def test_concatenates_all_actions(self):
        seg1 = _make_candidate(n_actions=4, score=0.8)
        seg2 = _make_candidate(n_actions=4, score=0.7)
        seg3 = _make_candidate(n_actions=4, score=0.9)

        traj = got_aggregate([seg1, seg2, seg3])

        self.assertEqual(len(traj), 12)  # 4 + 4 + 4

    def test_action_values_preserved(self):
        """Values must be preserved exactly — no averaging/blending."""
        seg1 = ActionCandidate(actions=[np.array([1.0]*7)])
        seg2 = ActionCandidate(actions=[np.array([2.0]*7)])

        traj = got_aggregate([seg1, seg2])

        np.testing.assert_allclose(traj[0], np.array([1.0]*7))
        np.testing.assert_allclose(traj[1], np.array([2.0]*7))

    def test_no_modality_collapse(self):
        """
        Left-action candidate vs right-action candidate.
        Averaging would give [0]*7. Concatenation preserves both.
        """
        left = ActionCandidate(actions=[np.array([-1.0, 0, 0, 0, 0, 0, 0])])
        right = ActionCandidate(actions=[np.array([+1.0, 0, 0, 0, 0, 0, 0])])

        traj = got_aggregate([left, right])

        self.assertEqual(len(traj), 2)
        self.assertAlmostEqual(traj[0][0], -1.0)
        self.assertAlmostEqual(traj[1][0], +1.0)
        # The mean of [-1, +1] would be 0 — but our result is NOT 0
        mean_x = (traj[0][0] + traj[1][0]) / 2
        # We do not take the mean, so neither action should BE the mean
        self.assertNotAlmostEqual(traj[0][0], mean_x)

    def test_empty_segments_skipped(self):
        seg1 = _make_candidate(n_actions=3, score=0.9)
        seg2 = ActionCandidate(actions=[])  # empty
        traj = got_aggregate([seg1, seg2])
        self.assertEqual(len(traj), 3)

    def test_none_segments_handled(self):
        seg1 = _make_candidate(n_actions=2, score=0.8)
        traj = got_aggregate([seg1, None])
        self.assertEqual(len(traj), 2)


class TestGoTKeepBest(unittest.TestCase):
    """GoT KeepBest(1): highest-scoring candidate wins."""

    def test_returns_highest_score(self):
        c1 = _make_candidate(score=0.3)
        c2 = _make_candidate(score=0.9)
        c3 = _make_candidate(score=0.6)

        best = got_keep_best([c1, c2, c3])
        self.assertIs(best, c2)

    def test_empty_list_returns_none(self):
        self.assertIsNone(got_keep_best([]))

    def test_single_candidate(self):
        c = _make_candidate(score=0.5)
        self.assertIs(got_keep_best([c]), c)

    def test_negative_inf_score(self):
        c1 = ActionCandidate(actions=[_make_action()])
        c1.score = -float("inf")
        c2 = _make_candidate(score=0.1)
        self.assertIs(got_keep_best([c1, c2]), c2)


class TestCollisionScore(unittest.TestCase):
    """S_collision heuristic."""

    def test_returns_float_in_range(self):
        img = _make_image()
        score = compute_collision_score(img)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_none_returns_neutral(self):
        score = compute_collision_score(None)
        self.assertAlmostEqual(score, 0.5)

    def test_uniform_image_low_score(self):
        """Uniform colour → low variance → low score (possibly blocked)."""
        img = Image.new("RGB", (64, 64), (100, 100, 100))
        score = compute_collision_score(img)
        self.assertLessEqual(score, 0.3)

    def test_noisy_image_high_score(self):
        """Random noise → high variance → higher score (clear scene)."""
        arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        score = compute_collision_score(img)
        self.assertGreater(score, 0.3)


class TestGoTConfig(unittest.TestCase):
    """GoTConfig defaults and validation."""

    def test_defaults(self):
        cfg = GoTConfig()
        self.assertEqual(cfg.n_segments, 3)
        self.assertEqual(cfg.k_candidates, 3)
        self.assertAlmostEqual(cfg.w_collision + cfg.w_consistency, 1.0)

    def test_custom_values(self):
        cfg = GoTConfig(n_segments=5, segment_len=6, k_candidates=4)
        self.assertEqual(cfg.n_segments, 5)
        self.assertEqual(cfg.segment_len, 6)


class TestGoTStructuralCorrespondence(unittest.TestCase):
    """
    Verify the structural correspondence between GoT-Sorting and GoT-VLA
    (from the proposal's core argument).
    """

    def test_got_sorting_analogy(self):
        """
        GoT Sorting:  split 12 nums → 3×4 → sort each → merge
        GoT-VLA:      split 12 steps → 3 segments × 4 → plan each → concatenate

        Here we verify the concatenation produces exactly n_segments × segment_len
        actions (same as sorting produces n_elements output).
        """
        n_segments = 3
        segment_len = 4

        # Simulate best candidate from each segment
        seg_results = [
            ActionCandidate(actions=[_make_action(float(s * segment_len + a))
                                     for a in range(segment_len)])
            for s in range(n_segments)
        ]

        traj = got_aggregate(seg_results)

        # Total actions = n_segments × segment_len (GoT sorting: 3×4=12)
        self.assertEqual(len(traj), n_segments * segment_len)

        # Actions are in the right time order (segment 0 first)
        self.assertAlmostEqual(traj[0][0], 0.0)   # seg 0, action 0
        self.assertAlmostEqual(traj[4][0], 4.0)   # seg 1, action 0
        self.assertAlmostEqual(traj[8][0], 8.0)   # seg 2, action 0


# ─────────────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main(verbosity=2)
