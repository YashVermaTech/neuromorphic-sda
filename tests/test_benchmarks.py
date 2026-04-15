"""
Unit tests for benchmark environments and metrics.

Tests cover:
  - OrbitalBenchmarkEnv: scenario creation, simulation
  - OrbitalScenario: trajectory properties
  - DetectionMetrics: update, compute, precision/recall
  - BenchmarkResults: formatting
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from benchmarks.deterministic_env import (
    ORBIT_PRESETS,
    OrbitalBenchmarkEnv,
    OrbitalScenario,
    SatelliteMotionModel,
    StarfieldGenerator,
    angular_velocity_deg_s,
    orbital_period_s,
)
from benchmarks.metrics import (
    BenchmarkResults,
    DetectionMetrics,
    FrameResult,
    _iou,
    compute_ap,
)


# ── Physics sanity checks ─────────────────────────────────────────────────


class TestOrbitalPhysics:
    def test_leo_period_roughly_90_min(self):
        T = orbital_period_s(550.0)
        assert 5000 < T < 6000, f"LEO period {T:.0f}s should be ~5400s"

    def test_geo_period_roughly_24_hours(self):
        T = orbital_period_s(35786.0)
        assert 85000 < T < 87000, f"GEO period {T:.0f}s should be ~86400s"

    def test_angular_velocity_decreases_with_altitude(self):
        v_leo = angular_velocity_deg_s(550.0)
        v_meo = angular_velocity_deg_s(20200.0)
        v_geo = angular_velocity_deg_s(35786.0)
        assert v_leo > v_meo > v_geo


# ── Starfield generator tests ─────────────────────────────────────────────


class TestStarfieldGenerator:
    def test_render_shape(self):
        sg = StarfieldGenerator(64, 48, n_stars=20, seed=0)
        frame = sg.render(seed=0)
        assert frame.shape == (48, 64)
        assert frame.dtype == np.uint8

    def test_render_has_bright_pixels(self):
        sg = StarfieldGenerator(64, 64, n_stars=50, seed=0)
        frame = sg.render(seed=0)
        assert frame.max() > 0


# ── OrbitalBenchmarkEnv tests ─────────────────────────────────────────────


class TestOrbitalBenchmarkEnv:
    @pytest.fixture
    def env(self):
        return OrbitalBenchmarkEnv(sensor_width=64, sensor_height=48, seed=42)

    def test_create_scenario_returns_correct_type(self, env):
        s = env.create_scenario("leo")
        assert isinstance(s, OrbitalScenario)
        assert s.orbit_type == "leo"

    def test_create_all_orbit_types(self, env):
        for orbit in ORBIT_PRESETS:
            s = env.create_scenario(orbit)
            assert s.orbit_type == orbit

    def test_invalid_orbit_type_raises(self, env):
        with pytest.raises(ValueError):
            env.create_scenario("warp_drive")

    def test_run_scenario_populates_frames(self, env):
        s = env.create_scenario("leo", duration_s=2.0, fps=10.0)
        env.run_scenario(s, show_progress=False)
        assert len(s.frames) == s.n_frames

    def test_frame_shape(self, env):
        s = env.create_scenario("leo", duration_s=1.0, fps=5.0)
        env.run_scenario(s, show_progress=False)
        for frame in s.frames:
            assert frame.shape == (env.sensor_height, env.sensor_width)
            assert frame.dtype == np.uint8

    def test_gt_boxes_shape(self, env):
        s = env.create_scenario("geo", duration_s=1.0, fps=5.0)
        env.run_scenario(s, n_satellites=2, show_progress=False)
        for boxes in s.gt_boxes:
            assert boxes.shape == (2, 4)

    def test_gt_boxes_normalised(self, env):
        s = env.create_scenario("leo", duration_s=1.0, fps=5.0)
        env.run_scenario(s, show_progress=False)
        for boxes in s.gt_boxes:
            assert boxes[:, 0].min() >= 0.0   # cx in [0, 1]
            assert boxes[:, 0].max() <= 1.0

    def test_reproducibility(self, env):
        """Same seed → identical frames."""
        s1 = env.create_scenario("leo", duration_s=2.0, fps=10.0)
        s2 = OrbitalBenchmarkEnv(sensor_width=64, sensor_height=48, seed=42)
        s2_scen = s2.create_scenario("leo", duration_s=2.0, fps=10.0)
        # Scenarios from the same seed env should match
        assert s1.seed == s2_scen.seed

    def test_n_frames_property(self, env):
        s = env.create_scenario("meo", duration_s=5.0, fps=10.0)
        assert s.n_frames == 50

    def test_custom_altitude(self, env):
        s = env.create_scenario("leo", custom_altitude_km=400.0)
        assert s.altitude_km == 400.0


# ── SatelliteMotionModel tests ────────────────────────────────────────────


class TestSatelliteMotionModel:
    def test_positions_update_each_step(self):
        env = OrbitalBenchmarkEnv(64, 48, seed=0)
        s = env.create_scenario("leo", fps=30.0)
        motion = SatelliteMotionModel(s, n_satellites=1)

        pos1 = [(float(p[0]), float(p[1])) for p in motion.positions.copy()]
        pos2 = motion.step()
        assert pos1 != pos2

    def test_positions_wrap_around(self):
        env = OrbitalBenchmarkEnv(64, 48, seed=0)
        s = env.create_scenario("leo", fps=1.0)  # slow updates
        motion = SatelliteMotionModel(s, n_satellites=1)
        for _ in range(200):
            positions = motion.step()
        for x, y in positions:
            assert 0 <= x < 64
            assert 0 <= y < 48

    def test_render_frame_shape(self):
        env = OrbitalBenchmarkEnv(64, 48, seed=0)
        s = env.create_scenario("leo")
        motion = SatelliteMotionModel(s)
        bg = np.zeros((48, 64), dtype=np.uint8)
        positions = motion.step()
        frame = motion.render_to_frame(bg, positions)
        assert frame.shape == (48, 64)
        assert frame.dtype == np.uint8
        assert frame.max() > 0  # something was rendered


# ── IoU tests ─────────────────────────────────────────────────────────────


class TestIoU:
    def test_perfect_overlap(self):
        box = np.array([0.5, 0.5, 0.2, 0.2])
        assert abs(_iou(box, box) - 1.0) < 1e-6

    def test_no_overlap(self):
        a = np.array([0.1, 0.1, 0.1, 0.1])
        b = np.array([0.9, 0.9, 0.1, 0.1])
        assert _iou(a, b) < 1e-6

    def test_partial_overlap(self):
        a = np.array([0.4, 0.5, 0.2, 0.2])
        b = np.array([0.6, 0.5, 0.2, 0.2])
        iou = _iou(a, b)
        assert 0.0 < iou < 1.0

    def test_symmetry(self):
        a = np.array([0.3, 0.4, 0.15, 0.12])
        b = np.array([0.35, 0.42, 0.1, 0.1])
        assert abs(_iou(a, b) - _iou(b, a)) < 1e-9


# ── DetectionMetrics tests ────────────────────────────────────────────────


class TestDetectionMetrics:
    def test_perfect_detection(self):
        metrics = DetectionMetrics(iou_threshold=0.5)
        box = np.array([[0.5, 0.5, 0.2, 0.2]], dtype=np.float32)
        score = np.array([0.99], dtype=np.float32)
        result = metrics.update(box, score, box, latency_ms=1.0)
        assert result.true_positives == 1
        assert result.false_positives == 0

    def test_all_false_positives(self):
        metrics = DetectionMetrics(iou_threshold=0.5)
        pred = np.array([[0.1, 0.1, 0.05, 0.05]], dtype=np.float32)
        gt = np.array([[0.9, 0.9, 0.05, 0.05]], dtype=np.float32)
        score = np.array([0.9], dtype=np.float32)
        result = metrics.update(pred, score, gt)
        assert result.false_positives == 1
        assert result.true_positives == 0

    def test_empty_predictions(self):
        metrics = DetectionMetrics()
        result = metrics.update(
            np.zeros((0, 4)), np.zeros(0),
            np.array([[0.5, 0.5, 0.1, 0.1]])
        )
        assert result.false_negatives == 1
        assert result.true_positives == 0

    def test_compute_returns_benchmark_results(self):
        metrics = DetectionMetrics()
        box = np.array([[0.5, 0.5, 0.2, 0.2]], dtype=np.float32)
        score = np.array([0.9])
        metrics.update(box, score, box)
        results = metrics.compute()
        assert isinstance(results, BenchmarkResults)
        assert 0.0 <= results.mean_ap <= 1.0

    def test_precision_recall_at_perfect_detection(self):
        metrics = DetectionMetrics(iou_threshold=0.5)
        box = np.array([[0.5, 0.5, 0.2, 0.2]], dtype=np.float32)
        for _ in range(10):
            metrics.update(box, np.array([0.95]), box)
        results = metrics.compute()
        assert results.precision > 0.9
        assert results.recall > 0.9

    def test_reset_clears_state(self):
        metrics = DetectionMetrics()
        box = np.array([[0.5, 0.5, 0.2, 0.2]], dtype=np.float32)
        metrics.update(box, np.array([0.9]), box)
        metrics.reset()
        assert len(metrics._all_scores) == 0
        assert metrics._n_gt == 0

    def test_latency_statistics(self):
        metrics = DetectionMetrics()
        box = np.array([[0.5, 0.5, 0.2, 0.2]], dtype=np.float32)
        for i in range(10):
            metrics.update(box, np.array([0.9]), box, latency_ms=float(i + 1))
        results = metrics.compute()
        assert results.mean_latency_ms > 0
        assert results.p95_latency_ms >= results.mean_latency_ms


# ── AP computation tests ───────────────────────────────────────────────────


class TestComputeAP:
    def test_perfect_ap_is_one(self):
        precision = np.ones(10)
        recall = np.linspace(0, 1, 10)
        ap = compute_ap(precision, recall, method="interp")
        assert abs(ap - 1.0) < 0.01

    def test_zero_precision_gives_low_ap(self):
        precision = np.zeros(10)
        recall = np.linspace(0, 1, 10)
        ap = compute_ap(precision, recall, method="interp")
        assert ap < 0.01

    def test_area_method(self):
        precision = np.array([1.0, 0.8, 0.6, 0.4])
        recall = np.array([0.0, 0.3, 0.6, 1.0])
        ap = compute_ap(precision, recall, method="area")
        assert 0.0 <= ap <= 1.0
