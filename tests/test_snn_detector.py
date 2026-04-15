"""
Unit tests for the SNN satellite detector.

Tests cover:
  - SNNConfig default values
  - SNNConvBlock forward pass
  - SNNBackbone output shapes
  - SNNSatelliteDetector forward pass and output keys
  - Loss computation
  - Latency benchmarking
  - Centroiding algorithms
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.snn_detector import (
    SNNBackbone,
    SNNConfig,
    SNNConvBlock,
    SNNSatelliteDetector,
)
from models.centroiding import (
    Centroider,
    CentroidResult,
    detect_star_candidates,
)


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def tiny_config():
    """Minimal config for fast tests."""
    return SNNConfig(
        sensor_height=32,
        sensor_width=32,
        time_steps=4,
        hidden_channels=[4, 8, 16, 32],
        num_classes=2,
        lif_beta=0.9,
    )


@pytest.fixture
def tiny_model(tiny_config):
    return SNNSatelliteDetector(tiny_config)


@pytest.fixture
def tiny_input(tiny_config):
    # (B=2, T=4, C=2, H=32, W=32)
    return torch.zeros(2, tiny_config.time_steps, 2,
                       tiny_config.sensor_height, tiny_config.sensor_width)


# ── SNNConvBlock tests ────────────────────────────────────────────────────


class TestSNNConvBlock:
    def test_output_shape(self):
        block = SNNConvBlock(2, 8)
        x = torch.zeros(1, 2, 16, 16)
        spk, mem = block(x)
        assert spk.shape == (1, 8, 16, 16)
        assert mem.shape == (1, 8, 16, 16)

    def test_spike_binary(self):
        block = SNNConvBlock(2, 8)
        x = torch.ones(1, 2, 8, 8) * 100  # large input to trigger spikes
        spk, _ = block(x)
        unique = spk.unique()
        for v in unique:
            assert v.item() in {0.0, 1.0}

    def test_membrane_initialised_to_zero_when_none(self):
        block = SNNConvBlock(2, 4)
        x = torch.zeros(1, 2, 8, 8)
        spk, mem = block(x, mem=None)
        assert spk.shape[1] == 4


# ── SNNBackbone tests ─────────────────────────────────────────────────────


class TestSNNBackbone:
    def test_output_shapes(self, tiny_config, tiny_input):
        backbone = SNNBackbone(tiny_config)
        with torch.no_grad():
            p3, p4, sr = backbone(tiny_input)

        B = tiny_input.shape[0]
        ch = tiny_config.hidden_channels
        H, W = tiny_config.sensor_height, tiny_config.sensor_width

        assert p3.shape == (B, ch[2], H // 4, W // 4)
        assert p4.shape == (B, ch[3], H // 8, W // 8)

    def test_spike_rate_in_range(self, tiny_config, tiny_input):
        backbone = SNNBackbone(tiny_config)
        with torch.no_grad():
            _, _, sr = backbone(tiny_input)
        assert 0.0 <= sr <= 1.0

    def test_zero_input_low_spike_rate(self, tiny_config):
        backbone = SNNBackbone(tiny_config)
        x = torch.zeros(1, tiny_config.time_steps, 2,
                        tiny_config.sensor_height, tiny_config.sensor_width)
        with torch.no_grad():
            _, _, sr = backbone(x)
        assert sr < 0.5  # zero input should produce few spikes


# ── SNNSatelliteDetector tests ────────────────────────────────────────────


class TestSNNSatelliteDetector:
    def test_forward_output_keys(self, tiny_model, tiny_input):
        with torch.no_grad():
            out = tiny_model(tiny_input)
        expected_keys = {"cls_p3", "box_p3", "cls_p4", "box_p4", "spike_rate"}
        assert set(out.keys()) == expected_keys

    def test_cls_output_shape(self, tiny_model, tiny_config, tiny_input):
        with torch.no_grad():
            out = tiny_model(tiny_input)
        B = tiny_input.shape[0]
        cls_p3 = out["cls_p3"]
        # (B, num_anchors, num_classes, H/4, W/4)
        assert cls_p3.shape[0] == B
        assert cls_p3.shape[2] == tiny_config.num_classes

    def test_box_output_shape(self, tiny_model, tiny_input):
        with torch.no_grad():
            out = tiny_model(tiny_input)
        B = tiny_input.shape[0]
        box_p3 = out["box_p3"]
        assert box_p3.shape[0] == B
        assert box_p3.shape[2] == 4  # (dx, dy, dw, dh)

    def test_spike_rate_tensor(self, tiny_model, tiny_input):
        with torch.no_grad():
            out = tiny_model(tiny_input)
        assert isinstance(out["spike_rate"], torch.Tensor)
        assert 0.0 <= out["spike_rate"].item() <= 1.0

    def test_compute_loss_returns_dict(self, tiny_model, tiny_input):
        out = tiny_model(tiny_input)
        targets = [{"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.long)}
                   for _ in range(tiny_input.shape[0])]
        loss = tiny_model.compute_loss(out, targets)
        assert "total" in loss
        assert "cls" in loss
        assert "loc" in loss
        assert "spike" in loss

    def test_loss_total_is_finite(self, tiny_model, tiny_input):
        out = tiny_model(tiny_input)
        targets = []
        loss = tiny_model.compute_loss(out, targets)
        assert torch.isfinite(loss["total"])

    def test_predict_returns_list(self, tiny_model, tiny_input):
        with torch.no_grad():
            preds = tiny_model.predict(tiny_input)
        assert isinstance(preds, list)
        assert len(preds) == tiny_input.shape[0]

    def test_predict_result_keys(self, tiny_model, tiny_input):
        with torch.no_grad():
            preds = tiny_model.predict(tiny_input, confidence_threshold=0.0)
        for pred in preds:
            assert "scores" in pred
            assert "labels" in pred
            assert "boxes" in pred

    def test_benchmark_latency_returns_dict(self, tiny_config):
        model = SNNSatelliteDetector(tiny_config)
        result = model.benchmark_latency(
            batch_size=1, time_steps=2, n_runs=5, device="cpu", warmup=2
        )
        expected = {"mean_ms", "std_ms", "min_ms", "max_ms",
                    "spike_rate", "synaptic_ops_estimate", "n_parameters"}
        assert expected.issubset(set(result.keys()))
        assert result["mean_ms"] > 0

    def test_parameter_count_positive(self, tiny_model):
        n = sum(p.numel() for p in tiny_model.parameters())
        assert n > 0


# ── Centroiding tests ─────────────────────────────────────────────────────


class TestCentroider:
    @pytest.fixture
    def star_image(self):
        """Small image with a bright star at (25, 20)."""
        img = np.zeros((50, 50), dtype=np.float32)
        # Gaussian PSF
        for dy in range(-5, 6):
            for dx in range(-5, 6):
                y, x = 20 + dy, 25 + dx
                if 0 <= y < 50 and 0 <= x < 50:
                    img[y, x] += 100.0 * np.exp(-(dx ** 2 + dy ** 2) / 4.0)
        return img

    def test_weighted_centroid_near_truth(self, star_image):
        c = Centroider(method="weighted")
        result = c.centroid(star_image, 25.0, 20.0)
        assert abs(result.x - 25.0) < 1.0
        assert abs(result.y - 20.0) < 1.0

    def test_iterative_centroid_converges(self, star_image):
        c = Centroider(method="iterative", tolerance=0.01)
        result = c.centroid(star_image, 24.0, 19.0)  # offset start
        assert result.converged
        assert abs(result.x - 25.0) < 1.0

    def test_threshold_centroid_shape(self, star_image):
        c = Centroider(method="threshold")
        result = c.centroid(star_image, 25.0, 20.0)
        assert isinstance(result, CentroidResult)

    def test_flux_positive(self, star_image):
        c = Centroider(method="weighted")
        result = c.centroid(star_image, 25.0, 20.0)
        assert result.flux > 0

    def test_batch_centroid_length(self, star_image):
        c = Centroider()
        positions = [(25.0, 20.0), (10.0, 10.0)]
        results = c.centroid_batch(star_image, positions)
        assert len(results) == 2

    def test_empty_image_no_crash(self):
        c = Centroider()
        img = np.zeros((30, 30), dtype=np.float32)
        result = c.centroid(img, 15.0, 15.0)
        assert isinstance(result, CentroidResult)

    def test_detect_star_candidates_finds_star(self, star_image):
        candidates = detect_star_candidates(star_image, min_sigma=2.0)
        assert len(candidates) >= 1
        xs = [p[0] for p in candidates]
        ys = [p[1] for p in candidates]
        # The brightest candidate should be near (25, 20)
        nearest_idx = np.argmin([(x - 25) ** 2 + (y - 20) ** 2 for x, y in zip(xs, ys)])
        assert abs(xs[nearest_idx] - 25.0) < 5.0

    def test_detect_no_candidates_on_blank(self):
        img = np.zeros((30, 30), dtype=np.float32)
        candidates = detect_star_candidates(img)
        assert len(candidates) == 0
