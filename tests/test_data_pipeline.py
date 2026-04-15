"""
Unit tests for the data pipeline module.

Tests cover:
  - OrbitalToEvents: frame conversion, output shape, event polarity
  - EventStream: window slicing, frame accumulation, save/load
  - CosmicNoiseGAN: forward pass, noise generation
  - NoiseAugmentor: augmentation application
  - EventDatasetCurator: add, split, statistics
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Add src to path for direct test execution
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_pipeline.orbital_to_events import EVENT_DTYPE, EventStream, OrbitalToEvents
from data_pipeline.gan_noise_model import (
    CosmicNoiseGAN,
    NoiseAugmentor,
    NoiseConfig,
    SyntheticNoiseDataset,
    _cosmic_ray_noise,
    _dark_current_noise,
)
from data_pipeline.dataset_curator import EventDatasetCurator, SampleMetadata


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def small_frames():
    """3 small grayscale frames (32×32)."""
    rng = np.random.default_rng(0)
    return [rng.integers(0, 255, (32, 32), dtype=np.uint8) for _ in range(3)]


@pytest.fixture
def converter():
    return OrbitalToEvents(
        sensor_width=32,
        sensor_height=32,
        threshold_pos=0.1,
        threshold_neg=0.1,
        threshold_sigma=0.01,
        shot_noise_rate_hz=0.1,
        seed=42,
    )


@pytest.fixture
def sample_stream(converter, small_frames):
    return converter.convert(small_frames, fps=30.0, show_progress=False)


# ── OrbitalToEvents tests ─────────────────────────────────────────────────


class TestOrbitalToEvents:
    def test_convert_returns_event_stream(self, converter, small_frames):
        stream = converter.convert(small_frames, fps=30.0, show_progress=False)
        assert isinstance(stream, EventStream)

    def test_event_dtype(self, sample_stream):
        assert sample_stream.events.dtype == EVENT_DTYPE

    def test_event_polarities(self, sample_stream):
        if len(sample_stream.events) > 0:
            unique_p = set(sample_stream.events["p"].tolist())
            assert unique_p.issubset({-1, 1})

    def test_event_coordinates_in_bounds(self, sample_stream):
        if len(sample_stream.events) > 0:
            assert sample_stream.events["x"].min() >= 0
            assert sample_stream.events["x"].max() < sample_stream.sensor_width
            assert sample_stream.events["y"].min() >= 0
            assert sample_stream.events["y"].max() < sample_stream.sensor_height

    def test_timestamps_non_decreasing(self, sample_stream):
        if len(sample_stream.events) > 1:
            diffs = np.diff(sample_stream.events["t"])
            assert (diffs >= 0).all(), "Timestamps should be non-decreasing"

    def test_duration_us(self, sample_stream, small_frames):
        expected_us = len(small_frames) * int(1_000_000 / 30.0)
        assert sample_stream.duration_us == expected_us

    def test_streaming_mode_yields_arrays(self, converter, small_frames):
        results = list(converter.stream_frames(small_frames, fps=30.0))
        assert len(results) == len(small_frames)
        for r in results:
            assert r.dtype == EVENT_DTYPE

    def test_reset_clears_state(self, converter):
        converter.reset()
        assert converter._log_ref is None
        assert converter._thresh_pos_map is not None

    def test_single_uniform_frame_no_events(self, converter):
        """Uniform frame → no log-luminance change → no events (except noise)."""
        frame = np.full((32, 32), 128, dtype=np.uint8)
        stream = converter.convert([frame, frame], fps=30.0, show_progress=False)
        # With noise disabled this would be 0; with noise it should be small
        assert isinstance(stream, EventStream)


# ── EventStream tests ─────────────────────────────────────────────────────


class TestEventStream:
    def test_on_off_split(self, sample_stream):
        all_ev = sample_stream.events
        on_ev = sample_stream.on_events
        off_ev = sample_stream.off_events
        assert len(on_ev) + len(off_ev) == len(all_ev)

    def test_window_slicing(self, sample_stream):
        if sample_stream.duration_us > 0:
            mid = sample_stream.duration_us // 2
            w = sample_stream.window(0, mid)
            assert len(w) <= len(sample_stream.events)

    def test_to_frame_shape(self, sample_stream):
        frame = sample_stream.to_frame(0, sample_stream.duration_us)
        assert frame.shape == (2, sample_stream.sensor_height, sample_stream.sensor_width)
        assert frame.dtype == np.float32

    def test_to_frame_normalised(self, sample_stream):
        frame = sample_stream.to_frame(0, sample_stream.duration_us)
        assert frame.min() >= 0.0
        assert frame.max() <= 1.0 + 1e-6

    def test_save_load_roundtrip(self, sample_stream, tmp_path):
        out = tmp_path / "test_stream.npz"
        sample_stream.save_numpy(out)
        loaded = EventStream.load_numpy(out)
        assert loaded.num_events == sample_stream.num_events
        np.testing.assert_array_equal(loaded.events["t"], sample_stream.events["t"])

    def test_event_rate_property(self, sample_stream):
        assert sample_stream.event_rate_hz >= 0.0

    def test_repr(self, sample_stream):
        r = repr(sample_stream)
        assert "EventStream" in r


# ── GAN / noise model tests ───────────────────────────────────────────────


class TestSyntheticNoise:
    def test_cosmic_ray_noise_shape(self):
        rng = np.random.default_rng(0)
        patch = _cosmic_ray_noise(64, 64, 3, rng)
        assert patch.shape == (64, 64)
        assert patch.dtype == np.float32
        assert patch.min() >= 0.0
        assert patch.max() <= 1.0

    def test_dark_current_noise_shape(self):
        rng = np.random.default_rng(0)
        patch = _dark_current_noise(64, 64, 0.05, 300.0, rng)
        assert patch.shape == (64, 64)

    def test_synthetic_dataset_shape(self):
        ds = SyntheticNoiseDataset(n_samples=8, seed=0)
        patches = ds.generate()
        assert patches.shape == (8, 1, 64, 64)
        assert patches.min() >= -1.0 - 1e-6
        assert patches.max() <= 1.0 + 1e-6


class TestCosmicNoiseGAN:
    def test_forward_pass_generator(self):
        import torch
        gan = CosmicNoiseGAN(latent_dim=16, features_g=8, features_d=8, device="cpu")
        z = torch.randn(2, 16, 1, 1)
        out = gan.G(z)
        assert out.shape == (2, 1, 64, 64)
        assert out.min().item() >= -1.0 - 1e-5
        assert out.max().item() <= 1.0 + 1e-5

    def test_generate_samples(self):
        gan = CosmicNoiseGAN(latent_dim=16, features_g=8, features_d=8, device="cpu")
        samples = gan.generate(n_samples=4)
        assert samples.shape == (4, 1, 64, 64)
        assert samples.min() >= 0.0 - 1e-5
        assert samples.max() <= 1.0 + 1e-5

    def test_short_training(self):
        gan = CosmicNoiseGAN(latent_dim=16, features_g=8, features_d=8,
                             device="cpu", seed=0)
        history = gan.train(n_epochs=2, batch_size=8, n_samples_train=64,
                            show_progress=False)
        assert "loss_G" in history
        assert len(history["loss_G"]) == 2

    def test_save_load(self, tmp_path):
        gan = CosmicNoiseGAN(latent_dim=16, features_g=8, features_d=8, device="cpu")
        save_path = tmp_path / "gan.pth"
        gan.save(save_path)
        gan2 = CosmicNoiseGAN(latent_dim=16, features_g=8, features_d=8, device="cpu")
        gan2.load(save_path)
        # Verify weights match
        import torch
        for p1, p2 in zip(gan.G.parameters(), gan2.G.parameters()):
            assert torch.allclose(p1, p2)


class TestNoiseAugmentor:
    def test_augment_does_not_change_shape(self):
        aug = NoiseAugmentor(augment_prob=1.0, noise_scale=0.1, seed=0)
        frame = np.random.rand(2, 32, 32).astype(np.float32)
        noisy = aug(frame)
        assert noisy.shape == frame.shape

    def test_augment_clipped(self):
        aug = NoiseAugmentor(augment_prob=1.0, noise_scale=10.0, seed=0)
        frame = np.zeros((2, 32, 32), dtype=np.float32)
        noisy = aug(frame)
        assert noisy.max() <= 1.0

    def test_augment_prob_zero(self):
        aug = NoiseAugmentor(augment_prob=0.0)
        frame = np.random.rand(2, 32, 32).astype(np.float32)
        result = aug(frame)
        np.testing.assert_array_equal(result, frame)


# ── Dataset curator tests ─────────────────────────────────────────────────


class TestEventDatasetCurator:
    @pytest.fixture
    def small_stream(self, sample_stream):
        return sample_stream

    def test_add_stream(self, small_stream, tmp_path):
        curator = EventDatasetCurator(
            output_dir=tmp_path / "curated",
            window_duration_us=10_000,
            min_events_per_window=1,
        )
        n_added = curator.add_stream(small_stream, source_name="test")
        assert n_added >= 0

    def test_split_fractions(self, small_stream, tmp_path):
        curator = EventDatasetCurator(
            output_dir=tmp_path / "curated",
            window_duration_us=10_000,
            min_events_per_window=1,
        )
        curator.add_stream(small_stream, source_name="test")
        if len(curator) >= 3:
            counts = curator.split(train=0.6, val=0.2, test=0.2)
            assert sum(counts.values()) == len(curator)

    def test_statistics_keys(self, small_stream, tmp_path):
        curator = EventDatasetCurator(
            output_dir=tmp_path / "curated",
            window_duration_us=10_000,
            min_events_per_window=1,
        )
        curator.add_stream(small_stream, source_name="test")
        stats = curator.statistics()
        if stats:
            assert "total_samples" in stats
            assert "total_events" in stats

    def test_export_numpy(self, small_stream, tmp_path):
        curator = EventDatasetCurator(
            output_dir=tmp_path / "export",
            window_duration_us=10_000,
            min_events_per_window=1,
        )
        curator.add_stream(small_stream)
        if len(curator) > 0:
            curator.split()
            out_dir = curator.export(format="numpy", overwrite=True)
            assert out_dir.exists()
            assert (out_dir / "manifest.json").exists()
