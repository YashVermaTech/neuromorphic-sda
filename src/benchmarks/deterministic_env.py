"""
Deterministic Orbital Benchmark Environments.

Simulates orbital scenarios (LEO, MEO, GEO) with ground-truth satellite
trajectories for reproducible evaluation of SDA detection systems.

Each scenario generates:
  - A sequence of event frames (via OrbitalToEvents)
  - Ground-truth bounding boxes and trajectory
  - Environmental conditions (sun angle, Earth albedo, star density)

All scenarios are seeded for perfect reproducibility.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Dict, Generator, Iterator, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ── Orbital parameters ────────────────────────────────────────────────────

GM_EARTH = 3.986e14    # m³/s²  – Earth's gravitational parameter
R_EARTH = 6.371e6      # m      – Earth's mean radius

ORBIT_PRESETS: Dict[str, Dict] = {
    "leo": {
        "altitude_km": 550,
        "inclination_deg": 53.0,
        "description": "Low Earth Orbit (Starlink-like)",
    },
    "meo": {
        "altitude_km": 20200,
        "inclination_deg": 55.0,
        "description": "Medium Earth Orbit (GPS-like)",
    },
    "geo": {
        "altitude_km": 35786,
        "inclination_deg": 0.0,
        "description": "Geostationary Orbit",
    },
    "sso": {
        "altitude_km": 500,
        "inclination_deg": 97.4,
        "description": "Sun-Synchronous Orbit",
    },
    "heo": {
        "altitude_km": 39000,
        "inclination_deg": 63.4,
        "description": "Highly Elliptical Orbit (Molniya-like)",
    },
}


def orbital_period_s(altitude_km: float) -> float:
    """Return orbital period in seconds for a circular orbit at *altitude_km*."""
    r = R_EARTH + altitude_km * 1e3
    return 2 * math.pi * math.sqrt(r ** 3 / GM_EARTH)


def angular_velocity_deg_s(altitude_km: float) -> float:
    """Return angular velocity in degrees per second for a circular orbit."""
    T = orbital_period_s(altitude_km)
    return 360.0 / T


# ── Trajectory generation ─────────────────────────────────────────────────


@dataclass
class OrbitalScenario:
    """
    A single orbital benchmark scenario.

    Attributes
    ----------
    scenario_id : str
    orbit_type : str
    altitude_km : float
    inclination_deg : float
    duration_s : float
    fps : float
    seed : int
    sensor_width, sensor_height : int
    description : str
    """
    scenario_id: str
    orbit_type: str
    altitude_km: float
    inclination_deg: float
    duration_s: float = 60.0
    fps: float = 30.0
    seed: int = 42
    sensor_width: int = 346
    sensor_height: int = 260
    description: str = ""

    # Filled in during simulation
    frames: List[np.ndarray] = field(default_factory=list, repr=False)
    gt_boxes: List[np.ndarray] = field(default_factory=list, repr=False)
    gt_timestamps_us: List[int] = field(default_factory=list, repr=False)
    gt_positions_px: List[Tuple[float, float]] = field(default_factory=list, repr=False)

    @property
    def n_frames(self) -> int:
        return int(self.duration_s * self.fps)

    @property
    def angular_velocity(self) -> float:
        return angular_velocity_deg_s(self.altitude_km)

    def __repr__(self) -> str:
        return (
            f"OrbitalScenario(id='{self.scenario_id}', orbit={self.orbit_type}, "
            f"alt={self.altitude_km:.0f}km, fps={self.fps}, frames={self.n_frames})"
        )


class SatelliteMotionModel:
    """
    Generate pixel-space satellite trajectory for a given orbit.

    The satellite moves across the sensor as a point source, with
    Gaussian PSF and adjustable apparent magnitude based on altitude
    and solar angle.

    Parameters
    ----------
    scenario : OrbitalScenario
    n_satellites : int
        Number of satellites to simulate simultaneously.
    psf_sigma : float
        Point spread function width (pixels).
    """

    def __init__(
        self,
        scenario: OrbitalScenario,
        n_satellites: int = 1,
        psf_sigma: float = 1.5,
    ) -> None:
        self.scenario = scenario
        self.n_sats = n_satellites
        self.psf_sigma = psf_sigma
        self.rng = np.random.default_rng(scenario.seed)

        W, H = scenario.sensor_width, scenario.sensor_height
        # Initial positions (random, not too close to edge)
        margin = 20
        self.positions = np.stack([
            self.rng.uniform(margin, W - margin, n_satellites),
            self.rng.uniform(margin, H - margin, n_satellites),
        ], axis=1)  # (N, 2) [x, y]

        # Pixel velocity: map orbital angular velocity → pixels/frame
        pix_per_deg = W / 20.0  # assume 20° FOV
        ang_vel = scenario.angular_velocity  # deg/s
        speed_px_s = ang_vel * pix_per_deg
        angles = self.rng.uniform(0, 2 * math.pi, n_satellites)
        self.velocities = np.stack([
            np.cos(angles) * speed_px_s,
            np.sin(angles) * speed_px_s,
        ], axis=1)  # (N, 2)

        dt = 1.0 / scenario.fps
        self.velocities_px_frame = self.velocities * dt

        # Apparent brightness (inverse-square scaled by altitude)
        self.brightness = 200.0 * (550.0 / max(scenario.altitude_km, 1.0)) ** 2
        self.brightness = float(np.clip(self.brightness, 5.0, 240.0))

    def step(self) -> List[Tuple[float, float]]:
        """
        Advance all satellites by one frame.

        Returns
        -------
        list of (x, y) current positions
        """
        self.positions += self.velocities_px_frame
        W, H = self.scenario.sensor_width, self.scenario.sensor_height
        # Wrap around sensor edges
        self.positions[:, 0] %= W
        self.positions[:, 1] %= H
        return [(float(p[0]), float(p[1])) for p in self.positions]

    def render_to_frame(
        self,
        background: np.ndarray,
        positions: Optional[List[Tuple[float, float]]] = None,
    ) -> np.ndarray:
        """
        Render satellite PSFs onto a background frame.

        Parameters
        ----------
        background : np.ndarray  shape (H, W), dtype uint8
        positions : list of (x, y) | None
            If None, uses current self.positions.

        Returns
        -------
        np.ndarray  shape (H, W), dtype uint8
        """
        frame = background.copy().astype(np.float32)
        H, W = frame.shape
        positions = positions or [(float(p[0]), float(p[1])) for p in self.positions]

        for x, y in positions:
            # Render Gaussian PSF
            xi, yi = int(round(x)), int(round(y))
            r = int(self.psf_sigma * 4)
            x_lo = max(0, xi - r)
            x_hi = min(W, xi + r + 1)
            y_lo = max(0, yi - r)
            y_hi = min(H, yi + r + 1)

            xs = np.arange(x_lo, x_hi)
            ys = np.arange(y_lo, y_hi)
            xx, yy = np.meshgrid(xs, ys)
            dist2 = (xx - x) ** 2 + (yy - y) ** 2
            psf = self.brightness * np.exp(-dist2 / (2 * self.psf_sigma ** 2))
            frame[y_lo:y_hi, x_lo:x_hi] += psf

        return np.clip(frame, 0, 255).astype(np.uint8)


class StarfieldGenerator:
    """
    Generate a synthetic starfield background for an event-camera scene.

    Parameters
    ----------
    sensor_width, sensor_height : int
    n_stars : int
        Number of background stars.
    seed : int
    """

    def __init__(
        self,
        sensor_width: int = 346,
        sensor_height: int = 260,
        n_stars: int = 150,
        seed: int = 42,
    ) -> None:
        self.W = sensor_width
        self.H = sensor_height
        rng = np.random.default_rng(seed)

        # Fixed star positions and brightnesses
        self.star_x = rng.uniform(0, sensor_width, n_stars)
        self.star_y = rng.uniform(0, sensor_height, n_stars)
        self.star_bright = rng.exponential(20.0, n_stars).clip(5.0, 200.0)

    def render(self, twinkle_std: float = 2.0, seed: int = 0) -> np.ndarray:
        """
        Render the starfield as a uint8 grayscale frame.

        Parameters
        ----------
        twinkle_std : float
            Brightness variation per frame (atmospheric scintillation).
        seed : int
            Per-frame random seed for twinkle variation.

        Returns
        -------
        np.ndarray  shape (H, W), dtype uint8
        """
        rng = np.random.default_rng(seed)
        frame = np.zeros((self.H, self.W), dtype=np.float32)
        bright = self.star_bright + rng.normal(0, twinkle_std, len(self.star_bright))
        bright = bright.clip(0, 255)

        for x, y, b in zip(self.star_x, self.star_y, bright):
            xi, yi = int(round(x)), int(round(y))
            if 0 <= xi < self.W and 0 <= yi < self.H:
                frame[yi, xi] += b

        return np.clip(frame, 0, 255).astype(np.uint8)


# ── Main benchmark environment ────────────────────────────────────────────


class OrbitalBenchmarkEnv:
    """
    Deterministic benchmark environment for SDA evaluation.

    Generates reproducible orbital scenarios with known ground truth
    for rigorous evaluation of event-based detection systems.

    Parameters
    ----------
    sensor_width, sensor_height : int
    seed : int
        Master random seed (all sub-components derive from this).
    n_background_stars : int
        Number of background stars in the starfield.

    Examples
    --------
    >>> env = OrbitalBenchmarkEnv(seed=42)
    >>> scenario = env.create_scenario("leo", duration_s=30.0, fps=30.0)
    >>> env.run_scenario(scenario)
    >>> print(f"Generated {scenario.n_frames} frames with GT boxes")
    """

    def __init__(
        self,
        sensor_width: int = 346,
        sensor_height: int = 260,
        seed: int = 42,
        n_background_stars: int = 150,
    ) -> None:
        self.sensor_width = sensor_width
        self.sensor_height = sensor_height
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.starfield = StarfieldGenerator(
            sensor_width, sensor_height,
            n_stars=n_background_stars,
            seed=seed,
        )
        self._scenario_counter = 0

    def create_scenario(
        self,
        orbit_type: str = "leo",
        duration_s: float = 60.0,
        fps: float = 30.0,
        n_satellites: int = 1,
        custom_altitude_km: Optional[float] = None,
    ) -> OrbitalScenario:
        """
        Create an OrbitalScenario with preset parameters.

        Parameters
        ----------
        orbit_type : str
            One of ``"leo"``, ``"meo"``, ``"geo"``, ``"sso"``, ``"heo"``.
        duration_s : float
        fps : float
        n_satellites : int
        custom_altitude_km : float | None
            Override the preset altitude.

        Returns
        -------
        OrbitalScenario
        """
        if orbit_type not in ORBIT_PRESETS:
            raise ValueError(
                f"Unknown orbit type '{orbit_type}'. "
                f"Choose from {list(ORBIT_PRESETS)}"
            )
        preset = ORBIT_PRESETS[orbit_type]
        altitude = custom_altitude_km or preset["altitude_km"]
        scenario_id = f"{orbit_type}_{self._scenario_counter:04d}"
        self._scenario_counter += 1

        scenario = OrbitalScenario(
            scenario_id=scenario_id,
            orbit_type=orbit_type,
            altitude_km=float(altitude),
            inclination_deg=float(preset["inclination_deg"]),
            duration_s=duration_s,
            fps=fps,
            seed=int(self.rng.integers(0, 2 ** 31)),
            sensor_width=self.sensor_width,
            sensor_height=self.sensor_height,
            description=preset["description"],
        )
        return scenario

    def run_scenario(
        self,
        scenario: OrbitalScenario,
        n_satellites: int = 1,
        *,
        show_progress: bool = True,
    ) -> OrbitalScenario:
        """
        Simulate the scenario: generate frames + ground-truth boxes.

        Parameters
        ----------
        scenario : OrbitalScenario
        n_satellites : int
        show_progress : bool

        Returns
        -------
        OrbitalScenario  (mutated in-place with frames and gt_boxes)
        """
        from tqdm import tqdm

        motion = SatelliteMotionModel(scenario, n_satellites=n_satellites)
        box_half = max(3, int(motion.psf_sigma * 3))

        scenario.frames.clear()
        scenario.gt_boxes.clear()
        scenario.gt_timestamps_us.clear()
        scenario.gt_positions_px.clear()

        frame_interval_us = int(1_000_000 / scenario.fps)

        iterator = tqdm(range(scenario.n_frames), desc=f"Scenario {scenario.scenario_id}",
                        disable=not show_progress)

        for frame_idx in iterator:
            t_us = frame_idx * frame_interval_us
            bg = self.starfield.render(twinkle_std=2.0, seed=scenario.seed + frame_idx)
            positions = motion.step()
            frame = motion.render_to_frame(bg, positions)

            # Ground truth bounding boxes  [cx, cy, w, h]
            boxes = []
            for (x, y) in positions:
                cx = float(x) / self.sensor_width
                cy = float(y) / self.sensor_height
                w = float(box_half * 2) / self.sensor_width
                h = float(box_half * 2) / self.sensor_height
                boxes.append([cx, cy, w, h])

            scenario.frames.append(frame)
            scenario.gt_boxes.append(np.array(boxes, dtype=np.float32))
            scenario.gt_timestamps_us.append(t_us)
            scenario.gt_positions_px.extend(positions)

        logger.info(
            "Scenario '%s' complete: %d frames, %d satellites, orbit=%s %.0f km",
            scenario.scenario_id, scenario.n_frames, n_satellites,
            scenario.orbit_type, scenario.altitude_km,
        )
        return scenario

    def run_standard_suite(
        self,
        duration_s: float = 30.0,
        fps: float = 30.0,
    ) -> Dict[str, OrbitalScenario]:
        """
        Run the standard benchmark suite (LEO, MEO, GEO).

        Parameters
        ----------
        duration_s : float
        fps : float

        Returns
        -------
        dict  orbit_type → OrbitalScenario
        """
        results: Dict[str, OrbitalScenario] = {}
        for orbit in ["leo", "meo", "geo"]:
            scenario = self.create_scenario(orbit, duration_s=duration_s, fps=fps)
            results[orbit] = self.run_scenario(scenario)
        return results

    def frames_to_event_stream(
        self,
        scenario: OrbitalScenario,
        *,
        threshold_pos: float = 0.15,
        threshold_neg: float = 0.15,
        shot_noise: float = 0.5,
    ):
        """
        Convert a scenario's frames to an event stream.

        Requires the data_pipeline module.

        Parameters
        ----------
        scenario : OrbitalScenario
        threshold_pos, threshold_neg : float
        shot_noise : float

        Returns
        -------
        EventStream
        """
        from ..data_pipeline.orbital_to_events import OrbitalToEvents

        converter = OrbitalToEvents(
            sensor_width=self.sensor_width,
            sensor_height=self.sensor_height,
            threshold_pos=threshold_pos,
            threshold_neg=threshold_neg,
            shot_noise_rate_hz=shot_noise,
            seed=scenario.seed,
        )
        return converter.convert(scenario.frames, fps=scenario.fps)


def main() -> None:
    """CLI entry point for running benchmarks."""
    import argparse

    parser = argparse.ArgumentParser(description="Run neuromorphic SDA benchmarks")
    parser.add_argument("--orbit", default="leo",
                        choices=list(ORBIT_PRESETS),
                        help="Orbital scenario type")
    parser.add_argument("--duration", type=float, default=30.0,
                        help="Scenario duration in seconds")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    env = OrbitalBenchmarkEnv(seed=args.seed)
    scenario = env.create_scenario(args.orbit, duration_s=args.duration, fps=args.fps)
    env.run_scenario(scenario)
    print(scenario)
    print(f"First frame shape: {scenario.frames[0].shape}")
    print(f"GT boxes (frame 0): {scenario.gt_boxes[0]}")


if __name__ == "__main__":
    main()
