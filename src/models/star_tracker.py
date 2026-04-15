"""
Event-Based Star Tracker with Kalman Filter Prediction.

Detects and tracks stars from event streams, distinguishes satellites
from stars via angular velocity, and outputs attitude determination data.

Pipeline
--------
1. Accumulate events in a time window → 2-D frame
2. Detect star candidates (connected components)
3. Refine positions with centroiding
4. Predict positions with per-star Kalman filters
5. Match detections to catalogue (HYG database subset)
6. Identify moving objects (satellites) by velocity threshold
7. Output RA/Dec attitude solution + satellite candidates

References
----------
Liebe, C. C. (1995). Star trackers for attitude determination.
  IEEE Aerospace and Electronic Systems Magazine, 10(6), 10-16.

Mughal, M. R., et al. (2014). Event-based Star Tracker for Attitude
  Determination. AIAA Guidance Navigation and Control Conference.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .centroiding import Centroider, CentroidResult, detect_star_candidates

logger = logging.getLogger(__name__)


# ── Data classes ──────────────────────────────────────────────────────────


@dataclass
class StarCandidate:
    """A detected star / point source candidate."""
    track_id: int
    x: float                 # Pixel column (sub-pixel)
    y: float                 # Pixel row    (sub-pixel)
    flux: float              # Integrated flux (ADU)
    snr: float               # Signal-to-noise ratio
    timestamp_us: int        # Detection timestamp
    ra: Optional[float] = None   # Right ascension (degrees) – set after catalogue match
    dec: Optional[float] = None  # Declination (degrees)
    is_satellite: bool = False   # Flagged as satellite by velocity test
    angular_velocity: float = 0.0  # arcsec/s


@dataclass
class AttitudeSolution:
    """Attitude determination output."""
    timestamp_us: int
    ra_boresight: float       # Boresight RA (degrees)
    dec_boresight: float      # Boresight Dec (degrees)
    roll_deg: float           # Roll angle
    n_matched_stars: int
    residual_arcsec: float    # RMS residual of matched stars
    confidence: float         # 0–1 solution confidence


# ── Simplified star catalogue ─────────────────────────────────────────────


def _build_hyg_subset(n_stars: int = 200, seed: int = 42) -> np.ndarray:
    """
    Build a synthetic HYG-like catalogue subset for offline use.

    In production this would load from the real HYG database CSV.
    Here we generate a plausible distribution of bright stars.

    Returns
    -------
    np.ndarray with dtype [("ra", f8), ("dec", f8), ("mag", f4), ("hip", i4)]
    """
    rng = np.random.default_rng(seed)
    dtype = np.dtype([("ra", np.float64), ("dec", np.float64),
                      ("mag", np.float32), ("hip", np.int32)])
    catalogue = np.empty(n_stars, dtype=dtype)

    # Realistic sky distribution (all-sky, biased toward equatorial band)
    catalogue["ra"] = rng.uniform(0, 360, n_stars)
    catalogue["dec"] = np.degrees(np.arcsin(rng.uniform(-0.8, 0.8, n_stars)))
    catalogue["mag"] = rng.uniform(1.0, 6.5, n_stars).astype(np.float32)
    catalogue["hip"] = rng.integers(1, 119_000, n_stars)

    return catalogue


# ── Kalman filter for single-star tracking ───────────────────────────────


class StarKalmanFilter:
    """
    Constant-velocity Kalman filter for tracking a single star pixel position.

    State vector: [x, y, vx, vy]

    Parameters
    ----------
    x0, y0 : float
        Initial position.
    dt : float
        Time step in seconds.
    process_noise_std : float
        Process noise standard deviation (pixels/step²).
    measurement_noise_std : float
        Measurement noise standard deviation (pixels).
    """

    def __init__(
        self,
        x0: float,
        y0: float,
        dt: float = 0.01,
        process_noise_std: float = 0.1,
        measurement_noise_std: float = 0.5,
    ) -> None:
        self.dt = dt
        self.state = np.array([x0, y0, 0.0, 0.0], dtype=np.float64)

        # State transition matrix
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1],
        ], dtype=np.float64)

        # Observation matrix (observe x, y only)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float64)

        q = process_noise_std ** 2
        self.Q = np.eye(4, dtype=np.float64) * q

        r = measurement_noise_std ** 2
        self.R = np.eye(2, dtype=np.float64) * r

        self.P = np.eye(4, dtype=np.float64) * 1.0

    def predict(self) -> np.ndarray:
        """Predict the next state. Returns predicted [x, y]."""
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.state[:2].copy()

    def update(self, measurement: np.ndarray) -> np.ndarray:
        """
        Update with a new measurement.

        Parameters
        ----------
        measurement : np.ndarray  shape (2,)  [x, y]

        Returns
        -------
        np.ndarray  shape (2,)  updated [x, y]
        """
        z = measurement.reshape(2)
        y = z - self.H @ self.state  # innovation
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        return self.state[:2].copy()

    @property
    def velocity(self) -> Tuple[float, float]:
        """Current velocity estimate (pixels / second)."""
        return float(self.state[2]), float(self.state[3])

    @property
    def speed(self) -> float:
        """Scalar speed (pixels / second)."""
        vx, vy = self.velocity
        return math.sqrt(vx ** 2 + vy ** 2)


# ── Main star tracker class ───────────────────────────────────────────────


class EventStarTracker:
    """
    Event-based star tracker with Kalman filter prediction.

    Parameters
    ----------
    sensor_width : int
    sensor_height : int
    fov_deg : float
        Camera field of view (diagonal, degrees).
    event_window_ms : float
        Time window for event accumulation (milliseconds).
    min_star_snr : float
        Minimum SNR for a detection to be accepted.
    magnitude_limit : float
        Limiting magnitude for catalogue stars.
    satellite_vel_threshold_arcsec_s : float
        Angular velocity above which an object is classified as a satellite.
    dt_s : float
        Kalman filter time step in seconds.
    catalogue_size : int
        Number of stars in the built-in synthetic HYG catalogue.
    seed : int
        RNG seed.

    Examples
    --------
    >>> tracker = EventStarTracker(sensor_width=346, sensor_height=260)
    >>> # Feed event stream windows
    >>> for window_events in event_windows:
    ...     solution = tracker.update(window_events, timestamp_us=t)
    ...     if solution:
    ...         print(f"RA={solution.ra_boresight:.2f} Dec={solution.dec_boresight:.2f}")
    """

    def __init__(
        self,
        sensor_width: int = 346,
        sensor_height: int = 260,
        fov_deg: float = 20.0,
        event_window_ms: float = 10.0,
        min_star_snr: float = 3.0,
        magnitude_limit: float = 6.5,
        satellite_vel_threshold_arcsec_s: float = 10.0,
        dt_s: float = 0.01,
        catalogue_size: int = 200,
        seed: int = 42,
    ) -> None:
        self.sensor_width = sensor_width
        self.sensor_height = sensor_height
        self.fov_deg = fov_deg
        self.event_window_ms = event_window_ms
        self.min_star_snr = min_star_snr
        self.magnitude_limit = magnitude_limit
        self.satellite_vel_threshold_arcsec_s = satellite_vel_threshold_arcsec_s
        self.dt_s = dt_s

        # Pixel scale in arcsec/px
        fov_arcsec = fov_deg * 3600.0
        diagonal_px = math.sqrt(sensor_width ** 2 + sensor_height ** 2)
        self.pixel_scale_arcsec_px = fov_arcsec / diagonal_px

        # Centroider
        self.centroider = Centroider(method="iterative", box_half_size=5)

        # Catalogue
        self.catalogue = _build_hyg_subset(catalogue_size, seed=seed)

        # Track registry: track_id → StarKalmanFilter
        self._tracks: Dict[int, StarKalmanFilter] = {}
        self._track_meta: Dict[int, StarCandidate] = {}
        self._next_id: int = 0

        # Internal accumulation buffer
        self._event_buffer: List[np.ndarray] = []
        self._buffer_t_start: int = 0

        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Main interface
    # ------------------------------------------------------------------

    def update(
        self,
        events: np.ndarray,
        timestamp_us: int,
    ) -> Optional[AttitudeSolution]:
        """
        Process a batch of events and update tracking state.

        Parameters
        ----------
        events : np.ndarray
            Structured event array (EVENT_DTYPE) for this time step.
        timestamp_us : int
            Timestamp of the end of the event batch (microseconds).

        Returns
        -------
        AttitudeSolution | None
            Attitude solution if enough stars are matched, else None.
        """
        if len(events) == 0:
            return None

        # Accumulate ON events into a frame
        frame = self._accumulate_on_events(events)

        # Detect candidates
        raw_positions = detect_star_candidates(frame, min_sigma=self.min_star_snr)
        if not raw_positions:
            return None

        # Refine with centroiding
        centroids: List[CentroidResult] = self.centroider.centroid_batch(
            frame, raw_positions
        )

        # Filter by SNR
        good_centroids = [c for c in centroids if c.snr >= self.min_star_snr]
        if not good_centroids:
            return None

        # Data association (greedy nearest-neighbour)
        matched_tracks = self._associate(good_centroids, timestamp_us)

        # Update Kalman filters
        updated_stars: List[StarCandidate] = []
        for star_cand, track_id in matched_tracks:
            kf = self._tracks[track_id]
            kf.predict()
            kf.update(np.array([star_cand.x, star_cand.y]))

            # Convert pixel velocity → angular velocity in arcsec/s
            ang_vel = kf.speed * self.pixel_scale_arcsec_px / self.dt_s
            star_cand.angular_velocity = ang_vel
            star_cand.is_satellite = (ang_vel >= self.satellite_vel_threshold_arcsec_s)
            star_cand.timestamp_us = timestamp_us
            self._track_meta[track_id] = star_cand
            updated_stars.append(star_cand)

        # Catalogue matching for stationary objects (stars)
        star_objects = [s for s in updated_stars if not s.is_satellite]
        solution = self._solve_attitude(star_objects, timestamp_us)

        return solution

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _accumulate_on_events(self, events: np.ndarray) -> np.ndarray:
        """Create a 2-D event count frame from ON events."""
        frame = np.zeros((self.sensor_height, self.sensor_width), dtype=np.float32)
        on_mask = events["p"] > 0
        ev_on = events[on_mask]
        x = np.clip(ev_on["x"].astype(int), 0, self.sensor_width - 1)
        y = np.clip(ev_on["y"].astype(int), 0, self.sensor_height - 1)
        np.add.at(frame, (y, x), 1.0)
        return frame

    def _associate(
        self,
        centroids: List[CentroidResult],
        timestamp_us: int,
    ) -> List[Tuple[StarCandidate, int]]:
        """
        Associate new detections with existing tracks (greedy NN match).

        New tracks are created for unmatched detections.

        Returns
        -------
        list of (StarCandidate, track_id) pairs
        """
        max_dist_px = self.sensor_width * 0.05  # 5% sensor width
        assigned: List[Tuple[StarCandidate, int]] = []
        used_tracks = set()

        for c in centroids:
            best_id = None
            best_dist = max_dist_px

            for tid, kf in self._tracks.items():
                if tid in used_tracks:
                    continue
                px, py = kf.state[0], kf.state[1]
                dist = math.sqrt((c.x - px) ** 2 + (c.y - py) ** 2)
                if dist < best_dist:
                    best_dist = dist
                    best_id = tid

            if best_id is None:
                # Create new track
                new_id = self._next_id
                self._next_id += 1
                self._tracks[new_id] = StarKalmanFilter(
                    c.x, c.y, dt=self.dt_s
                )
                best_id = new_id

            used_tracks.add(best_id)
            star = StarCandidate(
                track_id=best_id,
                x=c.x,
                y=c.y,
                flux=c.flux,
                snr=c.snr,
                timestamp_us=timestamp_us,
            )
            assigned.append((star, best_id))

        return assigned

    def _solve_attitude(
        self,
        stars: List[StarCandidate],
        timestamp_us: int,
    ) -> Optional[AttitudeSolution]:
        """
        Simplified attitude determination via catalogue matching.

        Maps pixel positions to RA/Dec using a synthetic plate solution.
        In production this would use QUEST or DAVENPORT algorithms.

        Parameters
        ----------
        stars : list of StarCandidate
        timestamp_us : int

        Returns
        -------
        AttitudeSolution | None
        """
        if len(stars) < 3:
            return None

        # Assign synthetic RA/Dec for each matched star
        # (In a real system, triangle-match against catalogue)
        n_matched = min(len(stars), len(self.catalogue))
        sorted_cat = self.catalogue[self.catalogue["mag"].argsort()][:n_matched]

        for i, star in enumerate(stars[:n_matched]):
            star.ra = float(sorted_cat["ra"][i])
            star.dec = float(sorted_cat["dec"][i])

        # Compute boresight as weighted mean of matched stars
        weights = np.array([s.flux for s in stars[:n_matched]])
        weights /= weights.sum() + 1e-9

        ra_vals = np.array([s.ra for s in stars[:n_matched]])
        dec_vals = np.array([s.dec for s in stars[:n_matched]])

        ra_boresight = float(np.dot(weights, ra_vals))
        dec_boresight = float(np.dot(weights, dec_vals))

        # Synthetic residual (pixel_scale * 0.1 px RMS)
        residual = self.pixel_scale_arcsec_px * 0.1

        confidence = min(1.0, n_matched / 10.0)

        return AttitudeSolution(
            timestamp_us=timestamp_us,
            ra_boresight=ra_boresight,
            dec_boresight=dec_boresight,
            roll_deg=0.0,
            n_matched_stars=n_matched,
            residual_arcsec=residual,
            confidence=confidence,
        )

    @property
    def satellite_candidates(self) -> List[StarCandidate]:
        """Return currently tracked satellite candidates."""
        return [s for s in self._track_meta.values() if s.is_satellite]

    @property
    def n_tracked_stars(self) -> int:
        """Number of currently active tracks."""
        return len(self._tracks)

    def reset(self) -> None:
        """Clear all tracks."""
        self._tracks.clear()
        self._track_meta.clear()
        self._next_id = 0

    def __repr__(self) -> str:
        return (
            f"EventStarTracker(fov={self.fov_deg}°, "
            f"scale={self.pixel_scale_arcsec_px:.2f}\"/px, "
            f"tracks={self.n_tracked_stars})"
        )
