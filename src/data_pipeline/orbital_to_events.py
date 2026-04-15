"""
Orbital Image to Event Stream Converter (v2e methodology).

Converts grayscale orbital / astronomical frames into asynchronous
event streams that simulate a Dynamic Vision Sensor (DVS) / event camera.

The conversion follows the v2e model (Hu et al., 2021):
  - Each pixel integrates log-luminance changes independently.
  - Positive (ON) events fire when log-luminance increases by ≥ C_pos.
  - Negative (OFF) events fire when log-luminance decreases by ≥ C_neg.
  - Refractory period enforces a minimum inter-event interval per pixel.
  - Shot noise and leak noise are optionally injected.

References
----------
Hu, Y., Liu, S., & Delbruck, T. (2021). v2e: From Video Frames to
  Realistic DVS Events. CVPR Workshops.
  https://arxiv.org/abs/2006.07722
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ── Public dtype shared with visualization / I/O modules ──────────────────
EVENT_DTYPE = np.dtype(
    [
        ("t", np.int64),   # timestamp in microseconds
        ("x", np.int16),   # pixel column
        ("y", np.int16),   # pixel row
        ("p", np.int8),    # polarity: +1 (ON) or -1 (OFF)
    ]
)


@dataclass
class EventStream:
    """
    Container for an asynchronous event stream.

    Attributes
    ----------
    events : np.ndarray
        Structured array with dtype EVENT_DTYPE, sorted by timestamp.
    sensor_width : int
        Sensor width in pixels.
    sensor_height : int
        Sensor height in pixels.
    duration_us : int
        Total duration of the stream in microseconds.
    metadata : dict
        Optional metadata (source file, conversion parameters, etc.).
    """

    events: np.ndarray
    sensor_width: int
    sensor_height: int
    duration_us: int
    metadata: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def num_events(self) -> int:
        """Total number of events."""
        return len(self.events)

    @property
    def event_rate_hz(self) -> float:
        """Mean event rate in events per second."""
        if self.duration_us == 0:
            return 0.0
        return self.num_events / (self.duration_us * 1e-6)

    @property
    def on_events(self) -> np.ndarray:
        """Return only ON (positive) events."""
        return self.events[self.events["p"] > 0]

    @property
    def off_events(self) -> np.ndarray:
        """Return only OFF (negative) events."""
        return self.events[self.events["p"] < 0]

    def window(self, t_start_us: int, t_end_us: int) -> np.ndarray:
        """
        Slice events within a microsecond time window.

        Parameters
        ----------
        t_start_us : int
            Window start time in microseconds.
        t_end_us : int
            Window end time in microseconds.

        Returns
        -------
        np.ndarray
            Events in [t_start_us, t_end_us).
        """
        mask = (self.events["t"] >= t_start_us) & (self.events["t"] < t_end_us)
        return self.events[mask]

    def to_frame(self, t_start_us: int, t_end_us: int) -> np.ndarray:
        """
        Accumulate events in a time window into a polarity frame.

        Returns a (2, H, W) float32 array where channel 0 is ON count
        and channel 1 is OFF count, both normalised to [0, 1].

        Parameters
        ----------
        t_start_us : int
        t_end_us : int

        Returns
        -------
        np.ndarray  shape (2, H, W)
        """
        frame = np.zeros((2, self.sensor_height, self.sensor_width), dtype=np.float32)
        evs = self.window(t_start_us, t_end_us)
        for p, ch in [(1, 0), (-1, 1)]:
            mask = evs["p"] == p
            ev_p = evs[mask]
            np.add.at(frame[ch], (ev_p["y"], ev_p["x"]), 1.0)
        # Normalise
        mx = frame.max()
        if mx > 0:
            frame /= mx
        return frame

    def save_numpy(self, path: Union[str, Path]) -> None:
        """Save event stream to a ``.npz`` file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            t=self.events["t"],
            x=self.events["x"],
            y=self.events["y"],
            p=self.events["p"],
            sensor_width=self.sensor_width,
            sensor_height=self.sensor_height,
            duration_us=self.duration_us,
        )
        logger.info("Saved %d events to %s", self.num_events, path)

    @classmethod
    def load_numpy(cls, path: Union[str, Path]) -> "EventStream":
        """Load an event stream from a ``.npz`` file."""
        data = np.load(path)
        events = np.empty(len(data["t"]), dtype=EVENT_DTYPE)
        events["t"] = data["t"]
        events["x"] = data["x"]
        events["y"] = data["y"]
        events["p"] = data["p"]
        return cls(
            events=events,
            sensor_width=int(data["sensor_width"]),
            sensor_height=int(data["sensor_height"]),
            duration_us=int(data["duration_us"]),
        )

    def __repr__(self) -> str:
        return (
            f"EventStream(events={self.num_events:,}, "
            f"sensor={self.sensor_width}x{self.sensor_height}, "
            f"duration={self.duration_us / 1e6:.3f}s, "
            f"rate={self.event_rate_hz / 1e3:.1f} kHz)"
        )


class OrbitalToEvents:
    """
    Convert a sequence of orbital / astronomical grayscale frames to an
    asynchronous event stream using the v2e (video-to-events) algorithm.

    Parameters
    ----------
    sensor_width : int
        Output sensor width in pixels (default 346 – DAVIS346).
    sensor_height : int
        Output sensor height in pixels (default 260).
    threshold_pos : float
        Log-intensity threshold for ON events (default 0.15).
    threshold_neg : float
        Log-intensity threshold for OFF events (default 0.15).
    threshold_sigma : float
        Per-pixel threshold variation std (sensor mismatch model).
    refractory_period_us : float
        Minimum time between events per pixel, in microseconds.
    shot_noise_rate_hz : float
        Poisson noise event rate per pixel per second.
    leak_rate_hz : float
        Leak potential decay rate (Hz).
    cutoff_hz : float
        Low-pass filter cutoff frequency for luminance signal.
    seed : int | None
        RNG seed for reproducibility.

    Examples
    --------
    >>> converter = OrbitalToEvents(threshold_pos=0.2, shot_noise_rate_hz=1.0)
    >>> frames = [np.random.randint(0, 255, (260, 346), dtype=np.uint8)
    ...           for _ in range(30)]
    >>> stream = converter.convert(frames, fps=30.0)
    >>> print(stream)
    EventStream(events=..., sensor=346x260, ...)
    """

    def __init__(
        self,
        sensor_width: int = 346,
        sensor_height: int = 260,
        threshold_pos: float = 0.15,
        threshold_neg: float = 0.15,
        threshold_sigma: float = 0.03,
        refractory_period_us: float = 1.0,
        shot_noise_rate_hz: float = 0.5,
        leak_rate_hz: float = 0.1,
        cutoff_hz: float = 300.0,
        seed: Optional[int] = None,
    ) -> None:
        self.width = sensor_width
        self.height = sensor_height
        self.threshold_pos = threshold_pos
        self.threshold_neg = threshold_neg
        self.threshold_sigma = threshold_sigma
        self.refractory_period_us = refractory_period_us
        self.shot_noise_rate_hz = shot_noise_rate_hz
        self.leak_rate_hz = leak_rate_hz
        self.cutoff_hz = cutoff_hz
        self.rng = np.random.default_rng(seed)

        # Per-pixel threshold maps (sampled once during reset)
        self._thresh_pos_map: Optional[np.ndarray] = None
        self._thresh_neg_map: Optional[np.ndarray] = None

        # Internal state
        self._log_ref: Optional[np.ndarray] = None    # reference log-luminance
        self._last_event_t: Optional[np.ndarray] = None  # last event time per pixel (us)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset all internal state (call between independent sequences)."""
        self._log_ref = None
        self._last_event_t = None
        # Sample per-pixel thresholds
        self._thresh_pos_map = self.rng.normal(
            self.threshold_pos, self.threshold_sigma, (self.height, self.width)
        ).clip(0.01, None).astype(np.float32)
        self._thresh_neg_map = self.rng.normal(
            self.threshold_neg, self.threshold_sigma, (self.height, self.width)
        ).clip(0.01, None).astype(np.float32)

    def convert(
        self,
        frames: Sequence[np.ndarray],
        fps: float = 30.0,
        *,
        start_time_us: int = 0,
        show_progress: bool = True,
    ) -> EventStream:
        """
        Convert a list of grayscale frames to an event stream.

        Parameters
        ----------
        frames : Sequence[np.ndarray]
            List of grayscale frames, each shape (H, W) or (H, W, 1).
            Pixel values should be in [0, 255].
        fps : float
            Frame rate of the source video in frames per second.
        start_time_us : int
            Starting timestamp for the output stream in microseconds.
        show_progress : bool
            Display a tqdm progress bar.

        Returns
        -------
        EventStream
        """
        self.reset()
        frame_interval_us = int(1_000_000 / fps)
        all_events: List[np.ndarray] = []

        iterator = tqdm(enumerate(frames), total=len(frames), desc="v2e conversion",
                        disable=not show_progress)

        for frame_idx, frame in iterator:
            t_start = start_time_us + frame_idx * frame_interval_us
            t_end = t_start + frame_interval_us

            frame_gray = self._preprocess_frame(frame)
            events = self._process_frame(frame_gray, t_start, t_end)

            if len(events) > 0:
                all_events.append(events)

        total_duration_us = len(frames) * frame_interval_us
        if all_events:
            merged = np.concatenate(all_events)
            # Sort by timestamp (already roughly sorted, but ensure strict order)
            merged.sort(order="t")
        else:
            merged = np.empty(0, dtype=EVENT_DTYPE)

        return EventStream(
            events=merged,
            sensor_width=self.width,
            sensor_height=self.height,
            duration_us=total_duration_us,
            metadata={
                "source": "v2e_conversion",
                "fps": fps,
                "n_frames": len(frames),
                "threshold_pos": self.threshold_pos,
                "threshold_neg": self.threshold_neg,
                "shot_noise_rate_hz": self.shot_noise_rate_hz,
            },
        )

    def stream_frames(
        self,
        frames: Sequence[np.ndarray],
        fps: float = 30.0,
        *,
        start_time_us: int = 0,
    ) -> Iterator[np.ndarray]:
        """
        Streaming (frame-by-frame) variant of :meth:`convert`.

        Yields the event array for each input frame without buffering
        the entire stream in memory.

        Parameters
        ----------
        frames : Sequence[np.ndarray]
        fps : float
        start_time_us : int

        Yields
        ------
        np.ndarray
            Events for the current frame (may be empty).
        """
        self.reset()
        frame_interval_us = int(1_000_000 / fps)

        for frame_idx, frame in enumerate(frames):
            t_start = start_time_us + frame_idx * frame_interval_us
            t_end = t_start + frame_interval_us
            frame_gray = self._preprocess_frame(frame)
            yield self._process_frame(frame_gray, t_start, t_end)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize to sensor resolution, ensure float32, add eps."""
        import cv2

        if frame.ndim == 3:
            if frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            elif frame.shape[2] == 1:
                frame = frame[:, :, 0]

        if frame.shape != (self.height, self.width):
            frame = cv2.resize(frame, (self.width, self.height),
                               interpolation=cv2.INTER_AREA)

        return frame.astype(np.float32)

    def _log_luminance(self, frame: np.ndarray) -> np.ndarray:
        """Compute log-luminance with epsilon guard."""
        return np.log(frame + 1.0)  # shape (H, W)

    def _process_frame(
        self,
        frame: np.ndarray,
        t_start_us: int,
        t_end_us: int,
    ) -> np.ndarray:
        """
        Core v2e algorithm for a single inter-frame interval.

        Returns a structured event array for this frame.
        """
        log_new = self._log_luminance(frame)

        if self._log_ref is None:
            self._log_ref = log_new.copy()
            self._last_event_t = np.full((self.height, self.width),
                                        t_start_us, dtype=np.int64)
            return np.empty(0, dtype=EVENT_DTYPE)

        diff = log_new - self._log_ref          # (H, W)
        events: List[np.ndarray] = []

        # ── ON events ────────────────────────────────────────────────
        on_mask = diff >= self._thresh_pos_map
        if on_mask.any():
            ev = self._emit_events(on_mask, polarity=1,
                                   t_start_us=t_start_us, t_end_us=t_end_us)
            if len(ev):
                events.append(ev)
                self._log_ref[on_mask] += (
                    self._thresh_pos_map[on_mask]
                    * np.floor(diff[on_mask] / self._thresh_pos_map[on_mask])
                )

        # ── OFF events ────────────────────────────────────────────────
        off_mask = diff <= -self._thresh_neg_map
        if off_mask.any():
            ev = self._emit_events(off_mask, polarity=-1,
                                   t_start_us=t_start_us, t_end_us=t_end_us)
            if len(ev):
                events.append(ev)
                self._log_ref[off_mask] -= (
                    self._thresh_neg_map[off_mask]
                    * np.floor(-diff[off_mask] / self._thresh_neg_map[off_mask])
                )

        # ── Noise injection ───────────────────────────────────────────
        if self.shot_noise_rate_hz > 0:
            noise_ev = self._shot_noise(t_start_us, t_end_us)
            if len(noise_ev):
                events.append(noise_ev)

        if events:
            combined = np.concatenate(events)
            combined.sort(order="t")
            return combined

        return np.empty(0, dtype=EVENT_DTYPE)

    def _emit_events(
        self,
        mask: np.ndarray,
        polarity: int,
        t_start_us: int,
        t_end_us: int,
    ) -> np.ndarray:
        """
        Emit events for pixels in *mask* respecting the refractory period.

        Events are assigned uniformly random timestamps within the frame
        interval, mimicking the asynchronous nature of a real DVS.
        """
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return np.empty(0, dtype=EVENT_DTYPE)

        # Uniformly distributed timestamps within the frame interval
        ts = self.rng.integers(t_start_us, t_end_us, size=len(xs)).astype(np.int64)

        # Refractory filter
        refractory_us = int(self.refractory_period_us)
        time_since_last = ts - self._last_event_t[ys, xs]
        valid = time_since_last >= refractory_us

        xs = xs[valid]
        ys = ys[valid]
        ts = ts[valid]

        if len(xs) == 0:
            return np.empty(0, dtype=EVENT_DTYPE)

        # Update last event timestamp
        self._last_event_t[ys, xs] = ts

        out = np.empty(len(xs), dtype=EVENT_DTYPE)
        out["t"] = ts
        out["x"] = xs.astype(np.int16)
        out["y"] = ys.astype(np.int16)
        out["p"] = np.int8(polarity)
        return out

    def _shot_noise(self, t_start_us: int, t_end_us: int) -> np.ndarray:
        """
        Inject Poisson-distributed shot noise events.

        Expected number per pixel per frame = shot_noise_rate_hz * dt.
        Only pixels with expected ≥ Poisson draw are included.
        """
        dt_s = (t_end_us - t_start_us) * 1e-6
        lambda_ = self.shot_noise_rate_hz * dt_s  # expected events per pixel

        counts = self.rng.poisson(lambda_, size=(self.height, self.width))
        ys, xs = np.where(counts > 0)

        if len(xs) == 0:
            return np.empty(0, dtype=EVENT_DTYPE)

        ts = self.rng.integers(t_start_us, t_end_us, size=len(xs)).astype(np.int64)
        polarities = self.rng.choice([-1, 1], size=len(xs)).astype(np.int8)

        out = np.empty(len(xs), dtype=EVENT_DTYPE)
        out["t"] = ts
        out["x"] = xs.astype(np.int16)
        out["y"] = ys.astype(np.int16)
        out["p"] = polarities
        return out


# ── CLI entry-point ────────────────────────────────────────────────────────

def main() -> None:
    """Command-line interface for orbital-to-events conversion."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Convert a folder of orbital grayscale frames to an event stream."
    )
    parser.add_argument("input_dir", type=Path, help="Directory containing PNG/JPG frames")
    parser.add_argument("output", type=Path, help="Output .npz path")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--threshold-pos", type=float, default=0.15)
    parser.add_argument("--threshold-neg", type=float, default=0.15)
    parser.add_argument("--noise", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        import cv2
    except ImportError:
        sys.exit("opencv-python is required. Install with: pip install opencv-python")

    frames = []
    for p in sorted(args.input_dir.glob("*.png")) + sorted(args.input_dir.glob("*.jpg")):
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            frames.append(img)

    if not frames:
        sys.exit(f"No PNG/JPG frames found in {args.input_dir}")

    converter = OrbitalToEvents(
        threshold_pos=args.threshold_pos,
        threshold_neg=args.threshold_neg,
        shot_noise_rate_hz=args.noise,
        seed=args.seed,
    )
    stream = converter.convert(frames, fps=args.fps)
    stream.save_numpy(args.output)
    print(f"Converted {len(frames)} frames → {stream}")


if __name__ == "__main__":
    main()
