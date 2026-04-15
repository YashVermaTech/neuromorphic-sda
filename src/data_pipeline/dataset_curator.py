"""
Event Camera Dataset Curation Tools.

Provides utilities for building, filtering, splitting, and exporting
event-camera datasets for space domain awareness tasks.  Works with
EventStream objects produced by OrbitalToEvents and can export to
standard formats (NPZ, HDF5, CSV).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SampleMetadata:
    """Metadata for a single dataset sample."""
    sample_id: str
    source_file: str
    duration_us: int
    n_events: int
    sensor_width: int
    sensor_height: int
    labels: List[Dict] = field(default_factory=list)  # bounding boxes / class labels
    split: str = "train"   # "train" | "val" | "test"
    tags: List[str] = field(default_factory=list)


class EventDatasetCurator:
    """
    Curate an event-stream dataset for satellite detection training.

    Parameters
    ----------
    output_dir : str | Path
        Root directory for the curated dataset.
    sensor_width : int
        Sensor width expected across all samples.
    sensor_height : int
        Sensor height expected across all samples.
    window_duration_us : int
        Duration of each time window used to slice streams into samples.
    stride_us : int | None
        Stride between consecutive windows.  Defaults to window_duration_us
        (non-overlapping).
    min_events_per_window : int
        Discard windows with fewer than this many events.
    seed : int
        Random seed for reproducible splits.

    Examples
    --------
    >>> curator = EventDatasetCurator("data/curated", window_duration_us=50_000)
    >>> curator.add_stream(stream, labels=[{"class": "satellite", "bbox": [10, 20, 50, 60]}])
    >>> curator.split(train=0.7, val=0.15, test=0.15)
    >>> curator.export()
    """

    def __init__(
        self,
        output_dir: Union[str, Path] = "data/curated",
        sensor_width: int = 346,
        sensor_height: int = 260,
        window_duration_us: int = 50_000,
        stride_us: Optional[int] = None,
        min_events_per_window: int = 100,
        seed: int = 42,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.sensor_width = sensor_width
        self.sensor_height = sensor_height
        self.window_duration_us = window_duration_us
        self.stride_us = stride_us if stride_us is not None else window_duration_us
        self.min_events_per_window = min_events_per_window
        self.rng = np.random.default_rng(seed)

        self._samples: List[Tuple[np.ndarray, SampleMetadata]] = []
        self._counter: int = 0

    # ------------------------------------------------------------------
    # Adding samples
    # ------------------------------------------------------------------

    def add_stream(
        self,
        stream,  # EventStream (imported lazily to avoid circular deps)
        *,
        source_name: str = "unknown",
        labels: Optional[List[Dict]] = None,
        tags: Optional[List[str]] = None,
    ) -> int:
        """
        Slice an EventStream into fixed windows and add them to the dataset.

        Parameters
        ----------
        stream : EventStream
            Input event stream.
        source_name : str
            Human-readable identifier for the source file.
        labels : list of dict | None
            Ground-truth annotations.  Each dict should contain at least
            ``{"class": str, "bbox": [x, y, w, h]}``.  Applied to all
            windows from this stream.
        tags : list of str | None
            Optional tags (e.g. "leo", "geo", "night").

        Returns
        -------
        int
            Number of windows added.
        """
        added = 0
        t_start = int(stream.events["t"].min()) if len(stream.events) else 0
        t_end = t_start + stream.duration_us

        for ws in range(t_start, t_end - self.window_duration_us + 1, self.stride_us):
            we = ws + self.window_duration_us
            window_events = stream.window(ws, we)

            if len(window_events) < self.min_events_per_window:
                continue

            sample_id = f"sample_{self._counter:06d}"
            meta = SampleMetadata(
                sample_id=sample_id,
                source_file=source_name,
                duration_us=self.window_duration_us,
                n_events=len(window_events),
                sensor_width=self.sensor_width,
                sensor_height=self.sensor_height,
                labels=labels or [],
                tags=tags or [],
            )
            self._samples.append((window_events, meta))
            self._counter += 1
            added += 1

        logger.debug("Added %d windows from '%s'", added, source_name)
        return added

    def add_raw_events(
        self,
        events: np.ndarray,
        metadata: SampleMetadata,
    ) -> None:
        """
        Directly add a pre-sliced event array with metadata.

        Parameters
        ----------
        events : np.ndarray
            Structured event array (EVENT_DTYPE).
        metadata : SampleMetadata
        """
        self._samples.append((events, metadata))
        self._counter += 1

    # ------------------------------------------------------------------
    # Splitting
    # ------------------------------------------------------------------

    def split(
        self,
        train: float = 0.7,
        val: float = 0.15,
        test: float = 0.15,
    ) -> Dict[str, int]:
        """
        Randomly assign samples to train / val / test splits.

        Parameters
        ----------
        train, val, test : float
            Fractions that must sum to 1.0.

        Returns
        -------
        dict
            Count per split.
        """
        assert abs(train + val + test - 1.0) < 1e-6, "Split fractions must sum to 1"
        n = len(self._samples)
        indices = self.rng.permutation(n)
        n_train = int(n * train)
        n_val = int(n * val)

        for i, idx in enumerate(indices):
            _, meta = self._samples[idx]
            if i < n_train:
                meta.split = "train"
            elif i < n_train + n_val:
                meta.split = "val"
            else:
                meta.split = "test"

        counts = {
            "train": n_train,
            "val": n_val,
            "test": n - n_train - n_val,
        }
        logger.info("Split: train=%d val=%d test=%d", *counts.values())
        return counts

    # ------------------------------------------------------------------
    # Exporting
    # ------------------------------------------------------------------

    def export(
        self,
        format: str = "numpy",
        overwrite: bool = False,
    ) -> Path:
        """
        Export dataset to disk.

        Parameters
        ----------
        format : str
            ``"numpy"`` (default) or ``"hdf5"``.
        overwrite : bool
            If False, raises if output directory already exists.

        Returns
        -------
        Path
            Path to the output directory.
        """
        if self.output_dir.exists() and not overwrite:
            raise FileExistsError(
                f"{self.output_dir} already exists.  Use overwrite=True."
            )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if format == "numpy":
            return self._export_numpy()
        elif format == "hdf5":
            return self._export_hdf5()
        else:
            raise ValueError(f"Unknown format '{format}'. Use 'numpy' or 'hdf5'.")

    def _export_numpy(self) -> Path:
        """Save each sample as a separate .npz + a manifest JSON."""
        manifest = []
        for events, meta in self._samples:
            out_path = self.output_dir / meta.split / f"{meta.sample_id}.npz"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                out_path,
                t=events["t"],
                x=events["x"],
                y=events["y"],
                p=events["p"],
            )
            manifest.append({
                "sample_id": meta.sample_id,
                "path": str(out_path.relative_to(self.output_dir)),
                "split": meta.split,
                "n_events": meta.n_events,
                "duration_us": meta.duration_us,
                "labels": meta.labels,
                "tags": meta.tags,
            })

        manifest_path = self.output_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        logger.info("Exported %d samples (numpy) to %s", len(self._samples), self.output_dir)
        return self.output_dir

    def _export_hdf5(self) -> Path:
        """Save all samples into a single HDF5 file."""
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py is required for HDF5 export: pip install h5py")

        h5_path = self.output_dir / "dataset.h5"
        with h5py.File(h5_path, "w") as f:
            for events, meta in self._samples:
                grp = f.create_group(meta.sample_id)
                grp.create_dataset("t", data=events["t"], compression="gzip")
                grp.create_dataset("x", data=events["x"], compression="gzip")
                grp.create_dataset("y", data=events["y"], compression="gzip")
                grp.create_dataset("p", data=events["p"], compression="gzip")
                grp.attrs["split"] = meta.split
                grp.attrs["n_events"] = meta.n_events
                grp.attrs["duration_us"] = meta.duration_us
                grp.attrs["labels"] = json.dumps(meta.labels)

        logger.info("Exported %d samples (HDF5) to %s", len(self._samples), h5_path)
        return self.output_dir

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def statistics(self) -> Dict:
        """Return dataset statistics."""
        if not self._samples:
            return {}

        all_counts = [m.n_events for _, m in self._samples]
        splits = {s: sum(1 for _, m in self._samples if m.split == s)
                  for s in ["train", "val", "test"]}

        return {
            "total_samples": len(self._samples),
            "splits": splits,
            "events_mean": float(np.mean(all_counts)),
            "events_std": float(np.std(all_counts)),
            "events_min": int(np.min(all_counts)),
            "events_max": int(np.max(all_counts)),
            "total_events": int(np.sum(all_counts)),
        }

    def __len__(self) -> int:
        return len(self._samples)

    def __repr__(self) -> str:
        stats = self.statistics()
        return (
            f"EventDatasetCurator(samples={len(self)}, "
            f"splits={stats.get('splits', {})}, "
            f"output_dir={self.output_dir})"
        )
