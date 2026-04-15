"""
Event stream visualization utilities.

Provides tools for rendering asynchronous neuromorphic event streams
as colored scatter plots, time-surface heatmaps, and animated sequences.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
from matplotlib.figure import Figure


# ── Event array dtype ──────────────────────────────────────────────────────
EVENT_DTYPE = np.dtype(
    [
        ("t", np.int64),   # timestamp in microseconds
        ("x", np.int16),   # pixel column
        ("y", np.int16),   # pixel row
        ("p", np.int8),    # polarity: +1 (ON) or -1 (OFF)
    ]
)

# Default colours
_POS_COLOR = "#00BFFF"   # deep sky blue  – ON events
_NEG_COLOR = "#FF4500"   # orange-red     – OFF events


class EventVisualizer:
    """
    Visualize neuromorphic event streams from orbital / astronomical cameras.

    Parameters
    ----------
    sensor_width : int
        Sensor width in pixels (default: 346 – DAVIS346).
    sensor_height : int
        Sensor height in pixels (default: 260).
    pos_color : str
        Matplotlib color string for ON (positive) events.
    neg_color : str
        Matplotlib color string for OFF (negative) events.
    dpi : int
        Figure DPI for saved images.
    """

    def __init__(
        self,
        sensor_width: int = 346,
        sensor_height: int = 260,
        pos_color: str = _POS_COLOR,
        neg_color: str = _NEG_COLOR,
        dpi: int = 150,
    ) -> None:
        self.width = sensor_width
        self.height = sensor_height
        self.pos_color = pos_color
        self.neg_color = neg_color
        self.dpi = dpi

    # ------------------------------------------------------------------
    # Primary visualizations
    # ------------------------------------------------------------------

    def scatter_plot(
        self,
        events: np.ndarray,
        *,
        title: str = "Event Stream",
        alpha: float = 0.6,
        marker_size: float = 1.5,
        ax: Optional[plt.Axes] = None,
        save_path: Optional[Union[str, Path]] = None,
    ) -> Figure:
        """
        Render events as a 2-D scatter plot on the sensor plane.

        Parameters
        ----------
        events : np.ndarray
            Structured array with fields ``t, x, y, p`` (see EVENT_DTYPE).
        title : str
            Figure title.
        alpha : float
            Point transparency (0–1).
        marker_size : float
            Scatter marker size.
        ax : plt.Axes | None
            Existing axes to draw on.  Creates a new figure if ``None``.
        save_path : str | Path | None
            If provided, save the figure to this path.

        Returns
        -------
        matplotlib.figure.Figure
        """
        create_fig = ax is None
        if create_fig:
            fig, ax = plt.subplots(figsize=(10, 7), dpi=self.dpi)
        else:
            fig = ax.get_figure()

        pos_mask = events["p"] > 0
        neg_mask = ~pos_mask

        ax.scatter(
            events["x"][pos_mask],
            events["y"][pos_mask],
            c=self.pos_color,
            s=marker_size,
            alpha=alpha,
            label=f"ON events  ({pos_mask.sum():,})",
            rasterized=True,
        )
        ax.scatter(
            events["x"][neg_mask],
            events["y"][neg_mask],
            c=self.neg_color,
            s=marker_size,
            alpha=alpha,
            label=f"OFF events ({neg_mask.sum():,})",
            rasterized=True,
        )

        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.invert_yaxis()
        ax.set_xlabel("X (pixels)", fontsize=11)
        ax.set_ylabel("Y (pixels)", fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend(loc="upper right", markerscale=5, fontsize=9)
        ax.set_facecolor("#0d1117")
        if create_fig:
            fig.patch.set_facecolor("#0d1117")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")

        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=self.dpi)

        return fig

    def time_surface(
        self,
        events: np.ndarray,
        *,
        tau_us: float = 5000.0,
        title: str = "Time Surface",
        colormap: str = "hot",
        ax: Optional[plt.Axes] = None,
        save_path: Optional[Union[str, Path]] = None,
    ) -> Figure:
        """
        Compute and render a polarity-aware time surface.

        The time surface :math:`T_{p}(x,y) = \\exp(-(t_{last} - t(x,y)) / \\tau)`
        decays exponentially with the time since the last event at each pixel.

        Parameters
        ----------
        events : np.ndarray
            Structured array with fields ``t, x, y, p``.
        tau_us : float
            Decay time constant in microseconds.
        title : str
            Figure title.
        colormap : str
            Matplotlib colormap name.
        ax : plt.Axes | None
            Existing axes to plot on.
        save_path : str | Path | None
            Optional output path for saving.

        Returns
        -------
        matplotlib.figure.Figure
        """
        surface = np.zeros((2, self.height, self.width), dtype=np.float32)
        t_last = np.full((2, self.height, self.width), -np.inf, dtype=np.float64)

        if len(events) == 0:
            pass
        else:
            t_max = float(events["t"].max())
            for ev in events:
                ch = 0 if ev["p"] > 0 else 1
                xi, yi = int(ev["x"]), int(ev["y"])
                if 0 <= xi < self.width and 0 <= yi < self.height:
                    t_last[ch, yi, xi] = float(ev["t"])

            valid = t_last > -np.inf
            surface[valid] = np.exp(-(t_max - t_last[valid]) / tau_us)

        create_fig = ax is None
        if create_fig:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=self.dpi)
        else:
            fig = ax.get_figure()
            axes = [ax, ax]

        for ch, (a, label) in enumerate(zip(axes, ["ON (positive)", "OFF (negative)"])):
            im = a.imshow(
                surface[ch],
                cmap=colormap,
                vmin=0,
                vmax=1,
                aspect="auto",
                origin="upper",
            )
            a.set_title(f"{title} – {label}", fontsize=11)
            a.set_xlabel("X (pixels)")
            a.set_ylabel("Y (pixels)")
            fig.colorbar(im, ax=a, label="Surface value")

        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=self.dpi)

        return fig

    def space_time_plot(
        self,
        events: np.ndarray,
        *,
        row: Optional[int] = None,
        col: Optional[int] = None,
        title: str = "Space-Time Event Plot",
        save_path: Optional[Union[str, Path]] = None,
    ) -> Figure:
        """
        3-D space-time plot of events (x, y, t).

        Parameters
        ----------
        events : np.ndarray
            Structured event array.
        row : int | None
            If given, only show events at this pixel row.
        col : int | None
            If given, only show events at this pixel column.
        title : str
            Figure title.
        save_path : str | Path | None
            Save path.

        Returns
        -------
        matplotlib.figure.Figure
        """
        fig = plt.figure(figsize=(12, 8), dpi=self.dpi)
        ax = fig.add_subplot(111, projection="3d")

        mask = np.ones(len(events), dtype=bool)
        if row is not None:
            mask &= events["y"] == row
        if col is not None:
            mask &= events["x"] == col

        ev = events[mask]
        pos_mask = ev["p"] > 0

        t_us = ev["t"] * 1e-3  # convert to ms for readability

        ax.scatter(
            ev["x"][pos_mask],
            t_us[pos_mask],
            ev["y"][pos_mask],
            c=self.pos_color,
            s=1.0,
            alpha=0.5,
            label="ON",
        )
        ax.scatter(
            ev["x"][~pos_mask],
            t_us[~pos_mask],
            ev["y"][~pos_mask],
            c=self.neg_color,
            s=1.0,
            alpha=0.5,
            label="OFF",
        )

        ax.set_xlabel("X (px)")
        ax.set_ylabel("Time (ms)")
        ax.set_zlabel("Y (px)")
        ax.set_title(title, fontsize=12)
        ax.legend()

        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=self.dpi)

        return fig

    def comparison_plot(
        self,
        original_frame: np.ndarray,
        events: np.ndarray,
        *,
        title: str = "Frame vs. Event Stream",
        save_path: Optional[Union[str, Path]] = None,
    ) -> Figure:
        """
        Side-by-side comparison of the original grayscale frame and the
        resulting event stream.

        Parameters
        ----------
        original_frame : np.ndarray
            2-D grayscale image array (H x W), values in [0, 255].
        events : np.ndarray
            Structured event array for the frame interval.
        title : str
            Overall figure title.
        save_path : str | Path | None
            Save path.

        Returns
        -------
        matplotlib.figure.Figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=self.dpi)
        fig.suptitle(title, fontsize=14, fontweight="bold", color="white")
        fig.patch.set_facecolor("#0d1117")

        # ── left: original frame ────────────────────────────────────────
        axes[0].imshow(original_frame, cmap="gray", vmin=0, vmax=255, origin="upper")
        axes[0].set_title("Original Frame", color="white", fontsize=11)
        axes[0].axis("off")

        # ── right: event scatter ────────────────────────────────────────
        axes[1].set_facecolor("#0d1117")
        if len(events) > 0:
            pos_mask = events["p"] > 0
            axes[1].scatter(
                events["x"][pos_mask],
                events["y"][pos_mask],
                c=self.pos_color,
                s=1.2,
                alpha=0.7,
                label=f"ON ({pos_mask.sum():,})",
                rasterized=True,
            )
            axes[1].scatter(
                events["x"][~pos_mask],
                events["y"][~pos_mask],
                c=self.neg_color,
                s=1.2,
                alpha=0.7,
                label=f"OFF ({(~pos_mask).sum():,})",
                rasterized=True,
            )
            axes[1].legend(loc="upper right", markerscale=5, fontsize=9, labelcolor="white",
                           framealpha=0.3)

        axes[1].set_xlim(0, self.width)
        axes[1].set_ylim(0, self.height)
        axes[1].invert_yaxis()
        axes[1].set_title("Event Stream", color="white", fontsize=11)
        axes[1].tick_params(colors="white")
        axes[1].set_xlabel("X (pixels)", color="white")
        axes[1].set_ylabel("Y (pixels)", color="white")
        for spine in axes[1].spines.values():
            spine.set_edgecolor("#444")

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=self.dpi, facecolor=fig.get_facecolor())

        return fig

    def event_rate_plot(
        self,
        events: np.ndarray,
        *,
        bin_ms: float = 1.0,
        title: str = "Event Rate over Time",
        save_path: Optional[Union[str, Path]] = None,
    ) -> Figure:
        """
        Plot total ON/OFF event rate as a function of time.

        Parameters
        ----------
        events : np.ndarray
            Structured event array.
        bin_ms : float
            Histogram bin width in milliseconds.
        title : str
            Figure title.
        save_path : str | Path | None
            Optional save path.

        Returns
        -------
        matplotlib.figure.Figure
        """
        fig, ax = plt.subplots(figsize=(12, 4), dpi=self.dpi)

        if len(events) == 0:
            ax.set_title(title)
            return fig

        t_ms = events["t"] * 1e-3
        bins = np.arange(t_ms.min(), t_ms.max() + bin_ms, bin_ms)

        pos_mask = events["p"] > 0
        ax.hist(t_ms[pos_mask], bins=bins, color=self.pos_color, alpha=0.7,
                label="ON events", histtype="stepfilled")
        ax.hist(t_ms[~pos_mask], bins=bins, color=self.neg_color, alpha=0.7,
                label="OFF events", histtype="stepfilled")

        ax.set_xlabel("Time (ms)", fontsize=11)
        ax.set_ylabel(f"Events per {bin_ms:.1f} ms bin", fontsize=11)
        ax.set_title(title, fontsize=13)
        ax.legend()
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=self.dpi)

        return fig


# ── Convenience wrapper ────────────────────────────────────────────────────

def plot_event_stream(
    events: np.ndarray,
    sensor_width: int = 346,
    sensor_height: int = 260,
    *,
    mode: str = "scatter",
    save_path: Optional[Union[str, Path]] = None,
    **kwargs,
) -> Figure:
    """
    One-call convenience wrapper around :class:`EventVisualizer`.

    Parameters
    ----------
    events : np.ndarray
        Structured event array with fields ``t, x, y, p``.
    sensor_width : int
        Sensor width in pixels.
    sensor_height : int
        Sensor height in pixels.
    mode : str
        Visualization mode: ``"scatter"`` | ``"time_surface"`` |
        ``"space_time"`` | ``"rate"``.
    save_path : str | Path | None
        Optional output path.
    **kwargs
        Additional keyword arguments passed to the selected plot method.

    Returns
    -------
    matplotlib.figure.Figure
    """
    viz = EventVisualizer(sensor_width=sensor_width, sensor_height=sensor_height)
    dispatch: Dict[str, any] = {
        "scatter": viz.scatter_plot,
        "time_surface": viz.time_surface,
        "space_time": viz.space_time_plot,
        "rate": viz.event_rate_plot,
    }
    if mode not in dispatch:
        raise ValueError(f"Unknown mode '{mode}'. Choose from {list(dispatch)}")

    return dispatch[mode](events, save_path=save_path, **kwargs)
