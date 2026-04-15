"""
Star Centroiding Algorithms for Event-Based Star Trackers.

Implements intensity-weighted centroiding, iterative Gaussian centroiding,
and threshold-based centroiding suitable for sparse event-camera star maps.

These algorithms are used downstream by the EventStarTracker to produce
sub-pixel star position estimates from event cluster data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy import ndimage

logger = logging.getLogger(__name__)


@dataclass
class CentroidResult:
    """Result of a centroiding computation."""
    x: float            # Sub-pixel column position
    y: float            # Sub-pixel row position
    flux: float         # Total integrated intensity (proxy for magnitude)
    fwhm: float         # Full-width half-maximum estimate (pixels)
    snr: float          # Signal-to-noise ratio
    converged: bool     # Whether iterative method converged
    n_pixels: int       # Number of pixels used in centroid computation


class Centroider:
    """
    Star centroiding algorithms for sparse event-camera star maps.

    Parameters
    ----------
    method : str
        Centroiding method: ``"weighted"`` (intensity-weighted),
        ``"iterative"`` (iterative Gaussian), or ``"threshold"``
        (threshold-based, fastest).
    box_half_size : int
        Half-size of the centroiding aperture in pixels.
    max_iterations : int
        Maximum iterations for the iterative method.
    tolerance : float
        Convergence tolerance for iterative method (pixels).
    min_snr : float
        Minimum signal-to-noise ratio; results below this are flagged.
    sigma_threshold : float
        Background sigma clipping threshold.

    Examples
    --------
    >>> centroider = Centroider(method="weighted", box_half_size=5)
    >>> image = np.zeros((50, 50), dtype=np.float32)
    >>> image[25, 25] = 100.0  # star at (25, 25)
    >>> result = centroider.centroid(image, 25.0, 25.0)
    >>> print(f"Star at ({result.x:.3f}, {result.y:.3f})")
    """

    def __init__(
        self,
        method: str = "weighted",
        box_half_size: int = 5,
        max_iterations: int = 50,
        tolerance: float = 0.001,
        min_snr: float = 3.0,
        sigma_threshold: float = 3.0,
    ) -> None:
        if method not in {"weighted", "iterative", "threshold"}:
            raise ValueError(f"Unknown method '{method}'")
        self.method = method
        self.box_half_size = box_half_size
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.min_snr = min_snr
        self.sigma_threshold = sigma_threshold

    def centroid(
        self,
        image: np.ndarray,
        x_guess: float,
        y_guess: float,
    ) -> CentroidResult:
        """
        Compute sub-pixel centroid near an initial guess position.

        Parameters
        ----------
        image : np.ndarray  shape (H, W), dtype float
            Accumulated event frame or reconstructed intensity image.
        x_guess : float
            Initial column estimate.
        y_guess : float
            Initial row estimate.

        Returns
        -------
        CentroidResult
        """
        if self.method == "weighted":
            return self._weighted_centroid(image, x_guess, y_guess)
        elif self.method == "iterative":
            return self._iterative_centroid(image, x_guess, y_guess)
        else:
            return self._threshold_centroid(image, x_guess, y_guess)

    def centroid_batch(
        self,
        image: np.ndarray,
        positions: List[Tuple[float, float]],
    ) -> List[CentroidResult]:
        """
        Centroid multiple star candidates in a single image.

        Parameters
        ----------
        image : np.ndarray  shape (H, W)
        positions : list of (x, y) tuples

        Returns
        -------
        list of CentroidResult
        """
        return [self.centroid(image, x, y) for x, y in positions]

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _extract_box(
        self,
        image: np.ndarray,
        x_center: float,
        y_center: float,
    ) -> Tuple[np.ndarray, int, int]:
        """
        Extract a sub-image around (x_center, y_center).

        Returns
        -------
        sub : np.ndarray  sub-image (possibly smaller near borders)
        x0  : int  left column of sub in original image
        y0  : int  top row of sub in original image
        """
        H, W = image.shape
        h = self.box_half_size
        xi, yi = int(round(x_center)), int(round(y_center))

        x0 = max(0, xi - h)
        x1 = min(W, xi + h + 1)
        y0 = max(0, yi - h)
        y1 = min(H, yi + h + 1)

        return image[y0:y1, x0:x1], x0, y0

    def _estimate_background(self, sub: np.ndarray) -> Tuple[float, float]:
        """Sigma-clipped background mean and std for a sub-image."""
        flat = sub.ravel()
        med = float(np.median(flat))
        sigma = float(np.std(flat))
        for _ in range(3):
            mask = np.abs(flat - med) < self.sigma_threshold * sigma
            if mask.sum() < 3:
                break
            med = float(np.mean(flat[mask]))
            sigma = float(np.std(flat[mask]))
        return med, max(sigma, 1e-9)

    def _weighted_centroid(
        self,
        image: np.ndarray,
        x_guess: float,
        y_guess: float,
    ) -> CentroidResult:
        """Intensity-weighted (first moment) centroiding."""
        sub, x0, y0 = self._extract_box(image, x_guess, y_guess)
        bg, bg_std = self._estimate_background(sub)

        weights = (sub - bg).clip(0, None)
        total = weights.sum()
        snr = float(total / (bg_std * np.sqrt(weights.size)))

        if total <= 0:
            return CentroidResult(x_guess, y_guess, 0.0, 0.0, snr, False, 0)

        rows = np.arange(sub.shape[0], dtype=np.float64)
        cols = np.arange(sub.shape[1], dtype=np.float64)
        cx = float(np.dot(weights.sum(axis=0), cols) / total) + x0
        cy = float(np.dot(weights.sum(axis=1), rows) / total) + y0

        fwhm = self._estimate_fwhm(sub, bg)
        n_pixels = int((weights > 0).sum())

        return CentroidResult(cx, cy, float(total), fwhm, snr, True, n_pixels)

    def _iterative_centroid(
        self,
        image: np.ndarray,
        x_guess: float,
        y_guess: float,
    ) -> CentroidResult:
        """
        Iterative refinement of the weighted centroid.

        Re-centres the aperture each iteration until convergence.
        """
        cx, cy = x_guess, y_guess

        for iteration in range(self.max_iterations):
            result = self._weighted_centroid(image, cx, cy)
            dx = result.x - cx
            dy = result.y - cy
            cx, cy = result.x, result.y

            if np.sqrt(dx ** 2 + dy ** 2) < self.tolerance:
                result.converged = True
                return result

        result.converged = False
        return result

    def _threshold_centroid(
        self,
        image: np.ndarray,
        x_guess: float,
        y_guess: float,
    ) -> CentroidResult:
        """
        Threshold-based centroiding: only pixels above bg + n*sigma are used.

        Fastest method; suitable for high-SNR stars.
        """
        sub, x0, y0 = self._extract_box(image, x_guess, y_guess)
        bg, bg_std = self._estimate_background(sub)
        threshold = bg + self.sigma_threshold * bg_std

        mask = sub >= threshold
        if not mask.any():
            return CentroidResult(x_guess, y_guess, 0.0, 0.0, 0.0, False, 0)

        weights = (sub - bg) * mask
        total = weights.sum()
        snr = float(total / (bg_std * np.sqrt(mask.sum())))

        rows = np.arange(sub.shape[0], dtype=np.float64)
        cols = np.arange(sub.shape[1], dtype=np.float64)
        cx = float(np.dot(weights.sum(axis=0), cols) / total) + x0
        cy = float(np.dot(weights.sum(axis=1), rows) / total) + y0

        fwhm = self._estimate_fwhm(sub * mask, bg)
        n_pixels = int(mask.sum())

        return CentroidResult(cx, cy, float(total), fwhm, snr, True, n_pixels)

    @staticmethod
    def _estimate_fwhm(sub: np.ndarray, bg: float) -> float:
        """
        Estimate FWHM from the marginal profiles of a star sub-image.

        Uses the half-maximum crossing of the 1-D column and row profiles.
        """
        signal = (sub - bg).clip(0, None)
        if signal.max() <= 0:
            return 0.0

        # Column profile
        profile_x = signal.sum(axis=0)
        profile_y = signal.sum(axis=1)

        def _profile_fwhm(profile: np.ndarray) -> float:
            max_val = profile.max()
            if max_val <= 0:
                return 0.0
            half_max = max_val / 2.0
            above = profile >= half_max
            if not above.any():
                return 0.0
            indices = np.where(above)[0]
            return float(indices[-1] - indices[0] + 1)

        fx = _profile_fwhm(profile_x)
        fy = _profile_fwhm(profile_y)
        return float((fx + fy) / 2.0)


def detect_star_candidates(
    accumulation_frame: np.ndarray,
    *,
    min_sigma: float = 3.0,
    min_cluster_size: int = 3,
    max_cluster_size: int = 50,
) -> List[Tuple[float, float]]:
    """
    Detect star candidate positions from an event accumulation frame.

    Uses connected-component labelling to find clusters of events
    and returns their approximate centre positions.

    Parameters
    ----------
    accumulation_frame : np.ndarray  shape (H, W)
        Summed event frame (e.g. ON events only over a time window).
    min_sigma : float
        Threshold multiplier above background std for detection.
    min_cluster_size : int
        Minimum cluster area in pixels.
    max_cluster_size : int
        Maximum cluster area in pixels (reject large cosmic ray tracks).

    Returns
    -------
    list of (x, y) float tuples
        Approximate pixel positions of candidate stars.
    """
    bg = float(np.median(accumulation_frame))
    std = float(np.std(accumulation_frame))
    if std < 1e-9:
        return []

    binary = accumulation_frame > (bg + min_sigma * std)
    labeled, n_labels = ndimage.label(binary)

    candidates: List[Tuple[float, float]] = []
    for label_id in range(1, n_labels + 1):
        mask = labeled == label_id
        size = int(mask.sum())
        if size < min_cluster_size or size > max_cluster_size:
            continue
        cy, cx = ndimage.center_of_mass(accumulation_frame * mask)
        candidates.append((float(cx), float(cy)))

    return candidates
