"""
Detection Metrics and Benchmark Evaluation.

Implements standard object detection metrics (AP, mAP, F1, precision-recall)
plus SDA-specific metrics: detection rate, false alarm rate, latency.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """
    Compute IoU between two boxes in [cx, cy, w, h] format.

    Parameters
    ----------
    box_a, box_b : np.ndarray  shape (4,)

    Returns
    -------
    float  IoU in [0, 1]
    """
    ax1 = box_a[0] - box_a[2] / 2
    ay1 = box_a[1] - box_a[3] / 2
    ax2 = box_a[0] + box_a[2] / 2
    ay2 = box_a[1] + box_a[3] / 2

    bx1 = box_b[0] - box_b[2] / 2
    by1 = box_b[1] - box_b[3] / 2
    bx2 = box_b[0] + box_b[2] / 2
    by2 = box_b[1] + box_b[3] / 2

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    area_a = box_a[2] * box_a[3]
    area_b = box_b[2] * box_b[3]
    union_area = area_a + area_b - inter_area

    return float(inter_area / (union_area + 1e-9))


def compute_ap(
    precision: np.ndarray,
    recall: np.ndarray,
    method: str = "interp",
) -> float:
    """
    Compute Average Precision from a precision-recall curve.

    Parameters
    ----------
    precision, recall : np.ndarray  shape (N,)
    method : str
        ``"interp"`` (11-point interpolation, VOC 2007) or
        ``"area"`` (exact area under curve).

    Returns
    -------
    float  AP in [0, 1]
    """
    if method == "interp":
        ap = 0.0
        for thresh in np.linspace(0, 1, 11):
            prec_at = precision[recall >= thresh]
            ap += prec_at.max() if len(prec_at) else 0.0
        return ap / 11.0
    else:  # area under curve
        # Append sentinel values
        r = np.concatenate(([0.0], recall, [1.0]))
        p = np.concatenate(([1.0], precision, [0.0]))
        # Ensure monotonically decreasing precision
        for i in range(len(p) - 2, -1, -1):
            p[i] = max(p[i], p[i + 1])
        return float(np.sum((r[1:] - r[:-1]) * p[1:]))


@dataclass
class FrameResult:
    """Evaluation result for a single frame."""
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    latency_ms: float = 0.0
    n_gt: int = 0
    n_pred: int = 0


@dataclass
class BenchmarkResults:
    """Aggregate benchmark results across all frames / scenarios."""
    # Detection performance
    mean_ap: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    detection_rate: float = 0.0
    false_alarm_rate: float = 0.0

    # Latency
    mean_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0

    # Energy (proxy)
    mean_spike_rate: float = 0.0
    synaptic_ops: float = 0.0

    # Per-orbit breakdown
    per_orbit: Dict[str, Dict] = field(default_factory=dict)

    def print_table(self) -> None:
        """Print a formatted results table."""
        print("=" * 60)
        print("  NEUROMORPHIC SDA BENCHMARK RESULTS")
        print("=" * 60)
        print(f"  mAP@0.5            : {self.mean_ap:.4f}")
        print(f"  Precision          : {self.precision:.4f}")
        print(f"  Recall             : {self.recall:.4f}")
        print(f"  F1 Score           : {self.f1:.4f}")
        print(f"  Detection Rate     : {self.detection_rate * 100:.1f}%")
        print(f"  False Alarm Rate   : {self.false_alarm_rate * 100:.2f}%")
        print("-" * 60)
        print(f"  Mean Latency       : {self.mean_latency_ms:.2f} ms")
        print(f"  P95  Latency       : {self.p95_latency_ms:.2f} ms")
        print(f"  P99  Latency       : {self.p99_latency_ms:.2f} ms")
        print("-" * 60)
        print(f"  Mean Spike Rate    : {self.mean_spike_rate:.4f}")
        print(f"  Synaptic Ops (est) : {self.synaptic_ops:.2e}")
        if self.per_orbit:
            print("-" * 60)
            print("  Per-Orbit Breakdown:")
            for orbit, metrics in self.per_orbit.items():
                dr = metrics.get("detection_rate", 0.0)
                far = metrics.get("false_alarm_rate", 0.0)
                print(f"    {orbit:10s} | DR={dr*100:.1f}%  FAR={far*100:.2f}%")
        print("=" * 60)


class DetectionMetrics:
    """
    Compute detection metrics from a list of prediction / ground-truth pairs.

    Parameters
    ----------
    iou_threshold : float
        IoU threshold for a detection to count as a true positive.
    num_classes : int
        Number of object classes (default: 1, satellite).

    Examples
    --------
    >>> metrics = DetectionMetrics(iou_threshold=0.5)
    >>> metrics.update(pred_boxes, pred_scores, gt_boxes)
    >>> results = metrics.compute()
    >>> results.print_table()
    """

    def __init__(
        self,
        iou_threshold: float = 0.5,
        num_classes: int = 1,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.num_classes = num_classes

        self._all_scores: List[float] = []
        self._all_tp: List[int] = []
        self._all_fp: List[int] = []
        self._n_gt: int = 0
        self._latencies_ms: List[float] = []
        self._spike_rates: List[float] = []

    def update(
        self,
        pred_boxes: np.ndarray,
        pred_scores: np.ndarray,
        gt_boxes: np.ndarray,
        *,
        latency_ms: float = 0.0,
        spike_rate: float = 0.0,
    ) -> FrameResult:
        """
        Update metric state with one frame's predictions and ground truth.

        Parameters
        ----------
        pred_boxes : np.ndarray  shape (N_pred, 4)  [cx, cy, w, h]
        pred_scores : np.ndarray  shape (N_pred,)
        gt_boxes : np.ndarray  shape (N_gt, 4)  [cx, cy, w, h]
        latency_ms : float
            Inference latency for this frame.
        spike_rate : float
            Mean spike rate for energy estimation.

        Returns
        -------
        FrameResult
        """
        self._latencies_ms.append(latency_ms)
        self._spike_rates.append(spike_rate)
        self._n_gt += len(gt_boxes)

        if len(pred_boxes) == 0:
            result = FrameResult(
                false_negatives=len(gt_boxes),
                n_gt=len(gt_boxes),
                latency_ms=latency_ms,
            )
            return result

        # Sort predictions by score (descending)
        order = np.argsort(pred_scores)[::-1]
        pred_boxes = pred_boxes[order]
        pred_scores = pred_scores[order]

        matched_gt: set = set()
        tp_list: List[int] = []
        fp_list: List[int] = []

        for i, pb in enumerate(pred_boxes):
            best_iou = 0.0
            best_gt_idx = -1

            for j, gb in enumerate(gt_boxes):
                if j in matched_gt:
                    continue
                iou = _iou(pb, gb)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j

            if best_iou >= self.iou_threshold and best_gt_idx >= 0:
                tp_list.append(1)
                fp_list.append(0)
                matched_gt.add(best_gt_idx)
            else:
                tp_list.append(0)
                fp_list.append(1)

        self._all_scores.extend(pred_scores.tolist())
        self._all_tp.extend(tp_list)
        self._all_fp.extend(fp_list)

        tp_count = sum(tp_list)
        fp_count = sum(fp_list)
        fn_count = len(gt_boxes) - tp_count

        return FrameResult(
            true_positives=tp_count,
            false_positives=fp_count,
            false_negatives=fn_count,
            latency_ms=latency_ms,
            n_gt=len(gt_boxes),
            n_pred=len(pred_boxes),
        )

    def compute(self) -> BenchmarkResults:
        """
        Aggregate all accumulated updates into final metrics.

        Returns
        -------
        BenchmarkResults
        """
        if not self._all_scores:
            return BenchmarkResults()

        scores = np.array(self._all_scores)
        tp_arr = np.array(self._all_tp)
        fp_arr = np.array(self._all_fp)

        order = np.argsort(scores)[::-1]
        tp_arr = tp_arr[order]
        fp_arr = fp_arr[order]

        tp_cum = np.cumsum(tp_arr)
        fp_cum = np.cumsum(fp_arr)

        recall = tp_cum / (self._n_gt + 1e-9)
        precision = tp_cum / (tp_cum + fp_cum + 1e-9)

        ap = compute_ap(precision, recall)

        total_tp = int(tp_arr.sum())
        total_fp = int(fp_arr.sum())
        total_fn = self._n_gt - total_tp

        prec = total_tp / (total_tp + total_fp + 1e-9)
        rec = total_tp / (total_tp + total_fn + 1e-9)
        f1 = 2 * prec * rec / (prec + rec + 1e-9)

        latencies = np.array(self._latencies_ms)
        spike_rates = np.array(self._spike_rates)

        return BenchmarkResults(
            mean_ap=ap,
            precision=float(prec),
            recall=float(rec),
            f1=float(f1),
            detection_rate=float(rec),
            false_alarm_rate=float(total_fp / (total_fp + total_tp + 1e-9)),
            mean_latency_ms=float(latencies.mean()),
            p95_latency_ms=float(np.percentile(latencies, 95)),
            p99_latency_ms=float(np.percentile(latencies, 99)),
            mean_spike_rate=float(spike_rates.mean()),
            synaptic_ops=float(spike_rates.mean() * 1e8),  # rough proxy
        )

    def reset(self) -> None:
        """Clear all accumulated state."""
        self._all_scores.clear()
        self._all_tp.clear()
        self._all_fp.clear()
        self._n_gt = 0
        self._latencies_ms.clear()
        self._spike_rates.clear()
