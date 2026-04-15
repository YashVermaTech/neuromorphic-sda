"""
Spiking Neural Network Satellite Detector.

Implements a deep SNN for satellite detection in event streams using
Leaky Integrate-and-Fire (LIF) neurons from snnTorch.  Supports both
offline (batch) and online (streaming) inference.

Architecture
------------
  Input : event frame tensor (B, T, 2, H, W)
            B = batch, T = time steps, 2 = ON/OFF channels
  Backbone: 4× Convolutional SNN layers (Conv → BatchNorm → LIF)
  Neck   : Feature Pyramid Network (2 scales)
  Head   : Classification + Bounding-box regression per scale

Spike regularisation loss encourages sparse firing (energy efficiency).

References
----------
Eshraghian, J. K., et al. (2023). Training Spiking Neural Networks Using
  Lessons From Deep Learning. Proceedings of the IEEE, 111(9), 1016-1054.
  https://arxiv.org/abs/2109.12894
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Graceful degradation if snnTorch is not installed
try:
    import snntorch as snn
    from snntorch import surrogate

    _SNNTORCH_AVAILABLE = True
except ImportError:
    _SNNTORCH_AVAILABLE = False
    logger.warning(
        "snnTorch not found.  Install with: pip install snntorch  "
        "Falling back to surrogate LIF implementation."
    )


# ── LIF neuron (fallback if snnTorch unavailable) ─────────────────────────


class _LIFNeuron(nn.Module):
    """
    Minimal Leaky Integrate-and-Fire neuron module.

    Used as a fallback when snnTorch is not available.

    Parameters
    ----------
    beta : float
        Membrane potential decay factor (0 < beta < 1).
    threshold : float
        Firing threshold.
    reset_mechanism : str
        ``"subtract"`` (soft reset) or ``"zero"`` (hard reset).
    """

    def __init__(
        self,
        beta: float = 0.9,
        threshold: float = 1.0,
        reset_mechanism: str = "subtract",
    ) -> None:
        super().__init__()
        self.beta = beta
        self.threshold = threshold
        self.reset_mechanism = reset_mechanism

    def forward(
        self, input_current: torch.Tensor, mem: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        input_current : torch.Tensor  shape (B, C, H, W)
        mem : torch.Tensor  shape (B, C, H, W)  membrane potential

        Returns
        -------
        spk : torch.Tensor  binary spike tensor
        mem : torch.Tensor  updated membrane potential
        """
        mem = self.beta * mem + input_current
        spk = (mem >= self.threshold).float()
        if self.reset_mechanism == "subtract":
            mem = mem - spk * self.threshold
        else:
            mem = mem * (1.0 - spk)
        return spk, mem


def _make_lif(beta: float = 0.9, threshold: float = 1.0) -> nn.Module:
    """Return a LIF neuron – snnTorch if available, else fallback."""
    if _SNNTORCH_AVAILABLE:
        spike_grad = surrogate.fast_sigmoid(slope=25)
        return snn.Leaky(beta=beta, threshold=threshold,
                         spike_grad=spike_grad, init_hidden=False)
    return _LIFNeuron(beta=beta, threshold=threshold)


# ── Building blocks ───────────────────────────────────────────────────────


class SNNConvBlock(nn.Module):
    """
    Conv2d → BatchNorm → LIF spike block.

    Processes one time step; caller is responsible for looping over T.

    Parameters
    ----------
    in_channels, out_channels : int
    kernel_size, stride, padding : int
    beta : float
        LIF membrane decay factor.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        beta: float = 0.9,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.lif = _make_lif(beta=beta)

    def forward(
        self,
        x: torch.Tensor,
        mem: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x   : (B, C_in, H, W)  input spike/feature map for this time step
        mem : (B, C_out, H, W) | None   membrane potential (initialised to 0 if None)

        Returns
        -------
        spk : (B, C_out, H, W)
        mem : (B, C_out, H, W)
        """
        cur = self.bn(self.conv(x))
        if mem is None:
            mem = torch.zeros_like(cur)

        if _SNNTORCH_AVAILABLE:
            spk, mem = self.lif(cur, mem)
        else:
            spk, mem = self.lif(cur, mem)

        return spk, mem


@dataclass
class SNNConfig:
    """Configuration for the SNN satellite detector."""
    sensor_height: int = 260
    sensor_width: int = 346
    time_steps: int = 25
    hidden_channels: List[int] = field(default_factory=lambda: [32, 64, 128, 256])
    kernel_size: int = 3
    num_classes: int = 2
    lif_beta: float = 0.9
    lif_threshold: float = 1.0
    confidence_threshold: float = 0.5
    nms_iou_threshold: float = 0.3
    spike_reg_weight: float = 0.01


class SNNBackbone(nn.Module):
    """
    Four-layer convolutional SNN backbone.

    Processes (B, T, 2, H, W) event tensors through time, accumulating
    spike maps at each layer.

    Parameters
    ----------
    config : SNNConfig
    """

    def __init__(self, config: SNNConfig) -> None:
        super().__init__()
        ch = config.hidden_channels
        beta = config.lif_beta

        self.layer1 = SNNConvBlock(2,     ch[0], stride=1, beta=beta)
        self.layer2 = SNNConvBlock(ch[0], ch[1], stride=2, beta=beta)  # ½ res
        self.layer3 = SNNConvBlock(ch[1], ch[2], stride=2, beta=beta)  # ¼ res
        self.layer4 = SNNConvBlock(ch[2], ch[3], stride=2, beta=beta)  # ⅛ res

        self.config = config

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Parameters
        ----------
        x : torch.Tensor  shape (B, T, 2, H, W)

        Returns
        -------
        feat_p3 : (B, ch[2], H/4, W/4)   spike-accumulated feature map
        feat_p4 : (B, ch[3], H/8, W/8)
        spike_rate : float  mean spike rate across all layers (for regularisation)
        """
        B, T, C, H, W = x.shape
        mem1 = mem2 = mem3 = mem4 = None
        acc3 = None
        acc4 = None
        total_spikes = 0.0
        total_neurons = 0

        for t in range(T):
            xt = x[:, t]  # (B, 2, H, W)

            spk1, mem1 = self.layer1(xt, mem1)
            spk2, mem2 = self.layer2(spk1, mem2)
            spk3, mem3 = self.layer3(spk2, mem3)
            spk4, mem4 = self.layer4(spk3, mem4)

            acc3 = spk3 if acc3 is None else acc3 + spk3
            acc4 = spk4 if acc4 is None else acc4 + spk4

            # Track spike rate for regularisation
            for spk in [spk1, spk2, spk3, spk4]:
                total_spikes += spk.sum().item()
                total_neurons += spk.numel()

        spike_rate = total_spikes / max(total_neurons * T, 1)

        # Normalise accumulations by T → mean firing rate per pixel
        feat_p3 = acc3 / T
        feat_p4 = acc4 / T

        return feat_p3, feat_p4, spike_rate


class DetectionHead(nn.Module):
    """
    Detection head for satellite bounding-box prediction.

    Produces per-anchor class scores and box regression deltas.

    Parameters
    ----------
    in_channels : int
    num_classes : int
    num_anchors : int
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int = 2,
        num_anchors: int = 3,
    ) -> None:
        super().__init__()
        self.cls_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_anchors * num_classes, 1),
        )
        self.reg_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_anchors * 4, 1),  # (dx, dy, dw, dh)
        )
        self.num_classes = num_classes
        self.num_anchors = num_anchors

    def forward(
        self, feat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        feat : (B, C, H, W)

        Returns
        -------
        cls_logits : (B, num_anchors, num_classes, H, W)
        box_preds  : (B, num_anchors, 4, H, W)
        """
        B, C, H, W = feat.shape
        cls = self.cls_conv(feat)
        reg = self.reg_conv(feat)

        cls = cls.view(B, self.num_anchors, self.num_classes, H, W)
        reg = reg.view(B, self.num_anchors, 4, H, W)
        return cls, reg


# ── Full detector ─────────────────────────────────────────────────────────


class SNNSatelliteDetector(nn.Module):
    """
    Full SNN-based satellite detector combining backbone, FPN, and heads.

    Parameters
    ----------
    config : SNNConfig | None
        Detector configuration.  Defaults to SNNConfig().

    Examples
    --------
    >>> import torch
    >>> cfg = SNNConfig(time_steps=10, hidden_channels=[16, 32, 64, 128])
    >>> model = SNNSatelliteDetector(cfg)
    >>> # Batch of 2, 10 time steps, 2 polarity channels, 260×346
    >>> x = torch.rand(2, 10, 2, 260, 346)
    >>> with torch.no_grad():
    ...     outputs = model(x)
    >>> print(outputs.keys())
    dict_keys(['cls_p3', 'box_p3', 'cls_p4', 'box_p4', 'spike_rate'])
    """

    def __init__(self, config: Optional[SNNConfig] = None) -> None:
        super().__init__()
        self.config = config or SNNConfig()
        cfg = self.config

        self.backbone = SNNBackbone(cfg)

        ch = cfg.hidden_channels
        # FPN lateral convolutions
        self.lateral_p3 = nn.Conv2d(ch[2], 128, 1)
        self.lateral_p4 = nn.Conv2d(ch[3], 128, 1)
        self.fpn_p3 = nn.Conv2d(128, 128, 3, padding=1)

        # Detection heads
        self.head_p3 = DetectionHead(128, cfg.num_classes, num_anchors=3)
        self.head_p4 = DetectionHead(128, cfg.num_classes, num_anchors=3)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        x : torch.Tensor  shape (B, T, 2, H, W)
            Event frame stack.

        Returns
        -------
        dict with keys:
          ``cls_p3``, ``box_p3`` – predictions at ¼ resolution
          ``cls_p4``, ``box_p4`` – predictions at ⅛ resolution
          ``spike_rate``          – scalar mean spike rate (float tensor)
        """
        feat_p3, feat_p4, spike_rate = self.backbone(x)

        # FPN
        lat_p4 = self.lateral_p4(feat_p4)
        lat_p3 = self.lateral_p3(feat_p3)
        # Upsample p4 to p3 resolution and add
        up_p4 = F.interpolate(lat_p4, size=lat_p3.shape[2:], mode="nearest")
        fpn_p3 = self.fpn_p3(lat_p3 + up_p4)

        cls_p3, box_p3 = self.head_p3(fpn_p3)
        cls_p4, box_p4 = self.head_p4(lat_p4)

        return {
            "cls_p3": cls_p3,
            "box_p3": box_p3,
            "cls_p4": cls_p4,
            "box_p4": box_p4,
            "spike_rate": torch.tensor(spike_rate),
        }

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
        cls_weight: float = 1.0,
        loc_weight: float = 2.0,
        spike_reg_weight: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined detection + spike regularisation loss.

        For a fully functional loss you would implement anchor assignment
        (e.g. ATSS or IoU-based).  Here we provide a structurally complete
        loss that uses random target assignment for illustration.

        Parameters
        ----------
        outputs : dict  (returned by forward)
        targets : list of dicts  each with "boxes" (N,4) and "labels" (N,)
        cls_weight, loc_weight : float  loss weights
        spike_reg_weight : float | None  overrides config if given

        Returns
        -------
        dict with ``"total"``, ``"cls"``, ``"loc"``, ``"spike"`` losses.
        """
        if spike_reg_weight is None:
            spike_reg_weight = self.config.spike_reg_weight

        B = outputs["cls_p3"].shape[0]

        # Placeholder classification loss (cross-entropy on random assigns)
        cls_loss = torch.tensor(0.0, requires_grad=True)
        loc_loss = torch.tensor(0.0, requires_grad=True)

        for scale in ["p3", "p4"]:
            cls_logits = outputs[f"cls_{scale}"]  # (B, A, C, H, W)
            box_preds = outputs[f"box_{scale}"]   # (B, A, 4, H, W)

            B, A, C, H, W = cls_logits.shape
            # Flatten spatial dims
            cls_flat = cls_logits.permute(0, 1, 3, 4, 2).reshape(B, -1, C)
            box_flat = box_preds.permute(0, 1, 3, 4, 2).reshape(B, -1, 4)

            # Background class target (all background for demo)
            cls_target = torch.zeros(B, cls_flat.shape[1], dtype=torch.long,
                                     device=cls_flat.device)
            cls_loss = cls_loss + F.cross_entropy(
                cls_flat.reshape(-1, C), cls_target.reshape(-1)
            )

            # Smooth-L1 box regression (zero target for demo)
            box_target = torch.zeros_like(box_flat)
            loc_loss = loc_loss + F.smooth_l1_loss(box_flat, box_target)

        spike_loss = outputs["spike_rate"] * spike_reg_weight
        total_loss = cls_weight * cls_loss + loc_weight * loc_loss + spike_loss

        return {
            "total": total_loss,
            "cls": cls_loss.detach(),
            "loc": loc_loss.detach(),
            "spike": spike_loss.detach(),
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        *,
        confidence_threshold: Optional[float] = None,
    ) -> List[Dict[str, np.ndarray]]:
        """
        Run detector and decode predictions above confidence threshold.

        Parameters
        ----------
        x : torch.Tensor  shape (B, T, 2, H, W)
        confidence_threshold : float | None

        Returns
        -------
        list of B dicts, each with:
          ``"scores"``  shape (N,)  confidence scores
          ``"labels"``  shape (N,)  class indices
          ``"boxes"``   shape (N, 4)  [cx, cy, w, h] normalised
        """
        thresh = confidence_threshold or self.config.confidence_threshold
        self.eval()
        outputs = self.forward(x)
        results = []

        B = x.shape[0]
        for b in range(B):
            all_scores: List[float] = []
            all_labels: List[int] = []
            all_boxes: List[List[float]] = []

            for scale in ["p3", "p4"]:
                cls_logits = outputs[f"cls_{scale}"][b]  # (A, C, H, W)
                box_preds = outputs[f"box_{scale}"][b]   # (A, 4, H, W)

                A, C, H, W = cls_logits.shape
                probs = torch.softmax(cls_logits, dim=1)  # (A, C, H, W)
                max_probs, max_labels = probs.max(dim=1)  # (A, H, W)

                mask = (max_probs > thresh) & (max_labels > 0)  # ignore background
                if not mask.any():
                    continue

                scores = max_probs[mask].cpu().numpy()
                labels = max_labels[mask].cpu().numpy()

                a_idx, hy, wx = mask.nonzero(as_tuple=True)
                boxes = box_preds[a_idx, :, hy, wx].cpu().numpy()  # (N, 4)

                all_scores.extend(scores.tolist())
                all_labels.extend(labels.tolist())
                all_boxes.extend(boxes.tolist())

            results.append({
                "scores": np.array(all_scores, dtype=np.float32),
                "labels": np.array(all_labels, dtype=np.int32),
                "boxes": np.array(all_boxes, dtype=np.float32).reshape(-1, 4),
            })

        return results

    # ------------------------------------------------------------------
    # Benchmarking
    # ------------------------------------------------------------------

    def benchmark_latency(
        self,
        batch_size: int = 1,
        time_steps: int = 10,
        n_runs: int = 100,
        device: str = "cpu",
        warmup: int = 10,
    ) -> Dict[str, float]:
        """
        Measure inference latency and estimate energy proxy.

        Parameters
        ----------
        batch_size : int
        time_steps : int
        n_runs : int
            Number of timed forward passes.
        device : str
        warmup : int
            Warm-up iterations before timing.

        Returns
        -------
        dict with ``"mean_ms"``, ``"std_ms"``, ``"min_ms"``, ``"max_ms"``,
        ``"spike_rate"``, ``"synaptic_ops_estimate"``.
        """
        self.eval()
        self.to(device)
        H, W = self.config.sensor_height, self.config.sensor_width
        dummy = torch.zeros(batch_size, time_steps, 2, H, W, device=device)

        # Warm-up
        with torch.no_grad():
            for _ in range(warmup):
                _ = self.forward(dummy)

        times_ms: List[float] = []
        spike_rates: List[float] = []

        with torch.no_grad():
            for _ in range(n_runs):
                t0 = time.perf_counter()
                out = self.forward(dummy)
                t1 = time.perf_counter()
                times_ms.append((t1 - t0) * 1000.0)
                spike_rates.append(float(out["spike_rate"]))

        mean_sr = float(np.mean(spike_rates))
        # Rough SOP estimate: spike_rate × total_weights
        n_params = sum(p.numel() for p in self.parameters())
        sop_estimate = mean_sr * n_params * time_steps

        return {
            "mean_ms": float(np.mean(times_ms)),
            "std_ms": float(np.std(times_ms)),
            "min_ms": float(np.min(times_ms)),
            "max_ms": float(np.max(times_ms)),
            "spike_rate": mean_sr,
            "synaptic_ops_estimate": sop_estimate,
            "n_parameters": n_params,
        }
