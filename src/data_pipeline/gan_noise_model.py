"""
GAN-based Cosmic Radiation Noise Model.

Uses a Deep Convolutional GAN (DCGAN) to model realistic space noise
patterns including cosmic ray hits, dark current noise, readout noise,
and radiation belt artefacts.  The model can augment event streams with
physically plausible noise for robust training of downstream SNNs.

Architecture
------------
  Generator  : latent vector → 1-channel noise patch (64×64)
  Discriminator : noise patch → real/fake score

Training follows the DCGAN recipe (Radford et al., 2015) with optional
WGAN-GP gradient penalty for training stability.

References
----------
Radford, A., Metz, L., & Chintala, S. (2015). Unsupervised Representation
  Learning with Deep Convolutional Generative Adversarial Networks.
  https://arxiv.org/abs/1511.06434

Gulrajani, I., et al. (2017). Improved Training of Wasserstein GANs.
  https://arxiv.org/abs/1704.00028
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


# ── Architecture ──────────────────────────────────────────────────────────


class NoiseGenerator(nn.Module):
    """
    DCGAN generator: z → 64×64 noise patch.

    Parameters
    ----------
    latent_dim : int
        Dimension of the latent noise vector.
    features : int
        Base feature map width (scaled by powers of 2 through layers).
    channels : int
        Output image channels (1 for grayscale noise).
    """

    def __init__(
        self,
        latent_dim: int = 100,
        features: int = 64,
        channels: int = 1,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            # input: (latent_dim, 1, 1)
            self._block(latent_dim, features * 8, 4, 1, 0),   # → 4×4
            self._block(features * 8, features * 4, 4, 2, 1), # → 8×8
            self._block(features * 4, features * 2, 4, 2, 1), # → 16×16
            self._block(features * 2, features,     4, 2, 1), # → 32×32
            nn.ConvTranspose2d(features, channels, 4, 2, 1),  # → 64×64
            nn.Tanh(),
        )

    @staticmethod
    def _block(
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                               stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        z : torch.Tensor  shape (B, latent_dim, 1, 1)

        Returns
        -------
        torch.Tensor  shape (B, channels, 64, 64), values in [-1, 1]
        """
        return self.net(z)


class NoiseDiscriminator(nn.Module):
    """
    DCGAN discriminator: 64×64 patch → scalar real/fake score.

    Parameters
    ----------
    features : int
        Base feature map width.
    channels : int
        Input image channels.
    """

    def __init__(self, features: int = 64, channels: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, features, 4, 2, 1, bias=False),    # → 32×32
            nn.LeakyReLU(0.2, inplace=True),
            self._block(features,     features * 2, 4, 2, 1),      # → 16×16
            self._block(features * 2, features * 4, 4, 2, 1),      # → 8×8
            self._block(features * 4, features * 8, 4, 2, 1),      # → 4×4
            nn.Conv2d(features * 8, 1, 4, 1, 0),                   # → 1×1
        )

    @staticmethod
    def _block(
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor  shape (B, channels, 64, 64)

        Returns
        -------
        torch.Tensor  shape (B, 1, 1, 1)
        """
        return self.net(x)


# ── Noise synthesis functions ─────────────────────────────────────────────


def _cosmic_ray_noise(
    height: int,
    width: int,
    n_hits: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Simulate cosmic ray hit patterns.

    Each hit is a straight-line streak with random position, angle, and
    length — matching the appearance of heavy ion tracks in CCD/CMOS sensors.

    Parameters
    ----------
    height, width : int
        Patch dimensions.
    n_hits : int
        Number of cosmic ray hits to inject.
    rng : np.random.Generator

    Returns
    -------
    np.ndarray  shape (H, W), dtype float32, values in [0, 1]
    """
    patch = np.zeros((height, width), dtype=np.float32)
    for _ in range(n_hits):
        # Start point
        x0 = rng.integers(0, width)
        y0 = rng.integers(0, height)
        # Streak length and angle
        length = rng.integers(3, max(4, min(height, width) // 4))
        angle = rng.uniform(0, np.pi)
        dx = np.cos(angle)
        dy = np.sin(angle)
        # Draw streak
        intensity = rng.uniform(0.7, 1.0)
        for s in range(length):
            xi = int(x0 + s * dx) % width
            yi = int(y0 + s * dy) % height
            patch[yi, xi] = min(1.0, patch[yi, xi] + intensity)
    return patch


def _dark_current_noise(
    height: int,
    width: int,
    rate: float,
    temperature_k: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Simulate dark current noise (Poisson process).

    Parameters
    ----------
    height, width : int
    rate : float
        Mean dark current level (normalised, 0–1).
    temperature_k : float
        Sensor temperature (K) — higher → more dark current.
    rng : np.random.Generator

    Returns
    -------
    np.ndarray  shape (H, W), dtype float32
    """
    # Scale rate by temperature factor (approx. doubles every 6°C)
    temp_factor = 2.0 ** ((temperature_k - 293.0) / 6.0)
    lambda_ = rate * temp_factor
    dc = rng.poisson(lambda_, size=(height, width)).astype(np.float32)
    # Normalise
    if dc.max() > 0:
        dc /= dc.max()
    return dc


def _hot_pixel_noise(
    height: int,
    width: int,
    fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Inject hot pixels that fire at high rates.

    Parameters
    ----------
    height, width : int
    fraction : float
        Fraction of pixels that are 'hot' (e.g. 0.001 = 0.1%).
    rng : np.random.Generator

    Returns
    -------
    np.ndarray  shape (H, W), dtype float32
    """
    patch = np.zeros((height, width), dtype=np.float32)
    n_hot = max(1, int(height * width * fraction))
    xs = rng.integers(0, width, size=n_hot)
    ys = rng.integers(0, height, size=n_hot)
    intensities = rng.uniform(0.5, 1.0, size=n_hot)
    patch[ys, xs] = intensities
    return patch


def _readout_noise(
    height: int,
    width: int,
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Gaussian readout noise."""
    noise = rng.normal(0, sigma, size=(height, width)).astype(np.float32)
    noise = np.clip(noise, 0, None)
    if noise.max() > 0:
        noise /= noise.max()
    return noise


@dataclass
class NoiseConfig:
    """Configuration for synthetic noise generation."""
    patch_size: int = 64
    cosmic_ray_prob: float = 0.3      # Probability of CR hit per patch
    cosmic_ray_max: int = 5           # Max hits per patch
    dark_current_rate: float = 0.05
    temperature_k: float = 300.0
    hot_pixel_fraction: float = 0.001
    readout_sigma: float = 0.05
    orbital_altitude_km: float = 500.0
    van_allen_factor: float = 1.0     # Multiplicative radiation enhancement


class SyntheticNoiseDataset:
    """
    Generate a dataset of synthetic space noise patches for GAN training.

    Parameters
    ----------
    n_samples : int
        Number of patches to generate.
    config : NoiseConfig
    seed : int | None
    """

    def __init__(
        self,
        n_samples: int = 10_000,
        config: Optional[NoiseConfig] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.n_samples = n_samples
        self.config = config or NoiseConfig()
        self.rng = np.random.default_rng(seed)

    def generate(self) -> np.ndarray:
        """
        Generate ``n_samples`` noise patches.

        Returns
        -------
        np.ndarray  shape (N, 1, patch_size, patch_size), dtype float32
                    values in [-1, 1] (GAN-ready normalisation)
        """
        cfg = self.config
        H = W = cfg.patch_size
        patches = np.zeros((self.n_samples, H, W), dtype=np.float32)

        for i in range(self.n_samples):
            composite = np.zeros((H, W), dtype=np.float32)

            # Dark current
            composite += _dark_current_noise(H, W, cfg.dark_current_rate,
                                             cfg.temperature_k, self.rng)
            # Hot pixels
            composite += _hot_pixel_noise(H, W, cfg.hot_pixel_fraction, self.rng)
            # Readout
            composite += _readout_noise(H, W, cfg.readout_sigma, self.rng)
            # Cosmic rays (altitude-scaled)
            cr_scale = (cfg.van_allen_factor *
                        max(0.1, 1.0 - cfg.orbital_altitude_km / 40_000))
            if self.rng.random() < cfg.cosmic_ray_prob * cr_scale:
                n_hits = self.rng.integers(1, cfg.cosmic_ray_max + 1)
                composite += _cosmic_ray_noise(H, W, n_hits, self.rng)

            composite = np.clip(composite, 0, 1)
            patches[i] = composite

        # Reshape to (N, 1, H, W) and normalise to [-1, 1]
        patches = patches[:, np.newaxis, :, :]
        patches = patches * 2.0 - 1.0
        return patches


# ── GAN training class ────────────────────────────────────────────────────


class CosmicNoiseGAN:
    """
    DCGAN for modelling cosmic radiation noise in orbital sensors.

    Parameters
    ----------
    latent_dim : int
        Size of the generator input vector.
    features_g : int
        Generator base feature width.
    features_d : int
        Discriminator base feature width.
    lr_g : float
        Generator learning rate.
    lr_d : float
        Discriminator learning rate.
    beta1 : float
        Adam beta1 parameter.
    device : str | torch.device
        Training device.
    use_gradient_penalty : bool
        If True, uses WGAN-GP loss.  Default: False (vanilla DCGAN).
    lambda_gp : float
        Gradient penalty weight (only used if use_gradient_penalty=True).
    seed : int | None
        RNG seed.

    Examples
    --------
    >>> gan = CosmicNoiseGAN(device="cpu")
    >>> gan.train(n_epochs=5, batch_size=32, show_progress=False)
    >>> samples = gan.generate(n_samples=16)   # (16, 1, 64, 64)
    """

    def __init__(
        self,
        latent_dim: int = 100,
        features_g: int = 64,
        features_d: int = 64,
        lr_g: float = 2e-4,
        lr_d: float = 2e-4,
        beta1: float = 0.5,
        device: Union[str, torch.device] = "cpu",
        use_gradient_penalty: bool = False,
        lambda_gp: float = 10.0,
        seed: Optional[int] = None,
    ) -> None:
        self.latent_dim = latent_dim
        self.device = torch.device(device)
        self.use_gp = use_gradient_penalty
        self.lambda_gp = lambda_gp

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        self.G = NoiseGenerator(latent_dim, features_g).to(self.device)
        self.D = NoiseDiscriminator(features_d).to(self.device)

        self._init_weights()

        self.opt_G = optim.Adam(self.G.parameters(), lr=lr_g, betas=(beta1, 0.999))
        self.opt_D = optim.Adam(self.D.parameters(), lr=lr_d, betas=(beta1, 0.999))

        self.criterion = nn.BCEWithLogitsLoss()
        self.history: Dict[str, List[float]] = {"loss_G": [], "loss_D": []}

    def _init_weights(self) -> None:
        """Apply DCGAN weight initialisation (mean=0, std=0.02)."""
        for m in list(self.G.modules()) + list(self.D.modules()):
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)

    def train(
        self,
        n_epochs: int = 100,
        batch_size: int = 64,
        n_samples_train: int = 10_000,
        noise_config: Optional[NoiseConfig] = None,
        show_progress: bool = True,
        save_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, List[float]]:
        """
        Train the GAN on synthetically generated noise patches.

        Parameters
        ----------
        n_epochs : int
            Number of training epochs.
        batch_size : int
        n_samples_train : int
            Number of synthetic patches to generate for training.
        noise_config : NoiseConfig | None
            Noise generation parameters.
        show_progress : bool
            Display tqdm progress bar.
        save_path : str | Path | None
            If given, saves generator weights here after training.

        Returns
        -------
        dict
            Training loss history ``{"loss_G": [...], "loss_D": [...]}``.
        """
        from tqdm import tqdm

        dataset = SyntheticNoiseDataset(n_samples_train, noise_config)
        patches = dataset.generate()
        tensor = torch.from_numpy(patches)
        loader = DataLoader(TensorDataset(tensor), batch_size=batch_size,
                            shuffle=True, drop_last=True)

        real_label = 1.0
        fake_label = 0.0

        for epoch in range(n_epochs):
            epoch_loss_g: List[float] = []
            epoch_loss_d: List[float] = []

            pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{n_epochs}",
                        leave=False, disable=not show_progress)

            for (real_batch,) in pbar:
                real_batch = real_batch.to(self.device)
                B = real_batch.size(0)

                # ── Train Discriminator ───────────────────────────────
                self.D.zero_grad()

                # Real
                labels_real = torch.full((B, 1, 1, 1), real_label,
                                         device=self.device)
                out_real = self.D(real_batch)
                loss_d_real = self.criterion(out_real, labels_real)

                # Fake
                z = torch.randn(B, self.latent_dim, 1, 1, device=self.device)
                fake_batch = self.G(z).detach()
                labels_fake = torch.full((B, 1, 1, 1), fake_label,
                                          device=self.device)
                out_fake = self.D(fake_batch)
                loss_d_fake = self.criterion(out_fake, labels_fake)

                loss_D = (loss_d_real + loss_d_fake) * 0.5

                if self.use_gp:
                    gp = self._gradient_penalty(real_batch, fake_batch)
                    loss_D = loss_D + self.lambda_gp * gp

                loss_D.backward()
                self.opt_D.step()
                epoch_loss_d.append(loss_D.item())

                # ── Train Generator ───────────────────────────────────
                self.G.zero_grad()
                z = torch.randn(B, self.latent_dim, 1, 1, device=self.device)
                fake_batch = self.G(z)
                labels_real_for_g = torch.full((B, 1, 1, 1), real_label,
                                               device=self.device)
                out_fake_g = self.D(fake_batch)
                loss_G = self.criterion(out_fake_g, labels_real_for_g)
                loss_G.backward()
                self.opt_G.step()
                epoch_loss_g.append(loss_G.item())

                pbar.set_postfix({"G": f"{loss_G.item():.4f}",
                                  "D": f"{loss_D.item():.4f}"})

            mean_g = float(np.mean(epoch_loss_g))
            mean_d = float(np.mean(epoch_loss_d))
            self.history["loss_G"].append(mean_g)
            self.history["loss_D"].append(mean_d)

            if show_progress:
                logger.info("Epoch %d/%d | G=%.4f | D=%.4f",
                            epoch + 1, n_epochs, mean_g, mean_d)

        if save_path:
            self.save(save_path)

        return self.history

    def _gradient_penalty(
        self,
        real: torch.Tensor,
        fake: torch.Tensor,
    ) -> torch.Tensor:
        """Compute WGAN-GP gradient penalty."""
        B = real.size(0)
        alpha = torch.rand(B, 1, 1, 1, device=self.device)
        interpolated = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
        d_interpolated = self.D(interpolated)
        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True,
            retain_graph=True,
        )[0]
        gradients = gradients.view(B, -1)
        gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gp

    @torch.no_grad()
    def generate(self, n_samples: int = 16) -> np.ndarray:
        """
        Sample noise patches from the trained generator.

        Parameters
        ----------
        n_samples : int
            Number of patches to generate.

        Returns
        -------
        np.ndarray  shape (N, 1, 64, 64), values in [0, 1]
        """
        self.G.eval()
        z = torch.randn(n_samples, self.latent_dim, 1, 1, device=self.device)
        patches = self.G(z).cpu().numpy()
        self.G.train()
        # Map [-1, 1] → [0, 1]
        return (patches + 1.0) * 0.5

    def save(self, path: Union[str, Path]) -> None:
        """Save generator and discriminator weights."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "generator": self.G.state_dict(),
            "discriminator": self.D.state_dict(),
            "history": self.history,
        }, path)
        logger.info("Saved GAN weights to %s", path)

    def load(self, path: Union[str, Path]) -> None:
        """Load generator and discriminator weights."""
        checkpoint = torch.load(path, map_location=self.device)
        self.G.load_state_dict(checkpoint["generator"])
        self.D.load_state_dict(checkpoint["discriminator"])
        self.history = checkpoint.get("history", self.history)
        logger.info("Loaded GAN weights from %s", path)


# ── Augmentation pipeline ─────────────────────────────────────────────────


class NoiseAugmentor:
    """
    Augment event frames with GAN-generated or synthetic space noise.

    Can be used as a drop-in augmentation step in the SNN training loop.

    Parameters
    ----------
    gan : CosmicNoiseGAN | None
        Trained GAN for noise sampling.  If ``None``, falls back to
        direct synthetic noise generation.
    noise_config : NoiseConfig | None
        Parameters for fallback synthetic noise generation.
    augment_prob : float
        Probability of applying noise augmentation per sample (0–1).
    noise_scale : float
        Multiplicative scale for the noise added to the frame.
    seed : int | None
        RNG seed.

    Examples
    --------
    >>> augmentor = NoiseAugmentor(noise_scale=0.1)
    >>> frame = np.random.rand(2, 260, 346).astype(np.float32)
    >>> noisy = augmentor(frame)
    """

    def __init__(
        self,
        gan: Optional[CosmicNoiseGAN] = None,
        noise_config: Optional[NoiseConfig] = None,
        augment_prob: float = 0.5,
        noise_scale: float = 0.1,
        seed: Optional[int] = None,
    ) -> None:
        self.gan = gan
        self.config = noise_config or NoiseConfig()
        self.augment_prob = augment_prob
        self.noise_scale = noise_scale
        self.rng = np.random.default_rng(seed)
        self._synthetic_gen = SyntheticNoiseDataset(n_samples=1000,
                                                    config=self.config,
                                                    seed=seed)
        self._cache: Optional[np.ndarray] = None

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply noise augmentation to a polarity event frame.

        Parameters
        ----------
        frame : np.ndarray  shape (C, H, W)
            Event frame to augment (float32, values in [0, 1]).

        Returns
        -------
        np.ndarray  shape (C, H, W)
        """
        if self.rng.random() > self.augment_prob:
            return frame

        noise_patch = self._sample_noise_patch(frame.shape[1], frame.shape[2])
        return np.clip(frame + self.noise_scale * noise_patch, 0, 1)

    def _sample_noise_patch(self, height: int, width: int) -> np.ndarray:
        """Sample a single noise patch and resize to (height, width)."""
        if self.gan is not None:
            patch = self.gan.generate(n_samples=1)[0, 0]  # (64, 64)
        else:
            if self._cache is None:
                self._cache = self._synthetic_gen.generate()[:, 0, :, :]  # (N, 64, 64)
            idx = self.rng.integers(len(self._cache))
            patch = (self._cache[idx] + 1.0) * 0.5   # back to [0,1]

        if patch.shape != (height, width):
            import cv2
            patch = cv2.resize(patch, (width, height),
                               interpolation=cv2.INTER_LINEAR)

        return patch.astype(np.float32)

    def augment_orbital_altitude(
        self,
        altitude_km: float,
        van_allen_zone: bool = False,
    ) -> None:
        """
        Update noise configuration for a given orbital altitude.

        Parameters
        ----------
        altitude_km : float
            Orbital altitude in kilometres.
        van_allen_zone : bool
            Whether the orbit passes through the Van Allen belts (~2 000–
            6 000 km altitude).
        """
        va_factor = 1.5 if van_allen_zone else 1.0
        cr_scale = va_factor * max(0.1, 1.0 - altitude_km / 40_000)
        self.config.cosmic_ray_prob = min(0.9, 0.3 * cr_scale)
        self.config.van_allen_factor = va_factor
        self.config.orbital_altitude_km = altitude_km
        # Invalidate cache
        self._cache = None
        logger.debug(
            "Noise config updated for altitude %.0f km, CR prob=%.3f",
            altitude_km, self.config.cosmic_ray_prob,
        )
