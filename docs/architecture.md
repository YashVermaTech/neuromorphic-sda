# System Architecture

## Overview

The Neuromorphic SDA pipeline converts orbital imagery into asynchronous
event streams and processes them with a Spiking Neural Network for
microsecond-latency satellite detection.

```
┌─────────────────────────────────────────────────────────────────┐
│                  NEUROMORPHIC SDA PIPELINE                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────────┐    ┌──────────────┐  │
│  │ Orbital      │    │   GAN Noise      │    │  Dataset     │  │
│  │ Imagery      │───▶│   Augmentation   │───▶│  Curator     │  │
│  │ (NASA FITS / │    │  (DCGAN / WGAN)  │    │  (.npz/HDF5) │  │
│  │  Simulated)  │    └──────────────────┘    └──────┬───────┘  │
│  └──────┬───────┘                                   │           │
│         │ v2e Algorithm                             │           │
│         ▼                                           │           │
│  ┌──────────────┐                                   │           │
│  │ Event Camera │    Events: (t, x, y, polarity)   │           │
│  │ Simulator    │    Format: structured numpy /     │           │
│  │ (DVS/DAVIS)  │    AEDAT4                         │           │
│  └──────┬───────┘                                   │           │
│         │                                           │           │
│         ▼                                           ▼           │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              SPIKE ENCODING LAYER                         │  │
│  │   Window accumulation → Polarity frames (2, H, W)        │  │
│  │   Time surface computation (τ-decay)                     │  │
│  └──────────────────────┬───────────────────────────────────┘  │
│                          │                                       │
│                          ▼                                       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │          SPIKING NEURAL NETWORK DETECTOR                  │  │
│  │                                                           │  │
│  │  Input (B, T, 2, H, W) ──▶ SNN Backbone                 │  │
│  │                              │                            │  │
│  │  [LIF Layer 1: 32ch]         │ Spike trains               │  │
│  │  [LIF Layer 2: 64ch ÷2]      │ (asynchronous)             │  │
│  │  [LIF Layer 3: 128ch ÷2] ────┤                            │  │
│  │  [LIF Layer 4: 256ch ÷2]     │                            │  │
│  │                              ▼                            │  │
│  │                         FPN Neck                          │  │
│  │                        P3 ─── P4                          │  │
│  │                         │       │                          │  │
│  │                    Head P3   Head P4                       │  │
│  │                    (cls+box) (cls+box)                     │  │
│  └──────────────────────────────────────────────────────────┘  │
│                          │                                       │
│                          ▼                                       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              STAR TRACKER & ATTITUDE                      │  │
│  │                                                           │  │
│  │  Event clusters ──▶ Centroiding ──▶ Kalman filter        │  │
│  │  Star candidates ──▶ HYG catalogue match                 │  │
│  │  Moving objects  ──▶ Satellite discrimination             │  │
│  │  Output: RA/Dec attitude + satellite bounding boxes      │  │
│  └──────────────────────────────────────────────────────────┘  │
│                          │                                       │
│                          ▼                                       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                 BENCHMARK & EVALUATION                    │  │
│  │                                                           │  │
│  │  OrbitalBenchmarkEnv (LEO / MEO / GEO / SSO / HEO)      │  │
│  │  Ground truth trajectories + deterministic simulation    │  │
│  │  Metrics: mAP, DR, FAR, Latency, Spike rate              │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Module Dependencies

```
configs/default_config.yaml
        │
        ▼
src/utils/config.py ──────────────────────────────────┐
        │                                              │
        ▼                                              │
src/data_pipeline/                                     │
  ├── orbital_to_events.py   ◀── EventStream dtype     │
  ├── gan_noise_model.py     ◀── NoiseConfig            │
  └── dataset_curator.py    ◀── EventStream             │
        │                                              │
        ▼                                              │
src/models/                                            │
  ├── centroiding.py         (no internal deps)        │
  ├── star_tracker.py        ◀── centroiding            │
  └── snn_detector.py        ◀── (snnTorch / torch)    │
        │                                              │
        ▼                                              │
src/benchmarks/                                        │
  ├── deterministic_env.py  ◀── orbital_to_events      │
  └── metrics.py            (no internal deps)         │
        │                                              │
        ▼                                              │
src/utils/visualization.py ◀── EventStream dtype ──────┘
```

## Data Flow

### Event Stream Format

Events are stored as a structured numpy array with dtype:
```python
EVENT_DTYPE = np.dtype([
    ("t", np.int64),   # timestamp in microseconds
    ("x", np.int16),   # pixel column
    ("y", np.int16),   # pixel row
    ("p", np.int8),    # polarity: +1 (ON) or -1 (OFF)
])
```

### Temporal Processing

The SNN processes events in temporal windows of configurable duration.
Within each window:

1. Events are accumulated into a polarity frame `(2, H, W)` where
   channel 0 = ON events, channel 1 = OFF events
2. The frame is fed through T=25 time steps of the SNN
3. LIF neurons integrate and fire asynchronously
4. Accumulated spike maps feed the detection head

### Energy Efficiency Model

The system estimates energy consumption via Synaptic Operations (SOPs):
```
SOPs = Σ_layers (spike_rate_l × n_weights_l × T)
```
This is proportional to the fraction of neurons that fire, making sparse
event inputs highly energy efficient compared to dense CNN processing.

## Sensor Model

The DVS346 (used as the default sensor model) has:
- Resolution: 346 × 260 pixels
- Latency: < 1 μs per event
- Dynamic range: 120 dB
- Temporal resolution: 1 μs

The v2e conversion models:
- Per-pixel threshold variation (σ = 0.03)
- Refractory period (1 μs)
- Shot noise (0.5 Hz/px)
- Leak noise (0.1 Hz/px)

## Deployment Considerations

For on-orbit deployment:
- **Neuromorphic hardware**: Intel Loihi 2, BrainScaleS, or SpiNNaker
- **Latency target**: < 100 μs from event → detection
- **Power budget**: < 1 W (vs. ~10–50 W for GPU-based CNN)
- **Radiation tolerance**: GAN noise model trains detector on
  radiation-corrupted data for robustness
