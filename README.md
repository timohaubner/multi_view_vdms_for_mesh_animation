# Multi-View VDMs for Humanoid Mesh Animation

Research code for the practical project **"Evaluating Multi-View Video Diffusion for Humanoid Mesh Animation"**.

## Overview

This project investigates whether additional views synthesized by a pretrained multi-view video diffusion model can improve downstream 3D human motion reconstruction compared with reconstruction from a single video view.

The experimental pipeline focuses on two main stages:

1. Multi-view video synthesis using SV4D 2.0 to generate synchronized target-view videos from a reference video.
2. 3D human reconstruction and evaluation using an EasyMocap-based reconstruction pipeline and additional tracker baselines.

The experiments use motion sequences from the CAPE dataset in a controlled setup, enabling quantitative comparison against known 3D ground truth.

This repository is a research prototype intended for controlled experimentation and evaluation rather than a deployment-ready animation system.

## Repository Structure

```text
.
├── benchmark/              # 3D reconstruction evaluation, metrics, trackers, and configs
├── video_evaluation/       # Evaluation of synthesized target-view video quality
├── sv4d_scripts/           # Scripts and modifications for SV4D-based multi-view synthesis
├── scripts/                # Rendering, preprocessing, conversion, and utility scripts
├── environments/           # Conda environment definitions
├── data/                   # Dataset and rendering data
├── outputs/                # Generated experiment outputs
├── visualization/          # Visualization outputs
├── run_benchmark.py        # Entry point for reconstruction benchmarks
└── run_video_evaluation.py # Entry point for synthesized-video evaluation
```

## Environments

Different stages of the pipeline use separate Conda environments:

```text
environments/
├── rendering.yml
├── benchmark.yml
└── video_evaluation.yml
```

For example:

```bash
conda env create -f environments/benchmark.yml
```

## Documentation

More detailed instructions are kept with the corresponding part of the repository.

- [`benchmark/README.md`](benchmark/README.md) — reconstruction benchmark
- [`video_evaluation/README.md`](video_evaluation/README.md) — synthesized-video evaluation
- `sv4d_scripts/` — SV4D-based multi-view generation

## Benchmark

Reconstruction benchmarks are run through:

```bash
python run_benchmark.py <config>
```

See [`benchmark/README.md`](benchmark/README.md) for the available configurations and required environment variables.