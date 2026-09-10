# Multi-View VDMs for Humanoid Mesh Animation

Research code for the practical project **"Evaluating Multi-View Video Diffusion for Humanoid Mesh Animation"**.

## Overview

This project investigates whether additional views synthesized by a pretrained multi-view video diffusion model can improve downstream 3D human motion reconstruction compared with reconstruction from a single video view.

The experimental pipeline focuses on two main stages:

1. Multi-view video synthesis using SV4D 2.0 to generate synchronized target-view videos from a reference video.
2. Multi-view human reconstruction using an EasyMocap-based pipeline to recover an SMPL mesh sequence from the reference and synthesized views.

The experiments use motion sequences from the CAPE dataset in a controlled setup, enabling quantitative comparison against known 3D ground truth. The repository also contains evaluation code for reconstruction accuracy, synthesized-view quality, viewpoint configurations, and long-horizon generation with different reconditioning strategies.

## Repository Structure

```text
.
├── benchmark/              # 3D reconstruction evaluation, metrics, trackers, and configs
├── video_evaluation/       # Evaluation of synthesized target-view video quality
├── sv4d_scripts/           # Scripts and modifications for SV4D-based multi-view synthesis
├── scripts/                # Rendering, preprocessing, conversion, and utility scripts
├── data/                   # CAPE dataset and renderings (placeholder) 
├── outputs/                # Generated experiment outputs (placeholder)
├── visualization/          # Visualization outputs (placeholder)
├── run_benchmark.py        # Entry point for reconstruction benchmarks
└── run_video_evaluation.py # Entry point for synthesized-video evaluation
```

This repository is a research prototype intended for controlled experimentation and evaluation rather than a deployment-ready animation system.

## Environment Setup

Different stages of the experimental pipeline use separate Conda environments due to their differing software requirements.

The environment specifications are provided in `environments/`:

```text
environments/
├── rendering.yml
├── benchmark.yml
└── video_evaluation.yml
