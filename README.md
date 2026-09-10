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
├── Slurm_Scripts/          # Scripts for running experiments on a Slurm cluster
├── Data/                   # CAPE dataset and renderings (placeholder) 
├── results/                # Generated experiment outputs (placeholder)
├── visualization/          # Visualization outputs (placeholder)
├── run_benchmark.py        # Entry point for reconstruction benchmarks
└── run_video_evaluation.py # Entry point for synthesized-video evaluation
```

This repository is a research prototype intended for controlled experimentation and evaluation rather than a deployment-ready animation system.

## Documentation

Installation instructions, external dependencies, dataset setup, and commands for reproducing the experiments will be added during the repository cleanup.
