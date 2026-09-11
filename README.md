# Multi-View Video Diffusion for Humanoid Mesh Animation

Research code for the project **“Evaluating Multi-View Video Diffusion for Humanoid Mesh Animation”**, exploring how synthesized multi-view videos can improve 3D motion reconstruction for animating humanoid meshes.

<table>
  <tr>
    <td align="center" width="33%">
      <img src="assets/ours.gif" width="100%"><br>
      <strong>Ours</strong><br>
      <sub>SV4D 2.0 + EasyMocap</sub>
    </td>
    <td align="center" width="33%">
      <img src="assets/multihmr.gif" width="100%"><br>
      <strong>Multi-HMR</strong><br>
      <sub>Single-view baseline</sub>
    </td>
    <td align="center" width="33%">
      <img src="assets/wham.gif" width="100%"><br>
      <strong>WHAM</strong><br>
      <sub>Single-view baseline</sub>
    </td>
  </tr>
</table>

This project explores video diffusion as a motion prior for animating humanoid 3D meshes.
Instead of generating motion directly in 3D, motion is represented through video and
subsequently recovered as an explicit 3D mesh sequence.

A key limitation of existing video-based animation approaches is that recovering 3D
motion from a single view is inherently ambiguous due to missing depth information and
self-occlusions. This project investigates whether a pretrained multi-view video
diffusion model can synthesize additional viewpoints that provide stronger geometric
constraints for recovering the underlying human motion.

Using **SV4D 2.0** for synchronized novel-view synthesis and an **EasyMocap**-based
SMPL reconstruction pipeline, the experimental multi-view pipeline achieves
**58.55 mm MPVPE**, compared with **83.98 mm** for Multi-HMR and **112.57 mm**
for WHAM on the controlled CAPE evaluation.

📄 [Full Project Report](docs/ProjectReport.pdf)

## Approach

<p align="center">
  <img src="assets/ArchitectureDiagram.svg" width="95%" alt="Architecture overview">
</p>

The proposed architecture aims to animate a static human mesh from a semantic motion
description using video diffusion as an indirect motion representation.

A single-view video diffusion model first generates a video-based motion prior from a
rendering of the input mesh. Instead of reconstructing the animation directly from this
monocular video, a multi-view video diffusion model synthesizes synchronized observations
from additional viewpoints. These views are then jointly used by a human-specific
multi-view reconstruction pipeline to recover the final 3D mesh animation.

For the controlled experimental evaluation in this project, the single-view generation
stage is replaced by rendered motion sequences with known 3D ground truth (CAPE). This
isolates the multi-view synthesis and reconstruction stages and enables quantitative
evaluation.

## Key Results

The core experiment compares the proposed synthesized multi-view pipeline with the
single-view human reconstruction methods **WHAM** and **Multi-HMR**.

| Method | MPVPE ↓ [mm] | PA-MPVPE ↓ [mm] | Pelvis-Aligned MPJPE ↓ [mm] | PA-MPJPE ↓ [mm] | Acceleration Error ↓ [mm/frame²] |
|---|---:|---:|---:|---:|---:|
| WHAM | 112.57 | 47.41 | 66.03 | 38.25 | 11.98 |
| Multi-HMR | 83.98 | **41.06** | 64.11 | **32.83** | 17.08 |
| **SV4D 2.0 + EasyMocap (Ours)** | **58.55** | 44.72 | **49.55** | 36.37 | **10.48** |

The synthesized multi-view observations substantially improve reconstruction metrics
that retain information about **global human motion**. Multi-HMR remains stronger on
the Procrustes-aligned metrics, which focus more strongly on local pose accuracy after
removing global translation, rotation, and scale.

## Research Questions

- **RQ1 — Do synthesized target views improve 3D human motion reconstruction?**  
  Synthesized target views provide useful additional geometric constraints for recovering
  3D human motion. The proposed pipeline achieves **58.55 mm MPVPE**, compared with
  **83.98 mm** for Multi-HMR and **112.57 mm** for WHAM, with the main benefit appearing
  in reconstruction metrics that retain information about global motion.

- **RQ2a — How does angular viewpoint spacing affect reconstruction?**  
  Increasing the angular spacing introduces a trade-off between geometric coverage and
  synthesis quality. While wider viewpoints provide stronger geometric constraints when synthesis errors are
  removed using source-rendered views, synthesized-view fidelity decreases at larger offsets.
  Consequently, the **30° and 45°** configurations outperform the **72°** configuration
  in downstream reconstruction, with MPVPE increasing from **58.55 mm** at 30° to
  **68.65 mm** at 72°.

- **RQ2b — How does the number of views affect reconstruction?**  
  Additional views can improve reconstruction by providing complementary observations,
  but the improvement is not monotonic because each synthesized view can also introduce
  noise or cross-view inconsistencies. The full **five-view configuration** achieves the
  best overall performance with **58.55 mm MPVPE**, although the improvement over smaller
  view subsets remains comparatively small.

- **RQ3 — Does target-view conditioning improve multi-view synthesis?**  
  Providing source-rendered conditioning images from the target viewpoints substantially
  improves both synthesized-view fidelity and temporal stability. LPIPS decreases from
  **0.1078 to 0.0524**, while downstream reconstruction improves from **66.68 mm to
  58.55 mm MPVPE**. The stronger improvement in image fidelity than in reconstruction
  accuracy also indicates that visual similarity alone does not fully capture geometric
  consistency.

- **RQ4 — Can reconditioning improve long-horizon generation?**  
  To reduce error accumulation across consecutive generation windows, we propose a
  **mesh-level reconditioning strategy** that reconstructs the intermediate 3D motion
  and re-renders it from the target viewpoints before continuing generation. Compared
  with the native autoregressive continuation of SV4D 2.0, mesh-level reconditioning
  preserves synthesized-view fidelity better over longer horizons, reaching
  **0.0920 vs. 0.1014 LPIPS** after the second reconditioning boundary. However, this
  does not improve downstream reconstruction: the native SV4D 2.0 continuation achieves
  **66.34 mm MPVPE**, compared with **69.65 mm** for mesh-level reconditioning,
  leaving reliable long-horizon continuation as a remaining challenge.

## Repository Structure

```text
.
├── benchmark/              # 3D reconstruction evaluation, metrics, trackers, and configs
├── video_evaluation/       # Evaluation of synthesized target-view video quality
├── sv4d_scripts/           # Modified SV4D 2.0 scripts kept for documentation
├── scripts/                # Standalone research utilities for rendering, preprocessing, conversion and analysis
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
- [`scripts/README.md`](scripts/README.md) — standalone utility scripts used during the experimental workflow
- `sv4d_scripts/` — modified SV4D 2.0 scripts kept for documentation; they are not intended to be run from this repository

## Benchmark

Reconstruction benchmarks are run through:

```bash
python run_benchmark.py <config>
```

See [`benchmark/README.md`](benchmark/README.md) for the available configurations and required environment variables.