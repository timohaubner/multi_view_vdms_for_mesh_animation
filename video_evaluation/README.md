# Video Evaluation

Evaluation code for comparing synthesized target-view videos against rendered ground-truth videos.

## Structure

```text
video_evaluation/
├── configs/        # Experiment configurations
├── core/           # Path utilities
├── evaluation/     # Video evaluation runner and evaluator
├── metrics/        # PSNR, SSIM and LPIPS computation
└── results/        # Evaluation results
```

## Required Environment Variables

```text
SV4D_RESULTS
RENDERINGS_ROOT
```

Set the variables to the corresponding local data directories before running an evaluation.

## Run

From the repository root:

```bash
python run_video_evaluation.py video_evaluation/configs/azimuth/30.yml
python run_video_evaluation.py video_evaluation/configs/azimuth/45.yml
python run_video_evaluation.py video_evaluation/configs/auto/run1_GT.yml
```

Additional configurations are available under `video_evaluation/configs/`.

The evaluation reports PSNR, SSIM, and LPIPS. Output paths are defined in the corresponding YAML configuration.