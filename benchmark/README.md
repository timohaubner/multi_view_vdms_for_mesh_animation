# Reconstruction Benchmark

Evaluation code for comparing reconstructed mesh sequences against CAPE ground truth.

## Structure

```text
benchmark/
├── configs/        # Experiment configurations
├── core/           # Matching, alignment, SMPL and CAPE utilities
├── evaluation/     # Benchmark runner and evaluator
├── metrics/        # Metrics
├── trackers/       # EasyMocap, WHAM and Multi-HMR loaders
└── results/        # Benchmark results
```

## Required Environment Variables

```text
CAPE_ROOT
SMPL_ROOT
SMPLX2SMPL_PATH
EASYMOCAP_RESULTS
WHAM_RESULTS
MULTIHMR_RESULTS
```

Only the variables needed by the selected tracker have to be set.

## Run

From the repository root:

```bash
python run_benchmark.py benchmark/configs/wham.yml
python run_benchmark.py benchmark/configs/multihmr.yml
python run_benchmark.py benchmark/configs/easymocap/azimuth/30_generated.yml
```

Additional configurations are available under `benchmark/configs/`.

The benchmark reports MPVPE, MPJPE, aligned variants, and acceleration error. Output paths are defined in the corresponding YAML configuration.