from pathlib import Path

import pandas as pd
import torch
import yaml
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from video_evaluation.core.paths import resolve_output_csv, resolve_path
from video_evaluation.evaluation.evaluator import evaluate_sequence


VALID_OUTPUT_MODES = {"sequence", "frame"}


def parse_videos(raw_videos):
    videos = []

    if isinstance(raw_videos, dict):
        iterable = []

        for sequence_name, video_config in raw_videos.items():
            if not isinstance(video_config, dict):
                raise ValueError(f"Invalid config for sequence: {sequence_name}")

            iterable.append(
                {
                    "name": sequence_name,
                    **video_config,
                }
            )
    elif isinstance(raw_videos, list):
        iterable = raw_videos
    else:
        raise ValueError("Videos must be a mapping or a list.")

    for index, video_config in enumerate(iterable):
        if not isinstance(video_config, dict):
            raise ValueError(f"Invalid video config at index {index}")

        prediction = video_config.get("prediction")
        gt = video_config.get("gt")

        if not prediction or not gt:
            raise ValueError(
                f"Video config at index {index} requires prediction and gt."
            )

        prediction_path = resolve_path(prediction)
        gt_path = resolve_path(gt)

        sequence_name = video_config.get("name", prediction_path.stem)

        videos.append(
            {
                "sequence": str(sequence_name),
                "prediction": prediction_path,
                "gt": gt_path,
            }
        )

    if not videos:
        raise ValueError("No videos specified.")

    return videos


def add_summary_rows(
    dataframe: pd.DataFrame,
    label_column: str,
) -> pd.DataFrame:
    metric_columns = ["psnr", "ssim", "lpips"]

    average_row = {label_column: "AVERAGE"}
    std_row = {label_column: "STD"}

    for column in metric_columns:
        average_row[column] = dataframe[column].mean()
        std_row[column] = dataframe[column].std(ddof=1)

    return pd.concat(
        [
            dataframe,
            pd.DataFrame([average_row, std_row]),
        ],
        ignore_index=True,
    )


def evaluate_from_config(config_path: str | Path) -> Path:
    config_path = Path(config_path).expanduser().resolve()

    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Config must contain a mapping.")

    if "output_path" not in config:
        raise ValueError("Missing output_path in config.")

    if "videos" not in config:
        raise ValueError("Missing videos in config.")

    output_mode = str(config.get("output_mode", "sequence")).strip().lower()

    if output_mode not in VALID_OUTPUT_MODES:
        raise ValueError(f"Unknown output mode: {output_mode}")

    output_csv = resolve_output_csv(config["output_path"])
    videos = parse_videos(config["videos"])

    device = torch.device("cpu")

    lpips_metric = LearnedPerceptualImagePatchSimilarity(
        net_type="alex",
        normalize=True,
        reduction="none",
    ).to(device)
    lpips_metric.eval()

    results = []
    all_frame_metrics = []

    for index, video in enumerate(videos, start=1):
        print(f"[{index}/{len(videos)}] Evaluating {video['sequence']}")

        result = evaluate_sequence(
            sequence_name=video["sequence"],
            prediction_path=video["prediction"],
            gt_path=video["gt"],
            device=device,
            lpips_metric=lpips_metric,
            collect_frame_metrics=output_mode == "frame",
        )
        results.append(result)

        if output_mode == "frame":
            all_frame_metrics.extend(result["frame_metrics"])
        else:
            print(
                f"Frames: {result['frames']} | "
                f"PSNR: {result['psnr']:.4f} dB | "
                f"SSIM: {result['ssim']:.4f} | "
                f"LPIPS: {result['lpips']:.4f}"
            )

    if output_mode == "frame":
        dataframe = pd.DataFrame(all_frame_metrics)

        dataframe = (
            dataframe.groupby("frame", as_index=False)[
                ["psnr", "ssim", "lpips"]
            ]
            .mean()
            .sort_values("frame")
            .reset_index(drop=True)
        )

        dataframe = add_summary_rows(dataframe, "frame")
    else:
        dataframe = pd.DataFrame(results)
        dataframe = dataframe[
            [
                "sequence",
                "psnr",
                "ssim",
                "lpips",
            ]
        ]

        dataframe = add_summary_rows(dataframe, "sequence")

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    dataframe.to_csv(
        output_csv,
        index=False,
        float_format="%.6f",
    )

    print(f"Results saved to: {output_csv}")

    return output_csv