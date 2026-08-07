from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import pandas as pd
import torch
import yaml

from torchmetrics.functional.image import (
    peak_signal_noise_ratio,
    structural_similarity_index_measure,
)
from torchmetrics.image.lpip import (
    LearnedPerceptualImagePatchSimilarity,
)


DEFAULT_OUTPUT_FILENAME = "video_metrics.csv"
VALID_OUTPUT_MODES = {"sequence", "frame"}


def frame_to_tensor(frame, device: torch.device) -> torch.Tensor:
    """Convert an OpenCV BGR frame to a normalized RGB tensor."""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    tensor = torch.from_numpy(frame_rgb)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)
    tensor = tensor.float().div(255.0)

    return tensor.to(device)


def resolve_path(path_value: str | Path, config_dir: Path) -> Path:
    """Resolve relative paths against the directory of the YAML file."""
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = config_dir / path
    return path.resolve()


def resolve_output_csv(output_path: str | Path, config_dir: Path) -> Path:
    """
    Resolve the configured output path.

    If output_path ends in .csv, it is treated as the complete CSV path.
    Otherwise, it is treated as a directory and video_metrics.csv is created
    inside it.
    """
    resolved = resolve_path(output_path, config_dir)

    if resolved.suffix.lower() == ".csv":
        return resolved

    return resolved / DEFAULT_OUTPUT_FILENAME


def parse_videos(raw_videos: Any, config_dir: Path) -> list[dict[str, Path | str]]:
    """
    Parse either of these YAML forms:

    videos:
      sequence_1:
        prediction: path/to/prediction.mp4
        gt: path/to/gt.mp4

    or:

    videos:
      - name: sequence_1
        prediction: path/to/prediction.mp4
        gt: path/to/gt.mp4
    """
    parsed: list[dict[str, Path | str]] = []

    if isinstance(raw_videos, dict):
        iterable = []
        for sequence_name, video_config in raw_videos.items():
            if not isinstance(video_config, dict):
                raise ValueError(
                    f"Konfiguration für Sequenz '{sequence_name}' "
                    f"muss ein Mapping sein."
                )
            iterable.append(
                {
                    "name": sequence_name,
                    **video_config,
                }
            )
    elif isinstance(raw_videos, list):
        iterable = raw_videos
    else:
        raise ValueError(
            "'videos' muss entweder ein Mapping oder eine Liste sein."
        )

    for index, video_config in enumerate(iterable):
        if not isinstance(video_config, dict):
            raise ValueError(
                f"Videoeintrag {index} muss ein Mapping sein."
            )

        prediction_value = video_config.get("prediction")
        gt_value = video_config.get("gt")

        if not prediction_value or not gt_value:
            raise ValueError(
                f"Videoeintrag {index} benötigt 'prediction' und 'gt'."
            )

        prediction_path = resolve_path(prediction_value, config_dir)
        gt_path = resolve_path(gt_value, config_dir)

        sequence_name = video_config.get("name")
        if not sequence_name:
            sequence_name = prediction_path.stem

        parsed.append(
            {
                "sequence": str(sequence_name),
                "prediction": prediction_path,
                "gt": gt_path,
            }
        )

    if not parsed:
        raise ValueError("Die Konfiguration enthält keine Videos.")

    return parsed


def evaluate_sequence(
    sequence_name: str,
    prediction_path: Path,
    gt_path: Path,
    device: torch.device,
    lpips_metric: LearnedPerceptualImagePatchSimilarity,
    collect_frame_metrics: bool = False,
) -> dict[str, Any]:
    """Evaluate one prediction/ground-truth video pair."""
    prediction_capture = cv2.VideoCapture(str(prediction_path))
    gt_capture = cv2.VideoCapture(str(gt_path))

    if not prediction_capture.isOpened():
        raise RuntimeError(
            f"Prediction-Video konnte nicht geöffnet werden: "
            f"{prediction_path}"
        )

    if not gt_capture.isOpened():
        prediction_capture.release()
        raise RuntimeError(
            f"GT-Video konnte nicht geöffnet werden: {gt_path}"
        )

    psnr_sum = 0.0
    ssim_sum = 0.0
    lpips_sum = 0.0
    frame_count = 0
    frame_metrics: list[dict[str, Any]] = []

    try:
        with torch.inference_mode():
            while True:
                prediction_ok, prediction_frame = prediction_capture.read()
                gt_ok, gt_frame = gt_capture.read()

                if not prediction_ok and not gt_ok:
                    break

                if prediction_ok != gt_ok:
                    break

                if prediction_frame.shape != gt_frame.shape:
                    raise ValueError(
                        f"Unterschiedliche Auflösung bei Sequenz "
                        f"'{sequence_name}', Frame {frame_count}: "
                        f"prediction={prediction_frame.shape}, "
                        f"gt={gt_frame.shape}"
                    )

                prediction_tensor = frame_to_tensor(
                    prediction_frame,
                    device,
                )
                gt_tensor = frame_to_tensor(
                    gt_frame,
                    device,
                )

                psnr = peak_signal_noise_ratio(
                    prediction_tensor,
                    gt_tensor,
                    data_range=1.0,
                )
                ssim = structural_similarity_index_measure(
                    prediction_tensor,
                    gt_tensor,
                    data_range=1.0,
                )
                lpips = lpips_metric(
                    prediction_tensor,
                    gt_tensor,
                )

                psnr_value = float(psnr.item())
                ssim_value = float(ssim.item())
                lpips_value = float(lpips.mean().item())

                psnr_sum += psnr_value
                ssim_sum += ssim_value
                lpips_sum += lpips_value

                if collect_frame_metrics:
                    frame_metrics.append(
                        {
                            "sequence": sequence_name,
                            "frame": frame_count,
                            "psnr": psnr_value,
                            "ssim": ssim_value,
                            "lpips": lpips_value,
                        }
                    )

                frame_count += 1

    finally:
        prediction_capture.release()
        gt_capture.release()

    if frame_count == 0:
        raise RuntimeError(
            f"Für Sequenz '{sequence_name}' wurden keine Frames ausgewertet."
        )

    return {
        "sequence": sequence_name,
        "prediction": str(prediction_path),
        "gt": str(gt_path),
        "frames": frame_count,
        "psnr": psnr_sum / frame_count,
        "ssim": ssim_sum / frame_count,
        "lpips": lpips_sum / frame_count,
        "frame_metrics": frame_metrics,
    }


def add_summary_rows(
    dataframe: pd.DataFrame,
    label_column: str,
) -> pd.DataFrame:
    """
    Append AVERAGE and STD rows calculated across all result rows.

    STD uses ddof=0, i.e. the population standard deviation. This also yields
    0.0 for a single result row.
    """
    metric_columns = ["psnr", "ssim", "lpips"]

    average_row: dict[str, Any] = {
        label_column: "AVERAGE",
    }
    std_row: dict[str, Any] = {
        label_column: "STD",
    }

    for column in metric_columns:
        average_row[column] = dataframe[column].mean()
        std_row[column] = dataframe[column].std(ddof=0)

    return pd.concat(
        [
            dataframe,
            pd.DataFrame([average_row, std_row]),
        ],
        ignore_index=True,
    )


def evaluate_from_config(config_path: str | Path) -> Path:
    """Load a YAML configuration, evaluate all videos, and write one CSV."""
    config_path = Path(config_path).expanduser().resolve()

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Konfigurationsdatei wurde nicht gefunden: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Die YAML-Konfiguration muss ein Mapping enthalten.")

    if "output_path" not in config:
        raise ValueError("'output_path' fehlt in der Konfiguration.")

    if "videos" not in config:
        raise ValueError("'videos' fehlt in der Konfiguration.")

    output_mode = str(config.get("output_mode", "sequence")).strip().lower()
    if output_mode not in VALID_OUTPUT_MODES:
        raise ValueError(
            "'output_mode' muss entweder 'sequence' oder 'frame' sein."
        )

    config_dir = config_path.parent
    output_csv = resolve_output_csv(
        config["output_path"],
        config_dir,
    )
    videos = parse_videos(
        config["videos"],
        config_dir,
    )

    device = torch.device("cpu")

    lpips_metric = LearnedPerceptualImagePatchSimilarity(
        net_type="alex",
        normalize=True,
        reduction="none",
    ).to(device)
    lpips_metric.eval()

    results = []
    all_frame_metrics: list[dict[str, Any]] = []

    for index, video in enumerate(videos, start=1):
        print(
            f"[{index}/{len(videos)}] "
            f"Evaluiere Sequenz '{video['sequence']}' ..."
        )

        result = evaluate_sequence(
            sequence_name=str(video["sequence"]),
            prediction_path=Path(video["prediction"]),
            gt_path=Path(video["gt"]),
            device=device,
            lpips_metric=lpips_metric,
            collect_frame_metrics=output_mode == "frame",
        )
        results.append(result)

        if output_mode == "frame":
            all_frame_metrics.extend(result["frame_metrics"])
            print(f"  Frames: {result['frames']}")
        else:
            print(
                f"  Frames: {result['frames']} | "
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
        dataframe = add_summary_rows(dataframe, label_column="frame")
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
        dataframe = add_summary_rows(dataframe, label_column="sequence")

    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    dataframe.to_csv(
        output_csv,
        index=False,
        float_format="%.6f",
    )

    print()
    print(f"Ergebnisdatei: {output_csv}")

    return output_csv