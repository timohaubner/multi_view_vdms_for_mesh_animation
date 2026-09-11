from pathlib import Path

import cv2
import torch
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from video_evaluation.metrics.common import calculate_frame_metrics


def evaluate_sequence(
    sequence_name: str,
    prediction_path: Path,
    gt_path: Path,
    device: torch.device,
    lpips_metric: LearnedPerceptualImagePatchSimilarity,
    collect_frame_metrics: bool = False,
) -> dict:
    prediction_capture = cv2.VideoCapture(str(prediction_path))
    gt_capture = cv2.VideoCapture(str(gt_path))

    if not prediction_capture.isOpened():
        raise RuntimeError(f"Could not open prediction video: {prediction_path}")

    if not gt_capture.isOpened():
        prediction_capture.release()
        raise RuntimeError(f"Could not open ground-truth video: {gt_path}")

    psnr_sum = 0.0
    ssim_sum = 0.0
    lpips_sum = 0.0
    frame_count = 0
    frame_metrics = []

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
                        f"Different frame shapes for {sequence_name}, "
                        f"frame {frame_count}: "
                        f"prediction={prediction_frame.shape}, "
                        f"gt={gt_frame.shape}"
                    )

                psnr, ssim, lpips = calculate_frame_metrics(
                    prediction_frame,
                    gt_frame,
                    device,
                    lpips_metric,
                )

                psnr_sum += psnr
                ssim_sum += ssim
                lpips_sum += lpips

                if collect_frame_metrics:
                    frame_metrics.append(
                        {
                            "sequence": sequence_name,
                            "frame": frame_count,
                            "psnr": psnr,
                            "ssim": ssim,
                            "lpips": lpips,
                        }
                    )

                frame_count += 1

    finally:
        prediction_capture.release()
        gt_capture.release()

    if frame_count == 0:
        raise RuntimeError(f"No frames evaluated for sequence: {sequence_name}")

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