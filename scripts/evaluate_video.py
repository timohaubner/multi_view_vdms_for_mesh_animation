import argparse
from pathlib import Path

import cv2
import pandas as pd
import torch
from torchmetrics.functional.image import (
    peak_signal_noise_ratio,
    structural_similarity_index_measure,
)
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


def frame_to_tensor(frame, device):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    tensor = torch.from_numpy(frame_rgb)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)
    tensor = tensor.float().div(255.0)

    return tensor.to(device)


def evaluate_video(generated_path, gt_path, output_csv):
    device = torch.device("cpu")

    generated_capture = cv2.VideoCapture(str(generated_path))
    gt_capture = cv2.VideoCapture(str(gt_path))

    if not generated_capture.isOpened():
        raise RuntimeError(
            f"Could not open generated video: {generated_path}"
        )

    if not gt_capture.isOpened():
        generated_capture.release()
        raise RuntimeError(
            f"Could not open ground-truth video: {gt_path}"
        )

    generated_frame_count = int(
        generated_capture.get(cv2.CAP_PROP_FRAME_COUNT)
    )
    gt_frame_count = int(
        gt_capture.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    lpips_metric = LearnedPerceptualImagePatchSimilarity(
        net_type="alex",
        normalize=True,
        reduction="none",
    ).to(device)
    lpips_metric.eval()

    results = []
    frame_index = 0

    try:
        with torch.inference_mode():
            while True:
                generated_ok, generated_frame = generated_capture.read()
                gt_ok, gt_frame = gt_capture.read()

                if not generated_ok or not gt_ok:
                    break

                if generated_frame.shape != gt_frame.shape:
                    raise ValueError(
                        f"Different frame shapes at frame {frame_index}: "
                        f"generated={generated_frame.shape}, "
                        f"gt={gt_frame.shape}"
                    )

                generated_tensor = frame_to_tensor(
                    generated_frame,
                    device,
                )
                gt_tensor = frame_to_tensor(
                    gt_frame,
                    device,
                )

                psnr = peak_signal_noise_ratio(
                    generated_tensor,
                    gt_tensor,
                    data_range=1.0,
                )
                ssim = structural_similarity_index_measure(
                    generated_tensor,
                    gt_tensor,
                    data_range=1.0,
                )
                lpips = lpips_metric(
                    generated_tensor,
                    gt_tensor,
                )

                results.append(
                    {
                        "frame": frame_index,
                        "psnr": psnr.item(),
                        "ssim": ssim.item(),
                        "lpips": lpips.item(),
                    }
                )

                frame_index += 1

    finally:
        generated_capture.release()
        gt_capture.release()

    if not results:
        raise RuntimeError("No frames were evaluated.")

    dataframe = pd.DataFrame(results)

    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    dataframe.to_csv(output_csv, index=False)

    print()
    print(f"Generated frames: {generated_frame_count}")
    print(f"Ground-truth frames: {gt_frame_count}")
    print(f"Evaluated frames: {len(dataframe)}")
    print(f"Device: {device}")
    print()
    print("Mean values:")
    print(f"PSNR:  {dataframe['psnr'].mean():.4f} dB")
    print(f"SSIM:  {dataframe['ssim'].mean():.4f}")
    print(f"LPIPS: {dataframe['lpips'].mean():.4f}")
    print()
    print(f"Per-frame results: {output_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--generated",
        required=True,
    )
    parser.add_argument(
        "--gt",
        required=True,
    )
    parser.add_argument(
        "--output",
        default="metrics.csv",
    )
    args = parser.parse_args()

    evaluate_video(
        Path(args.generated),
        Path(args.gt),
        Path(args.output),
    )


if __name__ == "__main__":
    main()