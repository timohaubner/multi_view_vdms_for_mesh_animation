import cv2
import torch
from torchmetrics.functional.image import (
    peak_signal_noise_ratio,
    structural_similarity_index_measure,
)
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


def frame_to_tensor(frame, device: torch.device) -> torch.Tensor:
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    tensor = torch.from_numpy(frame_rgb)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)
    tensor = tensor.float().div(255.0)

    return tensor.to(device)


def calculate_frame_metrics(
    prediction_frame,
    gt_frame,
    device: torch.device,
    lpips_metric: LearnedPerceptualImagePatchSimilarity,
):
    prediction_tensor = frame_to_tensor(prediction_frame, device)
    gt_tensor = frame_to_tensor(gt_frame, device)

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

    return (
        float(psnr.item()),
        float(ssim.item()),
        float(lpips.mean().item()),
    )