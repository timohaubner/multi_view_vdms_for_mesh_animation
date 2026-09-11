import torch
import numpy as np


def point_error_mm(pred, gt) -> dict:
    pred = torch.as_tensor(pred, dtype=torch.float32)
    gt = torch.as_tensor(gt, dtype=torch.float32)

    err = torch.linalg.norm(pred - gt, dim=-1)
    per_frame = err.mean(dim=1) * 1000.0

    return {
        "mean": float(per_frame.mean().item()),
        "median": float(per_frame.median().item()),
        "per_frame": per_frame.cpu().numpy(),
    }


def acceleration_error_mm(pred_joints, gt_joints):
    pred_joints = np.asarray(pred_joints, dtype=np.float32)
    gt_joints = np.asarray(gt_joints, dtype=np.float32)

    assert pred_joints.shape == gt_joints.shape
    assert pred_joints.ndim == 3

    if pred_joints.shape[0] < 3:
        return {
            "mean": np.nan,
            "per_frame": np.array([]),
        }

    pred_acc = pred_joints[2:] - 2 * pred_joints[1:-1] + pred_joints[:-2]
    gt_acc = gt_joints[2:] - 2 * gt_joints[1:-1] + gt_joints[:-2]

    err = np.linalg.norm(pred_acc - gt_acc, axis=-1)
    per_frame = err.mean(axis=1) * 1000.0
    per_frame = per_frame[1:-1]

    return {
        "mean": float(per_frame.mean()),
        "median": float(np.median(per_frame)),
        "per_frame": per_frame,
    }