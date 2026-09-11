import torch


def batch_procrustes_align(pred, gt, eps: float = 1e-8):
    pred = torch.as_tensor(pred, dtype=torch.float32)
    gt = torch.as_tensor(gt, dtype=torch.float32)

    mu_pred = pred.mean(dim=1, keepdim=True)
    mu_gt = gt.mean(dim=1, keepdim=True)

    X = pred - mu_pred
    Y = gt - mu_gt

    var_X = (X ** 2).sum(dim=(1, 2), keepdim=True)

    K = X.transpose(1, 2) @ Y

    U, S, Vh = torch.linalg.svd(K)
    V = Vh.transpose(1, 2)

    Z = torch.eye(3, device=pred.device).unsqueeze(0).repeat(pred.shape[0], 1, 1)
    det = torch.det(V @ U.transpose(1, 2))
    Z[:, -1, -1] = torch.sign(det)

    R = V @ Z @ U.transpose(1, 2)

    scale = (
        S * torch.diagonal(Z, dim1=-2, dim2=-1)
    ).sum(dim=1, keepdim=True).unsqueeze(-1)

    scale = scale / (var_X + eps)

    aligned = scale * (X @ R.transpose(1, 2)) + mu_gt

    return aligned


def align_global_translation(pred_vertices, pred_joints, gt_vertices):
    offset = (gt_vertices - pred_vertices).mean(axis=(0, 1))

    pred_vertices_aligned = pred_vertices + offset[None, None, :]
    pred_joints_aligned = pred_joints + offset[None, None, :]

    return pred_vertices_aligned, pred_joints_aligned


def centroid_align(pred_vertices, pred_joints, gt_vertices):
    pred_centroid = pred_vertices.mean(axis=1)
    gt_centroid = gt_vertices.mean(axis=1)

    centroid_offset = gt_centroid - pred_centroid

    pred_vertices_aligned = pred_vertices + centroid_offset[:, None, :]
    pred_joints_aligned = pred_joints + centroid_offset[:, None, :]

    return pred_vertices_aligned, pred_joints_aligned


def pelvis_align(pred_vertices, pred_joints, gt_joints):
    pelvis_offset = gt_joints[:, 0, :] - pred_joints[:, 0, :]

    pred_vertices_aligned = pred_vertices + pelvis_offset[:, None, :]
    pred_joints_aligned = pred_joints + pelvis_offset[:, None, :]

    return pred_vertices_aligned, pred_joints_aligned