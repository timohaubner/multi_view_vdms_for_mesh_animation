import numpy as np

from benchmark.core.matching import match_by_frame_ids
from benchmark.core.smpl import vertices_to_joints
from benchmark.metrics.common import point_error_mm, acceleration_error_mm
from benchmark.core.alignment import (
    batch_procrustes_align,
    align_global_translation,
    centroid_align,
    pelvis_align,
)


class SequenceEvaluator:
    def __init__(self, j_regressor_male, j_regressor_female):
        self.j_regressor_male = j_regressor_male
        self.j_regressor_female = j_regressor_female

    def evaluate(self, gt, pred, gender, mode="default") -> dict:
        if gender == "male":
            j_regressor = self.j_regressor_male
        elif gender == "female":
            j_regressor = self.j_regressor_female
        else:
            raise ValueError("Unknown value for \"gender\"")

        matched = match_by_frame_ids(gt, pred, matching_strategy=mode)

        pred_v = matched.pred_vertices
        gt_v = matched.gt_vertices

        pred_j = vertices_to_joints(pred_v, j_regressor)
        gt_j = vertices_to_joints(gt_v, j_regressor)

        pred_v_initial_aligned, pred_j_initial_aligned = align_global_translation(pred_v, pred_j, gt_v)
        pred_v_centroid_aligned, pred_j_centroid_aligned = centroid_align(pred_v, pred_j, gt_v)
        pred_v_pelvis_aligned, pred_j_pelvis_aligned = pelvis_align(pred_v, pred_j, gt_j)

        pred_v_pa_aligned = batch_procrustes_align(pred_v, gt_v)
        pred_j_pa_aligned = batch_procrustes_align(pred_j, gt_j)

        accel = acceleration_error_mm(pred_j_pelvis_aligned, gt_j)

        mpvpe = point_error_mm(pred_v_initial_aligned, gt_v)
        mpjpe = point_error_mm(pred_j_initial_aligned, gt_j)

        centroid_aligned_mpvpe = point_error_mm(pred_v_centroid_aligned, gt_v)
        centroid_aligned_mpjpe = point_error_mm(pred_j_centroid_aligned, gt_j)

        pelvis_aligned_mpvpe = point_error_mm(pred_v_pelvis_aligned, gt_v)
        pelvis_aligned_mpjpe = point_error_mm(pred_j_pelvis_aligned, gt_j)

        pa_mpvpe = point_error_mm(pred_v_pa_aligned, gt_v)
        pa_mpjpe = point_error_mm(pred_j_pa_aligned, gt_j)

        return {
            "subject": gt.subject,
            "sequence": gt.sequence,
            "tracker": pred.tracker_name,
            "num_gt_frames": len(gt.vertices),
            "num_pred_frames": len(pred.vertices),
            "num_matched_frames": len(pred_v),
            "match_strategy": matched.match_strategy,
            "accel_error": accel["mean"],
            "per_frame_accel_error": accel["per_frame"],
            "mpvpe": mpvpe["mean"],
            "per_frame_mpvpe": mpvpe["per_frame"],
            "centroid_aligned_mpvpe": centroid_aligned_mpvpe["mean"],
            "per_frame_centroid_aligned_mpvpe": centroid_aligned_mpvpe["per_frame"],
            "pelvis_aligned_mpvpe": pelvis_aligned_mpvpe["mean"],
            "per_frame_pelvis_aligned_mpvpe": pelvis_aligned_mpvpe["per_frame"],
            "pa_mpvpe": pa_mpvpe["mean"],
            "per_frame_pa_mpvpe": pa_mpvpe["per_frame"],
            "mpjpe": mpjpe["mean"],
            "per_frame_mpjpe": mpjpe["per_frame"],
            "centroid_aligned_mpjpe": centroid_aligned_mpjpe["mean"],
            "per_frame_centroid_aligned_mpjpe": centroid_aligned_mpjpe["per_frame"],
            "pelvis_aligned_mpjpe": pelvis_aligned_mpjpe["mean"],
            "per_frame_pelvis_aligned_mpjpe": pelvis_aligned_mpjpe["per_frame"],
            "pa_mpjpe": pa_mpjpe["mean"],
            "per_frame_pa_mpjpe": pa_mpjpe["per_frame"],
            "pred_v_initial_aligned": pred_v_initial_aligned,
            "pred_v_centroid_aligned": pred_v_centroid_aligned,
            "pred_v_pelvis_aligned": pred_v_pelvis_aligned,
            "pred_v_pa_aligned": pred_v_pa_aligned,
            "gt_v": gt_v,
        }