import numpy as np

from benchmark.core.types import GTSequence, TrackerPrediction, MatchedSequence


#Hier werden die gt.frame_ids nicht benutzt. Für WHAM egal, bei anderen Trackern nochmal gedanken darüber machen!!
def match_by_frame_ids(gt: GTSequence, pred: TrackerPrediction, allow_drop_invalid: bool = False, matching_strategy="default") -> MatchedSequence:

    frame_ids = np.asarray(pred.frame_ids, dtype=int)

    if pred.tracker_name == "wham":

        if matching_strategy == "sparse_frames":
            frame_ids = np.arange(36) * 3

            pred_vertices = pred.vertices[frame_ids]

            gt_vertices = gt.vertices[frame_ids]
            gt_vertices = np.asarray(gt_vertices, dtype=np.float32)

            strategy = "sparse_frames"

        else:
            valid = (frame_ids >= 0) & (frame_ids < len(gt.vertices))

            if not valid.all():
                if not allow_drop_invalid:
                    bad = frame_ids[~valid]
                    raise ValueError(
                        f"Invalid frame_ids for {pred.subject}/{pred.sequence}. "
                        f"GT length={len(gt.vertices)}, invalid ids={bad[:10]}"
                    )

                frame_ids = frame_ids[valid]
                pred_vertices = pred.vertices[valid]
                strategy = "frame_ids_drop_invalid"
            else:
                pred_vertices = pred.vertices
                strategy = "frame_ids"

            gt_vertices = gt.vertices[frame_ids]

    elif pred.tracker_name == "multihmr":

        if matching_strategy == "sparse_frames":
            frame_ids = np.arange(36) * 3
            pred_vertices = pred.vertices[np.arange(36)]
            gt_vertices = gt.vertices[frame_ids]
            gt_vertices = np.asarray(gt_vertices, dtype=np.float32)
            strategy = "sparse_frames"

    elif pred.tracker_name in ("dmmr", "easymocap"):
        pred_vertices = pred.vertices

        gt_vertices = gt.vertices[frame_ids]
        gt_vertices = np.asarray(gt_vertices, dtype=np.float32)

        strategy = "sparse_frames"

    else:
        raise ValueError(f"Unknown tracker {pred.tracker_name}")

    return MatchedSequence(
        pred_vertices=pred_vertices,
        gt_vertices=gt_vertices,
        frame_ids=frame_ids,
        match_strategy=strategy,
    )