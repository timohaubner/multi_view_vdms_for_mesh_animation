from pathlib import Path
import joblib
import numpy as np

from benchmark.core.types import TrackerPrediction
from benchmark.trackers.base import BaseTrackerLoader


class WhamLoader(BaseTrackerLoader):
    tracker_name = "wham"

    def load(self, prediction_paths: list[Path], subject: str, sequence: str) -> TrackerPrediction:
        prediction_path = prediction_paths[0]

        if not prediction_path.exists():
            raise FileNotFoundError(prediction_path)

        wham = joblib.load(prediction_path)

        if not isinstance(wham, dict):
            raise TypeError(f"Expected WHAM output dict, got {type(wham)}")

        if len(wham) != 1:
            raise ValueError(f"Multiple persons found: {list(wham.keys())}")

        person_id = list(wham.keys())[0]
        pred = wham[person_id]

        conv = np.asarray([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1],
        ])

        vertices = np.asarray(pred["verts"], dtype=np.float32) @ conv
        frame_ids = np.asarray(pred["frame_ids"], dtype=int)

        return TrackerPrediction(
            tracker_name=self.tracker_name,
            subject=subject,
            sequence=sequence,
            vertices=vertices,
            frame_ids=frame_ids,
            source_path=prediction_path,
        )