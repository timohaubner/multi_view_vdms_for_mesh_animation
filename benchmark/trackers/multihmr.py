from pathlib import Path
import pickle
import numpy as np
from scipy import sparse

from benchmark.core.types import TrackerPrediction
from benchmark.trackers.base import BaseTrackerLoader


class MultiHMRLoader(BaseTrackerLoader):
    tracker_name = "multihmr"

    def __init__(self, smplx2smpl_path: Path):
        with smplx2smpl_path.open("rb") as file:
            data = pickle.load(file, encoding="latin1")

        if isinstance(data, dict):
            matrix = data.get("matrix", data.get("mtx", None))

            if matrix is None:
                raise KeyError(f"Could not find transfer matrix in keys: {data.keys()}")
        else:
            matrix = data

        if sparse.issparse(matrix):
            matrix = matrix.toarray()

        self.smplx2smpl = np.asarray(matrix, dtype=np.float32)

    def load(self, prediction_paths: list[Path], subject: str, sequence: str) -> TrackerPrediction:
        prediction_path = prediction_paths[0]

        files = sorted(prediction_path.glob("*.npy"))[:36]

        if not files:
            raise FileNotFoundError(f"No npy files in {prediction_path}")

        vertices = []

        for path in files:
            smplx = np.load(path)

            if smplx.ndim == 2:
                smplx = smplx[None]

            if len(smplx) != 1:
                raise ValueError(f"{path}: expected one person, got {len(smplx)}")

            smplx = smplx[0]
            smpl = self.smplx2smpl @ smplx

            vertices.append(smpl)

        vertices = np.stack(vertices).astype(np.float32)

        frame_ids = np.arange(len(files))

        conv = np.asarray([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1],
        ])

        vertices = vertices @ conv

        return TrackerPrediction(
            tracker_name=self.tracker_name,
            subject=subject,
            sequence=sequence,
            vertices=vertices,
            frame_ids=frame_ids,
            source_path=prediction_path,
        )