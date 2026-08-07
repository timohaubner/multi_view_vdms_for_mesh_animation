from pathlib import Path
import trimesh
import numpy as np

from benchmark.core.types import TrackerPrediction
from benchmark.trackers.base import BaseTrackerLoader

class DMMRLoader(BaseTrackerLoader):
    tracker_name = "dmmr"

    def load(self, prediction_paths: list[Path], subject: str, sequence: str) -> TrackerPrediction:

        prediction_path = prediction_paths[0]

        if not prediction_path.exists():
            raise FileNotFoundError(prediction_path)

        obj_files = sorted(prediction_path.glob("*.obj"))
        vertices_per_frame = []

        for i, obj_path in enumerate(obj_files):
            mesh = trimesh.load(obj_path, process=False)

            verts = np.asarray(mesh.vertices, dtype=np.float32)
            vertices_per_frame.append(verts)

        vertices = np.stack(vertices_per_frame, axis=0)

        print(vertices.shape)  # z. B. (num_frames, 6890, 3) bei SMPL

        frame_ids = np.arange(len(obj_files)) * 3

        print(frame_ids)

        return TrackerPrediction(
            tracker_name=self.tracker_name,
            subject=subject,
            sequence=sequence,
            vertices=vertices,
            frame_ids=frame_ids,
            source_path=prediction_path,
        )