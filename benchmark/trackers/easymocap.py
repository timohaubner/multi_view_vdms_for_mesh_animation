from pathlib import Path
import joblib
import trimesh
import numpy as np

from benchmark.core.types import TrackerPrediction
from benchmark.trackers.base import BaseTrackerLoader

class EasyMocapLoader(BaseTrackerLoader):
    tracker_name = "easymocap"

    def load(self, prediction_paths: list[Path], subject: str, sequence: str) -> TrackerPrediction:

        obj_files = []

        for i, prediction_path in enumerate(prediction_paths):
            if not prediction_path.exists():
                raise FileNotFoundError(prediction_path)

            files = sorted(prediction_path.glob("*.obj"))

            if i == 0:
                obj_files.extend(files)
            else:
                obj_files.extend(files[1:])

        vertices_per_frame = []

        for obj_path in obj_files:
            mesh = trimesh.load(obj_path, process=False)

            verts = np.asarray(mesh.vertices, dtype=np.float32)
            vertices_per_frame.append(verts)

        vertices = np.stack(vertices_per_frame, axis=0)

        frame_ids = np.arange(len(obj_files)) * 3

        return TrackerPrediction(
            tracker_name=self.tracker_name,
            subject=subject,
            sequence=sequence,
            vertices=vertices,
            frame_ids=frame_ids,
            source_path=prediction_paths[0],
        )

'''
    def load(self, prediction_paths: Path, subject: str, sequence: str) -> TrackerPrediction:
        if not prediction_paths.exists():
            raise FileNotFoundError(prediction_paths)

        obj_files = sorted(prediction_paths.glob("*.obj"))
        vertices_per_frame = []

        for i, obj_path in enumerate(obj_files):
            mesh = trimesh.load(obj_path, process=False)

            verts = np.asarray(mesh.vertices, dtype=np.float32)
            vertices_per_frame.append(verts)

        vertices = np.stack(vertices_per_frame, axis=0)

        frame_ids = np.arange(len(obj_files)) * 3

        return TrackerPrediction(
            tracker_name=self.tracker_name,
            subject=subject,
            sequence=sequence,
            vertices=vertices,
            frame_ids=frame_ids,
            source_path=prediction_paths,
        )
'''