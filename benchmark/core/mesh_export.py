from pathlib import Path
import pickle
import numpy as np

from benchmark.core.types import GTSequence


class MeshExporter:
    def __init__(self, smpl_male_path: Path, smpl_female_path: Path, export_path: Path):
        self.smpl_male_path = smpl_male_path
        self.smpl_female_path = smpl_female_path
        self.export_path = export_path

    def export_obj(self, vertices, faces, output_path):
        vertices = np.asarray(vertices)
        faces = np.asarray(faces)

        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError(f"Vertices are supposed to have shape (N, 3) but have: {vertices.shape}")

        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError(f"Faces are supposed to have shape (M, 3) but have: {faces.shape}")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as file:
            for x, y, z in vertices:
                file.write(f"v {x:.8f} {y:.8f} {z:.8f}\n")

            for i, j, k in faces:
                file.write(f"f {i + 1} {j + 1} {k + 1}\n")

    def export_mesh(self, vertices, faces, output_path):
        for frame_index, frame_vertices in enumerate(vertices):
            frame_output_path = output_path / f"frame_{frame_index:03d}.obj"

            self.export_obj(vertices=frame_vertices, faces=faces, output_path=frame_output_path)

            print(f"[{frame_index + 1}/{len(vertices)}] OBJ saved: {frame_output_path}")

    def export_meshes(self, gt: GTSequence, result):
        if gt.gender == "male":
            smpl_model_path = self.smpl_male_path
        elif gt.gender == "female":
            smpl_model_path = self.smpl_female_path
        else:
            raise ValueError(f"Unknown gender: {gt.gender}")

        with smpl_model_path.open("rb") as file:
            smpl_data = pickle.load(file, encoding="latin1")

        faces = np.asarray(smpl_data["f"], dtype=np.int64)

        output_directory = self.export_path / gt.subject / gt.sequence

        gt_v = result["gt_v"]
        pred_v_initial_aligned = result["pred_v_initial_aligned"]
        pred_v_centroid_aligned = result["pred_v_centroid_aligned"]
        pred_v_pa_aligned = result["pred_v_pa_aligned"]

        self.export_mesh(vertices=gt_v, faces=faces, output_path=output_directory / "gt_v")
        self.export_mesh(vertices=pred_v_initial_aligned, faces=faces, output_path=output_directory / "pred_v_initial_aligned")
        self.export_mesh(vertices=pred_v_centroid_aligned, faces=faces, output_path=output_directory / "pred_v_centroid_aligned")
        self.export_mesh(vertices=pred_v_pa_aligned, faces=faces, output_path=output_directory / "pred_v_pa_aligned")

        print("\nExport finished.\n")