from benchmark.trackers.wham import WhamLoader
from pathlib import Path
import numpy as np
import pickle


def export_obj(
    vertices: np.ndarray,
    faces: np.ndarray,
    output_path: Path,
) -> None:
    """Exportiert ein Dreiecks-Mesh als OBJ-Datei."""
    vertices = np.asarray(vertices)
    faces = np.asarray(faces)

    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(
            f"Vertices müssen die Form (N, 3) haben, erhalten: {vertices.shape}"
        )

    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(
            f"Faces müssen die Form (M, 3) haben, erhalten: {faces.shape}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        for x, y, z in vertices:
            file.write(f"v {x:.8f} {y:.8f} {z:.8f}\n")

        # OBJ-Indizes beginnen bei 1, NumPy-Indizes bei 0.
        for i, j, k in faces:
            file.write(f"f {i + 1} {j + 1} {k + 1}\n")


def get_vertices() -> None:
    loader = WhamLoader()

    prediction = loader.load(
        Path(
            "E:/DLHM/results/WHAM/FullRun/00032/"
            "00032_longshort_flying_eagle_front/wham_output.pkl"
        ),
        "00032",
        "longshort_flying_eagle",
    )

    # Erwartete Form: (Anzahl_Frames, Anzahl_Vertices, 3)
    vertices_all = np.asarray(prediction.vertices)

    print(f"Vertices insgesamt: {vertices_all.shape}")

    if vertices_all.ndim != 3 or vertices_all.shape[-1] != 3:
        raise ValueError(
            "prediction.vertices muss die Form "
            f"(Frames, Vertices, 3) haben, erhalten: {vertices_all.shape}"
        )

    print(f"Converted vertices: {vertices_all}")


    smpl_model_path = Path(
        "E:/DLHM/Data/smpl/models/SMPL_MALE.pkl"
    )

    with smpl_model_path.open("rb") as file:
        smpl_data = pickle.load(file, encoding="latin1")

    faces = np.asarray(smpl_data["f"], dtype=np.int64)

    output_directory = Path("E:/DLHM/mesh_exp")

    number_of_frames = vertices_all.shape[0]
    filename_digits = max(4, len(str(number_of_frames - 1)))

    for frame_index, frame_vertices in enumerate(vertices_all):
        output_path = output_directory / (
            f"frame_{frame_index:0{filename_digits}d}.obj"
        )

        export_obj(
            vertices=frame_vertices,
            faces=faces,
            output_path=output_path,
        )

        print(
            f"[{frame_index + 1}/{number_of_frames}] "
            f"OBJ gespeichert: {output_path}"
        )

    print(f"\nExport abgeschlossen.")
    print(f"Exportierte Frames: {number_of_frames}")
    print(f"Vertices pro Frame: {vertices_all.shape[1]}")
    print(f"Faces pro Frame: {faces.shape[0]}")
    print(f"Ausgabeordner: {output_directory}")


if __name__ == "__main__":
    get_vertices()