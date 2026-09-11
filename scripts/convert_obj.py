import argparse
from pathlib import Path

import numpy as np
from smplx import SMPL


def write_obj(
    output_path: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        for x, y, z in vertices:
            file.write(f"v {x:.8f} {y:.8f} {z:.8f}\n")

        # OBJ indices are 1-based
        for i, j, k in faces:
            file.write(f"f {i + 1} {j + 1} {k + 1}\n")


def npy_to_obj_sequence(
    npy_path: Path,
    output_dir: Path,
    smpl_model_dir: Path,
    gender: str = "male",
) -> None:
    vertices = np.load(npy_path)

    if vertices.ndim == 2:
        vertices = vertices[None, ...]

    if vertices.ndim != 3 or vertices.shape[-1] != 3:
        raise ValueError(
            f"Unexpected vertex shape: {vertices.shape}. "
            "Expected (V, 3) or (T, V, 3)."
        )

    smpl = SMPL(
        model_path=str(smpl_model_dir),
        gender=gender,
        batch_size=1,
    )

    faces = np.asarray(smpl.faces, dtype=np.int64)

    print("Vertices:", vertices.shape)
    print("Faces:", faces.shape)

    output_dir.mkdir(parents=True, exist_ok=True)

    for frame_idx, frame_vertices in enumerate(vertices):
        output_path = output_dir / f"frame_{frame_idx:03d}.obj"

        write_obj(
            output_path=output_path,
            vertices=frame_vertices,
            faces=faces,
        )

        print(f"\rSaved: {frame_idx + 1}/{len(vertices)}", end="")

    print(f"\nOBJ sequence saved to:\n{output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--npy_path",
        type=Path,
        required=True,
        help="Path to the gt_body_vertices_*.npy file",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Output directory for OBJ files",
    )
    parser.add_argument(
        "--smpl_model_dir",
        type=Path,
        default=Path(r"E:\DLHM\Data\smpl\models"),
    )
    parser.add_argument(
        "--gender",
        choices=["male", "female", "neutral"],
        default="male",
    )

    args = parser.parse_args()

    npy_to_obj_sequence(
        npy_path=args.npy_path,
        output_dir=args.output_dir,
        smpl_model_dir=args.smpl_model_dir,
        gender=args.gender,
    )