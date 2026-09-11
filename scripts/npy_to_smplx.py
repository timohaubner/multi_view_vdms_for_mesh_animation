import argparse
from pathlib import Path

import numpy as np


def write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray):
    with path.open("w") as file:
        for vertex in vertices:
            file.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")

        # OBJ indices are 1-based
        for face in faces:
            a, b, c = face + 1
            file.write(f"f {a} {b} {c}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Directory containing .npy files")
    parser.add_argument("--smplx", required=True, help="Path to SMPLX_NEUTRAL.npz")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = np.load(args.smplx, allow_pickle=True)
    faces = model["f"].astype(np.int64)

    for npy_path in sorted(input_dir.glob("*.npy")):
        vertices = np.load(npy_path)

        if vertices.ndim == 2:
            vertices = vertices[None]

        if vertices.ndim != 3 or vertices.shape[-1] != 3:
            print(f"SKIP {npy_path.name}: shape={vertices.shape}")
            continue

        for person_id, person_vertices in enumerate(vertices):
            name = npy_path.name

            if name.endswith(".png.npy"):
                name = name[:-8]
            else:
                name = npy_path.stem

            output_path = output_dir / f"{name}_person_{person_id:02d}.obj"

            write_obj(output_path, person_vertices, faces)
            print(f"{npy_path.name} -> {output_path}")


if __name__ == "__main__":
    main()