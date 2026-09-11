import argparse
from pathlib import Path

import cv2
import numpy as np
import trimesh


CAMERA_NAMES = ("00", "01", "02", "03", "04")
DEFAULT_YFOV_DEG = 45.0


def load_mesh(mesh_path):
    mesh = trimesh.load(mesh_path, process=False)

    if isinstance(mesh, trimesh.Scene):
        geometries = list(mesh.geometry.values())

        if not geometries:
            raise RuntimeError(f"No geometry found in {mesh_path}")

        mesh = trimesh.util.concatenate(geometries)

    if not isinstance(mesh, trimesh.Trimesh):
        raise RuntimeError(f"Unsupported mesh type: {type(mesh)}")

    return mesh


def get_mesh_center_and_size(mesh_path):
    mesh = load_mesh(mesh_path)

    center = mesh.bounds.mean(axis=0).astype(np.float64)
    size = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))

    if not np.all(np.isfinite(center)):
        raise RuntimeError(f"Invalid mesh center for {mesh_path}: {center}")

    if not np.isfinite(size) or size <= 0.0:
        raise RuntimeError(f"Invalid mesh size for {mesh_path}: {size}")

    return center, size


def get_intrinsics(width, height, yfov_rad):
    fy = height / (2.0 * np.tan(yfov_rad / 2.0))
    fx = fy

    cx = width / 2.0
    cy = height / 2.0

    return np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def get_camera_position(center, size, azimuth_deg):
    radius = 1.45 * size
    height_offset = 0.25 * size
    azimuth_rad = np.deg2rad(azimuth_deg)

    offset = np.array(
        [
            np.sin(azimuth_rad) * radius,
            height_offset,
            np.cos(azimuth_rad) * radius,
        ],
        dtype=np.float64,
    )

    return center + offset


def look_at(camera_position, target, up=None):
    if up is None:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    camera_position = np.asarray(camera_position, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)

    forward = target - camera_position
    forward_norm = np.linalg.norm(forward)

    if forward_norm < 1e-12:
        raise ValueError("Camera position and target must be different.")

    forward /= forward_norm

    right = np.cross(forward, up)
    right_norm = np.linalg.norm(right)

    if right_norm < 1e-12:
        raise ValueError(
            "The up vector must not be parallel to the viewing direction."
        )

    right /= right_norm
    true_up = np.cross(right, forward)

    pose = np.eye(4, dtype=np.float64)
    pose[:3, 0] = right
    pose[:3, 1] = true_up
    pose[:3, 2] = -forward
    pose[:3, 3] = camera_position

    return pose


def get_extrinsics(center, size, azimuth_deg):
    camera_position = get_camera_position(
        center=center,
        size=size,
        azimuth_deg=azimuth_deg,
    )

    target = center + np.array(
        [0.0, 0.05 * size, 0.0],
        dtype=np.float64,
    )

    # Convert OpenGL camera coordinates to OpenCV coordinates.
    camera_to_world_gl = look_at(camera_position, target)
    world_to_camera_gl = np.linalg.inv(camera_to_world_gl)

    gl_to_cv = np.diag([1.0, -1.0, -1.0, 1.0])
    world_to_camera_cv = gl_to_cv @ world_to_camera_gl

    rotation_matrix = world_to_camera_cv[:3, :3].astype(np.float64)
    translation = world_to_camera_cv[:3, 3:4].astype(np.float64)

    rotation_vector, _ = cv2.Rodrigues(rotation_matrix)

    return {
        "R": rotation_vector.astype(np.float64),
        "Rot": rotation_matrix,
        "T": translation,
    }


def write_opencv_matrix(file, key, matrix):
    matrix = np.asarray(matrix, dtype=np.float64)

    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)

    if matrix.ndim != 2:
        raise ValueError(
            f"{key} must be two-dimensional, got shape {matrix.shape}"
        )

    data = ", ".join(
        f"{value:.12g}"
        for value in matrix.reshape(-1)
    )

    file.write(f"{key}: !!opencv-matrix\n")
    file.write(f"  rows: {matrix.shape[0]}\n")
    file.write(f"  cols: {matrix.shape[1]}\n")
    file.write("  dt: d\n")
    file.write(f"  data: [{data}]\n")


def write_intrinsics(
    output_path,
    camera_names,
    K,
    width,
    height,
):
    with output_path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as file:
        file.write("%YAML:1.0\n")
        file.write("---\n")
        file.write("names:\n")

        for name in camera_names:
            file.write(f'  - "{name}"\n')

        file.write("\n")

        for name in camera_names:
            write_opencv_matrix(file, f"K_{name}", K)

        file.write("\n")

        distortion = np.zeros((1, 5), dtype=np.float64)

        for name in camera_names:
            write_opencv_matrix(file, f"dist_{name}", distortion)

        file.write("\n")

        for name in camera_names:
            file.write(f"H_{name}: {height}\n")

        file.write("\n")

        for name in camera_names:
            file.write(f"W_{name}: {width}\n")


def write_extrinsics(output_path, cameras):
    camera_names = list(cameras.keys())

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as file:
        file.write("%YAML:1.0\n")
        file.write("---\n")
        file.write("names:\n")

        for name in camera_names:
            file.write(f'  - "{name}"\n')

        file.write("\n")

        for name in camera_names:
            write_opencv_matrix(
                file,
                f"R_{name}",
                cameras[name]["R"],
            )

        file.write("\n")

        for name in camera_names:
            write_opencv_matrix(
                file,
                f"T_{name}",
                cameras[name]["T"],
            )

        file.write("\n")

        for name in camera_names:
            write_opencv_matrix(
                file,
                f"Rot_{name}",
                cameras[name]["Rot"],
            )


def export_camera_parameters(
    mesh_path,
    output_dir,
    azimuths,
    width,
    height,
    yfov_deg,
):
    if len(azimuths) != len(CAMERA_NAMES):
        raise ValueError(
            f"Exactly {len(CAMERA_NAMES)} angles are required, "
            f"got {len(azimuths)}."
        )

    if width <= 0 or height <= 0:
        raise ValueError("Width and height must be positive.")

    if not 0.0 < yfov_deg < 180.0:
        raise ValueError(
            "yfov_deg must be between 0 and 180 degrees."
        )

    mesh_path = Path(mesh_path)
    output_dir = Path(output_dir)

    if not mesh_path.is_file():
        raise FileNotFoundError(f"Mesh not found: {mesh_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    center, size = get_mesh_center_and_size(mesh_path)

    K = get_intrinsics(
        width=width,
        height=height,
        yfov_rad=np.deg2rad(yfov_deg),
    )

    cameras = {}

    for name, azimuth_deg in zip(CAMERA_NAMES, azimuths):
        cameras[name] = get_extrinsics(
            center=center,
            size=size,
            azimuth_deg=azimuth_deg,
        )

        print(f"    Camera {name}: {azimuth_deg:g}°")

    intri_path = output_dir / "intri.yml"
    extri_path = output_dir / "extri.yml"

    write_intrinsics(
        output_path=intri_path,
        camera_names=CAMERA_NAMES,
        K=K,
        width=width,
        height=height,
    )

    write_extrinsics(
        output_path=extri_path,
        cameras=cameras,
    )

    return intri_path, extri_path


def find_sequence_meshes(
    sequences_root,
    pose_folder="posed",
):
    sequences_root = Path(sequences_root)

    if not sequences_root.exists():
        raise RuntimeError(
            f"Input folder does not exist: {sequences_root}"
        )

    if not sequences_root.is_dir():
        raise RuntimeError(
            f"Input path is not a directory: {sequences_root}"
        )

    sequence_meshes = []

    for sequence_dir in sorted(sequences_root.iterdir()):
        if not sequence_dir.is_dir():
            continue

        found_any = False

        pose_dirs = sorted(
            sequence_dir.glob(f"*/{pose_folder}")
        )

        for pose_dir in pose_dirs:
            if not pose_dir.is_dir():
                continue

            obj_files = sorted(
                path
                for path in pose_dir.iterdir()
                if path.is_file() and path.suffix.lower() == ".obj"
            )

            if not obj_files:
                print(f"Skipping {pose_dir}: no .obj files found")
                continue

            animation_name = pose_dir.parent.name
            reference_mesh = obj_files[0]

            sequence_meshes.append(
                (
                    sequence_dir.name,
                    animation_name,
                    reference_mesh,
                )
            )

            found_any = True

        if not found_any:
            print(
                f"Skipping {sequence_dir.name}: "
                f"no .obj files found in */{pose_folder}/"
            )

    return sequence_meshes


def batch_export_camera_parameters(
    sequences_root,
    output_root,
    azimuths,
    width,
    height,
    yfov_deg,
    pose_folder="posed",
    skip_existing=False,
):
    sequences_root = Path(sequences_root)
    output_root = Path(output_root)

    output_root.mkdir(parents=True, exist_ok=True)

    sequence_meshes = find_sequence_meshes(
        sequences_root=sequences_root,
        pose_folder=pose_folder,
    )

    if not sequence_meshes:
        raise RuntimeError(
            f"No valid sequences found in {sequences_root}"
        )

    total = len(sequence_meshes)
    print(f"Found {total} animations")

    generated_count = 0
    skipped_count = 0

    for index, (
        sequence_name,
        animation_name,
        reference_mesh,
    ) in enumerate(sequence_meshes, start=1):
        animation_output_dir = (
            output_root
            / sequence_name
            / animation_name
        )

        intri_path = animation_output_dir / "intri.yml"
        extri_path = animation_output_dir / "extri.yml"

        if (
            skip_existing
            and intri_path.is_file()
            and extri_path.is_file()
        ):
            print(
                f"[{index}/{total}] "
                f"Skipping {sequence_name}/{animation_name}: "
                f"camera parameters already exist"
            )

            skipped_count += 1
            continue

        print(
            f"[{index}/{total}] "
            f"Processing {sequence_name}/{animation_name}"
        )
        print(f"  reference mesh: {reference_mesh}")
        print(f"  output dir:     {animation_output_dir}")

        generated_intri, generated_extri = export_camera_parameters(
            mesh_path=reference_mesh,
            output_dir=animation_output_dir,
            azimuths=azimuths,
            width=width,
            height=height,
            yfov_deg=yfov_deg,
        )

        print(f"  saved: {generated_intri}")
        print(f"  saved: {generated_extri}")

        generated_count += 1

    print()
    print("Batch camera export finished")
    print(f"Generated: {generated_count}")
    print(f"Skipped:   {skipped_count}")
    print(f"Total:     {total}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Export intri.yml and extri.yml for all mesh animations "
            "inside a sequence directory."
        )
    )

    parser.add_argument(
        "--sequences_root",
        required=True,
        help="Root directory containing sequence and animation folders.",
    )
    parser.add_argument(
        "--output_root",
        required=True,
        help=(
            "Root output directory. Results are written to "
            "<output_root>/<sequence>/<animation>/."
        ),
    )
    parser.add_argument(
        "--pose_folder",
        default="posed",
        help="Name of the folder containing OBJ meshes. Default: posed.",
    )
    parser.add_argument(
        "--azimuths",
        type=float,
        nargs=5,
        required=True,
        metavar=(
            "ANGLE_00",
            "ANGLE_01",
            "ANGLE_02",
            "ANGLE_03",
            "ANGLE_04",
        ),
        help=(
            "Five camera angles in degrees, assigned to cameras "
            "00, 01, 02, 03, and 04."
        ),
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Image width. Default: 512.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Image height. Default: 512.",
    )
    parser.add_argument(
        "--yfov_deg",
        type=float,
        default=DEFAULT_YFOV_DEG,
        help=(
            "Vertical field of view in degrees. "
            f"Default: {DEFAULT_YFOV_DEG:g}."
        ),
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help=(
            "Skip animations for which both intri.yml and extri.yml "
            "already exist."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    batch_export_camera_parameters(
        sequences_root=args.sequences_root,
        output_root=args.output_root,
        azimuths=args.azimuths,
        width=args.width,
        height=args.height,
        yfov_deg=args.yfov_deg,
        pose_folder=args.pose_folder,
        skip_existing=args.skip_existing,
    )


if __name__ == "__main__":
    main()