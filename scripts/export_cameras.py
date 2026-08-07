from pathlib import Path
import argparse

import cv2
import numpy as np
import trimesh


CAMERA_NAMES = ("00", "01", "02", "03", "04")
DEFAULT_YFOV_DEG = 45.0


def load_mesh(mesh_path):
    """Load an OBJ as a single trimesh.Trimesh."""
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

    if not np.isfinite(size) or size <= 0.0:
        raise RuntimeError(f"Invalid mesh size for {mesh_path}: {size}")

    return center, size


def get_intrinsics(width, height, yfov_rad):
    """Create the intrinsic matrix for a vertical field of view."""
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
    """
    Position a camera on a circle around the mesh.

    Convention:
      0°   = front
      90°  = right
      -90° = left
      180° = back
    """
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
    """Return a camera-to-world pose in OpenGL/pyrender convention."""
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
        raise ValueError("The up vector must not be parallel to the viewing direction.")
    right /= right_norm

    true_up = np.cross(right, forward)

    pose = np.eye(4, dtype=np.float64)
    pose[:3, 0] = right
    pose[:3, 1] = true_up
    pose[:3, 2] = -forward
    pose[:3, 3] = camera_position
    return pose


def get_extrinsics(center, size, azimuth_deg):
    """
    Create OpenCV extrinsics for one camera.

    Convention:
        X_camera = Rot @ X_world + T
    """
    camera_position = get_camera_position(center, size, azimuth_deg)
    target = center + np.array([0.0, 0.05 * size, 0.0], dtype=np.float64)

    # Camera-to-world in OpenGL coordinates.
    T_wc_gl = look_at(camera_position, target)

    # World-to-camera in OpenGL coordinates.
    T_cw_gl = np.linalg.inv(T_wc_gl)

    # OpenGL camera coordinates: x right, y up, z backward.
    # OpenCV camera coordinates: x right, y down, z forward.
    gl_to_cv = np.diag([1.0, -1.0, -1.0, 1.0])
    T_cw_cv = gl_to_cv @ T_cw_gl

    rotation_matrix = T_cw_cv[:3, :3].astype(np.float64)
    translation = T_cw_cv[:3, 3:4].astype(np.float64)
    rotation_vector, _ = cv2.Rodrigues(rotation_matrix)

    return {
        "R": rotation_vector.astype(np.float64),
        "Rot": rotation_matrix,
        "T": translation,
    }


def write_opencv_matrix(file, key, matrix):
    """Write a matrix using OpenCV YAML syntax."""
    matrix = np.asarray(matrix, dtype=np.float64)

    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    if matrix.ndim != 2:
        raise ValueError(f"{key} must be two-dimensional, got {matrix.shape}")

    data = ", ".join(f"{value:.12g}" for value in matrix.reshape(-1))

    file.write(f"{key}: !!opencv-matrix\n")
    file.write(f"  rows: {matrix.shape[0]}\n")
    file.write(f"  cols: {matrix.shape[1]}\n")
    file.write("  dt: d\n")
    file.write(f"  data: [{data}]\n")


def write_intrinsics(output_path, camera_names, K, width, height):
    with output_path.open("w", encoding="utf-8", newline="\n") as file:
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

    with output_path.open("w", encoding="utf-8", newline="\n") as file:
        file.write("%YAML:1.0\n")
        file.write("---\n")
        file.write("names:\n")
        for name in camera_names:
            file.write(f'  - "{name}"\n')
        file.write("\n")

        for name in camera_names:
            write_opencv_matrix(file, f"R_{name}", cameras[name]["R"])
        file.write("\n")

        for name in camera_names:
            write_opencv_matrix(file, f"T_{name}", cameras[name]["T"])
        file.write("\n")

        for name in camera_names:
            write_opencv_matrix(file, f"Rot_{name}", cameras[name]["Rot"])


def export_camera_parameters(mesh_path, output_dir, azimuths, width, height, yfov_deg):
    if len(azimuths) != 5:
        raise ValueError(f"Exactly five angles are required, got {len(azimuths)}.")
    if width <= 0 or height <= 0:
        raise ValueError("Width and height must be positive.")
    if not 0.0 < yfov_deg < 180.0:
        raise ValueError("yfov_deg must be between 0 and 180 degrees.")

    mesh_path = Path(mesh_path)
    output_dir = Path(output_dir)

    if not mesh_path.is_file():
        raise FileNotFoundError(f"Mesh not found: {mesh_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    center, size = get_mesh_center_and_size(mesh_path)
    K = get_intrinsics(width, height, np.deg2rad(yfov_deg))

    cameras = {}
    for name, azimuth_deg in zip(CAMERA_NAMES, azimuths):
        cameras[name] = get_extrinsics(center, size, azimuth_deg)
        print(f"Camera {name}: {azimuth_deg:g}°")

    intri_path = output_dir / "intri.yml"
    extri_path = output_dir / "extri.yml"

    write_intrinsics(intri_path, CAMERA_NAMES, K, width, height)
    write_extrinsics(extri_path, cameras)

    print(f"Saved: {intri_path}")
    print(f"Saved: {extri_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export intri.yml and extri.yml for four cameras."
    )
    parser.add_argument(
        "--mesh_path",
        required=True,
        help="Path to the reference OBJ mesh used to determine center and size.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory in which intri.yml and extri.yml are written.",
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
        help="Five camera angles in degrees, assigned to 00, 01, 02, 03, and 04.",
    )
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument(
        "--yfov_deg",
        type=float,
        default=DEFAULT_YFOV_DEG,
        help=f"Vertical field of view in degrees. Default: {DEFAULT_YFOV_DEG:g}.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    export_camera_parameters(
        mesh_path=args.mesh_path,
        output_dir=args.output_dir,
        azimuths=args.azimuths,
        width=args.width,
        height=args.height,
        yfov_deg=args.yfov_deg,
    )


if __name__ == "__main__":
    main()