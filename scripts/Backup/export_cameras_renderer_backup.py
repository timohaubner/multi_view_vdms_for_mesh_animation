#!/usr/bin/env python3
from pathlib import Path
import argparse
import json
import numpy as np
import trimesh
import cv2


def get_intrinsics(width, height, yfov):
    """
    Intrinsic camera matrix K for pyrender.PerspectiveCamera(yfov=...).

    Assumes:
      - square pixels
      - principal point at image center
      - pyrender/OpenGL vertical field of view
    """
    fy = height / (2.0 * np.tan(yfov / 2.0))
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
    Same convention as your renderer:

      0°    = front
      90°   = right
      -90°  = left
      180°  = back
    """
    radius = 1.45 * size
    height_offset = 0.25 * size

    azimuth_rad = np.deg2rad(azimuth_deg)

    x = np.sin(azimuth_rad) * radius
    z = np.cos(azimuth_rad) * radius
    y = height_offset

    return center + np.array([x, y, z], dtype=np.float64)


def look_at(camera_pos, target, up=np.array([0.0, 1.0, 0.0])):
    """
    Camera-to-world pose in pyrender/OpenGL convention.

    OpenGL/pyrender camera:
      x right
      y up
      camera looks along -Z
    """
    camera_pos = np.asarray(camera_pos, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)

    forward = target - camera_pos
    forward /= np.linalg.norm(forward)

    right = np.cross(forward, up)
    right /= np.linalg.norm(right)

    true_up = np.cross(right, forward)

    T_wc = np.eye(4, dtype=np.float64)
    T_wc[:3, 0] = right
    T_wc[:3, 1] = true_up
    T_wc[:3, 2] = -forward
    T_wc[:3, 3] = camera_pos

    return T_wc


def get_mesh_bounds(first_mesh_path):
    mesh = trimesh.load(first_mesh_path, process=False)

    if mesh.bounds is None:
        raise RuntimeError(f"Could not read mesh bounds from: {first_mesh_path}")

    center = mesh.bounds.mean(axis=0).astype(np.float64)
    size = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))

    if not np.isfinite(size) or size <= 0:
        raise RuntimeError(f"Invalid mesh size from: {first_mesh_path}")

    return center, size


def get_camera_calibration(
    first_mesh_path,
    width,
    height,
    azimuth_deg,
    yfov=np.pi / 4.0,
    world_scale_to_meters=1.0,
):
    """
    Returns EasyMocap/OpenCV-style camera matrices.

    Important convention:
      X_cam = R @ X_world + T

    This matches EasyMocap/OpenCV extrinsics.

    The renderer works in pyrender/OpenGL convention internally.
    We convert from OpenGL camera coordinates to OpenCV camera coordinates:

      OpenGL camera: x right, y up,   z backward, camera looks along -Z
      OpenCV camera: x right, y down, z forward
    """
    center, size = get_mesh_bounds(first_mesh_path)

    camera_pos = get_camera_position(center, size, azimuth_deg)
    target = center + np.array([0.0, 0.05 * size, 0.0], dtype=np.float64)

    T_wc_gl = look_at(
        camera_pos=camera_pos,
        target=target,
        up=np.array([0.0, 1.0, 0.0], dtype=np.float64),
    )

    # World-to-camera in OpenGL/pyrender convention.
    T_cw_gl = np.linalg.inv(T_wc_gl)

    # Convert OpenGL camera coordinates to OpenCV camera coordinates.
    gl_to_cv = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    T_cw_cv = gl_to_cv @ T_cw_gl

    R = T_cw_cv[:3, :3].astype(np.float64)
    T = T_cw_cv[:3, 3:4].astype(np.float64)

    # If your mesh/world units are not meters, convert translation to meters.
    # Example:
    #   mesh in meters:      world_scale_to_meters = 1.0
    #   mesh in centimeters: world_scale_to_meters = 0.01
    #   mesh in millimeters: world_scale_to_meters = 0.001
    T = T * float(world_scale_to_meters)

    K = get_intrinsics(width=width, height=height, yfov=yfov)
    dist = np.zeros((1, 5), dtype=np.float64)

    Rvec, _ = cv2.Rodrigues(R)
    P = K @ np.hstack([R, T])

    return {
        "K": K,
        "dist": dist,
        "R": Rvec.astype(np.float64),      # rotation vector, EasyMocap key R_xx
        "Rot": R.astype(np.float64),       # rotation matrix, useful/debug, EasyMocap often writes Rot_xx too
        "T": T.astype(np.float64),
        "P": P.astype(np.float64),
        "T_wc_gl": T_wc_gl,
        "T_cw_gl": T_cw_gl,
        "T_cw_cv": T_cw_cv,
        "camera_position": camera_pos * float(world_scale_to_meters),
        "target": target * float(world_scale_to_meters),
        "center": center * float(world_scale_to_meters),
        "size": size * float(world_scale_to_meters),
        "azimuth_deg": float(azimuth_deg),
        "yfov": float(yfov),
        "width": int(width),
        "height": int(height),
        "world_scale_to_meters": float(world_scale_to_meters),
    }


def write_opencv_matrix(f, key, mat):
    """
    Write one matrix in OpenCV YAML format.
    """
    mat = np.asarray(mat, dtype=np.float64)

    if mat.ndim == 1:
        mat = mat.reshape(-1, 1)

    f.write(f"{key}: !!opencv-matrix\n")
    f.write(f"  rows: {mat.shape[0]}\n")
    f.write(f"  cols: {mat.shape[1]}\n")
    f.write("  dt: d\n")
    f.write("  data: [{}]\n".format(", ".join(f"{x:.12g}" for x in mat.reshape(-1))))


def write_easymocap_yml(output_dir, cameras, width, height, write_rot_matrix=True):
    """
    Writes:
      output_dir/intri.yml
      output_dir/extri.yml

    cameras dict format:
      {
        "01": {
          "K": ...,
          "dist": ...,
          "R": rotation_vector,
          "Rot": rotation_matrix,
          "T": translation_vector,
        },
        ...
      }
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    intri_path = output_dir / "intri.yml"
    extri_path = output_dir / "extri.yml"

    cam_names = list(cameras.keys())

    with open(intri_path, "w", encoding="utf-8") as f:
        f.write("%YAML:1.0\n")
        f.write("---\n")
        f.write("names:\n")
        for name in cam_names:
            f.write(f'  - "{name}"\n')

        for name in cam_names:
            cam = cameras[name]
            write_opencv_matrix(f, f"K_{name}", cam["K"])
            write_opencv_matrix(f, f"dist_{name}", cam["dist"])
            f.write(f"H_{name}: {int(height)}\n")
            f.write(f"W_{name}: {int(width)}\n")

    with open(extri_path, "w", encoding="utf-8") as f:
        f.write("%YAML:1.0\n")
        f.write("---\n")
        f.write("names:\n")
        for name in cam_names:
            f.write(f'  - "{name}"\n')

        for name in cam_names:
            cam = cameras[name]
            write_opencv_matrix(f, f"R_{name}", cam["R"])
            write_opencv_matrix(f, f"T_{name}", cam["T"])

            # EasyMocap camera_utils often supports/writes Rot_xx as matrix too.
            # Keeping it is useful for debugging and usually harmless.
            if write_rot_matrix:
                write_opencv_matrix(f, f"Rot_{name}", cam["Rot"])

    print(f"Saved: {intri_path}")
    print(f"Saved: {extri_path}")


def write_debug_json(output_dir, cameras):
    """
    Writes human-readable debug values.
    This is not needed by EasyMocap, but useful to inspect the camera setup.
    """
    output_dir = Path(output_dir)
    debug_path = output_dir / "camera_debug.json"

    serializable = {}

    for name, cam in cameras.items():
        serializable[name] = {}
        for key, value in cam.items():
            if isinstance(value, np.ndarray):
                serializable[name][key] = value.tolist()
            elif isinstance(value, (np.floating, np.integer)):
                serializable[name][key] = value.item()
            else:
                serializable[name][key] = value

    with open(debug_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)

    print(f"Saved: {debug_path}")


def find_first_obj(mesh_dir):
    mesh_dir = Path(mesh_dir)
    obj_files = sorted(mesh_dir.glob("*.obj"))

    if not obj_files:
        raise RuntimeError(f"No .obj files found in: {mesh_dir}")

    return obj_files[0]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export EasyMocap intri.yml/extri.yml from the same camera setup as the pyrender renderer."
    )

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--first_mesh",
        type=str,
        help="Path to one .obj mesh frame. This is used to compute center and scale.",
    )
    src.add_argument(
        "--mesh_dir",
        type=str,
        help="Directory containing .obj mesh frames. The first sorted .obj is used.",
    )

    parser.add_argument(
        "--output_dir",
        required=True,
        type=str,
        help="Directory where intri.yml and extri.yml will be written.",
    )

    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)

    parser.add_argument(
        "--yfov",
        type=float,
        default=np.pi / 4.0,
        help="Vertical field of view in radians. Default matches your renderer: pi/4.",
    )

    parser.add_argument(
        "--azimuths",
        type=float,
        nargs="+",
        default=[0.0],
        help="Camera azimuths in degrees. Example: --azimuths 0 90 180 270",
    )

    parser.add_argument(
        "--cam_names",
        type=str,
        nargs="+",
        default=None,
        help='Camera names matching video filenames. Example: --cam_names 01 02 03 04 for videos/01.mp4 etc.',
    )

    parser.add_argument(
        "--world_scale_to_meters",
        type=float,
        default=1.0,
        help=(
            "Scale applied to translation T. "
            "Use 1.0 if meshes are in meters, 0.01 for centimeters, 0.001 for millimeters."
        ),
    )

    parser.add_argument(
        "--no_rot_matrix",
        action="store_true",
        help="Do not write Rot_xx rotation matrices, only R_xx rotation vectors and T_xx.",
    )

    parser.add_argument(
        "--debug_json",
        action="store_true",
        help="Also write camera_debug.json with K/R/T/P and poses.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.first_mesh is not None:
        first_mesh_path = Path(args.first_mesh)
    else:
        first_mesh_path = find_first_obj(args.mesh_dir)

    if not first_mesh_path.exists():
        raise RuntimeError(f"Mesh does not exist: {first_mesh_path}")

    azimuths = args.azimuths

    if args.cam_names is None:
        cam_names = [f"{i + 1:02d}" for i in range(len(azimuths))]
    else:
        cam_names = args.cam_names

    if len(cam_names) != len(azimuths):
        raise RuntimeError(
            f"Number of cam_names must match number of azimuths. "
            f"Got {len(cam_names)} cam_names and {len(azimuths)} azimuths."
        )

    cameras = {}

    print(f"Using mesh: {first_mesh_path}")
    print(f"Image size: {args.width} x {args.height}")
    print(f"yfov: {args.yfov}")
    print(f"world_scale_to_meters: {args.world_scale_to_meters}")

    for cam_name, azimuth in zip(cam_names, azimuths):
        cam = get_camera_calibration(
            first_mesh_path=first_mesh_path,
            width=args.width,
            height=args.height,
            azimuth_deg=azimuth,
            yfov=args.yfov,
            world_scale_to_meters=args.world_scale_to_meters,
        )

        cameras[cam_name] = cam

        print("")
        print(f"Camera {cam_name}")
        print(f"  azimuth_deg: {azimuth}")
        print(f"  K:\n{cam['K']}")
        print(f"  Rvec:\n{cam['R']}")
        print(f"  Rot:\n{cam['Rot']}")
        print(f"  T:\n{cam['T']}")

    write_easymocap_yml(
        output_dir=args.output_dir,
        cameras=cameras,
        width=args.width,
        height=args.height,
        write_rot_matrix=not args.no_rot_matrix,
    )

    if args.debug_json:
        write_debug_json(args.output_dir, cameras)

    print("")
    print("Done.")
    print("For EasyMocap, your videos should use the same camera names, e.g.:")
    print("  videos/01.mp4 -> K_01, dist_01, R_01, T_01")


if __name__ == "__main__":
    main()