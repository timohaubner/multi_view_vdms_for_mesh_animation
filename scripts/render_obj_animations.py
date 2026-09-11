#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from pathlib import Path
from typing import Iterable, Sequence

import imageio.v2 as imageio
import numpy as np
import trimesh


DEFAULT_COLORS = (
    (0.20, 0.48, 0.90, 1.0),
    (0.95, 0.42, 0.20, 1.0),
    (0.25, 0.72, 0.42, 1.0),
)


def natural_key(path: Path) -> list[object]:
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def collect_obj_files(
    folder: Path,
    recursive: bool,
) -> list[Path]:
    pattern = "**/*.obj" if recursive else "*.obj"
    files = sorted(folder.glob(pattern), key=natural_key)

    if not files:
        raise FileNotFoundError(f"No OBJ files found: {folder}")

    return files


def load_obj_mesh(path: Path) -> trimesh.Trimesh:
    try:
        loaded = trimesh.load_mesh(path, process=False)
    except (AttributeError, TypeError):
        # Compatibility fallback for older trimesh versions.
        loaded = trimesh.load(
            path,
            force="mesh",
            process=False,
        )

    if isinstance(loaded, trimesh.Scene):
        if hasattr(loaded, "to_mesh"):
            mesh = loaded.to_mesh()
        else:
            mesh = loaded.dump(concatenate=True)
    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    else:
        raise TypeError(
            f"{path} was not loaded as a triangle mesh "
            f"(type: {type(loaded).__name__})."
        )

    if mesh.is_empty or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError(f"Empty or invalid mesh: {path}")

    if not np.isfinite(mesh.vertices).all():
        raise ValueError(f"Mesh contains NaN/Inf coordinates: {path}")

    return mesh


def sampled_paths(
    paths: Sequence[Path],
    step: int,
) -> list[Path]:
    result = list(paths[::step])

    if paths[-1] not in result:
        result.append(paths[-1])

    return result


def compute_global_bounds(
    sequences: Sequence[Sequence[Path]],
    bounds_step: int,
) -> np.ndarray:
    global_min = np.full(
        3,
        np.inf,
        dtype=np.float64,
    )
    global_max = np.full(
        3,
        -np.inf,
        dtype=np.float64,
    )

    files_to_scan = sum(
        len(sampled_paths(sequence, bounds_step))
        for sequence in sequences
    )
    scanned = 0

    print(
        f"Computing shared bounding box from "
        f"{files_to_scan} OBJ files ..."
    )

    for sequence in sequences:
        for path in sampled_paths(
            sequence,
            bounds_step,
        ):
            mesh = load_obj_mesh(path)
            bounds = np.asarray(
                mesh.bounds,
                dtype=np.float64,
            )

            global_min = np.minimum(
                global_min,
                bounds[0],
            )
            global_max = np.maximum(
                global_max,
                bounds[1],
            )

            scanned += 1

            if (
                scanned == files_to_scan
                or scanned % max(1, files_to_scan // 10) == 0
            ):
                print(
                    f"  Bounding box: {scanned}/{files_to_scan}",
                    flush=True,
                )

    if (
        not np.isfinite(global_min).all()
        or not np.isfinite(global_max).all()
    ):
        raise ValueError(
            "The global bounding box could not be determined."
        )

    if np.allclose(global_min, global_max):
        raise ValueError(
            "The global bounding box has no spatial extent."
        )

    return np.vstack(
        (
            global_min,
            global_max,
        )
    )


def bounds_corners(bounds: np.ndarray) -> np.ndarray:
    lower, upper = bounds

    return np.array(
        [
            [x, y, z]
            for x in (lower[0], upper[0])
            for y in (lower[1], upper[1])
            for z in (lower[2], upper[2])
        ],
        dtype=np.float64,
    )


def make_shared_transform(
    bounds: np.ndarray,
    center_mode: str,
    scale: float,
    up_axis: str,
) -> np.ndarray:
    lower, upper = bounds
    center = (lower + upper) * 0.5

    offset = np.zeros(
        3,
        dtype=np.float64,
    )

    if center_mode == "bbox":
        offset = -center
    elif center_mode == "ground":
        up_index = 1 if up_axis == "y" else 2
        offset = -center
        offset[up_index] = -lower[up_index]
    elif center_mode != "none":
        raise ValueError(
            f"Unknown centering mode: {center_mode}"
        )

    transform = np.eye(
        4,
        dtype=np.float64,
    )
    transform[:3, :3] *= scale
    transform[:3, 3] = offset * scale

    return transform


def transform_points(
    points: np.ndarray,
    transform: np.ndarray,
) -> np.ndarray:
    homogeneous = np.column_stack(
        (
            points,
            np.ones(
                len(points),
                dtype=np.float64,
            ),
        )
    )

    return (transform @ homogeneous.T).T[:, :3]


def transform_bounds(
    bounds: np.ndarray,
    transform: np.ndarray,
) -> np.ndarray:
    corners = transform_points(
        bounds_corners(bounds),
        transform,
    )

    return np.vstack(
        (
            corners.min(axis=0),
            corners.max(axis=0),
        )
    )


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))

    if norm < 1e-12:
        raise ValueError(
            "Zero vector cannot be normalized."
        )

    return vector / norm


def view_backward_axis(
    view: str,
    up_axis: str,
) -> tuple[np.ndarray, np.ndarray]:
    if up_axis == "y":
        up = np.array([0.0, 1.0, 0.0])
        directions = {
            "front": np.array([0.0, 0.0, 1.0]),
            "back": np.array([0.0, 0.0, -1.0]),
            "side": np.array([1.0, 0.0, 0.0]),
            "iso": np.array([1.0, 0.45, 1.0]),
        }
    else:
        up = np.array([0.0, 0.0, 1.0])
        directions = {
            "front": np.array([0.0, -1.0, 0.0]),
            "back": np.array([0.0, 1.0, 0.0]),
            "side": np.array([1.0, 0.0, 0.0]),
            "iso": np.array([1.0, -1.0, 0.55]),
        }

    return normalize(directions[view]), up


def camera_basis(
    backward: np.ndarray,
    world_up: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z_axis = normalize(backward)
    x_axis = normalize(
        np.cross(world_up, z_axis)
    )
    y_axis = normalize(
        np.cross(z_axis, x_axis)
    )

    return x_axis, y_axis, z_axis


def fit_camera_to_bounds(
    bounds: np.ndarray,
    width: int,
    height: int,
    yfov_degrees: float,
    view: str,
    up_axis: str,
    margin: float,
) -> tuple[
    np.ndarray,
    float,
    float,
    np.ndarray,
    float,
]:
    corners = bounds_corners(bounds)
    target = bounds.mean(axis=0)

    backward, world_up = view_backward_axis(
        view,
        up_axis,
    )
    x_axis, y_axis, z_axis = camera_basis(
        backward,
        world_up,
    )

    relative = corners - target
    projected_x = relative @ x_axis
    projected_y = relative @ y_axis
    projected_z = relative @ z_axis

    yfov = math.radians(yfov_degrees)
    aspect = width / height

    xfov = 2.0 * math.atan(
        math.tan(yfov * 0.5) * aspect
    )
    tan_x = math.tan(xfov * 0.5)
    tan_y = math.tan(yfov * 0.5)

    required = projected_z + np.maximum(
        np.abs(projected_x) / tan_x,
        np.abs(projected_y) / tan_y,
    )

    diagonal = float(
        np.linalg.norm(
            bounds[1] - bounds[0]
        )
    )

    distance = max(
        float(required.max()) * margin,
        diagonal * 0.55,
        1e-3,
    )

    eye = target + z_axis * distance

    pose = np.eye(
        4,
        dtype=np.float64,
    )
    pose[:3, 0] = x_axis
    pose[:3, 1] = y_axis
    pose[:3, 2] = z_axis
    pose[:3, 3] = eye

    depths = distance - projected_z

    min_depth = max(
        float(depths.min()),
        1e-6,
    )
    max_depth = max(
        float(depths.max()),
        min_depth + 1e-6,
    )

    znear = max(
        min_depth * 0.25,
        diagonal * 1e-5,
        1e-5,
    )
    zfar = max(
        max_depth * 2.0,
        znear * 100.0,
    )

    return (
        pose,
        znear,
        zfar,
        target,
        diagonal,
    )


def parse_rgb(
    value: str,
) -> tuple[float, float, float]:
    try:
        parts = [
            float(part.strip())
            for part in value.split(",")
        ]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Color must be specified as R,G,B."
        ) from exc

    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "Color must contain exactly three values: R,G,B"
        )

    if max(parts) > 1.0:
        parts = [
            part / 255.0
            for part in parts
        ]

    if any(
        part < 0.0 or part > 1.0
        for part in parts
    ):
        raise argparse.ArgumentTypeError(
            "Color values must be in the range 0..1 or 0..255."
        )

    return (
        float(parts[0]),
        float(parts[1]),
        float(parts[2]),
    )


def choose_total_frames(
    lengths: Sequence[int],
    policy: str,
) -> int:
    if policy == "strict":
        if len(set(lengths)) != 1:
            raise ValueError(
                "With --length-policy strict, all sequences must "
                f"have the same length; found: {list(lengths)}"
            )

        return lengths[0]

    if policy == "min":
        return min(lengths)

    if policy == "max-hold":
        return max(lengths)

    raise ValueError(
        f"Unknown length policy: {policy}"
    )


def frame_path(
    sequence: Sequence[Path],
    frame_index: int,
    policy: str,
) -> Path:
    if frame_index < len(sequence):
        return sequence[frame_index]

    if policy == "max-hold":
        return sequence[-1]

    raise IndexError(frame_index)


def make_materials(
    pyrender_module,
    colors: Sequence[
        tuple[float, float, float]
    ],
):
    materials = []

    for index in range(3):
        rgb = (
            colors[index]
            if index < len(colors)
            else DEFAULT_COLORS[index][:3]
        )

        materials.append(
            pyrender_module.MetallicRoughnessMaterial(
                baseColorFactor=(*rgb, 1.0),
                metallicFactor=0.0,
                roughnessFactor=0.75,
                alphaMode="OPAQUE",
                doubleSided=True,
            )
        )

    return materials


def make_directional_light_pose(
    direction_to_scene: np.ndarray,
    world_up: np.ndarray,
) -> np.ndarray:
    backward = -normalize(direction_to_scene)

    x_axis, y_axis, z_axis = camera_basis(
        backward,
        world_up,
    )

    pose = np.eye(
        4,
        dtype=np.float64,
    )
    pose[:3, 0] = x_axis
    pose[:3, 1] = y_axis
    pose[:3, 2] = z_axis

    return pose


def render_video(
    args: argparse.Namespace,
) -> None:
    if not 1 <= len(args.folders) <= 3:
        raise ValueError(
            "Provide between 1 and 3 input folders."
        )

    if args.width % 2 or args.height % 2:
        raise ValueError(
            "Width and height must be even for H.264."
        )

    if args.scale <= 0:
        raise ValueError(
            "--scale must be greater than 0."
        )

    if args.fps <= 0:
        raise ValueError(
            "--fps must be greater than 0."
        )

    if not 0.0 <= args.quality <= 10.0:
        raise ValueError(
            "--quality must be between 0 and 10."
        )

    if not 1.0 < args.fov < 179.0:
        raise ValueError(
            "--fov must be between 1 and 179 degrees."
        )

    if args.camera_margin < 1.0:
        raise ValueError(
            "--camera-margin must be at least 1.0."
        )

    if args.bounds_step < 1 or args.stride < 1:
        raise ValueError(
            "--bounds-step and --stride must be at least 1."
        )

    folders = [
        Path(folder).expanduser().resolve()
        for folder in args.folders
    ]

    for folder in folders:
        if not folder.is_dir():
            raise NotADirectoryError(
                f"Invalid directory: {folder}"
            )

    sequences = [
        collect_obj_files(
            folder,
            args.recursive,
        )
        for folder in folders
    ]

    lengths = [
        len(sequence)
        for sequence in sequences
    ]

    print("Sequences:")

    for folder, length in zip(
        folders,
        lengths,
    ):
        print(
            f"  {folder}: {length} frames"
        )

    total_frames = choose_total_frames(
        lengths,
        args.length_policy,
    )

    start = args.start
    end = (
        total_frames
        if args.end is None
        else min(args.end, total_frames)
    )

    if not 0 <= start < end:
        raise ValueError(
            f"Invalid frame range: start={start}, end={end}"
        )

    timeline = list(
        range(
            start,
            end,
            args.stride,
        )
    )

    global_bounds = compute_global_bounds(
        sequences,
        args.bounds_step,
    )

    shared_transform = make_shared_transform(
        global_bounds,
        args.center,
        args.scale,
        args.up_axis,
    )

    rendered_bounds = transform_bounds(
        global_bounds,
        shared_transform,
    )

    print("Global bounding box:")
    print(
        f"  min={global_bounds[0].tolist()}"
    )
    print(
        f"  max={global_bounds[1].tolist()}"
    )
    print("Shared mesh transform:")
    print(shared_transform)

    # Must be set before importing pyrender and PyOpenGL.
    if args.backend != "auto":
        os.environ["PYOPENGL_PLATFORM"] = args.backend

    try:
        import pyrender
    except ImportError as exc:
        raise RuntimeError(
            "pyrender is not installed. Install with: "
            "pip install pyrender trimesh numpy imageio[ffmpeg]"
        ) from exc

    (
        camera_pose,
        znear,
        zfar,
        target,
        diagonal,
    ) = fit_camera_to_bounds(
        rendered_bounds,
        args.width,
        args.height,
        args.fov,
        args.view,
        args.up_axis,
        args.camera_margin,
    )

    background = np.array(
        [*args.background, 1.0],
        dtype=np.float32,
    )

    scene = pyrender.Scene(
        bg_color=background,
        ambient_light=np.array(
            [0.22, 0.22, 0.22],
            dtype=np.float32,
        ),
    )

    camera = pyrender.PerspectiveCamera(
        yfov=math.radians(args.fov),
        aspectRatio=args.width / args.height,
        znear=znear,
        zfar=zfar,
    )
    scene.add(
        camera,
        pose=camera_pose,
        name="camera",
    )

    _, world_up = view_backward_axis(
        args.view,
        args.up_axis,
    )

    key_light = pyrender.DirectionalLight(
        color=np.ones(3),
        intensity=3.0,
    )
    scene.add(
        key_light,
        pose=camera_pose,
        name="key_light",
    )

    fill_direction = normalize(
        target
        - (
            target
            + np.array([1.0, -0.5, 0.75])
            * max(diagonal, 1.0)
        )
    )

    if args.up_axis == "z":
        fill_direction = normalize(
            np.array([-1.0, 0.7, -0.6])
        )

    fill_pose = make_directional_light_pose(
        fill_direction,
        world_up,
    )

    scene.add(
        pyrender.DirectionalLight(
            color=np.ones(3),
            intensity=1.4,
        ),
        pose=fill_pose,
        name="fill_light",
    )

    rim_direction = -normalize(
        camera_pose[:3, 2]
        + world_up * 0.35
    )

    rim_pose = make_directional_light_pose(
        rim_direction,
        world_up,
    )

    scene.add(
        pyrender.DirectionalLight(
            color=np.ones(3),
            intensity=1.0,
        ),
        pose=rim_pose,
        name="rim_light",
    )

    materials = make_materials(
        pyrender,
        args.color,
    )

    renderer = pyrender.OffscreenRenderer(
        viewport_width=args.width,
        viewport_height=args.height,
        point_size=1.0,
    )

    output = (
        Path(args.output)
        .expanduser()
        .resolve()
    )
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    flags = pyrender.RenderFlags.NONE

    if args.shadows:
        flags |= (
            pyrender.RenderFlags.SHADOWS_DIRECTIONAL
        )

    print(
        f"Rendering {len(timeline)} frames to {output} "
        f"({args.width}x{args.height}, {args.fps:g} fps) ..."
    )

    writer = None

    try:
        writer = imageio.get_writer(
            output,
            fps=args.fps,
            codec="libx264",
            quality=args.quality,
            pixelformat="yuv420p",
            macro_block_size=2,
            ffmpeg_log_level="warning",
        )

        for output_index, source_frame in enumerate(
            timeline
        ):
            frame_nodes = []

            try:
                for sequence_index, sequence in enumerate(
                    sequences
                ):
                    path = frame_path(
                        sequence,
                        source_frame,
                        args.length_policy,
                    )

                    mesh = load_obj_mesh(path)

                    material = (
                        None
                        if args.material_mode == "obj"
                        else materials[sequence_index]
                    )

                    render_mesh = pyrender.Mesh.from_trimesh(
                        mesh,
                        material=material,
                        smooth=not args.flat,
                    )

                    # Use the same pose for every sequence and frame.
                    node = scene.add(
                        render_mesh,
                        pose=shared_transform,
                        name=(
                            f"sequence_{sequence_index}_"
                            f"frame_{source_frame}"
                        ),
                    )
                    frame_nodes.append(node)

                color, _ = renderer.render(
                    scene,
                    flags=flags,
                )

                writer.append_data(
                    np.asarray(
                        color,
                        dtype=np.uint8,
                    )
                )

            finally:
                for node in frame_nodes:
                    scene.remove_node(node)

            done = output_index + 1

            if (
                done == len(timeline)
                or done % max(1, len(timeline) // 20) == 0
            ):
                print(
                    f"  Rendering: {done}/{len(timeline)}",
                    flush=True,
                )

    finally:
        if writer is not None:
            writer.close()

        renderer.delete()

    print(f"Finished: {output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render 1-3 OBJ animation folders together as MP4. "
            "All meshes use the same global transformation."
        ),
        formatter_class=(
            argparse.ArgumentDefaultsHelpFormatter
        ),
    )

    parser.add_argument(
        "folders",
        nargs="+",
        help="1-3 folders containing OBJ frames",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="obj_animations.mp4",
        help="Output video",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Video frame rate",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Video width",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Video height",
    )
    parser.add_argument(
        "--quality",
        type=float,
        default=8.0,
        help="imageio/FFmpeg quality in the range 0..10",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "egl", "osmesa"),
        default="auto",
        help="OpenGL backend; egl is commonly used on headless Linux",
    )
    parser.add_argument(
        "--center",
        choices=("none", "bbox", "ground"),
        default="bbox",
        help=(
            "Shared centering mode: bbox centers the bounding box "
            "at the origin; ground also places the lower edge at 0"
        ),
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Shared uniform scale factor",
    )
    parser.add_argument(
        "--up-axis",
        choices=("y", "z"),
        default="y",
        help="Vertical world axis",
    )
    parser.add_argument(
        "--view",
        choices=(
            "front",
            "back",
            "side",
            "iso",
        ),
        default="front",
    )
    parser.add_argument(
        "--fov",
        type=float,
        default=45.0,
        help="Vertical camera field of view in degrees",
    )
    parser.add_argument(
        "--camera-margin",
        type=float,
        default=1.12,
        help="Additional camera margin",
    )
    parser.add_argument(
        "--length-policy",
        choices=(
            "max-hold",
            "min",
            "strict",
        ),
        default="max-hold",
        help=(
            "Handling of unequal sequence lengths: max-hold repeats "
            "the final frame, min stops at the shortest sequence, "
            "strict requires equal lengths"
        ),
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="First source frame, inclusive",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="Last source frame, exclusive",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Render every nth frame",
    )
    parser.add_argument(
        "--bounds-step",
        type=int,
        default=1,
        help=(
            "Use every nth OBJ frame for bounding-box computation; "
            "the final frame is always included"
        ),
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search recursively for OBJ files",
    )
    parser.add_argument(
        "--flat",
        action="store_true",
        help="Use flat instead of smooth shading",
    )
    parser.add_argument(
        "--shadows",
        action="store_true",
        help="Enable directional shadows",
    )
    parser.add_argument(
        "--material-mode",
        choices=("folder", "obj"),
        default="folder",
        help=(
            "folder uses one color per folder; "
            "obj uses OBJ/MTL materials"
        ),
    )
    parser.add_argument(
        "--color",
        type=parse_rgb,
        action="append",
        default=[],
        metavar="R,G,B",
        help=(
            "Color for one sequence; may be repeated. "
            "Values can use 0..1 or 0..255"
        ),
    )
    parser.add_argument(
        "--background",
        type=parse_rgb,
        default=(0.96, 0.96, 0.96),
        metavar="R,G,B",
        help="Background color",
    )

    return parser


def main(
    argv: Iterable[str] | None = None,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        render_video(args)
    except KeyboardInterrupt:
        print(
            "Aborted.",
            file=sys.stderr,
        )
        return 130
    except Exception as exc:
        print(
            f"Error: {exc}",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())