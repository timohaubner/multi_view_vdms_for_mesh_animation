#!/usr/bin/env python3
"""Render 1-3 OBJ animation sequences together with pyrender.

Each input directory is interpreted as one animation sequence. OBJ files are
sorted naturally (frame_2.obj before frame_10.obj) and rendered at the same
frame index. A single shared transform is computed and applied to every mesh,
so the original relative coordinates between sequences are preserved.

Example:
    python render_obj_animations.py person_a person_b \
        --output together.mp4 --fps 30 --backend egl --center bbox
"""

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
    """Return a case-insensitive natural-sort key for a path name."""
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", path.name)]


def collect_obj_files(folder: Path, recursive: bool) -> list[Path]:
    pattern = "**/*.obj" if recursive else "*.obj"
    files = sorted(folder.glob(pattern), key=natural_key)
    if not files:
        raise FileNotFoundError(f"Keine OBJ-Dateien gefunden: {folder}")
    return files


def load_obj_mesh(path: Path) -> trimesh.Trimesh:
    """Load an OBJ as one Trimesh and bake any internal scene transforms."""
    try:
        loaded = trimesh.load_mesh(path, process=False)
    except (AttributeError, TypeError):
        # Compatibility fallback for older trimesh versions.
        loaded = trimesh.load(path, force="mesh", process=False)

    if isinstance(loaded, trimesh.Scene):
        if hasattr(loaded, "to_mesh"):
            mesh = loaded.to_mesh()
        else:  # pragma: no cover - only for older trimesh releases
            mesh = loaded.dump(concatenate=True)
    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    else:
        raise TypeError(
            f"{path} wurde nicht als Dreiecks-Mesh geladen "
            f"(Typ: {type(loaded).__name__})."
        )

    if mesh.is_empty or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError(f"Leeres oder ungültiges Mesh: {path}")
    if not np.isfinite(mesh.vertices).all():
        raise ValueError(f"Mesh enthält NaN/Inf-Koordinaten: {path}")

    return mesh


def sampled_paths(paths: Sequence[Path], step: int) -> list[Path]:
    """Sample paths for bounds computation, always including the final frame."""
    result = list(paths[::step])
    if paths[-1] not in result:
        result.append(paths[-1])
    return result


def compute_global_bounds(
    sequences: Sequence[Sequence[Path]], bounds_step: int
) -> np.ndarray:
    """Compute union AABB over sampled frames of every sequence."""
    global_min = np.full(3, np.inf, dtype=np.float64)
    global_max = np.full(3, -np.inf, dtype=np.float64)

    files_to_scan = sum(
        len(sampled_paths(sequence, bounds_step)) for sequence in sequences
    )
    scanned = 0

    print(f"Bestimme gemeinsame Bounding Box aus {files_to_scan} OBJ-Dateien ...")
    for sequence in sequences:
        for path in sampled_paths(sequence, bounds_step):
            mesh = load_obj_mesh(path)
            bounds = np.asarray(mesh.bounds, dtype=np.float64)
            global_min = np.minimum(global_min, bounds[0])
            global_max = np.maximum(global_max, bounds[1])
            scanned += 1
            if scanned == files_to_scan or scanned % max(1, files_to_scan // 10) == 0:
                print(f"  Bounding Box: {scanned}/{files_to_scan}", flush=True)

    if not np.isfinite(global_min).all() or not np.isfinite(global_max).all():
        raise ValueError("Die globale Bounding Box konnte nicht bestimmt werden.")
    if np.allclose(global_min, global_max):
        raise ValueError("Die globale Bounding Box hat keine räumliche Ausdehnung.")

    return np.vstack((global_min, global_max))


def bounds_corners(bounds: np.ndarray) -> np.ndarray:
    lo, hi = bounds
    return np.array(
        [[x, y, z]
         for x in (lo[0], hi[0])
         for y in (lo[1], hi[1])
         for z in (lo[2], hi[2])],
        dtype=np.float64,
    )


def make_shared_transform(
    bounds: np.ndarray,
    center_mode: str,
    scale: float,
    up_axis: str,
) -> np.ndarray:
    """Build one transform that will be applied unchanged to every mesh."""
    lo, hi = bounds
    center = (lo + hi) * 0.5
    offset = np.zeros(3, dtype=np.float64)

    if center_mode == "bbox":
        offset = -center
    elif center_mode == "ground":
        up_index = 1 if up_axis == "y" else 2
        offset = -center
        offset[up_index] = -lo[up_index]
    elif center_mode != "none":
        raise ValueError(f"Unbekannter Zentrierungsmodus: {center_mode}")

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] *= scale
    transform[:3, 3] = offset * scale
    return transform


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack((points, np.ones(len(points), dtype=np.float64)))
    return (transform @ homogeneous.T).T[:, :3]


def transform_bounds(bounds: np.ndarray, transform: np.ndarray) -> np.ndarray:
    corners = transform_points(bounds_corners(bounds), transform)
    return np.vstack((corners.min(axis=0), corners.max(axis=0)))


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        raise ValueError("Nullvektor kann nicht normalisiert werden.")
    return vector / norm


def view_backward_axis(view: str, up_axis: str) -> tuple[np.ndarray, np.ndarray]:
    """Return camera backward axis (+Z local) and world-up vector."""
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


def camera_basis(backward: np.ndarray, world_up: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create an orthonormal camera basis in world coordinates."""
    z_axis = normalize(backward)  # Camera local +Z; it looks along local -Z.
    x_axis = normalize(np.cross(world_up, z_axis))
    y_axis = normalize(np.cross(z_axis, x_axis))
    return x_axis, y_axis, z_axis


def fit_camera_to_bounds(
    bounds: np.ndarray,
    width: int,
    height: int,
    yfov_degrees: float,
    view: str,
    up_axis: str,
    margin: float,
) -> tuple[np.ndarray, float, float, np.ndarray, float]:
    """Fit a perspective camera to the transformed global AABB."""
    corners = bounds_corners(bounds)
    target = bounds.mean(axis=0)
    backward, world_up = view_backward_axis(view, up_axis)
    x_axis, y_axis, z_axis = camera_basis(backward, world_up)

    relative = corners - target
    projected_x = relative @ x_axis
    projected_y = relative @ y_axis
    projected_z = relative @ z_axis

    yfov = math.radians(yfov_degrees)
    aspect = width / height
    xfov = 2.0 * math.atan(math.tan(yfov * 0.5) * aspect)
    tan_x = math.tan(xfov * 0.5)
    tan_y = math.tan(yfov * 0.5)

    required = projected_z + np.maximum(
        np.abs(projected_x) / tan_x,
        np.abs(projected_y) / tan_y,
    )
    diagonal = float(np.linalg.norm(bounds[1] - bounds[0]))
    distance = max(float(required.max()) * margin, diagonal * 0.55, 1e-3)
    eye = target + z_axis * distance

    pose = np.eye(4, dtype=np.float64)
    pose[:3, 0] = x_axis
    pose[:3, 1] = y_axis
    pose[:3, 2] = z_axis
    pose[:3, 3] = eye

    depths = distance - projected_z
    min_depth = max(float(depths.min()), 1e-6)
    max_depth = max(float(depths.max()), min_depth + 1e-6)
    znear = max(min_depth * 0.25, diagonal * 1e-5, 1e-5)
    zfar = max(max_depth * 2.0, znear * 100.0)
    return pose, znear, zfar, target, diagonal


def parse_rgb(value: str) -> tuple[float, float, float]:
    """Parse R,G,B as either 0..1 floats or 0..255 integers."""
    try:
        parts = [float(part.strip()) for part in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Farbe muss R,G,B sein.") from exc
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Farbe muss genau drei Werte enthalten: R,G,B")
    if max(parts) > 1.0:
        parts = [part / 255.0 for part in parts]
    if any(part < 0.0 or part > 1.0 for part in parts):
        raise argparse.ArgumentTypeError("Farbwerte müssen in 0..1 oder 0..255 liegen.")
    return float(parts[0]), float(parts[1]), float(parts[2])


def choose_total_frames(lengths: Sequence[int], policy: str) -> int:
    if policy == "strict":
        if len(set(lengths)) != 1:
            raise ValueError(
                "Bei --length-policy strict müssen alle Sequenzen gleich lang sein; "
                f"gefunden: {list(lengths)}"
            )
        return lengths[0]
    if policy == "min":
        return min(lengths)
    if policy == "max-hold":
        return max(lengths)
    raise ValueError(f"Unbekannte Längenstrategie: {policy}")


def frame_path(sequence: Sequence[Path], frame_index: int, policy: str) -> Path:
    if frame_index < len(sequence):
        return sequence[frame_index]
    if policy == "max-hold":
        return sequence[-1]
    raise IndexError(frame_index)


def make_materials(pyrender_module, colors: Sequence[tuple[float, float, float]]):
    materials = []
    for index in range(3):
        rgb = colors[index] if index < len(colors) else DEFAULT_COLORS[index][:3]
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


def make_directional_light_pose(direction_to_scene: np.ndarray, world_up: np.ndarray) -> np.ndarray:
    """Pose a directional light so its local -Z points in direction_to_scene."""
    backward = -normalize(direction_to_scene)
    x_axis, y_axis, z_axis = camera_basis(backward, world_up)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 0] = x_axis
    pose[:3, 1] = y_axis
    pose[:3, 2] = z_axis
    return pose


def render_video(args: argparse.Namespace) -> None:
    if not (1 <= len(args.folders) <= 3):
        raise ValueError("Bitte 1 bis 3 Eingabeordner angeben.")
    if args.width % 2 or args.height % 2:
        raise ValueError("Breite und Höhe müssen für H.264 gerade Zahlen sein.")
    if args.scale <= 0:
        raise ValueError("--scale muss größer als 0 sein.")
    if args.fps <= 0:
        raise ValueError("--fps muss größer als 0 sein.")
    if not 0.0 <= args.quality <= 10.0:
        raise ValueError("--quality muss zwischen 0 und 10 liegen.")
    if not 1.0 < args.fov < 179.0:
        raise ValueError("--fov muss zwischen 1 und 179 Grad liegen.")
    if args.camera_margin < 1.0:
        raise ValueError("--camera-margin muss mindestens 1.0 sein.")
    if args.bounds_step < 1 or args.stride < 1:
        raise ValueError("--bounds-step und --stride müssen mindestens 1 sein.")

    folders = [Path(folder).expanduser().resolve() for folder in args.folders]
    for folder in folders:
        if not folder.is_dir():
            raise NotADirectoryError(f"Kein gültiger Ordner: {folder}")

    sequences = [collect_obj_files(folder, args.recursive) for folder in folders]
    lengths = [len(sequence) for sequence in sequences]
    print("Sequenzen:")
    for folder, length in zip(folders, lengths):
        print(f"  {folder}: {length} Frames")

    total_frames = choose_total_frames(lengths, args.length_policy)
    start = args.start
    end = total_frames if args.end is None else min(args.end, total_frames)
    if not (0 <= start < end):
        raise ValueError(f"Ungültiger Framebereich: start={start}, end={end}")
    timeline = list(range(start, end, args.stride))

    global_bounds = compute_global_bounds(sequences, args.bounds_step)
    shared_transform = make_shared_transform(
        global_bounds, args.center, args.scale, args.up_axis
    )
    rendered_bounds = transform_bounds(global_bounds, shared_transform)

    print("Globale Bounding Box (Originalkoordinaten):")
    print(f"  min={global_bounds[0].tolist()}")
    print(f"  max={global_bounds[1].tolist()}")
    print("Gemeinsame Transformation für alle Meshes:")
    print(shared_transform)

    # Must be set before importing pyrender / PyOpenGL.
    if args.backend != "auto":
        os.environ["PYOPENGL_PLATFORM"] = args.backend

    try:
        import pyrender
    except ImportError as exc:
        raise RuntimeError(
            "pyrender ist nicht installiert. Installation: "
            "pip install pyrender trimesh numpy imageio[ffmpeg]"
        ) from exc

    camera_pose, znear, zfar, target, diagonal = fit_camera_to_bounds(
        rendered_bounds,
        args.width,
        args.height,
        args.fov,
        args.view,
        args.up_axis,
        args.camera_margin,
    )

    background = np.array([*args.background, 1.0], dtype=np.float32)
    scene = pyrender.Scene(
        bg_color=background,
        ambient_light=np.array([0.22, 0.22, 0.22], dtype=np.float32),
    )

    camera = pyrender.PerspectiveCamera(
        yfov=math.radians(args.fov),
        aspectRatio=args.width / args.height,
        znear=znear,
        zfar=zfar,
    )
    scene.add(camera, pose=camera_pose, name="camera")

    _, world_up = view_backward_axis(args.view, args.up_axis)
    key_light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    scene.add(key_light, pose=camera_pose, name="key_light")

    fill_direction = normalize(target - (target + np.array([1.0, -0.5, 0.75]) * max(diagonal, 1.0)))
    if args.up_axis == "z":
        fill_direction = normalize(np.array([-1.0, 0.7, -0.6]))
    fill_pose = make_directional_light_pose(fill_direction, world_up)
    scene.add(
        pyrender.DirectionalLight(color=np.ones(3), intensity=1.4),
        pose=fill_pose,
        name="fill_light",
    )

    rim_direction = -normalize(camera_pose[:3, 2] + world_up * 0.35)
    rim_pose = make_directional_light_pose(rim_direction, world_up)
    scene.add(
        pyrender.DirectionalLight(color=np.ones(3), intensity=1.0),
        pose=rim_pose,
        name="rim_light",
    )

    materials = make_materials(pyrender, args.color)
    renderer = pyrender.OffscreenRenderer(
        viewport_width=args.width,
        viewport_height=args.height,
        point_size=1.0,
    )

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    flags = pyrender.RenderFlags.NONE
    if args.shadows:
        flags |= pyrender.RenderFlags.SHADOWS_DIRECTIONAL

    print(
        f"Rendere {len(timeline)} Frames nach {output} "
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

        for output_index, source_frame in enumerate(timeline):
            frame_nodes = []
            try:
                for sequence_index, sequence in enumerate(sequences):
                    path = frame_path(sequence, source_frame, args.length_policy)
                    mesh = load_obj_mesh(path)
                    material = None if args.material_mode == "obj" else materials[sequence_index]
                    render_mesh = pyrender.Mesh.from_trimesh(
                        mesh,
                        material=material,
                        smooth=not args.flat,
                    )
                    # The exact same pose is used for every sequence and every frame.
                    node = scene.add(
                        render_mesh,
                        pose=shared_transform,
                        name=f"sequence_{sequence_index}_frame_{source_frame}",
                    )
                    frame_nodes.append(node)

                color, _depth = renderer.render(scene, flags=flags)
                writer.append_data(np.asarray(color, dtype=np.uint8))
            finally:
                for node in frame_nodes:
                    scene.remove_node(node)

            done = output_index + 1
            if done == len(timeline) or done % max(1, len(timeline) // 20) == 0:
                print(f"  Rendering: {done}/{len(timeline)}", flush=True)
    finally:
        if writer is not None:
            writer.close()
        renderer.delete()

    print(f"Fertig: {output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Rendert 1-3 Ordner mit OBJ-Animationsframes gemeinsam als MP4. "
            "Alle Meshes erhalten dieselbe globale Transformation."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("folders", nargs="+", help="1-3 Ordner mit OBJ-Frames")
    parser.add_argument("-o", "--output", default="obj_animations.mp4", help="Ausgabevideo")
    parser.add_argument("--fps", type=float, default=30.0, help="Video-Framerate")
    parser.add_argument("--width", type=int, default=1280, help="Videobreite")
    parser.add_argument("--height", type=int, default=720, help="Videohöhe")
    parser.add_argument("--quality", type=float, default=8.0, help="imageio/FFmpeg-Qualität 0..10")
    parser.add_argument(
        "--backend",
        choices=("auto", "egl", "osmesa"),
        default="auto",
        help="OpenGL-Backend; auf Headless-Linux meist egl",
    )
    parser.add_argument(
        "--center",
        choices=("none", "bbox", "ground"),
        default="bbox",
        help=(
            "Gemeinsame Zentrierung: bbox=Boxzentrum auf Ursprung, "
            "ground=horizontal zentrieren und Unterkante auf 0"
        ),
    )
    parser.add_argument("--scale", type=float, default=1.0, help="Gemeinsamer uniformer Skalierungsfaktor")
    parser.add_argument("--up-axis", choices=("y", "z"), default="y", help="Vertikale Weltachse")
    parser.add_argument("--view", choices=("front", "back", "side", "iso"), default="front")
    parser.add_argument("--fov", type=float, default=45.0, help="Vertikales Kamera-Sichtfeld in Grad")
    parser.add_argument("--camera-margin", type=float, default=1.12, help="Zusätzlicher Bildrand")
    parser.add_argument(
        "--length-policy",
        choices=("max-hold", "min", "strict"),
        default="max-hold",
        help=(
            "Umgang mit ungleichen Sequenzlängen: max-hold wiederholt den letzten Frame, "
            "min stoppt bei der kürzesten, strict verlangt gleiche Längen"
        ),
    )
    parser.add_argument("--start", type=int, default=0, help="Erster Quellframe, inklusiv")
    parser.add_argument("--end", type=int, default=None, help="Letzter Quellframe, exklusiv")
    parser.add_argument("--stride", type=int, default=1, help="Nur jeden n-ten Frame rendern")
    parser.add_argument(
        "--bounds-step",
        type=int,
        default=1,
        help="Für Bounding Box nur jeden n-ten OBJ-Frame prüfen; letzter wird immer geprüft",
    )
    parser.add_argument("--recursive", action="store_true", help="OBJ-Dateien rekursiv suchen")
    parser.add_argument("--flat", action="store_true", help="Flache statt geglätteter Schattierung")
    parser.add_argument("--shadows", action="store_true", help="Richtungsschatten aktivieren")
    parser.add_argument(
        "--material-mode",
        choices=("folder", "obj"),
        default="folder",
        help="folder=je Ordner eine Farbe; obj=OBJ/MTL-Materialien verwenden",
    )
    parser.add_argument(
        "--color",
        type=parse_rgb,
        action="append",
        default=[],
        metavar="R,G,B",
        help="Farbe für eine Sequenz, wiederholbar; Werte 0..1 oder 0..255",
    )
    parser.add_argument(
        "--background",
        type=parse_rgb,
        default=(0.96, 0.96, 0.96),
        metavar="R,G,B",
        help="Hintergrundfarbe",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        render_video(args)
    except KeyboardInterrupt:
        print("Abgebrochen.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Fehler: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())