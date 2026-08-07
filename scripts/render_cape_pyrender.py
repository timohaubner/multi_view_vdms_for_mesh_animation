from pathlib import Path
import argparse
import numpy as np
import trimesh
import pyrender
import imageio.v2 as imageio


def get_camera_position(center, size, azimuth_deg):
    """
    Positioniert die Kamera auf einem Kreis um das Mesh.

    Konvention:
      0°    = front, Kamera vor der Person
      90°   = right, Kamera rechts von der Person
      -90°  = left, Kamera links von der Person
      180°  = back, Kamera hinter der Person
    """

    radius = 1.45 * size
    height_offset = 0.25 * size

    azimuth_rad = np.deg2rad(azimuth_deg)

    x = np.sin(azimuth_rad) * radius
    z = np.cos(azimuth_rad) * radius
    y = height_offset

    return center + np.array([x, y, z])


def look_at(camera_pos, target, up=np.array([0.0, 1.0, 0.0])):
    camera_pos = np.asarray(camera_pos, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    forward = target - camera_pos
    forward /= np.linalg.norm(forward)

    right = np.cross(forward, up)
    right /= np.linalg.norm(right)

    true_up = np.cross(right, forward)

    pose = np.eye(4)
    pose[:3, 0] = right
    pose[:3, 1] = true_up
    pose[:3, 2] = -forward
    pose[:3, 3] = camera_pos
    return pose


def setup_scene(first_mesh_path, width, height, azimuth_deg):
    """
    Erstellt Scene, Renderer, Kamera, Licht und Material.
    Gibt alles zurück, was für das Rendern benötigt wird.
    """

    first = trimesh.load(first_mesh_path, process=False)

    center = first.bounds.mean(axis=0)
    size = np.linalg.norm(first.bounds[1] - first.bounds[0])

    camera_pos = get_camera_position(center, size, azimuth_deg)
    target = center + np.array([0.0, 0.05 * size, 0.0])

    camera_pose = look_at(
        camera_pos,
        target,
        up=np.array([0.0, 1.0, 0.0])
    )

    scene = pyrender.Scene(
        bg_color=[255, 255, 255, 255],
        ambient_light=[0.6, 0.6, 0.6]
    )

    camera = pyrender.PerspectiveCamera(yfov=np.pi / 4.0)
    scene.add(camera, pose=camera_pose)

    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)

    scene.add(light, pose=camera_pose)

    renderer = pyrender.OffscreenRenderer(
        viewport_width=width,
        viewport_height=height
    )

    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0.35, 0.35, 0.35, 1.0],
        metallicFactor=0.0,
        roughnessFactor=0.75
    )

    return scene, renderer, material


def render_single_frame(scene, renderer, material, obj_path):
    """
    Rendert ein einzelnes Mesh und gibt das Farbbild zurück.
    """

    tri = trimesh.load(obj_path, process=False)

    render_mesh = pyrender.Mesh.from_trimesh(
        tri,
        material=material,
        smooth=True
    )

    mesh_node = scene.add(render_mesh)

    try:
        color, depth = renderer.render(scene)
    finally:
        scene.remove_node(mesh_node)

    return color


def render_sequence(
    mesh_dir,
    out_video,
    fps=60,
    width=512,
    height=512,
    azimuth_deg=0.0,
    image_only=False,
    image_filename=None,
    reduce_frames=1,
):

    mesh_dir = Path(mesh_dir)
    out_video = Path(out_video)

    # Immer das erste Mesh der gesamten Sequenz als Kamera-Referenz verwenden
    all_obj_files = sorted(mesh_dir.glob("*.obj"))

    if not all_obj_files:
        raise RuntimeError(f"No .obj files found in {mesh_dir}")

    camera_reference_obj = all_obj_files[0]

    # Separat bestimmen, welche Meshes tatsächlich gerendert werden
    if image_only and image_filename is not None:
        selected_obj = mesh_dir / image_filename

        if selected_obj.suffix.lower() != ".obj":
            raise ValueError(
                f"--image_filename must refer to an .obj file: {selected_obj}"
            )

        if not selected_obj.is_file():
            raise FileNotFoundError(
                f"OBJ file does not exist: {selected_obj}"
            )

        obj_files = [selected_obj]

    else:
        obj_files = all_obj_files[::reduce_frames]

    scene, renderer, material = setup_scene(
        first_mesh_path=camera_reference_obj,
        width=width,
        height=height,
        azimuth_deg=azimuth_deg
    )

    try:
        if image_only:
            out_image = out_video.with_suffix(".png")
            out_image.parent.mkdir(parents=True, exist_ok=True)

            print(f"Rendering first frame only: {obj_files[0].name}")

            color = render_single_frame(
                scene=scene,
                renderer=renderer,
                material=material,
                obj_path=obj_files[0]
            )

            imageio.imwrite(str(out_image), color)

            print(f"Saved image to: {out_image}")
            return

        out_video.parent.mkdir(parents=True, exist_ok=True)

        writer = imageio.get_writer(
            str(out_video),
            fps=fps,
            codec="libx264",
            quality=8
        )

        mesh_node = None

        try:
            for i, obj_path in enumerate(obj_files):
                print(f"[{i+1}/{len(obj_files)}] Rendering {obj_path.name}")

                tri = trimesh.load(obj_path, process=False)

                render_mesh = pyrender.Mesh.from_trimesh(
                    tri,
                    material=material,
                    smooth=True
                )

                if mesh_node is not None:
                    scene.remove_node(mesh_node)

                mesh_node = scene.add(render_mesh)

                color, depth = renderer.render(scene)
                writer.append_data(color)

        finally:
            writer.close()

        print(f"Saved video to: {out_video}")

    finally:
        renderer.delete()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh_dir", required=True)
    parser.add_argument("--out_video", required=True)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)

    parser.add_argument(
        "--azimuth_deg",
        type=float,
        default=0.0,
        help="Camera azimuth angle in degrees. 0=front, 90=right, -90/270=left, 180=back."
    )

    parser.add_argument(
        "--image_only",
        action="store_true",
        help="If set, only the first mesh frame is rendered and saved as an image."
    )

    parser.add_argument(
        "--reduce_frames",
        type=int,
        default=1,
        help="Only use every xth frame of the mesh sequence if activated."
    )

    parser.add_argument(
        "--image_filename",
        type=str,
        default=None,
        help=(
            "Name of the OBJ file to render when --image_only is active, "
            "for example frame_0123.obj."
        )
    )

    args = parser.parse_args()

    render_sequence(
        mesh_dir=args.mesh_dir,
        out_video=args.out_video,
        fps=args.fps / args.reduce_frames,
        width=args.width,
        height=args.height,
        azimuth_deg=args.azimuth_deg,
        image_only=args.image_only,
        image_filename=args.image_filename,
        reduce_frames=args.reduce_frames,
    )