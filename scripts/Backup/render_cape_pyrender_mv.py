from pathlib import Path
import argparse
import numpy as np
import trimesh
import pyrender
import imageio.v2 as imageio


def get_camera_position(center, size, view):
    """
    view:
      front = Kamera vor der Person
      left  = Kamera links von der Person
      right = Kamera rechts von der Person
    """

    distance = 1.65 * size
    height_offset = 0.18 * size

    if view == "front":
        return center + np.array([0.0, height_offset, distance])

    elif view == "left":
        return center + np.array([-distance, height_offset, 0.0])

    elif view == "right":
        return center + np.array([distance, height_offset, 0.0])

    else:
        raise ValueError(f"Unknown view: {view}")


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


def render_sequence(mesh_dir, out_video, fps=60, width=512, height=512, view ="front"):
    mesh_dir = Path(mesh_dir)
    out_video = Path(out_video)
    out_video.parent.mkdir(parents=True, exist_ok=True)

    obj_files = sorted(mesh_dir.glob("*.obj"))
    obj_files = obj_files[::3]
    if not obj_files:
        raise RuntimeError(f"No .obj files found in {mesh_dir}")

    # Load first mesh to estimate center and scale
    first = trimesh.load(obj_files[0], process=False)
    center = first.bounds.mean(axis=0)
    extent = first.bounds[1] - first.bounds[0]
    size = extent.max()

    camera_pos = get_camera_position(center, size, view)

    target = center + np.array([0.0, 0.05 * size, 0.0])

    camera_pose = look_at(camera_pos, target, up=np.array([0.0, 1.0, 0.0]))

    scene = pyrender.Scene(
        bg_color=[255, 255, 255, 255],
        ambient_light=[0.6, 0.6, 0.6]
    )

    camera = pyrender.PerspectiveCamera(yfov=np.pi / 4.0)
    scene.add(camera, pose=camera_pose)

    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    scene.add(light, pose=camera_pose)

    renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)

    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0.42, 0.42, 0.42, 1.0],
        metallicFactor=0.0,
        roughnessFactor=0.75
    )

    writer = imageio.get_writer(
        str(out_video),
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=1
    )

    mesh_node = None

    try:
        for i, obj_path in enumerate(obj_files):
            print(f"[{i+1}/{len(obj_files)}] Rendering {obj_path.name}")

            tri = trimesh.load(obj_path, process=False)

            render_mesh = pyrender.Mesh.from_trimesh(tri, material=material, smooth=True)

            if mesh_node is not None:
                scene.remove_node(mesh_node)

            mesh_node = scene.add(render_mesh)

            color, depth = renderer.render(scene)
            writer.append_data(color)

    finally:
        writer.close()
        renderer.delete()

    print(f"Saved video to: {out_video}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh_dir", required=True)
    parser.add_argument("--out_video", required=True)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument(
        "--view",
        choices=["front", "left", "right"],
        default="front",
        help="Camera view: front, left, or right"
    )
    args = parser.parse_args()

    render_sequence(
        mesh_dir=args.mesh_dir,
        out_video=args.out_video,
        fps=args.fps,
        width=args.width,
        height=args.height,
        view=args.view
    )