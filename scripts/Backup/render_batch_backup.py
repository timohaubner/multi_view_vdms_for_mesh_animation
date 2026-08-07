from pathlib import Path
import argparse
import subprocess
import sys


def find_sequence_mesh_dirs(sequences_root, pose_folder="posed"):
    """
    Sucht alle Animationen mit folgender Struktur:

    sequences_root/
      00145/
        shortlong_ATUSquat/
          posed/
            *.obj
        shortlong_hips/
          posed/
            *.obj
        shortlong_punching/
          posed/
            *.obj

    Gibt Tripel zurück:
      (sequence_name, animation_name, mesh_dir)
    """
    sequences_root = Path(sequences_root)

    if not sequences_root.exists():
        raise RuntimeError(f"Input folder does not exist: {sequences_root}")

    sequence_mesh_dirs = []

    for seq_dir in sorted(sequences_root.iterdir()):
        if not seq_dir.is_dir():
            continue

        found_any = False

        # Alle Animationen innerhalb der Sequence suchen
        for posed_dir in sorted(seq_dir.glob(f"*/{pose_folder}")):
            if not posed_dir.is_dir():
                continue

            obj_files = list(posed_dir.glob("*.obj"))
            if not obj_files:
                print(f"Skipping {posed_dir}: no .obj files found")
                continue

            animation_name = posed_dir.parent.name

            sequence_mesh_dirs.append(
                (seq_dir.name, animation_name, posed_dir)
            )

            found_any = True

        if not found_any:
            print(f"Skipping {seq_dir.name}: no .obj files found in */{pose_folder}/")

    return sequence_mesh_dirs


def batch_render_sequences(
    renderer_script,
    sequences_root,
    output_root,
    fps=60,
    width=1000,
    height=800,
    view="front",
    pose_folder="posed",
    output_name=None,
    overwrite=False,
):
    renderer_script = Path(renderer_script)
    output_root = Path(output_root)

    if not renderer_script.exists():
        raise RuntimeError(f"Renderer script does not exist: {renderer_script}")

    output_root.mkdir(parents=True, exist_ok=True)

    sequence_mesh_dirs = find_sequence_mesh_dirs(
        sequences_root=sequences_root,
        pose_folder=pose_folder,
    )

    if not sequence_mesh_dirs:
        raise RuntimeError(f"No valid sequences found in {sequences_root}")

    print(f"Found {len(sequence_mesh_dirs)} animations")

    for i, (seq_name, animation_name, mesh_dir) in enumerate(sequence_mesh_dirs, start=1):
        # Ausgabeordner pro Sequence und Animation
        animation_output_dir = output_root / seq_name / animation_name
        animation_output_dir.mkdir(parents=True, exist_ok=True)

        if output_name is None:
            video_name = f"{seq_name}_{animation_name}_{view}.mp4"
        else:
            video_name = output_name

        out_video = animation_output_dir / video_name

        if out_video.exists() and not overwrite:
            print(
                f"[{i}/{len(sequence_mesh_dirs)}] "
                f"Skipping {seq_name}/{animation_name}: output already exists"
            )
            continue

        print(f"[{i}/{len(sequence_mesh_dirs)}] Rendering {seq_name}/{animation_name}")
        print(f"  mesh_dir:  {mesh_dir}")
        print(f"  out_video: {out_video}")

        cmd = [
            sys.executable,
            str(renderer_script),
            "--mesh_dir",
            str(mesh_dir),
            "--out_video",
            str(out_video),
            "--fps",
            str(fps),
            "--width",
            str(width),
            "--height",
            str(height),
            "--view",
            view,
        ]

        subprocess.run(cmd, check=True)

    print("Batch rendering finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--renderer_script",
        required=True,
        help="Path to the original render script",
    )

    parser.add_argument(
        "--sequences_root",
        required=True,
        help="Folder containing sequence folders, e.g. sequences/",
    )

    parser.add_argument(
        "--output_root",
        required=True,
        help="Output folder. One subfolder per sequence and animation will be created.",
    )

    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--width", type=int, default=1000)
    parser.add_argument("--height", type=int, default=800)

    parser.add_argument(
        "--view",
        choices=["front", "left", "right"],
        default="front",
    )

    parser.add_argument(
        "--pose_folder",
        default="posed",
        help="Name of the folder containing the .obj files. Default: posed",
    )

    parser.add_argument(
        "--output_name",
        default=None,
        help="Optional fixed output filename, e.g. render.mp4. Default: <sequence>_<animation>_<view>.mp4",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing videos",
    )

    args = parser.parse_args()

    batch_render_sequences(
        renderer_script=args.renderer_script,
        sequences_root=args.sequences_root,
        output_root=args.output_root,
        fps=args.fps,
        width=args.width,
        height=args.height,
        view=args.view,
        pose_folder=args.pose_folder,
        output_name=args.output_name,
        overwrite=args.overwrite,
    )