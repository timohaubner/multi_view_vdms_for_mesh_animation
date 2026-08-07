from pathlib import Path
import argparse
import subprocess
import sys


def find_sequence_mesh_dirs(sequences_root, pose_folder="posed"):
    sequences_root = Path(sequences_root)

    if not sequences_root.exists():
        raise RuntimeError(f"Input folder does not exist: {sequences_root}")

    sequence_mesh_dirs = []

    for seq_dir in sorted(sequences_root.iterdir()):
        if not seq_dir.is_dir():
            continue

        found_any = False

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
    output_name=None,
    renderer_args=None,
):
    renderer_script = Path(renderer_script)
    output_root = Path(output_root)
    renderer_args = renderer_args or []

    if not renderer_script.exists():
        raise RuntimeError(f"Renderer script does not exist: {renderer_script}")

    output_root.mkdir(parents=True, exist_ok=True)

    sequence_mesh_dirs = find_sequence_mesh_dirs(
        sequences_root=sequences_root,
        pose_folder="posed",
    )

    if not sequence_mesh_dirs:
        raise RuntimeError(f"No valid sequences found in {sequences_root}")

    print(f"Found {len(sequence_mesh_dirs)} animations")

    for i, (seq_name, animation_name, mesh_dir) in enumerate(sequence_mesh_dirs, start=1):
        animation_output_dir = output_root / seq_name / animation_name
        animation_output_dir.mkdir(parents=True, exist_ok=True)

        if output_name is None:
            video_name = f"{seq_name}_{animation_name}.mp4"
        else:
            video_name = f"{output_name}_{seq_name}_{animation_name}.mp4"

        out_video = animation_output_dir / video_name

        '''
        if out_video.exists():
            print(
                f"[{i}/{len(sequence_mesh_dirs)}] "
                f"Skipping {seq_name}/{animation_name}: output already exists"
            )
            continue
            
        '''

        print(f"[{i}/{len(sequence_mesh_dirs)}] Rendering {seq_name}/{animation_name}")
        print(f"  mesh_dir:  {mesh_dir}")
        print(f"  out_video: {out_video}")

        cmd = [
            sys.executable,
            str(renderer_script),

            # Diese zwei Argumente setzt der Wrapper pro Animation automatisch:
            "--mesh_dir",
            str(mesh_dir),
            "--out_video",
            str(out_video),

            # Alles hier wird unverändert an den Renderer durchgereicht:
            *renderer_args,
        ]

        print("  command:")
        print("  " + " ".join(cmd))

        subprocess.run(cmd, check=True)

    print("Batch rendering finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Nur Parameter, die wirklich dem Batch-Skript gehören:
    parser.add_argument("--renderer_script", required=True)
    parser.add_argument("--sequences_root", required=True)
    parser.add_argument("--output_root", required=True)

    parser.add_argument(
        "--output_name",
        default=None,
        help="Optional prefix for the output file name. Default: <sequence>_<animation>.mp4",
    )

    # Alles nach "--" gehört dem Renderer und wird nicht interpretiert.
    parser.add_argument(
        "renderer_args",
        nargs=argparse.REMAINDER,
        help="Arguments passed through to the renderer script after --",
    )

    args = parser.parse_args()

    renderer_args = args.renderer_args

    # argparse.REMAINDER enthält das Trennzeichen "--" selbst nicht immer,
    # aber falls doch, entfernen wir es sicherheitshalber.
    if renderer_args and renderer_args[0] == "--":
        renderer_args = renderer_args[1:]

    batch_render_sequences(
        renderer_script=args.renderer_script,
        sequences_root=args.sequences_root,
        output_root=args.output_root,
        output_name=args.output_name,
        renderer_args=renderer_args,
    )