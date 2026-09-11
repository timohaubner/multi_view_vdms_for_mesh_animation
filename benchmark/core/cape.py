from pathlib import Path
import numpy as np

from benchmark.core.types import GTSequence


def load_cape_gt(
    cape_root: Path,
    subject: str,
    sequence: str,
    gt_type: str = "unclothed",
    gender: str = "male",
) -> GTSequence:
    if gt_type == "unclothed":
        path = (
            cape_root
            / "unclothed"
            / subject
            / sequence
            / f"gt_body_vertices_{subject}_{sequence}.npy"
        )

        if not path.exists():
            raise FileNotFoundError(path)

        vertices = np.load(path).astype(np.float32)

        return GTSequence(
            subject=subject,
            sequence=sequence,
            vertices=vertices,
            frame_ids=np.arange(len(vertices)),
            source_path=path,
            gender=gender,
        )

    if gt_type == "clothed":
        seq_dir = cape_root / "sequences" / subject / sequence
        files = sorted(seq_dir.glob("*.npz"))

        if not files:
            raise FileNotFoundError(seq_dir)

        vertices = []

        for fn in files:
            data = np.load(fn)
            vertices.append(data["v_posed"])

        vertices = np.stack(vertices).astype(np.float32)

        return GTSequence(
            subject=subject,
            sequence=sequence,
            vertices=vertices,
            frame_ids=np.arange(len(vertices)),
            source_path=seq_dir,
            gender=gender,
        )

    raise ValueError(f"Unknown gt_type: {gt_type}")