import os
from pathlib import Path


def resolve_path(path: str | Path) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(path))))


def resolve_output_csv(path: str | Path) -> Path:
    path = resolve_path(path)

    if path.suffix.lower() == ".csv":
        return path

    return path / "video_metrics.csv"