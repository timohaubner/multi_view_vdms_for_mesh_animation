import os
from pathlib import Path


def resolve_path(path: str) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(path)))