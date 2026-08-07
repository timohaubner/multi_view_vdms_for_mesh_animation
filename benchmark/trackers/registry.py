from pathlib import Path

from benchmark.trackers.dmmr import DMMRLoader
from benchmark.trackers.easymocap import EasyMocapLoader
from benchmark.trackers.wham import WhamLoader
from benchmark.trackers.multihmr import MultiHMRLoader

TRACKER_LOADERS = {
    "wham": WhamLoader,
    "dmmr": DMMRLoader,
    "easymocap": EasyMocapLoader,
    "multihmr": MultiHMRLoader
}


def get_tracker_loader(name: str, smplx2smpl_path: Path):
    loader_cls = TRACKER_LOADERS.get(name)

    if loader_cls is None:
        raise KeyError(f"Unknown tracker: {name}")

    if name == "multihmr":
        return loader_cls(smplx2smpl_path)

    return loader_cls()