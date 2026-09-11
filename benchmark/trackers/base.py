from pathlib import Path

from benchmark.core.types import TrackerPrediction


class BaseTrackerLoader:
    tracker_name: str

    def load(self, prediction_paths: list[Path], subject: str, sequence: str) -> TrackerPrediction:
        raise NotImplementedError