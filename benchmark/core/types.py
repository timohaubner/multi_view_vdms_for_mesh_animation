from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class GTSequence:
    subject: str
    sequence: str
    gender: str
    vertices: np.ndarray
    frame_ids: np.ndarray
    source_path: Path


@dataclass
class TrackerPrediction:
    tracker_name: str
    subject: str
    sequence: str
    vertices: np.ndarray
    frame_ids: np.ndarray
    source_path: Path
    joints: np.ndarray | None = None
    confidence: np.ndarray | None = None


@dataclass
class MatchedSequence:
    pred_vertices: np.ndarray
    gt_vertices: np.ndarray
    frame_ids: np.ndarray
    match_strategy: str