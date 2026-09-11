from pathlib import Path
import pickle
import numpy as np


def load_smpl_j_regressor(smpl_model_path: Path) -> np.ndarray:
    if not smpl_model_path.exists():
        raise FileNotFoundError(smpl_model_path)

    if smpl_model_path.suffix == ".pkl":
        with smpl_model_path.open("rb") as file:
            smpl_data = pickle.load(file, encoding="latin1")
            j_regressor = smpl_data["J_regressor"]
    else:
        raise ValueError(f"Unsupported SMPL model format: {smpl_model_path.suffix}")

    if hasattr(j_regressor, "toarray"):
        j_regressor = j_regressor.toarray()

    j_regressor = np.asarray(j_regressor, dtype=np.float32)

    if j_regressor.shape != (24, 6890):
        raise ValueError(f"Expected J_regressor shape (24, 6890), got {j_regressor.shape}")

    return j_regressor


def vertices_to_joints(vertices: np.ndarray, j_regressor: np.ndarray) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=np.float32)
    j_regressor = np.asarray(j_regressor, dtype=np.float32)

    return np.einsum("jv,tvc->tjc", j_regressor, vertices)