from pathlib import Path
import pickle
import numpy as np


def load_smpl_j_regressor(smpl_model_path: Path) -> np.ndarray:
    if not smpl_model_path.exists():
        raise FileNotFoundError(smpl_model_path)

    if smpl_model_path.suffix == ".pkl":
        with open(smpl_model_path, "rb") as f:
            smpl_data = pickle.load(f, encoding="latin1")
            J = smpl_data["J_regressor"]
    else:
        raise ValueError(f"Unsupported SMPL model format: {smpl_model_path.suffix}")

    if hasattr(J, "toarray"):
        J = J.toarray()

    J = np.asarray(J, dtype=np.float32)

    if J.shape != (24, 6890):
        raise ValueError(f"Expected J_regressor shape (24, 6890), got {J.shape}")

    return J


def vertices_to_joints(vertices: np.ndarray, J_regressor: np.ndarray) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=np.float32)
    J_regressor = np.asarray(J_regressor, dtype=np.float32)

    return np.einsum("jv,tvc->tjc", J_regressor, vertices)