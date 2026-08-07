from pathlib import Path
import argparse
import joblib
import numpy as np
import torch
import pickle

def load_smpl_j_regressor(smpl_model_path: Path):
    """
    Lädt den offiziellen SMPL J_regressor aus einer SMPL .pkl oder .npz Datei.

    Rückgabe:
      J_regressor: np.ndarray, shape (24, 6890)
    """
    if not smpl_model_path.exists():
        raise FileNotFoundError(f"SMPL model not found: {smpl_model_path}")

    with open(smpl_model_path, "rb") as f:
        smpl_data = pickle.load(f, encoding="latin1")
        J_regressor = smpl_data["J_regressor"]

    J_regressor = J_regressor.toarray()
    J_regressor = np.asarray(J_regressor, dtype=np.float32)

    if J_regressor.shape != (24, 6890):
        raise ValueError(
            f"Expected J_regressor shape (24, 6890), got {J_regressor.shape}"
        )

    return J_regressor

def vertices_to_joints(vertices, J_regressor):
    """
    Berechnet SMPL-Joints aus Mesh-Vertices.

    vertices:    (T, 6890, 3)
    J_regressor: (24, 6890)

    return:
      joints:    (T, 24, 3)
    """
    vertices = np.asarray(vertices, dtype=np.float32)
    J_regressor = np.asarray(J_regressor, dtype=np.float32)

    all_joints = []

    for t in range(vertices.shape[0]):
        frame_vertices = vertices[t]  # (6890, 3)

        frame_joints = J_regressor @ frame_vertices
        # (24, 6890) @ (6890, 3) = (24, 3)

        all_joints.append(frame_joints)

    joints = np.stack(all_joints)  # (T, 24, 3)

    return joints

def load_cape_sequence(cape_root: Path, subj: str, seq_name: str):
    """
    Lädt eine CAPE-Sequenz.

    Erwartet:
      <cape_root>/sequences/<subj>/<seq_name>/*.npz

    Jede .npz enthält u.a.:
      v_posed: (6890, 3) clothed posed mesh
      pose:    (72,)
      transl:  (3,)
    """
    seq_dir = cape_root / "sequences" / subj / seq_name
    npz_files = sorted(seq_dir.glob("*.npz"))

    if len(npz_files) == 0:
        raise FileNotFoundError(f"No .npz files found in: {seq_dir}")

    verts = []
    poses = []
    transls = []

    for fn in npz_files:
        data = np.load(fn)
        verts.append(data["v_posed"])
        poses.append(data["pose"])
        transls.append(data["transl"])

    gt_vertices = np.stack(verts)   # (T, 6890, 3)
    gt_poses = np.stack(poses)      # (T, 72)
    gt_transl = np.stack(transls)   # (T, 3)

    return {
        "vertices": gt_vertices,
        "poses": gt_poses,
        "transl": gt_transl,
        "files": npz_files,
        "seq_dir": seq_dir,
    }


def load_wham_output(path_wham: Path):
    """
    Lädt WHAM wham_output.pkl.
    Erwartet Dict:
      wham[person_id=0]["verts"]     -> (T, 6890, 3)
      wham[person_id=0]["frame_ids"] -> (T,)
    """
    if not path_wham.exists():
        raise FileNotFoundError(f"WHAM output not found: {path_wham}")

    wham = joblib.load(path_wham)

    if not isinstance(wham, dict):
        raise TypeError(f"Expected WHAM output to be dict, got: {type(wham)}")

    print("WHAM top-level person ids:", list(wham.keys()))

    person_id = list(wham.keys())[0]

    if person_id not in wham:
        raise KeyError(f"person_id {person_id} not found. Available: {list(wham.keys())}")

    pred = wham[person_id]

    required = ["verts", "frame_ids"]
    for k in required:
        if k not in pred:
            raise KeyError(f"Missing key in WHAM output: {k}")

    return pred, person_id


def batch_procrustes_align(pred, gt):
    """
    Procrustes Alignment pro Frame.

    pred, gt: (T, N, 3)

    Gibt pred aligned zu gt zurück.
    Entfernt Rotation, Translation und Scale.
    """
    pred = torch.as_tensor(pred).float()
    gt = torch.as_tensor(gt).float()

    mu_pred = pred.mean(dim=1, keepdim=True)
    mu_gt = gt.mean(dim=1, keepdim=True)

    X = pred - mu_pred
    Y = gt - mu_gt

    norm_X = torch.sqrt((X ** 2).sum(dim=(1, 2), keepdim=True))
    norm_Y = torch.sqrt((Y ** 2).sum(dim=(1, 2), keepdim=True))

    eps = 1e-8
    Xn = X / (norm_X + eps)
    Yn = Y / (norm_Y + eps)

    H = Xn.transpose(1, 2) @ Yn
    U, S, Vh = torch.linalg.svd(H)

    R = Vh.transpose(1, 2) @ U.transpose(1, 2)

    # Reflection fix
    det = torch.det(R)
    if torch.any(det < 0):
        Vh[det < 0, -1, :] *= -1
        R = Vh.transpose(1, 2) @ U.transpose(1, 2)

    scale = norm_Y / (norm_X + eps)

    pred_aligned = scale * (X @ R.transpose(1, 2)) + mu_gt
    return pred_aligned


def pa_point_error_mm(pred, gt):
    """
    PA-Fehler in mm.

    pred, gt: (T, N, 3)
    Annahme: Koordinaten sind in Metern.
    """
    gt_t = torch.as_tensor(gt).float()
    pred_aligned = batch_procrustes_align(pred, gt)

    err = torch.linalg.norm(pred_aligned - gt_t, dim=-1)  # (T, N)
    mean_per_frame = err.mean(dim=1)                      # (T,)

    return {
        "mean_mm": mean_per_frame.mean().item() * 1000.0,
        "per_frame_mm": mean_per_frame.cpu().numpy() * 1000.0,
    }

def pa_mpjpe_mm(pred_joints, gt_joints):
    """
    PA-MPJPE in mm.

    pred_joints, gt_joints: (T, J, 3)
    """
    return pa_point_error_mm(pred_joints, gt_joints)


def match_frames(pred_vertices, frame_ids, gt_vertices):
    """
    Matcht CAPE-GT zu WHAM anhand frame_ids.

    Falls frame_ids nicht passen, nimmt es fallback first-min-length.
    """
    frame_ids = np.asarray(frame_ids).astype(int)

    print("\nFrame matching")
    print("-------------")
    print("WHAM pred_vertices:", pred_vertices.shape)
    print("WHAM frame_ids:", frame_ids.shape)
    print("frame_ids min/max:", frame_ids.min(), frame_ids.max())
    print("CAPE gt_vertices:", gt_vertices.shape)

    if frame_ids.max() < len(gt_vertices):
        print("Using WHAM frame_ids to index CAPE.")
        gt_matched = gt_vertices[frame_ids]
        pred_matched = pred_vertices
        used_frame_ids = frame_ids
    else:
        print("WARNING: frame_ids exceed CAPE length.")
        print("Fallback: using first min(len(WHAM), len(CAPE)) frames.")
        T = min(len(pred_vertices), len(gt_vertices))
        pred_matched = pred_vertices[:T]
        gt_matched = gt_vertices[:T]
        used_frame_ids = np.arange(T)

    print("Matched pred:", pred_matched.shape)
    print("Matched gt:  ", gt_matched.shape)

    return pred_matched, gt_matched, used_frame_ids


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cape_root",
        type=str,
        default=r"E:\DLHM\Data\cape_release",
        help="Pfad zum CAPE root directory",
    )
    parser.add_argument(
        "--subj",
        type=str,
        default="03375",
        help="CAPE subject id",
    )
    parser.add_argument(
        "--seq_name",
        type=str,
        default="blazerlong_babysit_trial2",
        help="CAPE sequence name",
    )
    parser.add_argument(
        "--wham_pkl",
        type=str,
        default=r"E:\DLHM\results\WHAM\Tracking\03375\03375_blazerlong_babysit_trial2_front\wham_output.pkl",
        help="Pfad zu WHAM wham_output.pkl",
    )
    parser.add_argument(
        "--clothed",
        action="store_true",
        help="Compare to clothed CAPE sequence instead of unclothed body vertices",
    )
    parser.add_argument(
        "--smpl_model",
        type=str,
        default=r"E:\DLHM\Data\smpl\models\SMPL_MALE.pkl",
        help="Pfad zu SMPL_MALE.pkl / SMPL_FEMALE.pkl",
    )

    args = parser.parse_args()

    cape_root = Path(args.cape_root)
    path_wham = Path(args.wham_pkl)

    if not args.clothed:
        print("Note: This is WHAM SMPL mesh vs CAPE unclothed v_posed mesh.")
        p = cape_root / "unclothed" / args.subj / args.seq_name / f"gt_body_vertices_{args.subj}_{args.seq_name}.npy"
        gt_vertices_all = np.load(p)
    else:
        print("Note: This is WHAM SMPL mesh vs CAPE clothed v_posed mesh.")
        print("\nLoading CAPE")
        print("============")
        cape = load_cape_sequence(cape_root, args.subj, args.seq_name)
        gt_vertices_all = cape["vertices"]
        print("CAPE seq dir:", cape["seq_dir"])
        print("CAPE frames:", len(cape["files"]))
        print("First CAPE frame:", cape["files"][0])
        print("gt_vertices:", cape["vertices"].shape)
        print("gt_poses:   ", cape["poses"].shape)
        print("gt_transl:  ", cape["transl"].shape)

    print("\nLoading WHAM")
    print("============")
    pred, person_id = load_wham_output(path_wham)

    print("WHAM keys:", list(pred.keys()))
    for k, v in pred.items():
        print(k, type(v), getattr(v, "shape", None))

    pred_vertices_all = pred["verts"]
    frame_ids = pred["frame_ids"]

    pred_vertices, gt_vertices, used_frame_ids = match_frames(
        pred_vertices_all,
        frame_ids,
        gt_vertices_all,
    )

    print("\nEvaluation")
    print("==========")

    # PA-MPVPE
    mpvpe_result = pa_point_error_mm(pred_vertices, gt_vertices)
    print(f"PA-MPVPE: {mpvpe_result['mean_mm']:.2f} mm")

    #PA-MPJPE
    print("\nLoading SMPL J_regressor")
    print("========================")

    smpl_model_path = Path(args.smpl_model)
    J_regressor = load_smpl_j_regressor(smpl_model_path)

    print("J_regressor:", J_regressor.shape)

    print("\nRegressing joints from vertices")
    print("===============================")

    pred_joints = vertices_to_joints(pred_vertices, J_regressor)
    gt_joints = vertices_to_joints(gt_vertices, J_regressor)

    print("pred_joints:", pred_joints.shape)
    print("gt_joints:  ", gt_joints.shape)

    mpjpe_result = pa_mpjpe_mm(pred_joints, gt_joints)
    print(f"PA-MPJPE: {mpjpe_result['mean_mm']:.2f} mm")


if __name__ == "__main__":
    main()