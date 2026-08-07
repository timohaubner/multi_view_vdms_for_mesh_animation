import inspect

if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec

import argparse
from pathlib import Path
import pickle
import numpy as np
import torch
from smplx import SMPL



def extract_gt_body_vertices(subj: str, seq_name: str, gender: str, cape_root: Path = Path(r"E:\DLHM\Data\cape_release")):

    param_path = cape_root / "minimal_body_shape" / subj / f"{subj}_param.pkl"

    # HIER model directory rein:
    smpl_model_dir = r"E:\DLHM\Data\smpl\models"


    # CAPE sequence poses/transl laden
    seq_dir = cape_root / "sequences" / subj / seq_name
    npz_files = sorted(seq_dir.glob("*.npz"))

    poses = []
    transls = []

    for fn in npz_files:
        data = np.load(fn)
        poses.append(data["pose"])
        transls.append(data["transl"])

    gt_poses = np.stack(poses)      # (T, 72)
    gt_transl = np.stack(transls)   # (T, 3)

    print("gt_poses:", gt_poses.shape)
    print("gt_transl:", gt_transl.shape)


    # Betas laden
    with open(param_path, "rb") as f:
        params = pickle.load(f, encoding="latin1")

    betas = params["betas"].reshape(1, 10)

    print("betas:", betas.shape)
    print("betas:", betas)



    # SMPL forward
    device = "cpu"
    T = gt_poses.shape[0]

    smpl = SMPL(
        model_path=smpl_model_dir,
        gender=gender,
        batch_size=T,
    ).to(device)

    global_orient = torch.tensor(gt_poses[:, :3], dtype=torch.float32, device=device)
    body_pose = torch.tensor(gt_poses[:, 3:], dtype=torch.float32, device=device)
    transl = torch.tensor(gt_transl, dtype=torch.float32, device=device)

    betas_t = torch.tensor(betas, dtype=torch.float32, device=device).repeat(T, 1)

    with torch.no_grad():
        out = smpl(
            global_orient=global_orient,
            body_pose=body_pose,
            betas=betas_t,
            transl=transl,
            pose2rot=True,
        )

    gt_body_vertices = out.vertices.detach().cpu().numpy()

    print("gt_body_vertices:", gt_body_vertices.shape)



    out_path = cape_root / "unclothed" / subj / seq_name / f"gt_body_vertices_{subj}_{seq_name}.npy"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, gt_body_vertices)

    print("saved:", out_path)





if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gender",
        required=False,
        default="male",
    )

    parser.add_argument(
        "--subj",
        required=False,
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

    args = parser.parse_args()

    extract_gt_body_vertices(subj=args.subj, seq_name=args.seq_name, gender=args.gender)