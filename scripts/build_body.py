import inspect

if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
from smplx import SMPL


def extract_gt_body_vertices(
    subj: str,
    seq_name: str,
    gender: str,
    cape_root: Path = Path(r"E:\DLHM\Data\cape_release"),
):
    param_path = cape_root / "minimal_body_shape" / subj / f"{subj}_param.pkl"
    smpl_model_dir = r"E:\DLHM\Data\smpl\models"

    seq_dir = cape_root / "sequences" / subj / seq_name
    npz_files = sorted(seq_dir.glob("*.npz"))

    poses = []
    transls = []

    for npz_path in npz_files:
        data = np.load(npz_path)
        poses.append(data["pose"])
        transls.append(data["transl"])

    gt_poses = np.stack(poses)
    gt_transl = np.stack(transls)

    print("gt_poses:", gt_poses.shape)
    print("gt_transl:", gt_transl.shape)

    with param_path.open("rb") as file:
        params = pickle.load(file, encoding="latin1")

    betas = params["betas"].reshape(1, 10)

    print("betas:", betas.shape)
    print("betas:", betas)

    device = "cpu"
    num_frames = gt_poses.shape[0]

    smpl = SMPL(
        model_path=smpl_model_dir,
        gender=gender,
        batch_size=num_frames,
    ).to(device)

    global_orient = torch.tensor(
        gt_poses[:, :3],
        dtype=torch.float32,
        device=device,
    )
    body_pose = torch.tensor(
        gt_poses[:, 3:],
        dtype=torch.float32,
        device=device,
    )
    transl = torch.tensor(
        gt_transl,
        dtype=torch.float32,
        device=device,
    )
    betas_tensor = torch.tensor(
        betas,
        dtype=torch.float32,
        device=device,
    ).repeat(num_frames, 1)

    with torch.no_grad():
        output = smpl(
            global_orient=global_orient,
            body_pose=body_pose,
            betas=betas_tensor,
            transl=transl,
            pose2rot=True,
        )

    gt_body_vertices = output.vertices.detach().cpu().numpy()

    print("gt_body_vertices:", gt_body_vertices.shape)

    output_path = (
        cape_root
        / "unclothed"
        / subj
        / seq_name
        / f"gt_body_vertices_{subj}_{seq_name}.npy"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, gt_body_vertices)

    print("saved:", output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gender",
        default="male",
    )
    parser.add_argument(
        "--subj",
        type=str,
        default="03375",
        help="CAPE subject ID",
    )
    parser.add_argument(
        "--seq_name",
        type=str,
        default="blazerlong_babysit_trial2",
        help="CAPE sequence name",
    )

    args = parser.parse_args()

    extract_gt_body_vertices(
        subj=args.subj,
        seq_name=args.seq_name,
        gender=args.gender,
    )