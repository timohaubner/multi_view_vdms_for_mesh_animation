import os
import sys
from glob import glob
from typing import List, Optional

from pathlib import Path
from PIL import Image
from torchvision.transforms import ToTensor

from tqdm import tqdm

sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), "../../")))
import numpy as np
import torch
from fire import Fire
import scripts.demo.sv4d_helpers as sv4d_helpers

from scripts.demo.sv4d_helpers import (
    load_model,
    preprocess_video,
    read_video,
    run_img2vid,
    save_video,
)

from sgm.modules.encoders.modules import VideoPredictionEmbedderWithEncoder


print("DEBUG sv4d_helpers file:", sv4d_helpers.__file__)


sv4d2_configs = {
    "sv4d2": {
        "T": 12,  # number of frames per sample
        "V": 4,  # number of views per sample
        "model_config": "scripts/sampling/configs/sv4d2.yaml",
        "version_dict": {
            "T": 12 * 4,
            "options": {
                "discretization": 1,
                "cfg": 2.0,
                "min_cfg": 2.0,
                "num_views": 4,
                "sigma_min": 0.002,
                "sigma_max": 700.0,
                "rho": 7.0,
                "guider": 2,
                "force_uc_zero_embeddings": [
                    "cond_frames",
                    "cond_frames_without_noise",
                    "cond_view",
                    "cond_motion",
                ],
                "additional_guider_kwargs": {
                    "additional_cond_keys": ["cond_view", "cond_motion"]
                },
            },
        },
    },
    "sv4d2_8views": {
        "T": 5,  # number of frames per sample
        "V": 8,  # number of views per sample
        "model_config": "scripts/sampling/configs/sv4d2_8views.yaml",
        "version_dict": {
            "T": 5 * 8,
            "options": {
                "discretization": 1,
                "cfg": 2.5,
                "min_cfg": 1.5,
                "num_views": 8,
                "sigma_min": 0.002,
                "sigma_max": 700.0,
                "rho": 7.0,
                "guider": 5,
                "force_uc_zero_embeddings": [
                    "cond_frames",
                    "cond_frames_without_noise",
                    "cond_view",
                    "cond_motion",
                ],
                "additional_guider_kwargs": {
                    "additional_cond_keys": ["cond_view", "cond_motion"]
                },
            },
        },
    },
}

def apply_video_crop_to_reference_image(
    image: Image.Image,
    crop_info: dict,
    H: int,
    W: int,
):
    """
    Applies the exact crop/pad/resize transform from preprocess_video
    to a reference image.

    This assumes the reference image has the same original canvas framing
    as the input video frames.
    """

    if image.mode == "RGB":
        image = image.convert("RGBA")
    elif image.mode != "RGBA":
        image = image.convert("RGBA")

    image_arr = np.array(image)

    x = crop_info["x"]
    y = crop_info["y"]
    w = crop_info["w"]
    h = crop_info["h"]
    box_size = crop_info["box_size"]
    side_len = crop_info["side_len"]

    padded_image = np.zeros((side_len, side_len, 4), dtype=np.uint8)
    center = side_len // 2

    box_size_w = min(w, box_size)
    box_size_h = min(h, box_size)

    crop = image_arr[x : x + w, y : y + h]

    padded_image[
        center - box_size_w // 2 : center - box_size_w // 2 + box_size_w,
        center - box_size_h // 2 : center - box_size_h // 2 + box_size_h,
    ] = crop[:box_size_w, :box_size_h]

    rgba = Image.fromarray(padded_image).resize((W, H), Image.LANCZOS)
    rgba_arr = np.array(rgba) / 255.0

    rgb = rgba_arr[..., :3] * rgba_arr[..., -1:] + (1 - rgba_arr[..., -1:])
    image = (rgb * 255).astype(np.uint8)

    return Image.fromarray(image)



def load_reference_views(
    reference_views_path: str,
    expected_views: int,
    H: int,
    W: int,
    device: str = "cuda",
    crop_info: Optional[dict] = None,
    debug_output_folder: Optional[str] = None,
    base_count: int = 0,
):
    """
    Loads fixed-name PNG reference images for novel views.

    Expected filenames:
      reference_view_1.png -> view 1
      reference_view_2.png -> view 2
      ...
      reference_view_N.png -> view N

    For sv4d2:
      N = 4
    """

    path = Path(reference_views_path)

    if not path.is_dir():
        raise ValueError(
            f"reference_views_path must be a folder containing fixed PNG files. "
            f"Got: {reference_views_path}"
        )

    img_paths = [
        path / f"reference_view_{i}.png"
        for i in range(1, expected_views + 1)
    ]

    missing = [p.name for p in img_paths if not p.is_file()]
    if missing:
        raise ValueError(
            "Missing reference PNG files in "
            f"{reference_views_path}: {missing}. "
            f"Expected exactly: {[p.name for p in img_paths]}"
        )

    refs = []
    for i, img_path in enumerate(img_paths, start=1):
        print(f"Loading reference view {i}: {img_path}")

        img = Image.open(img_path).convert("RGBA")

        if img.size != (W, H):
            raise ValueError(
                f"{img_path} must be exactly {W}x{H}, got {img.size}. "
                "Render all reference views with the same square canvas as the input video."
            )

        tensor = ToTensor()(img).unsqueeze(0).to(device)
        tensor = tensor * 2.0 - 1.0
        refs.append(tensor)

    return refs


def sample(
    input_path: str = "assets/sv4d_videos/camel.gif",  # Can either be image file or folder with image files
    reference_views_path: Optional[str] = None,
    model_path: Optional[str] = "checkpoints/sv4d2.safetensors",
    output_folder: Optional[str] = "outputs",
    num_steps: Optional[int] = 50,
    img_size: int = 576,  # image resolution
    n_frames: int = 21,  # number of input and output video frames
    seed: int = 23,
    encoding_t: int = 8,  # Number of frames encoded at a time! This eats most VRAM. Reduce if necessary.
    decoding_t: int = 4,  # Number of frames decoded at a time! This eats most VRAM. Reduce if necessary.
    device: str = "cuda",
    elevations_deg: Optional[List[float]] = 0.0,
    azimuths_deg: Optional[List[float]] = None,
    image_frame_ratio: Optional[float] = 0.9,
    verbose: Optional[bool] = False,
    remove_bg: bool = False,
):
    """
    Simple script to generate multiple novel-view videos conditioned on a video `input_path` or multiple frames, one for each
    image file in folder `input_path`. If you run out of VRAM, try decreasing `decoding_t` and `encoding_t`.
    """
    # Set model config
    assert os.path.basename(model_path) in [
        "sv4d2.safetensors",
        "sv4d2_8views.safetensors",
    ]
    sv4d2_model = os.path.splitext(os.path.basename(model_path))[0]
    config = sv4d2_configs[sv4d2_model]
    print(sv4d2_model, config)
    T = config["T"]
    V = config["V"]
    model_config = config["model_config"]
    version_dict = config["version_dict"]
    F = 8  # vae factor to downsize image->latent
    C = 4
    H, W = img_size, img_size
    n_views = V + 1  # number of output video views (1 input view + 8 novel views)
    subsampled_views = np.arange(n_views)
    version_dict["H"] = H
    version_dict["W"] = W
    version_dict["C"] = C
    version_dict["f"] = F
    version_dict["options"]["num_steps"] = num_steps

    torch.manual_seed(seed)
    output_folder = os.path.join(output_folder, sv4d2_model)
    os.makedirs(output_folder, exist_ok=True)

    # Read input video frames i.e. images at view 0
    print(f"Reading {input_path}")
    base_count = len(glob(os.path.join(output_folder, "*.mp4"))) // n_views

    '''
    ret = preprocess_video(
        input_path,
        remove_bg=remove_bg,
        n_frames=n_frames,
        W=W,
        H=H,
        output_folder=output_folder,
        image_frame_ratio=image_frame_ratio,
        base_count=base_count,
    )
    
    print("DEBUG preprocess_video return type:", type(ret))
    print("DEBUG preprocess_video return repr:", repr(ret))

    try:
        print("DEBUG preprocess_video return len:", len(ret))
    except Exception as e:
        print("DEBUG no len:", e)
    
    '''

    processed_input_path = input_path

    images_v0 = read_video(processed_input_path, n_frames=n_frames, device=device)
    images_t0 = torch.zeros(n_views, 3, H, W).float().to(device)

    if reference_views_path is not None:
        reference_views = load_reference_views(
            reference_views_path=reference_views_path,
            expected_views=V,
            H=H,
            W=W,
            device=device,
            debug_output_folder=output_folder,
            base_count=base_count,
        )

        # view 0 remains the input video view.
        # views 1..V are initialized from reference images.
        for i, ref_img in enumerate(reference_views):
            images_t0[i + 1] = ref_img[0]

    # Get camera viewpoints
    if isinstance(elevations_deg, float) or isinstance(elevations_deg, int):
        elevations_deg = [elevations_deg] * n_views
    assert (
        len(elevations_deg) == n_views
    ), f"Please provide 1 value, or a list of {n_views} values for elevations_deg! Given {len(elevations_deg)}"
    if azimuths_deg is None:
        # azimuths_deg = np.linspace(0, 360, n_views + 1)[1:] % 360
        azimuths_deg = (
            np.array([0, 60, 120, 180, 240])
            if sv4d2_model == "sv4d2"
            else np.array([0, 30, 75, 120, 165, 210, 255, 300, 330])
        )
    assert (
        len(azimuths_deg) == n_views
    ), f"Please provide a list of {n_views} values for azimuths_deg! Given {len(azimuths_deg)}"
    polars_rad = np.array([np.deg2rad(90 - e) for e in elevations_deg])
    azimuths_rad = np.array(
        [np.deg2rad((a - azimuths_deg[-1]) % 360) for a in azimuths_deg]
    )

    # Initialize image matrix
    img_matrix = [[None] * n_views for _ in range(n_frames)]
    for i, v in enumerate(subsampled_views):
        img_matrix[0][i] = images_t0[v].unsqueeze(0)
    for t in range(n_frames):
        img_matrix[t][0] = images_v0[t]

    # Load SV4D++ model
    model, _ = load_model(
        model_config,
        device,
        version_dict["T"],
        num_steps,
        verbose,
        model_path,
    )
    model.en_and_decode_n_samples_a_time = decoding_t
    for emb in model.conditioner.embedders:
        if isinstance(emb, VideoPredictionEmbedderWithEncoder):
            emb.en_and_decode_n_samples_a_time = encoding_t

    # Sampling novel-view videos
    v0 = 0
    view_indices = np.arange(V) + 1
    t0_list = (
        range(0, n_frames, T-1)
        if sv4d2_model == "sv4d2"
        else range(0, n_frames - T + 1, T - 1)
    )
    for t0 in tqdm(t0_list):
        if t0 + T > n_frames:
            t0 = n_frames - T
        frame_indices = t0 + np.arange(T)
        print(f"Sampling frames {frame_indices}")
        image = img_matrix[t0][v0]
        cond_motion = torch.cat([img_matrix[t][v0] for t in frame_indices], 0)
        cond_view = torch.cat([img_matrix[t0][v] for v in view_indices], 0)
        polars = polars_rad[subsampled_views[1:]][None].repeat(T, 0).flatten()
        azims = azimuths_rad[subsampled_views[1:]][None].repeat(T, 0).flatten()
        polars = (polars - polars_rad[v0] + torch.pi / 2) % (torch.pi * 2)
        azims = (azims - azimuths_rad[v0]) % (torch.pi * 2)

        cond_mv = True if reference_views_path is not None else (False if t0 == 0 else True)
        samples = run_img2vid(
            version_dict,
            model,
            image,
            seed,
            polars,
            azims,
            cond_motion,
            cond_view,
            decoding_t,
            cond_mv=cond_mv,
        )
        samples = samples.view(T, V, 3, H, W)

        for i, t in enumerate(frame_indices):
            for j, v in enumerate(view_indices):
                img_matrix[t][v] = samples[i, j][None] * 2 - 1

    # Save output videos
    for v in view_indices:
        vid_file = os.path.join(output_folder, f"{base_count:06d}_v{v:03d}.mp4")
        print(f"Saving {vid_file}")
        save_video(
            vid_file,
            [img_matrix[t][v] for t in range(n_frames) if img_matrix[t][v] is not None],
        )


if __name__ == "__main__":
    Fire(sample)