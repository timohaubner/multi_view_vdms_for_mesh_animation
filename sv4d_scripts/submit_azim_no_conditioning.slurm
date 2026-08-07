#!/bin/bash
#SBATCH --job-name=sv4d2-azim
#SBATCH --partition=24g
#SBATCH --qos=students_normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --output=logs/sv4d2-azim-%j.out
#SBATCH --error=logs/sv4d2-azim-%j.err

set -euo pipefail

###############################################################################
# USER SETTINGS
###############################################################################

# Repository
REPO_DIR="/mnt/home/timohaubner/projects/sv4d20/generative-models"

# Eingabe- und Ausgabeordner
INPUT_DIR="/mnt/home/timohaubner/projects/data/azim_first/Azim_videos"
OUTPUT_DIR="/mnt/home/timohaubner/projects/sv4d20/generative-models/outputs/azim_unconditioned"

# Stuttgart Working Image
IMAGE_NAME="localhost/sv4d2:stuttgart-xformers"
IMAGE_TAR="/mnt/home/timohaubner/sv4d2-stuttgart-xformers.tar"

###############################################################################
# AZIMUTH SETTINGS
###############################################################################

# Azimut des Eingabevideos.
#
# Erwartetes Dateimuster:
#   0_00032_longshort_flying_eagle.mp4
INPUT_VIDEO_AZIM=0

# Vier Novel Views für sv4d2.safetensors.
#
# Diese Werte definieren nur die Kamerapositionen der zu generierenden Views.
# Es werden keine PNG-Referenzbilder geladen.
OUTPUT_AZIMS=(30 60 300 330)

# Daraus wird:
#   [0,25,50,310,335]
#
# Der erste Wert gehört zum Eingabevideo.
# Die restlichen vier Werte gehören zu den vier generierten Novel Views.
OUTPUT_AZIMS_STR="$(IFS=,; echo "${OUTPUT_AZIMS[*]}")"
AZIMUTHS_DEG_STR="${INPUT_VIDEO_AZIM},${OUTPUT_AZIMS_STR}"

###############################################################################
# SV4D2 PARAMETER
###############################################################################

IMG_SIZE=512
N_FRAMES=36
ENCODING_T=1
DECODING_T=1
NUM_STEPS=50

# Zusätzliche optionale Fire-Argumente.
#
# Beispiel:
# EXTRA_ARGS="--seed 42"
EXTRA_ARGS=""

# Für Stuttgart/rootless Podman
PODMAN_EXTRA_ARGS=(
    --storage-opt ignore_chown_errors=true
)

###############################################################################
# HOST SETUP
###############################################################################

mkdir -p "${REPO_DIR}/logs"
mkdir -p "${OUTPUT_DIR}"

echo "============================================================"
echo "SLURM job: ${SLURM_JOB_ID:-manual}"
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Repo: ${REPO_DIR}"
echo "Input root: ${INPUT_DIR}"
echo "Output root: ${OUTPUT_DIR}"
echo "Image: ${IMAGE_NAME}"
echo "Image tar: ${IMAGE_TAR}"
echo "Input video azimuth: ${INPUT_VIDEO_AZIM}"
echo "Output azimuths: ${OUTPUT_AZIMS_STR}"
echo "Azimuths passed to sampler: [${AZIMUTHS_DEG_STR}]"
echo "N_FRAMES: ${N_FRAMES}"
echo "IMG_SIZE: ${IMG_SIZE}"
echo "ENCODING_T: ${ENCODING_T}"
echo "DECODING_T: ${DECODING_T}"
echo "NUM_STEPS: ${NUM_STEPS}"
echo "Reference views: disabled"
echo "Conditioning:"
echo "  first temporal window: cond_mv=False"
echo "  later temporal windows: cond_mv=True"
echo "============================================================"

nvidia-smi || true

cd "${REPO_DIR}"

export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

echo "PYTHONPATH=${PYTHONPATH}"

if [[ ! -d "${INPUT_DIR}" ]]; then
    echo "ERROR: INPUT_DIR existiert nicht: ${INPUT_DIR}"
    exit 1
fi

if [[ ! -f "${IMAGE_TAR}" ]]; then
    echo "ERROR: IMAGE_TAR existiert nicht: ${IMAGE_TAR}"
    exit 1
fi

###############################################################################
# LOAD CONTAINER IMAGE
###############################################################################

if ! podman images --format '{{.Repository}}:{{.Tag}}' |
    grep -qx "${IMAGE_NAME}"; then

    echo "Image ${IMAGE_NAME} nicht gefunden."
    echo "Lade Image aus ${IMAGE_TAR} ..."

    podman "${PODMAN_EXTRA_ARGS[@]}" load -i "${IMAGE_TAR}"
else
    echo "Image ${IMAGE_NAME} ist bereits vorhanden."
fi

###############################################################################
# RUN INSIDE CONTAINER
###############################################################################

podman "${PODMAN_EXTRA_ARGS[@]}" run --rm \
    --name "sv4d2-folder-${SLURM_JOB_ID:-manual}" \
    -v /mnt:/mnt:rw \
    -v /tmp:/tmp:rw \
    -w "${REPO_DIR}" \
    --device=nvidia.com/gpu=all \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e CUDA_VISIBLE_DEVICES=0 \
    -e PYTHONPATH="${REPO_DIR}" \
    -e SLURM_JOB_ID="${SLURM_JOB_ID:-manual}" \
    -e REPO_DIR="${REPO_DIR}" \
    -e INPUT_DIR="${INPUT_DIR}" \
    -e OUTPUT_DIR="${OUTPUT_DIR}" \
    -e IMG_SIZE="${IMG_SIZE}" \
    -e N_FRAMES="${N_FRAMES}" \
    -e ENCODING_T="${ENCODING_T}" \
    -e DECODING_T="${DECODING_T}" \
    -e NUM_STEPS="${NUM_STEPS}" \
    -e EXTRA_ARGS="${EXTRA_ARGS}" \
    -e INPUT_VIDEO_AZIM="${INPUT_VIDEO_AZIM}" \
    -e OUTPUT_AZIMS_STR="${OUTPUT_AZIMS_STR}" \
    -e AZIMUTHS_DEG_STR="${AZIMUTHS_DEG_STR}" \
    -e PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128" \
    --network=host \
    "${IMAGE_NAME}" \
    bash -lc '
set -euo pipefail

echo "============================================================"
echo "Inside container"
echo "Hostname: $(hostname)"
echo "Working dir: $(pwd)"
echo "Input root: ${INPUT_DIR}"
echo "Output root: ${OUTPUT_DIR}"
echo "Input video azimuth: ${INPUT_VIDEO_AZIM}"
echo "Output azimuths: ${OUTPUT_AZIMS_STR}"
echo "Azimuths passed to sampler: [${AZIMUTHS_DEG_STR}]"
echo "Reference views: disabled"
echo "============================================================"

###############################################################################
# CUDA CHECK
###############################################################################

python - <<PY
import torch

print("cuda:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
    print(
        "vram_gb:",
        round(
            torch.cuda.get_device_properties(0).total_memory / 1024**3,
            2,
        ),
    )
PY

cd "${REPO_DIR}"

###############################################################################
# CHECK MODEL
###############################################################################

if [[ ! -f checkpoints/sv4d2.safetensors ]]; then
    echo "ERROR: checkpoints/sv4d2.safetensors nicht gefunden."
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

###############################################################################
# FIND INPUT VIDEOS
###############################################################################

echo "Suche Input-Videos nach Muster:"
echo "  ${INPUT_DIR}/**/${INPUT_VIDEO_AZIM}_*.mp4"

mapfile -d "" VIDEOS < <(
    find "${INPUT_DIR}" \
        -type f \
        -iname "${INPUT_VIDEO_AZIM}_*.mp4" \
        -print0 |
        sort -z
)

echo "Gefundene Input-Videos: ${#VIDEOS[@]}"

if [[ "${#VIDEOS[@]}" -eq 0 ]]; then
    echo "Keine passenden Input-Videos gefunden."
    exit 0
fi

###############################################################################
# PROCESS VIDEOS
###############################################################################

for VIDEO in "${VIDEOS[@]}"; do
    VIDEO_DIR="$(dirname "${VIDEO}")"
    BASENAME="$(basename "${VIDEO}")"
    STEM="${BASENAME%.*}"

    if [[ "${VIDEO_DIR}" == "${INPUT_DIR}" ]]; then
        REL_DIR="_root"
    else
        REL_DIR="${VIDEO_DIR#${INPUT_DIR}/}"
    fi

    VIDEO_OUTPUT="${OUTPUT_DIR}/${REL_DIR}"
    LOG_FILE="${VIDEO_OUTPUT}/${STEM}.log"

    mkdir -p "${VIDEO_OUTPUT}"

    echo "============================================================"
    echo "Ordner: ${VIDEO_DIR}"
    echo "Input video: ${VIDEO}"
    echo "Output: ${VIDEO_OUTPUT}"
    echo "Log: ${LOG_FILE}"
    echo "Azimuths: [${AZIMUTHS_DEG_STR}]"
    echo "Reference views: disabled"
    echo "============================================================"

    echo "Starte Sampling ..."

    set +e

    # Es wird absichtlich KEIN --reference_views_path übergeben.
    #
    # Mit der vorhandenen Python-Logik:
    #
    #   cond_mv = (
    #       True
    #       if reference_views_path is not None
    #       else (False if t0 == 0 else True)
    #   )
    #
    # gilt dadurch:
    #
    #   t0 == 0  -> cond_mv=False
    #   t0 > 0   -> cond_mv=True
    #
    python scripts/sampling/simple_video_sample_4d2.py \
        --input_path "${VIDEO}" \
        --model_path "checkpoints/sv4d2.safetensors" \
        --output_folder "${VIDEO_OUTPUT}" \
        --n_frames "${N_FRAMES}" \
        --encoding_t "${ENCODING_T}" \
        --decoding_t "${DECODING_T}" \
        --img_size "${IMG_SIZE}" \
        --num_steps "${NUM_STEPS}" \
        --azimuths_deg "[${AZIMUTHS_DEG_STR}]" \
        ${EXTRA_ARGS} \
        > "${LOG_FILE}" 2>&1

    STATUS=$?

    set -e

    if [[ "${STATUS}" -ne 0 ]]; then
        echo "FEHLER bei ${VIDEO}"
        echo "Exit-Code: ${STATUS}"
        echo "Log: ${LOG_FILE}"
        echo "Letzte 100 Log-Zeilen:"
        tail -n 100 "${LOG_FILE}" || true
        echo "Mache mit dem nächsten Video weiter."
    else
        echo "Fertig: ${VIDEO}"
        echo "Output liegt in: ${VIDEO_OUTPUT}"
    fi

    ############################################################################
    # CLEANUP BETWEEN VIDEOS
    ############################################################################

    python - <<PY || true
import gc
import torch

gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("CUDA cache cleared")
PY

done

echo "============================================================"
echo "Alle passenden Videos wurden verarbeitet."
echo "Output root: ${OUTPUT_DIR}"
echo "============================================================"
'