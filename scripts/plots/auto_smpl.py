import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

files = {
    "Ground Truth Reconditioning": "E:/DLHM/benchmark/results/auto/GT/easymocap_auto_GT.csv",
    "Mesh Based Reconditioning": "E:/DLHM/benchmark/results/auto/EM/easymocap_auto_EM.csv",
    "Frame Based Reconditioning": "E:/DLHM/benchmark/results/auto/SV/easymocap_auto_SV.csv",
}


def parse_per_frame(values):
    return np.fromstring(values.strip("[]"), sep=" ")


fig, ax = plt.subplots(figsize=(13.33, 7.5))

for file_name, csv_path in files.items():
    df = pd.read_csv(csv_path)

    # Leere Zeilen entfernen
    df = df[df["sequence"].notna()]

    sequences = [
        parse_per_frame(values)
        for values in df["per_frame_pa_mpjpe"]
    ]

    mean_values = np.mean(sequences, axis=0)

    # Frame-Indizes beginnen bei 0
    frames = np.arange(len(mean_values))

    ax.plot(
        frames,
        mean_values,
        linewidth=3,
        label=file_name
    )


# Abschnittsgrenzen bei Frame 35 und 70
for boundary in [35, 70]:
    ax.axvline(
        x=boundary,
        linestyle="--",
        linewidth=2.5,
        alpha=0.75,
        color="black",
        zorder=1
    )


ax.set_xlabel("Frame-Index", fontsize=22, labelpad=12)
ax.set_ylabel("Per-Frame PA-MPVPE", fontsize=22, labelpad=12)

ax.tick_params(axis="both", labelsize=18)

# Linke Grenze exakt bei 0
ax.set_xlim(left=0)

ax.grid(alpha=0.25, linewidth=1.2)

ax.legend(
    loc="upper left",
    fontsize=17,
    frameon=True,
    framealpha=0.95,
    borderpad=0.8,
    labelspacing=0.7,
    handlelength=2.5
)

fig.tight_layout()

fig.savefig(
    "vergleich_durchschnitt_alle_dateien_ppt.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()