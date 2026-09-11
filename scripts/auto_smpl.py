import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


files = {
    "Ground Truth": "E:/DLHM/benchmark/results/auto/GT/easymocap_auto_GT.csv",
    "Frame-Level": "E:/DLHM/benchmark/results/auto/SV/easymocap_auto_SV.csv",
    "Mesh-Level": "E:/DLHM/benchmark/results/auto/EM/easymocap_auto_EM.csv",
}

metric_column = "per_frame_pa_mpvpe"
metric_label = "PA-MPVPE [mm]"


def parse_per_frame(values):
    return np.fromstring(values.strip("[]"), sep=" ")


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 10,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

fig, ax = plt.subplots(figsize=(6.8, 3.8))

for label, csv_path in files.items():
    df = pd.read_csv(csv_path)
    df = df[df["sequence"].notna()]

    sequences = [
        parse_per_frame(values)
        for values in df[metric_column]
    ]

    lengths = {len(sequence) for sequence in sequences}

    if len(lengths) != 1:
        raise ValueError(
            f"Sequences in '{csv_path}' have different lengths: {lengths}"
        )

    sequences = np.stack(sequences)
    mean_values = np.mean(sequences, axis=0)
    frames = np.arange(len(mean_values))

    ax.plot(
        frames,
        mean_values,
        label=label,
        zorder=3,
    )

boundaries = [
    (35, r"$\tau_4 = 35$"),
    (70, r"$\tau_8 = 70$"),
]

for boundary, boundary_label in boundaries:
    ax.axvline(
        x=boundary,
        color="black",
        linestyle="--",
        linewidth=1.2,
        alpha=0.65,
        zorder=1,
    )

    ax.text(
        boundary,
        0.97,
        boundary_label,
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=9,
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.8,
            "pad": 1.0,
        },
    )

ax.set_xlabel("Frame")
ax.set_ylabel(metric_label)
ax.set_xlim(0, len(mean_values) - 1)

ax.grid(
    True,
    linewidth=0.6,
    alpha=0.25,
)
ax.set_axisbelow(True)

ax.legend(
    loc="upper left",
    frameon=True,
    framealpha=0.95,
    borderpad=0.5,
    labelspacing=0.4,
    handlelength=2.5,
)

fig.tight_layout(pad=0.5)

fig.savefig(
    "reconditioning_per_frame_pa_mpvpe.pdf",
    bbox_inches="tight",
)
fig.savefig(
    "reconditioning_per_frame_pa_mpvpe.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
)

plt.show()