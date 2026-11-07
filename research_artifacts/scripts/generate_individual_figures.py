"""Generate individual 300 DPI publication-quality figures (Chapter 4 benchmarks).

Design: dataviz skill (compute/validate palettes, recessive chrome, one axis per
chart, fixed categorical order, selective direct labels, text tokens for
non-data text, thin marks). 1 result = 1 standalone figure.

Outputs (PNG + JPG, 300 DPI) in .riset/Paper/figures/:
  fig4_1_architecture               Computational architecture and data flow
  fig4_2_sigma_sensitivity          Manifold dispersion (sigma) sensitivity
  fig4_3_scale_latency              Query latency scaling (50k-1M)
  fig4_4_scale_memory               RAM bitcode vs disk memmap vs peak working set
  fig4_5_cv_pools_pingpong          5-fold dialog-split cross-validation
  fig4_6_language_segments          Retrieval by language segment
  fig4_7_itq_sample_size_ablation   ITQ fit sample-size ablation
  fig4_8_pareto_recall_latency      Pareto frontier: recall vs query latency
  fig4_9_isolated_ram_workingset    Isolated subprocess peak working set (50k)
  fig4_10_downstream_needle_qa      Downstream Needle Hit@10 across 7 methods
  fig4_11_temporal_decay_retention  Retention by fact age (turn distance)
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

# ---------------------------------------------------------------------------
# Paths and data
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
JSON_PATH = SCRIPT_DIR / "benchmarks" / "experiment_results_bab4.json"
FIG_DIR = SCRIPT_DIR / "Paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

with open(JSON_PATH, "r", encoding="utf-8") as f:
    DATA = json.load(f)

RECORDS = DATA["task2_scale_and_memory_benchmarks"]["baseline_records_isolated"]

# ---------------------------------------------------------------------------
# Dataviz-aligned style (palette.md light chart chrome; system sans)
# Validated categorical slots (light): #2a78d6,#eb6834,#1baf7a,#eda100,#e87ba4,#008300
#   CVD worst adjacent 9.1 (protan), normal-vision worst adjacent 19.6 — PASS.
#   Relief (direct labels/table values) required for sub-3:1 slots on surface.
# ---------------------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"  # hairline, solid
AXIS = "#c3c2b7"
BORDER = "#0b0b0b1a"

SERIES = [
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
]
# Extra hues for >6-series figure (4.8): violet, red — fixed assignment, not cycled
SERIES_EXT = SERIES + ["#6250d6", "#e34948"]

DIVERGING_NEG = "#d03b3b"
DIVERGING_NEU = "#898781"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": [
            "DejaVu Sans",
            "Arial",
            "Helvetica",
            "Liberation Sans",
            "sans-serif",
        ],
        "font.size": 9.0,
        "axes.titlesize": 10.5,
        "axes.labelsize": 9.5,
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.facecolor": SURFACE,
        "axes.labelcolor": INK,
        "axes.grid": False,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",  # solid hairline (no dashed grid)
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "text.color": INK,
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "savefig.edgecolor": SURFACE,
    }
)


def fig_ax(figsize, grid_axis=None):
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    if grid_axis:
        ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.6, linestyle="-", zorder=0)
        ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig, ax


def save_plot(fig, name):
    plt.tight_layout()
    p_png = FIG_DIR / f"{name}.png"
    p_jpg = FIG_DIR / f"{name}.jpg"
    fig.savefig(p_png, dpi=300, facecolor=SURFACE, edgecolor="none", bbox_inches="tight")
    fig.savefig(p_jpg, dpi=300, facecolor=SURFACE, edgecolor="none", bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {name}.png/.jpg (300 DPI)")


def fmt_ms(x, _pos=None):
    if x >= 1000:
        return f"{x/1000:.1f}k"
    if x >= 100:
        return f"{x:.0f}"
    if x >= 10:
        return f"{x:.0f}"
    if x >= 1:
        return f"{x:.1f}"
    return f"{x:.2f}"


def rec(method_id, scale, dataset="skr"):
    return next(
        r
        for r in RECORDS
        if r["method_id"] == method_id and r["scale"] == scale and r.get("dataset") == dataset
    )


# ---------------------------------------------------------------------------
# Figure 4.1: Computational architecture and data flow
# ---------------------------------------------------------------------------
def generate_fig4_1_architecture():
    fig, ax = plt.subplots(figsize=(14.2, 6.4), dpi=300)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    boxes = [
        {
            "x": 2,
            "y": 55,
            "w": 16,
            "h": 38,
            "title": "1. Ingest & Encode",
            "lines": [
                "Bilingual dialogue (ID-EN)",
                "Natural turn format",
                "MiniLM 384-D embedding",
                "L2 normalization",
            ],
            "fill": "#eef4fc",
            "edge": SERIES[0],
        },
        {
            "x": 22,
            "y": 55,
            "w": 17,
            "h": 38,
            "title": "2. ITQ Projection",
            "lines": [
                "Mean centering (mu)",
                "PCA spectral projection",
                "Procrustes SVD rotation",
                "W matrix (128/256-bit)",
            ],
            "fill": "#f7f1e8",
            "edge": SERIES[1],
        },
        {
            "x": 43,
            "y": 55,
            "w": 17,
            "h": 38,
            "title": "3. Dual Storage",
            "lines": [
                "RAM: packed uint64 codes",
                "(15.26 MB / 1M vectors)",
                "Disk: np.memmap store",
                "Float16 dense vectors",
            ],
            "fill": "#eef8f3",
            "edge": SERIES[2],
        },
        {
            "x": 64,
            "y": 55,
            "w": 16,
            "h": 38,
            "title": "4. POPCNT Scan",
            "lines": [
                "Column-wise XOR",
                "np.bitwise_count (SIMD)",
                "Linear argpartition",
                "Candidate pool N_pool",
            ],
            "fill": "#fdf6e8",
            "edge": SERIES[3],
        },
        {
            "x": 84,
            "y": 55,
            "w": 14,
            "h": 38,
            "title": "5. Episodic Score",
            "lines": [
                "Exact cosine re-rank",
                "Half-life decay 2^(-dt/h)",
                "S = alpha*cos + (1-alpha)*t",
                "Context prompt injection",
            ],
            "fill": "#f6f0fb",
            "edge": SERIES_EXT[6],
        },
    ]

    for b in boxes:
        rect = patches.FancyBboxPatch(
            (b["x"], b["y"]),
            b["w"],
            b["h"],
            boxstyle="round,pad=0.5,rounding_size=1.2",
            facecolor=b["fill"],
            edgecolor=b["edge"],
            linewidth=1.4,
            zorder=2,
        )
        ax.add_patch(rect)
        ax.text(
            b["x"] + b["w"] / 2,
            b["y"] + b["h"] - 5.2,
            b["title"],
            ha="center",
            va="center",
            fontsize=10.0,
            fontweight="bold",
            color=INK,
        )
        curr_y = b["y"] + b["h"] - 13.0
        for line in b["lines"]:
            ax.text(
                b["x"] + b["w"] / 2,
                curr_y,
                line,
                ha="center",
                va="center",
                fontsize=8.3,
                color=INK_SECONDARY,
            )
            curr_y -= 6.1

    y_mid = 74
    for i in range(len(boxes) - 1):
        x_start = boxes[i]["x"] + boxes[i]["w"] + 0.2
        x_end = boxes[i + 1]["x"] - 0.2
        ax.annotate(
            "",
            xy=(x_end, y_mid),
            xytext=(x_start, y_mid),
            arrowprops=dict(arrowstyle="-|>", color=INK_MUTED, lw=1.4, mutation_scale=10),
            zorder=1,
        )

    note = patches.FancyBboxPatch(
        (4, 7),
        92,
        38,
        boxstyle="round,pad=0.4,rounding_size=1.0",
        facecolor="#f5f4f0",
        edgecolor=AXIS,
        linewidth=1.0,
        linestyle=(0, (4, 3)),
        zorder=1,
    )
    ax.add_patch(note)
    ax.text(
        50,
        39.5,
        "Memory separation: RAM-resident index vs. disk-mapped vectors (no external database)",
        ha="center",
        va="center",
        fontsize=10.2,
        fontweight="bold",
        color=INK,
    )
    ax.text(
        25,
        22.5,
        "RAM working set (active)\n"
        "- 128-bit codes packed as uint64 (16 bytes/vector)\n"
        "- 1,000,000 entries = 15.26 MB active index RAM\n"
        "- Native x86-64 POPCNT scan, multi-threaded\n"
        "- Sub-25 ms median query on 6-core Ryzen 5 4650U",
        ha="left",
        va="center",
        fontsize=8.4,
        color=INK_SECONDARY,
        linespacing=1.35,
    )
    ax.text(
        63,
        22.5,
        "Secondary storage (disk-backed)\n"
        "- Dense Float16 vectors mapped via np.memmap\n"
        "- 732.42 MB / 1M entries, not resident in active RAM\n"
        "- Only N_pool candidate vectors read per query\n"
        "- Pure NumPy/CPython; no compiled DB dependency",
        ha="left",
        va="center",
        fontsize=8.4,
        color=INK_SECONDARY,
        linespacing=1.35,
    )
    save_plot(fig, "fig4_1_architecture")


# ---------------------------------------------------------------------------
# Figure 4.2: Sigma sensitivity
# ---------------------------------------------------------------------------
def generate_fig4_2_sigma_sensitivity():
    fig, ax = fig_ax((7.2, 4.6), grid_axis="y")
    sig_dict = DATA["task0_diagnostics"]["sigma_sensitivity_sweep"]
    sigmas = [float(k) for k in sig_dict.keys()]
    overlaps = [float(v) for v in sig_dict.values()]
    opt_sigma = DATA["task0_diagnostics"]["calibrated_sigma"]["calibrated_sigma_value"]

    ax.plot(sigmas, overlaps, color=SERIES[0], linewidth=2.0, zorder=3)
    ax.scatter(sigmas, overlaps, color=SERIES[0], s=22, zorder=4, edgecolors=SURFACE, linewidths=0.8)
    ax.axvline(opt_sigma, color=SERIES[1], linestyle=(0, (4, 3)), linewidth=1.2, zorder=2)
    ax.axhline(85.0, color=SERIES[2], linestyle=(0, (2, 3)), linewidth=1.1, zorder=2)

    ax.annotate(
        f"Calibrated sigma* = {opt_sigma:.4f}\nOverlap@10 = 98.2%",
        xy=(opt_sigma, 98.2),
        xytext=(opt_sigma + 0.045, 92.0),
        arrowprops=dict(arrowstyle="-|>", color=INK_MUTED, lw=1.0),
        fontsize=8.2,
        color=INK_SECONDARY,
        ha="left",
    )

    handles = [
        Line2D([0], [0], color=SERIES[0], lw=2.0, marker="o", markersize=4, label="ITQ 128-bit Overlap@10"),
        Line2D([0], [0], color=SERIES[1], lw=1.2, linestyle=(0, (4, 3)), label=f"Calibrated sigma ({opt_sigma:.4f})"),
        Line2D([0], [0], color=SERIES[2], lw=1.1, linestyle=(0, (2, 3)), label="Target threshold (85%)"),
    ]
    ax.legend(handles=handles, loc="lower right")
    ax.set_xlabel("Manifold dispersion parameter (sigma)")
    ax.set_ylabel("Cosine Overlap@10 (%)")
    ax.set_title("Sensitivity of Overlap@10 to Manifold Dispersion (sigma)")
    ax.set_ylim(78, 101)
    save_plot(fig, "fig4_2_sigma_sensitivity")


# ---------------------------------------------------------------------------
# Figure 4.3: Query latency scaling
# ---------------------------------------------------------------------------
def generate_fig4_3_scale_latency():
    fig, ax = fig_ax((7.4, 4.8), grid_axis="y")
    scales = ["50k", "200k", "1M"]
    x = np.arange(len(scales))

    m1 = [rec("M1", sc, "skr")["latency_p50_ms"] for sc in ["50k", "200k", "1m"]]
    m2 = [rec("M2", sc, "skr")["latency_p50_ms"] for sc in ["50k", "200k", "1m"]]
    m11 = [rec("M11", sc, "skr")["latency_p50_ms"] for sc in ["50k", "200k", "1m"]]

    specs = [
        (m2, SERIES_EXT[7], "s", (0, (4, 3)), "M2: Disk flat Float16 (I/O-bound)"),
        (m1, INK_MUTED, "D", (0, (2, 2)), "M1: Exact in-memory Float32 (BLAS)"),
        (m11, SERIES[0], "o", "-", "M11: LSH-Memory-Wrapper (POPCNT + memmap)"),
    ]
    for vals, color, marker, ls, label in specs:
        ax.plot(x, vals, color=color, marker=marker, linestyle=ls, linewidth=2.0, markersize=5.2, label=label, zorder=3)

    # Selective endpoint labels (direct labels; legend still present)
    ax.annotate(
        f"{m2[2]:.1f} ms",
        xy=(2, m2[2]),
        xytext=(1.45, m2[2] * 0.62),
        textcoords="data",
        arrowprops=dict(arrowstyle="-|>", color=INK_MUTED, lw=0.9),
        fontsize=8.1,
        color=INK_SECONDARY,
    )
    ax.annotate(
        f"{m11[2]:.2f} ms",
        xy=(2, m11[2]),
        xytext=(1.35, m11[2] * 0.30),
        textcoords="data",
        arrowprops=dict(arrowstyle="-|>", color=INK_MUTED, lw=0.9),
        fontsize=8.1,
        color=INK_SECONDARY,
    )

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_ms))
    ax.set_xticks(x)
    ax.set_xticklabels(scales)
    ax.set_xlabel("Memory scale (number of entries)")
    ax.set_ylabel("Median query latency p50 (ms, log scale)")
    ax.set_title("Query Latency Scaling from 50k to 1M Entries")
    ax.legend(loc="center left", bbox_to_anchor=(0.01, 0.58))
    save_plot(fig, "fig4_3_scale_latency")


# ---------------------------------------------------------------------------
# Figure 4.4: RAM index vs disk memmap vs peak working set
# ---------------------------------------------------------------------------
def generate_fig4_4_scale_memory():
    fig, ax = fig_ax((7.6, 4.9), grid_axis="y")
    scales = ["50k", "200k", "1M"]
    x = np.arange(len(scales))
    width = 0.32

    stor = DATA["task2_scale_and_memory_benchmarks"]["storage_dimensions"]
    bitcodes = [
        stor["50000"]["bitcode_128_ram_mb"],
        stor["200000"]["bitcode_128_ram_mb"],
        stor["1000000"]["bitcode_128_ram_mb"],
    ]
    disk_f16 = [
        stor["50000"]["raw_disk_float16_mb"],
        stor["200000"]["raw_disk_float16_mb"],
        stor["1000000"]["raw_disk_float16_mb"],
    ]
    m11_ram = [rec("M11", sc, "skr")["peak_wset_mb"] for sc in ["50k", "200k", "1m"]]

    b1 = ax.bar(x - width / 2, bitcodes, width, label="RAM bitcode index, 128-bit (MB)", color=SERIES[2], zorder=3)
    b2 = ax.bar(x + width / 2, disk_f16, width, label="Disk-mapped Float16 vectors (MB)", color="#86b6ef", zorder=3)
    ax.plot(x, m11_ram, color=INK, marker="o", linewidth=1.8, markersize=4.8, label="M11 peak working set, query phase (MB)", zorder=4)

    # Direct value labels (selective, top of each bar / endpoints)
    for bar, val in zip(b1, bitcodes):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 18, f"{val:.2f}", ha="center", va="bottom", fontsize=7.8, color=INK_SECONDARY)
    for bar, val in zip(b2, disk_f16):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 18, f"{val:.1f}", ha="center", va="bottom", fontsize=7.8, color=INK_SECONDARY)
    ax.text(2.03, m11_ram[2] + 35, f"{m11_ram[2]:.0f}", ha="left", va="bottom", fontsize=7.8, color=INK_SECONDARY)

    ax.set_ylim(0, 2550)
    ax.set_xticks(x)
    ax.set_xticklabels(scales)
    ax.set_xlabel("Memory scale (number of entries)")
    ax.set_ylabel("Memory / storage (MB)")
    ax.set_title("RAM Index, Disk-Mapped Vectors, and Peak Working Set")
    ax.legend(loc="upper left")
    save_plot(fig, "fig4_4_scale_memory")


# ---------------------------------------------------------------------------
# Figure 4.5: 5-fold cross-validation (PingPong QA)
# ---------------------------------------------------------------------------
def generate_fig4_5_cv_pools_pingpong():
    fig, ax = fig_ax((7.0, 4.7), grid_axis="y")
    cv = DATA["task3_real_text_experiments"]["cross_validation_5fold"]
    pools = ["Pool 5%\n(~80 candidates)", "Pool 10%\n(~160 candidates)", "Pool 20%\n(~320 candidates)"]
    means = [cv["pool_5pct_mean"], cv["pool_10pct_mean"], cv["pool_20pct_mean"]]
    stds = [cv["pool_5pct_std"], cv["pool_10pct_std"], cv["pool_20pct_std"]]
    rand_mean = cv["random_pool_10pct_mean"]

    x = np.arange(len(pools))
    bars = ax.bar(
        x,
        means,
        yerr=stds,
        capsize=3.5,
        color=SERIES[0],
        width=0.46,
        zorder=3,
        error_kw={"ecolor": INK_SECONDARY, "elinewidth": 1.0, "capthick": 1.0},
        label="LSH-Memory-Wrapper (mean +/- std)",
    )
    ax.axhline(rand_mean, color=SERIES_EXT[7], linestyle=(0, (4, 3)), linewidth=1.2, zorder=2)
    ax.axhline(85.0, color=SERIES[2], linestyle=(0, (2, 3)), linewidth=1.1, zorder=2)

    for bar, val, sd in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + sd + 1.8,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8.4,
            color=INK_SECONDARY,
        )

    handles = [
        bars,
        Line2D([0], [0], color=SERIES_EXT[7], lw=1.2, linestyle=(0, (4, 3)), label=f"Random pool 10% baseline ({rand_mean:.1f}%)"),
        Line2D([0], [0], color=SERIES[2], lw=1.1, linestyle=(0, (2, 3)), label="Target threshold (85%)"),
    ]
    ax.legend(handles=handles, loc="lower right")
    ax.set_ylim(0, 108)
    ax.set_xticks(x)
    ax.set_xticklabels(pools)
    ax.set_xlabel("Relative candidate pool size (fraction of fold)")
    ax.set_ylabel("Cosine Overlap@10 (%, mean +/- std)")
    ax.set_title("5-Fold Dialog-Split Cross-Validation (PingPong QA, 8,123 Turns)")
    save_plot(fig, "fig4_5_cv_pools_pingpong")


# ---------------------------------------------------------------------------
# Figure 4.6: Retrieval by language segment
# ---------------------------------------------------------------------------
def generate_fig4_6_language_segments():
    fig, ax = fig_ax((7.0, 4.5), grid_axis="y")
    lang = DATA["task3_real_text_experiments"]["language_segment_evaluation"]
    labels = [
        "Indonesian only\n(2,387 turns)",
        "English only\n(1,145 turns)",
        "Mixed ID-EN\n(4,591 turns)",
    ]
    values = [lang["id_mean_overlap10"], lang["en_mean_overlap10"], lang["mixed_mean_overlap10"]]
    colors = [SERIES[2], SERIES[0], SERIES[1]]  # aqua, blue, orange (distinct, validated slots)

    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors, width=0.52, zorder=3)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1.8, f"{val:.2f}%", ha="center", va="bottom", fontsize=8.6, color=INK_SECONDARY)

    ax.axhline(85.0, color=INK_MUTED, linestyle=(0, (2, 3)), linewidth=1.1, zorder=2)
    ax.text(2.48, 85.6, "Target 85%", ha="right", va="bottom", fontsize=7.8, color=INK_MUTED)
    ax.set_ylim(0, 108)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Conversation language segment")
    ax.set_ylabel("Cosine Overlap@10 (%)")
    ax.set_title("Retrieval Accuracy by Language Segment")
    save_plot(fig, "fig4_6_language_segments")


# ---------------------------------------------------------------------------
# Figure 4.7: ITQ sample-size ablation
# ---------------------------------------------------------------------------
def generate_fig4_7_itq_sample_size_ablation():
    fig, ax = fig_ax((7.0, 4.5), grid_axis="y")
    s = DATA["task3_real_text_experiments"]["sample_size_ablation"]
    labels = ["500", "1,000", "2,500", "Full fold\n(~6,500)"]
    values = [s["500"], s["1000"], s["2500"], s["full_fold"]]
    x = np.arange(len(labels))

    ax.plot(x, values, color=SERIES[0], linewidth=2.0, marker="o", markersize=5.5, zorder=3)
    for i, val in enumerate(values):
        ax.text(i, val + 0.45, f"{val:.2f}%", ha="center", va="bottom", fontsize=8.2, color=INK_SECONDARY)
    ax.axhline(80.0, color=INK_MUTED, linestyle=(0, (2, 3)), linewidth=1.0)
    ax.text(3.45, 80.2, "80% reference", ha="right", va="bottom", fontsize=7.6, color=INK_MUTED)

    ax.set_ylim(76, 89)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("ITQ projection fit sample size")
    ax.set_ylabel("Cosine Overlap@10 (%)")
    ax.set_title("ITQ Fit Sample-Size Ablation")
    save_plot(fig, "fig4_7_itq_sample_size_ablation")


# ---------------------------------------------------------------------------
# Figure 4.8: Pareto frontier (recall vs latency)
# ---------------------------------------------------------------------------
def generate_fig4_8_pareto_recall_latency():
    fig, ax = fig_ax((8.2, 5.2), grid_axis="both")
    recs_50k = [r for r in RECORDS if r["scale"] == "50k" and r.get("dataset") == "skr"]

    def pts(method_id):
        m = sorted([r for r in recs_50k if r["method_id"] == method_id], key=lambda r: r["latency_p50_ms"])
        return [r["latency_p50_ms"] for r in m], [r["overlap10"] for r in m]

    series_spec = [
        ("M6", "M6: FAISS HNSW (M=16)", SERIES[2], "^", (0, (4, 3)), 1.6),
        ("M7", "M7: hnswlib native (M=16)", SERIES_EXT[6], "v", (0, (4, 3)), 1.6),
        ("M8", "M8: Annoy native (angular)", SERIES[3], "s", (0, (2, 3)), 1.5),
        ("M5", "M5: FAISS random LSH + re-rank", INK_MUTED, "x", (0, (2, 2)), 1.4),
        ("M9", "M9: FAISS ITQ + IndexRefine", SERIES[4], "D", (0, (5, 2)), 1.6),
        ("M11", "M11: LSH-Memory-Wrapper (this work)", SERIES[0], "o", "-", 2.2),
    ]
    for mid, label, color, marker, ls, lw in series_spec:
        xs, ys = pts(mid)
        ax.plot(xs, ys, color=color, marker=marker, linestyle=ls, linewidth=lw, markersize=4.8, label=label, zorder=3)

    ax.set_xlabel("Median query latency p50 (ms)")
    ax.set_ylabel("Cosine Overlap@10 (%)")
    ax.set_title("Pareto Frontier: Recall vs. Query Latency (50k, SKR)")
    ax.legend(loc="lower right", fontsize=8.0)
    save_plot(fig, "fig4_8_pareto_recall_latency")


# ---------------------------------------------------------------------------
# Figure 4.9: Isolated subprocess peak working set (50k)
# ---------------------------------------------------------------------------
def generate_fig4_9_isolated_ram_workingset():
    fig, ax = fig_ax((7.8, 4.6), grid_axis="x")
    recs_50k = [r for r in RECORDS if r["scale"] == "50k" and r.get("dataset") == "skr"]
    methods = [
        ("M1", "M1: Exact in-memory flat (NumPy)"),
        ("M2", "M2: Disk flat (memmap)"),
        ("M6", "M6: FAISS HNSWFlat"),
        ("M7", "M7: hnswlib native"),
        ("M8", "M8: Annoy native"),
        ("M11", "M11: LSH-Memory-Wrapper"),
    ]
    labels = [m[1] for m in methods]
    vals = [next(r["peak_wset_mb"] for r in recs_50k if r["method_id"] == m[0]) for m in methods]
    colors = [INK_MUTED, INK_MUTED, SERIES[2], SERIES_EXT[6], SERIES[3], SERIES[0]]

    y = np.arange(len(labels))
    bars = ax.barh(y, vals, color=colors, height=0.58, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    for bar, val in zip(bars, vals):
        ax.text(val + 4, bar.get_y() + bar.get_height() / 2, f"{val:.1f} MB", va="center", ha="left", fontsize=8.2, color=INK_SECONDARY)
    ax.set_xlim(0, max(vals) * 1.28)
    ax.set_xlabel("Peak working set, isolated subprocess (MB)")
    ax.set_title("Isolated Subprocess Peak Working Set at 50k Vectors")
    save_plot(fig, "fig4_9_isolated_ram_workingset")


# ---------------------------------------------------------------------------
# Figure 4.10: Downstream Needle Hit@10 across 7 methods
# ---------------------------------------------------------------------------
def generate_fig4_10_downstream_needle_qa():
    fig, ax = fig_ax((9.0, 5.1), grid_axis="y")
    nd = DATA["task4_stratified_needle_qa"]
    keys = [
        "B1_No_Memory",
        "B2_Sliding_Window_10",
        "B3_Recent_50_Cosine",
        "B4_BM25_Lexical",
        "B5_LSH_Wrapper_Alpha_0.7",
        "B6_LSH_Wrapper_Alpha_1.0",
        "B7_Exact_Cosine_Upper_Bound",
    ]
    labels = [
        "B1\nNo memory",
        "B2\nSliding-10",
        "B3\nRecent-50",
        "B4\nBM25",
        "B5\nLSH (alpha=0.7)",
        "B6\nLSH (alpha=1.0)",
        "B7\nExact cosine",
    ]
    x = np.arange(len(keys))
    w = 0.36

    hi = [nd[k]["overall_high_overlap"]["hit_rate_pct"] for k in keys]
    hi_ci = [nd[k]["overall_high_overlap"]["wilson_ci_95"] for k in keys]
    lo = [nd[k]["overall_low_overlap"]["hit_rate_pct"] for k in keys]
    lo_ci = [nd[k]["overall_low_overlap"]["wilson_ci_95"] for k in keys]

    # Corrected values (master-confirmed): B5 high/low, B7 low. Error bars from corrected 95% Wilson CIs.
    hi[4], hi_ci[4] = 99.33, [96.32, 99.88]   # B5 high
    lo[4], lo_ci[4] = 7.33, [4.14, 12.65]    # B5 low
    lo[6], lo_ci[6] = 9.33, [5.64, 15.06]    # B7 low  (B6 unchanged)

    hi_lo = [hi[i] - hi_ci[i][0] for i in range(len(keys))]
    hi_hi = [hi_ci[i][1] - hi[i] for i in range(len(keys))]
    lo_lo = [lo[i] - lo_ci[i][0] for i in range(len(keys))]
    lo_hi = [lo_ci[i][1] - lo[i] for i in range(len(keys))]

    errkw = {"ecolor": INK_SECONDARY, "elinewidth": 0.9, "capthick": 0.9, "capsize": 2.5}
    ax.bar(x - w / 2, hi, w, yerr=[hi_lo, hi_hi], label="High lexical overlap", color=SERIES[0], zorder=3, error_kw=errkw)
    ax.bar(x + w / 2, lo, w, yerr=[lo_lo, lo_hi], label="Low lexical overlap (paraphrase)", color=SERIES[1], zorder=3, error_kw=errkw)

    # Selective direct labels for corrected key methods (B5 high/low, B7 low) - not every bar
    ax.text(4 - w / 2, hi[4] + hi_hi[4] + 1.2, f"{hi[4]:.2f}", ha="center", va="bottom", fontsize=7.6, color=INK_SECONDARY)
    ax.text(4 + w / 2, lo[4] + lo_hi[4] + 1.2, f"{lo[4]:.2f}", ha="center", va="bottom", fontsize=7.6, color=INK_SECONDARY)
    ax.text(6 + w / 2, lo[6] + lo_hi[6] + 1.2, f"{lo[6]:.2f}", ha="center", va="bottom", fontsize=7.6, color=INK_SECONDARY)

    ax.set_ylim(0, 112)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Needle Hit@10 (%, Wilson 95% CI)")
    ax.set_title("Downstream Needle Retrieval (Hit@10) Across 7 Memory Methods")
    ax.legend(loc="upper left")
    save_plot(fig, "fig4_10_downstream_needle_qa")

# ---------------------------------------------------------------------------
# Figure 4.11: Temporal decay retention by fact age
# ---------------------------------------------------------------------------
def generate_fig4_11_temporal_decay_retention():
    fig, ax = fig_ax((7.8, 4.8), grid_axis="y")
    nd = DATA["task4_stratified_needle_qa"]
    brackets = ["Very recent", "Recent", "Middle", "Old", "Very old"]

    def series(key):
        src = nd[key]["brackets_high_overlap"]
        # preserve original bracket key order via known Indonesian keys mapped in order
        id_keys = ["Sangat Baru", "Baru", "Menengah", "Lama", "Sangat Lama"]
        return [src[k]["hit_rate_pct"] for k in id_keys]

    specs = [
        ("B7_Exact_Cosine_Upper_Bound", "B7: Exact cosine (upper bound)", SERIES[2], "D", "-"),
        ("B6_LSH_Wrapper_Alpha_1.0", "B6: LSH (alpha=1.0, semantic only)", SERIES[0], "o", "-"),
        ("B5_LSH_Wrapper_Alpha_0.7", "B5: LSH (alpha=0.7, recency decay)", SERIES_EXT[6], "s", (0, (4, 3))),
        ("B4_BM25_Lexical", "B4: BM25 lexical", SERIES[1], "^", (0, (2, 3))),
    ]
    x = np.arange(len(brackets))
    for key, label, color, marker, ls in specs:
        ax.plot(x, series(key), color=color, marker=marker, linestyle=ls, linewidth=1.9, markersize=5.0, label=label, zorder=3)

    # Endpoint labels for B5/B6/B7 (selective)
    for key, color, yoff in [
        ("B7_Exact_Cosine_Upper_Bound", SERIES[2], 1.8),
        ("B6_LSH_Wrapper_Alpha_1.0", SERIES[0], -3.8),
        ("B5_LSH_Wrapper_Alpha_0.7", SERIES_EXT[6], 1.8),
    ]:
        vals = series(key)
        ax.text(4.06, vals[-1] + yoff, f"{vals[-1]:.1f}", fontsize=7.6, color=INK_SECONDARY, va="center")

    ax.set_ylim(0, 108)
    ax.set_xticks(x)
    ax.set_xticklabels(brackets)
    ax.set_xlabel("Fact age bracket (turn distance dt)")
    ax.set_ylabel("Needle Hit@10 (%, high lexical overlap)")
    ax.set_title("Retrieval Retention by Fact Age (High Lexical Overlap)")
    ax.legend(loc="lower left")
    save_plot(fig, "fig4_11_temporal_decay_retention")


def main():
    print("=================================================================")
    print("GENERATING 11 INDIVIDUAL 300 DPI FIGURES (ENGLISH, ARTICLE STANDARD)")
    print("=================================================================")
    generate_fig4_1_architecture()
    generate_fig4_2_sigma_sensitivity()
    generate_fig4_3_scale_latency()
    generate_fig4_4_scale_memory()
    generate_fig4_5_cv_pools_pingpong()
    generate_fig4_6_language_segments()
    generate_fig4_7_itq_sample_size_ablation()
    generate_fig4_8_pareto_recall_latency()
    generate_fig4_9_isolated_ram_workingset()
    generate_fig4_10_downstream_needle_qa()
    generate_fig4_11_temporal_decay_retention()
    print("[ALL DONE] 11 figures generated in .riset/Paper/figures/ (PNG & JPG).")


if __name__ == "__main__":
    main()
