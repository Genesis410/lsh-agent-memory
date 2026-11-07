"""Generate combined 300 DPI publication figures (Chapter 4), English, dataviz-aligned.

Outputs 5 combined figures (PNG + JPG) in .riset/Paper/figures/:
  fig4_1_architecture_combined
  fig4_2_scale_latency_ram_combined
  fig4_3_sigma_and_cv_combined
  fig4_4_pareto_and_ram_combined
  fig4_5_downstream_retention_combined
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

SCRIPT_DIR = Path(__file__).resolve().parent
JSON_PATH = SCRIPT_DIR / "benchmarks" / "experiment_results_bab4.json"
FIG_DIR = SCRIPT_DIR / "Paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

with open(JSON_PATH, "r", encoding="utf-8") as f:
    DATA = json.load(f)

RECORDS = DATA["task2_scale_and_memory_benchmarks"]["baseline_records_isolated"]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
SERIES_EXT = SERIES + ["#6250d6", "#e34948"]

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "Liberation Sans", "sans-serif"],
        "font.size": 9.0,
        "axes.titlesize": 10.0,
        "axes.labelsize": 9.2,
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.facecolor": SURFACE,
        "axes.labelcolor": INK,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "legend.frameon": False,
        "legend.fontsize": 8.0,
        "text.color": INK,
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
    }
)


def save_plot(fig, name):
    plt.tight_layout()
    fig.savefig(FIG_DIR / f"{name}.png", dpi=300, facecolor=SURFACE, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.jpg", dpi=300, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {name}.png/.jpg (300 DPI)")


def style_ax(ax, grid_axis=None):
    if grid_axis:
        ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


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
    return next(r for r in RECORDS if r["method_id"] == method_id and r["scale"] == scale and r.get("dataset") == dataset)


def generate_fig4_1_architecture():
    fig, ax = plt.subplots(figsize=(14.2, 6.4), dpi=300)
    ax.set_facecolor(SURFACE)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    boxes = [
        {"x": 2, "y": 55, "w": 16, "h": 38, "title": "1. Ingest & Encode", "lines": ["Bilingual dialogue (ID-EN)", "Natural turn format", "MiniLM 384-D embedding", "L2 normalization"], "fill": "#eef4fc", "edge": SERIES[0]},
        {"x": 22, "y": 55, "w": 17, "h": 38, "title": "2. ITQ Projection", "lines": ["Mean centering (mu)", "PCA spectral projection", "Procrustes SVD rotation", "W matrix (128/256-bit)"], "fill": "#f7f1e8", "edge": SERIES[1]},
        {"x": 43, "y": 55, "w": 17, "h": 38, "title": "3. Dual Storage", "lines": ["RAM: packed uint64 codes", "(15.26 MB / 1M vectors)", "Disk: np.memmap store", "Float16 dense vectors"], "fill": "#eef8f3", "edge": SERIES[2]},
        {"x": 64, "y": 55, "w": 16, "h": 38, "title": "4. POPCNT Scan", "lines": ["Column-wise XOR", "np.bitwise_count (SIMD)", "Linear argpartition", "Candidate pool N_pool"], "fill": "#fdf6e8", "edge": SERIES[3]},
        {"x": 84, "y": 55, "w": 14, "h": 38, "title": "5. Episodic Score", "lines": ["Exact cosine re-rank", "Half-life decay 2^(-dt/h)", "S = alpha*cos + (1-alpha)*t", "Context prompt injection"], "fill": "#f6f0fb", "edge": SERIES_EXT[6]},
    ]
    for b in boxes:
        ax.add_patch(patches.FancyBboxPatch((b["x"], b["y"]), b["w"], b["h"], boxstyle="round,pad=0.5,rounding_size=1.2", facecolor=b["fill"], edgecolor=b["edge"], linewidth=1.4))
        ax.text(b["x"] + b["w"] / 2, b["y"] + b["h"] - 5.2, b["title"], ha="center", va="center", fontsize=10.0, fontweight="bold", color=INK)
        cy = b["y"] + b["h"] - 13.0
        for line in b["lines"]:
            ax.text(b["x"] + b["w"] / 2, cy, line, ha="center", va="center", fontsize=8.3, color=INK_SECONDARY)
            cy -= 6.1
    for i in range(len(boxes) - 1):
        ax.annotate("", xy=(boxes[i + 1]["x"] - 0.2, 74), xytext=(boxes[i]["x"] + boxes[i]["w"] + 0.2, 74), arrowprops=dict(arrowstyle="-|>", color=INK_MUTED, lw=1.4, mutation_scale=10))
    ax.add_patch(patches.FancyBboxPatch((4, 7), 92, 38, boxstyle="round,pad=0.4,rounding_size=1.0", facecolor="#f5f4f0", edgecolor=AXIS, linewidth=1.0, linestyle=(0, (4, 3))))
    ax.text(50, 39.5, "Memory separation: RAM-resident index vs. disk-mapped vectors (no external database)", ha="center", va="center", fontsize=10.2, fontweight="bold", color=INK)
    ax.text(25, 22.5, "RAM working set (active)\n- 128-bit codes packed as uint64 (16 bytes/vector)\n- 1M entries = 15.26 MB active index RAM\n- Native POPCNT scan, multi-threaded\n- Sub-25 ms median on 6-core Ryzen 5 4650U", ha="left", va="center", fontsize=8.4, color=INK_SECONDARY, linespacing=1.35)
    ax.text(63, 22.5, "Secondary storage (disk-backed)\n- Dense Float16 vectors via np.memmap\n- 732.42 MB / 1M entries, not active RAM\n- Only N_pool candidates read per query\n- Pure NumPy/CPython; no DB dependency", ha="left", va="center", fontsize=8.4, color=INK_SECONDARY, linespacing=1.35)
    save_plot(fig, "fig4_1_architecture_combined")


def generate_fig4_2_scale_latency_ram():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.2), dpi=300)
    style_ax(ax1, "y")
    style_ax(ax2, "y")
    scales = ["50k", "200k", "1M"]
    x = np.arange(len(scales))
    width = 0.32
    m1 = [rec("M1", sc, "skr")["latency_p50_ms"] for sc in ["50k", "200k", "1m"]]
    m2 = [rec("M2", sc, "skr")["latency_p50_ms"] for sc in ["50k", "200k", "1m"]]
    m11 = [rec("M11", sc, "skr")["latency_p50_ms"] for sc in ["50k", "200k", "1m"]]
    ax1.plot(x, m2, marker="s", lw=2.0, ls=(0, (4, 3)), color=SERIES_EXT[7], label="M2: Disk flat Float16")
    ax1.plot(x, m1, marker="D", lw=1.8, ls=(0, (2, 2)), color=INK_MUTED, label="M1: Exact in-memory Float32")
    ax1.plot(x, m11, marker="o", lw=2.2, color=SERIES[0], label="M11: LSH-Memory-Wrapper")
    ax1.set_yscale("log")
    ax1.yaxis.set_major_formatter(FuncFormatter(fmt_ms))
    ax1.set_xticks(x)
    ax1.set_xticklabels(scales)
    ax1.set_xlabel("Memory scale (entries)")
    ax1.set_ylabel("Median query latency p50 (ms, log)")
    ax1.set_title("(a) Query latency scaling")
    ax1.legend(loc="center left", bbox_to_anchor=(0.0, 0.58), fontsize=7.6)

    stor = DATA["task2_scale_and_memory_benchmarks"]["storage_dimensions"]
    bitcodes = [stor["50000"]["bitcode_128_ram_mb"], stor["200000"]["bitcode_128_ram_mb"], stor["1000000"]["bitcode_128_ram_mb"]]
    disk_f16 = [stor["50000"]["raw_disk_float16_mb"], stor["200000"]["raw_disk_float16_mb"], stor["1000000"]["raw_disk_float16_mb"]]
    m11_ram = [rec("M11", sc, "skr")["peak_wset_mb"] for sc in ["50k", "200k", "1m"]]
    ax2.bar(x - width / 2, bitcodes, width, label="RAM bitcode index 128-bit (MB)", color=SERIES[2], zorder=3)
    ax2.bar(x + width / 2, disk_f16, width, label="Disk-mapped Float16 (MB)", color="#86b6ef", zorder=3)
    ax2.plot(x, m11_ram, marker="o", color=INK, lw=1.8, label="M11 peak working set (MB)", zorder=4)
    ax2.set_ylim(0, 2550)
    ax2.set_xticks(x)
    ax2.set_xticklabels(scales)
    ax2.set_xlabel("Memory scale (entries)")
    ax2.set_ylabel("Memory / storage (MB)")
    ax2.set_title("(b) RAM index, disk vectors, peak working set")
    ax2.legend(loc="upper left", fontsize=7.6)
    save_plot(fig, "fig4_2_scale_latency_ram_combined")


def generate_fig4_3_sigma_and_cv():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 5.1), dpi=300)
    style_ax(ax1, "y")
    style_ax(ax2, "y")
    sig = DATA["task0_diagnostics"]["sigma_sensitivity_sweep"]
    xs = [float(k) for k in sig.keys()]
    ys = [float(v) for v in sig.values()]
    opt = DATA["task0_diagnostics"]["calibrated_sigma"]["calibrated_sigma_value"]
    ax1.plot(xs, ys, color=SERIES[0], marker="o", ms=4, lw=1.9)
    ax1.axvline(opt, color=SERIES[1], ls=(0, (4, 3)), lw=1.1)
    ax1.axhline(85.0, color=SERIES[2], ls=(0, (2, 3)), lw=1.0)
    ax1.set_xlabel("Manifold dispersion (sigma)")
    ax1.set_ylabel("Cosine Overlap@10 (%)")
    ax1.set_title("(a) Sensitivity to manifold dispersion")
    ax1.set_ylim(78, 101)
    ax1.legend(handles=[
        Line2D([0], [0], color=SERIES[0], marker="o", lw=1.9, label="ITQ 128-bit Overlap@10"),
        Line2D([0], [0], color=SERIES[1], ls=(0, (4, 3)), label=f"Calibrated sigma ({opt:.4f})"),
        Line2D([0], [0], color=SERIES[2], ls=(0, (2, 3)), label="Target 85%"),
    ], fontsize=7.6)

    cv = DATA["task3_real_text_experiments"]["cross_validation_5fold"]
    labels = ["Pool 5%\n(~80)", "Pool 10%\n(~160)", "Pool 20%\n(~320)"]
    means = [cv["pool_5pct_mean"], cv["pool_10pct_mean"], cv["pool_20pct_mean"]]
    stds = [cv["pool_5pct_std"], cv["pool_10pct_std"], cv["pool_20pct_std"]]
    xp = np.arange(len(labels))
    bars = ax2.bar(xp, means, yerr=stds, capsize=3, color=SERIES[0], width=0.46, zorder=3, error_kw={"ecolor": INK_SECONDARY, "elinewidth": 0.9})
    ax2.axhline(cv["random_pool_10pct_mean"], color=SERIES_EXT[7], ls=(0, (4, 3)), lw=1.1)
    ax2.axhline(85.0, color=SERIES[2], ls=(0, (2, 3)), lw=1.0)
    for b, v, s in zip(bars, means, stds):
        ax2.text(b.get_x() + b.get_width() / 2, v + s + 1.5, f"{v:.1f}", ha="center", va="bottom", fontsize=7.8, color=INK_SECONDARY)
    ax2.set_xticks(xp)
    ax2.set_xticklabels(labels)
    ax2.set_ylim(0, 108)
    ax2.set_xlabel("Relative candidate pool")
    ax2.set_ylabel("Cosine Overlap@10 (%, mean +/- std)")
    ax2.set_title("(b) 5-fold dialog-split cross-validation")
    save_plot(fig, "fig4_3_sigma_and_cv_combined")


def generate_fig4_4_pareto_and_ram():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.4, 5.3), dpi=300)
    style_ax(ax1, "both")
    style_ax(ax2, "x")
    recs_50k = [r for r in RECORDS if r["scale"] == "50k" and r.get("dataset") == "skr"]

    def pts(mid):
        m = sorted([r for r in recs_50k if r["method_id"] == mid], key=lambda r: r["latency_p50_ms"])
        return [r["latency_p50_ms"] for r in m], [r["overlap10"] for r in m]

    for mid, label, color, marker, ls, lw in [
        ("M6", "M6: FAISS HNSW", SERIES[2], "^", (0, (4, 3)), 1.5),
        ("M7", "M7: hnswlib", SERIES_EXT[6], "v", (0, (4, 3)), 1.5),
        ("M8", "M8: Annoy", SERIES[3], "s", (0, (2, 3)), 1.4),
        ("M5", "M5: FAISS random LSH+rerank", INK_MUTED, "x", (0, (2, 2)), 1.3),
        ("M9", "M9: FAISS ITQ+Refine", SERIES[4], "D", (0, (5, 2)), 1.5),
        ("M11", "M11: LSH-Memory-Wrapper", SERIES[0], "o", "-", 2.1),
    ]:
        xs, ys = pts(mid)
        ax1.plot(xs, ys, color=color, marker=marker, ls=ls, lw=lw, ms=4.2, label=label)
    ax1.set_xlabel("Median query latency p50 (ms)")
    ax1.set_ylabel("Cosine Overlap@10 (%)")
    ax1.set_title("(a) Pareto frontier (50k, SKR)")
    ax1.legend(loc="lower right", fontsize=7.2)

    methods = [("M1", "M1: Exact in-memory"), ("M2", "M2: Disk flat"), ("M6", "M6: FAISS HNSW"), ("M7", "M7: hnswlib"), ("M8", "M8: Annoy"), ("M11", "M11: LSH-Wrapper")]
    vals = [next(r["peak_wset_mb"] for r in recs_50k if r["method_id"] == m[0]) for m in methods]
    colors = [INK_MUTED, INK_MUTED, SERIES[2], SERIES_EXT[6], SERIES[3], SERIES[0]]
    y = np.arange(len(methods))
    bars = ax2.barh(y, vals, color=colors, height=0.58, zorder=3)
    ax2.set_yticks(y)
    ax2.set_yticklabels([m[1] for m in methods], fontsize=8.2)
    ax2.invert_yaxis()
    for b, v in zip(bars, vals):
        ax2.text(v + 3, b.get_y() + b.get_height() / 2, f"{v:.1f}", va="center", fontsize=7.6, color=INK_SECONDARY)
    ax2.set_xlim(0, max(vals) * 1.3)
    ax2.set_xlabel("Peak working set (MB)")
    ax2.set_title("(b) Isolated subprocess peak working set (50k)")
    save_plot(fig, "fig4_4_pareto_and_ram_combined")


def generate_fig4_5_downstream_retention():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.4, 5.2), dpi=300)
    style_ax(ax1, "y")
    style_ax(ax2, "y")
    nd = DATA["task4_stratified_needle_qa"]
    keys = ["B1_No_Memory", "B2_Sliding_Window_10", "B3_Recent_50_Cosine", "B4_BM25_Lexical", "B5_LSH_Wrapper_Alpha_0.7", "B6_LSH_Wrapper_Alpha_1.0", "B7_Exact_Cosine_Upper_Bound"]
    labels = ["B1\nNo mem", "B2\nSlide-10", "B3\nRecent-50", "B4\nBM25", "B5\nLSH a=0.7", "B6\nLSH a=1.0", "B7\nExact"]
    x = np.arange(len(keys))
    w = 0.36
    hi = [nd[k]["overall_high_overlap"]["hit_rate_pct"] for k in keys]
    hi_e = [[hi[i] - nd[k]["overall_high_overlap"]["wilson_ci_95"][0] for i, k in enumerate(keys)], [nd[k]["overall_high_overlap"]["wilson_ci_95"][1] - hi[i] for i, k in enumerate(keys)]]
    lo = [nd[k]["overall_low_overlap"]["hit_rate_pct"] for k in keys]
    lo_e = [[lo[i] - nd[k]["overall_low_overlap"]["wilson_ci_95"][0] for i, k in enumerate(keys)], [nd[k]["overall_low_overlap"]["wilson_ci_95"][1] - lo[i] for i, k in enumerate(keys)]]
    ek = {"ecolor": INK_SECONDARY, "elinewidth": 0.8, "capsize": 2.2}
    ax1.bar(x - w / 2, hi, w, yerr=hi_e, label="High lexical overlap", color=SERIES[0], zorder=3, error_kw=ek)
    ax1.bar(x + w / 2, lo, w, yerr=lo_e, label="Low lexical overlap", color=SERIES[1], zorder=3, error_kw=ek)
    ax1.set_ylim(0, 112)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=7.6)
    ax1.set_ylabel("Needle Hit@10 (%, Wilson 95% CI)")
    ax1.set_title("(a) Downstream needle retrieval")
    ax1.legend(loc="upper left", fontsize=7.6)

    id_keys = ["Sangat Baru", "Baru", "Menengah", "Lama", "Sangat Lama"]
    brackets = ["Very recent", "Recent", "Middle", "Old", "Very old"]
    xb = np.arange(len(brackets))
    def ser(k):
        return [nd[k]["brackets_high_overlap"][b]["hit_rate_pct"] for b in id_keys]
    for k, label, color, marker, ls in [
        ("B7_Exact_Cosine_Upper_Bound", "B7: Exact cosine", SERIES[2], "D", "-"),
        ("B6_LSH_Wrapper_Alpha_1.0", "B6: LSH a=1.0", SERIES[0], "o", "-"),
        ("B5_LSH_Wrapper_Alpha_0.7", "B5: LSH a=0.7", SERIES_EXT[6], "s", (0, (4, 3))),
        ("B4_BM25_Lexical", "B4: BM25", SERIES[1], "^", (0, (2, 3))),
    ]:
        ax2.plot(xb, ser(k), color=color, marker=marker, ls=ls, lw=1.8, ms=4.5, label=label)
    ax2.set_ylim(0, 108)
    ax2.set_xticks(xb)
    ax2.set_xticklabels(brackets)
    ax2.set_xlabel("Fact age bracket (turn distance)")
    ax2.set_ylabel("Needle Hit@10 (%)")
    ax2.set_title("(b) Retention by fact age (high overlap)")
    ax2.legend(loc="lower left", fontsize=7.4)
    save_plot(fig, "fig4_5_downstream_retention_combined")


def main():
    print("=================================================================")
    print("GENERATING 5 COMBINED 300 DPI FIGURES (ENGLISH, ARTICLE STANDARD)")
    print("=================================================================")
    generate_fig4_1_architecture()
    generate_fig4_2_scale_latency_ram()
    generate_fig4_3_sigma_and_cv()
    generate_fig4_4_pareto_and_ram()
    generate_fig4_5_downstream_retention()
    print("[ALL DONE] 5 combined figures generated in .riset/Paper/figures/.")


if __name__ == "__main__":
    main()
