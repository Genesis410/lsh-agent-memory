"""Generate high-resolution (300 DPI) publication-quality figures for Bab 4 based on verified experiment results.

Produces:
- fig4_1_architecture.png / .jpg: Complete computational architecture and dataflow pipeline.
- fig4_2_scale_latency_ram.png / .jpg: Scaling of query latency (p50/p95) and RAM peak_wset across scales.
- fig4_3_sigma_and_memory_unit.png / .jpg: Empirical sigma sensitivity sweep and 5-fold CV relative pools vs random baseline.
- fig4_4_pareto_recall_latency.png / .jpg: Multi-point Pareto Recall vs Latency frontier and isolated RAM footprint.
- fig4_5_downstream_retention.png / .jpg: Stratified Needle Hit@10 across 7 methods with Wilson 95% error bars.
"""

import json
import os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Load master validated JSON data
with open("benchmarks/experiment_results_bab4.json", "r", encoding="utf-8") as f:
    DATA = json.load(f)

fig_dir = Path("paper/figures")
fig_dir.mkdir(parents=True, exist_ok=True)

# Dataviz typography and styles
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8


def save_plot(fig, name: str):
    """Save plot in both PNG and JPG format at 300 DPI."""
    p_png = fig_dir / f"{name}.png"
    p_jpg = fig_dir / f"{name}.jpg"
    plt.tight_layout()
    fig.savefig(p_png, dpi=300, facecolor='white', bbox_inches='tight')
    fig.savefig(p_jpg, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] Generated {p_png} and {p_jpg}")


def generate_fig4_1_architecture():
    """Figure 4.1: Architecture & Dataflow Pipeline Diagram."""
    fig, ax = plt.subplots(figsize=(15, 6.8), dpi=300)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    boxes = [
        {"x": 2, "y": 55, "w": 16, "h": 38, "title": "1. Ingesti & Ekstraksi", "lines": ["Percakapan Dwibahasa", "Format ID-EN Alami", "MiniLM 384-D Vector", "Normalisasi Bola L2"]},
        {"x": 22, "y": 55, "w": 17, "h": 38, "title": "2. ITQ Projection", "lines": ["Mean Centering (μ)", "PCA Spectral Project", "Procrustes SVD Rotasi", "Matriks W (128/256-bit)"]},
        {"x": 43, "y": 55, "w": 17, "h": 38, "title": "3. Penyimpanan Ganda", "lines": ["RAM: Packed uint64", "(15,26 MB / 1M vektor)", "Disk: np.memmap", "Float16 (Presisi 10^-3)"]},
        {"x": 64, "y": 55, "w": 16, "h": 38, "title": "4. Pemindaian POPCNT", "lines": ["XOR Kolom 1D", "np.bitwise_count native", "Linear argpartition", "Pool Kandidat N (10%)"]},
        {"x": 84, "y": 55, "w": 14, "h": 38, "title": "5. Skoring Episodik", "lines": ["Re-ranking Cosine Exact", "Half-Life Decay 2^(-Δt/h)", "Score = α·cos + (1-α)·t", "Injeksi Konteks Prompt"]}
    ]

    for b in boxes:
        rect = patches.FancyBboxPatch((b["x"], b["y"]), b["w"], b["h"], boxstyle="round,pad=0.8,rounding_size=1.5",
                                      facecolor="#F8FAFC", edgecolor="#1E293B", linewidth=1.6)
        ax.add_patch(rect)
        ax.text(b["x"] + b["w"]/2, b["y"] + b["h"] - 5, b["title"], ha='center', va='center',
                fontsize=10.5, fontweight='bold', color='#0F172A')
        curr_y = b["y"] + b["h"] - 13
        for line in b["lines"]:
            ax.text(b["x"] + b["w"]/2, curr_y, line, ha='center', va='center',
                    fontsize=8.5, color='#334155')
            curr_y -= 6.0

    for i in range(len(boxes) - 1):
        x_start = boxes[i]["x"] + boxes[i]["w"]
        x_end = boxes[i+1]["x"]
        y_mid = 74
        ax.annotate('', xy=(x_end, y_mid), xytext=(x_start, y_mid),
                    arrowprops=dict(facecolor='#0F172A', edgecolor='#0F172A', width=2.0, headwidth=7, headlength=7))

    note_box = patches.FancyBboxPatch((4, 7), 92, 38, boxstyle="square,pad=0.5",
                                      facecolor="#F1F5F9", edgecolor="#64748B", linewidth=1.2, linestyle="--")
    ax.add_patch(note_box)
    ax.text(50, 39, "Arsitektur Pemisahan Working Set: RAM Terikat vs Disk Terpetakan (Zero-Dependency)",
            ha='center', va='center', fontsize=10.5, fontweight='bold', color='#0F172A')

    ax.text(25, 23, "Working Set RAM (In-Memory):\n- Bitcode 128-bit dipadatkan ke uint64 (16 byte/vektor)\n- Skala 1.000.000 entri = tepat 15,26 MB RAM aktif\n- Pemindaian POPCNT native x86-64 paralel tanpa GIL\n- Waktu respon sub-25 ms pada CPU 6-core Ryzen 5 4650U",
            ha='left', va='center', fontsize=8.5, color='#1E293B')

    ax.text(63, 23, "Penyimpanan Sekunder (Disk-Backed):\n- Vektor mentah Float16 terpetakan via np.memmap\n- Alokasi 732,42 MB / 1M entri tanpa konsumsi RAM aktif\n- Pembacaan sequential dihindari; hanya memuat N_pool kandidat\n- Zero-dependency: murni NumPy & CPython tanpa mesin database",
            ha='left', va='center', fontsize=8.5, color='#1E293B')

    save_plot(fig, "fig4_1_architecture")


def generate_fig4_2_scale_latency_ram():
    """Figure 4.2: Scalability Latency & RAM Usage (50k to 1M entries)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    fig.patch.set_facecolor('white')

    scales = ['50k', '200k', '1M']
    x = np.arange(len(scales))
    width = 0.35

    # Extract isolated records for M1, M2, M11
    records = DATA["task2_scale_and_memory_benchmarks"]["baseline_records_isolated"]

    m1_lats = [next(r["latency_p50_ms"] for r in records if r["method_id"] == "M1" and r["scale"] == sc and r.get("dataset") == "skr") for sc in ["50k", "200k", "1m"]]
    m2_lats = [next(r["latency_p50_ms"] for r in records if r["method_id"] == "M2" and r["scale"] == sc and r.get("dataset") == "skr") for sc in ["50k", "200k", "1m"]]
    m11_lats = [next(r["latency_p50_ms"] for r in records if r["method_id"] == "M11" and r["scale"] == sc and r.get("dataset") == "skr") for sc in ["50k", "200k", "1m"]]

    # Panel A: Latency comparison
    ax1.plot(x, m2_lats, marker='^', linewidth=2.0, color='#DC2626', linestyle='--', label='M2: Disk Flat Float16 (I/O Bottleneck)')
    ax1.plot(x, m1_lats, marker='s', linewidth=2.0, color='#64748B', linestyle='-.', label='M1: Exact In-Memory Float32 (BLAS)')
    ax1.plot(x, m11_lats, marker='o', linewidth=2.5, color='#2563EB', label='M11: LSH-Wrapper (POPCNT + Memmap)')

    ax1.set_yscale('log')
    ax1.set_xlabel('Skala Jumlah Entri Memori', fontsize=10.5, fontweight='bold', labelpad=8)
    ax1.set_ylabel('Latensi Median p50 (ms, skala log)', fontsize=10.5, fontweight='bold', labelpad=8)
    ax1.set_title('(a) Skalabilitas Latensi Kueri (H2 & H3 Inflection)', fontsize=11, fontweight='bold', pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(scales, fontsize=10)
    ax1.legend(frameon=True, facecolor='white', edgecolor='#E2E8F0', fontsize=8.5)
    ax1.grid(True, linestyle=':', alpha=0.6, color='#CBD5E1')

    # Panel B: Memory Working Set vs Index Size
    stor = DATA["task2_scale_and_memory_benchmarks"]["storage_dimensions"]
    bitcodes = [stor["50000"]["bitcode_128_ram_mb"], stor["200000"]["bitcode_128_ram_mb"], stor["1000000"]["bitcode_128_ram_mb"]]
    disk_f16 = [stor["50000"]["raw_disk_float16_mb"], stor["200000"]["raw_disk_float16_mb"], stor["1000000"]["raw_disk_float16_mb"]]

    # Peak Wset for M11
    m11_ram = [next(r["peak_wset_mb"] for r in records if r["method_id"] == "M11" and r["scale"] == sc and r.get("dataset") == "skr") for sc in ["50k", "200k", "1m"]]

    ax2.bar(x - width/2, bitcodes, width, label='RAM Bitcode 128-bit Murni (MB)', color='#16A34A', edgecolor='#15803D')
    ax2.bar(x + width/2, disk_f16, width, label='Disk Memmap Float16 (MB)', color='#93C5FD', edgecolor='#3B82F6')
    ax2.plot(x, m11_ram, marker='o', color='#0F172A', linewidth=2.0, label='Peak Working Set Total (MB)')

    ax2.axhline(2048.0, color='#DC2626', linestyle='--', linewidth=1.5, label='Batas Budget RAM (< 2.048 MB)')
    ax2.set_xlabel('Skala Jumlah Entri Memori', fontsize=10.5, fontweight='bold', labelpad=8)
    ax2.set_ylabel('Konsumsi Memori & Storage (MB)', fontsize=10.5, fontweight='bold', labelpad=8)
    ax2.set_title('(b) Alokasi Memori Terikat vs Storage Terpetakan', fontsize=11, fontweight='bold', pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(scales, fontsize=10)
    ax2.legend(frameon=True, facecolor='white', edgecolor='#E2E8F0', fontsize=8.5)
    ax2.grid(axis='y', linestyle=':', alpha=0.6, color='#CBD5E1')

    save_plot(fig, "fig4_2_scale_latency_ram")


def generate_fig4_3_sigma_and_memory_unit():
    """Figure 4.3: Empirical Sigma Sensitivity Sweep & 5-Fold CV Pools."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    fig.patch.set_facecolor('white')

    # Panel A: Sigma Sensitivity Sweep
    sig_dict = DATA["task0_diagnostics"]["sigma_sensitivity_sweep"]
    sigmas = [float(k) for k in sig_dict.keys()]
    overlaps = [float(v) for v in sig_dict.values()]
    opt_sigma = DATA["task0_diagnostics"]["calibrated_sigma"]["calibrated_sigma_value"]

    ax1.plot(sigmas, overlaps, marker='o', linewidth=2.2, color='#2563EB', label='ITQ 128-bit Overlap@10 (%)')
    ax1.axvline(opt_sigma, color='#DC2626', linestyle='--', linewidth=1.5, label=f'Optimal σ Terkalibrasi ({opt_sigma:.4f})')
    ax1.axhline(85.0, color='#16A34A', linestyle=':', linewidth=1.5, label='Ambang Target Riset (85%)')

    ax1.set_xlabel('Parameter Dispersi Derau (σ)', fontsize=10.5, fontweight='bold', labelpad=8)
    ax1.set_ylabel('Cosine Overlap@10 (%)', fontsize=10.5, fontweight='bold', labelpad=8)
    ax1.set_title('(a) Uji Sensitivitas Dispersi Manifold (σ)', fontsize=11, fontweight='bold', pad=10)
    ax1.legend(frameon=True, facecolor='white', edgecolor='#E2E8F0', fontsize=8.5)
    ax1.grid(True, linestyle=':', alpha=0.6, color='#CBD5E1')

    # Panel B: 5-Fold Cross Validation Relative Pools vs Random Baseline
    cv = DATA["task3_real_text_experiments"]["cross_validation_5fold"]
    pools = ['Pool 5%\n(~80 cands)', 'Pool 10%\n(~160 cands)', 'Pool 20%\n(~320 cands)']
    pool_means = [cv["pool_5pct_mean"], cv["pool_10pct_mean"], cv["pool_20pct_mean"]]
    pool_stds = [cv["pool_5pct_std"], cv["pool_10pct_std"], cv["pool_20pct_std"]]
    rand_mean = cv["random_pool_10pct_mean"]
    rand_std = cv["random_pool_10pct_std"]

    x_p = np.arange(len(pools))
    bars = ax2.bar(x_p, pool_means, yerr=pool_stds, capsize=5, color='#2563EB', edgecolor='#1D4ED8', width=0.45, label='LSH-Memory-Wrapper')
    ax2.axhline(rand_mean, color='#DC2626', linestyle='--', linewidth=1.5, label=f'Random Pool 10% ({rand_mean:.1f}%)')
    ax2.axhline(85.0, color='#16A34A', linestyle=':', linewidth=1.5, label='Target Riset (85%)')

    for bar, val in zip(bars, pool_means):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 2.5, f"{val:.1f}%", ha='center', va='bottom', fontsize=9.5, fontweight='bold')

    ax2.set_ylim(0, 105)
    ax2.set_xlabel('Ukuran Pool Relatif (Persentase N_fold)', fontsize=10.5, fontweight='bold', labelpad=8)
    ax2.set_ylabel('Cosine Overlap@10 (Mean ± Std, %)', fontsize=10.5, fontweight='bold', labelpad=8)
    ax2.set_title('(b) Kinerja 5-Fold CV Dialog-Split PingPong QA (8.123 Turn)', fontsize=11, fontweight='bold', pad=10)
    ax2.set_xticks(x_p)
    ax2.set_xticklabels(pools, fontsize=9.5)
    ax2.legend(frameon=True, facecolor='white', edgecolor='#E2E8F0', fontsize=8.5)
    ax2.grid(axis='y', linestyle=':', alpha=0.6, color='#CBD5E1')

    save_plot(fig, "fig4_3_sigma_and_memory_unit")


def generate_fig4_4_pareto_recall_latency():
    """Figure 4.4: Multi-point Pareto Recall vs Latency Frontier and Isolated Working Set."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.8), dpi=300)
    fig.patch.set_facecolor('white')

    records = DATA["task2_scale_and_memory_benchmarks"]["baseline_records_isolated"]
    recs_50k = [r for r in records if r["scale"] == "50k" and r.get("dataset") == "skr"]

    # Filter by methods
    def get_points(m_id):
        m_recs = [r for r in recs_50k if r["method_id"] == m_id]
        m_recs.sort(key=lambda r: r["latency_p50_ms"])
        lats = [r["latency_p50_ms"] for r in m_recs]
        ovs = [r["overlap10"] for r in m_recs]
        return lats, ovs

    m5_lat, m5_ov = get_points("M5")
    m6_lat, m6_ov = get_points("M6")
    m7_lat, m7_ov = get_points("M7")
    m8_lat, m8_ov = get_points("M8")
    m9_lat, m9_ov = get_points("M9")
    m11_lat, m11_ov = get_points("M11")

    # Panel A: Pareto Curves
    ax1.plot(m6_lat, m6_ov, marker='^', linestyle='--', color='#0D9488', label='M6: FAISS HNSW (M=16)')
    ax1.plot(m7_lat, m7_ov, marker='v', linestyle='--', color='#059669', label='M7: hnswlib Native (M=16)')
    ax1.plot(m8_lat, m8_ov, marker='s', linestyle=':', color='#D97706', label='M8: Annoy Native (Angular)')
    ax1.plot(m5_lat, m5_ov, marker='x', linestyle=':', color='#94A3B8', label='M5: FAISS Random LSH + Re-rank')
    ax1.plot(m9_lat, m9_ov, marker='d', linestyle='-.', color='#7C3AED', label='M9: FAISS ITQ + IndexRefine')
    ax1.plot(m11_lat, m11_ov, marker='o', linewidth=2.5, color='#2563EB', label='M11: LSH-Wrapper (Riset Ini)')

    ax1.set_xlabel('Latensi Kueri Median p50 (milidetik)', fontsize=10.5, fontweight='bold', labelpad=8)
    ax1.set_ylabel('Cosine Overlap@10 (%)', fontsize=10.5, fontweight='bold', labelpad=8)
    ax1.set_title('(a) Kurva Pareto Recall vs Latensi Kueri (50k SKR Manifold)', fontsize=11, fontweight='bold', pad=10)
    ax1.legend(frameon=True, facecolor='white', edgecolor='#E2E8F0', fontsize=8.5)
    ax1.grid(True, linestyle=':', alpha=0.6, color='#CBD5E1')

    # Panel B: Isolated Process Working Set RAM on 50k
    methods_disp = ['M1: In-Mem Flat', 'M2: Disk Flat', 'M6: FAISS HNSW', 'M7: hnswlib', 'M8: Annoy', 'M11: LSH-Wrapper']
    m_ids = ['M1', 'M2', 'M6', 'M7', 'M8', 'M11']
    wset_vals = [next(r["peak_wset_mb"] for r in recs_50k if r["method_id"] == mid) for mid in m_ids]

    colors_wset = ['#64748B', '#64748B', '#0D9488', '#059669', '#D97706', '#2563EB']
    y_pos = np.arange(len(methods_disp))

    bars_w = ax2.barh(y_pos, wset_vals, color=colors_wset, edgecolor='#0F172A', linewidth=0.8, height=0.55)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(methods_disp, fontsize=9.5)
    ax2.invert_yaxis()

    for bar, val in zip(bars_w, wset_vals):
        ax2.text(val + 5, bar.get_y() + bar.get_height()/2, f"{val:.1f} MB", ha='left', va='center', fontsize=9.0, fontweight='bold')

    ax2.set_xlim(0, max(wset_vals) * 1.25)
    ax2.set_xlabel('Peak Working Set RAM Terisolasi (MB)', fontsize=10.5, fontweight='bold', labelpad=8)
    ax2.set_title('(b) Konsumsi Memori Subproses Mandiri (50k Vektor)', fontsize=11, fontweight='bold', pad=10)
    ax2.grid(axis='x', linestyle=':', alpha=0.6, color='#CBD5E1')

    save_plot(fig, "fig4_4_pareto_recall_latency")


def generate_fig4_5_downstream_retention():
    """Figure 4.5: Downstream Needle Hit@10 across 7 methods with Wilson 95% Error Bars."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.8), dpi=300)
    fig.patch.set_facecolor('white')

    needle_data = DATA["task4_stratified_needle_qa"]
    method_keys = [
        "B1_No_Memory",
        "B2_Sliding_Window_10",
        "B3_Recent_50_Cosine",
        "B4_BM25_Lexical",
        "B5_LSH_Wrapper_Alpha_0.7",
        "B6_LSH_Wrapper_Alpha_1.0",
        "B7_Exact_Cosine_Upper_Bound"
    ]
    method_labels = [
        "B1: Tanpa Memori",
        "B2: Sliding-10",
        "B3: Recent-50",
        "B4: BM25 Leksikal",
        "B5: LSH (α=0,7)",
        "B6: LSH (α=1,0)",
        "B7: Exact Upper Bound"
    ]

    # Panel A: Overall Comparison (High Overlap vs Low Overlap)
    x = np.arange(len(method_keys))
    width = 0.38

    high_rates = [needle_data[m]["overall_high_overlap"]["hit_rate_pct"] for m in method_keys]
    high_err_low = [high_rates[i] - needle_data[m]["overall_high_overlap"]["wilson_ci_95"][0] for i, m in enumerate(method_keys)]
    high_err_high = [needle_data[m]["overall_high_overlap"]["wilson_ci_95"][1] - high_rates[i] for i, m in enumerate(method_keys)]

    low_rates = [needle_data[m]["overall_low_overlap"]["hit_rate_pct"] for m in method_keys]
    low_err_low = [low_rates[i] - needle_data[m]["overall_low_overlap"]["wilson_ci_95"][0] for i, m in enumerate(method_keys)]
    low_err_high = [needle_data[m]["overall_low_overlap"]["wilson_ci_95"][1] - low_rates[i] for i, m in enumerate(method_keys)]

    ax1.bar(x - width/2, high_rates, width, yerr=[high_err_low, high_err_high], capsize=3, label='Tumpang-Tindih Tinggi (High Overlap)', color='#2563EB', edgecolor='#1D4ED8')
    ax1.bar(x + width/2, low_rates, width, yerr=[low_err_low, low_err_high], capsize=3, label='Parafrase Rendah (Low Overlap)', color='#F59E0B', edgecolor='#D97706')

    ax1.set_ylabel('Needle Hit@10 (%, Wilson 95% CI)', fontsize=10.5, fontweight='bold', labelpad=8)
    ax1.set_title('(a) Kinerja Temu Balik Jarum Informasi (Needle QA)', fontsize=11, fontweight='bold', pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(method_labels, rotation=35, ha='right', fontsize=8.5)
    ax1.legend(frameon=True, facecolor='white', edgecolor='#E2E8F0', fontsize=8.5)
    ax1.grid(axis='y', linestyle=':', alpha=0.6, color='#CBD5E1')

    # Panel B: Breakdown by Age Bracket (High Overlap)
    brackets = ["Sangat Baru", "Baru", "Menengah", "Lama", "Sangat Lama"]
    b4_age = [needle_data["B4_BM25_Lexical"]["brackets_high_overlap"][b]["hit_rate_pct"] for b in brackets]
    b5_age = [needle_data["B5_LSH_Wrapper_Alpha_0.7"]["brackets_high_overlap"][b]["hit_rate_pct"] for b in brackets]
    b6_age = [needle_data["B6_LSH_Wrapper_Alpha_1.0"]["brackets_high_overlap"][b]["hit_rate_pct"] for b in brackets]
    b7_age = [needle_data["B7_Exact_Cosine_Upper_Bound"]["brackets_high_overlap"][b]["hit_rate_pct"] for b in brackets]

    x_b = np.arange(len(brackets))
    ax2.plot(x_b, b7_age, marker='d', linestyle='-', linewidth=2.0, color='#10B981', label='B7: Exact Upper Bound')
    ax2.plot(x_b, b6_age, marker='o', linestyle='-', linewidth=2.2, color='#2563EB', label='B6: LSH (α=1,0 Semantik Murni)')
    ax2.plot(x_b, b5_age, marker='s', linestyle='--', linewidth=2.0, color='#6366F1', label='B5: LSH (α=0,7 Recency Decay)')
    ax2.plot(x_b, b4_age, marker='^', linestyle=':', linewidth=1.8, color='#F59E0B', label='B4: BM25 Leksikal')

    ax2.set_xlabel('Kelompok Usia Jarum (Turn Distance Δt)', fontsize=10.5, fontweight='bold', labelpad=8)
    ax2.set_ylabel('Needle Hit@10 (%)', fontsize=10.5, fontweight='bold', labelpad=8)
    ax2.set_title('(b) Retensi Berdasarkan Usia Fakta (Tumpang-Tindih Tinggi)', fontsize=11, fontweight='bold', pad=10)
    ax2.set_xticks(x_b)
    ax2.set_xticklabels(brackets, fontsize=9.5)
    ax2.legend(frameon=True, facecolor='white', edgecolor='#E2E8F0', fontsize=8.5)
    ax2.grid(True, linestyle=':', alpha=0.6, color='#CBD5E1')

    save_plot(fig, "fig4_5_downstream_retention")


def main():
    print("=================================================================")
    print("TASK 6: GENERATING 300 DPI PUBLICATION FIGURES (FIG 4.1 - 4.5)")
    print("=================================================================")
    generate_fig4_1_architecture()
    generate_fig4_2_scale_latency_ram()
    generate_fig4_3_sigma_and_memory_unit()
    generate_fig4_4_pareto_recall_latency()
    generate_fig4_5_downstream_retention()
    print("[ALL DONE] All 5 Bab 4 figures successfully generated in paper/figures/ (PNG & JPG)!")


if __name__ == '__main__':
    main()
