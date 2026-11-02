"""Generate high-resolution monochrome flowchart for Kerangka Pikiran."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_kerangka_pikiran(output_path='paper/kerangka_pikiran.jpg'):
    fig, ax = plt.subplots(figsize=(18, 12), dpi=300)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Styles
    box_edge_thick = 2.0
    box_edge_normal = 1.4
    font_family = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'

    # 1. Main Background Columns (Subgraphs)
    # Col 1: Masalah (X: 3 to 29)
    col1_bg = patches.FancyBboxPatch((3, 4), 26, 92, boxstyle="round,pad=1.0,rounding_size=2.0",
                                    facecolor="#FAFAFA", edgecolor="#555555", linewidth=1.5, linestyle="--")
    ax.add_patch(col1_bg)

    # Col 2: Solusi (X: 37 to 63)
    col2_bg = patches.FancyBboxPatch((37, 4), 26, 92, boxstyle="round,pad=1.0,rounding_size=2.0",
                                    facecolor="#FAFAFA", edgecolor="#222222", linewidth=2.0)
    ax.add_patch(col2_bg)

    # Col 3: Evaluasi (X: 71 to 97)
    col3_bg = patches.FancyBboxPatch((71, 4), 26, 92, boxstyle="round,pad=1.0,rounding_size=2.0",
                                    facecolor="#FAFAFA", edgecolor="#555555", linewidth=1.5, linestyle="--")
    ax.add_patch(col3_bg)

    # Column Headers
    def add_col_header(x, y, w, h, title, subtitle):
        hdr = patches.FancyBboxPatch((x, y), w, h, boxstyle="square,pad=0.5",
                                     facecolor="#E0E0E0", edgecolor="#000000", linewidth=box_edge_thick)
        ax.add_patch(hdr)
        ax.text(x + w/2, y + h*0.62, title, ha='center', va='center',
                fontsize=11.5, fontweight='bold', family=font_family, color='#000000')
        ax.text(x + w/2, y + h*0.26, subtitle, ha='center', va='center',
                fontsize=9.0, fontstyle='italic', family=font_family, color='#222222')

    add_col_header(4.5, 87.5, 23, 7, "1. MASALAH PENELITIAN", "Tantangan Fisik & Bahasa")
    add_col_header(38.5, 87.5, 23, 7, "2. SOLUSI TEKNOLOGI", "LSH-Memory-Wrapper (NumPy CPU)")
    add_col_header(72.5, 87.5, 23, 7, "3. EVALUASI & VALIDASI", "Tolok Ukur Kinerja & Akurasi")

    # Major Inter-Column Flow Arrows (Cause & Effect at header level)
    ax.annotate('', xy=(37, 91), xytext=(29, 91),
                arrowprops=dict(facecolor='#000000', edgecolor='#000000', width=3.5, headwidth=10, headlength=10))
    ax.text(33, 93, "Menjawab\nMasalah", ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#111111')

    ax.annotate('', xy=(71, 91), xytext=(63, 91),
                arrowprops=dict(facecolor='#000000', edgecolor='#000000', width=3.5, headwidth=10, headlength=10))
    ax.text(67, 93, "Menguji\nArtefak", ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#111111')

    # Helper function for drawing boxes
    def draw_node(x, y, w, h, title, lines, fill_color="#FFFFFF", edge_color="#000000", bold_title=True):
        box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.2",
                                     facecolor=fill_color, edgecolor=edge_color, linewidth=box_edge_normal)
        ax.add_patch(box)
        text_y = y + h - 2.2
        if title:
            ax.text(x + w/2, text_y, title, ha='center', va='center',
                    fontsize=10.0 if bold_title else 9.5, fontweight='bold' if bold_title else 'normal',
                    family=font_family, color='#000000')
            text_y -= 2.6
        for line in lines:
            ax.text(x + w/2, text_y, line, ha='center', va='center',
                    fontsize=8.5, family=font_family, color='#222222')
            text_y -= 2.1
        return (x, y, w, h)

    # ==================== COLUMN 2: SOLUSI TEKNOLOGI (TOP TO BOTTOM) ====================
    # 5 Sequential Process Nodes
    s_w = 23
    s_x = 38.5
    s_h = 9.5

    # Node S1 (Top)
    s1_y = 73.5
    draw_node(s_x, s1_y, s_w, s_h,
              "Tahap 1: Ekstraksi Dense Embedding",
              ["Model MiniLM 384-Dimensi", "L2-Normalization (||x||_2 = 1)", "Format Vektor Float32"])

    # Node S2
    s2_y = 58.5
    draw_node(s_x, s2_y, s_w, s_h,
              "Tahap 2: Hashing Data-Dependent",
              ["Mean Centering (X - mu)", "Principal Component Analysis (PCA)", "Iterative Quantization (ITQ 64/128-bit)"])

    # Node S3
    s3_y = 43.5
    draw_node(s_x, s3_y, s_w, s_h,
              "Tahap 3: Indeks Bitcode & Hamming Scan",
              ["Kompak Bit-Packing uint64 di RAM", "Scan Cepat: np.bitwise_count (POPCNT)", "Top-N Partial Sorting (np.argpartition)"])

    # Node S4
    s4_y = 28.5
    draw_node(s_x, s4_y, s_w, s_h,
              "Tahap 4: Raw Store & Re-ranking Episodik",
              ["Disk-backed np.memmap (Float32/Float16)", "Cosine Similarity Eksak pada N_pool", "Skor Gabungan: alpha*Cos + (1-alpha)*2^(-dt/h)"])

    # Node S5 (Bottom)
    s5_y = 13.5
    draw_node(s_x, s5_y, s_w, s_h,
              "Tahap 5: Prompt Context Injection",
              ["Adapter Llama.cpp (Token Budget Terkontrol)", "LangChain VectorStore / BaseRetriever", "Injeksi Memori Lampau Relevan"])

    # DOWNWARD ARROWS in Column 2 (FATAL ISSUE FIX: S1 -> S2 -> S3 -> S4 -> S5)
    def add_down_arrow(y_top, y_bottom, label=""):
        mid_y = (y_top + y_bottom) / 2
        ax.annotate('', xy=(s_x + s_w/2, y_bottom), xytext=(s_x + s_w/2, y_top),
                    arrowprops=dict(facecolor='#000000', edgecolor='#000000', width=2.2, headwidth=8, headlength=7))
        if label:
            ax.text(s_x + s_w/2 + 1.2, mid_y, label, ha='left', va='center',
                    fontsize=7.8, family=font_family, fontstyle='italic', color='#333333')

    add_down_arrow(s1_y, s2_y + s_h, "Vektor 384-D")
    add_down_arrow(s2_y, s3_y + s_h, "Bitcode biner")
    add_down_arrow(s3_y, s4_y + s_h, "N_pool kandidat")
    add_down_arrow(s4_y, s5_y + s_h, "k memori teratas")

    # ==================== COLUMN 1: MASALAH PENELITIAN ====================
    m_w = 23
    m_x = 4.5
    m_h = 13.5

    # M3 (Teks Campuran ID-EN) -> Maps to S2
    m3_y = 56.5
    draw_node(m_x, m3_y, m_w, m_h,
              "M3: Degenerasi Teks Campuran ID-EN",
              ["Teks bilingual (code-switching ID-EN)",
               "Ruang embedding alami anisotropik",
               "(cone effect / μ != 0)",
               "LSH acak (SimHash) menderita",
               "tingkat tabrakan biner & galat tinggi"],
              fill_color="#FDFDFD")

    # M2 (Keterbatasan Resource) -> Maps to S3 & S4
    m2_y = 35.0
    draw_node(m_x, m2_y, m_w, m_h,
              "M2: Keterbatasan Sumber Daya Edge",
              ["Budget RAM komputasi < 2GB di CPU",
               "VectorDB eksternal (Milvus/Pinecone)",
               "sangat berat (overhead RAM > 2GB)",
               "Indeks graf HNSW overhead 50-60%",
               "Penyimpanan disk harus efisien"],
              fill_color="#FDFDFD")

    # M1 (Amnesia Episodik) -> Maps to S5
    m1_y = 11.5
    draw_node(m_x, m1_y, m_w, m_h,
              "M1: Amnesia Episodik LLM Lokal",
              ["Model ringkas CPU (Phi-3, Llama-3 8B)",
               "Jendela konteks terbatas (context window)",
               "KV-Cache bottleneck (O(S^2) & DDR4 bandwidth)",
               "Trunkasi riwayat memicu amnesia fakta",
               "Lost in the Middle problem pada sekuens panjang"],
              fill_color="#FDFDFD")

    # ==================== COLUMN 3: EVALUASI & VALIDASI ====================
    e_w = 23
    e_x = 72.5
    e_h = 13.5

    # E1: Akurasi Temu Balik -> Tests S2
    e1_y = 56.5
    draw_node(e_x, e1_y, e_w, e_h,
              "E1: Akurasi Temu Balik Semantik",
              ["Recall@10 >= 85% vs Cosine Exact",
               "Mean Reciprocal Rank (MRR)",
               "Ablasi: Random LSH vs ITQ Hashing",
               "Ablasi: 64-bit vs 128-bit bitcode",
               "Uji terstratifikasi segmen teks ID & EN"],
              fill_color="#FDFDFD")

    # E2: Efisiensi Resource & Waktu -> Tests S3 & S4
    e2_y = 35.0
    draw_node(e_x, e2_y, e_w, e_h,
              "E2: Efisiensi Sumber Daya & Waktu",
              ["Waktu Encode < 2.0 s (50.000 vektor)",
               "Latensi Kueri p50 < 10 ms (12 thread)",
               "Peak Working Set RAM < 2 GB (target < 500MB)",
               "Ukuran Indeks RAM < 16 MB (1M entri)",
               "Komparasi: Float32 vs Float16 disk store"],
              fill_color="#FDFDFD")

    # E3: Evaluasi Downstream Needle QA -> Tests S5 & Solves M1
    e3_y = 11.5
    draw_node(e_x, e3_y, e_w, e_h,
              "E3: Evaluasi Downstream Needle QA",
              ["Akurasi Pemulihan Fakta Jangka Panjang",
               "Uji Multi-Turn Dialog (4 kondisi komparasi):",
               "- No Memory (0%)",
               "- Sliding Window Truncation (0%)",
               "- LSH-Memory-Wrapper (100% pada alpha=1.0)",
               "- Exact Cosine Upper Bound (100%)"],
              fill_color="#FDFDFD")

    # ==================== SPECIFIC CROSS-COLUMN MAPPING ARROWS ====================
    # Style: Dashed line with clear arrowheads showing cause -> solution -> evaluation

    # Mapping Row 1: M3 -> S2 -> E1
    ax.annotate('', xy=(s_x, s2_y + s_h/2), xytext=(m_x + m_w, m3_y + m_h/2),
                arrowprops=dict(facecolor='#333333', edgecolor='#333333', width=1.5, headwidth=7, headlength=7,
                                linestyle='--', linewidth=1.2))
    ax.annotate('', xy=(e_x, e1_y + e_h/2), xytext=(s_x + s_w, s2_y + s_h/2),
                arrowprops=dict(facecolor='#333333', edgecolor='#333333', width=1.5, headwidth=7, headlength=7,
                                linestyle='--', linewidth=1.2))

    # Mapping Row 2: M2 -> S3/S4 -> E2
    # M2 connects between S3 and S4
    s_mid_34_y = (s3_y + s4_y + s_h) / 2
    ax.annotate('', xy=(s_x, s_mid_34_y), xytext=(m_x + m_w, m2_y + m_h/2),
                arrowprops=dict(facecolor='#333333', edgecolor='#333333', width=1.5, headwidth=7, headlength=7,
                                linestyle='--', linewidth=1.2))
    ax.annotate('', xy=(e_x, e2_y + e_h/2), xytext=(s_x + s_w, s_mid_34_y),
                arrowprops=dict(facecolor='#333333', edgecolor='#333333', width=1.5, headwidth=7, headlength=7,
                                linestyle='--', linewidth=1.2))

    # Mapping Row 3: M1 -> S5 -> E3
    ax.annotate('', xy=(s_x, s5_y + s_h/2), xytext=(m_x + m_w, m1_y + m_h/2),
                arrowprops=dict(facecolor='#333333', edgecolor='#333333', width=1.5, headwidth=7, headlength=7,
                                linestyle='--', linewidth=1.2))
    ax.annotate('', xy=(e_x, e3_y + e_h/2), xytext=(s_x + s_w, s5_y + s_h/2),
                arrowprops=dict(facecolor='#333333', edgecolor='#333333', width=1.5, headwidth=7, headlength=7,
                                linestyle='--', linewidth=1.2))

    # Footer Note
    ax.text(50, 1.8,
            "Garis tebal padat: Alur proses sekuensial modul memori (Tahap 1 -> Tahap 5).   "
            "Garis putus-putus berpanah: Pemetaan spesifik Masalah -> Solusi Teknis -> Target Evaluasi.",
            ha='center', va='center', fontsize=8.5, fontstyle='italic', family=font_family, color='#444444')

    plt.tight_layout()
    plt.savefig(output_path, format='jpg', dpi=300, facecolor='white', bbox_inches='tight')
    plt.close()
    print(f"Diagram successfully generated and saved to {output_path}")

if __name__ == '__main__':
    draw_kerangka_pikiran()
