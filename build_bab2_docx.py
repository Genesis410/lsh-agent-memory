"""Script to generate complete Bab 2 docx document with embedded chart and exact IEEE citations."""

import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import qn, nsdecls
import os

doc = docx.Document()

# Margins
for s in doc.sections:
    s.top_margin = Inches(1.0)
    s.bottom_margin = Inches(1.0)
    s.left_margin = Inches(1.0)
    s.right_margin = Inches(1.0)

def set_font(run, name='Calibri', size=11, bold=False, italic=False, color=None):
    run.font.name = name
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    if color:
        run.font.color.rgb = color

def add_p(text, space_after=6, line_spacing=1.15, align=WD_ALIGN_PARAGRAPH.JUSTIFY):
    p = doc.add_paragraph()
    p.alignment = align
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = line_spacing
    r = p.add_run(text)
    set_font(r, size=11)
    return p

def add_heading_1(text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(18)
    p.paragraph_format.space_after = Pt(8)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(text)
    set_font(r, size=15, bold=True)
    return p

def add_heading_2(text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(text)
    set_font(r, size=13, bold=True)
    return p

def add_heading_3(text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(text)
    set_font(r, size=11.5, bold=True)
    return p

def add_formula(text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(6)
    r = p.add_run(text)
    set_font(r, name='Cambria Math', size=11, italic=True)
    return p

def style_table(table):
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for row_idx, row in enumerate(table.rows):
        trPr = row._tr.get_or_add_trPr()
        trPr.append(parse_xml(f'<w:cantSplit {nsdecls("w")}/>'))
        for cell in row.cells:
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            tcPr = cell._tc.get_or_add_tcPr()
            if row_idx == 0:
                shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="F2F2F2"/>')
                tcPr.append(shd)

# Title
add_heading_1("BAB 2. TINJAUAN PUSTAKA")

# 2.1 Landasan Teori
add_heading_2("2.1 Landasan Teori")

# 2.1.1
add_heading_3("2.1.1 Memori Jangka Panjang Agen AI dan Keterbatasan Context Window LLM Lokal")
add_p(
    "Perkembangan agen cerdas otonom (agentic AI) mentransformasikan interaksi manusia-komputer "
    "dari sistem tanya-jawab reaktif menjadi entitas kolaboratif yang menjalankan peran pendamping "
    "personal, asisten produktivitas, maupun eksekutor alur kerja multi-tahap [1], [3]. Dalam arsitektur "
    "kognitif agen, memori jangka panjang (long-term memory) memegang peranan krusial dalam mempertahankan "
    "kontinuitas identitas, preferensi personal pengguna, dan status tugas lintas sesi percakapan [2]. "
    "Secara psikologis dan komputasional, memori agen umumnya diklasifikasikan ke dalam working memory "
    "(konteks aktif dalam prompt saat ini), semantic memory (pengetahuan faktual umum), dan episodic memory "
    "(rekaman sekuensial peristiwa, percakapan masa lalu, dan pengalaman interaksi temporal) [1], [2]."
)
add_p(
    "Pada model bahasa besar (LLM) lokal berukuran ringkas seperti Phi-3 [4] maupun Llama-3 8B terkuantisasi [5] "
    "yang dieksekusi pada Central Processing Unit (CPU) komputer meja atau laptop, retensi memori jangka panjang "
    "mengalami hambatan fisik yang berat akibat keterbatasan context window. Arsitektur dasar Transformer beroperasi "
    "dengan mekanisme self-attention penuh yang membutuhkan komputasi berorde O(S^2), di mana S adalah panjang sekuens token. "
    "Pada tahap inferensi autoregresif (decoding), model menyimpan representasi aktivasi kunci dan nilai dalam Key-Value Cache (KV-Cache) "
    "agar tidak menghitung ulang token masa lalu [7]. Kebutuhan memori fisik untuk KV-Cache dirumuskan secara analitis sebagai:"
)
add_formula("M_KV = 2 × B × S × L × H_KV × D × P")
add_p(
    "di mana B merepresentasikan batch size (B=1 pada inferensi lokal pengguna tunggal), S adalah panjang sekuens total, "
    "L adalah jumlah lapisan Transformer, H_KV adalah jumlah kepala attention khusus key-value (menggunakan Grouped-Query Attention/GQA), "
    "D adalah dimensi per kepala attention, dan P adalah presisi bita tipe data (P=2 untuk FP16/BF16) [7]. "
    "Pada model Llama-3 8B (L=32, H_KV=8, D=128, P=2), kebutuhan KV-Cache mencapai 1,07 GB RAM pada S=8.192 token, "
    "melonjak menjadi 4,29 GB pada S=32.768 token, dan mencapai 16,78 GB pada jendela 128.000 token [7]. "
    "Di perangkat komputasi personal dengan RAM 16 GB DDR4 (bandwidth memori 35–45 GB/s), membaca seluruh matriks KV-Cache "
    "sebesar belasan gigabita untuk setiap token baru yang digenerasi menyebabkan degradasi kecepatan inferensi drastis (< 1–2 token/detik). "
    "Akibatnya, sistem terpaksa memotong (truncate) riwayat percakapan lama, yang memicu fenomena amnesia episodik [1], [6]."
)
add_p(
    "Lebih jauh, temuan empiris Liu dkk. (2024) mengenai fenomena Lost in the Middle [6] menegaskan bahwa memperbesar "
    "jendela konteks secara naif tidak menyelesaikan permasalahan retrieval informasi. Model bahasa terbukti memiliki bias posisi "
    "berbentuk kurva U, di mana performa ekstraksi fakta sangat tinggi jika informasi berada di awal (primacy effect) atau di akhir "
    "(recency effect) prompt, namun anjlok drastis ketika fakta berada di tengah-tengah rentang konteks panjang [6]. "
    "Kenyataan ini menuntut hadirnya mekanisme penarikan konteks selektif dari basis memori eksternal yang ringkas."
)

# 2.1.2
add_heading_3("2.1.2 Konsep Embedding Kalimat dan Cosine Similarity")
add_p(
    "Untuk memungkinkan temu balik semantik atas memori percakapan masa lalu, teks percakapan harus ditransformasikan "
    "ke dalam representasi vektor kontinu dense pada ruang berdimensi d (x ∈ R^d). Model sentence embedding modern "
    "seperti Sentence-BERT [17] dan MiniLM [16] memanfaatkan arsitektur jaringan siam (siamese) atau triplet yang dilatih "
    "untuk memetakan kalimat-kalimat yang memiliki kesamaan semantik ke koordinat yang berdekatan dalam ruang vektor euklidian."
)
add_p(
    "MiniLM (Wang dkk., 2020) mengaplikasikan distilasi self-attention mendalam (deep self-attention distillation) "
    "dari model guru berukuran besar ke model siswa yang ringkas, mentransfer relasi nilai dot-product perhatian dan matriks "
    "kemiripan antar token [16]. Hasilnya, varian seperti all-MiniLM-L6-v2 menghasilkan vektor berdimensi d=384 dengan bobot model "
    "kurang dari 90 MB dan memori kerja < 300 MB, menjadikannya standar representasi kalimat paling efisien di lingkungan CPU [16]. "
    "Tingkat kemiripan semantik antara vektor query q dan vektor memori m diukur menggunakan metrik kemiripan kosinus (cosine similarity):"
)
add_formula("cos(q, m) = (q · m) / (||q||_2 × ||m||_2) = (∑_{i=1}^d q_i m_i) / (√(∑_{i=1}^d q_i^2) × √(∑_{i=1}^d m_i^2))")
add_p(
    "Apabila seluruh vektor embedding dinormalisasi terhadap norma L2 (||v||_2 = 1), perhitungan kemiripan kosinus "
    "tereduksi secara elegan menjadi operasi perkalian dot product matriks: cos(q, m) = q^T m. "
    "Hal ini sangat menguntungkan secara komputasi pada CPU x86-64 karena dapat dihitung melalui operasi Basic Linear "
    "Algebra Subprograms (BLAS) level 3 yang terparalelisasi penuh."
)

# 2.1.3
add_heading_3("2.1.3 Approximate Nearest Neighbor (ANN) dan Kutukan Dimensi (Curse of Dimensionality)")
add_p(
    "Pencarian tetangga terdekat eksak (Exact Nearest Neighbor Search) atas basis data berisi N vektor memerlukan "
    "komparasi dot product penuh terhadap seluruh entri dengan kompleksitas waktu O(N × d). Meskipun terjangkau pada skala "
    "kecil (N ≤ 10.000), pendekatan ini menimbulkan latensi yang tidak dapat ditoleransi saat memori agen bertumbuh mencapai "
    "ratusan ribu hingga jutaan catatan interaksi [15]."
)
add_p(
    "Pada ruang berdimensi tinggi (d ≥ 100), metode partisi spasial konvensional seperti k-d tree, R-tree, atau Ball tree "
    "mengalami degradasi total menjadi pemindaian linier akibat fenomena kutukan dimensi (curse of dimensionality) [15]. "
    "Secara matematis, seiring bertambahnya dimensi d menuju tak hingga, selisih antara jarak terjauh (dist_max) dan "
    "jarak terdekat (dist_min) dari suatu titik query terhadap sekumpulan titik acak memudar menuju nol relatif terhadap jarak minimumnya:"
)
add_formula("lim_{d → ∞} (dist_max - dist_min) / dist_min = 0")
add_p(
    "Fenomena ini menyebabkan hampir seluruh titik data berada pada jarak yang nyaris ekuidistan dari titik query, "
    "sehingga partisi hiper-kotak (hyper-rectangle) gagal memangkas ruang pencarian [15]. Untuk mengatasi batas teoritis ini, "
    "paradigma Approximate Nearest Neighbor (ANN) melonggarkan jaminan pencarian eksak dengan menukar sejumlah kecil akurasi "
    "(recall) demi memperoleh kecepatan temu balik sub-linier O(log N) atau O(1) [8], [15]."
)

# 2.1.4
add_heading_3("2.1.4 Locality-Sensitive Hashing (LSH)")
add_p(
    "Locality-Sensitive Hashing (LSH) merupakan salah satu pilar algoritma ANN fundamental yang menyediakan jaminan matematis "
    "sub-linier tanpa memerlukan struktur graf yang masif [10], [14]. Konsep dasar LSH bertumpu pada perancangan keluarga fungsi hash H "
    "sedemikian rupa sehingga probabilitas tabrakan (collision) antara dua vektor berbanding lurus dengan tingkat kemiripannya [10], [14]. "
    "Secara formal, keluarga fungsi hash H disebut (d_1, d_2, p_1, p_2)-sensitive terhadap metrik jarak D jika untuk setiap pasang titik u, v ∈ R^d berlaku:\n"
    "1. Jika D(u, v) ≤ d_1, maka Pr_{h ∈ H}[h(u) = h(v)] ≥ p_1\n"
    "2. Jika D(u, v) ≥ d_2, maka Pr_{h ∈ H}[h(u) = h(v)] ≤ p_2\n"
    "dengan syarat d_1 < d_2 dan p_1 > p_2 [10], [14]."
)
add_p(
    "Untuk metrik kemiripan sudut kosinus, Charikar (2002) memperkenalkan skema proyeksi bidang batas acak (Random Hyperplane Projection "
    "atau SimHash) [20]. Diberikan vektor proyeksi acak r ∈ R^d yang disampel dari distribusi Gaussian baku multivariat N(0, I_d), "
    "fungsi hash biner didefinisikan berdasarkan tanda dot product (sign):"
)
add_formula("h_r(x) = 1 jika r^T x ≥ 0; h_r(x) = 0 jika r^T x < 0")
add_p(
    "Probabilitas tabrakan bit biner dari dua vektor u dan v sama dengan fraksi ruang hiper-bola di mana bidang batas normal r "
    "tidak memisahkan kedua vektor tersebut. Persamaan probabilitas tabrakan dirumuskan secara eksak sebagai [20]:"
)
add_formula("Pr[h_r(u) = h_r(v)] = 1 - (θ(u, v) / π)")
add_p(
    "di mana θ(u, v) = arccos(cos(u, v)) merupakan sudut rotasi dalam radian antara vektor u dan v. "
    "Untuk memperkuat diskriminasi pencarian, LSH menggabungkan K fungsi hash acak independen untuk membentuk satu signature biner "
    "g(x) = (h_1(x), ..., h_K(x)), dan mendistribusikan data ke dalam L tabel hash independen. "
    "Probabilitas bahwa dua vektor u dan v bertabrakan pada setidaknya satu dari L tabel hash dirumuskan sebagai [10]:"
)
add_formula("P_collision = 1 - (1 - p^K)^L")
add_p(
    "di mana p = 1 - (θ / π). Peningkatan nilai K menekan tingkat false positive (meningkatkan presisi), "
    "sedangkan penambahan jumlah tabel L menekan false negative (meningkatkan recall) namun meningkatkan konsumsi memori secara proporsional [10], [14]."
)

# 2.1.5
add_heading_3("2.1.5 Hashing Data-Dependent: Centering, PCA, dan ITQ")
add_p(
    "Meskipun SimHash klasik memiliki keanggunan teoritis, kinerjanya merosot tajam ketika diterapkan pada representasi embedding kalimat modern. "
    "Hal ini disebabkan oleh fenomena degenerasi representasi teks atau anisotropi (representation degeneration problem / cone effect) [12]. "
    "Vektor-vektor embedding bahasa alami tidak tersebar seragam secara isotropik di seluruh permukaan bola satuan hiper-bola, "
    "melainkan mengumpul dalam sebuah kerucut sempit dengan korelasi tinggi antar-dimensi serta vektor rata-rata yang menyimpang jauh dari titik nol (μ ≠ 0) [12]. "
    "Akibatnya, proyeksi acak isotropik memotong ruang representasi secara tidak optimal: banyak bidang batas memotong ruang kosong tanpa membagi data, "
    "sementara bidang lainnya membelah data secara sangat timpang, memicu lonjakan bit redundancy dan galat kuantisasi biner yang masif [11], [18]."
)
add_p(
    "Untuk mengatasi patologi anisotropi pada teks campuran bilingual, diperlukan pendekatan hashing data-dependent "
    "yang mengadaptasi orientasi bidang pemotong sesuai dengan struktur kovarians data riil [11], [18]. "
    "Metodologi ini terdiri dari tiga tahapan sistematis:\n"
    "1. Pemusatan Rata-Rata (Mean Centering):\n"
    "Matriks data X ∈ R^{N × d} ditransformasikan dengan mengurangkan vektor rata-rata empiris μ = (1/N) ∑_{i=1}^N x_i, menghasilkan data terpusat X_c = X - μ. "
    "Langkah ini memindahkan pusat massa distribusi anisotropik tepat ke titik asal koordinat.\n"
    "2. Reduksi Dimensi Principal Component Analysis (PCA):\n"
    "Dihitung matriks kovarians sampel C = (X_c^T X_c) / N ∈ R^{d × d}. Melalui dekomposisi nilai eigen C = V Λ V^T, diekstraksi c komponen utama teratas "
    "P = V_{:, -c:} ∈ R^{d × c} yang menangkap varians data terbesar, menghasilkan proyeksi awal Z = X_c P ∈ R^{N × c}.\n"
    "3. Iterative Quantization (ITQ):\n"
    "Gong dan Lazebnik (2013) membuktikan bahwa proyeksi PCA mentah masih menghasilkan galat kuantisasi tinggi karena verteks biner {-1, 1}^c "
    "tidak sejajar dengan sumbu-sumbu komponen utama [18]. ITQ merumuskan pencarian matriks rotasi ortogonal optimal R ∈ R^{c × c} "
    "yang meminimalkan galat aproksimasi Frobenius antara proyeksi kontinu terotasi Z R dan kode biner B ∈ {-1, 1}^{N × c}:"
)
add_formula("min_{B, R} ||B - Z R||_F^2   dengan kendala   R^T R = I_c,  B ∈ {-1, 1}^{N × c}")
add_p(
    "Solusi optimal dicapai secara iteratif melalui optimasi bergantian (alternating optimization). "
    "Pada setiap iterasi t: (a) Matriks biner diperbarui melalui B = sign(Z R); "
    "(b) Matriks rotasi diperbarui dengan menghitung Singular Value Decomposition (SVD) dari matriks B^T Z = U Σ V^T, "
    "dan menetapkan R = (U V^T)^T [18]. Matriks proyeksi akhir gabungan W = P R ∈ R^{d × c} "
    "memastikan setiap bit biner memiliki varians maksimal dan korelasi antar-bit mendekati nol [11], [18]."
)

# 2.1.6
add_heading_3("2.1.6 Multi-probe LSH, Hamming Distance, dan Re-ranking")
add_p(
    "Dalam skema hashing biner c-bit (misalnya c=64 atau c=128 bit), setiap representasi vektor dikonversi menjadi bitcode biner. "
    "Untuk efisiensi perangkat keras pada arsitektur x86-64, bitcode dipadatkan (bit-packing) menjadi array bertipe integer 64-bit tak bertanda (uint64), "
    "di mana kode 128-bit hanya membutuhkan w = 2 word per entri (16 bita). "
    "Jarak Hamming antara bitcode query b_q dan bitcode basis data b_i dihitung melalui operasi bitwise XOR yang diikuti instruksi hardware population count (popcount):"
)
add_formula("D_H(b_q, b_i) = ∑_{j=1}^w popcount(b_{q, j} ⊕ b_{i, j})")
add_p(
    "NumPy versi 2.0+ mengekspos instruksi native CPU ini secara langsung melalui fungsi ufunc np.bitwise_count, "
    "yang beroperasi pada register SIMD dan melepas Global Interpreter Lock (GIL) Python, memungkinkan pemindaian multi-thread penuh pada 12 thread Ryzen [11]. "
    "Untuk mengatasi boundary effect—kondisi di mana dua vektor tetangga dekat semantik terpisah oleh bidang pemotong biner tipis—diterapkan "
    "prinsip Multi-probe LSH, yakni memeriksa kandidat dengan jarak Hamming D_H ≤ r di sekitar kode query [10]."
)
add_p(
    "Sistem menerapkan paradigma temu balik dua tahap (Two-Stage Retrieval):\n"
    "1. Tahap Penyaringan Hamming (Filter Stage): Memindai seluruh bitcode di RAM berorde O(N) dan melakukan seleksi sebagian (partial selection) "
    "menggunakan algoritma np.argpartition untuk menyaring N_pool kandidat teratas (misalnya N_pool = 250 hingga 500) dengan jarak Hamming terkecil.\n"
    "2. Tahap Pemeringkatan Ulang (Re-ranking Stage): Mengambil vektor mentah Float32 dari N_pool kandidat terpilih dari disk via np.memmap, "
    "kemudian menghitung exact cosine similarity dan mengurutkan k hasil teratas. "
    "Pendekatan ini membatasi komputasi floating-point berat hanya pada fraksi kecil data (≤ 0,05% pada 1 juta entri)."
)

# 2.1.7
add_heading_3("2.1.7 Skor Memori Episodik: Relevansi Semantik dan Peluruhan Waktu Paruh (Half-Life)")
add_p(
    "Sistem memori agen kognitif yang realistis tidak boleh hanya bertumpu pada kemiripan semantik statis. "
    "Memori episodik manusia secara alami menyeimbangkan antara relevansi topik dan aktualitas waktu (recency), "
    "sebagaimana diadaptasi dalam studi agen otonom Generative Agents [3] dan MemoryBank [2]. "
    "Dalam penelitian ini, skor akhir memori episodik dirumuskan sebagai kombinasi linier terbobot antara relevansi semantik dan fungsi peluruhan waktu paruh:"
)
add_formula("Score(q, m) = α · cos(q, m) + (1 - α) · Decay(Δt)")
add_p(
    "di mana α ∈ [0, 1] adalah hiperparameter bobot komparatif. Fungsi peluruhan waktu dirancang mengadopsi formulasi waktu paruh fisika (half-life):"
)
add_formula("Decay(Δt) = exp(-λ · Δt) = 2^(-Δt / h),   di mana λ = ln(2) / h")
add_p(
    "di mana Δt merepresentasikan selisih waktu atau jarak giliran dialog (turn distance) antara saat memori dicatat dan query saat ini, "
    "dan h adalah parameter waktu paruh (half-life). Keunggulan formulasi berbasis h terletak pada interpretabilitasnya yang intuitif: "
    "nilai aktualitas memori akan menyusut tepat sebesar 50% setelah h unit waktu berlalu (misalnya h=200 giliran atau h=7 hari). "
    "Peluruhan ini hanya diaplikasikan pada tahap pemeringkatan ulang (re-ranking) kandidat, sehingga fakta needle lampau yang memiliki kesamaan semantik "
    "sangat tinggi tidak tereliminasi secara prematur pada tahap penyaringan Hamming."
)

# 2.1.8
add_heading_3("2.1.8 Metrik Evaluasi Sistem Memori")
add_p(
    "Untuk mengevaluasi kinerja sistem memori episodik secara objektif dan kuantitatif, digunakan empat dimensi metrik terstandarisasi [15]:\n"
    "1. Recall@k:\n"
    "Mengukur fraksi tetangga terdekat sejati (ground truth exact cosine G_k) yang berhasil masuk ke dalam himpunan hasil temu balik sistem R_k:"
)
add_formula("Recall@k = |R_k ∩ G_k| / k")
add_p(
    "2. Mean Reciprocal Rank (MRR):\n"
    "Mengevaluasi presisi urutan posisi dokumen atau memori target pertama yang relevan dalam daftar peringkat hasil temu balik:"
)
add_formula("MRR = (1 / |Q|) ∑_{i=1}^{|Q|} (1 / rank_i)")
add_p(
    "3. Persentil Latensi Query (p50 dan p95):\n"
    "Mengukur waktu respons sistem dalam milidetik (ms) untuk memproses query dari pengkodean biner hingga pemeringkatan akhir. "
    "Nilai p50 mencerminkan latensi tipikal (median), sedangkan p95 merepresentasikan jaminan performa batas atas pada kondisi beban ekor (tail latency).\n"
    "4. Konsumsi RAM Puncak (Peak Process Working Set):\n"
    "Mengukur pemakaian memori fisik riil (Resident Set Size/Working Set) proses Python pada sistem operasi Windows 11 menggunakan pustaka psutil, "
    "memastikan isolasi pengukuran dari file page cache sistem operasi."
)

# 2.1.9
add_heading_3("2.1.9 Design Science Research Methodology (DSRM)")
add_p(
    "Penelitian perancangan artefak perangkat lunak ini berpedoman pada kerangka kerja Design Science Research Methodology (DSRM) "
    "yang dikembangkan oleh Peffers dkk. (2007) [19]. DSRM menyediakan pendekatan terstruktur berbasis pembuktian ilmiah untuk rekayasa artefak sistem informasi. "
    "Enam tahapan DSRM diintegrasikan ke dalam siklus penelitian ini:\n"
    "1. Identifikasi Masalah dan Motivasi: Menganalisis batasan context window model bahasa lokal pada CPU dan kelemahan Vector DB konvensional.\n"
    "2. Definisi Tujuan Solusi: Menetapkan sasaran kuantitatif terukur (waktu encode 50k < 2s, RAM < 2GB, Recall@10 ≥ 85%).\n"
    "3. Desain dan Pengembangan Artefak: Mengonstruksi modul arsitektural LSH-Memory-Wrapper (Hasher ITQ, BitIndex, MemmapStore, Scorer, Adapter).\n"
    "4. Demonstrasi Artefak: Mengintegrasikan pustaka pada eksekusi inferensi agen percakapan lokal berbasis Llama.cpp.\n"
    "5. Evaluasi Kinerja: Menguji efisiensi komputasi, konsumsi memori, ablasi parameter, dan akurasi temu balik needle-in-haystack.\n"
    "6. Komunikasi Hasil: Menyusun laporan riset akademik, dokumentasi teknis, dan mempublikasikan kode sumber terbuka."
)

# 2.2 Studi Pustaka
add_heading_2("2.2 Studi Pustaka")
add_p(
    "Sebagai landasan komparatif dan penegasan posisi penelitian terhadap perkembangan ilmu pengetahuan terkini, "
    "dilakukan sintesis komprehensif atas literatur bereputasi internasional terbitan tahun 2020 hingga 2026. "
    "Literatur dikelompokkan ke dalam empat pilar tematik utama: Penelitian LSH dan Kuantisasi Biner, Sistem ANN Modern, "
    "Model Embedding Teks Bahasa Campuran/Ringan, serta Sistem Memori Agen Cerdas."
)

# Table 2.1
p_t1 = doc.add_paragraph()
r_t1 = p_t1.add_run("Tabel 2.1. Sintesis Perbandingan Literatur Terkait (2020–2026)")
set_font(r_t1, size=11, bold=True)
p_t1.paragraph_format.space_before = Pt(8)
p_t1.paragraph_format.space_after = Pt(4)

table_data = [
    ("Kategori & Peneliti", "Metode Utama", "Hasil Utama", "Kelemahan & Research Gap"),
    # Kategori 1: LSH
    ("1. LSH & Hashing\nJafari dkk. (2021) [10]", "Survei komprehensif algoritma LSH dan Distributed LSH untuk NNS.", "Karakterisasi teoritis sub-linear query time dan varian hashing multi-probe.", "Fokus komputasi terdistribusi; tidak mengulas kasus khusus anisotropi teks LLM."),
    ("Liu dkk. (2024) [11]\nIEEE TKDE", "Reduksi dan penyempurnaan bit LSH (Two-stage bit refinement) untuk data berskala besar.", "Meningkatkan daya diskriminasi bitcode dan menurunkan redundansi korelasi antar-bit.", "Diuji pada korpus citra umum; belum diintegrasikan sebagai modul memori agen CPU."),
    ("Singh & Gupta (2022) [14]\nKnowl. Inf. Syst.", "Taksonomi metode deep learning-based supervised dan unsupervised hashing.", "Menunjukkan keunggulan kuantisasi ortogonal dalam menjaga topologi jarak semantik.", "Kompleksitas pelatihan deep hashing tinggi; tidak praktis untuk cold-start on-device."),
    # Kategori 2: Modern ANN
    ("2. Modern ANN\nMalkov & Yashunin (2020) [8]\nIEEE TPAMI", "Hierarchical Navigable Small World (HNSW) graph indexing.", "Mencapai recall mendekati 99% dengan kecepatan kueri sub-milidetik.", "Overhead memori graf 50-60% di atas data vektor; melebihi batas 2GB RAM pada skala 1M."),
    ("Johnson dkk. (2021) [9]\nIEEE TBD", "Billion-scale similarity search (FAISS) pada arsitektur GPU dan CPU.", "Sangat efisien untuk komputasi batch paralel masif berbasis instruksi SIMD.", "Dependensi kompilasi C++ besar; tidak memiliki lapisan peluruhan memori episodik."),
    ("Aumüller dkk. (2020) [15]\nInformation Systems", "Kerangka pembandingan terstandarisasi ANN-Benchmarks.", "Analisis trade-off recall vs throughput lintas puluhan algoritma ANN.", "Evaluasi terfokus pada dataset statis tanpa faktor peluruhan temporal memori interaktif."),
    # Kategori 3: Embedding
    ("3. Embedding & Teks\nWang dkk. (2020) [16]\nNeurIPS", "MiniLM: Deep self-attention distillation untuk kompresi Transformer.", "Model embedding 384-d dengan ukuran < 90MB, inferensi sangat cepat di CPU.", "Korpus pelatihan didominasi bahasa Inggris; performa menurun pada alih kode lokal."),
    ("Reimers & Gurevych (2020) [17]\nEMNLP", "Multilingual sentence embeddings via knowledge distillation (paraphrase-multilingual).", "Menyelaraskan representasi multibahasa ke dalam ruang vektor bersama.", "Ruang representasi tetap menderita anisotropi distribusi kerucut pada CPU."),
    ("Gao dkk. (2020) [12]\nICLR", "Analisis representation degeneration problem (anisotropi) pada representasi teks.", "Membuktikan adanya korelasi spurious dimensi dan pergeseran nilai ekspektasi μ ≠ 0.", "Menyarankan regularisasi pelatihan, namun belum merumuskan kuantisasi hashing data-dependent."),
    # Kategori 4: Memory Agen
    ("4. Memori Agen\nPacker dkk. (2023) [1]\nMemGPT", "Virtual context management berbasis analogi hierarki memori sistem operasi.", "Memungkinkan obrolan multi-sesi tanpa batas context window melalui function calling.", "Sangat bergantung pada server database PostgreSQL eksternal dan API model komputasi awan."),
    ("Zhong dkk. (2024) [2]\nAAAI", "MemoryBank: Mekanisme memori jangka panjang berbasis Ebbinghaus Forgetting Curve.", "Pembaruan memori dinamis dan empati percakapan yang meningkat lintas sesi.", "Menggunakan brute-force atau vector store konvensional tanpa optimalisasi kuantisasi bitcode CPU."),
    ("Park dkk. (2023) [3]\nACM UIST", "Generative Agents: Arsitektur memori retrieval, refleksi, dan perencanaan 25 agen simulasi.", "Menghasilkan dinamika perilaku sosial agen yang konsisten berhari-hari.", "Biaya penarikan memori sangat mahal; latensi membengkak seiring bertambahnya riwayat interaksi.")
]

t1 = doc.add_table(rows=len(table_data), cols=4)
style_table(t1)
col_widths_t1 = [Inches(1.5), Inches(1.8), Inches(1.8), Inches(1.9)]
for row_idx, row in enumerate(t1.rows):
    for col_idx, cell in enumerate(row.cells):
        cell.width = col_widths_t1[col_idx]
        text_val = table_data[row_idx][col_idx]
        cell.paragraphs[0].text = text_val
        p_c = cell.paragraphs[0]
        p_c.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p_c.paragraph_format.space_before = Pt(2)
        p_c.paragraph_format.space_after = Pt(2)
        p_c.paragraph_format.line_spacing = 1.05
        for r_c in p_c.runs:
            set_font(r_c, size=9, bold=(row_idx == 0))

# Novelty Matrix
add_p("", space_after=6)
p_t2 = doc.add_paragraph()
r_t2 = p_t2.add_run("Tabel 2.2. Matriks Kebaruan Fitur LSH-Memory-Wrapper terhadap Sistem Eksisting")
set_font(r_t2, size=11, bold=True)
p_t2.paragraph_format.space_before = Pt(8)
p_t2.paragraph_format.space_after = Pt(4)

matrix_data = [
    ("Pendekatan / Sistem", "CPU-only Ringan", "Tanpa DB Eksternal", "RAM < 2GB (1M)", "Teks Campuran ID-EN", "Hash Data-Dependent", "Lapisan Memori Agen", "Integrasi Llama.cpp"),
    ("FAISS [9]", "✓", "✗", "✗", "✗", "✗", "✗", "✗"),
    ("hnswlib / Annoy [8]", "✓", "✓", "✗", "✗", "✗", "✗", "✗"),
    ("sqlite-vec / Chroma", "✓", "✗", "✗", "✗", "✗", "✗", "✗"),
    ("MemGPT & Sejenisnya [1]", "✗", "✗", "✗", "✗", "✗", "✓", "✗"),
    ("LSH-Memory-Wrapper (Riset Ini)", "✓", "✓", "✓", "✓", "✓", "✓", "✓")
]

t2 = doc.add_table(rows=len(matrix_data), cols=8)
style_table(t2)
col_widths_t2 = [Inches(1.8), Inches(0.7), Inches(0.8), Inches(0.8), Inches(0.8), Inches(0.8), Inches(0.7), Inches(0.7)]
for row_idx, row in enumerate(t2.rows):
    for col_idx, cell in enumerate(row.cells):
        cell.width = col_widths_t2[col_idx]
        text_val = matrix_data[row_idx][col_idx]
        cell.paragraphs[0].text = text_val
        p_c = cell.paragraphs[0]
        p_c.alignment = WD_ALIGN_PARAGRAPH.CENTER if col_idx > 0 else WD_ALIGN_PARAGRAPH.LEFT
        p_c.paragraph_format.space_before = Pt(2)
        p_c.paragraph_format.space_after = Pt(2)
        p_c.paragraph_format.line_spacing = 1.05
        for r_c in p_c.runs:
            is_bold = (row_idx == 0 or row_idx == 5)
            set_font(r_c, size=9, bold=is_bold)

# 2.3 Kerangka Pikiran
add_heading_2("2.3 Kerangka Pikiran")
add_p(
    "Kerangka pemikiran penelitian ini dibangun di atas hubungan kausal antara kendala komputasi fisik, "
    "solusi rekayasa perangkat lunak berbasis struktur data, dan pembuktian empiris. "
    "Alur kerangka pikiran terbagi menjadi tiga tahapan terintegrasi:\n"
    "1. MASALAH:\n"
    "Keterbatasan mendasar berakar pada amnesia kontekstual model bahasa lokal kecil (Phi-3, Llama-3 8B terkuantisasi) "
    "akibat sempitnya context window (M1); keterbatasan sumber daya RAM fisik pada perangkat laptop/desktop CPU (< 2 GB alokasi proses) "
    "serta tingginya overhead memori server VectorDB eksternal (M2); dan tantangan pemetaan representasi teks campuran "
    "Indonesia-Inggris yang mengalami degenerasi anisotropik pada ruang embedding (M3).\n"
    "2. SOLUSI TEKNOLOGI:\n"
    "Dikonstruksi modul LSH-Memory-Wrapper yang beroperasi secara sekuensial dari atas ke bawah: "
    "(a) Tahap 1: Ekstraksi kalimat ke dalam dense embedding MiniLM 384 dimensi bernorma satuan; "
    "(b) Tahap 2: Transformasi hashing data-dependent melalui pemusatan rata-rata (centering), reduksi kovarians PCA, dan iterasi rotasi ortogonal ITQ guna memitigasi galat kuantisasi pada teks bilingual; "
    "(c) Tahap 3: Penyimpanan bitcode biner padat uint64 di RAM yang dipindai secara cepat via operasi np.bitwise_count dan seleksi argpartition O(N) multi-thread; "
    "(d) Tahap 4: Penyimpanan vektor mentah di disk (np.memmap) dan pemeringkatan ulang menggabungkan exact cosine similarity dengan skor aktualitas waktu paruh (half-life decay); "
    "(e) Tahap 5: Injeksi konteks terformat ke prompt Llama.cpp di bawah batasan token budget tertentu serta penyediaan adapter LangChain VectorStore.\n"
    "3. EVALUASI DAN VALIDASI:\n"
    "Keberhasilan arsitektur divalidasi melalui tiga pilar pengujian: "
    "(a) E1: Pengukuran akurasi temu balik Recall@k dan MRR terhadap ground truth exact cosine brute-force serta ablasi ITQ vs Random LSH; "
    "(b) E2: Pengujian efisiensi komputasi mencakup waktu encode 50k entri (< 2s), latensi kueri p50 (< 10 ms), konsumsi working set RAM (< 2 GB), dan trade-off presisi disk Float32 vs Float16; "
    "(c) E3: Pengujian hilir (downstream QA) needle-in-haystack untuk membuktikan peningkatan akurasi penarikan fakta percakapan multi-sesi dari 0% menjadi 100% pada model bahasa lokal."
)

add_p(
    "Representasi alur logika kerangka pikiran tersebut dimodelkan dalam diagram alir Mermaid.js berikut:"
)

# Mermaid Code Block
mermaid_code = (
    "```mermaid\n"
    "flowchart LR\n"
    "    subgraph MASALAH [\"1. MASALAH PENELITIAN\"]\n"
    "        direction TB\n"
    "        M1[\"M1: Amnesia Episodik LLM Lokal<br/>(Context window terbatas pada CPU)\"]\n"
    "        M2[\"M2: Keterbatasan Sumber Daya Edge<br/>(RAM < 2GB, VectorDB eksternal berat)\"]\n"
    "        M3[\"M3: Degenerasi Teks Campuran ID-EN<br/>(Anisotropi ruang embedding)\"]\n"
    "    end\n\n"
    "    subgraph SOLUSI [\"2. SOLUSI TEKNOLOGI (LSH-MEMORY-WRAPPER)\"]\n"
    "        direction TB\n"
    "        S1[\"Tahap 1: Ekstraksi Dense Embedding<br/>(MiniLM 384-D, L2 Norm)\"]\n"
    "        S2[\"Tahap 2: Hashing Data-Dependent<br/>(Centering + PCA + ITQ)\"]\n"
    "        S3[\"Tahap 3: Indeks Bitcode uint64 di RAM<br/>(Parallel Hamming Scan via bitwise_count)\"]\n"
    "        S4[\"Tahap 4: Raw Store & Re-ranking Episodik<br/>(np.memmap + Cosine Exact + Half-Life Decay)\"]\n"
    "        S5[\"Tahap 5: Prompt Context Injection<br/>(Llama.cpp & LangChain Adapter)\"]\n"
    "        S1 --> S2 --> S3 --> S4 --> S5\n"
    "    end\n\n"
    "    subgraph EVALUASI [\"3. EVALUASI DAN VALIDASI\"]\n"
    "        direction TB\n"
    "        E1[\"E1: Akurasi Temu Balik Semantik<br/>(Recall@10 >= 85%, MRR, Ablasi ITQ)\"]\n"
    "        E2[\"E2: Efisiensi Sumber Daya & Waktu<br/>(Encode < 2s, Latensi < 10ms, RAM < 2GB)\"]\n"
    "        E3[\"E3: Evaluasi Downstream Needle QA<br/>(Akurasi Pemulihan Fakta 100%)\"]\n"
    "    end\n\n"
    "    %% Hubungan Kausal Antar-Kolom\n"
    "    MASALAH ==>|\"Menjawab Masalah\"| SOLUSI\n"
    "    SOLUSI ==>|\"Menguji Artefak\"| EVALUASI\n\n"
    "    %% Pemetaan Spesifik (Mapping)\n"
    "    M3 -.->|\"Mitigasi Anisotropi\"| S2\n"
    "    S2 -.->|\"Validasi Recall\"| E1\n\n"
    "    M2 -.->|\"Alokasi RAM Ringan\"| S3\n"
    "    S4 -.->|\"Validasi Benchmark\"| E2\n\n"
    "    M1 -.->|\"Restorasi Konteks\"| S5\n"
    "    S5 -.->|\"Validasi Downstream\"| E3\n"
    "```"
)

p_code = doc.add_paragraph()
p_code.paragraph_format.space_before = Pt(4)
p_code.paragraph_format.space_after = Pt(6)
r_code = p_code.add_run(mermaid_code)
set_font(r_code, name='Consolas', size=9.5)

# Insert Picture
img_path = 'paper/kerangka_pikiran.jpg'
if os.path.exists(img_path):
    p_img = doc.add_paragraph()
    p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_img.paragraph_format.space_before = Pt(8)
    p_img.paragraph_format.space_after = Pt(4)
    run_img = p_img.add_run()
    run_img.add_picture(img_path, width=Inches(6.2))

    p_cap = doc.add_paragraph()
    p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cap.paragraph_format.space_after = Pt(12)
    r_cap = p_cap.add_run("Gambar 2.1. Diagram Kerangka Pikiran Penelitian LSH-Memory-Wrapper (Monokrom)")
    set_font(r_cap, size=10, italic=True)

# Daftar Pustaka Sementara Bab 2
add_heading_2("Daftar Pustaka Sementara Bab 2")

references = [
    "[1] C. Packer, S. Wooders, K. Lin, V. Fang, S. G. Patil, I. Stoica, and J. E. Gonzalez, \"MemGPT: Towards LLMs as operating systems,\" arXiv preprint arXiv:2310.08560, 2023.",
    "[2] W. Zhong, L. Guo, Q. Gao, H. Ye, and Y. Wang, \"MemoryBank: Enhancing large language models with long-term memory,\" in Proc. AAAI Conf. Artif. Intell., vol. 38, no. 17, pp. 19729–19737, 2024.",
    "[3] J. S. Park, J. C. O'Brien, C. J. Cai, M. R. Morris, P. Liang, and M. S. Bernstein, \"Generative agents: Interactive simulacra of human behavior,\" in Proc. 36th Annu. ACM Symp. User Interface Softw. Technol. (UIST), pp. 1–22, 2023.",
    "[4] M. Abdin et al., \"Phi-3 technical report: A highly capable language model locally on your phone,\" arXiv preprint arXiv:2404.14219, 2024.",
    "[5] A. Dubey et al., \"The Llama 3 herd of models,\" arXiv preprint arXiv:2407.21783, 2024.",
    "[6] N. F. Liu, K. Lin, J. Hewitt, A. Paranjape, M. Bevilacqua, F. Petroni, and P. Liang, \"Lost in the middle: How language models use long contexts,\" Trans. Assoc. Comput. Linguist., vol. 12, pp. 157–173, 2024.",
    "[7] Y. Li et al., \"A KV cache-centric analysis of long-context methods,\" arXiv preprint arXiv:2412.10319, 2024.",
    "[8] Y. A. Malkov and D. A. Yashunin, \"Efficient and robust approximate nearest neighbor search using hierarchical navigable small world graphs,\" IEEE Trans. Pattern Anal. Mach. Intell., vol. 42, no. 4, pp. 824–836, 2020.",
    "[9] J. Johnson, M. Douze, and H. Jégou, \"Billion-scale similarity search with GPUs,\" IEEE Trans. Big Data, vol. 7, no. 3, pp. 535–547, 2021.",
    "[10] O. Jafari, P. Maurya, P. Nagarkar, K. M. Islam, and C. Crushev, \"A survey on locality sensitive hashing algorithms and their applications,\" arXiv preprint arXiv:2102.08942, 2021.",
    "[11] H. Liu, W. Zhou, Z. Wu, S. Zhang, and X. Li, \"Refining codes for locality sensitive hashing,\" IEEE Trans. Knowl. Data Eng., vol. 36, no. 3, pp. 1274–1284, 2024.",
    "[12] J. Gao, D. He, X. Tan, T. Qin, L. Wang, and T. Liu, \"Representation degeneration problem in training natural language generation models,\" in Proc. Int. Conf. Learn. Represent. (ICLR), 2020.",
    "[13] Republik Indonesia, \"Undang-Undang Republik Indonesia Nomor 27 Tahun 2022 tentang Pelindungan Data Pribadi,\" Lembaran Negara Republik Indonesia Tahun 2022 Nomor 196, 2022.",
    "[14] A. Singh and S. Gupta, \"Learning to hash: a comprehensive survey of deep learning-based hashing methods,\" Knowl. Inf. Syst., vol. 64, no. 10, pp. 2567–2618, 2022.",
    "[15] M. Aumüller, E. Bernhardsson, and A. Faithfull, \"ANN-Benchmarks: A benchmarking tool for approximate nearest neighbor algorithms,\" Inf. Syst., vol. 87, p. 101374, 2020.",
    "[16] W. Wang et al., \"MiniLM: Deep self-attention distillation for task-agnostic compression of pre-trained transformers,\" in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), vol. 33, pp. 5776–5788, 2020.",
    "[17] N. Reimers and I. Gurevych, \"Making monolingual sentence embeddings multilingual using knowledge distillation,\" in Proc. Conf. Empir. Methods Nat. Lang. Process. (EMNLP), pp. 4512–4525, 2020.",
    "[18] Y. Gong, S. Lazebnik, A. Gordo, and F. Perronnin, \"Iterative quantization: A procrustean approach to learning binary codes for large-scale image retrieval,\" IEEE Trans. Pattern Anal. Mach. Intell., vol. 35, no. 12, pp. 2916–2929, 2013.",
    "[19] K. Peffers, T. Tuunanen, M. A. Rothenberger, and S. Chatterjee, \"A design science research methodology for information systems research,\" J. Manage. Inf. Syst., vol. 24, no. 3, pp. 45–77, 2007.",
    "[20] M. S. Charikar, \"Similarity estimation techniques from rounding algorithms,\" in Proc. 34th Annu. ACM Symp. Theory Comput. (STOC), pp. 380–388, 2002."
]

for ref in references:
    p_ref = doc.add_paragraph()
    p_ref.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p_ref.paragraph_format.space_before = Pt(2)
    p_ref.paragraph_format.space_after = Pt(4)
    p_ref.paragraph_format.line_spacing = 1.15
    r_ref = p_ref.add_run(ref)
    set_font(r_ref, size=10)

out_path = 'paper/bab2.docx'
out_path_rev = 'paper/bab2_revisi.docx'
try:
    doc.save(out_path)
    print(f'Document successfully created and saved to {out_path}')
except PermissionError:
    doc.save(out_path_rev)
    print(f'Warning: {out_path} is currently locked by Word. Saved to {out_path_rev} instead.')
