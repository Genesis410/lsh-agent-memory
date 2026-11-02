"""Script to generate complete Bab 3 in both DOCX and TXT formats."""

import os
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

def create_bab3_documents():
    doc = docx.Document()

    # Set Margins (1 inch / 2.54 cm all sides)
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

    # Document Header
    add_heading_1("BAB 3. METODOLOGI PENELITIAN")

    # 3.1 Jenis Pendekatan
    add_heading_2("3.1 Jenis Pendekatan")
    add_p(
        "Penelitian ini menggunakan pendekatan kuantitatif-eksperimental yang dipadukan dengan kerangka kerja "
        "Design Science Research Methodology (DSRM) yang dirumuskan oleh Peffers dkk. [21]. Pendekatan "
        "kuantitatif diterapkan untuk mengukur kinerja komputasi dan akurasi sistem temu balik secara objektif "
        "melalui metrik terstandarisasi, seperti Recall@k, Mean Reciprocal Rank (MRR), waktu pembentukan indeks, "
        "serta konsumsi memori fisik (RAM) [15]. Metode eksperimental diwujudkan melalui pengujian terkontrol "
        "terhadap variabel arsitektur, parameter kuantisasi, dan variasi data masukan guna menguji hipotesis kinerja sistem."
    )
    add_p(
        "Kerangka kerja DSRM dipilih karena fokus utama penelitian ini adalah perekayasaan artefak perangkat lunak "
        "(software artifact) berupa modul memori episodik ringan (LSH-Memory-Wrapper) untuk memecahkan masalah komputasi "
        "nyata pada model bahasa lokal di perangkat edge [21]. DSRM menyediakan panduan metodologis berulang (iteratif) "
        "yang menjamin keterkaitan logis antara identifikasi kendala komputasi fisik, perancangan struktur data biner, "
        "implementasi algoritma, serta validasi empiris berbasis pembuktian ilmiah yang dapat direproduksi."
    )

    # 3.2 Alat dan Bahan
    add_heading_2("3.2 Alat dan Bahan")
    add_p(
        "Pelaksanaan eksperimen dan pengembangan artefak dalam penelitian ini menggunakan konfigurasi perangkat keras, "
        "perangkat lunak pendukung, pustaka pembanding (baseline), serta dataset pengujian yang dirancang untuk merepresentasikan "
        "lingkungan komputasi personal terbatas."
    )
    add_p(
        "1. Perangkat Keras (Hardware):\n"
        "Eksperimen dijalankan pada komputer jinjing (laptop) berspesifikasi komersial standar dengan prosesor AMD Ryzen 5 PRO 4650U. "
        "Prosesor ini memiliki 6 inti fisik (cores) dan 12 utas logika (threads) berbasis arsitektur x86-64 Zen 2, dengan frekuensi kerja dasar "
        "2,1 GHz dan boost clock hingga 4,0 GHz, serta dilengkapi cache L3 sebesar 8 MB. Sistem didukung oleh memori utama (RAM) berkapasitas "
        "16 GB DDR4 dual-channel (kecepatan 3.200 MT/s, bandwidth memori teoritis 45–51 GB/s) dan media penyimpanan solid-state drive (SSD) "
        "NVMe PCIe Gen3 x4 berkapasitas 512 GB dengan kecepatan baca sekuensial ~2.100 MB/s. Seluruh eksperimen dijalankan murni pada CPU "
        "tanpa bantuan akselerator grafis (GPU terdedikasi) guna menguji ketahanan modul di bawah batasan perangkat keras minimal."
    )
    add_p(
        "2. Perangkat Lunak (Software):\n"
        "Lingkungan pengembangan berjalan di atas sistem operasi 64-bit Windows 11 Pro (build 26200). "
        "Kode program ditulis menggunakan bahasa pemrograman Python versi 3.10+ (64-bit). Komputasi numerik dan manipulasi bitcode "
        "menggunakan pustaka NumPy versi 2.0+ [22] yang mendukung operasi vektorisasi SIMD native, seperti np.bitwise_count (instruksi x86 POPCNT), "
        "np.packbits, serta np.argpartition. Operasi dekomposisi nilai eigen dan Singular Value Decomposition (SVD) untuk fitting ITQ "
        "menggunakan SciPy [23]. Ekstraksi representasi semantik teks menggunakan pustaka sentence-transformers [17] dengan model dasar "
        "all-MiniLM-L6-v2 (dimensi d=384) [16] dan pembanding paraphrase-multilingual-MiniLM-L12-v2 [17]. Eksekusi model bahasa lokal terkuantisasi "
        "menggunakan Llama.cpp versi rilis 2024 [24] melalui antarmuka Python (llama-cpp-python) dengan model Phi-3 Mini 4K Instruct [4] "
        "dan Llama-3 8B Instruct [5] terkuantisasi 4-bit (format GGUF Q4_K_M). Pemantauan konsumsi memori fisik sistem (Resident Set Size/Working Set) "
        "menggunakan pustaka psutil."
    )
    add_p(
        "3. Pustaka Pembanding (Baseline):\n"
        "Untuk mengevaluasi keunggulan komparatif LSH-Memory-Wrapper, digunakan lima sistem pembanding yang dieksekusi pada lingkungan pengujian yang identik:\n"
        "a. Exact Cosine Flat (NumPy): Pencarian tetangga terdekat brute-force menggunakan operasi perkalian matriks dot-product penuh tanpa indeks, "
        "berfungsi sebagai batas atas teoritis (upper bound) akurasi temu balik (Recall 100%).\n"
        "b. FAISS (Facebook AI Similarity Search) [9]: Menguji varian indeks IndexFlatIP (pencarian eksak teroptimasi C++/AVX2), IndexIVFFlat "
        "(inverter file clustering), dan IndexLSH (Locality-Sensitive Hashing proyeksi acak standar).\n"
        "c. hnswlib [8]: Implementasi referensi C++ untuk algoritma graf Hierarchical Navigable Small World, mewakili metode graf berakurasi tinggi.\n"
        "d. Annoy (Approximate Nearest Neighbors Oh Yeah) [25]: Pustaka ANN berbasis pohon hiper-bidang acak (random projection trees) yang dikembangkan oleh Spotify.\n"
        "e. sqlite-vec [26]: Ekstensi pencarian vektor in-process untuk basis data relasional SQLite tanpa peladen eksternal."
    )
    add_p(
        "4. Data Penelitian:\n"
        "Penelitian ini menggunakan dua kelompok dataset teks percakapan:\n"
        "a. Korpus Percakapan Sintetis Bilingual ID-EN: Dataset multi-sesi yang dikonstruksi secara programatik untuk merepresentasikan pola percakapan riil "
        "di Indonesia dengan fenomena alih kode (code-switching). Korpus memiliki komposisi terstratifikasi: 40% Bahasa Indonesia baku dan kolokial, "
        "30% Bahasa Inggris, serta 30% kalimat campuran dwibahasa dalam satu tuturan. Dataset diskalakan dari 10.000 hingga 1.000.000 entri interaksi. "
        "Ke dalam riwayat percakapan tersebut disisipkan fakta-fakta spesifik pengguna (temporal needles) pada jarak giliran waktu (turn distance) tertentu "
        "guna menguji kemampuan retensi memori episodik.\n"
        "b. Korpus Percakapan Publik: Subset data tanya-jawab percakapan terbuka multi-turn (seperti CoQA atau korpus dialog bilingual) yang diadaptasi "
        "untuk validasi silang (cross-validation) agar evaluasi semantik tidak bias terhadap data sintetis."
    )

    # 3.3 Tahapan Penelitian
    add_heading_2("3.3 Tahapan Penelitian")
    add_p(
        "Pelaksanaan penelitian mengadopsi enam tahapan terstruktur dari kerangka kerja DSRM Peffers dkk. [21]. "
        "Setiap tahapan memiliki sasaran keluaran yang dipetakan secara eksplisit ke dalam struktur laporan tesis ini:"
    )
    add_p(
        "1. Identifikasi Masalah dan Motivasi (Problem Identification and Motivation):\n"
        "Menganalisis fenomena amnesia episodik pada model bahasa lokal kecil (Phi-3 dan Llama-3 8B terkuantisasi) akibat sempitnya jendela konteks "
        "pada CPU laptop, batasan fisik memori KV-Cache (M_KV) [7], fenomena Lost in the Middle [6], serta tingginya overhead RAM basis data vektor konvensional "
        "yang melebihi budget operasional 2 GB. Tahap ini telah diuraikan secara mendalam pada Bab 1 (Sub-bab 1.1 dan 1.2)."
    )
    add_p(
        "2. Definisi Tujuan Solusi (Define the Objectives for a Solution):\n"
        "Menetapkan sasaran kuantitatif artefak yang dapat diuji: (a) waktu encode dan indeks < 2,0 detik untuk 50.000 vektor; "
        "(b) Recall@10 terhadap pencarian eksak >= 85%; (c) konsumsi working set RAM puncak < 2 GB (target < 500 MB); "
        "(d) jejak memori indeks biner aktif < 16 MB untuk 1 juta vektor; dan (e) eliminasi total terhadap dependensi peladen database eksternal. "
        "Tahap ini telah dipaparkan pada Bab 1 (Sub-bab 1.3)."
    )
    add_p(
        "3. Desain dan Pengembangan Artefak (Design and Development):\n"
        "Merancang arsitektur modul Python terpadu lsh-agent-memory yang terdiri atas modul Hasher (centering, PCA, dan ITQ 64/128-bit) [18], "
        "modul HammingBackend berbasis instruksi bitwise SIMD CPU np.bitwise_count [22], modul MemmapStore berbasis file biner terpetakan ke disk [22], "
        "modul EpisodicScorer dengan fungsi peluruhan waktu paruh fisika (half-life decay), serta adapter antarmuka Llama.cpp [24] dan LangChain VectorStore. "
        "Rincian arsitektur, diagram kelas, dan formulasi matematis komponen didetailkan secara komprehensif pada Bab 4 (Sub-bab 4.2)."
    )
    add_p(
        "4. Demonstrasi Artefak (Demonstration):\n"
        "Mengintegrasikan modul purwarupa ke dalam skenario agen percakapan personal lokal. Modul diuji saat menerima instruksi pengguna, "
        "mengekstrak kueri, menyaring memori relevan dari riwayat ribuan percakapan lampau di disk, dan menyuntikkan blok memori terformat "
        "ke dalam prompt inferensi Llama.cpp di bawah anggaran token terbatas. Demonstrasi alur kerja ini dilaporkan pada Bab 4 (Sub-bab 4.3)."
    )
    add_p(
        "5. Evaluasi Kinerja (Evaluation):\n"
        "Melakukan pengujian formal terhadap purwarupa mencakup validasi fungsional, pengujian kinerja sumber daya dan kecepatan, "
        "pengujian akurasi temu balik terhadap ground-truth, ablasi parameter komponen, pengujian varian bahasa, serta pengujian downstream needle QA. "
        "Hasil pengujian empiris dan analisis komparatif disajikan pada Bab 4 (Sub-bab 4.4 dan 4.5)."
    )
    add_p(
        "6. Komunikasi Hasil (Communication):\n"
        "Mendokumentasikan hasil temuan teoritis dan praktis ke dalam naskah laporan tesis, menyusun artikel ilmiah untuk diseminasi pada prosiding konferensi "
        "atau jurnal bereputasi, serta menerbitkan kode sumber modul lsh-agent-memory beserta dokumentasi dan script benchmark secara terbuka (open-source) "
        "di repositori GitHub agar dapat dimanfaatkan oleh komunitas pengembang agen AI."
    )

    # 3.4 Metode Pengujian dan Evaluasi
    add_heading_2("3.4 Metode Pengujian dan Evaluasi")
    add_p(
        "Untuk memastikan keabsahan dan keterandalan artefak yang dikembangkan, dirancang enam protokol pengujian terukur "
        "yang mencakup dimensi teknis sistem maupun efikasi kognitif hilir (downstream)."
    )

    add_heading_3("3.4.1 Uji Fungsionalitas")
    add_p(
        "Uji fungsionalitas bertujuan memverifikasi bahwa seluruh komponen modul bekerja secara deterministik sesuai spesifikasi "
        "tanpa mengalami kegagalan eksekusi (runtime error) atau kebocoran memori (memory leak). Pengujian mencakup lima skenario utama:\n"
        "1. Cold-start handling: Memastikan kueri sebelum fitting ITQ (< 5.000 entri) beralih mulus ke mekanisme brute-force fallback.\n"
        "2. Transisi status indeks: Memastikan peralihan otomatis dari status uninitialized menuju fitted dan indexed berjalan tanpa galat.\n"
        "3. Persistensi status: Memverifikasi penyimpanan (save) dan pemuatan ulang (load) matriks proyeksi W, vektor rata-rata mu, "
        "serta array bitcode dari disk menghasilkan keselarasan bitcode 100% identik.\n"
        "4. Integritas penyimpanan disk: Memastikan penambahan data sekuensial pada MemmapStore merefleksikan catatan teks JSONL dan koordinat biner secara konsisten.\n"
        "5. Adapter prompt injection: Memverifikasi pemotongan memori yang diinjeksikan tidak pernah melampaui batas max_context_tokens yang dialokasikan."
    )

    add_heading_3("3.4.2 Uji Kinerja (Performance Benchmark)")
    add_p(
        "Uji kinerja mengukur efisiensi komputasi dan konsumsi sumber daya fisik pada prosesor AMD Ryzen 5 4650U (12 thread). "
        "Parameter pengujian diskalakan secara bertahap pada ukuran basis data N = 10.000, 50.000, 200.000, hingga 1.000.000 vektor (dimensi d=384). "
        "Metrik kinerja yang diukur meliputi:\n"
        "1. Waktu Fit (Offline): Durasi penghitungan vektor rata-rata mu, dekomposisi nilai eigen kovarians PCA, dan iterasi optimasi rotasi ITQ (50 iterasi) "
        "pada sampel terstratifikasi 5.000 vektor.\n"
        "2. Waktu Encode dan Build Indeks (Online): Durasi transformasi matriks proyeksi data-dependent dan pemadatan bit (packbits) ke format uint64 "
        "untuk seluruh N entri data. Target keberhasilan adalah durasi < 2,0 detik untuk N = 50.000 vektor.\n"
        "3. Latensi Kueri (p50 dan p95): Waktu respons dalam milidetik (ms) untuk memproses 1 kueri dari tahap pengkodean biner, pemindaian Hamming multi-thread, "
        "pengambilan vektor mentah dari disk, hingga pemeringkatan ulang kosinus. Pengukuran diambil dari distribusi 100 kueri uji acak.\n"
        "4. Konsumsi RAM Puncak (Peak Working Set): Penggunaan memori fisik aktual proses Python yang dipantau melalui psutil.Process().memory_info().rss "
        "untuk memastikan beban sistem tidak melampaui batas anggaran aman 2 GB (target < 500 MB).\n"
        "5. Jejak Penyimpanan (Storage Footprint): Ukuran file indeks aktif di RAM (bitcode uint64) dan ukuran berkas vektor mentah di disk (Float32 vs Float16)."
    )

    add_heading_3("3.4.3 Uji Akurasi Temu Balik")
    add_p(
        "Uji akurasi mengevaluasi kedekatan hasil temu balik sistem LSH-Memory-Wrapper terhadap hasil temu balik eksak (ground-truth) "
        "yang dihasilkan oleh pencarian linier dot-product penuh (Exact Cosine). Metrik akurasi dihitung menggunakan formulasi standar temu balik informasi [15]:\n"
        "1. Recall@k:\n"
        "Mengukur proporsi tetangga terdekat relevan sejati (G_k) yang berhasil dijaring ke dalam k kandidat teratas sistem (R_k):"
    )
    add_formula("Recall@k = |R_k ∩ G_k| / k")
    add_p(
        "Pengujian difokuskan pada Recall@5 dan Recall@10, dengan target keberhasilan Recall@10 >= 85% pada basis data 50.000 entri.\n"
        "2. Mean Reciprocal Rank (MRR):\n"
        "Mengevaluasi posisi peringkat pertama di mana dokumen relevan target ditemukan di antara kueri uji Q:"
    )
    add_formula("MRR = (1 / |Q|) ∑_{i=1}^{|Q|} (1 / rank_i)")

    add_heading_3("3.4.4 Uji Ablasi (Ablation Study)")
    add_p(
        "Uji ablasi dirancang secara sistematis untuk mengisolasi dan membuktikan kontribusi kuantitatif dari setiap keputusan desain teknis "
        "terhadap peningkatan akurasi Recall@10 dan efisiensi latensi. Komparasi dilakukan bertahap meliputi:\n"
        "1. Pengaruh Transformasi Hashing: Menguji (a) Random Hyperplane LSH murni (SimHash isotropik) [20], (b) Centered Random LSH (X - mu), "
        "dan (c) Data-Dependent PCA + Iterative Quantization (ITQ) [18].\n"
        "2. Pengaruh Panjang Bitcode: Membandingkan performa kode biner 64-bit (8 bita/vektor) versus 128-bit (16 bita/vektor).\n"
        "3. Pengaruh Ukuran Candidate Pool (N_pool): Mengukur sensitivitas kurva saturasi recall pada variasi ukuran pool N_pool ∈ {50, 100, 250, 500, 1000, 2000}.\n"
        "4. Pengaruh Tahap Re-ranking: Membandingkan akurasi temu balik berbasis jarak Hamming biner murni versus integrasi tahap pemeringkatan ulang exact cosine.\n"
        "5. Pengaruh Presisi Vektor Mentah di Disk: Menguji trade-off akurasi antara format penyimpanan Float32 (1,54 GB per 1M entri) dan Float16 (0,77 GB per 1M entri)."
    )

    add_heading_3("3.4.5 Uji Varian Data dan Representasi Teks")
    add_p(
        "Untuk memastikan ketahanan arsitektur terhadap heterogenitas bahasa pada pengguna lokal di Indonesia, dilakukan pengujian terstratifikasi "
        "terhadap variasi komposisi linguistik dan model embedding:\n"
        "1. Stratifikasi Segmen Bahasa: Mengukur Recall@10 pada tiga kelompok uji terpisah: (a) 100% Bahasa Indonesia baku dan gaul, "
        "(b) 100% Bahasa Inggris teknis, serta (c) kalimat alih kode (code-switching) campuran dwibahasa.\n"
        "2. Evaluasi Model Sentence Embedding: Membandingkan karakteristik ruang representasi antara model all-MiniLM-L6-v2 (dilatih dominan teks Inggris) [16] "
        "dan paraphrase-multilingual-MiniLM-L12-v2 (model multilingual yang mencakup Bahasa Indonesia) [17] guna menganalisis dampak mitigasi anisotropi "
        "oleh modul ITQ pada model yang berbeda."
    )

    add_heading_3("3.4.6 Uji Downstream Needle-in-a-Haystack")
    add_p(
        "Uji hilir (downstream evaluation) bertujuan membuktikan bahwa temu balik memori yang diinjeksikan secara nyata memulihkan fakta lampau "
        "pada inferensi model bahasa lokal di bawah jendela konteks terbatas [6]. Skenario pengujian mengadopsi protokol Needle-in-a-Haystack:\n"
        "1. Konstruksi Skenario: Riwayat percakapan panjang (hingga 4.000 giliran dialog) dibuat dengan menyisipkan fakta target (needle) berupa informasi sensitif "
        "(misalnya: nomor brankas, preferensi personal, instruksi rahasia) pada posisi temporal lampau tertentu (turn distance Δt ∈ {10, 50, 200, 1000, 4000}).\n"
        "2. Kondisi Komparasi: Evaluasi akurasi jawaban model bahasa lokal (Llama-3 8B Instruct Q4_K_M) diuji di bawah empat kondisi eksperimental:\n"
        "   a. Tanpa Memori (No Memory): Model hanya menerima instruksi query terakhir tanpa riwayat percakapan lampau.\n"
        "   b. Sliding Window Truncation: Konfigurasi standar LLM lokal yang membatasi konteks hanya pada 10 giliran interaksi terakhir.\n"
        "   c. Exact Cosine Memory: Injeksi konteks berbasis temu balik exact cosine brute-force (batas atas teoritis).\n"
        "   d. LSH-Memory-Wrapper: Injeksi konteks menggunakan purwarupa modul yang dikembangkan, diuji pada konfigurasi murni semantik (α = 1,0) "
        "dan konfigurasi penskoran gabungan aktualitas (α = 0,7 dengan half-life h = 200 giliran).\n"
        "3. Kriteria Evaluasi: Akurasi penarikan fakta diukur menggunakan pencocokan token eksak (exact match token) dan evaluasi konsistensi semantik jawaban."
    )

    # 3.5 Teknik Analisis Data
    add_heading_2("3.5 Teknik Analisis Data")
    add_p(
        "Data empiris yang diperoleh dari seluruh skenario pengujian diolah dan dianalisis menggunakan metode statistik terstandarisasi "
        "guna menjamin validitas dan signifikansi kesimpulan ilmiah yang ditarik:"
    )
    add_p(
        "1. Statistik Deskriptif dan Pengendalian Variansi:\n"
        "Setiap parameter eksperimen (pengukuran latensi, waktu pembentukan indeks, dan recall) dijalankan secara berulang sebanyak 5 hingga 10 kali "
        "eksekusi independen dengan seed generator bilangan acak yang berbeda. Nilai tengah dilaporkan dalam bentuk rata-rata hitung (mean) "
        "yang disertai simpangan baku (standard deviation). Untuk parameter latensi, nilai persentil p50 (median) dan p95 disajikan untuk mencerminkan "
        "karakteristik latensi tipikal dan perilaku beban puncak (tail latency)."
    )
    add_p(
        "2. Uji Signifikansi Statistik:\n"
        "Untuk memvalidasi apakah peningkatan akurasi temu balik (Recall@10) yang dihasilkan oleh LSH data-dependent ITQ terhadap Random LSH murni "
        "bersifat signifikan secara statistik, dilakukan uji komparasi inferensial. Uji normalitas data diuji terlebih dahulu menggunakan uji Shapiro-Wilk. "
        "Apabila data terdistribusi normal, digunakan uji parametrik Paired Samples t-Test; jika tidak normal, digunakan uji non-parametrik Wilcoxon Signed-Rank Test. "
        "Taraf signifikansi yang ditetapkan adalah α = 0,05, di mana hipotesis keunggulan diterima jika nilai p < 0,05."
    )
    add_p(
        "3. Analisis Kompromi Kinerja (Trade-off Analysis):\n"
        "Hubungan tarik-menarik antara akurasi temu balik, latensi pemrosesan, dan konsumsi memori divisualisasikan melalui kurva analitis dua dimensi:\n"
        "a. Kurva Recall@10 versus Ukuran Candidate Pool (N_pool): Menentukan titik jenuh (knee point) ukuran pool di mana penambahan kandidat "
        "tidak lagi memberikan peningkatan recall yang signifikan namun melipatgandakan latensi re-ranking.\n"
        "b. Kurva Recall@10 versus Latensi Kueri (p50/p95): Memetakan kurva Pareto frontier efisiensi sistem terhadap pustaka baseline (FAISS, hnswlib, Annoy).\n"
        "c. Analisis Selisih Presisi Disk (Float16 vs Float32): Diterapkan aturan keputusan rilis kuantitatif: jika penurunan nilai Recall@10 pada penyimpanan "
        "vektor mentah Float16 lebih kecil dari 0,2 poin persentase (ΔRecall@10 < 0,2%) di seluruh skala pengujian, maka format Float16 ditetapkan sebagai konfigurasi rilis "
        "guna menghemat 50% kapasitas penyimpanan sekunder (disk storage) dan meminimalkan cache thrashing pada arsitektur CPU."
    )

    # Temporary Bibliography for Chapter 3
    add_heading_2("Daftar Pustaka Sementara Bab 3")

    references = [
        "[21] K. Peffers, T. Tuunanen, M. A. Rothenberger, and S. Chatterjee, \"A design science research methodology for information systems research,\" J. Manage. Inf. Syst., vol. 24, no. 3, pp. 45–77, 2007.",
        "[22] C. R. Harris et al., \"Array programming with NumPy,\" Nature, vol. 585, no. 7825, pp. 357–362, 2020.",
        "[23] P. Virtanen et al., \"SciPy 1.0: fundamental algorithms for scientific computing in Python,\" Nature Methods, vol. 17, no. 3, pp. 261–272, 2020.",
        "[24] G. Gerganov, \"llama.cpp: Port of Facebook's LLaMA model in C/C++,\" GitHub repository, 2023. [Online]. Available: https://github.com/ggerganov/llama.cpp",
        "[25] E. Bernhardsson, \"Annoy: Approximate Nearest Neighbors Oh Yeah in C++/Python,\" GitHub repository, 2020. [Online]. Available: https://github.com/spotify/annoy",
        "[26] A. Garcia, \"sqlite-vec: A vector search SQLite extension that runs anywhere,\" GitHub repository, 2024. [Online]. Available: https://github.com/asg017/sqlite-vec"
    ]

    for ref in references:
        p_ref = doc.add_paragraph()
        p_ref.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p_ref.paragraph_format.space_before = Pt(2)
        p_ref.paragraph_format.space_after = Pt(4)
        p_ref.paragraph_format.line_spacing = 1.15
        r_ref = p_ref.add_run(ref)
        set_font(r_ref, size=10)

    docx_path = 'paper/bab3.docx'
    doc.save(docx_path)
    print(f"File DOCX berhasil disimpan di: {docx_path}")

    # Also build clean txt version
    txt_content = []
    for p in doc.paragraphs:
        txt_content.append(p.text)

    txt_path = 'paper/bab3.txt'
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("\n\n".join([line for line in txt_content if line.strip()]))
    print(f"File TXT berhasil disimpan di: {txt_path}")

if __name__ == '__main__':
    create_bab3_documents()
