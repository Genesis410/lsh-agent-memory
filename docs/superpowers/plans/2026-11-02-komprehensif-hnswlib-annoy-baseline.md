# Rencana Komprehensif: Integrasi Native Baseline (hnswlib & Annoy) dan Uji Ulang Bab 4

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Mengompilasi dan menginstal baseline native C++ `hnswlib` dan `annoy` pada Windows 11 menggunakan MinGW-w64 GCC/G++ tanpa ketergantungan MSVC Build Tools besar (zero disk bloat), menjalankan benchmark head-to-head terukur langsung pada 50.000 entri, memperbarui master JSON dan visualisasi Gambar 4.4, serta menyusun revisi naskah Bab 4 ilmiah yang siap sidang.

**Architecture:**
1. Ekstraksi source package `hnswlib` (v0.8.0) dan `annoy` (v1.17.3) dari cache PyPI, kompilasi C++ extensions menggunakan compiler MinGW-w64 (`g++ -O -shared`) via `distutils` `--compiler=mingw32`.
2. Bundling dependensi runtime DLL (`libstdc++-6.dll`, `libgcc_s_seh-1.dll`, `libwinpthread-1.dll`) langsung ke dalam folder paket di Python `site-packages` agar self-contained.
3. Eksekusi pengujian komparatif head-to-head 50.000 entri terukur via `psutil` (Build time, Latensi p50/p95, Overlap@10, Peak Working Set RSS, Index RAM).
4. Pemutakhiran berkas master `benchmarks/experiment_results_bab4.json` dan perenderan ulang 5 gambar publikasi 300 DPI di `benchmarks/generate_bab4_figures.py`.
5. Penyusunan ulang naskah Bab 4 (`paper/bab4.txt` dan `paper/bab4.docx`) dengan narasi akademik objektif sesuai IEEE tanpa klaim berlebihan.

**Tech Stack:** Python 3.14, MinGW-w64 G++ 16.1.0, hnswlib 0.8.0, annoy 1.17.3, faiss-cpu 1.15.1, NumPy 2.2+, psutil, matplotlib, python-docx.

**Spec:** Dokumen Rencana Riset `docs/superpowers/plans/2026-11-02-komprehensif-hnswlib-annoy-baseline.md` dan Catatan Kritis Pembimbing.

---

## Global Constraints

- **Platform & Perangkat**: AMD Ryzen 5 PRO 4650U (6-Core / 12-Thread, 2.1 - 4.0 GHz), 16GB DDR4-3200, Windows 11 Pro 64-bit.
- **Reproducibility**: Parameter acak deterministik `rng = np.random.default_rng(42)`.
- **Zero Heavy Toolchain**: Tidak mengunduh 6-10 GB Visual Studio Build Tools; memanfaatkan MinGW GCC/G++ yang sudah terinstal di sistem.
- **Pengukuran Riil (No Estimates)**: Seluruh metrik latensi, build time, dan memory RSS diukur langsung via runtime `psutil` dan `time.perf_counter`.
- **Preservasi Metrik**: Memisahkan secara ketat antara `Cosine Overlap@k` (fidelity terhadap exact cosine) dan `Target Needle Hit@k`.

## Review Focus

1. **Kompatibilitas Runtime MinGW**: Memastikan paket `hnswlib` dan `annoy` yang dikompilasi dengan MinGW-w64 dapat di-import dan berjalan stabil di Python 3.14 tanpa `ImportError: DLL load failed`.
2. **Kesesuaian Parameter Baseline**:
   - `hnswlib`: Metric Inner Product (`ip`), `M=16`, `ef_construction=100`, `efSearch=64` (ekuivalen dengan FAISS IndexHNSWFlat).
   - `annoy`: Metric Dot Product (`dot`), `n_trees=10` dan `n_trees=50`, `search_k=-1`.
3. **Keadilan Komparasi Memori**: Membedakan antara Index Storage RAM (ukuran struktur data dalam memori) dan Total Process RSS (termasuk buffer runtime Python).
4. **Konsistensi Dataset Pengujian**: Semua baseline (NumPy Exact, FAISS Flat/LSH/HNSW, hnswlib, Annoy, LSH-Wrapper) diuji pada 50.000 vektor dan 100 kueri yang identik dengan kalibrasi $\sigma = 0,032$.
5. **Kualitas Naskah Bab 4**: Mengikuti kaidah /anti-ai-writing, menyajikan kelebihan dan kekurangan secara berimbang (misalnya mengakui hnswlib/HNSW memiliki latensi p50 sub-milidetik tercepat, namun LSH-Wrapper unggul mutlak dalam efisiensi RAM dan waktu inisialisasi).

---

### Task 1: Kompilasi & Instalasi Mandiri `hnswlib` dan `annoy` via MinGW-w64

**Files:**
- Create: `scripts/install_native_baselines.py`
- Target Path: `C:\Users\LENOVO THINKPAD\AppData\Roaming\Python\Python314\site-packages\`

**Interfaces:**
- Consumes: MinGW-w64 toolchain (`C:\ProgramData\mingw64\mingw64\bin`), PyPI tarball `hnswlib` dan `annoy`.
- Produces: Terinstalnya modul `hnswlib` dan `annoy` yang dapat di-import langsung di Python 3.14 (`import hnswlib`, `from annoy import AnnoyIndex`).

- [ ] **Step 1: Tulis skrip otomatisasi build `scripts/install_native_baselines.py`**
  Mengunduh source tarball dari PyPI cache, mengompilasi ekstensi C++ menggunakan flag `--compiler=mingw32`, menyalin DLL runtime (`libstdc++-6.dll`, `libgcc_s_seh-1.dll`, `libwinpthread-1.dll`) ke folder target, dan menyalin paket ke `site-packages`.

- [ ] **Step 2: Jalankan skrip instalasi**
  Run: `python scripts/install_native_baselines.py`
  Expected: Returncode 0, pesan `[SUCCESS] hnswlib and annoy installed successfully`.

- [ ] **Step 3: Verifikasi fungsional unit test kedua library**
  Run: `python -c "import hnswlib, annoy; print('HNSWLIB:', hnswlib.Index('ip', 4)); print('ANNOY:', annoy.AnnoyIndex(4, 'dot'))"`
  Expected: Kedua objek berhasil diinisialisasi tanpa error missing DLL.

---

### Task 2: Eksekusi Benchmark Baseline Komprehensif Head-to-Head

**Files:**
- Modify: `benchmarks/run_real_baselines.py`
- Output: `benchmarks/real_baseline_results.json`

**Interfaces:**
- Consumes: 50.000 vektor database terkalibrasi ($\sigma=0,032$, 384 dimensi), 100 kueri uji.
- Produces: JSON lengkap berisi hasil ukur 8 metode ANN:
  1. Exact Cosine Flat (NumPy BLAS)
  2. FAISS IndexFlatIP (C++ AVX2)
  3. FAISS IndexLSH (128-bit Raw)
  4. FAISS IndexLSH (128-bit + Re-rank Pool 500)
  5. FAISS IndexHNSWFlat (M=16, efSearch=64)
  6. hnswlib (M=16, efSearch=64)
  7. Annoy (n_trees=10 & n_trees=50)
  8. LSH-Memory-Wrapper (ITQ 128-bit & 256-bit, Pool 500 & 1000)

- [ ] **Step 1: Modifikasi `benchmarks/run_real_baselines.py`**
  Menambahkan blok pengujian `hnswlib` dan `annoy` dengan pencatatan waktu build, latensi query p50/p95, peak RSS (`psutil`), RAM indeks, dan Overlap@10.

- [ ] **Step 2: Jalankan pengujian baseline secara menyeluruh**
  Run: `python benchmarks/run_real_baselines.py`
  Expected: Semua 8 metode selesai dieksekusi dan tercatat di `benchmarks/real_baseline_results.json`.

- [ ] **Step 3: Validasi integritas hasil pengujian**
  Memastikan `hnswlib` dan `annoy` memiliki nilai numerik riil (bukan `None`) untuk semua kolom metrik.

---

### Task 3: Konsolidasi Master Experiment Results (JSON)

**Files:**
- Modify: `benchmarks/compile_master_results_v2.py`
- Output: `benchmarks/experiment_results_bab4.json`

**Interfaces:**
- Consumes: `benchmarks/diagnostic_report.json`, `benchmarks/real_baseline_results.json`, `benchmarks/real_text_experiments_results.json`.
- Produces: Master JSON mutakhir yang memuat data diagnostik, skalabilitas, presisi disk, eksperimen teks nyata PingPong, dan perbandingan baseline lengkap.

- [ ] **Step 1: Jalankan kompilasi data master**
  Run: `python benchmarks/compile_master_results_v2.py`
  Expected: Berkas `benchmarks/experiment_results_bab4.json` terisi dengan dataset lengkap.

---

### Task 4: Regenerasi Visualisasi Gambar Bab 4 (300 DPI, Kaidah Dataviz)

**Files:**
- Modify: `benchmarks/generate_bab4_figures.py`
- Outputs:
  - `paper/figures/fig4_1_architecture.png`
  - `paper/figures/fig4_2_scale_latency_ram.png`
  - `paper/figures/fig4_3_ablation_recall_curve.png`
  - `paper/figures/fig4_4_baseline_comparison.png`
  - `paper/figures/fig4_5_downstream_retention.png`

**Interfaces:**
- Consumes: Data riil dari `benchmarks/experiment_results_bab4.json`.
- Produces: Gambar ilmiah beresolusi tinggi 300 DPI dengan palet warna colorblind-safe, garis kisi lembut (recessive grid), dan tipografi standar publikasi.

- [ ] **Step 1: Perbarui data sumber pada Gambar 4.4 (`fig4_4_baseline_comparison.png`)**
  Menyertakan batang `hnswlib` dan `Annoy` berdampingan dengan `FAISS HNSW`, `FAISS LSH`, dan `LSH-Memory-Wrapper`.

- [ ] **Step 2: Eksekusi perenderan seluruh gambar**
  Run: `python benchmarks/generate_bab4_figures.py`
  Expected: Kelima gambar berhasil dirender ulang dengan takarir resmi.

---

### Task 5: Penulisan Revisi Naskah Ilmiah Bab 4 (`paper/bab4.txt` & `paper/bab4.docx`)

**Files:**
- Modify: `paper/bab4.txt`
- Create: `build_bab4_documents.py` (untuk konversi otomatis ke DOCX dengan tabel dan gambar tersemat)
- Output: `paper/bab4.docx`

**Interfaces:**
- Consumes: Angka-angka terverifikasi dari `benchmarks/experiment_results_bab4.json` dan gambar dari `paper/figures/`.
- Produces: Naskah Bab 4 lengkap (Subbab 4.1 s.d. 4.5) setebal ~25-35 halaman format skripsi/tesis akademik.

- [ ] **Step 1: Perbarui naskah teks `paper/bab4.txt`**
  - Subbab 4.1: Gambaran Umum Penelitian & Konteks Solusi.
  - Subbab 4.2: Perancangan Artefak (Prinsip Zero-Dependency, Arsitektur, Formula Matematis ITQ & Hamming).
  - Subbab 4.3: Pengembangan dan Implementasi (Katalog Komponen, Lingkungan Uji, Penanganan Bilingual).
  - Subbab 4.4: Hasil Pengujian Empiris:
    - 4.4.1 Kualitas Hashing & Korelasi Spearman (Global vs Local, Kalibrasi $\sigma$).
    - 4.4.2 Kinerja Skalabilitas & Presisi Disk (10k - 1M entri, Float32 vs Float16).
    - 4.4.3 Pengujian Teks Nyata Percakapan PingPong (Segmen ID/EN/Mixed, Ablasi Unit Memori, Cross-Domain Generalization, Sample Size).
    - 4.4.4 Komparasi Head-to-Head Baseline (FAISS, hnswlib, Annoy, LSH-Wrapper).
    - 4.4.5 Evaluasi Downstream Needle-in-Haystack (30 Needles, Wilson 95% CI).
  - Subbab 4.5: Pembahasan Ilmiah Kritis (Analisis Trade-Off, Kondisi Batas di mana HNSW lebih unggul, Implikasi UU PDP).

- [ ] **Step 2: Buat skrip kompilasi dokumen `build_bab4_documents.py`**
  Menyusun naskah Word berformat standar (Times New Roman 12, Spasi 1.5, Heading terstruktur, Tabel formal IEEE dengan garis horizontal tipis, Gambar tersemat di tengah halaman).

- [ ] **Step 3: Bangun naskah akhir `paper/bab4.docx`**
  Run: `python build_bab4_documents.py`
  Expected: Berkas `paper/bab4.docx` selesai dibuat tanpa error formatting.
