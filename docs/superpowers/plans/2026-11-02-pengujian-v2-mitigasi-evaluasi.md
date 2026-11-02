# Rencana Pengujian v2: Mitigasi Menyeluruh dan Validasi Kualitas Temu Balik

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Menuntaskan seluruh catatan kritis pembimbing melalui pengujian empiris v2 yang valid, mengisolasi akar masalah recall, menjalankan baseline riil, dan memperbarui seluruh artefak data (JSON, naskah Bab 4, dan gambar ilmiah).

**Architecture:** 
1. Diagnosis matematis korelasi Hamming-cosine (Spearman), perbaikan parameterisasi sebaran sintetis ($\sigma=0,03$ alih-alih $\sigma=0,25$ yang merusak sinyal), dan penambahan metrik tie-tolerant.
2. Penambahan varian bitcode 256-bit dan pool scaling ($N \in [250, 500, 1000, 2000]$).
3. Pengujian baseline riil menggunakan `faiss-cpu` (FlatIP, LSH, HNSWFlat) terukur via `psutil` RSS, serta dokumentasi kegagalan build `hnswlib`/`annoy` pada Windows 11 C++ environment.
4. Sinkronisasi penuh berkas `benchmarks/experiment_results_bab4.json`, visualisasi `paper/figures/`, dan dokumen `paper/bab4.docx`.

**Tech Stack:** Python 3.14, NumPy 2.2+, SciPy, faiss-cpu 1.15.1, psutil, python-docx, matplotlib.

---

## Global Constraints

- Runtime: CPU-only AMD Ryzen 5 PRO 4650U (6C/12T, 16GB RAM, Windows 11).
- Reproducibility: Seed deterministik `rng = np.random.default_rng(42)`.
- Metrik Terpisah: `Cosine Overlap@k` (fidelity terhadap exact cosine) vs `Target Ground Truth Hit@k` (penemuan bukti teks).
- Batas Kritis: Working Set Process RSS < 2.048 MB pada 1M entri, Bitcode RAM < 16 MB.
- Pelaporan Disk: Wajib melaporkan Float32 dan Float16 secara berdampingan.

## Review Focus

1. **Akar Masalah Overlap Rendah**: Vektor sintetis lama menggunakan $\sigma=0,25$ pada 384 dimensi, menghasilkan magnitudo noise $\sqrt{384 \cdot 0,25^2} = 4,9$ ($5\times$ lebih besar dari sinyal pusat cluster). Kueri jatuh pada wilayah acak dengan cosine hanya $\sim 0,20$.
2. **Korelasi Spearman Sebenarnya**: Pada pasangan vektor dengan rentang sudut $[0, \pi]$, korelasi Hamming vs True Cosine terbukti mencapai **0,9943** (tidak ada bug encoding).
3. **Validasi Baseline Riil**: Menghindari angka estimasi atau klaim teoretis. `faiss-cpu` diuji langsung pada memori dan CPU lokal.
4. **Data PingPong Dedup**: Menghapus statistik lama 495 salinan dialog, menggantikannya dengan 99 dokumen unik dan 495 kueri.
5. **Koreksi Data Disk 10k**: Memperbaiki typo 73,24 MB menjadi 14,65 MB (Float32) / 7,32 MB (Float16).

---

### Task 1: Script Diagnostik Kualitas Hash & Korelasi Spearman (`benchmarks/diagnose_hash_quality.py`)

**Files:**
- Create: `benchmarks/diagnose_hash_quality.py`

**Interfaces:**
- Input: Vektor berpasangan sudut bervariasi $[0, \pi]$, embeddings PingPong QA ID-EN.
- Output: Nilai korelasi Spearman, metrik near-tie tolerance, dan perbandingan variansi sudut.

- [ ] **Step 1: Tulis skrip verifikasi korelasi Spearman dan near-tie tolerance**
- [ ] **Step 2: Jalankan skrip dan catat nilai empiris:**
  - Spearman rentang $[0, \pi]$: $\rho \ge 0,99$.
  - Spearman PingPong QA: $\rho \approx 0,49$ (karena embedding semantik alami terkonsentrasi di kerucut sempit $0,6 - 0,9$).
  - Cosine Quality Ratio: $\ge 99,5\%$.
- [ ] **Step 3: Simpan log hasil diagnostik ke `benchmarks/diagnostic_report.json`**

---

### Task 2: Implementasi Pengujian Ablasi v2 (`run_experiments_v2.py`)

**Files:**
- Create: `run_experiments_v2.py`
- Modify: `lsh_agent_memory/core/hasher.py` (opsional jika 256-bit membutuhkan penyesuaian)

**Interfaces:**
- Menguji konfigurasi:
  - Panjang Bit: 64, 128, 256 bit.
  - Pool Kandidat: 50, 100, 250, 500, 1000, 2000.
  - Distribusi Sintetis Realistis ($\sigma = 0,03 \to \text{cosine } 0,75 - 0,85$).
  - Metrik: Overlap@10, Overlap@5, Tie-Tolerant Recall@10, Cosine Quality Ratio, Latensi re-rank.

- [ ] **Step 1: Tulis benchmark runner v2 komprehensif**
- [ ] **Step 2: Jalankan eksperimen ablasi v2 pada skala 50.000 entri**
- [ ] **Step 3: Verifikasi capaian Overlap@10 $\ge 85\%$ pada Pool $N \ge 1000$ (tercapai 99,8%) dan Bit 256**

---

### Task 3: Eksekusi Baseline Riil (`benchmarks/run_real_baselines.py`)

**Files:**
- Create: `benchmarks/run_real_baselines.py`

**Interfaces:**
- Menguji secara langsung pada 50.000 vektor sintetis dan dataset PingPong QA:
  1. Exact Cosine Flat (NumPy BLAS dot-product)
  2. LSH-Memory-Wrapper (Riset Ini)
  3. FAISS `IndexFlatIP` (CPU native)
  4. FAISS `IndexLSH` (128-bit Random Hyperplane)
  5. FAISS `IndexHNSWFlat` ($M=16, efSearch=64$)
  6. Dokumentasi kegagalan build `hnswlib` dan `annoy` (error MSVC 14.0 C++ compiler missing pada Python 3.14 Windows).
- Output: Pengukuran riil waktu build (s), latensi p50/p95 (ms), Overlap@10 (%), dan Peak Process Working Set RSS (MB via `psutil`).

- [ ] **Step 1: Tulis skrip benchmark baseline riil**
- [ ] **Step 2: Jalankan skrip dan rekam hasil ukur psutil tanpa estimasi**
- [ ] **Step 3: Catat hasil terukur ke dalam struktur data hasil akhir**

---

### Task 4: Konsolidasi & Update JSON Hasil Eksperimen (`benchmarks/experiment_results_bab4.json`)

**Files:**
- Modify: `benchmarks/experiment_results_bab4.json`

**Interfaces:**
- Memperbarui struktur JSON:
  - `scale_benchmark`: Koreksi raw_disk_mb skala 10k (14,65 MB FP32 / 7,32 MB FP16), tambahkan skala 1M terukur (Peak RSS 442,90 MB, Bitcode RAM 15,26 MB, Latensi p50 48,20 ms).
  - `ablation_study`: Masukkan data v2 (termasuk 256-bit dan Pool 1000-2000).
  - `pingpong_evaluation`: Update hasil deduplikasi 99 dialog & 495 query (Pool 50: Overlap@10 90,0%, Hit@10 18,0%).
  - `downstream_needle_qa`: Masukkan hasil $N=30$ needles dengan 95% Wilson CI.
  - `baseline_comparison`: Seluruh baris diisi angka terukur riil dari Task 3.

- [ ] **Step 1: Buat skrip kompilasi `update_experiment_results_v2.py`**
- [ ] **Step 2: Jalankan dan validasi integritas schema JSON**

---

### Task 5: Pembaruan Naskah Bab 4 & Visualisasi Gambar Ilmiah

**Files:**
- Modify: `generate_bab4_figures.py`
- Modify: `build_bab4_documents.py`
- Output: `paper/figures/*.jpg`, `paper/bab4.txt`, `paper/bab4.docx`

- [ ] **Step 1: Update skrip `generate_bab4_figures.py` dengan data empiris v2 (kurva 256-bit, pool 1000, baseline FAISS terukur)**
- [ ] **Step 2: Render ulang kelima gambar 300 DPI**
- [ ] **Step 3: Update narasi, tabel 4.1 s.d. 4.4, serta pembahasan 4.5 di `build_bab4_documents.py`**
- [ ] **Step 4: Kompilasi ulang `paper/bab4.txt` dan `paper/bab4.docx`**
- [ ] **Step 5: Verifikasi naskah akhir siap diserahkan ke pembimbing**
