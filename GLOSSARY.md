# GLOSSARY

## Bitcode
Array biner ringkas bertipe `uint8` atau `uint64` hasil binarisasi tanda $(X - \mu) W > 0$. Untuk kode 64-bit atau 128-bit, bitcode disimpan sebagai 1 atau 2 word `uint64` per entri.

## Hasher
Modul proyeksi data-dependent yang mempelajari vektor rata-rata $\mu$ dan matriks proyeksi ortogonal $W$ melalui reduksi dimensi Principal Component Analysis (PCA) dan iterasi kuantisasi rotasi ortogonal Iterative Quantization (ITQ).

## HammingBackend
Antarmuka eksekusi pencarian jarak Hamming antara query bitcode dan basis data bitcode. Implementasi default menggunakan operasi XOR bitwise dan `np.bitwise_count` terparalelisasi multi-thread dengan seleksi $O(N)$ via `np.argpartition`.

## EpisodicStore
Komponen penyimpanan persisten pada disk berbasis memory-mapped file (`np.memmap`) yang memuat vektor mentah (`float32` atau `float16`) serta atribut metadata (ID, timestamp, pointer teks) tanpa membebani working set RAM utama.

## Candidate Pool
Himpunan berukuran $N$ entri teratas hasil penyaringan jarak Hamming terkecil yang diteruskan ke tahap re-ranking exact cosine.

## EpisodicScorer
Fungsi penskoran akhir yang menggabungkan kesamaan semantik exact cosine dengan peluruhan eksponensial berbasis waktu paruh (half-life):
$$\text{score} = \alpha \cdot \cos(q, m) + (1 - \alpha) \cdot 2^{-\frac{\Delta t}{h}}$$

## Needle Retrieval
Metodologi pengujian downstream untuk mengukur kemampuan sistem memori dalam menemukan kembali fakta spesifik (needle) yang disisipkan pada sesi percakapan lampau dalam rentang konteks multi-turn yang panjang.
