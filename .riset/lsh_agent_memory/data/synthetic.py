"""Synthetic multi-turn Indonesian-English mixed conversational data generator with stratified needle injection."""

import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any


@dataclass
class ConversationTurn:
    turn_id: int
    speaker: str
    text: str
    language: str  # 'id', 'en', or 'mixed'
    is_needle: bool = False
    needle_key: str = ""


# Pre-defined templates for episodic dialogue simulation
ID_TEMPLATES = [
    "Saya sedang mengerjakan laporan keuangan untuk kuartal ketiga.",
    "Bisa tolong jelaskan kembali konsep machine learning yang kemarin?",
    "Kemarin saya sempat diskusi dengan tim sales tentang target bulanan.",
    "Bagaimana cara mengoptimalkan performa database SQLite di server lokal?",
    "Besok pagi ada meeting koordinasi sprint dengan tim engineering.",
    "Tolong buatkan rangkuman dari catatan pertemuan kemarin sore.",
    "Proses deployment backend kemarin sempat tertunda karena issue network.",
    "Apakah kita perlu update dependensi library Python minggu ini?",
    "Analisis log menunjukkan peningkatan traffic sekitar tiga puluh persen.",
    "Koneksi ke endpoint API pembayaran sempat timeout beberapa detik.",
    "Rapat evaluasi arsitektur microservices dijadwalkan ulang minggu depan.",
    "Kita perlu backup database PostgreSQL sebelum menjalankan migrasi schema."
]

EN_TEMPLATES = [
    "I am currently reviewing the pull request for the authentication service.",
    "Can you help me analyze the memory consumption of this background worker?",
    "We need to ensure backward compatibility with the legacy REST API.",
    "The latency benchmark showed significant improvement after caching.",
    "Please verify the configuration settings in the staging environment.",
    "Let's schedule a code review session tomorrow morning at 10 AM.",
    "The CPU utilization spiked during the batch processing job.",
    "Make sure to add unit tests covering the edge cases in tokenizer.",
    "We observed increased disk I/O wait times on the primary worker node.",
    "The response payload size should be minimized for mobile clients.",
    "Network latency between availability zones increased slightly today.",
    "Please check if the background worker thread pool is properly saturated."
]

MIXED_TEMPLATES = [
    "Tolong check PR di GitHub yang baru saya push tadi siang.",
    "Meeting sprint planning kita reschedule ke jam 2 siang ya.",
    "Query database ini masih kena bottleneck di table indexing-nya.",
    "Log error menunjukkan service crash karena out of memory issue.",
    "Bisa tolong refactor function ini supaya lebih clean dan maintainable?",
    "Kita perlu benchmark latency untuk endpoint retrieval memory ini.",
    "Fitur baru ini butuh approval dari team lead sebelum release ke production.",
    "Tolong update documentation di repo supaya onboarding developer baru lebih smooth.",
    "Memory usage worker ini sudah lewat threshold aman 80 persen.",
    "Mohon push commit fix bug autentikasi sebelum build staging sore ini.",
    "Monitoring dashboard menunjukkan spike request setelah campaign blast jalan.",
    "Koneksi cache hit ratio turun drastis setelah restart container Redis."
]

# 30 Stratified Needles: (Fact, High-Overlap Query, Low-Overlap Paraphrased Query, Expected Answer)
STRATIFIED_NEEDLES = [
    # Bracket 1: Sangat Baru (Δt in [10, 50])
    (
        "Kunci rahasia brankas kantor pusat adalah KODELSH2026.",
        "Berapa kunci rahasia brankas kantor pusat?",
        "Bisa tolong share passkey pengaman uang di head office kita?",
        "KODELSH2026"
    ),
    (
        "The client requested all invoice summaries to be sent to billing@perusahaan.co.id.",
        "What email should invoice summaries be sent to?",
        "Klien minta rekap tagihan di-forward ke kontak mana?",
        "billing@perusahaan.co.id"
    ),
    (
        "Preferensi bahasa saya untuk coding adalah Python dengan style guide Google.",
        "Apa preferensi bahasa coding pengguna beserta style guide-nya?",
        "Gaya penulisan dan bahasa pemrograman kesukaan user waktu develop script apa ya?",
        "Python dengan style guide Google"
    ),
    (
        "Project deployment deadline is officially set to 15 November 2026.",
        "When is the project deployment deadline?",
        "Kapan tanggal rilis final aplikasi ke production?",
        "15 November 2026"
    ),
    (
        "Makanan favorit saya saat lembur adalah Nasi Goreng Kambing Kebon Sirih.",
        "Apa makanan favorit pengguna saat lembur?",
        "Menu santapan yang disukai user kalau kerja larut malam apa?",
        "Nasi Goreng Kambing Kebon Sirih"
    ),
    (
        "Nomor darurat server room gedung Cyber 1 adalah 0812-9988-7766.",
        "Berapa nomor darurat server room gedung Cyber 1?",
        "Kontak genting untuk ruang infrastruktur data center di Kuningan berapa?",
        "0812-9988-7766"
    ),

    # Bracket 2: Baru (Δt in [51, 200])
    (
        "Our database replica port is mapped to 5433 instead of default 5432.",
        "Which port is mapped for the database replica?",
        "Jalur koneksi cadangan backend Postgres kita dibuka lewat pintu nomor berapa?",
        "5433"
    ),
    (
        "Alamat IP static server staging adalah 192.168.100.45.",
        "Berapa alamat IP static server staging?",
        "Host address tetap untuk mesin testing pra-produksi berapa?",
        "192.168.100.45"
    ),
    (
        "CEO requested to keep the quarterly budget under 250 million rupiah.",
        "What is the maximum quarterly budget requested by the CEO?",
        "Pimpinan minta alokasi dana operasional 3 bulan dibatasi sampai nominal berapa?",
        "250 million rupiah"
    ),
    (
        "Format tanggal default untuk audit log sistem harus YYYY-MM-DD HH:mm:ss.",
        "Apa format tanggal default untuk audit log sistem?",
        "Standar penulisan waktu rekaman histori aktivitas platform bagaimana?",
        "YYYY-MM-DD HH:mm:ss"
    ),
    (
        "Token autentikasi webhook payment gateway menggunakan secret KEY-SEC-9921.",
        "Apa secret token autentikasi webhook payment gateway?",
        "Kredensial rahasia buat verifikasi callback transaksi pembayaran apa?",
        "KEY-SEC-9921"
    ),
    (
        "The primary caching layer uses Redis cluster with 3 master nodes.",
        "How many master nodes does the primary Redis cluster have?",
        "Berapa jumlah simpul pengendali utama pada penyimpanan memori sementara in-memory kita?",
        "3 master nodes"
    ),

    # Bracket 3: Menengah (Δt in [201, 800])
    (
        "Nama sandi proyek migrasi microservices adalah Project Garuda Blue.",
        "Apa nama sandi proyek migrasi microservices?",
        "Sebutan internal untuk inisiatif pemecahan arsitektur monolitik apa?",
        "Project Garuda Blue"
    ),
    (
        "User prefers all technical documentation in Indonesian language.",
        "What language does the user prefer for technical documentation?",
        "Bahasa pengantar apa yang dipilih user buat berkas panduan teknis?",
        "Indonesian language"
    ),
    (
        "Target throughput sistem baru adalah minimal 5000 request per detik.",
        "Berapa target throughput minimal sistem baru per detik?",
        "Beban kapasitas query traffic terendah yang harus ditangani platform per sekon berapa?",
        "5000 request"
    ),
    (
        "Batas waktu retensi data log user diatur selama 90 hari sesuai kebijakan PDP.",
        "Berapa hari batas waktu retensi data log user?",
        "Berapa lama riwayat jejak digital pengguna boleh disimpan sebelum dihapus demi regulasi privasi?",
        "90 hari"
    ),
    (
        "The backup NAS storage is mounted on directory /mnt/secure_backup/daily.",
        "What is the mount path for the backup NAS storage?",
        "Di folder mana media penyimpanan cadangan berkas dipasang?",
        "/mnt/secure_backup/daily"
    ),
    (
        "Vendor penyedia SSL certificate resmi kita adalah DigiCert EV Pro.",
        "Siapa vendor penyedia SSL certificate resmi perusahaan?",
        "Perusahaan rekanan penerbit sertifikat enkripsi domain HTTPS kita siapa?",
        "DigiCert EV Pro"
    ),

    # Bracket 4: Lama (Δt in [801, 2000])
    (
        "Maximum token budget allocated for context retrieval prompt is 1024 tokens.",
        "What is the maximum token budget allocated for context retrieval?",
        "Berapa kuota panjang teks pengambilan memori yang diizinkan masuk ke prompt?",
        "1024 tokens"
    ),
    (
        "Jadwal maintenance bulanan selalu diadakan hari Minggu pertama jam 02:00 WIB.",
        "Kapan jadwal rutin maintenance bulanan diadakan?",
        "Waktu pemeliharaan rutin server setiap bulan diset kapan?",
        "Hari Minggu pertama jam 02:00 WIB"
    ),
    (
        "Default branch name for core repository is trunk rather than main.",
        "What is the default branch name for the core repository?",
        "Nama cabang utama di repositori git pusat pakai istilah apa?",
        "trunk"
    ),
    (
        "Batas toleransi memory leak pada worker background adalah 50 MB per jam.",
        "Berapa batas toleransi memory leak pada worker background?",
        "Ambang batas kenaikan konsumsi RAM abnormal pada proses latar belakang berapa?",
        "50 MB per jam"
    ),
    (
        "The secondary contact person for infrastructure outage is Ahmad Fauzi.",
        "Who is the secondary contact person for infrastructure outage?",
        "Kalau server tumbang dan PIC utama berhalangan, hubungi siapa?",
        "Ahmad Fauzi"
    ),
    (
        "Kapasitas maksimum heap memory untuk Java service diatur 4096 megabytes.",
        "Berapa kapasitas maksimum heap memory Java service?",
        "Batas atas alokasi RAM JVM backend diset di angka berapa?",
        "4096 megabytes"
    ),

    # Bracket 5: Sangat Lama (Δt in [2001, 5000])
    (
        "Default model inference parameter temperature is locked at 0.2.",
        "What is the locked default temperature parameter for model inference?",
        "Setelan derajat keacakan output generasi teks dipatok di angka berapa?",
        "0.2"
    ),
    (
        "Kode kupon diskon testing internal QA adalah DISKONDEV2026.",
        "Apa kode kupon diskon testing internal QA?",
        "Voucher potongan harga yang dipakai tim penguji aplikasi kodenya apa?",
        "DISKONDEV2026"
    ),
    (
        "The emergency incident room on Slack is channel #war-room-alpha.",
        "Which Slack channel is designated as the emergency incident room?",
        "Kanal obrolan darurat buat koordinasi trouble shooting di platform chat namanya apa?",
        "#war-room-alpha"
    ),
    (
        "Akun monitoring Grafana read-only menggunakan username auditor_external.",
        "Apa username akun monitoring Grafana read-only?",
        "User ID untuk login dasbor pantauan metrik bagi pihak luar apa?",
        "auditor_external"
    ),
    (
        "The minimum supported Python runtime version for this wrapper is 3.10.",
        "What is the minimum supported Python runtime version?",
        "Spesifikasi versi bahasa pemrograman terendah agar modul ini jalan versi berapa?",
        "3.10"
    ),
    (
        "Lokasi fisik rak server colocation berada di Lantai 3 Ruang 304 IDC Duren Tiga.",
        "Di mana lokasi fisik rak server colocation perusahaan?",
        "Di ruangan dan gedung mana lemari mesin server kita ditempatkan?",
        "Lantai 3 Ruang 304 IDC Duren Tiga"
    )
]

# Aliases for backward compatibility
NEEDLE_PAIRS = [(f, qh, ans) for (f, qh, ql, ans) in STRATIFIED_NEEDLES]

AGE_BRACKETS = [
    {"name": "Sangat Baru", "dt_range": (10, 50), "indices": range(0, 6)},
    {"name": "Baru", "dt_range": (51, 200), "indices": range(6, 12)},
    {"name": "Menengah", "dt_range": (201, 800), "indices": range(12, 18)},
    {"name": "Lama", "dt_range": (801, 2000), "indices": range(18, 24)},
    {"name": "Sangat Lama", "dt_range": (2001, 5000), "indices": range(24, 30)},
]


def wilson_score_interval(successes: int, total: int, z: float = 1.95996) -> Tuple[float, float, float]:
    """Calculate Wilson score interval for a binomial proportion.

    Returns (center, lower_bound, upper_bound) in percentage [0, 100].
    """
    if total == 0:
        return 0.0, 0.0, 0.0
    p = successes / total
    denom = 1 + (z**2) / total
    center = (p + (z**2) / (2 * total)) / denom
    margin = (z / denom) * math.sqrt((p * (1 - p) / total) + (z**2) / (4 * total**2))
    low = max(0.0, center - margin) * 100.0
    high = min(1.0, center + margin) * 100.0
    return round(p * 100.0, 2), round(low, 2), round(high, 2)


def generate_stratified_needle_conversations(
    total_turns: int = 5_000,
    seed: int = 42,
) -> Tuple[List[ConversationTurn], List[Dict[str, Any]]]:
    """Generate multi-turn conversation with 30 needles stratified across 5 age brackets.

    Args:
        total_turns: Number of dialogue turns to generate (default 5,000).
        seed: Random seed for positioning.

    Returns:
        Tuple of (turns_list, needle_test_queries).
    """
    rng = random.Random(seed)
    turns: List[ConversationTurn] = []

    # Map target positions based on age brackets
    needle_positions: Dict[int, int] = {}
    used_positions = set()

    for bracket in AGE_BRACKETS:
        low_dt, high_dt = bracket["dt_range"]
        for needle_i in bracket["indices"]:
            # Sample delta_t
            # turn_pos = total_turns - delta_t
            for _ in range(100):
                dt = rng.randint(low_dt, high_dt)
                pos = total_turns - dt
                pos = max(0, min(total_turns - 1, pos))
                if pos not in used_positions:
                    used_positions.add(pos)
                    needle_positions[pos] = needle_i
                    break

    needles_info: List[Dict[str, Any]] = []

    for i in range(total_turns):
        if i in needle_positions:
            needle_idx = needle_positions[i]
            fact, query_high, query_low, answer = STRATIFIED_NEEDLES[needle_idx]
            turn = ConversationTurn(
                turn_id=i,
                speaker="user",
                text=fact,
                language="mixed",
                is_needle=True,
                needle_key=f"needle_{needle_idx}",
            )
            bracket_idx = needle_idx // 6
            needles_info.append({
                "needle_idx": needle_idx,
                "needle_key": f"needle_{needle_idx}",
                "turn_id": i,
                "delta_t": total_turns - i,
                "bracket_name": AGE_BRACKETS[bracket_idx]["name"],
                "bracket_idx": bracket_idx,
                "fact": fact,
                "query_high": query_high,
                "query_low": query_low,
                "expected_answer": answer,
            })
        else:
            lang_choice = rng.choices(["id", "en", "mixed"], weights=[0.4, 0.3, 0.3])[0]
            if lang_choice == "id":
                text = rng.choice(ID_TEMPLATES)
            elif lang_choice == "en":
                text = rng.choice(EN_TEMPLATES)
            else:
                text = rng.choice(MIXED_TEMPLATES)

            speaker = "user" if i % 2 == 0 else "assistant"
            turn = ConversationTurn(
                turn_id=i,
                speaker=speaker,
                text=text,
                language=lang_choice,
            )
        turns.append(turn)

    needles_info.sort(key=lambda x: x["needle_idx"])
    return turns, needles_info
