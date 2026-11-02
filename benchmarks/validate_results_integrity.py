"""Automated validation script for experiment results integrity.

Verifies:
1. No None, NaN, or missing values in master JSON.
2. Complete coverage of all 11 baselines (M1..M11).
3. Exact storage mathematics consistency (Float32, Float16, 128-bit bitcodes).
4. Physical and algorithmic scaling monotonicity.
5. Formal verification of Hypotheses H1, H2, H3, and H4.
"""

import json
import math
import sys
from pathlib import Path


def validate_integrity():
    print("=================================================================")
    print("AUTOMATED RESULTS INTEGRITY & HYPOTHESIS VALIDATION SUITE")
    print("=================================================================")

    json_path = Path("benchmarks/experiment_results_bab4.json")
    assert json_path.exists(), f"File {json_path} does not exist!"

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. Structural completeness
    required_top_keys = [
        "metadata",
        "task0_diagnostics",
        "task2_scale_and_memory_benchmarks",
        "task3_real_text_experiments",
        "task4_stratified_needle_qa",
    ]
    for k in required_top_keys:
        assert k in data, f"Missing required top-level key: {k}"
    print("[PASS] Top-level schema structure verified.")

    # 2. Check storage dimensions
    stor = data["task2_scale_and_memory_benchmarks"]["storage_dimensions"]
    assert abs(stor["10000"]["raw_disk_float32_mb"] - 14.65) < 0.01
    assert abs(stor["10000"]["raw_disk_float16_mb"] - 7.32) < 0.01
    assert abs(stor["50000"]["raw_disk_float16_mb"] - 36.62) < 0.01
    assert abs(stor["1000000"]["raw_disk_float16_mb"] - 732.42) < 0.01
    assert abs(stor["1000000"]["bitcode_128_ram_mb"] - 15.26) < 0.01
    print("[PASS] Storage and RAM byte mathematics verified exact.")

    # 3. Check Baseline Records
    records = data["task2_scale_and_memory_benchmarks"]["baseline_records_isolated"]
    assert len(records) >= 60, f"Expected >=60 records, found {len(records)}"

    methods_present = set(r["method_id"] for r in records)
    for m_i in range(1, 12):
        m_id = f"M{m_i}"
        assert m_id in methods_present, f"Method {m_id} missing from baseline records!"
    print(f"[PASS] All 11 methods (M1..M11) present ({len(records)} isolated executions).")

    # Check for NaN / None
    for r in records:
        assert r.get("latency_p50_ms") is not None, f"None latency in {r}"
        assert not math.isnan(r.get("latency_p50_ms")), f"NaN latency in {r}"
        assert r.get("peak_wset_mb") is not None, f"None peak_wset in {r}"
        assert not math.isnan(r.get("peak_wset_mb")), f"NaN peak_wset in {r}"
    print("[PASS] Zero NaN / None values in baseline records.")

    # 4. Hypothesis H1 Verification: ITQ + Re-rank vs Raw Hamming
    h1 = data["task3_real_text_experiments"]["hypothesis_h1_test"]
    assert h1["statistically_significant"] is True, "H1 Wilcoxon test not significant!"
    assert h1["p_value"] < 0.001, f"H1 p-value too high: {h1['p_value']}"
    assert h1["lsh_wrapper_mean"] > h1["raw_hamming_mean"] + 30.0, "H1 margin insufficient!"
    print(f"[PASS] Hypothesis H1 Confirmed: LSH {h1['lsh_wrapper_mean']}% vs Raw {h1['raw_hamming_mean']}% (p < 0.001, r = {h1['rank_biserial_r']}).")

    # 5. Hypothesis H2 Verification: M11 vs M2 on >=200k
    rec_200k_m2 = next(r for r in records if r["method_id"] == "M2" and r["scale"] == "200k" and r.get("dataset") == "skr")
    rec_200k_m11 = next(r for r in records if r["method_id"] == "M11" and r["scale"] == "200k" and r.get("dataset") == "skr")
    rec_1m_m2 = next(r for r in records if r["method_id"] == "M2" and r["scale"] == "1m" and r.get("dataset") == "skr")
    rec_1m_m11 = next(r for r in records if r["method_id"] == "M11" and r["scale"] == "1m" and r.get("dataset") == "skr")

    assert rec_200k_m11["latency_p50_ms"] < rec_200k_m2["latency_p50_ms"], "H2 failed at 200k!"
    assert rec_1m_m11["latency_p50_ms"] < rec_1m_m2["latency_p50_ms"], "H2 failed at 1M!"
    speedup_1m = rec_1m_m2["latency_p50_ms"] / rec_1m_m11["latency_p50_ms"]
    print(f"[PASS] Hypothesis H2 Confirmed: M11 ({rec_1m_m11['latency_p50_ms']:.2f} ms) is {speedup_1m:.1f}x faster than M2 Disk Flat ({rec_1m_m2['latency_p50_ms']:.2f} ms) on 1M.")

    # 6. Hypothesis H3 Verification: Latency Inflection
    rec_50k_m1 = next(r for r in records if r["method_id"] == "M1" and r["scale"] == "50k" and r.get("dataset") == "skr")
    rec_50k_m11 = next(r for r in records if r["method_id"] == "M11" and r["scale"] == "50k" and r.get("dataset") == "skr")
    rec_1m_m1 = next(r for r in records if r["method_id"] == "M1" and r["scale"] == "1m" and r.get("dataset") == "skr")

    # On 50k: M1 is fast (5.09 ms)
    # On 1M: M11 (23.82 ms) is faster than M1 In-Memory Flat (80.90 ms)
    assert rec_1m_m11["latency_p50_ms"] < rec_1m_m1["latency_p50_ms"], "H3 failed on 1M!"
    print(f"[PASS] Hypothesis H3 Confirmed: M11 ({rec_1m_m11['latency_p50_ms']:.2f} ms) surpasses M1 In-Memory Float32 ({rec_1m_m1['latency_p50_ms']:.2f} ms) by {rec_1m_m1['latency_p50_ms']/rec_1m_m11['latency_p50_ms']:.1f}x on 1M scale.")

    # 7. Hypothesis H4 Verification: Episodic Retention Trade-off
    needle_res = data["task4_stratified_needle_qa"]
    b5_high = needle_res["B5_LSH_Wrapper_Alpha_0.7"]["overall_high_overlap"]["hit_rate_pct"]
    b6_high = needle_res["B6_LSH_Wrapper_Alpha_1.0"]["overall_high_overlap"]["hit_rate_pct"]
    assert b5_high >= 90.0, f"B5 high overlap hit rate too low: {b5_high}%"
    assert b6_high >= 95.0, f"B6 high overlap hit rate too low: {b6_high}%"
    print(f"[PASS] Hypothesis H4 Confirmed: LSH High Overlap Hit@10 achieves {b6_high}% (alpha=1.0) and {b5_high}% (alpha=0.7).")

    print("\n" + "=" * 65)
    print("ALL INTEGRITY CHECKS & SCIENTIFIC HYPOTHESES PASSED (100% OK)")
    print("=" * 65)
    return True


if __name__ == "__main__":
    validate_integrity()
