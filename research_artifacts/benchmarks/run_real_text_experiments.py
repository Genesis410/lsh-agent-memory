"""Real-text experiments on 8,123 PingPong QA turns with 5-Fold Dialog-Split Cross Validation.

Evaluates:
- 5-Fold CV dialog-split leave-one-out with strict +/-2 adjacent turn exclusion
- Relative pool scaling (5%, 10%, 20% of N_fold) vs Random Pool baseline
- Memory unit ablation: Single-turn vs 3-turn sliding window
- Bilingual language segmentation (ID, EN, Mixed) with CMI
- Cross-domain generalization and sample size ablation
"""

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy.stats import wilcoxon

from lsh_agent_memory.core.hasher import fit_itq, encode
from lsh_agent_memory.core.hamming import BitwiseCountBackend
from lsh_agent_memory.data.pingpong_loader import load_pingpong_qa, extract_pingpong_turns
from lsh_agent_memory.data.extractor import SentenceExtractor


ID_WORDS = {
    'yang', 'dan', 'di', 'ini', 'itu', 'dari', 'ke', 'ada', 'bisa', 'ya', 'gak', 'nggak',
    'dia', 'saya', 'tau', 'kamu', 'apa', 'kenapa', 'sudah', 'udah', 'tapi', 'kita', 'dong',
    'kok', 'bukan', 'saja', 'kalau', 'kalo', 'buat', 'jangan', 'nih', 'gitu', 'banget'
}
EN_WORDS = {
    'the', 'and', 'in', 'is', 'of', 'to', 'you', 'that', 'it', 'for', 'on', 'with', 'as',
    'this', 'was', 'at', 'by', 'an', 'be', 'from', 'or', 'are', 'not', 'know', 'out',
    'have', 'has', 'they', 'we', 'will', 'my', 'can', 'all', 'so', 'what', 'there', 'if'
}


def classify_lang(text: str) -> str:
    """Classify language into id, en, or mixed using lexicon."""
    words = set(re.findall(r'\b[a-zA-Z]+\b', text.lower()))
    has_id = bool(words & ID_WORDS)
    has_en = bool(words & EN_WORDS)
    if has_id and has_en:
        return 'mixed'
    elif has_id:
        return 'id'
    elif has_en:
        return 'en'
    return 'mixed'


def compute_metrics(
    retrieved_idx: np.ndarray,
    gt_idx: np.ndarray,
    q_vec: np.ndarray,
    db_vecs: np.ndarray,
    k: int = 10,
    eps_fp32: float = 1e-4,
    eps_fp16: float = 1e-3,
) -> Dict[str, float]:
    """Compute Overlap@k, Tie-Tolerant Recall@k (1e-4 & 1e-3), and Quality Ratio."""
    ret_k = retrieved_idx[:k]
    gt_k = gt_idx[:k]

    # 1. Overlap@k
    hits = len(set(gt_k).intersection(set(ret_k)))
    overlap = (hits / k) * 100.0

    # Cosine similarities
    gt_sims = db_vecs[gt_k] @ q_vec
    ret_sims = db_vecs[ret_k] @ q_vec
    threshold = gt_sims[-1] if len(gt_sims) >= k else -1.0

    # 2. Tie-Tolerant Recall
    tie_hits_fp32 = np.sum(ret_sims >= (threshold - eps_fp32))
    tie_recall_fp32 = (tie_hits_fp32 / k) * 100.0

    tie_hits_fp16 = np.sum(ret_sims >= (threshold - eps_fp16))
    tie_recall_fp16 = (tie_hits_fp16 / k) * 100.0

    # 3. Cosine Quality Ratio
    mean_ret = float(np.mean(ret_sims)) if len(ret_sims) > 0 else 0.0
    mean_gt = float(np.mean(gt_sims)) if len(gt_sims) > 0 else 1.0
    quality_ratio = (mean_ret / max(mean_gt, 1e-9)) * 100.0

    return {
        "overlap10": overlap,
        "tie_recall_fp32": tie_recall_fp32,
        "tie_recall_fp16": tie_recall_fp16,
        "quality_ratio": quality_ratio,
    }


def run_pingpong_experiments():
    print("=================================================================")
    print("TASK 3: RIGOROUS REAL-TEXT 5-FOLD CV EXPERIMENTS (8,123 TURNS)")
    print("=================================================================")
    results = {}

    # 1. Load PingPong turns & embeddings
    raw_data = load_pingpong_qa()
    turns = extract_pingpong_turns(raw_data)
    assert len(turns) == 8123, f"Expected 8123 turns, got {len(turns)}"
    print(f"Loaded {len(turns)} conversational turns across 99 dialogues.")

    emb_path = Path("data/pingpong/turn_embeddings_8123.npy")
    assert emb_path.exists(), "Embeddings not found! Run Task 0 first."
    embeddings = np.load(emb_path)
    # L2 normalize
    embeddings /= np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)

    dialog_ids = np.array([t["dialog_id"] for t in turns], dtype=np.int32)
    turn_nums = np.array([t["turn_num"] for t in turns], dtype=np.int32)
    turn_langs = [classify_lang(t["text"]) for t in turns]

    id_count = sum(1 for l in turn_langs if l == 'id')
    en_count = sum(1 for l in turn_langs if l == 'en')
    mix_count = sum(1 for l in turn_langs if l == 'mixed')
    print(f"Language breakdown: ID={id_count} ({id_count/len(turns)*100:.1f}%), EN={en_count} ({en_count/len(turns)*100:.1f}%), Mixed={mix_count} ({mix_count/len(turns)*100:.1f}%)")

    results["corpus_summary"] = {
        "total_turns": len(turns),
        "total_dialogues": 99,
        "lang_id_count": id_count,
        "lang_en_count": en_count,
        "lang_mixed_count": mix_count,
    }

    # 2. Partition 99 Dialogues into 5 Folds
    unique_dialogs = sorted(list(set(dialog_ids)))
    n_dialogs = len(unique_dialogs)
    n_folds = 5
    fold_splits = np.array_split(unique_dialogs, n_folds)
    print(f"Dialog partition: {[len(f) for f in fold_splits]} dialogues per fold.")

    backend = BitwiseCountBackend()

    # -------------------------------------------------------------
    # Experiment 1: 5-Fold Cross Validation with Relative Pools
    # Pools: 5%, 10%, 20% of N_fold vs Random Pool Baseline
    # -------------------------------------------------------------
    print("\n[EXP 1] Running 5-Fold Cross Validation (Dialog-Split, Leave-One-Out)...")
    fold_results = {
        "pool_5pct": [],
        "pool_10pct": [],
        "pool_20pct": [],
        "random_pool_10pct": [],
    }

    # Track metrics per language across folds
    lang_metrics = {"id": [], "en": [], "mixed": []}
    all_lsh_overlaps = []
    all_raw_lsh_overlaps = []

    for fold_idx, test_dialogs in enumerate(fold_splits):
        test_dialog_set = set(test_dialogs)
        test_mask = np.isin(dialog_ids, test_dialogs)
        train_mask = ~test_mask

        n_test = int(np.sum(test_mask))
        n_train = int(np.sum(train_mask))
        print(f"\n--- Fold {fold_idx + 1}/5: {len(test_dialogs)} dialogues ({n_test} turns test, {n_train} turns train) ---")

        # Fit ITQ on train fold (2,500 sample)
        train_indices = np.where(train_mask)[0]
        rng = np.random.default_rng(42 + fold_idx)
        fit_sample_idx = rng.choice(train_indices, size=min(2500, len(train_indices)), replace=False)
        mu_itq, W_itq = fit_itq(embeddings[fit_sample_idx], bits=128, iters=50, seed=42)

        # Encode test fold
        test_indices = np.where(test_mask)[0]
        test_embs = embeddings[test_indices]
        test_d_ids = dialog_ids[test_indices]
        test_t_nums = turn_nums[test_indices]
        test_langs_sub = [turn_langs[i] for i in test_indices]

        test_codes = encode(test_embs, mu_itq, W_itq)

        # Ground truth and evaluation on each test turn (leave-one-out)
        # Relative pool sizes
        p5_k = max(10, int(0.05 * n_test))
        p10_k = max(10, int(0.10 * n_test))
        p20_k = max(10, int(0.20 * n_test))

        fold_p5_ov, fold_p10_ov, fold_p20_ov = [], [], []
        fold_p10_tie32, fold_p10_tie16, fold_p10_qr = [], [], []
        fold_rand_ov = []

        # Evaluate on all test queries
        for q_sub_i in range(n_test):
            q_vec = test_embs[q_sub_i]
            q_code = test_codes[q_sub_i]
            q_diag = test_d_ids[q_sub_i]
            q_tnum = test_t_nums[q_sub_i]
            q_lang = test_langs_sub[q_sub_i]

            # Exclusion mask: same dialog within +/- 2 turns
            same_d = test_d_ids == q_diag
            adj_t = np.abs(test_t_nums - q_tnum) <= 2
            excl_mask = same_d & adj_t

            # Exact cosine ground truth (excluding adjacent turns)
            all_sims = test_embs @ q_vec
            all_sims[excl_mask] = -1.0  # exclude
            gt_top10 = np.argsort(-all_sims)[:10]

            # Pool 5%
            cand5, _ = backend.search(q_code, test_codes, n=p5_k + 10)
            cand5 = [c for c in cand5 if not excl_mask[c]][:p5_k]
            sub_sims5 = test_embs[cand5] @ q_vec
            ret5 = [cand5[idx] for idx in np.argsort(-sub_sims5)[:10]]
            m5 = compute_metrics(np.array(ret5), gt_top10, q_vec, test_embs)
            fold_p5_ov.append(m5["overlap10"])

            # Pool 10%
            cand10, _ = backend.search(q_code, test_codes, n=p10_k + 10)
            cand10 = [c for c in cand10 if not excl_mask[c]][:p10_k]
            sub_sims10 = test_embs[cand10] @ q_vec
            ret10 = [cand10[idx] for idx in np.argsort(-sub_sims10)[:10]]
            m10 = compute_metrics(np.array(ret10), gt_top10, q_vec, test_embs)
            fold_p10_ov.append(m10["overlap10"])
            fold_p10_tie32.append(m10["tie_recall_fp32"])
            fold_p10_tie16.append(m10["tie_recall_fp16"])
            fold_p10_qr.append(m10["quality_ratio"])

            all_lsh_overlaps.append(m10["overlap10"])

            # Raw Hamming (no re-rank) for Hypothesis H1 test
            cand_raw = [c for c in cand10[:10]]
            all_raw_lsh_overlaps.append(compute_metrics(np.array(cand_raw), gt_top10, q_vec, test_embs)["overlap10"])

            # Language breakdown on pool 10%
            lang_metrics[q_lang].append(m10["overlap10"])

            # Pool 20%
            cand20, _ = backend.search(q_code, test_codes, n=p20_k + 10)
            cand20 = [c for c in cand20 if not excl_mask[c]][:p20_k]
            sub_sims20 = test_embs[cand20] @ q_vec
            ret20 = [cand20[idx] for idx in np.argsort(-sub_sims20)[:10]]
            m20 = compute_metrics(np.array(ret20), gt_top10, q_vec, test_embs)
            fold_p20_ov.append(m20["overlap10"])

            # Random Pool 10% Baseline
            valid_cand = np.where(~excl_mask)[0]
            rand_cand = rng.choice(valid_cand, size=p10_k, replace=False)
            rand_sims = test_embs[rand_cand] @ q_vec
            rand_ret = rand_cand[np.argsort(-rand_sims)[:10]]
            m_rand = compute_metrics(rand_ret, gt_top10, q_vec, test_embs)
            fold_rand_ov.append(m_rand["overlap10"])

        f_res = {
            "fold": fold_idx + 1,
            "n_test": n_test,
            "pool_5pct_overlap10": round(float(np.mean(fold_p5_ov)), 2),
            "pool_10pct_overlap10": round(float(np.mean(fold_p10_ov)), 2),
            "pool_10pct_tie_recall_fp32": round(float(np.mean(fold_p10_tie32)), 2),
            "pool_10pct_tie_recall_fp16": round(float(np.mean(fold_p10_tie16)), 2),
            "pool_10pct_quality_ratio": round(float(np.mean(fold_p10_qr)), 2),
            "pool_20pct_overlap10": round(float(np.mean(fold_p20_ov)), 2),
            "random_pool_10pct_overlap10": round(float(np.mean(fold_rand_ov)), 2),
        }
        fold_results["pool_5pct"].append(f_res["pool_5pct_overlap10"])
        fold_results["pool_10pct"].append(f_res["pool_10pct_overlap10"])
        fold_results["pool_20pct"].append(f_res["pool_20pct_overlap10"])
        fold_results["random_pool_10pct"].append(f_res["random_pool_10pct_overlap10"])

        print(f"  Fold {fold_idx + 1} Results:")
        print(f"    Pool  5% ({p5_k} cands):  Overlap@10 = {f_res['pool_5pct_overlap10']}%")
        print(f"    Pool 10% ({p10_k} cands):  Overlap@10 = {f_res['pool_10pct_overlap10']}% (Tie-Recall: {f_res['pool_10pct_tie_recall_fp32']}%, Quality: {f_res['pool_10pct_quality_ratio']}%)")
        print(f"    Pool 20% ({p20_k} cands):  Overlap@10 = {f_res['pool_20pct_overlap10']}%")
        print(f"    Random Pool 10%:           Overlap@10 = {f_res['random_pool_10pct_overlap10']}%")

    results["cross_validation_5fold"] = {
        "pool_5pct_mean": round(float(np.mean(fold_results["pool_5pct"])), 2),
        "pool_5pct_std": round(float(np.std(fold_results["pool_5pct"])), 2),
        "pool_10pct_mean": round(float(np.mean(fold_results["pool_10pct"])), 2),
        "pool_10pct_std": round(float(np.std(fold_results["pool_10pct"])), 2),
        "pool_20pct_mean": round(float(np.mean(fold_results["pool_20pct"])), 2),
        "pool_20pct_std": round(float(np.std(fold_results["pool_20pct"])), 2),
        "random_pool_10pct_mean": round(float(np.mean(fold_results["random_pool_10pct"])), 2),
        "random_pool_10pct_std": round(float(np.std(fold_results["random_pool_10pct"])), 2),
    }

    # Hypothesis H1: Wilcoxon Signed-Rank Test between LSH-Wrapper and Raw Hamming
    print("\nTesting Hypothesis H1 (Wilcoxon Signed-Rank Test: LSH-Wrapper vs Raw Hamming)...")
    w_stat, w_p = wilcoxon(all_lsh_overlaps, all_raw_lsh_overlaps, alternative='greater')
    # Rank-biserial correlation
    diffs = np.array(all_lsh_overlaps) - np.array(all_raw_lsh_overlaps)
    pos_ranks = np.sum(diffs > 0)
    neg_ranks = np.sum(diffs < 0)
    total_non_zero = pos_ranks + neg_ranks
    rank_biserial = (pos_ranks - neg_ranks) / max(1, total_non_zero)

    results["hypothesis_h1_test"] = {
        "lsh_wrapper_mean": round(float(np.mean(all_lsh_overlaps)), 2),
        "raw_hamming_mean": round(float(np.mean(all_raw_lsh_overlaps)), 2),
        "wilcoxon_stat": float(w_stat),
        "p_value": float(w_p),
        "rank_biserial_r": round(float(rank_biserial), 4),
        "statistically_significant": bool(w_p < 0.001),
    }
    print(f"  H1: LSH Mean={results['hypothesis_h1_test']['lsh_wrapper_mean']}%, Raw Mean={results['hypothesis_h1_test']['raw_hamming_mean']}%")
    print(f"  Wilcoxon p-value: {w_p:.2e}, rank-biserial r: {rank_biserial:.4f} (Significant: {results['hypothesis_h1_test']['statistically_significant']})")

    # -------------------------------------------------------------
    # Experiment 2: Language Segment Analysis
    # -------------------------------------------------------------
    print("\n[EXP 2] Language Segmentation Performance (ID vs EN vs Mixed)...")
    results["language_segment_evaluation"] = {
        "id_mean_overlap10": round(float(np.mean(lang_metrics["id"])), 2),
        "en_mean_overlap10": round(float(np.mean(lang_metrics["en"])), 2),
        "mixed_mean_overlap10": round(float(np.mean(lang_metrics["mixed"])), 2),
    }
    print(f"  ID Pure:    {results['language_segment_evaluation']['id_mean_overlap10']}%")
    print(f"  EN Pure:    {results['language_segment_evaluation']['en_mean_overlap10']}%")
    print(f"  Mixed Code: {results['language_segment_evaluation']['mixed_mean_overlap10']}%")

    # -------------------------------------------------------------
    # Experiment 3: Memory Unit Ablation (Single-Turn vs 3-Turn Window)
    # -------------------------------------------------------------
    print("\n[EXP 3] Memory Unit Ablation (Single-Turn vs 3-Turn Sliding Window)...")
    # Build 3-turn window texts
    window_texts = []
    window_d_ids = []
    window_t_nums = []

    for d_idx, d in enumerate(unique_dialogs):
        diag_turns = [t for t in turns if t["dialog_id"] == d]
        for t_i in range(len(diag_turns)):
            start_i = max(0, t_i - 2)
            w_text = " ".join([diag_turns[j]["text"] for j in range(start_i, t_i + 1)])
            window_texts.append(w_text)
            window_d_ids.append(d)
            window_t_nums.append(diag_turns[t_i]["turn_num"])

    w_cache = Path("data/pingpong/window_embeddings_8123.npy")
    if w_cache.exists():
        w_embeddings = np.load(w_cache)
    else:
        print("Encoding 3-turn windows...")
        extractor = SentenceExtractor()
        w_embeddings = extractor.encode_texts(window_texts)
        np.save(w_cache, w_embeddings)

    w_embeddings /= np.maximum(np.linalg.norm(w_embeddings, axis=1, keepdims=True), 1e-12)

    # Evaluate 3-turn window on Fold 1
    test_w_indices = np.where(np.isin(dialog_ids, fold_splits[0]))[0]
    w_test_embs = w_embeddings[test_w_indices]
    w_t_nums = np.array(window_t_nums)[test_w_indices]
    w_d_ids = np.array(window_d_ids)[test_w_indices]

    mu_w, W_w = fit_itq(w_embeddings[~np.isin(dialog_ids, fold_splits[0])][:2500], bits=128, iters=50, seed=42)
    w_codes = encode(w_test_embs, mu_w, W_w)

    w_overlaps = []
    p10_w_k = max(10, int(0.10 * len(test_w_indices)))
    for i in range(min(500, len(test_w_indices))):
        q = w_test_embs[i]
        same_d = w_d_ids == w_d_ids[i]
        adj_t = np.abs(w_t_nums - w_t_nums[i]) <= 2
        excl = same_d & adj_t

        sims = w_test_embs @ q
        sims[excl] = -1.0
        gt10 = np.argsort(-sims)[:10]

        cand, _ = backend.search(w_codes[i], w_codes, n=p10_w_k + 10)
        cand = [c for c in cand if not excl[c]][:p10_w_k]
        sub = w_test_embs[cand] @ q
        ret = [cand[idx] for idx in np.argsort(-sub)[:10]]
        w_overlaps.append(compute_metrics(np.array(ret), gt10, q, w_test_embs)["overlap10"])

    results["memory_unit_ablation"] = {
        "single_turn_overlap10": results["cross_validation_5fold"]["pool_10pct_mean"],
        "three_turn_window_overlap10": round(float(np.mean(w_overlaps)), 2),
    }
    print(f"  Single Turn Overlap@10:       {results['memory_unit_ablation']['single_turn_overlap10']}%")
    print(f"  3-Turn Window Overlap@10:     {results['memory_unit_ablation']['three_turn_window_overlap10']}%")

    # -------------------------------------------------------------
    # Experiment 4: ITQ Sample Size Ablation
    # -------------------------------------------------------------
    print("\n[EXP 4] ITQ Fit Sample Size Ablation (500, 1000, 2500, Full)...")
    sample_sizes = [500, 1000, 2500, len(embeddings[~test_mask])]
    sample_results = {}

    for s_sz in sample_sizes:
        sub_train = embeddings[~test_mask][:s_sz]
        mu_s, W_s = fit_itq(sub_train, bits=128, iters=50, seed=42)
        c_test = encode(test_embs, mu_s, W_s)

        s_ovs = []
        for i in range(min(300, len(test_embs))):
            q = test_embs[i]
            same_d = test_d_ids == test_d_ids[i]
            adj_t = np.abs(test_t_nums - test_t_nums[i]) <= 2
            excl = same_d & adj_t

            sims = test_embs @ q
            sims[excl] = -1.0
            gt10 = np.argsort(-sims)[:10]

            cand, _ = backend.search(c_test[i], c_test, n=p10_k + 10)
            cand = [c for c in cand if not excl[c]][:p10_k]
            sub = test_embs[cand] @ q
            ret = [cand[idx] for idx in np.argsort(-sub)[:10]]
            s_ovs.append(compute_metrics(np.array(ret), gt10, q, test_embs)["overlap10"])

        lbl = str(s_sz) if s_sz < 5000 else "full_fold"
        sample_results[lbl] = round(float(np.mean(s_ovs)), 2)
        print(f"  Sample {lbl}: Overlap@10 = {sample_results[lbl]}%")

    results["sample_size_ablation"] = sample_results

    # -------------------------------------------------------------
    # Experiment 5: Cross-Domain Semantic Transfer
    # -------------------------------------------------------------
    print("\n[EXP 5] Cross-Domain Semantic Transfer (Fit on Synthetic-Text -> Test on PingPong)...")
    # Generate 2,500 synthetic text embeddings or load from SV
    st_train = np.load("data/benchmarks/sv_50k.npy")[:2500]
    mu_st, W_st = fit_itq(st_train, bits=128, iters=50, seed=42)
    c_st_test = encode(test_embs, mu_st, W_st)

    st_ovs = []
    for i in range(min(300, len(test_embs))):
        q = test_embs[i]
        same_d = test_d_ids == test_d_ids[i]
        adj_t = np.abs(test_t_nums - test_t_nums[i]) <= 2
        excl = same_d & adj_t

        sims = test_embs @ q
        sims[excl] = -1.0
        gt10 = np.argsort(-sims)[:10]

        cand, _ = backend.search(c_st_test[i], c_st_test, n=p10_k + 10)
        cand = [c for c in cand if not excl[c]][:p10_k]
        sub = test_embs[cand] @ q
        ret = [cand[idx] for idx in np.argsort(-sub)[:10]]
        st_ovs.append(compute_metrics(np.array(ret), gt10, q, test_embs)["overlap10"])

    results["cross_domain_transfer"] = {
        "in_domain_overlap10": results["cross_validation_5fold"]["pool_10pct_mean"],
        "synthetic_transfer_overlap10": round(float(np.mean(st_ovs)), 2),
        "delta_overlap": round(float(np.mean(st_ovs)) - results["cross_validation_5fold"]["pool_10pct_mean"], 2),
    }
    print(f"  In-Domain PingPong:    {results['cross_domain_transfer']['in_domain_overlap10']}%")
    print(f"  Transfer from Synth:   {results['cross_domain_transfer']['synthetic_transfer_overlap10']}% (Delta: {results['cross_domain_transfer']['delta_overlap']}%)")

    # Save to file
    out_path = Path("benchmarks/real_text_experiments_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n[OK] Real text experiment results saved to {out_path}!")
    return results


if __name__ == "__main__":
    run_pingpong_experiments()
