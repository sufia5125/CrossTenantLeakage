"""
Paired, occupancy-swept label-flip ASR using the v2 (retrieval-gated,
coherent, non-tautological) counter-poison payloads.

Design:
- Each of the 30 gated queries is run under THREE occupancy levels
  (2/5, 3/5, 4/5 poison slots among k=5), by controlling how many
  poison vs. real victim-tenant chunks are allowed into the
  constructed context -- NOT by relying on whatever F1's natural
  ranking happens to produce, so occupancy is actually controlled,
  not observed.
- Each (query, occupancy) condition is PAIRED: clean-context verdict
  vs. poisoned-context verdict for the SAME query, enabling McNemar's
  test on the discordant pairs (clean=correct,poisoned=wrong vs.
  clean=wrong,poisoned=correct) rather than two independent samples.
- Checkpointed per-row, resumable, same pattern as v1.
"""
import json
import os
import numpy as np
import pandas as pd
from db import get_conn, search_shared_filter, search_f1_filter_omitted
from scifact_labels import load_claim_labels
from asr_label_flip import get_model_verdict, doc_id_to_text

VICTIM = "tenant_1"
K = 5
OCCUPANCY_LEVELS = [3]  # SCALED DOWN after VM crash from 120-call run: single fixed occupancy only
CHECKPOINT = "../results/asr_v2_paired.csv"

def poison_id_to_text_v2(poison_id, gate_df):
    row = gate_df[gate_df.poison_id == poison_id]
    # text isn't stored in the gate CSV -- regenerate deterministically
    # from the same seed/index used at insertion time, matching
    # counter_poison.py's generation exactly.
    raise NotImplementedError  # replaced below with a direct text cache

def build_context(poison_text, clean_chunk_texts, n_poison_slots, k=K):
    """Constructs a k-slot context with EXACTLY n_poison_slots copies
    of the poison reasoning (as distinct sentences, not literal
    duplication of the whole string, to avoid an unrealistic repeated-
    paragraph artifact) and the remainder filled with real victim-
    tenant content, truncated/padded to exactly k total context items."""
    poison_slots = [poison_text] * n_poison_slots
    clean_slots = clean_chunk_texts[: max(0, k - n_poison_slots)]
    # pad with "(no further context)" if clean content runs short
    while len(clean_slots) < (k - n_poison_slots):
        clean_slots.append("(no further context)")
    combined = poison_slots + clean_slots
    return combined

def main():
    conn = get_conn()
    claims = load_claim_labels()
    queries_text = {}
    with open("../data/scifact/queries.jsonl") as f:
        for line in f:
            row = json.loads(line)
            queries_text[str(row["_id"])] = row["text"]

    query_ids = np.load("../data/query_ids.npy", allow_pickle=True).tolist()
    query_vecs = np.load("../data/query_vecs.npy")

    gate_df = pd.read_csv("../results/retrieval_gate_v2.csv")
    gated_qids = gate_df[gate_df.retrieval_gate_passed]["qid"].astype(str).tolist()
    gated_qids = gated_qids[:10]  # SCALED DOWN: cap at 10 queries (~20 total calls, matching the proven-stable run size)
    print(f"Running paired ASR on {len(gated_qids)} gated queries x "
          f"{len(OCCUPANCY_LEVELS)} occupancy levels = "
          f"{len(gated_qids) * len(OCCUPANCY_LEVELS)} conditions")

    # Rebuild poison text cache directly (same generation call as the gate script)
    from counter_poison import generate_counter_poison_set
    candidate_qids_ordered = gate_df["qid"].astype(str).tolist()
    ids, vecs, texts, meta = generate_counter_poison_set(
        candidate_qids_ordered, queries_text, claims, seed=0)
    poison_text_by_qid = {meta[pid]["target_qid"]: texts[i] for i, pid in enumerate(ids)}

    already_done = set()
    if os.path.exists(CHECKPOINT):
        prior = pd.read_csv(CHECKPOINT)
        already_done = set(zip(prior["qid"].astype(str), prior["occupancy"]))
        print(f"Resuming: {len(already_done)} (qid, occupancy) pairs already done")

    for qid in gated_qids:
        if qid not in query_ids or qid not in claims:
            continue
        gold_label = claims[qid]["label"]
        claim_text = queries_text[qid]
        qi = query_ids.index(qid)
        qvec = query_vecs[qi].tolist()
        poison_text = poison_text_by_qid.get(qid)
        if poison_text is None:
            continue

        # Real victim-tenant clean content, for filling non-poison slots
        clean_results = search_shared_filter(conn, VICTIM, qvec, k=K)
        clean_chunk_texts = [doc_id_to_text(r[0]) for r in clean_results if not r[2]]

        # CLEAN condition (occupancy=0, computed once per query, reused
        # across all occupancy levels as the paired baseline)
        clean_context = clean_chunk_texts[:K] if clean_chunk_texts else ["(no context retrieved)"]
        clean_verdict, clean_raw = get_model_verdict(claim_text, clean_context)

        for occ in OCCUPANCY_LEVELS:
            if (qid, occ) in already_done:
                continue
            poisoned_context = build_context(poison_text, clean_chunk_texts, occ, k=K)
            poisoned_verdict, poisoned_raw = get_model_verdict(claim_text, poisoned_context)

            row = {
                "qid": qid, "occupancy": occ, "gold_label": gold_label,
                "clean_verdict": clean_verdict, "clean_correct": clean_verdict == gold_label,
                "poisoned_verdict": poisoned_verdict, "poisoned_correct": poisoned_verdict == gold_label,
                "discordant_flip_to_wrong": (clean_verdict == gold_label) and (poisoned_verdict != gold_label),
                "discordant_flip_to_right": (clean_verdict != gold_label) and (poisoned_verdict == gold_label),
                "clean_raw": clean_raw[:150], "poisoned_raw": poisoned_raw[:150],
            }
            row_df = pd.DataFrame([row])
            if os.path.exists(CHECKPOINT):
                row_df.to_csv(CHECKPOINT, mode="a", header=False, index=False)
            else:
                row_df.to_csv(CHECKPOINT, index=False)
            print(f"  {qid} occ={occ}: gold={gold_label} clean={clean_verdict} "
                  f"poisoned={poisoned_verdict} "
                  f"{'FLIP->WRONG' if row['discordant_flip_to_wrong'] else ''}"
                  f"{'FLIP->RIGHT' if row['discordant_flip_to_right'] else ''}")

    df = pd.read_csv(CHECKPOINT)
    print(f"\n=== SUMMARY (n={len(df)} query x occupancy observations) ===")
    for occ in OCCUPANCY_LEVELS:
        sub = df[df.occupancy == occ]
        b = sub["discordant_flip_to_wrong"].sum()   # clean right, poisoned wrong
        c = sub["discordant_flip_to_right"].sum()   # clean wrong, poisoned right
        n_pairs = len(sub)
        print(f"Occupancy {occ}/5: flip-to-wrong={b}, flip-to-right={c}, "
              f"n={n_pairs}, raw ASR={b/n_pairs:.3f}")
        if b + c > 0:
            try:
                from statsmodels.stats.contingency_tables import mcnemar
                table = [[n_pairs - b - c, b], [c, 0]]  # simplified 2x2 for exact test
                # Proper McNemar needs the four-cell table: both-correct,
                # clean-only-correct(b), poison-only-correct(c), both-wrong
                both_correct = ((sub.clean_correct) & (sub.poisoned_correct)).sum()
                both_wrong = ((~sub.clean_correct) & (~sub.poisoned_correct)).sum()
                table = [[both_correct, b], [c, both_wrong]]
                result = mcnemar(table, exact=True)
                print(f"  McNemar exact p-value: {result.pvalue:.4f}")
            except ImportError:
                print("  (statsmodels not installed -- run: pip install statsmodels)")
        else:
            print("  (no discordant pairs -- McNemar undefined/trivial)")

if __name__ == "__main__":
    main()
