"""
Explicit retrieval gate for the v2 counter-poison set. A payload that
does not retrieve is EXCLUDED and LOGGED, never silently treated as a
generation-level null result.
"""
import json
import numpy as np
from db import get_conn, search_f1_filter_omitted, insert_chunks
from counter_poison import generate_counter_poison_set
from scifact_labels import load_claim_labels

VICTIM = "tenant_1"
ATTACKER = "tenant_0"
K = 5

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
    victim_queries = json.load(open("../data/victim_queries.json"))

    # Candidate pool: all labeled claims that are also victim queries
    # (so they're reachable through the normal victim query path)
    candidate_qids = [qid for qid in victim_queries
                       if qid in claims and claims[qid]["label"] is not None
                       and qid in queries_text][:30]
    print(f"Candidate pool: {len(candidate_qids)} labeled, victim-query claims")

    ids, vecs, texts, meta = generate_counter_poison_set(candidate_qids, queries_text, claims, seed=0)

    # Clean up any prior counter-poison rows before inserting fresh ones
    conn.execute("DELETE FROM chunks WHERE id LIKE 'counterpoison_%';")

    rows = [(ids[i], ATTACKER, True, vecs[i].tolist()) for i in range(len(ids))]
    insert_chunks(conn, rows)
    print(f"Inserted {len(rows)} v2 counter-poison candidates under tenant={ATTACKER}")

    gate_results = []
    for i, qid in enumerate(candidate_qids):
        if qid not in query_ids:
            continue
        qi = query_ids.index(qid)
        qvec = query_vecs[qi].tolist()
        poison_id = ids[i]

        res = search_f1_filter_omitted(conn, VICTIM, qvec, k=K)
        retrieved_ids = [r[0] for r in res]
        passed = poison_id in retrieved_ids
        rank = retrieved_ids.index(poison_id) + 1 if passed else None

        gate_results.append({
            "qid": qid, "poison_id": poison_id, "gold_label": claims[qid]["label"],
            "retrieval_gate_passed": passed, "rank_in_topk": rank,
        })
        status = f"PASS (rank {rank})" if passed else "FAIL -- excluded from ASR"
        print(f"  {qid} / {poison_id}: {status}")

    import pandas as pd
    df = pd.DataFrame(gate_results)
    df.to_csv("../results/retrieval_gate_v2.csv", index=False)

    n_pass = df["retrieval_gate_passed"].sum()
    n_total = len(df)
    print(f"\n=== GATE SUMMARY ===")
    print(f"Passed: {n_pass}/{n_total} ({100*n_pass/n_total:.1f}%)")
    print(f"Saved results/retrieval_gate_v2.csv")
    if n_pass < 10:
        print(f"\nWARNING: fewer than 10 payloads passed the gate. Consider widening")
        print(f"the candidate pool or adjusting the template's claim-term overlap")
        print(f"before proceeding to ASR.")

if __name__ == "__main__":
    main()
