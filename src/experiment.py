import json
import sys
import os
import numpy as np
import pandas as pd
from db import get_conn, search_shared_filter, search_schema, \
                search_rls_trusted, search_rls_attacker_sets_session, \
                search_f1_filter_omitted, search_f2_client_controlled, \
                search_f3_global_corpus, search_f5_null_falls_through
from metrics import ctrr_poison, ctrr_any, poison_precision_at_k, shortfall, bootstrap_ci

VICTIM = "tenant_1"
ATTACKER = "tenant_0"
K = 5

MECH_FUNCS = {
    "shared_filter": search_shared_filter,
    "schema": search_schema,
    "rls_trusted": search_rls_trusted,              # correct-config RLS
    # NOTE: rls_untrusted_session is intentionally excluded here -- it is
    # not a poison-leakage condition, it is an identity-forgery exploit
    # with its own metric. See rls_identity_forgery_check.py.
}

FAULT_FUNCS = {
    "none": None,
    "F1": search_f1_filter_omitted,
    "F2": search_f2_client_controlled,
    "F3": search_f3_global_corpus,
    "F5": search_f5_null_falls_through,
}

def load_query_vecs():
    query_ids = np.load("../data/query_ids.npy", allow_pickle=True).tolist()
    query_vecs = np.load("../data/query_vecs.npy")
    victim_queries = json.load(open("../data/victim_queries.json"))
    idx = [query_ids.index(q) for q in victim_queries if q in query_ids]
    qids = [victim_queries[j] for j, q in enumerate(victim_queries) if q in query_ids]
    return [query_vecs[i].tolist() for i in idx], qids

def load_targeted_qids():
    poison = json.load(open("../data/poison.json"))
    return set(meta["target_qid"] for meta in poison["targets"].values())

def run_condition(conn, mechanism, fault, qvecs, k=K):
    results_per_query = []
    for qv in qvecs:
        if mechanism == "rls_untrusted_session":
            # This mechanism IS the fault -- the attacker sets the session
            # variable themselves. Only its "none" (no additional fault
            # layered on top) condition is meaningful; it is itself F_RLS1.
            if fault != "none":
                return None
            res = MECH_FUNCS[mechanism](conn, VICTIM, qv, k)
        elif fault == "none":
            fn = MECH_FUNCS[mechanism]
            res = fn(conn, VICTIM, qv, k)
        elif fault in ("F1", "F3"):
            if mechanism != "shared_filter":
                return None  # not applicable combination
            res = FAULT_FUNCS[fault](conn, VICTIM, qv, k)
        elif fault == "F5":
            if mechanism != "shared_filter":
                return None
            res = FAULT_FUNCS[fault](conn, "", qv, k)
        elif fault == "F2":
            if mechanism != "shared_filter":
                return None
            res = FAULT_FUNCS[fault](conn, VICTIM, qv, k)
        else:
            return None
        results_per_query.append(res)
    return results_per_query

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    split_method = sys.argv[2] if len(sys.argv) > 2 else "kmeans"

    conn = get_conn()
    conn_appuser = get_conn(as_app_user=True)
    qvecs, qids = load_query_vecs()
    targeted_qids = load_targeted_qids()
    print(f"Seed {seed} ({split_method}): running on {len(qvecs)} victim queries "
          f"({len(targeted_qids)} targeted by poison)")

    agg_rows = []
    raw_rows = []
    for mechanism in MECH_FUNCS:
        for fault in FAULT_FUNCS:
            active_conn = conn_appuser if mechanism.startswith("rls") else conn
            res = run_condition(active_conn, mechanism, fault, qvecs)
            if res is None:
                continue

            agg_row = {
                "seed": seed,
                "split_method": split_method,
                "mechanism": mechanism,
                "fault": fault,
                "ctrr_poison": ctrr_poison(res, VICTIM),
                "ctrr_any": ctrr_any(res, VICTIM),
                "poison_precision_at_k": poison_precision_at_k(res, K),
                "shortfall": shortfall(res, K),
                "n_queries": len(res),
            }
            print(agg_row)
            agg_rows.append(agg_row)

            # RAW per-query rows -- needed for query-level bootstrap and
            # targeted/untargeted breakdown. One row per victim query.
            for i, query_result in enumerate(res):
                qid = qids[i] if i < len(qids) else None
                poison_hit = any(r[2] and r[1] != VICTIM for r in query_result)
                any_hit = any(r[1] != VICTIM for r in query_result)
                n_poison_in_topk = sum(1 for r in query_result if r[2])
                raw_rows.append({
                    "seed": seed,
                    "split_method": split_method,
                    "mechanism": mechanism,
                    "fault": fault,
                    "qid": qid,
                    "is_targeted": qid in targeted_qids,
                    "poison_hit": poison_hit,
                    "any_hit": any_hit,
                    "n_poison_in_topk": n_poison_in_topk,
                    "n_returned": len(query_result),
                })

    out_path = "../results/main_multiseed.csv"
    df = pd.DataFrame(agg_rows)
    if os.path.exists(out_path):
        df.to_csv(out_path, mode="a", header=False, index=False)
    else:
        df.to_csv(out_path, index=False)

    raw_path = "../results/raw_queries.csv"
    raw_df = pd.DataFrame(raw_rows)
    if os.path.exists(raw_path):
        raw_df.to_csv(raw_path, mode="a", header=False, index=False)
    else:
        raw_df.to_csv(raw_path, index=False)

    print(f"Appended seed={seed} ({split_method}) results to {out_path} and {raw_path}")

if __name__ == "__main__":
    main()
