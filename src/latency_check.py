"""
A8 -- latency and storage cost per isolation mechanism.

Design notes (addressing prior review points):
- All query embeddings are loaded from the precomputed .npy cache
  BEFORE any timer starts. No model.encode() calls happen inside the
  timed region -- this was the review's #11 concern (the original
  Step 14 sketch timed embedding generation, which dominates DB
  round-trip time by an order of magnitude and would make the
  mechanisms incomparable).
- Each mechanism gets its own warmup pass (a handful of untimed
  queries) before the timed pass, so the first-query cache/connection
  warmup cost doesn't bias one mechanism over another.
- Timed region is ONLY the search function call itself.
- Mechanisms sharing the same underlying physical table (shared_filter,
  rls_trusted) are timed on freshly-prepared connections to avoid any
  cross-contamination of session state (SET app.tenant_id, etc.)
  between conditions, consistent with the shortfall-measurement
  concern raised earlier about SET persisting across queries.
"""
import time
import json
import numpy as np
from db import get_conn, search_shared_filter, search_schema, search_rls_trusted

VICTIM = "tenant_1"
K = 5
N_WARMUP = 5

def load_query_vecs():
    query_ids = np.load("../data/query_ids.npy", allow_pickle=True).tolist()
    query_vecs = np.load("../data/query_vecs.npy")
    victim_queries = json.load(open("../data/victim_queries.json"))
    idx = [query_ids.index(q) for q in victim_queries if q in query_ids]
    return [query_vecs[i].tolist() for i in idx]

MECH_FUNCS = {
    "shared_filter": (search_shared_filter, False),
    "schema": (search_schema, False),
    "rls_trusted": (search_rls_trusted, True),
}

def time_mechanism(mechanism, fn, needs_app_user, qvecs, k=K):
    conn = get_conn(as_app_user=needs_app_user)

    # warmup (untimed) -- separate connection-level warmup so index
    # pages / connection setup cost doesn't land inside the timed pass
    for qv in qvecs[:N_WARMUP]:
        fn(conn, VICTIM, qv, k)

    latencies = []
    for qv in qvecs:
        t0 = time.perf_counter()
        fn(conn, VICTIM, qv, k)
        latencies.append(time.perf_counter() - t0)

    latencies = np.array(latencies)
    return {
        "mechanism": mechanism,
        "p50_ms": float(np.percentile(latencies, 50) * 1000),
        "p95_ms": float(np.percentile(latencies, 95) * 1000),
        "mean_ms": float(latencies.mean() * 1000),
        "n_queries": len(latencies),
    }

def measure_storage(conn, mechanism):
    if mechanism in ("shared_filter", "rls_trusted"):
        size = conn.execute(
            "SELECT pg_size_pretty(pg_total_relation_size('chunks'));"
        ).fetchone()[0]
        bytes_ = conn.execute(
            "SELECT pg_total_relation_size('chunks');"
        ).fetchone()[0]
    elif mechanism == "schema":
        row = conn.execute("""
            SELECT pg_size_pretty(
                pg_total_relation_size('t_attacker.chunks') +
                pg_total_relation_size('t_victim.chunks')
            );
        """).fetchone()
        size = row[0]
        bytes_row = conn.execute("""
            SELECT pg_total_relation_size('t_attacker.chunks') +
                   pg_total_relation_size('t_victim.chunks');
        """).fetchone()
        bytes_ = bytes_row[0]
    else:
        size, bytes_ = None, None
    return {"mechanism": mechanism, "storage_pretty": size, "storage_bytes": bytes_}

if __name__ == "__main__":
    qvecs = load_query_vecs()
    print(f"Timing {len(qvecs)} victim queries per mechanism "
          f"(after {N_WARMUP}-query warmup, embeddings precomputed)...")

    latency_rows = []
    storage_rows = []
    conn_super = get_conn()

    for mechanism, (fn, needs_app_user) in MECH_FUNCS.items():
        row = time_mechanism(mechanism, fn, needs_app_user, qvecs)
        print(row)
        latency_rows.append(row)
        storage_rows.append(measure_storage(conn_super, mechanism))

    import pandas as pd
    lat_df = pd.DataFrame(latency_rows)
    stor_df = pd.DataFrame(storage_rows)
    lat_df.to_csv("../results/latency_by_mechanism.csv", index=False)
    stor_df.to_csv("../results/storage_by_mechanism.csv", index=False)
    print("\nSaved results/latency_by_mechanism.csv and results/storage_by_mechanism.csv")
    print("\n=== Storage ===")
    print(stor_df.to_string(index=False))
