import sys, os, json
import numpy as np
import pandas as pd
from db import get_conn, search_f2_client_controlled, search_shared_filter

VICTIM, ATTACKER, K = "tenant_1", "tenant_0", 5
seed = sys.argv[1]; method = sys.argv[2]

qids_all = np.load("../data/query_ids.npy", allow_pickle=True).tolist()
qvecs = np.load("../data/query_vecs.npy")
victim_qs = [q for q in json.load(open("../data/victim_queries.json")) if q in qids_all]
conn = get_conn()

rows = []
for q in victim_qs:
    v = qvecs[qids_all.index(q)].tolist()
    forged = search_f2_client_controlled(conn, VICTIM, v, K)      # attacker claims victim
    honest = search_shared_filter(conn, ATTACKER, v, K)           # attacker as self
    rows.append({
        "seed": seed, "split_method": method, "qid": q,
        "forged_got_victim_data": any(r[1] == VICTIM for r in forged),
        "forged_got_poison": any(r[2] for r in forged),
        "honest_got_victim_data": any(r[1] == VICTIM for r in honest),
    })

out = "../results/f2_all_seeds.csv"
pd.DataFrame(rows).to_csv(out, mode="a", header=not os.path.exists(out), index=False)
print(f"seed={seed} {method}: n={len(rows)}")
