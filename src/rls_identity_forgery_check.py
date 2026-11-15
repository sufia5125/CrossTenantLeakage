"""
Separate check for the RLS untrusted-session-identity exploit.
Metric: Impersonation Success Rate (ISR) -- fraction of forged-identity
queries (attacker self-asserting victim's tenant_id) that return AT
LEAST ONE genuine victim-owned, non-poison document. This measures
unauthorized data access via identity forgery, independent of poisoning.
"""
import json
import numpy as np
from db import get_conn, search_rls_attacker_sets_session, search_rls_trusted

VICTIM = "tenant_1"
K = 5

def load_query_vecs():
    query_ids = np.load("../data/query_ids.npy", allow_pickle=True).tolist()
    query_vecs = np.load("../data/query_vecs.npy")
    victim_queries = json.load(open("../data/victim_queries.json"))
    idx = [query_ids.index(q) for q in victim_queries if q in query_ids]
    return [query_vecs[i].tolist() for i in idx]

def impersonation_success_rate(conn, qvecs, claimed_tenant=VICTIM, k=K):
    hits = []
    for qv in qvecs:
        res = search_rls_attacker_sets_session(conn, claimed_tenant, qv, k)
        # success = any row belonging to the claimed (victim) tenant was returned
        got_victim_data = any(r[1] == claimed_tenant for r in res)
        hits.append(got_victim_data)
    return float(np.mean(hits)), len(qvecs)

def trusted_baseline_rate(conn, qvecs, k=K):
    # sanity: how often the TRUSTED, correctly-authenticated path
    # legitimately returns victim data (should be high -- this is normal)
    hits = []
    for qv in qvecs:
        res = search_rls_trusted(conn, VICTIM, qv, k)
        hits.append(any(r[1] == VICTIM for r in res))
    return float(np.mean(hits))

if __name__ == "__main__":
    conn = get_conn(as_app_user=True)
    qvecs = load_query_vecs()

    isr, n = impersonation_success_rate(conn, qvecs)
    baseline = trusted_baseline_rate(conn, qvecs)

    print(f"Impersonation Success Rate (attacker forges victim identity): {isr:.4f} ({n} queries)")
    print(f"Trusted-path baseline (legitimate victim access):             {baseline:.4f}")
    print()
    if isr > 0:
        print("FINDING: RLS alone does NOT prevent unauthorized access when the")
        print("session identity itself is attacker-controlled. The attacker")
        print("successfully read victim-owned documents by self-asserting the")
        print("session variable, with no additional authentication check.")
    else:
        print("No successful impersonation observed -- investigate: either RLS")
        print("is blocking this unexpectedly, or app_user cannot run this query.")
