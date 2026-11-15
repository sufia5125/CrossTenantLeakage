"""
F2 -- client-controlled tenant identity.
Metric: whether an attacker, with no real victim access, can read
victim-owned data (including poison) by supplying the victim's
tenant_id as an unverified, caller-controlled parameter.

This mirrors rls_identity_forgery_check.py's structure: both test
"does trusting an unverified identity channel allow unauthorized
access," just at different layers (app-level param vs DB session var).
"""
import json
import numpy as np
from db import get_conn, search_f2_client_controlled, search_shared_filter

VICTIM = "tenant_1"
ATTACKER = "tenant_0"
K = 5

def load_query_vecs():
    query_ids = np.load("../data/query_ids.npy", allow_pickle=True).tolist()
    query_vecs = np.load("../data/query_vecs.npy")
    victim_queries = json.load(open("../data/victim_queries.json"))
    idx = [query_ids.index(q) for q in victim_queries if q in query_ids]
    return [query_vecs[i].tolist() for i in idx]

def forged_identity_success_rate(conn, qvecs, k=K):
    """Attacker (truly authorized only as ATTACKER) forges the
    client-supplied tenant_id to claim VICTIM. Success = the query
    returns genuine victim-owned data despite the attacker having no
    real authorization for it."""
    hits = []
    for qv in qvecs:
        res = search_f2_client_controlled(conn, VICTIM, qv, k)  # forged claim
        got_victim_data = any(r[1] == VICTIM for r in res)
        hits.append(got_victim_data)
    return float(np.mean(hits)), len(qvecs)

def honest_attacker_baseline(conn, qvecs, k=K):
    """Control: what does the attacker legitimately see when querying
    under their OWN true identity (no forgery)? Should show attacker's
    own data only, never victim data -- confirms the forgery, not a
    general bug, is what causes any leakage above."""
    hits = []
    for qv in qvecs:
        res = search_shared_filter(conn, ATTACKER, qv, k)
        got_victim_data = any(r[1] == VICTIM for r in res)
        hits.append(got_victim_data)
    return float(np.mean(hits))

if __name__ == "__main__":
    conn = get_conn()
    qvecs = load_query_vecs()

    forged_rate, n = forged_identity_success_rate(conn, qvecs)
    honest_rate = honest_attacker_baseline(conn, qvecs)

    print(f"F2 Forged-Identity Success Rate (claims to be VICTIM): {forged_rate:.4f} ({n} queries)")
    print(f"Honest-identity baseline (queries as true ATTACKER):   {honest_rate:.4f}")
    print()
    if forged_rate > 0 and honest_rate == 0.0:
        print("FINDING: trusting a caller-supplied tenant_id with no authentication")
        print("allows unauthorized access to victim data via simple identity forgery,")
        print("while an attacker querying honestly under their own identity gets")
        print("none of the victim's data. The vulnerability is the trust boundary,")
        print("not a general data-isolation failure.")
    elif forged_rate == 0.0:
        print("No leakage observed via forged identity -- this means either the")
        print("tenant filter correctly restricts results regardless of which value")
        print("is passed (expected/correct behavior for THIS specific layer), or")
        print("victim data wasn't retrievable for these queries. Compare against")
        print("rls_identity_forgery_check.py's ISR for a sanity cross-check.")
