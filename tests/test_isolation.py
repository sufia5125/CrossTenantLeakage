import sys
sys.path.insert(0, "../src")
import json
import numpy as np
from db import get_conn, search_shared_filter, search_schema, \
                search_rls_trusted, search_rls_attacker_sets_session, \
                search_f1_filter_omitted, search_f3_global_corpus, \
                search_f5_null_falls_through

VICTIM = "tenant_1"
K = 5

def dummy_qvec():
    return [0.01] * 384

def real_poison_qvec():
    """Use an ACTUAL poison target query vector, not a dummy, so the
    positive-control tests can meaningfully assert poison IS retrieved."""
    query_ids = np.load("../data/query_ids.npy", allow_pickle=True).tolist()
    query_vecs = np.load("../data/query_vecs.npy")
    poison = json.load(open("../data/poison.json"))
    first_target_qid = list(poison["targets"].values())[0]["target_qid"]
    idx = query_ids.index(first_target_qid)
    return query_vecs[idx].tolist()

# ---------- NEGATIVE CONTROLS: correct config must show zero leakage ----------

def test_shared_filter_no_leak():
    conn = get_conn()
    res = search_shared_filter(conn, VICTIM, dummy_qvec(), k=K)
    assert all(r[1] == VICTIM for r in res)

def test_rls_trusted_no_leak():
    conn = get_conn(as_app_user=True)
    res = search_rls_trusted(conn, VICTIM, dummy_qvec(), k=K)
    assert all(r[1] == VICTIM for r in res)

# ---------- POSITIVE CONTROLS: known-vulnerable paths MUST show leakage ----------
# Without these, a "0 leakage" result anywhere is ambiguous between
# "actually safe" and "fault never fired."

def test_f1_positive_control_leaks():
    conn = get_conn()
    qvec = real_poison_qvec()
    res = search_f1_filter_omitted(conn, VICTIM, qvec, k=K)
    assert any(r[2] for r in res), \
        "F1 is supposed to leak poison for its target query -- fault did not fire"

def test_f3_positive_control_leaks():
    conn = get_conn()
    qvec = real_poison_qvec()
    res = search_f3_global_corpus(conn, VICTIM, qvec, k=K)
    assert any(r[2] for r in res), \
        "F3 is supposed to leak poison via the global corpus -- fault did not fire"

def test_f5_positive_control_leaks():
    conn = get_conn()
    qvec = real_poison_qvec()
    res = search_f5_null_falls_through(conn, "", qvec, k=K)
    assert any(r[2] for r in res), \
        "F5 is supposed to leak poison when tenant_id is falsy -- fault did not fire"

def test_f5_all_sentinels_leak():
    conn = get_conn()
    qvec = real_poison_qvec()
    for sentinel in (None, "", "*"):
        res = search_f5_null_falls_through(conn, sentinel, qvec, k=K)
        assert any(r[2] for r in res), \
            f"F5 is supposed to leak for sentinel={sentinel!r} -- fault did not fire"

def test_rls_untrusted_session_grants_unauthorized_access():
    conn = get_conn(as_app_user=True)
    qvec = real_poison_qvec()
    # attacker (no real victim authorization) forges the session identity
    res = search_rls_attacker_sets_session(conn, VICTIM, qvec, k=K)
    got_victim_data = any(r[1] == VICTIM for r in res)
    assert got_victim_data, \
        "RLS is supposed to be bypassable via session-identity forgery -- it was not"
