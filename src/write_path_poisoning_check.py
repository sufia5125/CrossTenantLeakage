"""
Tests the actual claim in the paper's title: can an attacker WRITE a
poisoned document that becomes a permanent member of the VICTIM's
tenant partition, under RLS with only a USING clause (no explicit
WITH CHECK)? Then tests whether adding an explicit WITH CHECK blocks it.

This is distinct from all prior F1/F2/F3/F5 results, which test
READ-side leakage of documents that stayed in the ATTACKER's own
partition. This script tests WRITE-side poisoning: does the row
actually land inside the victim's partition and get served back
as legitimate victim content on every subsequent correctly-scoped
query, including under the "safe" mechanisms (rls_trusted, schema)?
"""
import json
import numpy as np
from db import get_conn, attacker_inserts_as_victim, add_explicit_with_check_policy, \
                search_rls_trusted, search_shared_filter

VICTIM = "tenant_1"
ATTACKER = "tenant_0"
K = 5

def load_one_target_query():
    query_ids = np.load("../data/query_ids.npy", allow_pickle=True).tolist()
    query_vecs = np.load("../data/query_vecs.npy")
    poison = json.load(open("../data/poison.json"))
    first_target_qid = list(poison["targets"].values())[0]["target_qid"]
    idx = query_ids.index(first_target_qid)
    return query_vecs[idx].tolist()

def run_check(conn, conn_super, label):
    qvec = load_one_target_query()
    poison_id = "writepath_poison_test"

    # clean up any previous attempt -- use superuser connection since
    # app_user is only granted SELECT, INSERT (not DELETE) by design,
    # matching a realistic application role's actual privileges.
    conn_super.execute("DELETE FROM chunks WHERE id = %s;", (poison_id,))

    succeeded, err = attacker_inserts_as_victim(
        conn, poison_id, claimed_victim_tenant=VICTIM,
        attacker_true_tenant=ATTACKER, poison_vec=qvec)

    print(f"\n--- {label} ---")
    print(f"INSERT with forged tenant_id=VICTIM, authenticated as ATTACKER: "
          f"{'SUCCEEDED' if succeeded else 'BLOCKED'}")
    if err:
        print(f"  (error: {err[:200]})")

    if succeeded:
        # Does it now show up as legitimate victim content under the
        # TRUSTED, correct-config read path? This is the real-world
        # consequence: every subsequent victim query is poisoned.
        res = search_rls_trusted(conn, VICTIM, qvec, k=K)
        appears = any(r[0] == poison_id for r in res)
        print(f"  -> Poisoned row served back under rls_trusted(VICTIM, ...): "
              f"{'YES -- POISONING CONFIRMED' if appears else 'no (not in top-k)'}")

    # cleanup (superuser connection, as above)
    conn_super.execute("DELETE FROM chunks WHERE id = %s;", (poison_id,))
    return succeeded

if __name__ == "__main__":
    conn = get_conn(as_app_user=True)
    conn_super = get_conn()  # needed for cleanup (app_user lacks DELETE) and ALTER POLICY

    vulnerable = run_check(conn, conn_super, "CURRENT POLICY (USING only, no explicit WITH CHECK)")

    print("\nApplying explicit WITH CHECK mitigation...")
    add_explicit_with_check_policy(conn_super)

    mitigated_still_vulnerable = run_check(conn, conn_super, "AFTER explicit WITH CHECK added")

    print("\n=== SUMMARY ===")
    if vulnerable and not mitigated_still_vulnerable:
        print("FINDING: the default USING-only RLS policy did NOT implicitly")
        print("protect INSERTs in this configuration -- an attacker could write")
        print("a row directly into the victim's tenant partition, which was then")
        print("served as legitimate content on every subsequent victim query.")
        print("Adding an explicit WITH CHECK clause fully closes this.")
    elif not vulnerable:
        print("FINDING: Postgres's USING-as-WITH-CHECK default DID protect this")
        print("policy as written -- the write-path attack was blocked even")
        print("without an explicit WITH CHECK clause. This contradicts a common")
        print("assumption and is worth reporting as a corrective result: relying")
        print("on the implicit default is still fragile/non-obvious practice,")
        print("but it is not automatically broken.")
    else:
        print("UNEXPECTED: attack succeeded even after explicit WITH CHECK was")
        print("added -- investigate policy application order, role grants, or")
        print("whether FORCE ROW LEVEL SECURITY is still in effect.")
