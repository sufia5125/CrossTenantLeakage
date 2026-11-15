"""
Tests the realistic failure mode identified after discovering that
Postgres's FOR-ALL implicit USING-as-WITH-CHECK protected the original
single-policy design. Here we test the common refactor where read and
write policies are SPLIT into FOR SELECT / FOR INSERT -- which has no
such implicit fallback, so an omitted or careless WITH CHECK on the
INSERT policy genuinely allows write-path tenant forgery.
"""
import json
import numpy as np
from db import get_conn, attacker_inserts_as_victim, \
                setup_split_policies_vulnerable, setup_split_policies_fixed, \
                search_rls_trusted

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
    poison_id = "splitpolicy_poison_test"
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
        res = search_rls_trusted(conn, VICTIM, qvec, k=K)
        appears = any(r[0] == poison_id for r in res)
        print(f"  -> Poisoned row served back under rls_trusted(VICTIM, ...): "
              f"{'YES -- POISONING CONFIRMED' if appears else 'no (not in top-k)'}")

    conn_super.execute("DELETE FROM chunks WHERE id = %s;", (poison_id,))
    return succeeded

if __name__ == "__main__":
    conn = get_conn(as_app_user=True)
    conn_super = get_conn()

    print("Setting up VULNERABLE split-policy configuration...")
    setup_split_policies_vulnerable(conn_super)
    vulnerable = run_check(conn, conn_super, "SPLIT POLICIES (FOR INSERT, no WITH CHECK)")

    print("\nApplying fix: explicit WITH CHECK on the FOR INSERT policy...")
    setup_split_policies_fixed(conn_super)
    fixed_result = run_check(conn, conn_super, "SPLIT POLICIES (FOR INSERT, WITH CHECK added)")

    print("\n=== SUMMARY ===")
    if vulnerable and not fixed_result:
        print("FINDING CONFIRMED: splitting read/write policies into FOR SELECT /")
        print("FOR INSERT removes Postgres's implicit USING-as-WITH-CHECK protection.")
        print("Without an explicit WITH CHECK on the INSERT policy, an attacker can")
        print("write a row directly into the victim's tenant partition, and it is")
        print("then served as legitimate content on every subsequent victim query.")
        print("This IS the realistic misconfiguration: the single-policy FOR ALL")
        print("design is safe by Postgres default, but the moment a developer")
        print("splits read/write logic -- a common and reasonable-looking refactor")
        print("-- that protection must be re-added explicitly or it silently")
        print("disappears.")
    else:
        print("Unexpected result -- investigate policy application and role grants.")
