"""
Blocker 1 fix: the explicit 2x2 decoupling identity authentication from
write-policy strictness. Four cells, each run and reported independently,
plus the DDL outcome is logged (not assumed) per the reviewer's caution
that an omitted WITH CHECK may be REJECTED by Postgres at policy-creation
time rather than silently defaulting open.
"""
import json
import numpy as np
from db import get_conn, attacker_insert_2x2, search_rls_trusted, \
                setup_split_policies_vulnerable, setup_split_policies_fixed

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

def try_create_omitted_check_policy(conn_super):
    """Empirically test whether Postgres accepts a FOR INSERT policy
    with NO WITH CHECK clause at all (as opposed to an explicit
    WITH CHECK (true)), and what it does if so. Logs the DDL outcome
    rather than assuming either way, per reviewer caution."""
    conn_super.execute("DROP POLICY IF EXISTS tenant_iso ON chunks;")
    conn_super.execute("DROP POLICY IF EXISTS tenant_read ON chunks;")
    conn_super.execute("DROP POLICY IF EXISTS tenant_write ON chunks;")
    conn_super.execute("""
        CREATE POLICY tenant_read ON chunks FOR SELECT
        USING (tenant_id = current_setting('app.tenant_id', true));
    """)
    try:
        conn_super.execute("CREATE POLICY tenant_write ON chunks FOR INSERT;")
        return "ACCEPTED -- policy created with no WITH CHECK clause at all"
    except Exception as e:
        return f"REJECTED by Postgres: {str(e)[:200]}"

def run_cell(conn, conn_super, label, identity_forged):
    qvec = load_one_target_query()
    poison_id = f"2x2_{label.replace(' ', '_')}"
    conn_super.execute("DELETE FROM chunks WHERE id = %s;", (poison_id,))

    succeeded, err = attacker_insert_2x2(
        conn, poison_id, qvec, identity_forged=identity_forged,
        attacker_tenant=ATTACKER, victim_tenant=VICTIM)

    print(f"\n--- {label} ---")
    print(f"  identity_forged={identity_forged}  "
          f"(session claims {'VICTIM' if identity_forged else 'ATTACKER (true identity)'})")
    print(f"  row claims tenant_id=VICTIM in all cases")
    print(f"  INSERT result: {'SUCCEEDED' if succeeded else 'BLOCKED'}")
    if err:
        print(f"  error: {err[:200]}")

    poisoned = False
    if succeeded:
        res = search_rls_trusted(conn, VICTIM, qvec, k=K)
        poisoned = any(r[0] == poison_id for r in res)
        print(f"  -> served back under correctly-scoped victim read: "
              f"{'YES -- POISONING CONFIRMED' if poisoned else 'no (not in top-k)'}")

    conn_super.execute("DELETE FROM chunks WHERE id = %s;", (poison_id,))
    return {"label": label, "identity_forged": identity_forged,
            "insert_succeeded": succeeded, "poisoning_confirmed": poisoned}

if __name__ == "__main__":
    conn = get_conn(as_app_user=True)
    conn_super = get_conn()

    results = []

    print("=" * 70)
    print("AXIS 2 = LAX (split policies, WITH CHECK (true) on FOR INSERT)")
    print("=" * 70)
    setup_split_policies_vulnerable(conn_super)

    # TOP-RIGHT CELL (the paper's core claim): authenticated identity,
    # lax write policy. Attacker's session correctly reflects their OWN
    # tenant -- no forgery anywhere -- yet the row still lands as victim.
    results.append(run_cell(conn, conn_super,
        "CELL: authenticated + lax policy (THE PAPER'S CLAIM)", identity_forged=False))

    # Degenerate case: identity ALSO forged, lax policy. Everything falls.
    results.append(run_cell(conn, conn_super,
        "CELL: forged identity + lax policy (degenerate)", identity_forged=True))

    print("\n" + "=" * 70)
    print("AXIS 2 = STRICT (split policies, WITH CHECK validates tenant_id)")
    print("=" * 70)
    setup_split_policies_fixed(conn_super)

    # Mitigation cell: authenticated identity, strict policy. No poisoning.
    results.append(run_cell(conn, conn_super,
        "CELL: authenticated + strict policy (mitigation)", identity_forged=False))

    # Read-side-only cell: identity forged, but write policy strict.
    # Row-level write is blocked; any leakage here would have to come
    # from the READ side (RLS session forgery), not from poisoning.
    results.append(run_cell(conn, conn_super,
        "CELL: forged identity + strict policy", identity_forged=True))

    print("\n" + "=" * 70)
    print("EMPIRICAL CHECK: omitted WITH CHECK clause (not WITH CHECK (true))")
    print("=" * 70)
    ddl_outcome = try_create_omitted_check_policy(conn_super)
    print(f"Result: {ddl_outcome}")

    if "ACCEPTED" in ddl_outcome:
        print("\nSince omission is ACCEPTED, testing the authenticated+omitted-check cell")
        print("(this is the MOST realistic bug: a developer simply forgot the clause,")
        print("as opposed to deliberately writing WITH CHECK (true)):")
        omitted_result = run_cell(conn, conn_super,
            "CELL: authenticated + OMITTED (not explicit true) write check", identity_forged=False)
        results.append(omitted_result)

    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    import pandas as pd
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    df.to_csv("../results/write_path_2x2.csv", index=False)
    with open("../results/write_path_ddl_check.txt", "w") as f:
        f.write(f"Omitted WITH CHECK clause on FOR INSERT policy: {ddl_outcome}\n")
    print("\nSaved results/write_path_2x2.csv and results/write_path_ddl_check.txt")

    # Restore the policy to the strict/safe version so the DB is left
    # in a known-good state for any subsequent experiment runs.
    setup_split_policies_fixed(conn_super)
    print("\nPolicy restored to strict (safe) state.")
