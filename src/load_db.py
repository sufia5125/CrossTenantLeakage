import json
import numpy as np
from db import get_conn, reset_schema, build_hnsw_index, insert_chunks, \
                setup_schema_mechanism, setup_rls, search_shared_filter, \
                populate_schema_mechanism

def load_everything(conn, attacker="tenant_0", victim="tenant_1"):
    doc_ids = np.load("../data/doc_ids.npy", allow_pickle=True)
    doc_vecs = np.load("../data/doc_vecs.npy")
    tenant_of_doc = json.load(open("../data/tenant_of_doc.json"))
    poison = json.load(open("../data/poison.json"))
    poison_vecs = np.load("../data/poison_vecs.npy")

    rows = []
    for i, did in enumerate(doc_ids):
        tid = tenant_of_doc.get(did, "tenant_0")
        rows.append((str(did), tid, False, doc_vecs[i].tolist()))
    insert_chunks(conn, rows)

    # E0 calibration: poison placed in the VICTIM's own tenant (temporary,
    # removed after calibration passes -- see __main__ below)
    calib_rows = []
    for i, pid in enumerate(poison["ids"]):
        calib_rows.append((pid + "_calib", victim, True, poison_vecs[i].tolist()))
    insert_chunks(conn, calib_rows)

    build_hnsw_index(conn)
    return tenant_of_doc, poison, poison_vecs, doc_ids, doc_vecs

def run_e0_calibration(conn, poison, victim="tenant_1", k=5):
    queries_vecs = np.load("../data/query_vecs.npy")
    query_ids = np.load("../data/query_ids.npy", allow_pickle=True).tolist()
    hits_total, checked = 0, 0
    for pid, meta in poison["targets"].items():
        qid = meta["target_qid"]
        if qid not in query_ids:
            continue
        qi = query_ids.index(qid)
        qvec = queries_vecs[qi].tolist()
        results = search_shared_filter(conn, victim, qvec, k=k)
        checked += 1
        if any(r[2] for r in results):  # is_poison column
            hits_total += 1
    rate = hits_total / checked if checked else 0
    print(f"E0 calibration: poison precision@{k} = {rate:.2f} ({hits_total}/{checked})")
    return rate

def insert_attacker_poison(conn, poison, poison_vecs, attacker="tenant_0"):
    """Insert the REAL attacker-owned poison used by every fault-injection
    experiment (F1, F3, F5, and the positive-control tests). This must
    run on every fresh load, or those experiments will show false
    zero-leakage because there is no attacker poison left to leak."""
    rows = []
    for i, pid in enumerate(poison["ids"]):
        rows.append((pid, attacker, True, poison_vecs[i].tolist()))
    insert_chunks(conn, rows)
    print(f"Inserted {len(rows)} attacker-owned poison chunks under tenant={attacker}")
    return rows

def insert_global_poison(conn, poison, poison_vecs):
    """Insert poison under the 'global' pseudo-tenant used by F3."""
    rows = []
    for i, pid in enumerate(poison["ids"]):
        rows.append((pid + "_global", "global", True, poison_vecs[i].tolist()))
    insert_chunks(conn, rows)
    print(f"Inserted {len(rows)} poison chunks under tenant=global (for F3)")
    return rows

def populate_schema_poison(conn, poison, poison_vecs, attacker="tenant_0"):
    """Insert attacker poison into the attacker's own schema table too."""
    rows = []
    for i, pid in enumerate(poison["ids"]):
        rows.append((pid + "_schema", attacker, True, poison_vecs[i].tolist()))
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO t_attacker.chunks (id,tenant_id,is_poison,embedding) "
            "VALUES (%s,%s,%s,%s) ON CONFLICT (id) DO NOTHING", rows)
    print(f"Inserted {len(rows)} attacker poison chunks into t_attacker.chunks")

if __name__ == "__main__":
    conn = get_conn()
    print("Resetting schema...")
    reset_schema(conn)
    print("Loading data...")
    tenant_of_doc, poison, poison_vecs, doc_ids, doc_vecs = load_everything(conn)

    print("Running E0 calibration...")
    rate = run_e0_calibration(conn, poison)
    if rate < 0.3:
        print("WARNING: poison signal is weak. Reconsider poison text before proceeding.")
    else:
        print("E0 passed: poison is detectable. Safe to proceed.")
        print("Removing calibration-only poison rows (they must not contaminate main experiments)...")
        conn.execute("DELETE FROM chunks WHERE id LIKE '%_calib';")

    print("Inserting real attacker poison (required for F1/F3/F5 and positive controls)...")
    insert_attacker_poison(conn, poison, poison_vecs)
    print("Inserting global-tenant poison (required for F3)...")
    insert_global_poison(conn, poison, poison_vecs)

    print("Setting up mechanism 2 (schema)...")
    setup_schema_mechanism(conn)
    populate_schema_mechanism(conn, tenant_of_doc, doc_ids, doc_vecs)
    populate_schema_poison(conn, poison, poison_vecs)

    print("Setting up mechanism 3 (RLS)...")
    setup_rls(conn)

    print("All mechanisms ready.")
