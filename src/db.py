import psycopg
from pgvector.psycopg import register_vector
from pgvector import Vector

DSN_SUPERUSER = "postgresql://postgres:devpass@localhost:5432/postgres"
DSN_APPUSER   = "postgresql://app_user:devpass@localhost:5432/postgres"

def get_conn(as_app_user=False):
    dsn = DSN_APPUSER if as_app_user else DSN_SUPERUSER
    conn = psycopg.connect(dsn, autocommit=True)
    register_vector(conn)
    return conn

def reset_schema(conn):
    conn.execute("DROP TABLE IF EXISTS chunks CASCADE;")
    conn.execute("DROP SCHEMA IF EXISTS t_attacker CASCADE;")
    conn.execute("DROP SCHEMA IF EXISTS t_victim CASCADE;")
    conn.execute("DROP ROLE IF EXISTS app_user;")
    conn.execute("""
        CREATE TABLE chunks (
            id text PRIMARY KEY,
            tenant_id text NOT NULL,
            is_poison boolean NOT NULL DEFAULT false,
            embedding vector(384) NOT NULL
        );
    """)
    conn.execute("CREATE INDEX ON chunks (tenant_id);")

def build_hnsw_index(conn):
    conn.execute("DROP INDEX IF EXISTS chunks_embedding_idx;")
    conn.execute("""
        CREATE INDEX chunks_embedding_idx ON chunks
        USING hnsw (embedding vector_cosine_ops);
    """)

def insert_chunks(conn, rows):
    # rows: list of (id, tenant_id, is_poison, embedding_list)
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO chunks (id, tenant_id, is_poison, embedding) "
            "VALUES (%s,%s,%s,%s) ON CONFLICT (id) DO NOTHING",
            rows
        )

def setup_schema_mechanism(conn):
    """Mechanism 2: separate schema+table per tenant (mirror of 'chunks')."""
    conn.execute("CREATE SCHEMA IF NOT EXISTS t_attacker;")
    conn.execute("CREATE SCHEMA IF NOT EXISTS t_victim;")
    for schema in ["t_attacker", "t_victim"]:
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {schema}.chunks (
                id text PRIMARY KEY,
                tenant_id text NOT NULL,
                is_poison boolean NOT NULL DEFAULT false,
                embedding vector(384) NOT NULL
            );
        """)
        conn.execute(f"""
            CREATE INDEX IF NOT EXISTS {schema}_embedding_idx
            ON {schema}.chunks USING hnsw (embedding vector_cosine_ops);
        """)

def populate_schema_mechanism(conn, tenant_of_doc, doc_ids, doc_vecs,
                                attacker="tenant_0", victim="tenant_1"):
    """Copy each tenant's docs into its own schema table (mechanism 2 data)."""
    attacker_rows, victim_rows = [], []
    for i, did in enumerate(doc_ids):
        tid = tenant_of_doc.get(did)
        if tid == attacker:
            attacker_rows.append((str(did), tid, False, doc_vecs[i].tolist()))
        elif tid == victim:
            victim_rows.append((str(did), tid, False, doc_vecs[i].tolist()))
    with conn.cursor() as cur:
        if attacker_rows:
            cur.executemany(
                "INSERT INTO t_attacker.chunks (id,tenant_id,is_poison,embedding) "
                "VALUES (%s,%s,%s,%s) ON CONFLICT (id) DO NOTHING", attacker_rows)
        if victim_rows:
            cur.executemany(
                "INSERT INTO t_victim.chunks (id,tenant_id,is_poison,embedding) "
                "VALUES (%s,%s,%s,%s) ON CONFLICT (id) DO NOTHING", victim_rows)
    print(f"Populated schema mechanism: attacker={len(attacker_rows)} victim={len(victim_rows)} rows")

def setup_rls(conn):
    """Mechanism 3: row-level security on the shared 'chunks' table."""
    conn.execute("ALTER TABLE chunks ENABLE ROW LEVEL SECURITY;")
    conn.execute("ALTER TABLE chunks FORCE ROW LEVEL SECURITY;")
    conn.execute("DROP POLICY IF EXISTS tenant_iso ON chunks;")
    conn.execute("""
        CREATE POLICY tenant_iso ON chunks
        USING (tenant_id = current_setting('app.tenant_id', true));
    """)
    conn.execute("DROP ROLE IF EXISTS app_user;")
    conn.execute("CREATE ROLE app_user LOGIN PASSWORD 'devpass';")
    conn.execute("GRANT SELECT, INSERT ON chunks TO app_user;")

# ---------- SEARCH FUNCTIONS (correct configuration) ----------
# NOTE: qvec is wrapped in Vector(...) because outside of an INSERT
# (where the destination column type gives Postgres a hint), a raw
# Python list gets sent as float8[] and `<=>` has no such overload.

def search_shared_filter(conn, tenant_id, qvec, k=5):
    return conn.execute(
        "SELECT id, tenant_id, is_poison FROM chunks "
        "WHERE tenant_id = %s ORDER BY embedding <=> %s LIMIT %s",
        (tenant_id, Vector(qvec), k)
    ).fetchall()

def search_schema(conn, tenant_id, qvec, k=5):
    schema = "t_attacker" if tenant_id == "tenant_0" else "t_victim"
    return conn.execute(
        f"SELECT id, tenant_id, is_poison FROM {schema}.chunks "
        f"ORDER BY embedding <=> %s LIMIT %s",
        (Vector(qvec), k)
    ).fetchall()

def search_rls(conn, tenant_id, qvec, k=5):
    # SET does not accept parameters ($1) -- use set_config() instead, which does.
    conn.execute("SELECT set_config('app.tenant_id', %s, false);", (tenant_id,))
    return conn.execute(
        "SELECT id, tenant_id, is_poison FROM chunks "
        "ORDER BY embedding <=> %s LIMIT %s",
        (Vector(qvec), k)
    ).fetchall()

# ---------- FAULT-INJECTED SEARCH FUNCTIONS ----------

def search_f1_filter_omitted(conn, tenant_id, qvec, k=5):
    # Buggy secondary path: forgets WHERE tenant_id = ...
    return conn.execute(
        "SELECT id, tenant_id, is_poison FROM chunks "
        "ORDER BY embedding <=> %s LIMIT %s",
        (Vector(qvec), k)
    ).fetchall()

def search_f2_client_controlled(conn, client_supplied_tenant_id, qvec, k=5):
    # Trusts whatever tenant_id the caller passes (e.g. a URL/body param)
    # instead of deriving identity from verified auth. The vulnerability
    # only manifests when the caller supplies an ID that is NOT their own
    # authorized tenant -- e.g. an attacker claiming to be the victim.
    return search_shared_filter(conn, client_supplied_tenant_id, qvec, k)

def search_f3_global_corpus(conn, tenant_id, qvec, k=5):
    # Shared "global" tenant is always included in results
    return conn.execute(
        "SELECT id, tenant_id, is_poison FROM chunks "
        "WHERE tenant_id = %s OR tenant_id = 'global' "
        "ORDER BY embedding <=> %s LIMIT %s",
        (tenant_id, Vector(qvec), k)
    ).fetchall()

def search_f5_null_falls_through(conn, tenant_id, qvec, k=5):
    # if tenant_id in (None, "", "*"): add_filter() -- these "unset /
    # wildcard" sentinel values silently skip filtering instead of being
    # rejected. None and "" are falsy in Python (would be caught by a
    # naive `if tenant_id:` check); "*" is NOT falsy, so a naive check
    # would miss it -- this is the more subtle, realistic version of the
    # bug (a common wildcard sentinel slipping past a falsy-only guard).
    UNSET_SENTINELS = {None, "", "*"}
    if tenant_id not in UNSET_SENTINELS:
        return search_shared_filter(conn, tenant_id, qvec, k)
    return conn.execute(
        "SELECT id, tenant_id, is_poison FROM chunks "
        "ORDER BY embedding <=> %s LIMIT %s",
        (Vector(qvec), k)
    ).fetchall()

# ---------- WRITE-PATH POISONING (CWE-639 on the INSERT path) ----------

def attacker_inserts_as_victim(conn, poison_id, claimed_victim_tenant,
                                 attacker_true_tenant, poison_vec, is_poison=True):
    """THE CORE ATTACK: the attacker, authenticated only as their own
    tenant (attacker_true_tenant), attempts to INSERT a row whose
    tenant_id COLUMN claims to be the victim's. This tests whether RLS
    policy enforcement (or application-layer write scoping) actually
    prevents a row from being written into another tenant's partition,
    as opposed to merely filtering reads.

    Threat model: attacker has a legitimate write path into the shared
    `chunks` table (e.g. via a document-upload API) but should only be
    able to set tenant_id to their own value. If this INSERT succeeds,
    the row becomes a PERMANENT, INDISTINGUISHABLE member of the
    victim's own tenant -- not merely "retrieved due to a missing
    filter" (the read-side faults), but genuinely and durably written
    as if it were the victim's own content. This is the actual
    poisoning primitive the paper's title claims.

    Returns: (succeeded: bool, error_message: str or None)
    """
    conn.execute("SELECT set_config('app.tenant_id', %s, false);",
                 (attacker_true_tenant,))
    try:
        conn.execute(
            "INSERT INTO chunks (id, tenant_id, is_poison, embedding) "
            "VALUES (%s, %s, %s, %s)",
            (poison_id, claimed_victim_tenant, is_poison, Vector(poison_vec))
        )
        return True, None
    except Exception as e:
        return False, str(e)

def add_explicit_with_check_policy(conn):
    """MITIGATION: explicitly add a WITH CHECK clause that re-validates
    the session identity on every INSERT/UPDATE, independent of whatever
    the implicit USING-as-WITH-CHECK default does. This makes the write
    protection explicit and auditable rather than relying on an easily
    overlooked Postgres default."""
    conn.execute("DROP POLICY IF EXISTS tenant_iso ON chunks;")
    conn.execute("""
        CREATE POLICY tenant_iso ON chunks
        USING (tenant_id = current_setting('app.tenant_id', true))
        WITH CHECK (tenant_id = current_setting('app.tenant_id', true));
    """)

# ---------- RLS THREAT-MODEL VARIANTS ----------

def search_rls_trusted(conn, tenant_id, qvec, k=5):
    """RLS under the INTENDED threat model: app.tenant_id is set only by
    trusted middleware, never by the end caller. This is the 'correct
    config' condition for RLS."""
    conn.execute("SELECT set_config('app.tenant_id', %s, false);", (tenant_id,))
    return conn.execute(
        "SELECT id, tenant_id, is_poison FROM chunks "
        "ORDER BY embedding <=> %s LIMIT %s",
        (Vector(qvec), k)
    ).fetchall()

def search_rls_attacker_sets_session(conn, claimed_tenant_id, qvec, k=5):
    """F_RLS1 -- IDENTITY FORGERY TEST, not a leakage-via-poison test.
    Simulates the ATTACKER (connected as app_user, with no real
    authorization as the victim) self-asserting app.tenant_id = victim's
    id. This tests whether RLS alone prevents unauthorized reads when
    session identity is not independently verified. Expected: if this
    succeeds in returning victim-owned rows at all, RLS has been
    bypassed via identity forgery -- regardless of whether any
    individual row happens to be poison."""
    conn.execute("SELECT set_config('app.tenant_id', %s, false);",
                 (claimed_tenant_id,))
    return conn.execute(
        "SELECT id, tenant_id, is_poison FROM chunks "
        "ORDER BY embedding <=> %s LIMIT %s",
        (Vector(qvec), k)
    ).fetchall()

# ---------- 2x2: IDENTITY AUTHENTICATION x WRITE POLICY STRICTNESS ----------
# These two axes were previously conflated in a single test. This makes
# each axis an independent, explicitly labeled experimental factor.
#
# Axis 1 (identity_forged: bool): whether app.tenant_id, at the moment
#   of INSERT, has been set to the attacker's OWN true tenant (False --
#   "authenticated") or to the VICTIM's tenant (True -- "forged"), with
#   the row's tenant_id column value held constant at VICTIM in all
#   cells (the attacker always attempts to write AS the victim).
#
# Axis 2 (write_policy: "strict" | "lax"): whether the FOR INSERT
#   policy's WITH CHECK clause actually validates tenant_id against the
#   session identity ("strict") or accepts any row ("lax", WITH CHECK
#   (true)).

def set_session_identity(conn, tenant_id):
    conn.execute("SELECT set_config('app.tenant_id', %s, false);", (tenant_id,))

def attacker_insert_2x2(conn, poison_id, poison_vec, identity_forged, attacker_tenant, victim_tenant):
    """The row ALWAYS claims tenant_id=victim_tenant in its column value
    (the attacker is always trying to write as the victim). What varies
    is ONLY whether the session's identity claim matches that forgery
    (identity_forged=True -> session also claims victim) or is honest
    (identity_forged=False -> session correctly reflects the attacker's
    own true tenant, decoupling write-authorization from authentication)."""
    session_identity = victim_tenant if identity_forged else attacker_tenant
    set_session_identity(conn, session_identity)
    try:
        conn.execute(
            "INSERT INTO chunks (id, tenant_id, is_poison, embedding) "
            "VALUES (%s, %s, %s, %s)",
            (poison_id, victim_tenant, True, Vector(poison_vec))
        )
        return True, None
    except Exception as e:
        return False, str(e)

# ---------- SPLIT READ/WRITE POLICY (realistic misconfiguration) ----------

def setup_split_policies_vulnerable(conn):
    """Common real-world refactor: a developer separates the read and
    write policies (e.g. to add different logic for each) using FOR
    SELECT and FOR INSERT instead of one FOR ALL policy. Critically,
    a FOR INSERT policy has NO 'visibility' clause to reuse -- Postgres
    requires WITH CHECK to be specified explicitly for INSERT-only
    policies, there is no implicit fallback. If the developer forgets
    this (very plausible: USING is the only clause most tutorials show),
    the INSERT policy ends up with no WITH CHECK, which Postgres treats
    as 'no restriction on inserted rows' -- any tenant_id is accepted."""
    conn.execute("DROP POLICY IF EXISTS tenant_iso ON chunks;")
    conn.execute("DROP POLICY IF EXISTS tenant_read ON chunks;")
    conn.execute("DROP POLICY IF EXISTS tenant_write ON chunks;")
    conn.execute("""
        CREATE POLICY tenant_read ON chunks FOR SELECT
        USING (tenant_id = current_setting('app.tenant_id', true));
    """)
    conn.execute("""
        CREATE POLICY tenant_write ON chunks FOR INSERT
        WITH CHECK (true);
    """)
    # NOTE: the line above is the realistic bug -- a developer writes
    # "WITH CHECK (true)" (or omits WITH CHECK, which Postgres then
    # defaults to TRUE for INSERT policies with none specified) because
    # they only thought about read-side filtering when adding this policy.

def setup_split_policies_fixed(conn):
    """The corrected version of the above: WITH CHECK is explicitly
    tied to tenant identity on the INSERT policy too."""
    conn.execute("DROP POLICY IF EXISTS tenant_iso ON chunks;")
    conn.execute("DROP POLICY IF EXISTS tenant_read ON chunks;")
    conn.execute("DROP POLICY IF EXISTS tenant_write ON chunks;")
    conn.execute("""
        CREATE POLICY tenant_read ON chunks FOR SELECT
        USING (tenant_id = current_setting('app.tenant_id', true));
    """)
    conn.execute("""
        CREATE POLICY tenant_write ON chunks FOR INSERT
        WITH CHECK (tenant_id = current_setting('app.tenant_id', true));
    """)

# ---------- MITIGATIONS ----------

def search_failclosed(search_fn, conn, tenant_id, qvec, k=5):
    if not tenant_id or tenant_id in {"*", "global"}:
        raise ValueError("invalid tenant_id")
    return search_fn(conn, tenant_id, qvec, k)
