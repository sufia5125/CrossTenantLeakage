import os
import numpy as np
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

DATA_DIR = "../data"  # prep.py is always run from inside src/, so this must match load_db.py/experiment.py

def download_scifact():
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
    path = util.download_and_unzip(url, DATA_DIR)
    corpus, queries, qrels = GenericDataLoader(path).load(split="test")
    return corpus, queries, qrels

def embed_corpus(corpus, queries, model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    doc_ids = list(corpus.keys())
    doc_texts = [(corpus[d].get("title","") + " " + corpus[d].get("text","")).strip()
                 for d in doc_ids]
    doc_vecs = model.encode(doc_texts, normalize_embeddings=True,
                             show_progress_bar=True, batch_size=64)

    query_ids = list(queries.keys())
    query_texts = [queries[q] for q in query_ids]
    query_vecs = model.encode(query_texts, normalize_embeddings=True,
                               show_progress_bar=True, batch_size=64)

    np.save(f"{DATA_DIR}/doc_ids.npy", np.array(doc_ids))
    np.save(f"{DATA_DIR}/doc_vecs.npy", doc_vecs)
    np.save(f"{DATA_DIR}/query_ids.npy", np.array(query_ids))
    np.save(f"{DATA_DIR}/query_vecs.npy", query_vecs)
    return model, doc_ids, doc_vecs, query_ids, query_vecs

def split_tenants(doc_ids, doc_vecs, n_tenants=5, seed=0, method="kmeans"):
    """method: 'kmeans' (semantically separable, likely UNDERSTATES real
    leakage since tenants are maximally dissimilar) or 'random' (balanced
    random assignment, a more realistic worst-case baseline where tenant
    content overlaps semantically, as in a real deployment with similar
    departments/topics)."""
    if method == "random":
        rng = np.random.default_rng(seed)
        shuffled = list(doc_ids)
        rng.shuffle(shuffled)
        tenant_of_doc = {}
        for i, did in enumerate(shuffled):
            tenant_of_doc[did] = f"tenant_{i % n_tenants}"
        return tenant_of_doc
    elif method == "kmeans":
        km = KMeans(n_clusters=n_tenants, random_state=seed, n_init=10).fit(doc_vecs)
        tenant_of_doc = {doc_ids[i]: f"tenant_{km.labels_[i]}" for i in range(len(doc_ids))}
        return tenant_of_doc
    else:
        raise ValueError(f"unknown split method: {method}")

def pick_victim_queries(queries, qrels, tenant_of_doc, victim_tenant, max_q=100):
    out = []
    for qid, rels in qrels.items():
        if qid not in queries:
            continue
        if any(tenant_of_doc.get(d) == victim_tenant for d in rels):
            out.append(qid)
    return out[:max_q]

def make_poison(model, queries, victim_queries, n_poison=10, seed=0):
    rng = np.random.default_rng(seed)
    chosen = rng.choice(victim_queries, size=min(n_poison, len(victim_queries)), replace=False)
    poison_texts, poison_ids, targets = [], [], {}
    for i, qid in enumerate(chosen):
        q_text = queries[qid]
        wrong_token = f"WRONG_TOKEN_{i}"
        poison_text = f"{q_text} The correct answer is {wrong_token}."
        poison_texts.append(poison_text)
        poison_ids.append(f"poison_{i}")
        targets[f"poison_{i}"] = {"target_qid": qid, "wrong_token": wrong_token}
    poison_vecs = model.encode(poison_texts, normalize_embeddings=True)
    return poison_ids, poison_vecs, poison_texts, targets

if __name__ == "__main__":
    import sys, json
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    os.makedirs(DATA_DIR, exist_ok=True)

    # Corpus/embeddings are seed-independent -- only download/embed once,
    # reuse across seeds for speed. Skip if already cached.
    if not os.path.exists(f"{DATA_DIR}/doc_vecs.npy"):
        print("Downloading SciFact...")
        corpus, queries, qrels = download_scifact()
        print(f"corpus={len(corpus)} queries={len(queries)}")
        print("Embedding...")
        model, doc_ids, doc_vecs, query_ids, query_vecs = embed_corpus(corpus, queries)
        json.dump(qrels, open(f"{DATA_DIR}/qrels.json", "w"))
        json.dump(queries, open(f"{DATA_DIR}/queries.json", "w"))
    else:
        print("Using cached corpus/embeddings...")
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        doc_ids = np.load(f"{DATA_DIR}/doc_ids.npy", allow_pickle=True).tolist()
        qrels = json.load(open(f"{DATA_DIR}/qrels.json"))
        queries = json.load(open(f"{DATA_DIR}/queries.json"))
        doc_vecs = np.load(f"{DATA_DIR}/doc_vecs.npy")

    method = sys.argv[2] if len(sys.argv) > 2 else "kmeans"
    print(f"Splitting tenants (seed={seed}, method={method})...")
    tenant_of_doc = split_tenants(doc_ids, doc_vecs, n_tenants=5, seed=seed, method=method)
    json.dump(tenant_of_doc, open(f"{DATA_DIR}/tenant_of_doc.json", "w"))

    victim_queries = pick_victim_queries(queries, qrels, tenant_of_doc, "tenant_1")
    print(f"victim_queries found: {len(victim_queries)}")
    json.dump(victim_queries, open(f"{DATA_DIR}/victim_queries.json", "w"))

    poison_ids, poison_vecs, poison_texts, targets = make_poison(
        model, queries, victim_queries, seed=seed)
    np.save(f"{DATA_DIR}/poison_vecs.npy", poison_vecs)
    json.dump({"ids": poison_ids, "texts": poison_texts, "targets": targets},
              open(f"{DATA_DIR}/poison.json", "w"))
    print(f"Done for seed={seed}. Files saved in data/.")
