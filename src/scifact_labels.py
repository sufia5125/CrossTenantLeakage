"""
Loads SciFact's ORIGINAL claim-verification labels (SUPPORT / CONTRADICT /
NOT_ENOUGH_INFO), which are separate from BEIR's repackaged binary
relevance qrels used elsewhere in this pipeline. Needed for the
label-flip ASR metric (see ../results/METRIC_DEFINITIONS.md).

SciFact's original release ships claims_train.jsonl / claims_dev.jsonl
with fields: id, claim, evidence (doc_id -> [{sentences, label}]),
cited_doc_ids. The BEIR queries.jsonl id for a claim corresponds to
this original claim's "id" field (BEIR preserves SciFact's claim ids
as query ids -- this is verified empirically below, not assumed).
"""
import json
import os
import urllib.request

SCIFACT_CLAIMS_URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
LABELS_DIR = "../data/scifact_labels"

def download_scifact_claims():
    """Downloads and extracts the original SciFact release (separate
    from the BEIR-repackaged version already in ../data/scifact/)."""
    os.makedirs(LABELS_DIR, exist_ok=True)
    tar_path = f"{LABELS_DIR}/data.tar.gz"
    if not os.path.exists(tar_path):
        print("Downloading original SciFact release (for claim labels)...")
        urllib.request.urlretrieve(SCIFACT_CLAIMS_URL, tar_path)
    import tarfile
    with tarfile.open(tar_path) as tar:
        tar.extractall(LABELS_DIR)
    print(f"Extracted to {LABELS_DIR}")

def load_claim_labels():
    """Returns {claim_id (str): {'claim_text': str, 'label': str or None,
    'evidence_doc_ids': [str]}}.

    label is the MAJORITY/first-found stance across all evidence docs
    for that claim: 'SUPPORT', 'CONTRADICT', or None if the claim has no
    evidence annotation in this split (common for NOT_ENOUGH_INFO claims,
    which SciFact records with empty evidence rather than an explicit
    label field -- this is a known quirk of the raw release, handled
    explicitly below rather than silently mis-parsed)."""
    claims = {}
    # SciFact splits claims across train/dev/test; dev has evidence
    # labels, test does not (held out for the original leaderboard).
    # We use train + dev, since those have usable ground truth.
    for split in ["claims_train.jsonl", "claims_dev.jsonl"]:
        path = f"{LABELS_DIR}/data/{split}"
        if not os.path.exists(path):
            print(f"WARNING: {path} not found -- skipping")
            continue
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                claim_id = str(row["id"])
                evidence = row.get("evidence", {})
                label = None
                evidence_doc_ids = list(evidence.keys())
                if evidence:
                    # take the label from the first doc's first evidence group
                    first_doc = next(iter(evidence.values()))
                    if first_doc:
                        label = first_doc[0].get("label")  # 'SUPPORT' or 'CONTRADICT'
                claims[claim_id] = {
                    "claim_text": row["claim"],
                    "label": label,  # SUPPORT / CONTRADICT / None (= NOT_ENOUGH_INFO or unlabeled)
                    "evidence_doc_ids": evidence_doc_ids,
                }
    return claims

def verify_id_alignment():
    """Confirms BEIR's queries.jsonl ids actually match SciFact's
    original claim ids, rather than assuming it. Prints overlap stats."""
    beir_query_ids = set()
    with open("../data/scifact/queries.jsonl") as f:
        for line in f:
            row = json.loads(line)
            beir_query_ids.add(str(row["_id"]))

    claims = load_claim_labels()
    claim_ids = set(claims.keys())

    overlap = beir_query_ids & claim_ids
    print(f"BEIR query ids: {len(beir_query_ids)}")
    print(f"Original SciFact claim ids (train+dev): {len(claim_ids)}")
    print(f"Overlap: {len(overlap)} "
          f"({100*len(overlap)/max(len(beir_query_ids),1):.1f}% of BEIR queries have a label)")
    labeled_overlap = sum(1 for qid in overlap if claims[qid]["label"] is not None)
    print(f"Of overlap, {labeled_overlap} have a non-null SUPPORT/CONTRADICT label")
    return overlap, claims

if __name__ == "__main__":
    download_scifact_claims()
    overlap, claims = verify_id_alignment()
    if len(overlap) == 0:
        print("\nALIGNMENT FAILED -- BEIR query ids do not match original SciFact")
        print("claim ids directly. The ASR script will need an alternate join")
        print("(e.g. via claim text matching) -- do not proceed until this is")
        print("resolved and re-verified.")
    else:
        print(f"\nAlignment OK. {len(overlap)} claims usable for label-flip ASR.")
