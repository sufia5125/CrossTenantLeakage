import numpy as np
# See ../results/METRIC_DEFINITIONS.md for the formal definition, denominator
# convention, and rationale for every metric in this file. Keep that file
# and these docstrings in sync if either changes.

def ctrr_poison(results_per_query, victim_tenant):
    hits = [any(r[2] and r[1] != victim_tenant for r in res) for res in results_per_query]
    return float(np.mean(hits))

def ctrr_any(results_per_query, victim_tenant):
    hits = [any(r[1] != victim_tenant for r in res) for res in results_per_query]
    return float(np.mean(hits))

def poison_precision_at_k(results_per_query, k):
    vals = [sum(1 for r in res if r[2]) / k for res in results_per_query]
    return float(np.mean(vals))

def shortfall(results_per_query, k):
    vals = [len(res) < k for res in results_per_query]
    return float(np.mean(vals))

def bootstrap_ci(values, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    values = np.array(values)
    if len(values) == 0:
        return (float("nan"), float("nan"))
    boots = [rng.choice(values, size=len(values), replace=True).mean() for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(lo), float(hi)
