"""
Query-level analysis, addressing three review points at once:
1. Bootstrap CI at the QUERY level (pooled across seeds), not the
   degenerate 5-seed-mean level.
2. Report targeted vs untargeted queries separately.
3. Compare kmeans (separable) vs random (overlapping) tenant splits.
"""
import pandas as pd
import numpy as np

df = pd.read_csv("../results/raw_queries.csv")

def bootstrap_ci_proportion(hits, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    hits = np.asarray(hits, dtype=float)
    n = len(hits)
    if n == 0:
        return np.nan, np.nan, np.nan, 0
    point = hits.mean()
    boots = rng.choice(hits, size=(n_boot, n), replace=True).mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, lo, hi, n

def summarize(group):
    poison_point, poison_lo, poison_hi, n = bootstrap_ci_proportion(group["poison_hit"])
    any_point, any_lo, any_hi, _ = bootstrap_ci_proportion(group["any_hit"])
    return pd.Series({
        "ctrr_poison_mean": poison_point, "ctrr_poison_ci_lo": poison_lo, "ctrr_poison_ci_hi": poison_hi,
        "ctrr_any_mean": any_point, "ctrr_any_ci_lo": any_lo, "ctrr_any_ci_hi": any_hi,
        "n_query_observations": n,
    })

print("=== Overall (all queries, query-level bootstrap, pooled across seeds) ===")
overall = df.groupby(["split_method", "mechanism", "fault"]).apply(summarize, include_groups=False).reset_index()
overall.to_csv("../results/leakage_query_level_ci.csv", index=False)
print(overall.to_string(index=False))

print("\n=== Targeted vs Untargeted queries (kmeans split, shared_filter mechanism) ===")
tu = df[(df.split_method == "kmeans") & (df.mechanism == "shared_filter")]
tu_summary = tu.groupby(["fault", "is_targeted"]).apply(summarize, include_groups=False).reset_index()
tu_summary.to_csv("../results/leakage_targeted_vs_untargeted.csv", index=False)
print(tu_summary.to_string(index=False))

print("\n=== KMeans vs Random split comparison (shared_filter mechanism) ===")
split_compare = df[df.mechanism == "shared_filter"].groupby(
    ["split_method", "fault"]).apply(summarize, include_groups=False).reset_index()
split_compare.to_csv("../results/leakage_split_method_comparison.csv", index=False)
print(split_compare.to_string(index=False))

print("\nSaved: leakage_query_level_ci.csv, leakage_targeted_vs_untargeted.csv, "
      "leakage_split_method_comparison.csv")
