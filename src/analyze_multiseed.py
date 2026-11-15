import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("../results/main_multiseed.csv")

# Per-seed means already ARE one row per (mechanism, fault, seed) since
# each seed run queries the full victim_queries set once. So the
# "seed-level mean" is just ctrr_poison itself per row; we bootstrap
# resample ACROSS SEEDS (5 values per condition), not across queries,
# per the reviewer's point about query correlation within a seed.

def seed_level_ci(group, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    vals = group["ctrr_poison"].values
    if len(vals) < 2:
        return pd.Series({"mean": vals.mean() if len(vals) else np.nan,
                           "ci_lo": np.nan, "ci_hi": np.nan, "n_seeds": len(vals)})
    boots = [rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return pd.Series({"mean": vals.mean(), "ci_lo": lo, "ci_hi": hi, "n_seeds": len(vals)})

summary = df.groupby(["mechanism", "fault"]).apply(seed_level_ci, include_groups=False).reset_index()
summary.to_csv("../results/leakage_summary_with_ci.csv", index=False)
print(summary)

pivot_mean = summary.pivot(index="mechanism", columns="fault", values="mean")
fig, ax = plt.subplots(figsize=(8, 5))
pivot_mean.plot(kind="bar", ax=ax)
ax.set_ylabel("Cross-Tenant Retrieval Rate (poison), mean across 5 seeds")
ax.set_title("Leakage by mechanism and injected fault (5-seed mean)")
plt.tight_layout()
plt.savefig("../results/leakage_chart_multiseed.png", dpi=150)
print("Saved results/leakage_summary_with_ci.csv and results/leakage_chart_multiseed.png")
