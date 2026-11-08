"""Post-hoc sensitivity / power analysis for the CNO cohort (n=11 vs 9).

Answers the reviewer question: 'is the negative CNO result simply underpowered?'
We report (i) the observed standardised effect size, (ii) the smallest effect
size the cohort could have detected with 80% power, and (iii) the power the
cohort actually had to detect the effect seen in bacterial osteomyelitis.
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

s = pd.read_csv(os.path.join(RES, "A_all_scores.csv"))
s = s.rename(columns={"Unnamed: 0": "dataset", "Unnamed: 1": "sample"})
cno = s[s.dataset == "GSE187429"]
g1 = cno.loc[cno.group == "CNO", "score"].values
g0 = cno.loc[cno.group == "HC", "score"].values
n1, n0 = len(g1), len(g0)

m1, m0 = g1.mean(), g0.mean()
v1, v0 = g1.var(ddof=1), g0.var(ddof=1)
sp = np.sqrt(((n1 - 1) * v1 + (n0 - 1) * v0) / (n1 + n0 - 2))
d_obs = (m1 - m0) / sp
t, p = stats.ttest_ind(g1, g0, equal_var=True)

# effect seen in bacterial osteomyelitis (discovery cohort), for comparison
om = s[s.dataset == "GSE30119"]
a1 = om.loc[om.group == "OM", "score"].values
a0 = om.loc[om.group == "HC", "score"].values
sp_om = np.sqrt(((len(a1) - 1) * a1.var(ddof=1) + (len(a0) - 1) * a0.var(ddof=1))
                / (len(a1) + len(a0) - 2))
d_om = (a1.mean() - a0.mean()) / sp_om

df = n1 + n0 - 2
ncp = np.sqrt(1 / n1 + 1 / n0)
alpha = 0.05


def power_at(d, n1=n1, n0=n0, alpha=alpha):
    df = n1 + n0 - 2
    nc = d / np.sqrt(1 / n1 + 1 / n0)
    crit = stats.t.ppf(1 - alpha / 2, df)
    hi = 1 - stats.nct.cdf(crit, df, nc)
    lo = stats.nct.cdf(-crit, df, nc)
    v = hi + lo
    if not np.isfinite(v):          # nct underflows at very large ncp
        v = 1.0
    return float(min(max(v, 0.0), 1.0))


def d_for_power(target=0.80, n1=n1, n0=n0, alpha=alpha):
    lo, hi = 0.01, 5.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if power_at(mid, n1, n0, alpha) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


d_mde = d_for_power(0.80)
pw_obs = power_at(abs(d_obs))
pw_om = power_at(abs(d_om))

# permutation test on the mean difference (two-sided, 20k shuffles)
rng = np.random.default_rng(20261108)
pool = np.concatenate([g1, g0])
obs = m1 - m0
cnt = 0
for _ in range(20000):
    rng.shuffle(pool)
    if abs(pool[:n1].mean() - pool[n1:].mean()) >= abs(obs):
        cnt += 1
p_perm = (cnt + 1) / 20001

out = pd.DataFrame([
    dict(analysis="CNO vs HC module score", n_cases=n1, n_controls=n0,
         mean_cases=round(m1, 4), mean_controls=round(m0, 4),
         cohens_d=round(d_obs, 3), t=round(float(t), 3), p_ttest=round(float(p), 3),
         p_permutation=round(p_perm, 3),
         power_for_observed_d=round(pw_obs, 3),
         min_detectable_d_80pct=round(d_mde, 3)),
    dict(analysis="reference: OM vs HC (GSE30119)", n_cases=len(a1), n_controls=len(a0),
         mean_cases=round(a1.mean(), 4), mean_controls=round(a0.mean(), 4),
         cohens_d=round(d_om, 3), t=np.nan, p_ttest=np.nan, p_permutation=np.nan,
         power_for_observed_d=round(pw_om, 3),
         min_detectable_d_80pct=round(d_mde, 3)),
])
fp = os.path.join(RES, "A_cno_power.csv")
out.to_csv(fp, index=False)
print(out.to_string(index=False))
print("\nsaved", fp)
print(f"\nCNO cohort (n={n1} vs {n0}) had 80% power to detect d >= {d_mde:.2f} "
      f"(alpha=0.05, two-sided); observed d = {d_obs:.3f}; "
      f"power at the observed effect = {pw_obs:.3f}.")
print(f"The same design would have had {pw_om:.3f} power for the effect seen in "
      f"bacterial osteomyelitis (d = {d_om:.2f}).")
