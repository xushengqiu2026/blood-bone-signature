# -*- coding: utf-8 -*-
"""
A 方向 · 无泄漏验证：嵌套交叉验证
问题：a1 中候选基因来自全数据 DEG，再在同一批数据上算 AUC 属特征选择泄漏。
做法：外层 5 折，每一折只在 4/5 训练样本内重算 DEG、重选基因、重训模型，
      在留出的 1/5 上预测；汇总 cross-validated AUC。
同时输出「固定 14 基因签名」在发现集上的乐观 AUC 作为对照。
"""
import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve

BASE = os.path.dirname(os.path.abspath(__file__))
DATA, RES = os.path.join(BASE, "data"), os.path.join(BASE, "results")

def bh(p):
    p = np.asarray(p, float); m = len(p)
    o = np.argsort(p); adj = np.empty(m); prev = 1.0
    for r, i in enumerate(o[::-1]):
        k = m - r; prev = min(prev, p[i] * m / k); adj[i] = prev
    return adj

z = np.load(os.path.join(DATA, "Xq_GSE30119_cache.npz"), allow_pickle=True)
X = pd.DataFrame(np.log2(np.clip(z["values"], 1, None)),
                 index=[str(s).upper() for s in z["index"]],
                 columns=[str(c) for c in z["columns"]])
X = X.loc[~X.index.duplicated(keep="first")]
grp = pd.read_csv(os.path.join(RES, "GSE30119_cibersort_fractions.csv"), index_col=0)["group"]
y = (grp.reindex(X.columns) == "OM").astype(int).values
R = X.rank(axis=0, pct=True)          # 样本内 rank，无跨样本信息，可全数据预计算
Xt = R.T.values                        # 样本 x 基因
genes = np.array(X.index)

skf = StratifiedKFold(5, shuffle=True, random_state=42)
oof = np.zeros(len(y))
selected_counts = []
for k, (tr, te) in enumerate(skf.split(Xt, y)):
    A = X.iloc[:, tr]; yy = y[tr]
    a, b = A.loc[:, yy == 1].values, A.loc[:, yy == 0].values
    t, p = stats.ttest_ind(a, b, axis=1, equal_var=False)
    lfc = a.mean(1) - b.mean(1)
    padj = bh(np.nan_to_num(p, nan=1.0))
    cand = np.where((padj < 0.05) & (lfc > 0.5))[0]
    cand = cand[np.argsort(-lfc[cand])][:20]
    selected_counts.append(len(cand))
    sc = StandardScaler().fit(Xt[np.ix_(tr, cand)])
    m = LogisticRegression(C=0.1, max_iter=2000).fit(sc.transform(Xt[np.ix_(tr, cand)]), yy)
    oof[te] = m.predict_proba(sc.transform(Xt[np.ix_(te, cand)]))[:, 1]
    print(f"  fold {k+1}: 训练折重选 {len(cand)} 基因, 留出折 AUC={roc_auc_score(y[te], oof[te]):.3f}", flush=True)

auc_nested = roc_auc_score(y, oof)
fpr, tpr, _ = roc_curve(y, oof)

# 固定 14 基因签名（a1 产出）在发现集上的乐观 AUC
sig = pd.read_csv(os.path.join(RES, "A_signature_genes.csv"))
gs = [g for g in sig.gene if g in X.index]
s_fixed = R.loc[gs].mean(axis=0).values
auc_fixed = roc_auc_score(y, s_fixed)

# bootstrap CI
rng = np.random.default_rng(0); boots = []
for _ in range(1000):
    i = rng.integers(0, len(y), len(y))
    if len(np.unique(y[i])) == 2: boots.append(roc_auc_score(y[i], oof[i]))
ci = np.percentile(boots, [2.5, 97.5])

print(f"\n嵌套 CV（无特征选择泄漏）AUC = {auc_nested:.3f}  95%CI {ci[0]:.3f}-{ci[1]:.3f}")
print(f"固定 14 基因签名在发现集（乐观/有泄漏）AUC = {auc_fixed:.3f}")
print("独立外部验证 AUC 见 A_external_validation.csv（GSE16129/6269，完全无泄漏）")

out = pd.DataFrame([
    dict(evaluation="Nested 5-fold CV (leakage-free, discovery)", AUC=auc_nested, ci_lo=ci[0], ci_hi=ci[1], note="feature selection inside each fold"),
    dict(evaluation="Fixed 14-gene signature, same data (optimistic)", AUC=auc_fixed, ci_lo=np.nan, ci_hi=np.nan, note="DO NOT report as primary"),
])
out.to_csv(os.path.join(RES, "A_nested_cv.csv"), index=False)
np.savez(os.path.join(RES, "A_nested_cv_oof.npz"), y=y, oof=oof, fpr=fpr, tpr=tpr)
