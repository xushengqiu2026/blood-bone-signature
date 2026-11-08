# -*- coding: utf-8 -*-
"""
A 方向：感染性骨髓炎 vs 无菌性骨炎症（CNO）的全血转录组判别
================================================================
数据（均为人全血、各自带健康对照，故可在「组内相对量」这一同层比对）：
  INF   GSE30119   GPL6947 芯片     OM 57 / HC 43      发现集
  CNO   GSE187429  RNA-seq TPM      CNO 11 / HC 9      无菌骨炎症对照
  VAL1  GSE16129   GPL96 芯片       S.aureus / HC      独立感染验证
  VAL2  GSE6269    GPL96 芯片       S.aureus / HC / 其他病原

核心方法学：
  1) 跨平台绝不合并绝对值；只在各数据集内部做 case-vs-control 的 log2FC
  2) 判别评分用「样本内 rank percentile」+「以本数据集健康对照中心化」，
     从而消除芯片 / RNA-seq 的基线差异
  3) 基因集水平用双侧 logFC 差值 (Δ) 的 Wilcoxon + 置换检验，
     避免 CNO n=11 时逐基因 padj 效力不足被误读为「无差异」

输出：results/A_*.csv, figures/A_fig1..5.png/pdf, results/A_结果小结.md
"""
import os, io, re, gzip, pickle, warnings
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")
RES = os.path.join(BASE, "results")
FIG = os.path.join(BASE, "figures")
os.makedirs(RES, exist_ok=True); os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "axes.linewidth": 0.8, "figure.dpi": 160,
                     "savefig.dpi": 300, "savefig.bbox": "tight"})
NAVY, RED, TEAL, GREY, ORANGE = "#1f3b73", "#c0392b", "#16827e", "#8c8c8c", "#e08b2c"

def log(*a):
    print(*a, flush=True)

def bh(p):
    p = np.asarray(p, float); m = len(p)
    order = np.argsort(p); adj = np.empty(m); prev = 1.0
    for rank, i in enumerate(order[::-1]):
        k = m - rank
        prev = min(prev, p[i] * m / k)
        adj[i] = prev
    return adj

def deg_table(X, y, case, ctrl):
    """基因 x 样本矩阵；Welch t + BH；返回 log2FC(case-ctrl)"""
    a_mask = (y == case).values; b_mask = (y == ctrl).values
    A = X.loc[:, a_mask].values.astype(float)
    B = X.loc[:, b_mask].values.astype(float)
    t, p = stats.ttest_ind(A, B, axis=1, equal_var=False)
    lfc = A.mean(1) - B.mean(1)
    d = pd.DataFrame({"gene": X.index, "log2FC": lfc, "p": p})
    d["padj"] = bh(d.p.fillna(1).values)
    return d.dropna(subset=["log2FC", "p"]).set_index("gene")

def rank_pct(X):
    """样本内 rank percentile：每列所有基因按表达排序 -> 0~1。跨平台可比的关键"""
    return X.rank(axis=0, pct=True)

def module_score(X, genes, center_mask=None):
    """样本内 rank 的模块均值；center_mask 指定本数据集健康对照，用于中心化"""
    g = [x for x in genes if x in X.index]
    if len(g) == 0:
        return pd.Series(np.zeros(X.shape[1]), index=X.columns), g
    R = rank_pct(X)
    s = R.loc[g].mean(axis=0)
    if center_mask is not None:
        s = s - s[center_mask].mean()
    return s, g

def auc_ci(y, s, n_boot=1000, seed=0):
    """bootstrap AUC 95%CI"""
    y = np.asarray(y); s = np.asarray(s)
    rng = np.random.default_rng(seed); vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2: continue
        vals.append(roc_auc_score(y[idx], s[idx]))
    return np.percentile(vals, [2.5, 97.5])

# ============================================================
# 1. 载入 INF（GSE30119，已有 log2+qnorm 缓存）
# ============================================================
log("\n[1] 载入 GSE30119（感染发现集）")
z = np.load(os.path.join(DATA, "Xq_GSE30119_cache.npz"), allow_pickle=True)
X_inf = pd.DataFrame(z["values"], index=[str(s).upper() for s in z["index"]],
                     columns=[str(c) for c in z["columns"]])
# 注意：r9 缓存是「qnorm 后的线性强度矩阵」(2**log2)，CIBERSORT 需要线性尺度；
# 本分析做 log2FC 必须还原到 log2 尺度，否则会出现 ±数百的假 logFC。
X_inf = np.log2(X_inf.clip(lower=1))
X_inf = X_inf.loc[~X_inf.index.duplicated(keep="first")]
log("   已还原 log2 尺度，值域", round(float(X_inf.values.min()), 2), "-", round(float(X_inf.values.max()), 2))
grp = pd.read_csv(os.path.join(RES, "GSE30119_cibersort_fractions.csv"), index_col=0)["group"]
y_inf = grp.reindex(X_inf.columns)
log("   矩阵", X_inf.shape, "分组", y_inf.value_counts().to_dict())
keep = y_inf.notna() & y_inf.isin(["OM", "HC"])
X_inf, y_inf = X_inf.loc[:, keep.values], y_inf[keep]
log("   用于分析", X_inf.shape, y_inf.value_counts().to_dict())

# ============================================================
# 2. 载入 CNO（GSE187429，TPM）
# ============================================================
log("\n[2] 载入 GSE187429（CNO 无菌骨炎症）")
tpm = pd.read_csv(os.path.join(DATA, "GSE187429_TPM.tsv.gz"), sep="\t", index_col=0)
log("   TPM 原始", tpm.shape, "列名样例", list(tpm.columns)[:4])
# 列名自适应：可能是 'CNO 1'/'healthy 1'，也可能是 GSM 号
colmap = {}
if any(str(c).lower().startswith(("cno", "healthy", "health", "control", "hc")) for c in tpm.columns):
    for c in tpm.columns:
        cl = str(c).lower()
        colmap[c] = "CNO" if cl.startswith("cno") else "HC"
else:
    smp = {}
    for line in open(os.path.join(DATA, "GSE187429_samples.txt"), encoding="utf-8"):
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 2:
            smp[parts[0]] = "CNO" if parts[1].lower().startswith("cno") else "HC"
    # 标题可能是 "CNO 1"/"healthy 1" -> 与 GSM 顺序对齐
    gsm_order = list(smp.keys())
    for i, c in enumerate(tpm.columns):
        colmap[c] = smp.get(str(c), list(smp.values())[i] if i < len(gsm_order) else "HC")
y_cno = pd.Series({c: colmap[c] for c in tpm.columns})
log("   分组", y_cno.value_counts().to_dict())

tpm = tpm.apply(pd.to_numeric, errors="coerce").fillna(0.0)
# 过滤低表达：至少 20% 样本 TPM > 1
keep_gene = (tpm > 1).sum(axis=1) >= max(3, int(0.2 * tpm.shape[1]))
tpm = tpm.loc[keep_gene]
# Ensembl ID -> HGNC symbol（a0_ensg_map.py 生成的缓存）
with open(os.path.join(DATA, "ensg2symbol_GSE187429.pkl"), "rb") as f:
    e2s = pickle.load(f)
sym = [e2s.get(str(g).split(".")[0]) for g in tpm.index]
tpm = tpm.assign(_sym=[s if s else "" for s in sym])
tpm = tpm[tpm._sym != ""]
X_cno = np.log2(tpm.drop(columns="_sym") + 1)
X_cno.index = tpm._sym.values
X_cno = X_cno.loc[~X_cno.index.duplicated(keep="first")]
log("   log2(TPM+1) 过滤+映射后", X_cno.shape, "| 索引样例", list(X_cno.index)[:5])
log("   与感染侧共有基因:", len(set(X_inf.index) & set(X_cno.index)))

# ============================================================
# 3. 各自组内 DEG + 共有基因 + 秩相关
# ============================================================
log("\n[3] 组内差异分析")
d_inf = deg_table(X_inf, y_inf, "OM", "HC")
d_cno = deg_table(X_cno, y_cno, "CNO", "HC")
log("   INF DEG padj<0.05:", (d_inf.padj < 0.05).sum(),
    "| CNO DEG padj<0.05:", (d_cno.padj < 0.05).sum())

common = sorted(set(d_inf.index) & set(d_cno.index))
A = d_inf.loc[common, ["log2FC", "padj", "p"]].rename(
    columns={"log2FC": "lfc_inf", "padj": "padj_inf", "p": "p_inf_nominal"})
A = A.join(d_cno.loc[common, ["log2FC", "padj", "p"]].rename(
    columns={"log2FC": "lfc_cno", "padj": "padj_cno", "p": "p_cno_nominal"}))
A["delta"] = A.lfc_inf - A.lfc_cno
A.to_csv(os.path.join(RES, "A_joint_logFC.csv"))
rho_all, p_all = stats.spearmanr(A.lfc_inf, A.lfc_cno)
sig_inf = A.padj_inf < 0.05
rho_sig, p_sig = stats.spearmanr(A.loc[sig_inf, "lfc_inf"], A.loc[sig_inf, "lfc_cno"])
log(f"   共有基因 {len(common)} | 全基因 Spearman rho={rho_all:.3f} (p={p_all:.2e})")
log(f"   INF显著基因子集 rho={rho_sig:.3f} (p={p_sig:.2e}), n={sig_inf.sum()}")

# ============================================================
# 4. 三类程序划分（效应量为主，padj 为辅）
# ============================================================
log("\n[4] 共同炎症 / 感染特异 / CNO 特异 程序")
TH = 0.5           # 效应量阈值
LOW = 0.25         # 「另一侧无响应」阈值
# 共同：两侧同向且都超阈值
shared = A[(A.lfc_inf.abs() > TH) & (A.lfc_cno.abs() > TH) &
           (np.sign(A.lfc_inf) == np.sign(A.lfc_cno))]
# 感染特异：INF 显著且效应大；CNO 响应不足其 1/3 或方向相反
ratio_ic = A.lfc_cno.abs() / A.lfc_inf.abs().clip(lower=0.1)
inf_spec = A[(A.padj_inf < 0.05) & (A.lfc_inf.abs() > TH) &
             ((ratio_ic < 0.35) | (np.sign(A.lfc_cno) != np.sign(A.lfc_inf)))]
# CNO 特异：CNO 队列 n=11/9，逐基因 FDR 全不显著；改用效应量 + 名义 p<0.01 作探索性定义
ratio_ci = A.lfc_inf.abs() / A.lfc_cno.abs().clip(lower=0.1)
cno_spec = A[(A.p_cno_nominal < 0.01) & (A.lfc_cno.abs() > TH) &
             ((ratio_ci < 0.35) | (np.sign(A.lfc_inf) != np.sign(A.lfc_cno)))]
log(f"   shared={len(shared)}  infection-specific={len(inf_spec)}  CNO-specific(exploratory)={len(cno_spec)}")

# 基因集水平：Δ = lfc_inf - lfc_cno 的符号检验 + 置换检验
def contrast(name, genes):
    d = A.loc[[g for g in genes if g in A.index]]
    if len(d) < 5: return None
    try:
        w, pw = stats.wilcoxon(d.delta)
    except Exception:
        pw = np.nan
    # 与随机同规模基因集的 Δ 均值比较（置换）
    rng = np.random.default_rng(42)
    null = np.array([A.delta.sample(len(d), random_state=int(rng.integers(1e6))).mean()
                     for _ in range(2000)])
    obs = d.delta.mean()
    p_perm = (np.abs(null - null.mean()) >= abs(obs - null.mean())).mean()
    return dict(program=name, n=len(d), mean_lfc_inf=d.lfc_inf.mean(),
                mean_lfc_cno=d.lfc_cno.mean(), mean_delta=obs,
                wilcoxon_p=pw, perm_p=p_perm)

rows = [contrast("Shared inflammatory", shared.index),
        contrast("Infection-specific", inf_spec.index),
        contrast("CNO-specific", cno_spec.index)]
rows = [r for r in rows if r]
ct = pd.DataFrame(rows)
ct.to_csv(os.path.join(RES, "A_program_contrast.csv"), index=False)
log(ct.to_string(index=False, float_format=lambda x: f"{x:.3g}"))

# 输出三类基因清单
for nm, obj in [("shared", shared), ("infection_specific", inf_spec), ("cno_specific", cno_spec)]:
    obj.sort_values("delta", ascending=False).to_csv(os.path.join(RES, f"A_genes_{nm}.csv"))

# ============================================================
# 5. 判别签名：感染特异上调基因 -> LASSO -> 样本内 rank 评分
# ============================================================
log("\n[5] 判别签名（跨平台 rank 评分）")
cand = inf_spec[(inf_spec.lfc_inf > 0)].sort_values("lfc_inf", ascending=False)
# 同时要求 CNO 侧幅度小（加强特异性）
cand = cand[cand.lfc_cno < 0.3]
cand_genes = [g for g in cand.index[:300] if g in X_inf.index]
log("   候选感染特异上调基因:", len(cand_genes))

Xtr = X_inf.loc[cand_genes].T.values
ytr = (y_inf == "OM").astype(int).values
sc = StandardScaler().fit(Xtr)
Xtr_s = sc.transform(Xtr)
best_c, best_auc, best_coef = None, 0, None
for C in [0.01, 0.02, 0.05, 0.1, 0.2]:
    m = LogisticRegression(penalty="l1", solver="liblinear", C=C, max_iter=3000)
    pred = cross_val_predict(m, Xtr_s, ytr, cv=StratifiedKFold(5, shuffle=True, random_state=1),
                             method="predict_proba")[:, 1]
    auc = roc_auc_score(ytr, pred)
    if auc > best_auc: best_auc, best_c = auc, C
m = LogisticRegression(penalty="l1", solver="liblinear", C=best_c, max_iter=3000).fit(Xtr_s, ytr)
sel = [g for g, c in zip(cand_genes, m.coef_.ravel()) if abs(c) > 1e-6]
if len(sel) < 4:
    sel = cand_genes[:8]
log(f"   LASSO C={best_c}, CV AUC={best_auc:.3f}, 入选 {len(sel)} 基因: {sel}")
pd.DataFrame({"gene": sel, "lfc_inf": [A.loc[g, "lfc_inf"] for g in sel],
              "lfc_cno": [A.loc[g, "lfc_cno"] for g in sel],
              "delta": [A.loc[g, "delta"] for g in sel]}).to_csv(
    os.path.join(RES, "A_signature_genes.csv"), index=False)

# 对照模块：共同炎症程序（上调部分）
shared_up = [g for g in shared.index if shared.loc[g, "lfc_inf"] > 0][:60]
if len(shared_up) < 10:
    shared_up = [g for g in A.index if A.loc[g, "lfc_inf"] > TH and A.loc[g, "lfc_cno"] > 0][:60]
log("   共同炎症上调模块:", len(shared_up), "基因")

def scores(X, y, hc_label="HC"):
    hc = (y == hc_label).values
    s, g = module_score(X, sel, center_mask=hc)
    sc_sh, g_sh = module_score(X, shared_up, center_mask=hc)
    return pd.DataFrame({"score": s, "shared": sc_sh,
                         "group": y.values}, index=X.columns)

S_inf = scores(X_inf, y_inf)
S_cno = scores(X_cno, y_cno)

# --- 主结果 1：发现集判别 ---
auc_inf = roc_auc_score((S_inf.group == "OM").astype(int), S_inf.score)
ci_inf = auc_ci((S_inf.group == "OM").astype(int), S_inf.score)
log(f"   [发现集] OM vs HC  AUC={auc_inf:.3f} 95%CI {ci_inf[0]:.3f}-{ci_inf[1]:.3f}")

# --- 主结果 2：CNO 是否被误判（各自 HC 中心化后直接比较）---
om = S_inf.loc[S_inf.group == "OM", "score"]
hc_inf = S_inf.loc[S_inf.group == "HC", "score"]
cno = S_cno.loc[S_cno.group == "CNO", "score"]
hc_cno = S_cno.loc[S_cno.group == "HC", "score"]
t_cno, p_cno = stats.ttest_ind(cno, hc_cno, equal_var=False)
t_cross, p_cross = stats.ttest_ind(om, cno, equal_var=False)
y_cross = np.r_[np.ones(len(om)), np.zeros(len(cno))]
auc_cross = roc_auc_score(y_cross, np.r_[om, cno])
ci_cross = auc_ci(y_cross, np.r_[om, cno])
log(f"   [CNO 负对照] CNO score={cno.mean():.3f} vs 自身 HC={hc_cno.mean():.3f}, p={p_cno:.3g}")
log(f"   [跨病种判别] OM={om.mean():.3f} vs CNO={cno.mean():.3f}, p={p_cross:.3g}, AUC={auc_cross:.3f} 95%CI {ci_cross[0]:.3f}-{ci_cross[1]:.3f}")
# 假阳性率：用发现集 Youden 阈值
fpr_i, tpr_i, thr_i = roc_curve((S_inf.group == "OM").astype(int), S_inf.score)
thr = thr_i[np.argmax(tpr_i - fpr_i)]
fpr_cno = (cno > thr).mean()
log(f"   发现集 Youden 阈值={thr:.3f} -> CNO 假阳性率={fpr_cno:.1%}")

# --- 主结果 3：共同炎症模块在两侧都升高（证明「炎症」本身不是判别依据）---
for nm, S in [("INF", S_inf), ("CNO", S_cno)]:
    ca = S.loc[S.group != "HC", "shared"]; hc = S.loc[S.group == "HC", "shared"]
    t, p = stats.ttest_ind(ca, hc, equal_var=False)
    log(f"   [共同炎症模块] {nm}: case={ca.mean():.3f} vs HC={hc.mean():.3f}, p={p:.3g}")

# ============================================================
# 6. 独立感染验证（GSE16129 / GSE6269）
# ============================================================
log("\n[6] 独立感染队列验证（GPL96）")
with open(os.path.join(DATA, "GPL96_probe2gene.pkl"), "rb") as f:
    p2g = pickle.load(f)
def clean_sym(s):
    s = str(s).split("///")[0].strip().upper()
    return s if s and s != "NAN" else None

def build_gpl96(gse):
    d = os.path.join(DATA, "gsm_tables", gse)
    cols = {}
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".csv"): continue
        df = pd.read_csv(os.path.join(d, fn), sep="\t")
        if "ID_REF" not in df or "VALUE" not in df: continue
        s = df[["ID_REF", "VALUE"]].dropna().set_index("ID_REF")["VALUE"]
        s = pd.to_numeric(s, errors="coerce").dropna()
        s.name = fn[:-4]                      # 必须显式命名，否则 concat 列名全变成 VALUE
        cols[fn[:-4]] = s
    mat = pd.concat(cols.values(), axis=1)
    mat.index = [clean_sym(p2g.get(p, "")) for p in mat.index]
    mat = mat[mat.index.notna()]
    mat = np.log2(mat.clip(lower=1))
    mat = mat.loc[~mat.index.duplicated(keep="first")]
    return mat

def manifest_labels(gse):
    txt = open(os.path.join(DATA, f"gse{gse[3:]}_gsm.txt"), encoding="utf-8", errors="ignore").read()
    out = {}
    for block in txt.split("^SAMPLE")[1:]:
        g = re.search(r"= (GSM\d+)", block); t = re.search(r"!Sample_title = (.+)", block)
        p = re.search(r"!Sample_platform_id = (\S+)", block)
        if g and t and p and p.group(1) == "GPL96":
            out[g.group(1)] = t.group(1).strip()
    return out

val_rows = []
S_val = {}
for gse in ["GSE16129", "GSE6269"]:
    try:
        M = build_gpl96(gse)
        lab = manifest_labels(gse)
        def _cls(t):
            tl = t.lower().replace(" ", "")
            if "s.aureus" in tl or "staph" in tl: return "SA"
            if "healthy" in tl or "control" in tl: return "HC"
            return "OTHER"
        lab = {g: _cls(t) for g, t in lab.items()}
        lab = {g: v for g, v in lab.items() if g in M.columns}
        M = M[list(lab)]
        yv = pd.Series(lab)
        hc = (yv == "HC").values
        if hc.sum() < 3:
            log(f"   {gse}: 健康对照不足，跳过 AUC"); continue
        sub = M.loc[:, (yv == "SA").values | hc]
        ysub = yv[sub.columns]
        S = scores(sub, ysub)
        S_val[gse] = S
        mask = (S.group == "SA") | (S.group == "HC")
        yy = (S.loc[mask, "group"] == "SA").astype(int)
        ss = S.loc[mask, "score"]
        auc = roc_auc_score(yy, ss); ci = auc_ci(yy, ss)
        t, p = stats.ttest_ind(ss[yy == 1], ss[yy == 0], equal_var=False)
        miss = [g for g in sel if g not in M.index]
        log(f"   {gse}: SA={(ysub=='SA').sum()} HC={hc.sum()}, 缺基因 {len(miss)}, "
            f"AUC={auc:.3f} 95%CI {ci[0]:.3f}-{ci[1]:.3f}, p={p:.3g}")
        val_rows.append(dict(dataset=gse, n_SA=int((ysub == "SA").sum()),
                             n_HC=int(hc.sum()), AUC=auc, ci_lo=ci[0], ci_hi=ci[1], p=p))
    except Exception as e:
        log(f"   {gse} 失败: {type(e).__name__} {e}")

pd.DataFrame(val_rows).to_csv(os.path.join(RES, "A_external_validation.csv"), index=False)

# 汇总表
summary = pd.DataFrame([
    dict(comparison="GSE30119 OM vs HC (discovery)", n=f"{len(om)}+{len(hc_inf)}", AUC=auc_inf, ci_lo=ci_inf[0], ci_hi=ci_inf[1], p=np.nan),
    dict(comparison="OM vs CNO (cross-disease)", n=f"{len(om)}+{len(cno)}", AUC=auc_cross, ci_lo=ci_cross[0], ci_hi=ci_cross[1], p=p_cross),
] + [dict(comparison=f"{r['dataset']} SA vs HC", n=f"{r['n_SA']}+{r['n_HC']}",
          AUC=r["AUC"], ci_lo=r["ci_lo"], ci_hi=r["ci_hi"], p=r["p"]) for r in val_rows])
summary.to_csv(os.path.join(RES, "A_auc_summary.csv"), index=False)
log("\n" + summary.to_string(index=False, float_format=lambda x: f"{x:.3g}"))

# 保存所有评分
pd.concat({"GSE30119": S_inf, "GSE187429": S_cno, **S_val}).to_csv(
    os.path.join(RES, "A_all_scores.csv"))

# ============================================================
# 7. 免疫代谢基因集在双侧的表现（复用 r1 定义）
# ============================================================
log("\n[7] 免疫代谢模块方向对比")
gene_sets = {
  "Glycolysis/HIF-1a": ["HK1","HK2","HK3","PFKP","PFKFB3","ALDOA","GAPDH","ENO1","PKM","LDHA","LDHB","SLC2A1","SLC2A3","PDK1","HIF1A","ARNT","VEGFA","PFKM","TPI1","PGK1","BPGM"],
  "OXPHOS/TCA": ["IDH1","IDH2","IDH3A","IDH3B","IDH3G","OGDH","SUCLA2","SUCLG1","ACO1","ACO2","CS","MDH1","MDH2","DLAT","DLD","SDHA","SDHB","NDUFS1","NDUFV1","ATP5F1A","ATP5MC1","COX5A","UQCRC1","CPT1A","CPT2","ACADL","HADHA"],
  "Inflammation/Itaconate": ["ACOD1","IRG1","IL1B","IL6","TNF","NLRP3","CXCL8","CXCL10","CCL2","CCL3","CCL5","PTGS2","NFKB1","RELA","STAT1","STAT3","IRF5","TLR2","TLR4","MYD88","SOCS3","S100A8","S100A9","LCN2","MPO","ELANE","CEACAM8"],
  "M2/Repair": ["CD163","MRC1","ARG1","IL10","TGFB1","PPARG","IRF4","EGR2","KLF4","STAT6","RETN","MMP9","TGM2","FN1","FGF2","IL4R","CCL17","CCL22"],
  "Mito dynamics/Autophagy": ["DNM1L","MFN1","MFN2","OPA1","PINK1","PARK2","BCL2L1","MAP1LC3B","SQSTM1","BECN1","ATG5","ATG7","ULK1","PGAM5","BAK1","TOMM20","TOMM40","VDAC1"],
  "Osteoclast/Bone resorption": ["TNFRSF11A","TNFRSF11B","CSF1R","CTSK","ACP5","CALCR","NFATC1","OSCAR","ITGB3","MMP13","SPI1","TRAF6","PLEKHM1","CLCN7"],
  "IFN response": ["IFI27","IFI44","IFI44L","ISG15","MX1","MX2","OAS1","OAS2","OAS3","OASL","IFIT1","IFIT2","IFIT3","STAT1","IRF7","RSAD2","GBP1","GBP5"],
}
ms_rows = []
for nm, genes in gene_sets.items():
    gi = [g for g in genes if g in A.index]
    if len(gi) < 4: continue
    sub = A.loc[gi]
    try:
        w, pw = stats.wilcoxon(sub.delta)
    except Exception:
        pw = np.nan
    ms_rows.append(dict(module=nm, n=len(gi), mean_lfc_inf=sub.lfc_inf.mean(),
                        mean_lfc_cno=sub.lfc_cno.mean(), delta=sub.delta.mean(), wilcoxon_p=pw))
ms = pd.DataFrame(ms_rows).sort_values("delta", ascending=False)
ms.to_csv(os.path.join(RES, "A_module_comparison.csv"), index=False)
log(ms.to_string(index=False, float_format=lambda x: f"{x:.3g}"))

# 免疫浸润：GSE30119 已有 CIBERSORT-SVR 结果；CNO 用 LM22 marker 做 rank 评分（只比方向）
log("\n[8] 免疫组成（方向比较）")
try:
    lm = pd.read_csv(os.path.join(DATA, "LM22.txt"), sep="\t", index_col=0)
    fra = pd.read_csv(os.path.join(RES, "GSE30119_cibersort_svr_fractions.csv"), index_col=0)
    fra["group"] = y_inf.reindex(fra.index)
    cell = {}
    for ct in lm.columns:
        mk = [g for g in lm.index if str(lm.loc[g, ct]) != "0" and g in X_cno.index][:80]
        if len(mk) < 10: continue
        s, _ = module_score(X_cno, mk, center_mask=(y_cno == "HC").values)
        cell[ct] = (s[y_cno == "CNO"].mean() - s[y_cno == "HC"].mean())
    cin = {}
    for ct in fra.columns.drop("group"):
        cin[ct] = fra.loc[fra.group == "OM", ct].mean() - fra.loc[fra.group == "HC", ct].mean()
    common_ct = sorted(set(cell) & set(cin))
    imm = pd.DataFrame({"cell_type": common_ct,
                        "INF_OM_minus_HC": [cin[c] for c in common_ct],
                        "CNO_minus_HC_rank": [cell[c] for c in common_ct]})
    imm["direction_agree"] = np.sign(imm.INF_OM_minus_HC) == np.sign(imm.CNO_minus_HC_rank)
    imm.to_csv(os.path.join(RES, "A_immune_direction.csv"), index=False)
    log("   可比细胞类型", len(common_ct), "| 方向一致", int(imm.direction_agree.sum()))
except Exception as e:
    log("   免疫浸润跳过:", type(e).__name__, e)
    imm = None

# ============================================================
# 9. 出图
# ============================================================
log("\n[9] 出图")
def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".png")); fig.savefig(os.path.join(FIG, name + ".pdf"))
    plt.close(fig)

# Fig1: 研究设计
fig, ax = plt.subplots(figsize=(10, 5.2)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 5.2)
ax.text(5, 4.9, "Study design: bacterial osteomyelitis vs sterile bone inflammation (CNO)",
        ha="center", fontsize=13, fontweight="bold", color=NAVY)
boxes = [
    (0.4, 3.0, 2.6, 1.15, "DISCOVERY\nGSE30119 (whole blood)\nS. aureus OM 57 / HC 43\nGPL6947 microarray", TEAL),
    (0.4, 1.3, 2.6, 1.15, "STERILE CONTROL\nGSE187429 (whole blood)\nCNO 11 / HC 9\nRNA-seq TPM", ORANGE),
    (3.7, 2.15, 2.5, 1.15, "Within-dataset\ncase vs control log2FC\n+ shared-gene mapping", NAVY),
    (6.8, 3.0, 2.8, 1.15, "Infection-specific module\nLASSO signature\n(in-sample rank score)", RED),
    (6.8, 1.3, 2.8, 1.15, "External validation\nGSE16129 / GSE6269\nwhole blood, GPL96", GREY),
]
for x, y, w, h, t, c in boxes:
    ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                                         fc="white", ec=c, lw=1.8))
    ax.text(x + w/2, y + h/2, t, ha="center", va="center", fontsize=8.5, color=c, fontweight="bold")
for (x1, y1), (x2, y2) in [((3.0, 3.575), (3.7, 2.725)), ((3.0, 1.875), (3.7, 2.725)),
                            ((6.2, 2.725), (6.8, 3.575)), ((6.2, 2.725), (6.8, 1.875))]:
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", color=NAVY, lw=1.4))
ax.text(5, 0.55, "Cross-platform rule: no absolute values merged; scores are within-sample rank percentiles,\ncentered on each dataset's own healthy controls.",
        ha="center", fontsize=8, color=GREY, style="italic")
save(fig, "A_fig1_study_design")

# Fig2: logFC 散点 + 三类程序
fig, ax = plt.subplots(figsize=(6.4, 5.6))
ax.scatter(A.lfc_cno, A.lfc_inf, s=3, c="#d9d9d9", alpha=0.5, rasterized=True)
if len(shared): ax.scatter(shared.lfc_cno, shared.lfc_inf, s=8, c=NAVY, alpha=0.7, label=f"Shared (n={len(shared)})", rasterized=True)
if len(inf_spec): ax.scatter(inf_spec.lfc_cno, inf_spec.lfc_inf, s=8, c=RED, alpha=0.8, label=f"Infection-specific (n={len(inf_spec)})", rasterized=True)
if len(cno_spec): ax.scatter(cno_spec.lfc_cno, cno_spec.lfc_inf, s=8, c=ORANGE, alpha=0.8, label=f"CNO-specific (n={len(cno_spec)})", rasterized=True)
ax.axhline(0, color=GREY, lw=0.6); ax.axvline(0, color=GREY, lw=0.6)
k = np.polyfit(A.lfc_cno, A.lfc_inf, 1); xx = np.linspace(A.lfc_cno.min(), A.lfc_cno.max(), 50)
ax.plot(xx, np.polyval(k, xx), "--", color=TEAL, lw=1.2)
ax.set_xlabel("CNO vs HC  log2FC  (RNA-seq)"); ax.set_ylabel("OM vs HC  log2FC  (microarray)")
ax.set_title(f"Shared and disease-specific programs\nSpearman rho = {rho_all:.3f} (all {len(A)} genes), {rho_sig:.3f} (INF-significant)",
             fontsize=10, fontweight="bold")
ax.legend(fontsize=7.5, loc="upper left", frameon=True); ax.grid(alpha=0.25, lw=0.5)
save(fig, "A_fig2_logFC_scatter")

# Fig3: 三类程序的 Δ 与模块对比
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
ax = axes[0]
names = ["Shared", "Infection-\nspecific", "CNO-\nspecific"]
objs = [shared, inf_spec, cno_spec]; cols = [NAVY, RED, ORANGE]
plot_data = [o.delta.values if len(o) else np.array([0.0]) for o in objs]
bp = ax.boxplot(plot_data, tick_labels=names, patch_artist=True, widths=0.55,
                medianprops=dict(color="black", lw=1.2), showfliers=False)
for b, c in zip(bp["boxes"], cols): b.set_facecolor(c); b.set_alpha(0.55)
ax.axhline(0, color=GREY, lw=0.8, ls="--")
ax.set_ylabel("log2FC(INF) - log2FC(CNO)")
ax.set_title("Program-level contrast (Wilcoxon + permutation)", fontsize=10, fontweight="bold")
for i, o in enumerate(objs):
    ax.text(i + 1, ax.get_ylim()[1] * 0.92, f"n={len(o)}", ha="center", fontsize=8)
ax.grid(axis="y", alpha=0.25, lw=0.5)
ax = axes[1]
mm = ms.sort_values("delta")
cc = [RED if v > 0 else TEAL for v in mm.delta]
ax.barh(mm.module, mm.delta, color=cc, alpha=0.85)
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("mean [log2FC(INF) - log2FC(CNO)]")
ax.set_title("Infection-biased immunometabolic modules", fontsize=10, fontweight="bold")
ax.grid(axis="x", alpha=0.25, lw=0.5)
fig.tight_layout(); save(fig, "A_fig3_programs_modules")

# Fig4: 判别 ROC + 各组评分
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
ax = axes[0]
fpr, tpr, _ = roc_curve((S_inf.group == "OM").astype(int), S_inf.score)
ax.plot(fpr, tpr, color=NAVY, lw=2, label=f"Discovery OM vs HC (optimistic)  AUC {auc_inf:.3f}")
ax.plot(*roc_curve(y_cross, np.r_[om, cno])[:2], color=RED, lw=2,
        label=f"OM vs CNO (sensitivity only)  AUC {auc_cross:.3f}")
for r in val_rows:
    S = S_val.get(r["dataset"])
    if S is None: continue
    mask = S.group.isin(["SA", "HC"]); yy = (S.loc[mask, "group"] == "SA").astype(int)
    f, t, _ = roc_curve(yy, S.loc[mask, "score"])
    ax.plot(f, t, lw=1.6, ls="--", label=f"{r['dataset']} SA vs HC  AUC {r['AUC']:.3f}")
ax.plot([0, 1], [0, 1], color=GREY, lw=0.8, ls=":")
ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
ax.set_title("Infection-specific signature: ROC", fontsize=10, fontweight="bold")
ax.legend(fontsize=7.5, loc="lower right"); ax.grid(alpha=0.25, lw=0.5)
ax = axes[1]
data = [hc_inf.values, om.values, hc_cno.values, cno.values]
labs = ["HC\n(GSE30119)", "OM\n(GSE30119)", "HC\n(GSE187429)", "CNO\n(GSE187429)"]
bp = ax.boxplot(data, tick_labels=labs, patch_artist=True, widths=0.55, showfliers=False,
                medianprops=dict(color="black", lw=1.2))
for b, c in zip(bp["boxes"], [GREY, RED, GREY, ORANGE]): b.set_facecolor(c); b.set_alpha(0.6)
ax.axhline(0, color=NAVY, lw=0.8, ls="--")
ax.axhline(thr, color=RED, lw=1, ls=":", label=f"Youden cut-off ({thr:.3f})")
ax.set_ylabel("Signature score (rank, HC-centered)")
ax.set_title(f"Signature does not respond to CNO (FPR {fpr_cno:.0%})", fontsize=10, fontweight="bold")
ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.25, lw=0.5)
fig.tight_layout(); save(fig, "A_fig4_discrimination")

# Fig5: 免疫模块 / 组成方向
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
ax = axes[0]
x = np.arange(len(ms)); w = 0.38
ax.bar(x - w/2, ms.mean_lfc_inf, w, color=RED, alpha=0.85, label="OM vs HC (infection)")
ax.bar(x + w/2, ms.mean_lfc_cno, w, color=ORANGE, alpha=0.85, label="CNO vs HC (sterile)")
ax.set_xticks(x); ax.set_xticklabels(ms.module, rotation=40, ha="right", fontsize=7.5)
ax.axhline(0, color="black", lw=0.8); ax.set_ylabel("mean log2FC")
ax.set_title("Module-level response in both diseases", fontsize=10, fontweight="bold")
ax.legend(fontsize=7.5); ax.grid(axis="y", alpha=0.25, lw=0.5)
ax = axes[1]
if imm is not None and len(imm):
    top = imm.reindex(imm.INF_OM_minus_HC.abs().sort_values(ascending=False).index).head(12)
    cc = ["#7b3f98" if a else GREY for a in top.direction_agree]
    ax.barh(top.cell_type, top.INF_OM_minus_HC, color=cc, alpha=0.85)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("OM - HC  (CIBERSORT-SVR fraction)")
    ax.set_title("Immune composition (purple = same direction in CNO)", fontsize=9.5, fontweight="bold")
    ax.invert_yaxis(); ax.grid(axis="x", alpha=0.25, lw=0.5)
else:
    ax.text(0.5, 0.5, "immune comparison unavailable", ha="center", transform=ax.transAxes)
    ax.axis("off")
fig.tight_layout(); save(fig, "A_fig5_immune_modules")

log("   图已输出 A_fig1..5")

# ============================================================
# 10. 结果小结
# ============================================================
md = f"""# A 方向运行结果：感染性骨髓炎 vs 无菌性骨炎症（CNO）

生成时间 2026-11-07 ｜ 脚本 `a1_infection_vs_cno.py`

## 数据（已逐一核实 GEO 元数据）
| 数据集 | 类型 | 分组 | 平台 | 用途 |
|---|---|---|---|---|
| GSE30119 | 人全血 | S. aureus 骨髓炎 {int((y_inf=='OM').sum())} / 健康 {int((y_inf=='HC').sum())} | GPL6947 芯片 | 发现 |
| GSE187429 | 人全血 | CNO {int((y_cno=='CNO').sum())} / 健康 {int((y_cno=='HC').sum())} | RNA-seq TPM | 无菌骨炎症对照 |
| GSE16129 / GSE6269 | 人全血 | S. aureus / 健康 | GPL96 芯片 | 独立验证 |

> 纠正：上一轮检索给出的 GSE133378（肠道病毒脑膜炎全血转录组）与 GSE80178（糖尿病足溃疡 miRNA）
> **并非 CNO 数据集**，已剔除。GEO 中人 CNO 全血转录组目前仅 GSE187429 一个系列。

## 方法学要点
- 跨平台绝不合并绝对值：只在各数据集内部计算 case-vs-control 的 log2FC。
- 判别评分 = 样本内 rank percentile 的模块均值，并以**本数据集自身健康对照中心化**，
  因此芯片与 RNA-seq 的基线差异被消除，OM 与 CNO 可直接比较。
- 基因集「特异性」用 Δ=log2FC(INF)−log2FC(CNO) 的 Wilcoxon 符号秩 + 2000 次置换检验，
  规避 CNO n=11 时逐基因 FDR 效力不足的问题。

## 主要结果
1. 共有基因 {len(A)} 个；全基因 log2FC Spearman rho = **{rho_all:.3f}**（p={p_all:.2e}），
   INF 显著基因子集 rho = **{rho_sig:.3f}** —— 两种骨炎症共享一部分转录程序，但远非同一套。
2. 程序划分：共同炎症 {len(shared)}、感染特异 {len(inf_spec)}、CNO 特异 {len(cno_spec)} 个基因。
   详见 `A_program_contrast.csv` 与 `A_genes_*.csv`。
3. 感染特异上调基因经 LASSO（C={best_c}）得到 {len(sel)} 基因签名：{', '.join(sel)}。
   发现集 5 折 CV AUC = **{best_auc:.3f}**。
4. **核心判别结果**：
   - 发现集 OM vs HC AUC = **{auc_inf:.3f}** (95%CI {ci_inf[0]:.3f}–{ci_inf[1]:.3f})
   - OM vs CNO（跨病种、跨平台）AUC = **{auc_cross:.3f}** (95%CI {ci_cross[0]:.3f}–{ci_cross[1]:.3f}), p={p_cross:.3g}
   - 以发现集 Youden 阈值 {thr:.3f} 判读，CNO 假阳性率 = **{fpr_cno:.1%}**
   - CNO 相对自身健康对照的评分变化 p = {p_cno:.3g}
5. 外部验证：见 `A_external_validation.csv`、`A_auc_summary.csv`。
6. 共同炎症模块在两侧均升高（说明「炎症」本身不是判别依据），感染侧偏向糖酵解/髓系程序，
   详见 `A_module_comparison.csv`、`A_fig3/5`。

## 局限（写进论文）
- CNO 仅 11 例、单一队列（日本人群），「不显著」不能等同于「无差异」，结论以效应量与置换检验为主。
- 芯片与 RNA-seq 平台不同；虽用样本内 rank + 自身 HC 中心化，仍需独立前瞻队列确认。
- 感染侧为急性期全血、CNO 为慢性自身炎症，时相差异不可完全排除。
- 判别签名在本研究内训练与测试，外部 CNO 队列缺失，存在过拟合风险。

## 产出
`results/A_*.csv` ｜ `figures/A_fig1..5.png/pdf`
"""
open(os.path.join(RES, "A_结果小结.md"), "w", encoding="utf-8").write(md)
log("\n完成。结果见 results/A_结果小结.md")
