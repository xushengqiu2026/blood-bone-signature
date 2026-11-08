# -*- coding: utf-8 -*-
"""
A 方向 · 病原特异性：14 基因签名是「S. aureus 特异」还是「泛炎症/泛感染」？
数据 GSE6269（GPL96 全血 PBMC，含金葡/大肠杆菌/肺炎链球菌/流感/健康对照）
输出 results/A_pathogen_scores.csv, A_pathogen_tests.csv, figures/A_fig8_pathogen.png/pdf
"""
import os, re, pickle
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
DATA, RES, FIG = (os.path.join(BASE, x) for x in ["data", "results", "figures"])
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "savefig.dpi": 300, "savefig.bbox": "tight"})
COL = {"HC": "#8c8c8c", "S. aureus": "#c0392b", "E. coli": "#e08b2c",
       "S. pneumoniae": "#16827e", "Influenza": "#1f3b73"}

with open(os.path.join(DATA, "GPL96_probe2gene.pkl"), "rb") as f:
    p2g = pickle.load(f)
def cs(s):
    s = str(s).split("///")[0].strip().upper()
    return s if s and s != "NAN" else None

d = os.path.join(DATA, "gsm_tables", "GSE6269")
cols = {}
for fn in sorted(os.listdir(d)):
    if not fn.endswith(".csv"): continue
    df = pd.read_csv(os.path.join(d, fn), sep="\t")
    if "ID_REF" not in df or "VALUE" not in df: continue
    s = pd.to_numeric(df[["ID_REF", "VALUE"]].dropna().set_index("ID_REF")["VALUE"],
                      errors="coerce").dropna()
    s.name = fn[:-4]; cols[fn[:-4]] = s
M = pd.concat(cols.values(), axis=1)
M.index = [cs(p2g.get(p, "")) for p in M.index]
M = M[M.index.notna()]
M = np.log2(M.clip(lower=1))
M = M.loc[~M.index.duplicated(keep="first")]

txt = open(os.path.join(DATA, "gse6269_gsm.txt"), encoding="utf-8", errors="ignore").read()
lab = {}
for block in txt.split("^SAMPLE")[1:]:
    g = re.search(r"= (GSM\d+)", block); t = re.search(r"!Sample_title = (.+)", block)
    p = re.search(r"!Sample_platform_id = (\S+)", block)
    if g and t and p and p.group(1) == "GPL96":
        lab[g.group(1)] = t.group(1).strip()

def cls(t):
    tl = t.lower().replace(" ", "")
    if "s.aureus" in tl: return "S. aureus"
    if "e.coli" in tl: return "E. coli"
    if "pneumoniae" in tl: return "S. pneumoniae"
    if "influenza" in tl: return "Influenza"
    if "healthy" in tl or "control" in tl: return "HC"
    return None
lab = {g: cls(t) for g, t in lab.items() if cls(t) and g in M.columns}
M = M[list(lab)]; y = pd.Series(lab)
print("分组:", y.value_counts().to_dict(), flush=True)

sig = pd.read_csv(os.path.join(RES, "A_signature_genes.csv"))
gs = [g for g in sig.gene if g in M.index]
R = M.rank(axis=0, pct=True)
s = R.loc[gs].mean(axis=0)
s = s - s[(y == "HC").values].mean()          # 以本数据集健康对照中心化
df = pd.DataFrame({"score": s, "group": y.values}, index=M.columns)
df.to_csv(os.path.join(RES, "A_pathogen_scores.csv"))
print("签名基因覆盖:", len(gs), "/", len(sig), flush=True)

groups = ["HC", "S. aureus", "E. coli", "S. pneumoniae", "Influenza"]
vals = [df.loc[df.group == g, "score"].values for g in groups]
H, p_kw = stats.kruskal(*[v for v in vals if len(v) > 1])
print(f"Kruskal-Wallis H={H:.2f}, p={p_kw:.3g}", flush=True)
rows = []
for g in groups[1:]:
    a, b = df.loc[df.group == g, "score"].values, df.loc[df.group == "HC", "score"].values
    if len(a) < 3: continue
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    rows.append(dict(group=g, n=len(a), mean=a.mean(), median=np.median(a),
                     p_vs_HC=p, AUC_vs_HC=roc_auc_score(np.r_[np.ones(len(a)), np.zeros(len(b))],
                                                          np.r_[a, b])))
# S. aureus vs 其他所有病原
a = df.loc[df.group == "S. aureus", "score"].values
b = df.loc[df.group.isin(["E. coli", "S. pneumoniae", "Influenza"]), "score"].values
u_sa, p_sa = stats.mannwhitneyu(a, b, alternative="two-sided")
auc_sa = roc_auc_score(np.r_[np.ones(len(a)), np.zeros(len(b))], np.r_[a, b])
print(f"S. aureus vs 其他病原: p={p_sa:.3g}, AUC={auc_sa:.3f}", flush=True)
rows.append(dict(group="S. aureus vs other pathogens", n=len(a), mean=a.mean(),
                 median=np.median(a), p_vs_HC=p_sa, AUC_vs_HC=auc_sa))
pd.DataFrame(rows).to_csv(os.path.join(RES, "A_pathogen_tests.csv"), index=False)
print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.3g}"), flush=True)

fig, ax = plt.subplots(figsize=(7.6, 4.4))
pos = np.arange(len(groups))
bp = ax.boxplot(vals, tick_labels=[f"{g}\n(n={len(v)})" for g, v in zip(groups, vals)],
                patch_artist=True, widths=0.58, showfliers=False,
                medianprops=dict(color="black", lw=1.2))
for b, g in zip(bp["boxes"], groups):
    b.set_facecolor(COL[g]); b.set_alpha(0.72)
ax.axhline(0, color="#1f3b73", lw=0.8, ls="--")
ax.set_ylabel("14-gene signature score (rank, HC-centered)")
ax.set_title(f"Pathogen specificity in GSE6269 whole blood\n"
             f"Kruskal p={p_kw:.2g}; S. aureus vs other pathogens AUC={auc_sa:.3f}",
             fontsize=10, fontweight="bold")
ax.grid(axis="y", alpha=0.25, lw=0.5)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "A_fig8_pathogen.png"))
fig.savefig(os.path.join(FIG, "A_fig8_pathogen.pdf"))
plt.close(fig)
print("DONE", flush=True)
