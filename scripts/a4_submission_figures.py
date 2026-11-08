# -*- coding: utf-8 -*-
"""A 方向投稿图（Clinical Immunology 版）：Figure 1-6, S1, S2
输出: output/20261103-nsfc/stage3/submission_ClinicalImmunology/figures/
每图 PNG (300 dpi) + PDF (矢量) 双格式。
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from sklearn.metrics import roc_curve, auc

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 300, "savefig.bbox": "tight",
})

BASE = r"C:\Users\邱旭升\WorkBuddy\2026-11-03-10-41-59"
RES = os.path.join(BASE, "geo_min", "results")
FIG = os.path.join(BASE, "geo_min", "figures")
OUT = os.path.join(BASE, "output", "20261103-nsfc", "stage3",
                   "submission_ClinicalImmunology", "figures")
os.makedirs(OUT, exist_ok=True)

NAVY = "#1F3A5F"; RED = "#C0392B"; TEAL = "#0E6B6B"; ORANGE = "#E67E22"
GREY = "#8A8A8A"; BLUE = "#2E5E9E"; GREEN = "#1E8449"

def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=300)
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    plt.close(fig)
    print("saved", name)

def box_data():
    s = pd.read_csv(os.path.join(RES, "A_all_scores.csv"), index_col=[0, 1])
    s.columns = ["score", "shared", "group"]
    return s

def roc(y, x):
    fpr, tpr, _ = roc_curve(y, x)
    return fpr, tpr, auc(fpr, tpr)

# =====================================================================
# Figure 1  Study design (4 blood cohorts + 1 interface cohort, 3 steps)
# =====================================================================
fig, ax = plt.subplots(figsize=(9.0, 5.4))
ax.set_xlim(0, 10); ax.set_ylim(0, 6.4); ax.axis("off")
ax.text(5, 6.15, "Study design", ha="center", fontsize=13,
        fontweight="bold", color=NAVY)

def box(x, y, w, h, title, lines, ec, fc="#FFFFFF", fs=7.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                                ec=ec, fc=fc, lw=1.6))
    ax.text(x + w/2, y + h - 0.28, title, ha="center", va="top",
            fontsize=fs + 0.8, fontweight="bold", color=ec)
    for i, ln in enumerate(lines):
        ax.text(x + w/2, y + h - 0.62 - i*0.26, ln, ha="center", va="top",
                fontsize=fs, color="#333333")

def arrow(x1, y1, x2, y2, color=NAVY):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=13, color=color, lw=1.4,
                                 shrinkA=2, shrinkB=2))

# left column: blood cohorts
box(0.2, 4.35, 2.9, 1.45, "DISCOVERY (blood)",
    ["GSE30119, GPL6947 array", "Haematogenous OM 57 / HC 43"], TEAL)
box(0.2, 2.45, 2.9, 1.45, "STERILE COMPARATOR (blood)",
    ["GSE187429, RNA-seq (TPM)", "CNO 11 / HC 9"], ORANGE)
box(0.2, 0.55, 2.9, 1.45, "VALIDATION (blood)",
    ["GSE16129 / GSE6269, GPL96", "S. aureus vs HC; 5 pathogens"], GREY)
# middle
box(3.75, 2.45, 2.5, 1.45, "HARMONISATION",
    ["Within-dataset log2FC;", "rank-based module scores", "centred on own HC"], NAVY)
# right column
box(6.9, 4.35, 2.9, 1.45, "SIGNATURE + SPECIFICITY",
    ["14-gene LASSO, nested CV;", "CNO and pathogen specificity"], RED)
box(6.9, 2.45, 2.9, 1.45, "TISSUE TRANSFER",
    ["GSE255786 sonicate fluid", "PJI 53 / aseptic failure 40"], BLUE)
box(6.9, 0.55, 2.9, 1.45, "FAILURE EXPLAINED?",
    ["4 exclusion tests: detection,", "cell composition, depth,", "direction concordance"], "#555555")

arrow(3.15, 5.05, 3.72, 3.6); arrow(3.15, 3.15, 3.72, 3.35); arrow(3.15, 1.3, 3.72, 2.7)
arrow(6.28, 3.6, 6.87, 5.05); arrow(6.28, 3.35, 6.87, 3.15); arrow(6.28, 3.0, 6.87, 1.35)
arrow(9.8, 4.3, 9.8, 3.95, color=BLUE)   # transfer down
arrow(9.8, 2.4, 9.8, 2.05, color="#555555")  # exclusion down
ax.text(5, 0.12, "Cross-platform rule: no absolute values merged; scores are within-sample rank percentiles "
                 "centred on each dataset's own healthy controls.",
        ha="center", fontsize=7.4, style="italic", color="#555555")
save(fig, "Figure_1")

# =====================================================================
# Figure 2  logFC scatter + programme contrast + module bars
# =====================================================================
fig, axs = plt.subplots(1, 3, figsize=(12.6, 4.0),
                        gridspec_kw={"width_ratios": [1.35, 0.9, 1.15]})
ax = axs[0]
j = pd.read_csv(os.path.join(RES, "A_joint_logFC.csv"))
shared_g = set(pd.read_csv(os.path.join(RES, "A_genes_shared.csv")).gene)
inf_g = set(pd.read_csv(os.path.join(RES, "A_genes_infection_specific.csv")).gene)
cno_g = set(pd.read_csv(os.path.join(RES, "A_genes_cno_specific.csv")).gene)
cat = np.where(j.gene.isin(shared_g), "Shared",
      np.where(j.gene.isin(inf_g), "Infection-specific",
      np.where(j.gene.isin(cno_g), "CNO-specific", "background")))
n_sh, n_inf, n_cn = len(shared_g), len(inf_g), len(cno_g)  # published counts (3 genes counted in both shared and infection-specific)
bg = cat == "background"
ax.scatter(j.lfc_cno[bg], j.lfc_inf[bg], s=4, color="#CCCCCC", alpha=0.5, lw=0,
           label="background")
for m, c, lab, sz in [("Shared", BLUE, f"Shared (n={n_sh})", 16),
                      ("Infection-specific", RED, f"Infection-specific (n={n_inf:,})", 9),
                      ("CNO-specific", ORANGE, f"CNO-specific (n={n_cn})", 24)]:
    mm = cat == m
    ax.scatter(j.lfc_cno[mm], j.lfc_inf[mm], s=sz, color=c, alpha=0.9, lw=0, label=lab)
# trend line (Theil-like via polyfit on robust range)
m = j[(j.lfc_cno > -2.5) & (j.lfc_cno < 2.5)]
z = np.polyfit(m.lfc_cno, m.lfc_inf, 1)
xs = np.linspace(-2.3, 2.5, 10)
ax.plot(xs, np.polyval(z, xs), "--", color=TEAL, lw=1.6)
rho_all = j[["lfc_inf", "lfc_cno"]].corr(method="spearman").iloc[0, 1]
ax.set_xlabel("CNO vs HC  log2FC (RNA-seq)")
ax.set_ylabel("OM vs HC  log2FC (microarray)")
ax.set_title(f"A  Whole-genome effect sizes\nSpearman rho = {rho_all:.3f} (all {len(j):,} genes)")
ax.legend(loc="upper right", fontsize=7, frameon=False)
ax.axhline(0, color="#999999", lw=0.6); ax.axvline(0, color="#999999", lw=0.6)

ax = axs[1]
pc = pd.read_csv(os.path.join(RES, "A_program_contrast.csv"))
order = ["CNO-specific", "Infection-specific", "Shared inflammatory"]
cols = [ORANGE, RED, BLUE]
for i, (prog, c) in enumerate(zip(order, cols)):
    vals = np.random.default_rng(0)
    row = pc[pc.program == prog].iloc[0]
    # box from summary not possible -> use module-level genes? Use bar of mean delta instead
ax.barh([0, 1, 2], [pc.set_index("program").loc[p, "mean_delta"] for p in order],
        color=cols, height=0.55)
ax.set_yticks([0, 1, 2]); ax.set_yticklabels(
    [f"{p}\n(n={int(pc.set_index('program').loc[p, 'n'])})" for p in order], fontsize=8)
ax.axvline(0, color="#555555", lw=0.8)
ax.set_xlabel("Mean delta = log2FC(OM) - log2FC(CNO)")
ax.set_title("B  Programme-level contrast\n(Wilcoxon + permutation)")
for i, p in enumerate(order):
    r = pc.set_index("program").loc[p]
    ax.text(r.mean_delta, i + 0.38, f"perm p = {r.perm_p:.3f}",
            ha="center", fontsize=7.2, color="#333333")

ax = axs[2]
mc = pd.read_csv(os.path.join(RES, "A_module_comparison.csv")).sort_values("delta")
colors = [TEAL if d < 0 else RED for d in mc.delta]
ax.barh(mc.module, mc.delta, color=colors, height=0.6)
for y, (d, p) in enumerate(zip(mc.delta, mc.wilcoxon_p)):
    star = "**" if p < 0.01 else ("*" if p < 0.05 else "ns")
    ax.text(d + (0.05 if d >= 0 else -0.05), y, star, va="center",
            ha="left" if d >= 0 else "right", fontsize=8, color="#333333")
ax.axvline(0, color="#555555", lw=0.8)
ax.set_xlabel("Mean delta = log2FC(OM) - log2FC(CNO)")
ax.set_title("C  Immunometabolic modules\n(* p<0.05, ** p<0.01, Wilcoxon)")
fig.tight_layout(w_pad=2.2)
save(fig, "Figure_2")

# =====================================================================
# Figure 3  Signature performance: nested CV ROC, external ROC, scores
# =====================================================================
s = box_data()
fig, axs = plt.subplots(1, 3, figsize=(12.6, 4.0))

# (A) nested CV ROC
d = np.load(os.path.join(RES, "A_nested_cv_oof.npz"), allow_pickle=True)
y, oof, fpr, tpr = d["y"], d["oof"], d["fpr"], d["tpr"]
a_cv = auc(fpr, tpr)
ax = axs[0]
ax.plot(fpr, tpr, color=NAVY, lw=2, label=f"Nested 5-fold CV (AUC {a_cv:.3f})")
ax.plot([0, 1], [0, 1], "--", color="#AAAAAA", lw=1)
ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
ax.set_title("A  Discovery cohort (GSE30119)\nleakage-free nested cross-validation")
ax.legend(loc="lower right", fontsize=8, frameon=False)

# (B) external validation ROCs
ax = axs[1]
for ds, c, lab in [("GSE16129", TEAL, "GSE16129 (S. aureus 46 / HC 10)"),
                   ("GSE6269", ORANGE, "GSE6269 (S. aureus 31 / HC 6)")]:
    sub = s.loc[ds]
    yy = (sub.group == "SA").astype(int).values if ds == "GSE16129" else \
         (sub.group == "SA").astype(int).values
    f, t, a_ = roc(yy, sub.score.values)
    ax.plot(f, t, color=c, lw=1.8, label=f"{lab}\nAUC {a_:.3f}")
ax.plot([0, 1], [0, 1], "--", color="#AAAAAA", lw=1)
ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
ax.set_title("B  Independent replication\n(fixed 14-gene signature, unchanged)")
ax.legend(loc="lower right", fontsize=7.2, frameon=False)

# (C) score distributions
ax = axs[2]
groups = [("GSE30119", "HC", "HC\n(GSE30119)", "#BFBFBF"),
          ("GSE30119", "OM", "OM\n(GSE30119)", RED),
          ("GSE187429", "HC", "HC\n(GSE187429)", "#BFBFBF"),
          ("GSE187429", "CNO", "CNO\n(GSE187429)", ORANGE)]
data = [s.xs(ds, level=0).query("group == @grp").score.values for ds, grp, _, _ in groups]
bp = ax.boxplot(data, tick_labels=[g[2] for g in groups], patch_artist=True,
                widths=0.55, medianprops=dict(color="black", lw=1.4))
for patch, (_, _, _, c) in zip(bp["boxes"], groups):
    patch.set_facecolor(c); patch.set_alpha(0.75)
ax.axhline(0, color=NAVY, lw=0.9, ls="--")
ax.set_ylabel("14-gene signature score (rank, HC-centred)")
ax.set_title("C  CNO patients overlap their own\nhealthy controls (p = 0.85)")
fig.tight_layout(w_pad=2.2)
save(fig, "Figure_3")

# =====================================================================
# Figure 4  Pathway architecture (reuse enrichment panel, 2 panels)
# =====================================================================
en = pd.read_csv(os.path.join(RES, "A_enrichment_all.csv"))
fig, axs = plt.subplots(1, 2, figsize=(11.6, 4.2))
for ax, setname, c, title in [
        (axs[0], "Infection-specific", RED, "A  Infection-specific programme"),
        (axs[1], "Shared inflammatory", BLUE, "B  Shared inflammatory programme")]:
    sub = en[(en["set"] == setname)].nsmallest(10, "Adjusted P-value").iloc[::-1]
    terms = [t.split(" (GO")[0] for t in sub["Term"]]
    terms = [t if len(t) <= 52 else t[:49] + "..." for t in terms]
    ax.barh(terms, -np.log10(sub["Adjusted P-value"]), color=c, height=0.62)
    ax.set_xlabel("-log10 (adjusted p)")
    ax.set_title(title)
fig.tight_layout(w_pad=2.4)
save(fig, "Figure_4")

# =====================================================================
# Figure 5  Pathogen specificity (GSE6269)
# =====================================================================
s = box_data()
fig, ax = plt.subplots(figsize=(6.8, 4.4))
order = [("HC", 6, "HC\n(n=6)", "#BFBFBF"), ("S. aureus", 31, "S. aureus\n(n=31)", RED),
         ("E. coli", 29, "E. coli\n(n=29)", ORANGE),
         ("S. pneumoniae", 13, "S. pneumoniae\n(n=13)", TEAL),
         ("Influenza", 18, "Influenza\n(n=18)", BLUE)]
p6 = pd.read_csv(os.path.join(RES, "A_pathogen_scores.csv"), index_col=0)
data, labels, colors = [], [], []
for key, n_exp, lab, c in order:
    m = p6[p6.group == key]
    data.append(m.score.values); labels.append(lab); colors.append(c)
bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.55,
                medianprops=dict(color="black", lw=1.4))
for patch, c in zip(bp["boxes"], colors):
    patch.set_facecolor(c); patch.set_alpha(0.75)
ax.axhline(0, color=NAVY, lw=0.9, ls="--")
ax.set_ylabel("14-gene signature score (rank, HC-centred)")
ax.set_title("Pan-pathogen response in GSE6269\nKruskal-Wallis p = 2.0e-04;  S. aureus vs others AUC = 0.626")
save(fig, "Figure_5")

# =====================================================================
# Figure 6  Tissue transfer to implant-bone interface (3 panels)
# =====================================================================
g = pd.read_csv(os.path.join(RES, "GSE255786_groups.csv"))
d = pd.read_csv(os.path.join(RES, "A_GSE255786_gene_direction.csv"))
fig, axs = plt.subplots(1, 3, figsize=(12.6, 4.1))

# (A) ROCs in GSE255786
ax = axs[0]
yy = (g.group == "PJI").astype(int).values
for col, c, lab in [("score_A14", RED, "14-gene blood signature (AUC 0.582)"),
                    ("score_M28", TEAL, "28 published interface markers (AUC 0.985)"),
                    ("score_N3", NAVY, "3 interface-derived markers (AUC 0.996)")]:
    f, t, a_ = roc(yy, g[col].values)
    ax.plot(f, t, color=c, lw=1.8, label=f"{lab}")
ax.plot([0, 1], [0, 1], "--", color="#AAAAAA", lw=1)
ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
ax.set_title("A  GSE255786 sonicate fluid\nPJI 53 vs aseptic failure 40")
ax.legend(loc="lower right", fontsize=7, frameon=False)

# (B) per-gene direction concordance
ax = axs[1]
for k, (_, r) in enumerate(d.iterrows()):
    c = GREEN if r.direction_agree else RED
    ax.scatter(r.lfc_whole_blood, r.lfc_local, s=42, color=c, zorder=3)
    dy = (4, 4) if k % 2 == 0 else (4, -9)
    ax.annotate(r.gene, (r.lfc_whole_blood, r.lfc_local),
                textcoords="offset points", xytext=dy, fontsize=6.6, color="#333333")
ax.axhline(0, color="#999999", lw=0.7); ax.axvline(0, color="#999999", lw=0.7)
n_agree = int(d.direction_agree.sum())
ax.set_xlabel("log2FC in whole blood (OM vs HC)")
ax.set_ylabel("log2FC at implant-bone interface (PJI vs AF)")
ax.set_title(f"B  Direction concordance {n_agree}/12\n(rank correlation of effect sizes = -0.13)")
ax.text(0.98, 0.03, "green = concordant, red = discordant", transform=ax.transAxes,
        fontsize=7, ha="right", color="#555555")

# (C) AUC before/after cell adjustment + depth
ax = axs[2]
vals = [0.582, 0.536, 0.985]
labs = ["Blood score\nunadjusted", "Blood score\nafter cell adjustment",
        "Published interface\npanel (control)"]
cs = [RED, "#E08573", TEAL]
bars = ax.bar(range(3), vals, color=cs, width=0.55)
ax.axhline(0.5, color="#555555", lw=1, ls="--")
ax.text(2.42, 0.505, "chance", fontsize=7.5, color="#555555", va="bottom")
for i, v in enumerate(vals):
    ax.text(i, v + 0.015, f"{v:.3f}", ha="center", fontsize=8.5, fontweight="bold")
ax.set_xticks(range(3)); ax.set_xticklabels(labs, fontsize=7.4)
ax.set_ylim(0, 1.08); ax.set_ylabel("AUC")
ax.set_title("C  Cell composition does not rescue\ntransfer failure (depth: rho = 0.16, p = 0.13)")
fig.tight_layout(w_pad=2.2)
save(fig, "Figure_6")

# =====================================================================
# Figure S1  Study flow with excluded datasets
# =====================================================================
fig, ax = plt.subplots(figsize=(8.8, 6.6))
ax.set_xlim(0, 10); ax.set_ylim(0, 8.3); ax.axis("off")
ax.text(5, 8.1, "Study flow", ha="center", fontsize=13, fontweight="bold", color=NAVY)

def sbox(x, y, w, h, title, lines, ec, fs=7.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                                ec=ec, fc="#FFFFFF", lw=1.6))
    ax.text(x + w/2, y + h - 0.05, title, ha="center", va="top",
            fontsize=fs + 0.9, fontweight="bold", color=ec)
    for i, ln in enumerate(lines):
        ax.text(x + w/2, y + h - 0.40 - i*0.27, ln, ha="center", va="top",
                fontsize=fs, color="#333333")

sbox(0.4, 6.35, 3.4, 1.25, "RETRIEVAL",
     ["GEO queries: osteomyelitis, CNO,", "PJI, sonicate fluid; whole blood",
      "and implant-interface tissue"], NAVY)
sbox(0.4, 4.15, 3.4, 1.70, "EXCLUDED AFTER METADATA CHECK",
     ["GSE133378 (viral meningitis,", "not bone infection); GSE80178",
      "(diabetic foot ulcer microRNA,", "not mRNA)"], RED)
sbox(0.4, 1.55, 3.4, 1.85, "LABEL RECOVERY",
     ["GSE255786: no labels in GEO/SRA;", "recovered from Gene 2022",
      "Supplemental Table 2; 93/93 matched;", "3 consistency checks passed"], ORANGE)
sbox(4.6, 6.35, 5.0, 1.55, "INCLUDED COHORTS",
     ["Blood: GSE30119 (57/43), GSE187429 (11/9),",
      "GSE16129 (46/10), GSE6269 (60)",
      "Interface: GSE255786 (53/40)"], GREEN)
sbox(4.6, 4.15, 5.0, 1.85, "ANALYSES",
     ["1. Programme decomposition (shared vs disease-specific)",
      "2. 14-gene signature, nested CV + external validation",
      "3. CNO and pathogen specificity",
      "4. Tissue transfer + 4 exclusion tests"], NAVY)
sbox(4.6, 1.55, 5.0, 1.85, "QUALITY CONTROL",
     ["Cached matrix was linear, not log2 - restored before statistics;",
      "Feature selection inside every CV fold (leakage avoided);",
      "No merging of absolute values across platforms"], GREY)

def sarrow(x1, y1, x2, y2, color=NAVY):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=13, color=color, lw=1.4,
                                 shrinkA=2, shrinkB=2))
sarrow(3.85, 6.95, 4.55, 7.05)
sarrow(2.1, 6.3, 2.1, 5.9, color=RED)
sarrow(2.1, 4.1, 2.1, 3.45, color=ORANGE)
sarrow(3.85, 2.6, 4.55, 6.5)
sarrow(7.1, 6.3, 7.1, 6.05)
sarrow(7.1, 4.1, 7.1, 3.45)
save(fig, "Figure_S1")

# Figure S2  Module scores & sensitivity analyses
# =====================================================================
s = box_data()
fig, axs = plt.subplots(1, 2, figsize=(11.0, 4.2))

# (A) shared inflammatory module score across cohorts
ax = axs[0]
combos = [("GSE30119", "HC", "HC", "#BFBFBF"), ("GSE30119", "OM", "OM", RED),
          ("GSE187429", "HC", "HC", "#BFBFBF"), ("GSE187429", "CNO", "CNO", ORANGE),
          ("GSE16129", "HC", "HC", "#BFBFBF"), ("GSE16129", "SA", "S. aureus", TEAL),
          ("GSE6269", "HC", "HC", "#BFBFBF"), ("GSE6269", "SA", "S. aureus", TEAL)]
data, labs, cols = [], [], []
for ds, grp, lab, c in combos:
    data.append(s.xs(ds, level=0).query("group == @grp")["shared"].values)
    labs.append(f"{lab}\n({ds})"); cols.append(c)
bp = ax.boxplot(data, tick_labels=labs, patch_artist=True, widths=0.55,
                medianprops=dict(color="black", lw=1.2))
for patch, c in zip(bp["boxes"], cols):
    patch.set_facecolor(c); patch.set_alpha(0.75)
ax.axhline(0, color=NAVY, lw=0.9, ls="--")
ax.tick_params(axis="x", labelsize=6.8)
ax.set_ylabel("Shared inflammatory module score (HC-centred)")
ax.set_title("A  Shared cytokine/interferon module\nmodest and consistent across cohorts")

# (B) infection-specific module score across cohorts (main asymmetry)
ax = axs[1]
data2, labs2, cols2 = [], [], []
for ds, grp, lab, c in combos:
    data2.append(s.xs(ds, level=0).query("group == @grp")["score"].values)
    labs2.append(f"{lab}\n({ds})"); cols2.append(c)
bp = ax.boxplot(data2, tick_labels=labs2, patch_artist=True, widths=0.55,
                medianprops=dict(color="black", lw=1.2))
for patch, c in zip(bp["boxes"], cols2):
    patch.set_facecolor(c); patch.set_alpha(0.75)
ax.axhline(0, color=NAVY, lw=0.9, ls="--")
ax.tick_params(axis="x", labelsize=6.8)
ax.set_ylabel("Infection-specific module score (HC-centred)")
ax.set_title("B  Infection-specific module: large in bacterial\ninfection, absent in CNO (sensitivity: rank-based, HC-centred)")
fig.tight_layout(w_pad=2.4)
save(fig, "Figure_S2")

print("ALL FIGURES DONE ->", OUT)
