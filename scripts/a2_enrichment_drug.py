# -*- coding: utf-8 -*-
"""
A 方向 · 第二步：通路富集 + 药物重定位（Enrichr）
输入 a1 产出的 A_genes_infection_specific.csv / A_genes_shared.csv / A_genes_cno_specific.csv
输出 results/A_enrich_*.csv、A_drug_*.csv、figures/A_fig6_enrichment.png/pdf
"""
import os, time
import pandas as pd
import numpy as np
import gseapy as gp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(BASE, "results"); FIG = os.path.join(BASE, "figures")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "savefig.dpi": 300,
                     "savefig.bbox": "tight"})
NAVY, RED, ORANGE, TEAL = "#1f3b73", "#c0392b", "#e08b2c", "#16827e"

LIBS = ["GO_Biological_Process_2021", "MSigDB_Hallmark_2020",
        "Reactome_Pathways_2024", "WikiPathways_2024_Human"]
DRUG_LIBS = ["Drug_Perturbations_from_GEO_down", "Drug_Perturbations_from_GEO_up",
             "DSigDB", "DrugMatrix"]

def read_genes(fn, top=300):
    p = os.path.join(RES, fn)
    if not os.path.exists(p): return []
    d = pd.read_csv(p, index_col=0)
    return [str(g).upper() for g in d.index[:top] if str(g) != "nan"]

def run_enrichr(genes, libs, tag, timeout=90):
    out = []
    for lib in libs:
        try:
            e = gp.enrichr(gene_list=genes, gene_sets=[lib], outdir=None,
                           no_plot=True, verbose=False)
            r = e.results.copy()
            r["library"] = lib; r["set"] = tag
            out.append(r)
            print(f"  {tag} | {lib}: {len(r)} terms, "
                  f"min adjP={pd.to_numeric(r['Adjusted P-value'],errors='coerce').min():.2e}", flush=True)
        except Exception as ex:
            print(f"  {tag} | {lib} FAILED {type(ex).__name__} {str(ex)[:70]}", flush=True)
        time.sleep(0.6)
    return out

sets = {
    "Infection-specific": read_genes("A_genes_infection_specific.csv"),
    "Shared inflammatory": read_genes("A_genes_shared.csv"),
    "CNO-specific": read_genes("A_genes_cno_specific.csv"),
}
for k, v in sets.items(): print(k, len(v), "genes", flush=True)

all_en = []
for tag, genes in sets.items():
    if len(genes) < 10: continue
    all_en += run_enrichr(genes, LIBS, tag)
if all_en:
    en = pd.concat(all_en, ignore_index=True)
    en.to_csv(os.path.join(RES, "A_enrichment_all.csv"), index=False)

# 药物重定位：感染特异上调基因里取最靠前的（Δ 最大），找「逆转」药物
inf_up = read_genes("A_genes_infection_specific.csv", top=200)
print("\nDrug repositioning on", len(inf_up), "infection-specific genes", flush=True)
all_dr = run_enrichr(inf_up, DRUG_LIBS, "Infection-specific")
if all_dr:
    dr = pd.concat(all_dr, ignore_index=True)
    dr.to_csv(os.path.join(RES, "A_drug_enrichment.csv"), index=False)

# 图：三类程序的 Top 富集条目
fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
for ax, (tag, col) in zip(axes, [("Infection-specific", RED),
                                  ("Shared inflammatory", NAVY),
                                  ("CNO-specific", ORANGE)]):
    if not all_en:
        ax.axis("off"); continue
    sub = en[(en["set"] == tag) & (en.library == "GO_Biological_Process_2021")].copy()
    sub["adj"] = pd.to_numeric(sub["Adjusted P-value"], errors="coerce")
    sub = sub.sort_values("adj").head(10).sort_values("adj", ascending=False)
    if len(sub) == 0:
        ax.axis("off"); ax.set_title(tag); continue
    ax.barh(sub.Term.str[:42], -np.log10(sub.adj.clip(lower=1e-20)), color=col, alpha=0.85)
    ax.set_xlabel("-log10 adjusted P"); ax.set_title(tag, fontsize=10, fontweight="bold")
    ax.tick_params(labelsize=7.5); ax.grid(axis="x", alpha=0.25, lw=0.5)
fig.suptitle("Pathway enrichment of shared and disease-specific programs",
             fontsize=12, fontweight="bold", color=NAVY)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "A_fig6_enrichment.png"))
fig.savefig(os.path.join(FIG, "A_fig6_enrichment.pdf"))
plt.close(fig)

# 药物 Top 条形
if all_dr:
    d = dr[dr.library == "Drug_Perturbations_from_GEO_down"].copy()
    if len(d) == 0: d = dr.copy()
    d["adj"] = pd.to_numeric(d["Adjusted P-value"], errors="coerce")
    d = d.sort_values("adj").head(15).sort_values("adj", ascending=True)
    fig, ax = plt.subplots(figsize=(7.2, 5))
    ax.barh(d.Term.str[:48], -np.log10(d.adj.clip(lower=1e-20)), color=TEAL, alpha=0.85)
    ax.set_xlabel("-log10 adjusted P")
    ax.set_title("Candidate compounds reversing the infection-specific program",
                 fontsize=10, fontweight="bold")
    ax.grid(axis="x", alpha=0.25, lw=0.5); ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "A_fig7_drugs.png"))
    fig.savefig(os.path.join(FIG, "A_fig7_drugs.pdf"))
    plt.close(fig)

print("DONE", flush=True)
