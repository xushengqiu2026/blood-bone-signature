# -*- coding: utf-8 -*-
"""Table S1-S6 补充表工作簿（Clinical Immunology 投稿用）"""
import os
import pandas as pd

BASE = r"C:\Users\邱旭升\WorkBuddy\2026-11-03-10-41-59"
RES = os.path.join(BASE, "geo_min", "results")
OUT = os.path.join(BASE, "output", "20261103-nsfc", "stage3",
                   "submission_ClinicalImmunology")
os.makedirs(OUT, exist_ok=True)

# S1: sample-level labels + clinical variables, GSE255786 (93 samples)
s1 = pd.read_csv(os.path.join(RES, "GSE255786_groups.csv"))
s1 = s1.rename(columns={c: c for c in s1.columns})

# S2: full differential-expression results (all 11,622 genes)
s2 = pd.read_csv(os.path.join(RES, "A_joint_logFC.csv"))

# S3: gene lists with category
sh = pd.read_csv(os.path.join(RES, "A_genes_shared.csv"))
inf = pd.read_csv(os.path.join(RES, "A_genes_infection_specific.csv"))
cn = pd.read_csv(os.path.join(RES, "A_genes_cno_specific.csv"))
sh["category"] = "Shared"
inf["category"] = "Infection-specific"
cn["category"] = "CNO-specific"
s3 = pd.concat([sh, inf, cn], ignore_index=True)
s3 = s3.sort_values(["category", "delta"], ascending=[True, False])

# S4: over-representation results
s4 = pd.read_csv(os.path.join(RES, "A_enrichment_all.csv"))

# S5: drug repositioning
s5 = pd.read_csv(os.path.join(RES, "A_drug_enrichment.csv"))

# S6: per-gene blood vs interface comparison
s6 = pd.read_csv(os.path.join(RES, "A_GSE255786_gene_direction.csv"))

desc = {
    "S1": ("Sample-level labels and clinical variables, GSE255786 (93 sonicate-fluid samples). "
           "Labels recovered from Supplemental Table 2 of Masters TL et al., Gene 2022;825:146400. "
           "group: PJI = periprosthetic joint infection; AF = aseptic failure. score_A14 = 14-gene "
           "whole-blood signature; score_M28 = 28 published interface markers; score_N3 = 3 "
           "interface-derived markers (CCL20, F7, FCRL4)."),
    "S2": ("Differential-expression results for all 11,622 genes shared across platforms. "
           "lfc_inf/padj_inf: bacterial osteomyelitis (GSE30119) vs its healthy controls; "
           "lfc_cno/padj_cno: chronic non-bacterial osteomyelitis (GSE187429) vs its controls; "
           "delta = lfc_inf - lfc_cno."),
    "S3": ("Shared, infection-specific and CNO-specific gene lists with within-dataset statistics "
           "and delta. Category definitions are given in Methods 2.4."),
    "S4": ("Complete over-representation results (Enrichr) across GO Biological Process 2021, "
           "Reactome 2024 and MSigDB Hallmark 2020 for the infection-specific and shared "
           "programmes."),
    "S5": ("Drug-repositioning results (Drug Perturbations from GEO, DSigDB). Gene-set overlaps "
           "only; no claim about efficacy."),
    "S6": ("Per-gene comparison between whole blood (GSE30119, OM vs HC) and the implant-bone "
           "interface (GSE255786, PJI vs AF) for the 12 signature genes present on the interface "
           "platform. direction_agree: same sign of log2 fold-change."),
}

fp = os.path.join(OUT, "Table_S1-S6.xlsx")
with pd.ExcelWriter(fp, engine="openpyxl") as xw:
    for name, df in [("TableS1", s1), ("TableS2", s2), ("TableS3", s3),
                     ("TableS4", s4), ("TableS5", s5), ("TableS6", s6)]:
        key = name.replace("Table", "")
        df.to_excel(xw, sheet_name=name, index=False)
        ws = xw.sheets[name]
        ws.insert_rows(1)
        ws.cell(row=1, column=1, value=f"Table {key}. " + desc[key])
        ws.column_dimensions["A"].width = 110
        for col in ws[2]:
            pass
        # freeze header + bold it
        from openpyxl.styles import Font, PatternFill
        for c in ws[3]:
            c.font = Font(bold=True)
            c.fill = PatternFill("solid", fgColor="E8EEF5")
        ws.freeze_panes = "A4"
print("saved", fp, "| rows:", {k: len(v) for k, v in
      [("S1", s1), ("S2", s2), ("S3", s3), ("S4", s4), ("S5", s5), ("S6", s6)]})
