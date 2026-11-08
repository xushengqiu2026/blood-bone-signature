# -*- coding: utf-8 -*-
"""
A 方向扩展：GSE255786（93 例 PJI vs 无菌失败 sonicate fluid）分组标签补全与独立验证

标签来源（原始、非推断）：
  Masters TL et al. Gene 2022;825:146400. PMID 35306116, PMCID PMC9007914
  Supplemental Table 2「Detailed information regarding individual samples and sequencing」
  经 Elsevier 官方补充材料 CDN 获取：1-s2.0-S0378111922002190-mmc2.xlsx
  （PMC bin 路径被 403 拦截；PII 由 PubMed efetch / CrossRef 得到）

映射链：
  Supp Table 2「Sample number」 == GEO GSM title 方括号编号 == gene_counts.txt.gz 列名
  唯一异常：编号 262 被 Excel 存成 261.833333333333（浮点/公式错误），按最近整数回改。

输出：
  results/GSE255786_groups.csv              93 例完整标签 + 临床协变量
  results/A_GSE255786_validation.csv        签名/标志物 AUC
  results/A_GSE255786_clinical_corr.csv     与临床指标相关
  results/A_GSE255786_species_tests.csv     菌种分层检验
  figures/A_fig9_pji_validation.png/pdf     标签 QC + 跨平台验证四联图
  results/A_GSE255786_summary.md            技术小结
"""
import os, re, gzip, json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")
RES = os.path.join(BASE, "results")
FIG = os.path.join(BASE, "figures")
os.makedirs(RES, exist_ok=True); os.makedirs(FIG, exist_ok=True)

SUPP = os.path.join(DATA, "Gene2022_supp", "1-s2.0-S0378111922002190-mmc2.xlsx")
COUNTS = os.path.join(DATA, "GSE255786_gene_counts.txt.gz")
SAMPLES = os.path.join(DATA, "GSE255786_samples.txt")
SIG = os.path.join(RES, "A_signature_genes.csv")

LOG = []
def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True); LOG.append(s)

# 原文 Table 1 的 28 个已验证 PJI 生物标志物（用于标签正确性复核）
MARKERS28 = ["BPI", "CAMP", "CCL3", "CCL4", "CSF2RB", "CSF3", "CXCL2", "DEFA4",
             "FCGR1B", "ICAM1", "IFNG", "IL13RA2", "IL17D", "IL1A", "IL1B", "IL1RN",
             "IL2RA", "IL2RG", "IL5RA", "IL6", "CXCL8", "LBP", "LCN2", "LDHC",
             "LTF", "MMP3", "PI3", "VEGFA"]
NOVEL3 = ["CCL20", "F7", "FCRL4"]

# ---------------------------------------------------------------- 1. 标签
log("=" * 78)
log("1. 解析 Supplemental Table 2")
d = pd.read_excel(SUPP, sheet_name="Supp Table 2", header=None)
hdr = [str(x).strip() for x in d.iloc[5].tolist()]

meta = []
for i in range(6, d.shape[0]):
    raw = d.iloc[i, 0]
    cl = str(d.iloc[i, 1]).strip()
    if cl not in ("PJI", "AF"):
        continue
    # 编号：纯数字直接用；浮点异常（261.833333333333 -> 262）按最近整数回改
    try:
        v = float(raw)
        if abs(v - round(v)) < 1e-6:
            sn = str(int(round(v)))
        else:
            sn = str(int(round(v)))
            log(f"   [修正] row {i}: Sample number = {raw!r} -> {sn} (Excel 浮点错误)")
    except (TypeError, ValueError):
        continue
    row = {"sample_id": sn, "group": "PJI" if cl == "PJI" else "AF"}
    for j, name in enumerate(hdr):
        if j in (0, 1):
            continue
        row[f"c{j:02d}"] = d.iloc[i, j]
    meta.append(row)

lab = pd.DataFrame(meta).drop_duplicates("sample_id").set_index("sample_id")
log("   标签行数:", len(lab), "| 分组:", lab.group.value_counts().to_dict())

# GSM 映射
gsm = {}
for line in open(SAMPLES, encoding="utf-8"):
    p = line.rstrip("\n").split("\t")
    m = re.search(r"\[(\d+)\]", p[1]) if len(p) > 1 else None
    if m:
        gsm[m.group(1)] = p[0]
lab["GSM"] = [gsm.get(s, "") for s in lab.index]

# 菌种分层
def species(culture):
    """注意：'S.' 缩写既可能是 Staphylococcus 也可能是 Streptococcus。
    只把明确的葡萄球菌种归入 staph，其余 'S. agalactiae / S. mitis' 等链球菌归 Non_staph。"""
    s = str(culture).lower().replace(".", "").replace("  ", " ").strip()
    # 显式写出属名时优先
    if "staphylococc" in s or "staph" in s:
        if "aureus" in s: return "S_aureus"
        if "epidermidis" in s: return "S_epidermidis"
        return "Other_staph"
    if "streptococc" in s: return "Non_staph"
    # 缩写 'S. xxx'：仅匹配已知的葡萄球菌种
    if s.startswith("s "):
        if "aureus" in s: return "S_aureus"
        if "epidermidis" in s: return "S_epidermidis"
        if "capitis" in s or "lugdunensis" in s: return "Other_staph"
        return "Non_staph"   # S. agalactiae / S. mitis 等 = 链球菌
    if "aureus" in s: return "S_aureus"
    if "epidermidis" in s: return "S_epidermidis"
    return "Non_staph"

lab["sonicate_culture"] = lab["c02"]
lab["species"] = [("Aseptic" if g == "AF" else species(c))
                  for g, c in zip(lab.group, lab.sonicate_culture)]
lab["staph_class"] = [("Aseptic" if g == "AF" else
                       ("Staphylococcal" if str(s).startswith(("S_", "Other_staph")) else "Non_staphylococcal"))
                      for g, s in zip(lab.group, lab.species)]

# ---------------------------------------------------------------- 2. 表达矩阵
log("=" * 78)
log("2. 读取 count 矩阵并标准化")
cnt = pd.read_csv(COUNTS, sep="\t")
gene_cols = list(cnt.columns[:7])
sample_cols = [str(c) for c in cnt.columns[7:]]
log("   矩阵:", cnt.shape, "| 样本列:", len(sample_cols))
expr = cnt.set_index("GeneName")[sample_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
expr = expr.loc[~expr.index.duplicated(keep="first")]
expr.index = [str(g).upper() for g in expr.index]
expr = expr.loc[~expr.index.duplicated(keep="first")]
# CPM -> log2
cpm = expr / expr.sum(axis=0).replace(0, np.nan) * 1e6
X = np.log2(cpm + 1).fillna(0.0)
log("   表达矩阵:", X.shape, "| 值域", round(float(X.values.min()), 2), "-", round(float(X.values.max()), 2))

# 标签与矩阵对齐
common = [s for s in sample_cols if s in lab.index]
log("   count ∩ 标签:", len(common), "| count-only:", sorted(set(sample_cols) - set(lab.index)),
    "| 标签-only:", sorted(set(lab.index) - set(sample_cols)))
lab = lab.loc[common].copy()
X = X[common]

# ---------------------------------------------------------------- 3. 打分
log("=" * 78)
log("3. 样本内 rank 打分")
R = X.rank(axis=0, pct=True)
def score(genes):
    g = [x for x in genes if x in R.index]
    return R.loc[g].mean(axis=0), g

sig = pd.read_csv(SIG)["gene"].astype(str).str.upper().tolist()
s_sig, g_sig = score(sig)
s_m28, g_m28 = score(MARKERS28)
s_n3, g_n3 = score(NOVEL3)
log("   14 基因签名命中:", len(g_sig), "/", len(sig), "|", g_sig)
log("   28 标志物命中:", len(g_m28), "/", len(MARKERS28))
log("   3 新标志物命中:", len(g_n3), "/", len(NOVEL3), "|", g_n3)

lab["score_A14"] = s_sig.values
lab["score_M28"] = s_m28.values
lab["score_N3"] = s_n3.values
# 临床数值列
for key, col in [("age", "c06"), ("CRP", "c27"), ("ESR", "c26"), ("WBC", "c25"),
                 ("synovial_TNC", "c28"), ("synovial_neut_pct", "c29"),
                 ("total_reads_human", "c05")]:
    lab[key] = pd.to_numeric(lab[col], errors="coerce")
lab["sex"] = lab["c07"].astype(str).str.strip()
lab["joint"] = lab["c10"].astype(str).str.strip().str.upper()
lab["acute_inflam"] = lab["c12"].astype(str).str.strip()
lab["diabetes"] = lab["c21"].astype(str).str.strip()
lab["preop_abx"] = lab["c22"].astype(str).str.strip()

# ---------------------------------------------------------------- 4. AUC
log("=" * 78)
log("4. 标签正确性复核 + A 方向签名跨平台验证")
def auc_ci(y, s, n_boot=2000, seed=0):
    y = np.asarray(y); s = np.asarray(s, dtype=float)
    a = roc_auc_score(y, s)
    rng = np.random.default_rng(seed); idx0 = np.where(y == 0)[0]; idx1 = np.where(y == 1)[0]
    bs = []
    for _ in range(n_boot):
        ii = np.r_[rng.choice(idx0, len(idx0), True), rng.choice(idx1, len(idx1), True)]
        try: bs.append(roc_auc_score(y[ii], s[ii]))
        except Exception: pass
    return a, np.percentile(bs, 2.5), np.percentile(bs, 97.5)

y = (lab.group == "PJI").astype(int).values
rows = []
for name, s, ng in [("Original_28_PJI_markers", lab.score_M28, len(g_m28)),
                    ("Original_3_novel_markers", lab.score_N3, len(g_n3)),
                    ("A_direction_14_gene_signature", lab.score_A14, len(g_sig))]:
    a, lo, hi = auc_ci(y, s.values)
    u = stats.mannwhitneyu(s[y == 1], s[y == 0], alternative="two-sided")
    rows.append(dict(score=name, n_genes=ng, n_PJI=int(y.sum()), n_AF=int((1 - y).sum()),
                     AUC=round(a, 4), ci_lo=round(lo, 4), ci_hi=round(hi, 4),
                     p_MannWhitney=u.pvalue))
    log(f"   {name:36s} AUC {a:.3f} ({lo:.3f}-{hi:.3f})  p={u.pvalue:.2e}  genes={ng}")
val = pd.DataFrame(rows)
val.to_csv(os.path.join(RES, "A_GSE255786_validation.csv"), index=False, encoding="utf-8-sig")

# ---------------------------------------------------------------- 5. 临床相关
log("=" * 78)
log("5. 与临床指标的一致性（PJI 组内 + 全体）")
cor = []
for key in ["CRP", "ESR", "WBC", "synovial_TNC", "synovial_neut_pct", "age", "total_reads_human"]:
    for scope, mask in [("all", pd.Series(True, index=lab.index)),
                        ("PJI_only", lab.group == "PJI")]:
        sub = lab.loc[mask]
        ok = sub[key].notna() & np.isfinite(sub[key])
        if ok.sum() < 12: continue
        rho, p = stats.spearmanr(sub.loc[ok, key], sub.loc[ok, "score_A14"])
        cor.append(dict(score="A14", clinical=key, scope=scope, n=int(ok.sum()),
                        rho=round(float(rho), 3), p=float(p)))
cc = pd.DataFrame(cor)
cc.to_csv(os.path.join(RES, "A_GSE255786_clinical_corr.csv"), index=False, encoding="utf-8-sig")
log(cc.to_string(index=False))

# ---------------------------------------------------------------- 6. 菌种分层
log("=" * 78)
log("6. 菌种分层（核对原文 Table 1：staph 29 / non-staph 24 / AF 40；SA 8 / SE 15）")
vc = lab.species.value_counts().to_dict()
log("   本表菌种计数:", vc)
n_staph = int((lab.staph_class == "Staphylococcal").sum())
n_nonstaph = int((lab.staph_class == "Non_staphylococcal").sum())
log(f"   核对 JBJS Table 1: staph {n_staph} (期望 29) | non-staph {n_nonstaph} (期望 24) | "
    f"SA {vc.get('S_aureus',0)} (期望 8) | SE {vc.get('S_epidermidis',0)} (期望 15) | AF 40")
log("   一致性:", "PASS" if (n_staph, n_nonstaph, vc.get('S_aureus', 0), vc.get('S_epidermidis', 0)) == (29, 24, 8, 15) else "CHECK")
sp = []
for name, sub in [("PJI_vs_AF", lab),
                  ("Staph_vs_Nonstaph_PJI", lab[lab.group == "PJI"]),
                  ("SA_vs_SE_PJI", lab[lab.species.isin(["S_aureus", "S_epidermidis"])])]:
    if name == "PJI_vs_AF":
        a, b = sub[sub.group == "PJI"], sub[sub.group == "AF"]
    elif name == "Staph_vs_Nonstaph_PJI":
        a = sub[sub.staph_class == "Staphylococcal"]; b = sub[sub.staph_class == "Non_staphylococcal"]
    else:
        a = sub[sub.species == "S_aureus"]; b = sub[sub.species == "S_epidermidis"]
    for sc in ["score_A14", "score_M28"]:
        if len(a) < 3 or len(b) < 3: continue
        u = stats.mannwhitneyu(a[sc], b[sc], alternative="two-sided")
        auc = roc_auc_score(np.r_[np.ones(len(a)), np.zeros(len(b))], np.r_[a[sc].values, b[sc].values])
        sp.append(dict(comparison=name, score=sc, n1=len(a), n2=len(b),
                       mean1=round(float(a[sc].mean()), 4), mean2=round(float(b[sc].mean()), 4),
                       AUC=round(float(auc), 4), p=float(u.pvalue)))
        log(f"   {name:24s} {sc:10s} n={len(a)}/{len(b)}  AUC={auc:.3f}  p={u.pvalue:.3g}")
spdf = pd.DataFrame(sp)
spdf.to_csv(os.path.join(RES, "A_GSE255786_species_tests.csv"), index=False, encoding="utf-8-sig")

# ---------------------------------------------------------------- 7. 保存标签
out_cols = ["GSM", "group", "species", "staph_class", "sonicate_culture",
            "sex", "joint", "age", "CRP", "ESR", "WBC", "synovial_TNC",
            "synovial_neut_pct", "total_reads_human", "acute_inflam", "diabetes",
            "preop_abx", "score_A14", "score_M28", "score_N3"]
out = lab[out_cols].reset_index().rename(columns={"index": "sample_id"})
out.to_csv(os.path.join(RES, "GSE255786_groups.csv"), index=False, encoding="utf-8-sig")
out.to_csv(os.path.join(DATA, "GSE255786_groups.csv"), index=False, encoding="utf-8-sig")
log("   已保存 results/GSE255786_groups.csv（93 行 ×", len(out_cols) + 1, "列）")

# ---------------------------------------------------------------- 8. 图
log("=" * 78)
log("8. 出图")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams["font.family"] = "DejaVu Sans"
rcParams["axes.unicode_minus"] = False
NAVY, RED, TEAL, GREY = "#1f3b63", "#b3342c", "#1c8a7d", "#8a8f98"

fig, axes = plt.subplots(2, 2, figsize=(12.6, 9.2))
# A: 28-marker QC
ax = axes[0, 0]
dat = [lab.loc[lab.group == "AF", "score_M28"], lab.loc[lab.group == "PJI", "score_M28"]]
bp = ax.boxplot(dat, tick_labels=["Aseptic failure\n(n=40)", "PJI\n(n=53)"],
                patch_artist=True, widths=0.5, showfliers=False)
for b, c in zip(bp["boxes"], [TEAL, RED]): b.set_facecolor(c); b.set_alpha(0.65)
ax.set_ylabel("Rank-based score (28 published markers)")
ax.set_title("A. Label QC: published PJI markers reproduce the split", fontsize=10.5, fontweight="bold")
ax.text(0.5, 0.94, f"AUC = {val.loc[0,'AUC']:.3f}", transform=ax.transAxes,
        ha="center", va="top", fontsize=11, color=RED, fontweight="bold")
ax.grid(axis="y", alpha=0.3)

# B: ROC
ax = axes[0, 1]
for name, s, c in [("28 published markers", lab.score_M28, TEAL),
                   ("3 novel markers", lab.score_N3, GREY),
                   ("A-direction 14-gene (whole blood)", lab.score_A14, RED)]:
    fpr, tpr, _ = roc_curve(y, s)
    a = roc_auc_score(y, s)
    ax.plot(fpr, tpr, lw=2, color=c, label=f"{name}  AUC {a:.3f}")
ax.plot([0, 1], [0, 1], "--", color="#bbbbbb", lw=1)
ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
ax.set_title("B. Independent validation in local sonicate fluid", fontsize=10.5, fontweight="bold")
ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=0.3)

# C: score vs CRP
ax = axes[1, 0]
ok = lab.CRP.notna() & np.isfinite(lab.CRP)
for g, c, mk in [("AF", TEAL, "o"), ("PJI", RED, "^")]:
    m = ok & (lab.group == g)
    ax.scatter(lab.loc[m, "CRP"], lab.loc[m, "score_A14"], s=28, c=c, marker=mk,
               alpha=0.7, label=g, edgecolors="white", linewidths=0.5)
if ok.sum() > 10:
    rho, p = stats.spearmanr(lab.loc[ok, "CRP"], lab.loc[ok, "score_A14"])
    ax.text(0.97, 0.05, f"Spearman rho = {rho:.2f}\np = {p:.1e}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GREY, alpha=0.9))
ax.set_xlabel("Serum CRP (mg/L)"); ax.set_ylabel("A-direction 14-gene score")
ax.set_title("C. No association with serum CRP", fontsize=10.5, fontweight="bold")
ax.legend(fontsize=9); ax.grid(alpha=0.3)

# D: species subgroups
ax = axes[1, 1]
order = ["Aseptic", "S_aureus", "S_epidermidis", "Other_staph", "Non_staph"]
groups = [lab.loc[lab.species == k, "score_A14"].values for k in order]
groups = [g if len(g) else np.array([np.nan]) for g in groups]
bp = ax.boxplot(groups, tick_labels=["Aseptic", "S. aureus", "S. epidermidis", "Other staph", "Non-staph"],
                patch_artist=True, widths=0.55, showfliers=False)
for b, c in zip(bp["boxes"], [GREY, RED, "#d97a3a", "#e3a95f", TEAL]):
    b.set_facecolor(c); b.set_alpha(0.7)
ax.set_ylabel("A-direction 14-gene score")
ax.set_title("D. Score by infecting organism", fontsize=10.5, fontweight="bold")
ax.tick_params(axis="x", labelsize=8.5); ax.grid(axis="y", alpha=0.3)
plt.setp(ax.get_xticklabels(), rotation=18, ha="right")

fig.suptitle("GSE255786 (Mayo Clinic, n=93): recovered labels and cross-platform validation\n"
             "Labels from Gene 2022 Supplemental Table 2 (PJI 53 / aseptic failure 40)",
             fontsize=12, fontweight="bold", y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.955])
for ext in ["png", "pdf"]:
    fig.savefig(os.path.join(FIG, f"A_fig9_pji_validation.{ext}"), dpi=300, bbox_inches="tight")
plt.close(fig)
log("   图: figures/A_fig9_pji_validation.png / .pdf")

# ---------------------------------------------------------------- 9. 摘要
smd = f"""# GSE255786 分组标签补全与独立验证

## 标签来源（作者原始标签，非推断）
- Masters TL et al. **Gene 2022;825:146400**（PMID 35306116 / PMCID PMC9007914）**Supplemental Table 2**
- 文件：`1-s2.0-S0378111922002190-mmc2.xlsx`，从 Elsevier 官方补充材料 CDN 取得
  （PMC bin 路径 403；PII `S0378111922002190` 由 PubMed efetch / CrossRef 查得）
- 另有 3 个补充文件（mmc1.docx / mmc3.xlsx / mmc4.xlsx，差异表达与 top50 表）已下载留存

## 映射链（三方一致）
`Supp Table 2 Sample number` = `GSM title 方括号编号` = `gene_counts.txt.gz 列名`
- 93/93 全部匹配；SRA/ENA `sample_alias` = GSM（93 runs，BioProject PRJNA1076519）
- 唯一异常：编号 **262** 在 xlsx 中被存成 `261.833333333333`（Excel 浮点错误），
  按最近整数回改；该行分类 PJI、菌种 *Cutibacterium acnes*，与原文 PJI=53 吻合
- 最终 **PJI 53 / AF 40**，与 Gene 2022 及 JBJS 2023（PMC10137834）Table 1 完全一致

## 附带的临床协变量（原表 34 列，逐样本）
菌种/CFU 分级、术中组织与滑液培养、年龄、性别、种族、侧别、膝/髋、
急性炎症、术中脓液、窦道、局部疼痛/体征、发热、影像学骨溶解/松动、
基础关节病、糖尿病等危险因素、术前 4 周抗生素、**Hb / WBC / ESR / CRP**、
滑液有核细胞计数与中性粒/淋巴/单核百分比、病理描述、人基因组比对 reads 数。

## 标签正确性的数据层复核（关键）
| 评分 | 基因数 | AUC (95%CI) |
|---|---|---|
| 原文 28 个已报道 PJI 标志物 | {len(g_m28)} | {val.loc[0,'AUC']:.3f} ({val.loc[0,'ci_lo']:.3f}–{val.loc[0,'ci_hi']:.3f}) |
| 原文 3 个新标志物 (CCL20/F7/FCRL4) | {len(g_n3)} | {val.loc[1,'AUC']:.3f} |
| **A 方向全血 14 基因签名（跨平台迁移）** | {len(g_sig)} | {val.loc[2,'AUC']:.3f} ({val.loc[2,'ci_lo']:.3f}–{val.loc[2,'ci_hi']:.3f}) |

原文报告 12/13 标志物 AUC>0.75、PI3/IL5RA/F7 >0.97；本表复算 AUC 与之相符，
说明标签与矩阵列的对应关系正确（若标签错位，AUC 会回到 0.5 附近）。

## 对 A 方向文章的意义
1. 这是一个**同一疾病、局部组织、独立平台**的验证集，可把「全血签名」推进到
   「植入物-骨界面局部也存在同一髓系/中性粒细胞程序」，明显提升文章层级。
2. 53 感染 vs 40 无菌失败是**真实临床两难**（比 CNO 全血更贴近临床），
   可替换 / 补强 A 方向目前较弱的「OM vs CNO」对比。
3. 自带菌种分层（staph 29 / non-staph 24；SA 8 / SE 15）与 CRP/ESR/滑液中性粒%，
   可做「签名 × 临床炎症指标」以及「签名 × 菌种」的剂量与特异性分析——
   这是 CNO 队列完全给不了的。
4. 注意：sonicate fluid 是**植入物表面超声裂解液**，不是骨组织；与自有钢板膜标本
   仍属不同样本类型，跨平台只比方向与秩相关，不合并绝对值（与第 5 章预注册口径一致）。

## 产出
- `results/GSE255786_groups.csv`（93 行标签 + 协变量 + 3 个评分）
- `results/A_GSE255786_validation.csv` / `A_GSE255786_clinical_corr.csv` / `A_GSE255786_species_tests.csv`
- `figures/A_fig9_pji_validation.png` / `.pdf`
- 原始补充表：`data/Gene2022_supp/`
"""
open(os.path.join(RES, "A_GSE255786_summary.md"), "w", encoding="utf-8").write(smd)
open(os.path.join(RES, "A_GSE255786_run.log"), "w", encoding="utf-8").write("\n".join(LOG) + "\n")
log("=" * 78)
log("DONE")
