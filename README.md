# Blood and bone tell different stories — analysis code

Code for the manuscript:

> **Blood and bone tell different stories: whole-blood signatures detect bacterial osteomyelitis  
> but are near-orthogonal to the immune programme at the implant–bone interface**  
> Rongrong Zhang†, Ying Liu†, Wusimanjiang Hailipitimu, Yi Xiao, Xusheng Qiu*

All analyses use public, de-identified data. No human samples were collected for this work.

## Data (download from GEO before running)

| Accession | Compartment                        | Groups                                                                        | Platform         |
| --------- | ---------------------------------- | ----------------------------------------------------------------------------- | ---------------- |
| GSE30119  | whole blood                        | haematogenous osteomyelitis 57 / healthy 43                                   | GPL6947          |
| GSE187429 | whole blood                        | CNO 11 / healthy 9                                                            | RNA-seq (TPM)    |
| GSE16129  | whole blood                        | *S. aureus* 46 / healthy 10                                                   | GPL96            |
| GSE6269   | whole blood                        | *S. aureus* 31 / *S. pneumoniae* 13 / *E. coli* 29 / influenza 18 / healthy 6 (n=97, GPL96 subset) | GPL96            |
| GSE255786 | implant interface (sonicate fluid) | PJI 53 / aseptic failure 40                                                   | RNA-seq (counts) |

**GSE255786 labels** are not present in GEO/SRA. They were recovered from Supplemental Table 2 of  
Masters TL *et al.*, *Gene* 2022;825:146400 and are redistributed here as  
`results/GSE255786_groups.csv` (93 samples × clinical variables × three scores) so that other  
groups do not have to repeat the exercise. Please cite the original publication if you use them.

## Order of execution

```
python scripts/a0_ensg_map.py                  # Ensembl -> gene symbol mapping (mygene.info)
python scripts/a1_infection_vs_cno.py          # DE, shared / infection-specific / CNO-specific decomposition,
                                               # module scores, LASSO signature, ROC, enrichment input
python scripts/a1b_nested_cv.py                # leakage-free nested 5-fold CV (primary AUC estimate)
python scripts/a1c_pathogen_specificity.py     # GSE6269 five-group comparison, S. aureus vs others
python scripts/a2_enrichment_drug.py           # Enrichr over-representation + drug reverse matching
python scripts/a3_gse255786_labels_validate.py # label recovery, 3 consistency checks, transfer ROC
python scripts/a3b_niche_specificity.py        # four exclusion tests for the failed transfer
python scripts/a4_submission_figures.py        # Figures 1-6, S1-S2 (PNG 300 dpi + PDF)
python scripts/a5_make_tableS_xlsx.py          # Supplementary Tables S1-S6 (single workbook)
```

Scripts expect a sibling `data/` directory containing the GEO series matrices and a `results/`  
directory for outputs; both paths are defined at the top of each script and can be edited.

## Key results reproduced

| Quantity                                   | Value               | Script |
| ------------------------------------------ | ------------------- | ------ |
| Genes shared across platforms              | 11,622              | a0, a1 |
| DE genes, osteomyelitis (FDR<0.05)         | 5,081               | a1     |
| DE genes, CNO (FDR<0.05)                   | 0                   | a1     |
| Shared / infection-specific / CNO-specific | 26 / 1,867 / 5      | a1     |
| Nested CV AUC (leakage-free)               | 0.927 (0.860–0.978) | a1b    |
| External AUC, GSE16129 / GSE6269           | 0.939 / 0.935       | a1     |
| Blood signature at the interface           | 0.582 (0.465–0.692) | a3     |
| Published 28-marker interface panel        | 0.985 (0.957–1.000) | a3     |
| Blood vs interface correlation             | ρ = 0.147           | a3b    |

## Notes

- The cached expression matrix distributed with GSE30119 is stored as **linear** quantile-normalised  
  intensities, not log2 values. `a1_infection_vs_cno.py` restores the log2 scale before any  
  statistic (`np.log2(X.clip(lower=1))`); without it, log fold-changes reach the hundreds.
- Feature selection is repeated inside every cross-validation fold (`a1b_nested_cv.py`). The  
  apparent AUC of the fixed signature on the discovery data (0.998) is reported only to document  
  optimistic bias.
- Absolute expression values are never merged across platforms; only within-dataset effect  
  directions and internally centred rank scores are compared.

## Environment

Python 3.11; see `requirements.txt`. Tested on Windows 11 and Linux.

## Citation

If you use this code or the recovered sample labels, please cite the manuscript above. The code is
maintained at <https://github.com/xushengqiu2026/blood-bone-signature>, which is the canonical source
for the version used in the manuscript. Issues and corrections are welcome there.

## Licence

MIT (see `LICENSE`). The GEO data remain subject to the terms of the original depositors.
