# Sanity check: FlowSOM (10x10 SOM, consensus meta-clustering k=40), all cells, Hungarian mean F1
| dataset | published (W&R 2016 Supp Fig S1-S4, single run, R FlowSOM) | this pipeline (python flowsom 0.2.2), 5 seeds |
|---|---|---|
| Levine_32dim | 0.780 | 0.751 ± 0.023 (0.725–0.794) |
| Samusik_all | 0.702 | 0.686 ± 0.016 (0.667–0.705) |
| Levine_13dim | 0.495 | 0.552 ± 0.003 |
Published values read from bar labels in the W&R supplementary PDF
(raw.githubusercontent.com/lmweber/cytometry-clustering-comparison/master/supplementary_files/).
