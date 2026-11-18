# PLAN — phenotype priors for mixture-model population identification

Status: written 2026-11-18 at the start of the work. Nothing here needs >2 GB of download
or multi-hour training, so per the working rules I proceed without waiting; the points
that *do* need the user's decision are listed at the end.

## Hardware / constraints found
- 4 CPU cores, 15 GB RAM, **no GPU** (torch auto-selects cuda/mps/cpu; here cpu).
- Network was policy-restricted; a project environment was created allowing the data and
  literature hosts.
- **FlowRepository is unusable**: `flowrepository.org` serves an *expired TLS certificate*
  (2026-11-18). TLS verification will not be disabled. Consequences below.
- Publisher sites (Wiley, Cell, Science, Nature) and PMC's supplementary-file links are
  behind bot walls; Europe PMC REST + PMC article pages/figure CDN work.

## Datasets (all from HDCytoData = Bioconductor copies of the Weber & Robinson FR-FCM-ZZPH files)
| dataset | cells (labelled) | samples | markers used | transform | labels | role |
|---|---|---|---|---|---|---|
| Samusik_all | 841,644 (514,386) | 10 mouse BM replicates | 39 | asinh(x/5) | 24 pops, 3-expert consensus hand-gating (Samusik 2016) | **main multi-sample set** |
| Levine_32dim | 265,627 (104,184) | 2 (H1,H2) | 32 | asinh(x/5) | 14 pops, manual gating (Levine 2015) | main, 2-sample transfer |
| Mosmann_rare | 396,460 (109 rare) | 1 | 14 | asinh(x/150) | 1 rare pop, W&R regated per Mosmann 2014 Fig 4A | rare |
| Nilsson_rare | 44,140 (358 rare) | 1 | 13 | asinh(x/150) | 1 rare pop (HSC), W&R regated per Nilsson 2013 Fig 2 | rare |
| Levine_13dim | 167,044 (81,747) | 1 | 13 | asinh(x/5) | 24 pops (Bendall 2011) | sanity check only |
| FlowCAP-I (GvHD, WNV, DLBCL, ND) | — | — | — | — | — | **blocked** (FlowRepository) |

## Splits (by sample, never by cell)
- Samusik_all: reference pool (labelled, only used by L5) = 01,02,03; validation = 04,05;
  test = 06–10. All mixture models are fitted per sample (transductive, unlabelled) —
  this is how a gating replacement would be used on a new sample.
- Levine_32dim: H1 = reference/validation, H2 = test (only 2 samples: weak).
- Single-sample sets (rare, Levine_13dim): no held-out sample exists. Methods are run
  unsupervised in-sample (as in W&R); hyper-parameters are **not** tuned on them (taken from
  Samusik/Levine validation). Reported as in-sample, flagged.

## Models (ablation ladder; same splits/metrics/seeds everywhere)
- L0 Gaussian mixture (sklearn, full cov), K = #populations.
- L1 multivariate-t mixture (torch, full cov, learned dof), K = #populations.
- L2 FlowSOM (python `flowsom`, 10×10 SOM, consensus meta-clustering k=40 as W&R).
- L3 L1 + ordering-constrained means: per marker a DAG between level groups
  (neg<dim<pos<bright); reparam μ_b = max_{a∈pa(b)} μ_a + softplus(η_b), half-normal prior on
  gaps; plus K_u unknown components with free means.
- L4 L3 + rule gate p(k|x) ∝ π_k exp(λ s_k(x)), s_k = mean soft-threshold log-score over the
  population's specified markers; thresholds per sample per marker from 1-D 2-component fits.
- L5 L4 + labelled reference samples (0 / 1 / 3), joint likelihood with labelled cells clamped.
- L6 flow mixture: x → z by (i) per-marker *monotone* flow (preserves every ordering by
  construction, acts as a learned per-sample normalisation) then (ii) optional affine coupling
  layers; mixture base with ordered means + rule gate; unknown components 0/1/several.
- Prior ablations on L3–L6: ordering only / rules only / both / neither.
- 3b exploration: decided after seeing 3a results.

## Metrics
Hungarian-matched F1 per population (W&R procedure: unassigned cells excluded, one-to-one
matching maximising F1, unmatched pops F1=0), macro F1, ARI, rare-pop F1 (best cluster, W&R),
**identity F1** for prior models (component k *is* population k — the real annotation use),
ordering-violation rate of fitted/empirical population medians on held-out samples,
reliability curve + ECE, runtime, peak memory. 5 seeds, mean ± sd.
Sanity check: FlowSOM k=40 on Levine_32dim vs published 0.780 (W&R Supp Fig S1).

## Time estimate (CPU)
Loader+priors 1–2 h; models 2–3 h; dev runs on 10–20k-cell subsamples; full runs ≈1–2 h
wall on 4 cores (fits on ≤30k-cell subsample per sample, prediction on all cells);
report/figures 1 h.

## Needs user decision
1. FlowCAP-I: supply files (see REPORT.md, "Data you need to supply").
2. Phenotype sources: Levine 2015 Supp. gating, Bendall 2011 Fig S-gating, Samusik 2016
   Supp Fig 5, Nilsson 2013 Fig 2 could not be read (bot walls). Populations whose
   definitions rely on name + canonical markers are flagged `inferred: true`.
