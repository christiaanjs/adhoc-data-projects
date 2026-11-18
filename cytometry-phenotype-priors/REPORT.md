# REPORT — do phenotype priors help mixture models identify gated cytometry populations?

**Short answer.** On the two multi-sample mass-cytometry sets they help.
- **Biggest gain: soft gating rules.** These are rules on marker levels taken from the publications. Turning them into a rule gate on p(k|x) gives the largest improvement over an unconstrained mixture.
- **Labels add more.** Combining the rules with one to three labelled reference samples beats the rules alone.
- **Ordering constraints alone do not help.** Constraining the marker orderings between populations neither keeps a component attached to the population it is meant to be, nor reliably improves Hungarian F1.
- **Flows: one kind earns its place.** A per-marker monotone flow, which acts as a learned per-sample normalisation, does. RealNVP-style coupling layers do not.
- **Rare populations: the priors fail.** On the two rare-population flow sets the priors are a clear negative result, because the automatic thresholds disagree with how the manual gates were drawn (§6).
- **FlowSOM still matches or beats the unsupervised prior models.** This is on the Weber & Robinson metric, with FlowSOM given k=40 clusters for 14–24 populations. FlowSOM, however, does not name its clusters.

Numbers: `results/results_table.md` (5 seeds; mean ± sd over seeds of the mean over test samples).
Figures: `results/figures/`.

---

## 1. Data (Phase 1)

All data are the Weber & Robinson (2016, *Cytometry A* 89:1084) benchmark files (FlowRepository
FR-FCM-ZZPH). They were obtained as the Bioconductor **HDCytoData** SummarizedExperiment objects from
the ExperimentHub bucket (`https://mghp.osn.xsede.org/bir190004-bucket01/ExperimentHub/HDCytoData/<name>/<name>_SE.rda`).
The raw `.rda` files are untouched (`data/SHA256SUMS_raw.txt`). They were extracted with base R
(`scripts/extract_hdcytodata.R`) and transformed in `src/cytoprior/data.py`.

| dataset | publication / label provenance | tech | transform | cells (labelled) | samples | markers | pops |
|---|---|---|---|---|---|---|---|
| Samusik_all | Samusik 2016 *Nat Methods*; consensus of 3 experts' hand gating | CyTOF | asinh(x/5) | 841,644 (514,386) | 10 mouse BM | 39 | 24 |
| Levine_32dim | Levine 2015 *Cell*; authors' manual gating | CyTOF | asinh(x/5) | 265,627 (104,184) | 2 (H1, H2) | 32 | 14 |
| Mosmann_rare | Mosmann 2014 *Cytometry A*; IFNγ+TNFα+ activated memory CD4 regated by W&R per Fig 4A | flow | asinh(x/150) | 396,460 (109) | 1 | 14 | 1 rare |
| Nilsson_rare | Rundberg Nilsson 2013 *Cytometry A*; HSC regated by W&R per Fig 2 | flow | asinh(x/150) | 44,140 (358) | 1 | 13 | 1 rare |
| Levine_13dim | Bendall 2011 *Science*; manual gating | CyTOF | asinh(x/5) | 167,044 (81,747) | 1 | 13 | 24 |

Clustering markers are those with `marker_class != "none"` in HDCytoData. Cells not in any gate are
coded −1 ("unassigned", or "other" in the rare sets). Levine_13dim is used only for the FlowSOM sanity check.

### Data you need to supply: FlowCAP-I (blocked)
`flowrepository.org` served an **expired TLS certificate** throughout this work (2026-11-18). TLS
verification was not disabled, so no FlowCAP-I data were downloaded and none were simulated. To add
them, place the following under `data/raw/flowcap1/<set>/`:

| set | FlowRepository ID | files needed |
|---|---|---|
| GvHD | FR-FCM-ZZY2 | all FCS files + the manual-gate label files (one label vector per FCS, FlowCAP-I "labels" CSVs) |
| WNV | FR-FCM-ZZY3 | same |
| DLBCL | FR-FCM-ZZYY | same |
| ND (normal donors) | FR-FCM-ZZYZ | same |

A loader stub would read FCS (e.g. `fcsparser`) plus the label CSVs. No priors were written for these
sets, because their gating definitions were never read.

## 2. Phenotype priors (Phase 2)

There is one prior file per dataset: `phenotype_priors/<dataset>.yaml`. Each records:
- the gating hierarchy;
- for each population, its marker levels (neg/dim/pos/bright) and citation keys;
- an `inferred` flag.

Per-marker ordering DAGs are derived automatically between consecutive level groups (`src/cytoprior/priors.py`).

**Honesty about sources.** Several gating supplements could not be read because of bot walls on publisher and PMC supplementary links: Samusik 2016 Supp Fig 5, Levine 2015 gating supplement, and Nilsson 2013 Fig 2 at full resolution. Where a population's definition comes from its name plus canonical marker definitions (Akashi 2000, Kondo 1997, Kiel 2005, Hardy 1991, Geissmann 2003, Maecker 2012 / HIPC, Notta 2011), it is flagged `inferred: true`:

| prior | populations | inferred | ordering edges | notes |
|---|---|---|---|---|
| Samusik_all | 24 | 22 entries flagged (21 of 24 pops) | 191 (on 39 markers) | MPP definition quoted verbatim from Samusik 2016 text |
| Levine_32dim | 14 | all 14 (lineage nodes canonical) | 83 | "CD34+ CD45low blasts" from Levine 2015 text |
| Mosmann_rare | 1 | 0 | — | live → CD3+CD14− → CD4+CD8a− → CD69+CD45RA− → IFNγ+TNFα+ (Mosmann 2014 Fig 4A) |
| Nilsson_rare | 1 | yes | — | Lin− CD34+ CD38− CD45RA− CD90+ CD49f+ (Notta 2011; matches Nilsson 2013 text) |

**Are the orderings true on held-out data?** Population medians of the *manual gates* on the test samples:
- Samusik 06–09: 0 of 191 edges violated.
- Samusik 10: 1 of 191 violated. Sca1 CLP 3.61 vs MPP 3.57, a tie in practice.
- Levine H2: 0 of 83 violated.

(`results/ordering_violations_heldout.json`.) So the encoded orderings are consistent with the expert gates. They are just not very informative for the mixture (§5).

## 3. Models (Phase 3)

`src/cytoprior/models.py`. PyTorch; device chosen automatically (cuda > mps > cpu; CPU here); seeds 0–4.

**PriorMixture** has K = K_known + K_unknown components:
- **Kernel.** Multivariate t or Gaussian, full covariance via a precision Cholesky factor.
- **Ordering prior.** Means are reparametrised so that, per marker, μ_hi = max(μ_lo over lower-level populations) + softplus(η), with a half-normal(σ=2) prior on the gaps. The ordering therefore holds by construction.
- **Rule gate.** log p(k|x) = log_softmax(log π + λ·s_k(x)). The score s_k is the mean over the population's specified markers of log-sigmoid soft memberships against per-sample, per-marker thresholds (t, h). These come from 1-D two-component Gaussian fits: t is the neg/pos boundary and h the pos/bright boundary. Unknown components get a constant score of log 0.5.
- **Semi-supervised (L5).** Labelled reference cells add a supervised log-likelihood term. Their "unassigned" cells are scored against the unknown components.
- **Flow (L6).** x → z through a per-marker monotone map (a·x + b + Σ w·tanh), which preserves every ordering, optionally followed by affine coupling layers. The mixture is fitted in z with the log-determinant term.
- **Fitting.** Minibatch Adam (4096 cells, 400 steps, cosine learning rate) on all cells of the sample. Initialisation defaults to soft-rule-weighted moments (`init: rules`); a rank-based variant (`L3_order_rankinit`) is kept as an ablation.

Baselines:
- `R_rules_classifier`: the rule gate alone, no mixture.
- `L0_gmm`: sklearn full-covariance GMM.
- `L2_flowsom`: python flowsom 0.2.2, 10×10 SOM, k=40.

Hyper-parameters were tuned on validation samples only (`scripts/tune_val.py`; Samusik 04, 05 and Levine H1). The grid was λ ∈ {1, 3, 10}, K_u ∈ {0, 6, 12}, τ ∈ {0.3, 0.6}, and the chosen values are λ=10, K_u=6, τ=0.6. The rare sets had no validation samples and were not tuned; they use K_u=15 (fixed a priori).

## 4. Evaluation (Phase 4)

- **Splits are by sample.**
  - Samusik: reference 01–03, validation 04–05, test 06–10.
  - Levine: H1 is both validation and reference, H2 is test.
  - The single-sample rare sets are evaluated in-sample, as in W&R.
- **Metrics.**
  - Hungarian-matched F1 per population and its macro mean (W&R procedure: unassigned cells removed, one-to-one matching).
  - **Identity F1**: component k is scored as population k without matching. This is the real annotation use and is available only for prior models.
  - ARI.
  - Rare F1: best-matching cluster for Hungarian; for identity, scored on *all* cells with "other" as negatives.
  - Ordering-violation rate of the matched model clusters' medians.
  - Reliability curves and ECE on gated cells.
  - Runtime, and peak RSS per worker process.

**Published-baseline reproduction** (`results/sanity_flowsom.md`). FlowSOM k=40, Hungarian mean F1, 5 seeds:

| dataset | ours | published |
|---|---|---|
| Levine_32dim | 0.751 ± 0.023 (range 0.725–0.794) | 0.780 |
| Samusik_all | 0.686 ± 0.016 | 0.702 |
| Levine_13dim | 0.552 | 0.495 |

Both main sets are within seed spread of the published value, so the metric and data pipeline reproduce W&R.

## 5. Results

RESULTS_PLACEHOLDER

## 6. Negative results and where the data pushed back

**Rare populations (Mosmann, Nilsson).** Every prior model is worse than FlowSOM, and most are worse than an unconstrained mixture.

The diagnosis is that the populations are defined by gates drawn *relative to a parent gate*, while the prior uses global per-sample thresholds:
- **Nilsson HSC.** CD90 and CD49f are "positive" only relative to the Lin−CD34+CD38− parent. Only 9% (CD90) and 13% (CD49f) of the gated HSCs pass the global thresholds, so the rule gate actively pushes the HSCs out of the HSC component.
- **Mosmann activated CD4.** CD14 on the gated cells has a median of about 2.67, above the bulk negative peak, so 79% fail the "CD14−" rule. The IFNγ/TNFα-positive cells sit at 5–6, but the two-component threshold lands inside the huge negative bulk (≈109 positives in 396k cells). A 1-D GMM cannot place a boundary for a 0.03% population.
- **Fixes tried and rejected.** None of these was adopted (main config unchanged):
  - MAD-based thresholds: worse everywhere.
  - Summing rule scores instead of averaging: Nilsson reached 0.60 only with an unknown score of log 0.02, which hurt Samusik and Levine validation.
  - Hierarchical thresholds re-fitted inside parent gates (`hierarchical_thresholds`): no rare-set gain; validation mixed (Levine H1 0.540 → 0.562, Samusik 04 0.590 → 0.576).

**Ordering alone doesn't keep identities.** Ordering constraints only bound means relative to each other. With 6 free unknown components, the "known" components still swap and merge (L3 identity F1 0.41–0.44, versus 0.57–0.60 for the rules). Rank-based initialisation without rules is worse still (0.18–0.29). Ordering adds almost nothing on top of rules (L4_rules vs L4_both: ±0.01).

**Small and transitional populations.** These stay hard for every model: HSC, CLP, γδ T, pro-B (Frac A-C), plasma cells, and the Classical/Intermediate monocyte split. The rule for Intermediate Monocytes has F1 0.01. It is defined by a Ly6C "dim/int" level that a two-level threshold cannot express.

## 7. Limitations

- **Splits.** Levine has only 2 samples, and H1 is both validation and reference, so its results rest on one test sample. The rare sets are single-sample, in-sample and untuned.
- **Inferred priors.** Most Samusik and Levine definitions are `inferred` from population names and canonical markers. The original gating supplements could not be read.
- **Initialisation confound.** Prior models use rules-based initialisation even in "order only" or "neither" variants (L3, L6_order, L6_neither, L5_neither_ref3 when rules are off). The init uses rule memberships, so ordering-only ablations are not fully free of rule information. The rank-init variant is the clean ordering-only control.
- **K differs from PLAN.md.** L0 and L1 use K = K_known + K_u (+6) rather than K = #populations as written there. FlowSOM gets k=40, which is more capacity than any other method, under the Hungarian metric that rewards over-clustering only through one-to-one matching.
- **Not implemented:** skew-t kernels and a Dirichlet-process number of unknowns.
- **Approximate figures.** Memory is ru_maxrss per worker process, so only approximate. Runtimes are single-thread CPU.
- **FlowCAP-I** not evaluated (§1).
