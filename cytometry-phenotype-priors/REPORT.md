# REPORT — do phenotype priors help mixture models identify gated cytometry populations?

**Short answer.** On the two multi-sample mass-cytometry sets they help.
- **Biggest gain: soft gating rules.** These are rules on marker levels taken from the publications. Turning them into a rule gate on p(k|x) gives the largest improvement over an unconstrained mixture.
- **Labels add more.** Combining the rules with one to three labelled reference samples beats the rules alone.
- **Ordering constraints alone do not help.** Constraining the marker orderings between populations neither keeps a component attached to the population it is meant to be, nor reliably improves Hungarian F1.
- **Flows: one kind earns its place on Samusik.** A per-marker monotone flow, which acts as a learned per-sample normalisation, earns its place there. With 3 labelled samples it is the best model: 0.754 test and 0.765 validation macro F1, against FlowSOM's 0.688 and 0.675. RealNVP-style coupling layers do not earn theirs. On Levine, with only 2 samples, the flow results are inconsistent between test and validation.
- **Rare populations: the priors fail.** On the two rare-population flow sets the priors are a clear negative result, because the automatic thresholds disagree with how the manual gates were drawn (§6).
- **FlowSOM is hard to beat without labels.** On the Weber & Robinson metric, with k=40 clusters for 14–24 populations, it beats every *unsupervised* prior model on Samusik (test and validation) and on Levine validation. On Levine test the best unsupervised prior model edges it, 0.712 vs 0.698, which is within seed spread. Unlike FlowSOM, the prior models name their clusters. Only the label-using models beat FlowSOM clearly.

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

Full tables: `results/results_table.md` and `.csv`. Raw per-run records: `results/runs/*_test.jsonl` (985 runs, 0 failures).

### 5.1 Main table: macro F1 on held-out test samples (mean ± sd over 5 seeds)

| method | priors | labels | Samusik Hungarian | Samusik identity | Levine Hungarian | Levine identity |
|---|---|---|---|---|---|---|
| L0 GMM | – | – | 0.486 ± 0.006 | – | 0.522 ± 0.024 | – |
| L1 t-mixture | – | – | 0.529 ± 0.004 | – | 0.549 ± 0.036 | – |
| L2 FlowSOM (k=40) | – | – | 0.688 ± 0.007 | – | 0.698 ± 0.029 | – |
| R rules only (no mixture) | rules | – | 0.559 | 0.557 | 0.592 | 0.563 |
| L3 t + ordering | order | – | 0.537 ± 0.005 | 0.441 ± 0.008 | 0.643 ± 0.034 | 0.413 ± 0.036 |
| L3 t + ordering, rank init (no rule info) | order | – | 0.473 ± 0.011 | 0.182 ± 0.017 | 0.596 ± 0.012 | 0.294 ± 0.030 |
| L4 t + rules | rules | – | 0.604 ± 0.002 | 0.565 ± 0.001 | 0.658 ± 0.008 | 0.596 ± 0.007 |
| L4 t + both | both | – | 0.613 ± 0.003 | 0.574 ± 0.002 | 0.652 ± 0.008 | 0.592 ± 0.008 |
| L5 t, neither, 3 refs | – | 3 | 0.670 ± 0.002 | 0.665 ± 0.003 | n/a | n/a |
| L5 t, order, 3 refs | order | 3 | 0.671 ± 0.001 | 0.667 ± 0.002 | n/a | n/a |
| L5 t, rules, 3 refs | rules | 3 | 0.692 ± 0.002 | 0.686 ± 0.002 | n/a | n/a |
| L5 t, both, 1 ref | both | 1 | 0.688 ± 0.003 | 0.681 ± 0.003 | 0.699 ± 0.041 | 0.667 ± 0.051 |
| L5 t, both, 3 refs | both | 3 | 0.701 ± 0.002 | 0.694 ± 0.002 | n/a | n/a |
| L6 marginal+coupling flow, neither | – | – | 0.582 ± 0.007 | – | 0.534 ± 0.033 | – |
| L6 marginal+coupling flow, order | order | – | 0.552 ± 0.007 | 0.431 ± 0.019 | 0.498 ± 0.022 | 0.282 ± 0.031 |
| L6 marginal+coupling flow, rules | rules | – | 0.623 ± 0.003 | 0.592 ± 0.003 | 0.626 ± 0.034 | 0.529 ± 0.013 |
| L6 marginal+coupling flow, both | both | – | 0.628 ± 0.004 | 0.597 ± 0.004 | 0.614 ± 0.027 | 0.523 ± 0.011 |
| L6 marginal flow only, both | both | – | 0.637 ± 0.005 | 0.609 ± 0.004 | 0.699 ± 0.016 | 0.676 ± 0.018 |
| *3b* marginal flow, neither | – | – | 0.591 ± 0.003 | – | 0.561 ± 0.027 | – |
| *3b* marginal flow, order | order | – | 0.563 ± 0.007 | 0.449 ± 0.021 | 0.584 ± 0.009 | 0.420 ± 0.049 |
| *3b* marginal flow, rules | rules | – | 0.627 ± 0.004 | 0.599 ± 0.004 | 0.703 ± 0.013 | 0.681 ± 0.012 |
| *3b* marginal flow + t kernel, both | both | – | 0.656 ± 0.002 | 0.617 ± 0.002 | **0.712 ± 0.013** | 0.689 ± 0.011 |
| *3b* marginal flow, both, 1 ref | both | 1 | 0.738 ± 0.002 | 0.734 ± 0.002 | 0.703 ± 0.022 | **0.702 ± 0.021** |
| *3b* marginal flow, both, 3 refs | both | 3 | **0.754 ± 0.002** | **0.752 ± 0.001** | n/a | n/a |

Notes on the table:
- **Two metrics.** Hungarian F1 is Weber & Robinson's metric, in which each cluster is matched to its best population after the fact. Identity F1 needs no matching: component k is scored as population k, which is how a model would actually be used to name cells.
- **Missing cells.** Levine has only one reference sample (H1), so its 3-reference cells are n/a.
- **Post-hoc rows.** Rows marked *3b* were added after seeing the 3a test results; their validation scores are in §5.3.
- **Context.** The best method in Weber & Robinson on Samusik_all (FlowSOM, 0.702 published) sits between the unsupervised prior models and the semi-supervised ones.

### 5.2 Answers to the Phase-3 questions

**Do priors help, and where?** Yes, on the multi-sample CyTOF sets. Against the matched model without priors:

| comparison | Samusik Hungarian | Samusik identity | Levine Hungarian | Levine identity |
|---|---|---|---|---|
| t-mixture (L1 → L4 both) | 0.529 → 0.613 | — | 0.549 → 0.652 | — |
| marginal flow (neither → both) | 0.591 → 0.637 | — | 0.561 → 0.699 | — |
| 3 labelled refs (neither → both) | 0.670 → 0.701 | 0.665 → 0.694 | — | — |

Priors also give identity F1 without matching, which unsupervised baselines cannot provide at all: 0.57–0.61 on Samusik and 0.59–0.69 on Levine.
- **Where they help.** The gain is concentrated in mid-abundance lineage populations whose rules are crisp: pDCs, CD4/CD8 T, NK, eosinophils, IgD/IgM B subsets, and MEP. See `figures/per_population_f1_*.png`.
- **Where they don't.** They do not help, and sometimes hurt, where a rule needs an intermediate level or a parent-relative gate:
  - Intermediate monocytes (F1 0.47 with rules vs 0.60 for L1)
  - Classical monocytes in L4
  - HSC (0 for every method except FlowSOM 0.07)
  - CLP, γδ T, and pro-B
- **Rare sets: negative** (§6).

**Which prior type helps most?** The rules, by a wide margin:
- **Rules alone.** They give almost all of the gain:

  | comparison | Samusik | Levine |
  |---|---|---|
  | L4_rules vs L1_t | +0.075 | +0.109 |
  | L4_both vs L4_rules (ordering on top) | +0.009 | −0.006 |
  | t-mixture + 3 refs: L5_rules_ref3 vs L5_neither_ref3 | +0.022 | — |
  | t-mixture + 3 refs: L5_order_ref3 vs L5_neither_ref3 | +0.001 | — |

- **Ordering alone.** It gives at best a modest Hungarian gain (Levine L3 0.643 vs 0.549, but this uses rule-based initialisation). With the clean rank initialisation it is *worse* than no prior on Samusik (0.473). Under any flow it hurts (L6_order < L6_neither; *3b* marginal order < neither).
- **Identity F1 under ordering only stays low (0.41–0.45).** The ordering bounds the component means relative to one another, but it does not stop components from swapping populations or absorbing unknown cells.
- **Labelled samples are worth more than either prior without labels.** One reference sample beats every unsupervised model. The priors still add +0.03 on top of three labelled samples.

**Is the flow worth its complexity?** On Samusik, the *per-marker monotone* flow is; the coupling layers are not. On Levine the evidence is mixed (§5.3).
- **Marginal monotone flow.** It preserves every marker ordering by construction and acts as a learned per-sample normalisation of each channel. Swapping it in for no flow:

  | setting | Samusik (no flow → marginal flow) | Levine (no flow → marginal flow) |
  |---|---|---|
  | both priors, no labels | L4_both 0.613 → 0.656 (t kernel) | 0.652 → 0.712 |
  | both priors, 3 refs | L5_both_ref3 0.701 → 0.754 | — |
  | identity F1, 3 refs | 0.694 → 0.752 | — |

  That makes it the best model on Samusik, beating FlowSOM by 0.07 on test and 0.09 on validation with no matching needed. The cost is about 1.2× the t-mixture runtime. On Levine the test gain (0.652 → 0.712) does not reproduce on the validation sample H1 (0.639 → 0.659).
- **Coupling layers (RealNVP-style).** Adding them on top of the marginal flow lowers F1 on Samusik test and validation (0.637 → 0.628; 0.661 → 0.651) and on Levine test (0.699 → 0.614). On Levine H1, however, they raise it (0.639 → 0.658). A plausible mechanism for the losses is that coupling layers let the latent space reshape freely, which undoes the alignment between components and marker levels that the rules rely on. They give no consistent benefit for their cost.
- **Number of unknown components.** It hardly matters on the multi-sample sets: K_u = 0, 1 or 6 gives 0.621, 0.627 and 0.628 on Samusik. On the rare sets, however, K_u = 0 or 1 collapses the model (Mosmann ECE 1.0).

**How often are orderings violated on held-out samples?**
- **By the expert gates.** Almost never (§2): 0 of 191 edges on Samusik 06–09, 1 of 191 on 10 (a 0.04 near-tie), and 0 of 83 on Levine H2. The encoded orderings are therefore correct, but also mostly *easy*: the edges join populations already well separated on that marker, which is why the ordering adds little information.
- **By model clusters.** Measured on data-space medians of the Hungarian-matched clusters:

  | model type | violation rate |
  |---|---|
  | unconstrained mixtures | 5–9% (L0 6.9/8.9%, L1 6.7/5.1%, L6_neither 7.4/4.8%) |
  | prior models | 0.2–3% |
  | FlowSOM | 2.8% / 0.2% |
  | best model (marginal flow, both, refs) | 0.9% (Samusik) / 0.2% (Levine) |

  Violations by prior models are not zero, even though the means are ordered by construction. Data-space medians of the cells assigned to a component differ from its mean in latent space, and unknown components can take a population's cells.

### 5.3 Validation check of the post-hoc (3b) variants
The marginal-flow variants were chosen after the 3a *test* results showed that L6_both_marginal beat the coupling flow. That makes their test scores optimistic by construction. To check them, the key methods were re-run on the **validation samples only** (`results/runs/*_val.jsonl`):
- Samusik: samples 04 and 05, with references 01–03.
- Levine: H1. Methods that need references cannot be run there, because H1 *is* the reference.

| method | Samusik val Hungarian | Samusik val identity | Levine H1 Hungarian | Levine H1 identity |
|---|---|---|---|---|
| L2 FlowSOM | 0.675 ± 0.004 | – | **0.728 ± 0.043** | – |
| L4 t + both | 0.636 ± 0.003 | 0.579 | 0.639 ± 0.010 | 0.601 |
| L6 marginal+coupling, both | 0.651 ± 0.006 | 0.610 | 0.658 ± 0.017 | 0.621 |
| L6 marginal only, both | 0.661 ± 0.006 | 0.624 | 0.639 ± 0.032 | 0.601 |
| *3b* marginal flow, rules | 0.651 ± 0.008 | 0.615 | 0.644 ± 0.029 | 0.605 |
| *3b* marginal flow + t, both | 0.672 ± 0.003 | 0.628 | 0.659 ± 0.002 | 0.620 |
| L5 t, both, 3 refs | 0.711 ± 0.004 | 0.703 | n/a | n/a |
| *3b* marginal flow, both, 3 refs | **0.765 ± 0.002** | **0.759** | n/a | n/a |

What the validation check shows:
- **Samusik: the ranking reproduces.** The marginal flow with both priors and 3 references is best by +0.05 over the same model without the flow, and beats FlowSOM by 0.09. The marginal flow also beats the coupling flow (0.661 vs 0.651). So the main Samusik conclusion is not an artefact of choosing the variant on the test set.
- **Levine: it does not reproduce.** On H1, adding a flow helps only about +0.02 and the coupling flow is as good as the marginal one. FlowSOM is best there (0.728).
- **Levine therefore supports only two conclusions:** priors beat an unconstrained mixture, and labels help. The test-set ranking among flow variants on Levine (marginal ≫ coupling) is one sample's worth of evidence and is contradicted by the other sample.

### 5.4 Calibration, runtime, memory
- **Calibration** (`figures/calibration.png`, gated cells):
  - The rules-only classifier is the best calibrated (ECE 0.06 / 0.03) and is slightly *under*-confident on Samusik.
  - All mixtures are over-confident. Without labels they are badly so: L3 ECE 0.56–0.61, L4 0.32–0.40.
  - Labels and the marginal flow fix most of this: marginal flow + both + 3 refs has ECE 0.042, and Levine with 1 ref 0.068.
  - Probabilities from unsupervised prior mixtures should not be read as gate confidence.
- **Runtime** is per sample, single CPU thread, about 80k cells for Samusik and 74k for Levine H2:

  | method | runtime per sample |
  |---|---|
  | FlowSOM | 7–9 s |
  | rules classifier | 10 s |
  | torch mixtures | 40–125 s (400 minibatch steps) |
  | sklearn GMM, Samusik (20k-cell fit + predict) | 311 s |

- **Memory.** Peak RSS per worker was 0.9–1.8 GB for every method. This is dominated by holding the whole dataset in each worker, so it does not discriminate between methods.


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
