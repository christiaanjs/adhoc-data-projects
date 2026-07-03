# Data provenance and references

All numbers in this project are extracted from the published orthopaedic
literature. **No individual patient records are used.** The two datasets have
different provenance and different intended use, described below.

## 1. `meta_analysis_trials.csv` — real aggregate trial data

Trial-level nonunion counts for **operative (plate / intramedullary fixation)
vs. nonoperative (sling / figure-of-eight)** treatment of **displaced midshaft
clavicle fractures**. These are the head-to-head randomized controlled trials
(RCTs) that anchor the modern evidence base. Event counts are reconstructed
from each trial's reported nonunion rates and arm sizes and cross-checked
against the pooled figures in the McKee/COTS and Woltz meta-analyses; where
dropout or rounding makes a count ambiguous it may differ by ±1 from a given
secondary source. The pooled estimate is robust to this.

| Study | Ref | Op nonunion | Nonop nonunion |
|-------|-----|-------------|----------------|
| Canadian Orthopaedic Trauma Society (COTS) | JBJS Am 2007;89:1-10 | 2/62 | 7/49 |
| Smekal et al. (elastic IM nail) | J Orthop Trauma 2009;23:106-12 | 0/30 | 4/30 |
| Mirzatolooei | Injury 2011 (comminuted subset) | 1/30 | 4/30 |
| Virtanen et al. | JBJS Am 2012;94:1546-53 | 0/28 | 3/23 |
| Robinson et al. | JBJS Am 2013;95:1576-84 | 1/86 | 15/95 |
| Woltz et al. | JBJS Am 2017;99:106-12 | 2/82 | 18/78 |

Published pooled effect for nonunion (operative vs nonoperative), for reference:
- McKee et al. meta-analysis: nonunion 3/212 (operative) vs 29/200 (nonoperative).
- Pooled relative risk ≈ 0.14 (95% CI 0.06–0.32) favouring operative fixation.

Our random-effects re-analysis of the six trials above reproduces an effect of
this magnitude.

## 1b. `distal_trials.csv` — distal (lateral-third) clavicle fractures

Comparative operative-vs-nonoperative nonunion data for **displaced distal
(Neer type II / lateral-third) clavicle fractures**, used by the hierarchical
meta-analysis. The distal evidence base is **much sparser and of lower certainty
than the midshaft base** — it is dominated by observational cohorts with a
single randomized trial — which is the motivation for partial pooling.

| Study | Ref | Design | Op nonunion | Nonop nonunion |
|-------|-----|--------|-------------|----------------|
| Hall et al. | JBJS Am 2021 (PMID 34128498) | multicentre RCT | 1/27 | 11/30 |
| Neer-II cohort A | representative comparative cohort | cohort | 2/40 | 13/44 |
| Neer-II cohort B | representative comparative cohort | cohort | 1/22 | 8/26 |

Provenance and honesty:
- **Hall et al. 2021** randomized 57 patients (27 operative, 30 nonoperative);
  union was 95% operative vs 64% nonoperative (adjusted absolute nonunion risk
  2.9% vs 20.2%). Counts above are the union-rate-derived nonunion events.
- The two **Neer-II cohort** rows are *representative comparative cohorts* whose
  event counts are reconstructed to be consistent with the pooled distal union
  rates reported in the Neer type II systematic review/meta-analysis of 2,284
  patients (operative union ≈ 96%, nonoperative union ≈ 63–70%; PMID 34779668).
  They stand in for the observational comparative literature; they are **not**
  attributed to a specific named study, and counts may be off by ±1–2.
- Reported nonoperative nonunion rates for displaced distal fractures range
  widely (≈ 28–44%), so the distal subgroup carries genuinely wider uncertainty.
  The hierarchical model expresses this by shrinking the distal estimate toward
  the overall effect.

## 2. `nonunion_predictors.csv` — published effect sizes for patient/fracture factors

Multivariable-adjusted associations with **nonunion after nonoperative
treatment**. These are used to build an interpretable risk score and to
generate a synthetic patient-level cohort for demonstrating and validating the
modelling pipeline (see below). Odds ratios are taken from:

- **Robinson CM, Court-Brown CM, McQueen MM, Wakefield AE.** Estimating the risk
  of nonunion following nonoperative treatment of a clavicular fracture.
  *JBJS Am* 2004;86(7):1359-65. Prospective cohort of 868 patients; overall
  nonunion 6.2%; advancing age, female sex, fracture displacement and
  comminution identified as risk factors (displacement and age independently
  significant in the middle-third model).
- **Murray IR, et al.** Risk factors for nonunion after nonoperative treatment
  of displaced midshaft fractures of the clavicle. *JBJS Am* 2013;95:1153-8.
  Multivariable predictors: smoking (OR ≈ 3.76), comminution (OR ≈ 1.75),
  displacement (OR ≈ 1.17 per mm).
- Systematic reviews of nonunion predictors (Int Orthop 2014; Injury 2015/2025)
  corroborating displacement, comminution, shortening, smoking, age and sex.

## 2b. `source_studies.csv` — cohorts behind the predictor odds ratios

The published multivariable odds ratios were each estimated in a specific
observational cohort. `source_studies.csv` records, for the estimation-process
evidence model (`src/ipd_evidence_model.py`), which predictors each cohort
reported, its sample size, its overall nonunion count, and its mean age:

- **Robinson 2004** (N=868, nonunion 6.2%): age, female sex, complete
  displacement, shortening. Prospective nonoperative cohort of all clavicle
  fractures.
- **Murray 2013** (N=200, nonunion ≈16.5%): smoking, comminution. Displaced
  midshaft nonoperative cohort.

Sample sizes and event counts are the reported cohort figures; the source
cohorts' internal covariate prevalences are not published, so the model uses the
shared latent prevalences for the Fisher-information calculation (documented
approximation).

## Important honesty note on the predictive model

Individual patient-level data (IPD) for clavicle fractures is not publicly
available. The predictive-modelling script therefore:

1. Builds a **transparent points-based risk score directly from the published
   odds ratios** (no data fitting required, fully interpretable), and
2. Generates a **synthetic cohort** whose outcomes are drawn from a logistic
   model parameterised by those same published odds ratios, then fits
   interpretable models (logistic regression, shallow decision tree) to it.

Because the synthetic outcome is generated from the published effect sizes, the
fitted logistic model *recovers* those effect sizes and the reported
discrimination reflects the separability implied by the literature — it is a
**validation of the pipeline and an estimate of achievable performance**, not
independent epidemiological evidence. The pipeline is written so it can be
re-run on real IPD by swapping in a real cohort CSV with the same columns.
