# Clavicle fracture outcomes — meta-analysis + interpretable risk model

A small, self-contained analysis of **clavicle fractures** built from the
published literature. It has three parts:

1. **Meta-analysis (midshaft)** — a random-effects (DerSimonian–Laird) synthesis
   of six RCTs comparing **operative fixation vs nonoperative treatment**, with
   nonunion as the outcome. Produces a forest plot, pooled RR/OR, NNT, and
   heterogeneity statistics.
2. **Hierarchical Bayesian meta-analysis (midshaft + distal)** — a 3-level
   multilevel model (studies nested within fracture location) fitted with
   **PyMC / NUTS**, extending the analysis to sparse **distal (lateral-third)**
   fractures via partial pooling.
3. **Interpretable predictive model** — a nonunion risk model based on patient
   and fracture factors (smoking, displacement, shortening, comminution, sex,
   age), delivered as a logistic regression, a bedside **points score**, and a
   decision tree, then compared on discrimination and calibration.
4. **Unified single-fit model** — one Bayesian model fit directly to all the
   published trial **counts** (each arm as a binomial), with the published
   predictor odds ratios folded in as an evidence block, so baseline risk,
   treatment effect and patient-factor slopes come from a single joint posterior
   (no normal approximation, no continuity corrections).
5. **Latent-covariate model** — the same idea done exactly: individual patient
   risk factors are treated as **latent and marginalised out** of the aggregate
   counts (exact enumeration of the binary profiles + Gauss–Hermite quadrature
   over age), with published **prevalences and odds ratios** both used as data.
   This removes the linear-offset (Jensen) approximation in the unified model.
6. **Individualised treatment benefit** — turns these into a per-patient
   absolute risk reduction and number-needed-to-treat, with credible intervals,
   exposed through a command-line predictor (`predict.py`).

The brief was to *favour models that are interpretable but have high predictive
power*. The headline finding is that the simple points score matches the full
logistic model's discrimination — interpretability is essentially free here.

## Results at a glance

- Operative fixation cuts nonunion risk by ~86% (pooled **RR ≈ 0.14**,
  95% CI 0.06–0.30; I² = 0%), NNT ≈ 7 — matching the published evidence base.
- The **hierarchical model** shows the same relative benefit holds for distal
  fractures (RR ≈ 0.14), with partial pooling tightening the sparse distal
  estimate; convergence is clean (0 divergences, R-hat = 1.00). Distal fractures
  carry a much higher *baseline* nonunion risk, so the *absolute* benefit of
  surgery is larger there.
- The interpretable nonunion model reaches **AUC ≈ 0.72** (5-fold CV) and is
  well calibrated; the points score performs the same as full logistic
  regression.

See [`report/REPORT.md`](report/REPORT.md) for the full write-up and figures.

## Layout

```
clavicle-fracture-analysis/
├── data/
│   ├── meta_analysis_trials.csv     # midshaft trial-level nonunion counts (real, cited)
│   ├── distal_trials.csv            # distal (lateral-third) comparative counts
│   ├── nonunion_predictors.csv      # published multivariable odds ratios
│   └── REFERENCES.md                # provenance + honesty note on modelling
├── src/
│   ├── meta_analysis.py             # DerSimonian–Laird RE meta-analysis + forest plot
│   ├── hierarchical_meta.py         # PyMC/NUTS 3-level Bayesian meta-analysis
│   ├── unified_model.py             # PyMC/NUTS single joint fit to arm counts + ORs
│   ├── latent_integration_model.py  # exact latent-covariate marginalisation fit
│   ├── risk_model.py                # points score, logistic regression, tree, calculator
│   └── treatment_benefit.py         # baseline risk x treatment effect -> ARR/NNT
├── outputs/                         # generated figures, tables, text reports
├── report/REPORT.md                 # auto-generated report
├── run_all.py                       # runs everything and regenerates the report
├── predict.py                       # command-line outcome predictor
└── requirements.txt
```

## Run it

The hierarchical model needs PyMC, which conflicts with the system `packaging`
on some Debian images, so use a virtualenv:

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python run_all.py
```

## Predict outcomes for a patient (CLI)

After `run_all.py` has generated the model artifacts, use the command-line
predictor. It reports the nonunion risk with and without surgery, the absolute
risk reduction, and the number-needed-to-treat (each with a 95% credible
interval), plus the interpretable points score.

```bash
python predict.py --age 28
python predict.py --age 62 --female --smoking --displacement --comminution --shortening
python predict.py --age 55 --displacement --location distal
python predict.py --age 40 --smoking --displacement --json          # machine-readable
python predict.py --age 40 --smoking --displacement --model two-stage  # cross-check
```

By default it uses the **latent-covariate model** (`src/latent_integration_model.py`);
`--model unified` uses the linear-offset joint fit and `--model two-stage`
combines the separate risk model and meta-analysis posteriors, both as
cross-checks.

Example output:

```
 Nonunion risk if treated NONoperatively : 67.5%  (95% CrI 62.1-72.5%)
 Nonunion risk if treated operatively    : 9.4%   (95% CrI 4.4-20.1%)
 Absolute risk reduction from surgery    : 57.8%  (95% CrI 46.6-65.1%)
 Number needed to treat (NNT)            : 2      (95% CrI 2-2)
```

Run `python predict.py --help` for all flags.

## Important caveats

- **No individual patient data is used.** The meta-analysis uses real, cited
  aggregate trial counts. The predictive model is parameterised from published
  odds ratios and validated on a synthetic cohort drawn from those same effect
  sizes — so the reported performance is an estimate of achievable performance,
  not independent epidemiological evidence. Full detail in
  [`data/REFERENCES.md`](data/REFERENCES.md).
- The pipeline is written to drop straight onto real data: swap `build_cohort()`
  in `src/risk_model.py` for a loader of a real cohort CSV with the same columns.
- This is a decision-support demonstration, **not** a validated clinical tool.
