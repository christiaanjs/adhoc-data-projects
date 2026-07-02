# Clavicle fracture outcomes — meta-analysis + interpretable risk model

A small, self-contained analysis of **displaced midshaft clavicle fractures**
built from the published literature. It has two parts:

1. **Meta-analysis** — a random-effects (DerSimonian–Laird) synthesis of six
   RCTs comparing **operative fixation vs nonoperative treatment**, with
   nonunion as the outcome. Produces a forest plot and pooled RR/OR, NNT, and
   heterogeneity statistics.
2. **Interpretable predictive model** — a nonunion risk model based on patient
   and fracture factors (smoking, displacement, shortening, comminution, sex,
   age), delivered as a logistic regression, a bedside **points score**, and a
   decision tree, then compared on discrimination and calibration.

The brief was to *favour models that are interpretable but have high predictive
power*. The headline finding is that the simple points score matches the full
logistic model's discrimination — interpretability is essentially free here.

## Results at a glance

- Operative fixation cuts nonunion risk by ~86% (pooled **RR ≈ 0.14**,
  95% CI 0.06–0.30; I² = 0%), NNT ≈ 7 — matching the published evidence base.
- The interpretable nonunion model reaches **AUC ≈ 0.72** (5-fold CV) and is
  well calibrated; the points score performs the same as full logistic
  regression.

See [`report/REPORT.md`](report/REPORT.md) for the full write-up and figures.

## Layout

```
clavicle-fracture-analysis/
├── data/
│   ├── meta_analysis_trials.csv     # trial-level nonunion counts (real, cited)
│   ├── nonunion_predictors.csv      # published multivariable odds ratios
│   └── REFERENCES.md                # provenance + honesty note on modelling
├── src/
│   ├── meta_analysis.py             # DerSimonian–Laird RE meta-analysis + forest plot
│   └── risk_model.py                # points score, logistic regression, tree, calculator
├── outputs/                         # generated figures, tables, text reports
├── report/REPORT.md                 # auto-generated report
├── run_all.py                       # runs everything and regenerates the report
└── requirements.txt
```

## Run it

```bash
pip install -r requirements.txt
python run_all.py
```

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
