"""
UNIFIED single-fit Bayesian model for clavicle-fracture nonunion.

Instead of (a) a separate meta-analysis for the treatment effect and (b) a
separate risk model for baseline risk, this fits ONE model to all the published
data at once:

  * Trial ARM-LEVEL COUNTS are modelled directly as binomial nonunion events
    (no log-RR normal approximation, no continuity correction for zero cells).
    These identify the baseline nonunion odds (by fracture location) and the
    operative-vs-nonoperative treatment effect.
  * The published multivariable PREDICTOR odds ratios enter as an evidence block
    (each log-OR is an observation of the corresponding coefficient), which
    identifies the patient/fracture risk-factor slopes.

Model (logit scale):

  # trial population, per study s and arm:
  logit p_nonop[s] = base_ref[loc] + offset[loc] + u_s
  logit p_op[s]    = base_ref[loc] + offset[loc] + u_s + delta[s]
  y_nonop[s] ~ Binomial(n_nonop[s], p_nonop[s])
  y_op[s]    ~ Binomial(n_op[s],    p_op[s])

  u_s        ~ Normal(0, sigma_base)                # study baseline heterogeneity
  delta[s]   ~ Normal(mu_delta[loc], sigma_delta)   # study treatment effect
  mu_delta[loc] = Delta + omega * z_loc             # location effect (partial pooling)
  base_ref[loc] ~ Normal(-2, 1.5)                   # reference-patient baseline log-odds

  # predictor evidence block (aggregate counts cannot identify these):
  logOR_k(published) ~ Normal(gamma_k, se_k)

`offset[loc] = sum_k gamma_k * xbar_k[loc]` converts the reference-patient
baseline (`base_ref`, i.e. age 40 with no risk factors) into the trial
POPULATION-average nonunion odds, using the assumed covariate profile `xbar` of
the trial populations. Individual prediction then reads straight off the joint
posterior:

  logit risk_nonop = base_ref[loc] + sum_k gamma_k * x_k
  logit risk_op    = logit risk_nonop + mu_delta[loc]

Honest limitations
------------------
* The predictor slopes (gamma) are pinned by the published-OR evidence block;
  the aggregate counts cannot update them. The joint fit's value is that
  baseline, treatment effect and predictor effects live in ONE posterior with
  coherent uncertainty, and the counts are used as counts.
* `offset` relies on an ecological assumption (the trial populations' covariate
  prevalences equal the assumed `xbar`). This only shifts the split between
  `base_ref` and `offset`; individual ARR/NNT are insensitive to it because the
  same offset cancels between arms.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pymc as pm
import arviz as az

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "outputs")
os.makedirs(OUT, exist_ok=True)

MIDSHAFT = os.path.join(ROOT, "data", "meta_analysis_trials.csv")
DISTAL = os.path.join(ROOT, "data", "distal_trials.csv")
PRED = os.path.join(ROOT, "data", "nonunion_predictors.csv")

LOC_LEVELS = ["midshaft", "distal"]
# Predictor order used everywhere downstream (age is per-decade, centred at 40).
PRED_ORDER = ["age", "female", "smoking", "complete_displacement",
              "comminution", "shortening_gt2cm"]
DRAWS, TUNE, CHAINS, SEED = 3000, 2000, 4, 20240702


def load_counts():
    mid = pd.read_csv(MIDSHAFT)
    mid["location"] = "midshaft"
    dist = pd.read_csv(DISTAL)
    dist["location"] = "distal"
    cols = ["study", "n_op", "events_op", "n_nonop", "events_nonop",
            "mean_age", "location"]
    return pd.concat([mid[cols], dist[cols]], ignore_index=True)


def load_predictor_evidence():
    df = pd.read_csv(PRED).set_index("predictor").loc[PRED_ORDER].reset_index()
    logor = np.log(df["odds_ratio"].to_numpy(float))
    se = (np.log(df["ci_high"].to_numpy(float)) -
          np.log(df["ci_low"].to_numpy(float))) / (2 * 1.959964)
    prev = df["baseline_prevalence"].fillna(0.0).to_numpy(float)  # age handled separately
    return df, logor, se, prev


def build_xbar(counts, prev):
    """Assumed trial-population covariate profile per location (for the offset)."""
    xbar = {}
    for loc in LOC_LEVELS:
        mean_age = counts.loc[counts["location"] == loc, "mean_age"].mean()
        vec = prev.copy()
        vec[PRED_ORDER.index("age")] = (mean_age - 40) / 10.0
        xbar[loc] = vec
    return xbar


def run():
    counts = load_counts()
    pred_df, logor, se, prev = load_predictor_evidence()
    xbar = build_xbar(counts, prev)

    loc_idx = counts["location"].map(
        {l: i for i, l in enumerate(LOC_LEVELS)}).to_numpy()
    xbar_mat = np.vstack([xbar[l] for l in LOC_LEVELS])          # (2, K)
    K = len(PRED_ORDER)

    coords = {"study": counts["study"].tolist(), "location": LOC_LEVELS,
              "pred": PRED_ORDER}
    with pm.Model(coords=coords) as model:
        y_no = pm.Data("y_no", counts["events_nonop"].to_numpy(int), dims="study")
        n_no = pm.Data("n_no", counts["n_nonop"].to_numpy(int), dims="study")
        y_op = pm.Data("y_op", counts["events_op"].to_numpy(int), dims="study")
        n_op = pm.Data("n_op", counts["n_op"].to_numpy(int), dims="study")
        g = pm.Data("loc_idx", loc_idx, dims="study")

        # Predictor slopes, informed by the published-OR evidence block.
        gamma = pm.Normal("gamma", 0.0, 2.0, dims="pred")
        pm.Normal("logor_obs", mu=gamma, sigma=se, observed=logor, dims="pred")

        # Reference-patient baseline log-odds of nonunion (nonoperative), by location.
        base_ref = pm.Normal("base_ref", -2.0, 1.5, dims="location")

        # Population covariate offset per location (constant given gamma).
        xbar_pt = pm.Data("xbar", xbar_mat, dims=("location", "pred"))
        offset = pm.Deterministic("offset", (xbar_pt * gamma).sum(axis=1),
                                  dims="location")

        # Study baseline heterogeneity (non-centred).
        sigma_base = pm.HalfNormal("sigma_base", 0.5)
        u = pm.Normal("u", 0.0, 1.0, dims="study") * sigma_base

        # Treatment effect: study within location within overall (partial pooling).
        Delta = pm.Normal("Delta", 0.0, 2.0)
        omega = pm.HalfNormal("omega", 1.0)
        z_loc = pm.Normal("z_loc", 0.0, 1.0, dims="location")
        mu_delta = pm.Deterministic("mu_delta", Delta + omega * z_loc,
                                    dims="location")
        sigma_delta = pm.HalfNormal("sigma_delta", 0.5)
        z_study = pm.Normal("z_study", 0.0, 1.0, dims="study")
        delta = pm.Deterministic("delta", mu_delta[g] + sigma_delta * z_study,
                                 dims="study")

        lin_no = base_ref[g] + offset[g] + u
        lin_op = lin_no + delta
        pm.Binomial("obs_no", n=n_no, logit_p=lin_no, observed=y_no, dims="study")
        pm.Binomial("obs_op", n=n_op, logit_p=lin_op, observed=y_op, dims="study")

        # Reportable transforms
        pm.Deterministic("OR_treat", pm.math.exp(mu_delta), dims="location")
        pm.Deterministic("OR_treat_overall", pm.math.exp(Delta))
        pm.Deterministic("baseline_risk_ref", pm.math.sigmoid(base_ref),
                         dims="location")

        idata = pm.sample(draws=DRAWS, tune=TUNE, chains=CHAINS, cores=CHAINS,
                          target_accept=0.99, random_seed=SEED, progressbar=False)

    post = idata.posterior

    def flat(name, idx=None):
        arr = post[name]
        if idx is not None:
            arr = arr.isel({arr.dims[-1]: idx})
        return arr.values.reshape(-1)

    # Save the joint posterior draws the predictor needs.
    gamma_draws = np.stack([flat("gamma", k) for k in range(K)], axis=1)  # (n,K)
    base_draws = np.stack([flat("base_ref", i) for i in range(2)], axis=1)
    delta_draws = np.stack([flat("mu_delta", i) for i in range(2)], axis=1)
    np.savez(os.path.join(OUT, "unified_posterior.npz"),
             gamma=gamma_draws, base_ref=base_draws, mu_delta=delta_draws,
             pred_order=np.array(PRED_ORDER, dtype=object),
             loc_levels=np.array(LOC_LEVELS, dtype=object))

    # Summaries
    def q(a):
        return np.median(a), np.percentile(a, 2.5), np.percentile(a, 97.5)

    rows = []
    for i, loc in enumerate(LOC_LEVELS):
        orr = np.exp(flat("mu_delta", i))
        br = flat("baseline_risk_ref", i)
        m, lo, hi = q(orr)
        bm, blo, bhi = q(br)
        rows.append(dict(location=loc, OR_treat=m, OR_lo=lo, OR_hi=hi,
                         base_risk_ref=bm, base_lo=blo, base_hi=bhi))
    orr_o = np.exp(flat("Delta"))
    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(OUT, "unified_summary.csv"), index=False)

    gamma_tab = pd.DataFrame({
        "predictor": PRED_ORDER,
        "OR_posterior": np.exp(gamma_draws).mean(axis=0),
        "OR_lo": np.percentile(np.exp(gamma_draws), 2.5, axis=0),
        "OR_hi": np.percentile(np.exp(gamma_draws), 97.5, axis=0),
        "OR_published": pred_df["odds_ratio"].to_numpy(float),
    })
    gamma_tab.to_csv(os.path.join(OUT, "unified_predictor_ORs.csv"), index=False)

    az_summ = az.summary(idata, var_names=["Delta", "mu_delta", "base_ref",
                                           "gamma", "sigma_base", "omega"],
                         hdi_prob=0.95)
    az_summ.to_csv(os.path.join(OUT, "unified_arviz_summary.csv"))
    diverging = int(idata.sample_stats["diverging"].values.sum())
    max_rhat = float(az_summ["r_hat"].max())

    _plot_treatment(summ, orr_o, os.path.join(OUT, "unified_treatment_effect.png"))

    # Console report
    lines = ["=== Unified single-fit model (binomial counts + predictor ORs) ==="]
    lines.append(f"Fitted to {len(counts)} studies x 2 arms = {2*len(counts)} "
                 f"binomial arm counts; {CHAINS} chains; "
                 f"divergences={diverging}; max R-hat={max_rhat:.3f}")
    lines.append(f"Treatment effect (operative vs nonoperative), odds ratio:")
    m, lo, hi = q(orr_o)
    lines.append(f"  Overall   OR {m:.2f} [{lo:.2f}, {hi:.2f}]")
    for _, r in summ.iterrows():
        lines.append(f"  {r['location']:<9} OR {r['OR_treat']:.2f} "
                     f"[{r['OR_lo']:.2f}, {r['OR_hi']:.2f}]   "
                     f"reference-patient baseline nonunion "
                     f"{r['base_risk_ref']*100:.1f}% "
                     f"[{r['base_lo']*100:.1f}, {r['base_hi']*100:.1f}]")
    lines.append("Predictor ORs recovered (posterior vs published):")
    for _, r in gamma_tab.iterrows():
        lines.append(f"  {r['predictor']:<22} {r['OR_posterior']:.2f} "
                     f"[{r['OR_lo']:.2f}, {r['OR_hi']:.2f}]  (pub {r['OR_published']:.2f})")
    report = "\n".join(lines)
    print(report)
    with open(os.path.join(OUT, "unified_report.txt"), "w") as f:
        f.write(report + "\n")

    return dict(summary=summ, OR_overall=q(orr_o), gamma=gamma_tab,
                diverging=diverging, max_rhat=max_rhat)


def _plot_treatment(summ, orr_overall, path):
    ink, c = "#1f2a44", "#2f6f8f"
    labels = list(summ["location"]) + ["overall"]
    meds = list(summ["OR_treat"]) + [np.median(orr_overall)]
    los = list(summ["OR_lo"]) + [np.percentile(orr_overall, 2.5)]
    his = list(summ["OR_hi"]) + [np.percentile(orr_overall, 97.5)]
    y = np.arange(len(labels))[::-1]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.errorbar(meds, y, xerr=[np.array(meds) - np.array(los),
                               np.array(his) - np.array(meds)],
                fmt="o", color=c, ecolor=ink, capsize=3, ms=8)
    for yi, m, lo, hi in zip(y, meds, los, his):
        ax.text(hi * 1.15, yi, f"{m:.2f} [{lo:.2f}, {hi:.2f}]", va="center",
                fontsize=9, color=ink)
    ax.axvline(1, ls="--", color="#888", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xscale("log")
    ax.set_xlim(0.02, 3)
    ax.set_xticks([0.05, 0.1, 0.25, 0.5, 1, 2])
    ax.set_xticklabels(["0.05", "0.1", "0.25", "0.5", "1", "2"])
    ax.set_xlabel("Odds ratio for nonunion, operative vs nonoperative")
    ax.set_title("Unified model: treatment effect (binomial arm counts)",
                 loc="left", fontsize=11)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    run()
