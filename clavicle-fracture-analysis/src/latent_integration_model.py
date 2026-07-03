"""
Latent-covariate single-fit model: integrate individual patient risk factors OUT
of the aggregate trial counts, and use the published prevalences AND odds ratios
as data alongside the counts.

Motivation
----------
The `unified_model.py` maps the reference-patient baseline to the trial
population rate with a *linear* offset (sum gamma*xbar). That ignores Jensen's
inequality: for a heterogeneous population the average of sigmoid(risk) is not
sigmoid(average risk). Here we do the marginalisation properly.

Each trial patient's risk factors are treated as LATENT and integrated out
exactly, so the arm-level nonunion probability is

    p_arm = E_x[ sigmoid( base_ref[loc] + treat*t + gamma . x ) ]

where x = (age, 5 binary factors). The expectation is computed exactly:
  * 5 binary factors  -> enumerate all 2^5 = 32 profiles, weighted by the
    (latent) prevalences (assumed independent within a trial population);
  * age               -> Gauss-Hermite quadrature over Normal(mean_age[loc], sd).

This keeps the likelihood deterministic and differentiable, so NUTS is happy
(no Monte-Carlo noise in the log-density). Three data blocks share parameters:

  counts      : y_arm ~ Binomial(n_arm, p_arm)              (uses counts as counts)
  prevalences : published prevalence ~ Binomial(N_eff, prev_k)   (latent x distribution)
  odds ratios : published logOR_k ~ Normal(gamma_k, se_k)        (risk-factor slopes)

Compared with the linear-offset unified model this (a) removes the Jensen
approximation, (b) lets the prevalences carry uncertainty, and (c) is a fully
generative model of the aggregate counts. The individual-slope information still
comes mostly from the OR block (aggregate data weakly identifies individual
associations - the ecological limitation), but everything is now one coherent
nonlinear fit.
"""
from __future__ import annotations

import itertools
import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pymc as pm
import pytensor.tensor as pt
import arviz as az

from unified_model import (load_counts, load_predictor_evidence, LOC_LEVELS,
                           PRED_ORDER)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "outputs")

AGE_SD_DECADES = 1.3          # assumed population SD of age (~13 y)
PREV_NEFF = 150               # effective sample size behind published prevalences
N_GH = 8                      # Gauss-Hermite nodes for the age integral
DRAWS, TUNE, CHAINS, SEED = 2000, 1500, 4, 20240702


def _age_nodes():
    """Per-location Gauss-Hermite nodes/weights for age (centred, per decade)."""
    x, w = np.polynomial.hermite.hermgauss(N_GH)
    gh_w = w / np.sqrt(np.pi)                       # weights sum to 1
    counts = load_counts()
    nodes = {}
    for loc in LOC_LEVELS:
        mean_age = counts.loc[counts["location"] == loc, "mean_age"].mean()
        m = (mean_age - 40) / 10.0
        nodes[loc] = m + np.sqrt(2.0) * AGE_SD_DECADES * x   # (Q,)
    return nodes, gh_w


def run():
    counts = load_counts()
    pred_df, logor, se, prev0 = load_predictor_evidence()   # PRED_ORDER
    loc_idx = counts["location"].map(
        {l: i for i, l in enumerate(LOC_LEVELS)}).to_numpy()

    # Binary profile enumeration (order: female, smoking, displacement, comminution, shortening)
    Bmat = np.array(list(itertools.product([0, 1], repeat=5)), dtype=float)  # (32,5)
    prev_obs = prev0[1:]                                    # binaries only
    prev_success = np.round(prev_obs * PREV_NEFF).astype(int)

    age_nodes, gh_w = _age_nodes()
    # numpy arrays indexed by the (fixed) location of each study
    study_age_nodes = np.vstack([age_nodes[LOC_LEVELS[i]] for i in loc_idx])  # (S,Q)

    coords = {"study": counts["study"].tolist(), "location": LOC_LEVELS,
              "pred": PRED_ORDER, "bin": PRED_ORDER[1:]}
    with pm.Model(coords=coords) as model:
        y_no = pm.Data("y_no", counts["events_nonop"].to_numpy(int), dims="study")
        n_no = pm.Data("n_no", counts["n_nonop"].to_numpy(int), dims="study")
        y_op = pm.Data("y_op", counts["events_op"].to_numpy(int), dims="study")
        n_op = pm.Data("n_op", counts["n_op"].to_numpy(int), dims="study")

        # Risk-factor slopes, informed by the published-OR evidence block.
        gamma = pm.Normal("gamma", 0.0, 2.0, dims="pred")
        pm.Normal("logor_obs", mu=gamma, sigma=se, observed=logor, dims="pred")
        gamma_age = gamma[0]
        gamma_bin = gamma[1:]                               # (5,)

        # Latent prevalences of the binary factors, informed by published data.
        prev = pm.Beta("prev", 1.5, 1.5, dims="bin")
        pm.Binomial("prev_obs", n=PREV_NEFF, p=prev,
                    observed=prev_success, dims="bin")

        # Reference-patient (age 40, no factors) baseline log-odds, by location.
        base_ref = pm.Normal("base_ref", -2.0, 1.5, dims="location")

        # Study baseline heterogeneity (non-centred).
        sigma_base = pm.HalfNormal("sigma_base", 0.4)
        u = pm.Normal("u", 0.0, 1.0, dims="study") * sigma_base

        # Treatment effect: study within location within overall (partial pooling).
        Delta = pm.Normal("Delta", 0.0, 2.0)
        omega = pm.HalfNormal("omega", 1.0)
        z_loc = pm.Normal("z_loc", 0.0, 1.0, dims="location")
        mu_delta = pm.Deterministic("mu_delta", Delta + omega * z_loc,
                                    dims="location")
        sigma_delta = pm.HalfNormal("sigma_delta", 0.4)
        z_study = pm.Normal("z_study", 0.0, 1.0, dims="study")
        delta = mu_delta[loc_idx] + sigma_delta * z_study   # (S,)

        # Marginalisation weights over the 32 binary profiles.
        logw = (Bmat @ pt.log(prev) + (1.0 - Bmat) @ pt.log1p(-prev))  # (32,)
        w = pt.exp(logw)                                    # (32,), sums to 1
        Bg = Bmat @ gamma_bin                               # (32,)

        # Exact-marginal arm probabilities, vectorised over all studies at once
        # (keeps the graph small so it compiles/runs fast on the C backend).
        age_term = gamma_age * study_age_nodes               # (S,Q) tensor
        const_no = base_ref[loc_idx] + u                     # (S,)
        const_op = const_no + delta                          # (S,)
        # grid[s,c,q] = Bg[c] + age_term[s,q] + const[s]
        grid_no = (Bg[None, :, None] + age_term[:, None, :]
                   + const_no[:, None, None])                # (S,32,Q)
        grid_op = (Bg[None, :, None] + age_term[:, None, :]
                   + const_op[:, None, None])
        weight = w[None, :, None] * gh_w[None, None, :]       # (1,32,Q)
        p_no = pt.clip((pt.sigmoid(grid_no) * weight).sum(axis=(1, 2)), 1e-6, 1 - 1e-6)
        p_op = pt.clip((pt.sigmoid(grid_op) * weight).sum(axis=(1, 2)), 1e-6, 1 - 1e-6)

        pm.Binomial("obs_no", n=n_no, p=p_no, observed=y_no, dims="study")
        pm.Binomial("obs_op", n=n_op, p=p_op, observed=y_op, dims="study")

        pm.Deterministic("OR_treat", pm.math.exp(mu_delta), dims="location")
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

    # Save the joint posterior (same layout as unified_posterior.npz).
    K = len(PRED_ORDER)
    gamma_draws = np.stack([flat("gamma", k) for k in range(K)], axis=1)
    base_draws = np.stack([flat("base_ref", i) for i in range(2)], axis=1)
    delta_draws = np.stack([flat("mu_delta", i) for i in range(2)], axis=1)
    np.savez(os.path.join(OUT, "latent_posterior.npz"),
             gamma=gamma_draws, base_ref=base_draws, mu_delta=delta_draws,
             pred_order=np.array(PRED_ORDER, dtype=object),
             loc_levels=np.array(LOC_LEVELS, dtype=object),
             prev=np.stack([flat("prev", k) for k in range(5)], axis=1))

    def q(a):
        return float(np.median(a)), float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))

    rows = []
    for i, loc in enumerate(LOC_LEVELS):
        orr = q(np.exp(flat("mu_delta", i)))
        br = q(flat("baseline_risk_ref", i))
        rows.append(dict(location=loc, OR_treat=orr[0], OR_lo=orr[1], OR_hi=orr[2],
                         base_risk_ref=br[0], base_lo=br[1], base_hi=br[2]))
    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(OUT, "latent_summary.csv"), index=False)

    prev_tab = pd.DataFrame({
        "factor": PRED_ORDER[1:],
        "prev_posterior": [np.median(flat("prev", k)) for k in range(5)],
        "prev_lo": [np.percentile(flat("prev", k), 2.5) for k in range(5)],
        "prev_hi": [np.percentile(flat("prev", k), 97.5) for k in range(5)],
        "prev_published": prev_obs,
    })
    prev_tab.to_csv(os.path.join(OUT, "latent_prevalences.csv"), index=False)

    az_summ = az.summary(idata, var_names=["Delta", "mu_delta", "base_ref",
                                           "gamma", "prev", "omega"],
                         hdi_prob=0.95)
    az_summ.to_csv(os.path.join(OUT, "latent_arviz_summary.csv"))
    diverging = int(idata.sample_stats["diverging"].values.sum())
    max_rhat = float(az_summ["r_hat"].max())

    lines = ["=== Latent-covariate single-fit model (exact marginalisation) ==="]
    lines.append(f"Counts + published prevalences + published ORs; {CHAINS} chains; "
                 f"divergences={diverging}; max R-hat={max_rhat:.3f}")
    for _, r in summ.iterrows():
        lines.append(f"  {r['location']:<9} treatment OR {r['OR_treat']:.2f} "
                     f"[{r['OR_lo']:.2f}, {r['OR_hi']:.2f}]   "
                     f"reference-patient baseline {r['base_risk_ref']*100:.1f}% "
                     f"[{r['base_lo']*100:.1f}, {r['base_hi']*100:.1f}]")
    lines.append("Latent prevalences (posterior vs published):")
    for _, r in prev_tab.iterrows():
        lines.append(f"  {r['factor']:<22} {r['prev_posterior']*100:.0f}% "
                     f"[{r['prev_lo']*100:.0f}, {r['prev_hi']*100:.0f}]  "
                     f"(pub {r['prev_published']*100:.0f}%)")
    report = "\n".join(lines)
    print(report)
    with open(os.path.join(OUT, "latent_report.txt"), "w") as f:
        f.write(report + "\n")

    return dict(summary=summ, prevalences=prev_tab,
                diverging=diverging, max_rhat=max_rhat)


if __name__ == "__main__":
    run()
