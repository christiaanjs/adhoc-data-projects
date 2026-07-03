"""
Principled use of the published odds ratios: model the ESTIMATION PROCESS that
produced them, instead of treating each reported OR as an independent Gaussian
observation with its reported standard error.

The idea
--------
Each source study fitted a multivariable logistic regression to its own
individual patient data. Its reported coefficient vector is (asymptotically) the
maximum-likelihood estimate, whose sampling distribution is

    gamma_hat_s ~ MVN( gamma , V_s ),   V_s = [ N_s * E_x( w(x) x x^T ) ]^-1

where the expectation is over the study's covariate distribution, w(x)=mu(1-mu),
mu = sigmoid(alpha_s + gamma . x), and x includes an intercept. V_s is the
inverse *expected Fisher information*: it is derived from the study's design
(sample size N_s, covariate prevalences, baseline outcome rate alpha_s) and the
shared true coefficients - NOT from the reported SEs. This automatically gives:
  * the correct across-coefficient CORRELATIONS (the study fitted them jointly);
  * precision that scales with the actual information (N_s, prevalence, and how
    close the outcome rate is to 50%, where a binary outcome is most informative);
  * a self-consistent likelihood tied to the same latent covariate distribution
    used for the trial-count marginalisation.

We also use each source study's overall nonunion count as data
(y_s ~ Binomial(N_s, marginal rate)), so the published marginal outcome rate
informs the fit too.

This replaces the `logOR_k ~ Normal(gamma_k, se_k)` block of the latent model.
The trial ARM COUNTS are still modelled exactly as in latent_integration_model
(binary profiles enumerated, age integrated by quadrature). Everything -
baseline, treatment effect, predictor slopes and their correlations - is one
coherent posterior.

Remaining approximations (documented): asymptotic normality of the MLE (the same
assumption behind any reported Wald CI); the source studies' covariate
prevalences are taken to equal the shared latent prevalences (we do not have
each source's covariate table); Fisher information uses the expected (not
observed) form.
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

from unified_model import load_counts, load_predictor_evidence, LOC_LEVELS, PRED_ORDER

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "outputs")
DATA = os.path.join(ROOT, "data")

AGE_SD_DECADES = 1.3
PREV_NEFF = 150
N_GH = 6
DRAWS, TUNE, CHAINS, SEED = 1500, 1500, 4, 20240702
TARGET_ACCEPT = 0.95


def _gh():
    x, w = np.polynomial.hermite.hermgauss(N_GH)
    return x, w / np.sqrt(np.pi)


def _design_matrix(mean_age_decades):
    """Constant (M,7) design matrix over 32 binary profiles x N_GH age nodes.
    Columns: [intercept, age, female, smoking, displacement, comminution, shortening]."""
    x_gh, gh_w = _gh()
    age_nodes = mean_age_decades + np.sqrt(2.0) * AGE_SD_DECADES * x_gh   # (Q,)
    B = np.array(list(itertools.product([0, 1], repeat=5)), dtype=float)  # (32,5)
    rows, ghw_rows, prof_rows = [], [], []
    for c in range(B.shape[0]):
        for q in range(len(age_nodes)):
            rows.append([1.0, age_nodes[q], *B[c]])
            ghw_rows.append(gh_w[q])
            prof_rows.append(c)
    return (np.array(rows), np.array(ghw_rows), np.array(prof_rows, dtype=int),
            B)


def run():
    counts = load_counts()
    pred_df, logor, se, prev0 = load_predictor_evidence()          # PRED_ORDER
    sources = pd.read_csv(os.path.join(DATA, "source_studies.csv"))
    loc_idx = counts["location"].map(
        {l: i for i, l in enumerate(LOC_LEVELS)}).to_numpy()

    # --- trial-count marginalisation pieces (as in the latent model) ---
    x_gh, gh_w = _gh()
    counts_age = {loc: (counts.loc[counts["location"] == loc, "mean_age"].mean() - 40) / 10.0
                  for loc in LOC_LEVELS}
    trial_age_nodes = {loc: counts_age[loc] + np.sqrt(2.0) * AGE_SD_DECADES * x_gh
                       for loc in LOC_LEVELS}
    study_age_nodes = np.vstack([trial_age_nodes[LOC_LEVELS[i]] for i in loc_idx])
    Bmat = np.array(list(itertools.product([0, 1], repeat=5)), dtype=float)   # (32,5)
    prev_obs = prev0[1:]
    prev_success = np.round(prev_obs * PREV_NEFF).astype(int)

    # --- source-study measurement-model pieces ---
    src = []
    for _, r in sources.iterrows():
        X, ghw, prof, _ = _design_matrix((r["mean_age"] - 40) / 10.0)
        rep_names = r["predictors"].split(";")
        # x-index (0=intercept, then PRED_ORDER+1) for each reported predictor
        xidx = [PRED_ORDER.index(nm) + 1 for nm in rep_names]
        gidx = [PRED_ORDER.index(nm) for nm in rep_names]
        rep_logor = np.array([logor[PRED_ORDER.index(nm)] for nm in rep_names])
        src.append(dict(name=r["study"], N=int(r["n"]), events=int(r["events"]),
                        X=X, ghw=ghw, prof=prof, xidx=xidx, gidx=gidx,
                        rep=rep_logor))

    K = len(PRED_ORDER)
    coords = {"study": counts["study"].tolist(), "location": LOC_LEVELS,
              "pred": PRED_ORDER, "bin": PRED_ORDER[1:]}
    with pm.Model(coords=coords) as model:
        gamma = pm.Normal("gamma", 0.0, 2.0, dims="pred")
        gamma_age = gamma[0]
        gamma_bin = gamma[1:]

        prev = pm.Beta("prev", 1.5, 1.5, dims="bin")
        pm.Binomial("prev_obs", n=PREV_NEFF, p=prev, observed=prev_success, dims="bin")
        logprev = pt.log(prev)
        log1m = pt.log1p(-prev)
        w_prof = pt.exp(Bmat @ logprev + (1.0 - Bmat) @ log1m)     # (32,)

        base_ref = pm.Normal("base_ref", -2.0, 1.5, dims="location")
        sigma_base = pm.HalfNormal("sigma_base", 0.4)
        u = pm.Normal("u", 0.0, 1.0, dims="study") * sigma_base

        Delta = pm.Normal("Delta", 0.0, 2.0)
        omega = pm.HalfNormal("omega", 1.0)
        z_loc = pm.Normal("z_loc", 0.0, 1.0, dims="location")
        mu_delta = pm.Deterministic("mu_delta", Delta + omega * z_loc, dims="location")
        sigma_delta = pm.HalfNormal("sigma_delta", 0.4)
        z_study = pm.Normal("z_study", 0.0, 1.0, dims="study")
        delta = mu_delta[loc_idx] + sigma_delta * z_study

        # ---- trial arm counts (exact marginalisation), vectorised over studies ----
        Bg = Bmat @ gamma_bin
        age_term = gamma_age * study_age_nodes               # (S,Q)
        const_no = base_ref[loc_idx] + u                     # (S,)
        const_op = const_no + delta                          # (S,)
        grid_no = Bg[None, :, None] + age_term[:, None, :] + const_no[:, None, None]
        grid_op = Bg[None, :, None] + age_term[:, None, :] + const_op[:, None, None]
        weight = w_prof[None, :, None] * gh_w[None, None, :]  # (1,32,Q)
        p_no = pt.clip((pt.sigmoid(grid_no) * weight).sum(axis=(1, 2)), 1e-6, 1 - 1e-6)
        p_op = pt.clip((pt.sigmoid(grid_op) * weight).sum(axis=(1, 2)), 1e-6, 1 - 1e-6)
        pm.Binomial("obs_no", n=counts["n_nonop"].to_numpy(int), p=p_no,
                    observed=counts["events_nonop"].to_numpy(int), dims="study")
        pm.Binomial("obs_op", n=counts["n_op"].to_numpy(int), p=p_op,
                    observed=counts["events_op"].to_numpy(int), dims="study")

        # ---- source-study IPD estimation-process evidence ----
        for s in src:
            X = pt.as_tensor_variable(s["X"])                     # (M,7) const
            # profile weight * age(GH) weight per row
            pw = w_prof[s["prof"]] * pt.as_tensor_variable(s["ghw"])   # (M,)
            alpha_s = pm.Normal(f"alpha_{s['name']}", -2.0, 1.5)
            eta = alpha_s + X[:, 1:] @ gamma                      # (M,)
            mu = pm.math.sigmoid(eta)
            wobs = mu * (1.0 - mu)
            # marginal outcome rate -> Binomial on reported event count
            pi_s = pt.sum(pw * mu)
            pm.Binomial(f"rate_{s['name']}", n=s["N"],
                        p=pt.clip(pi_s, 1e-6, 1 - 1e-6), observed=s["events"])
            # expected Fisher information J = N * X^T diag(pw*wobs) X
            Wrow = pw * wobs
            J = s["N"] * (X * Wrow[:, None]).T @ X                # (7,7)
            J = J + 1e-6 * pt.eye(J.shape[0])
            V = pt.linalg.inv(J)
            sub = np.array(s["xidx"])
            V_sub = V[sub][:, sub]
            gamma_sub = gamma[np.array(s["gidx"])]
            pm.MvNormal(f"or_obs_{s['name']}", mu=gamma_sub, cov=V_sub,
                        observed=s["rep"])

        pm.Deterministic("OR_treat", pm.math.exp(mu_delta), dims="location")
        pm.Deterministic("baseline_risk_ref", pm.math.sigmoid(base_ref), dims="location")

        idata = pm.sample(draws=DRAWS, tune=TUNE, chains=CHAINS, cores=CHAINS,
                          target_accept=TARGET_ACCEPT, random_seed=SEED,
                          progressbar=False)

    post = idata.posterior

    def flat(name, idx=None):
        arr = post[name]
        if idx is not None:
            arr = arr.isel({arr.dims[-1]: idx})
        return arr.values.reshape(-1)

    gamma_draws = np.stack([flat("gamma", k) for k in range(K)], axis=1)
    base_draws = np.stack([flat("base_ref", i) for i in range(2)], axis=1)
    delta_draws = np.stack([flat("mu_delta", i) for i in range(2)], axis=1)
    np.savez(os.path.join(OUT, "ipd_posterior.npz"),
             gamma=gamma_draws, base_ref=base_draws, mu_delta=delta_draws,
             pred_order=np.array(PRED_ORDER, dtype=object),
             loc_levels=np.array(LOC_LEVELS, dtype=object))

    # gamma OR posterior + correlation matrix (the new information)
    or_draws = np.exp(gamma_draws)
    gamma_tab = pd.DataFrame({
        "predictor": PRED_ORDER,
        "OR_posterior": np.median(or_draws, axis=0),
        "OR_lo": np.percentile(or_draws, 2.5, axis=0),
        "OR_hi": np.percentile(or_draws, 97.5, axis=0),
        "OR_published": pred_df["odds_ratio"].to_numpy(float),
    })
    gamma_tab.to_csv(os.path.join(OUT, "ipd_predictor_ORs.csv"), index=False)
    corr = np.corrcoef(gamma_draws.T)
    pd.DataFrame(corr, index=PRED_ORDER, columns=PRED_ORDER).to_csv(
        os.path.join(OUT, "ipd_gamma_correlation.csv"))

    rows = []
    for i, loc in enumerate(LOC_LEVELS):
        orr = np.exp(flat("mu_delta", i))
        br = flat("baseline_risk_ref", i)
        rows.append(dict(location=loc, OR_treat=np.median(orr),
                         OR_lo=np.percentile(orr, 2.5), OR_hi=np.percentile(orr, 97.5),
                         base_risk_ref=np.median(br), base_lo=np.percentile(br, 2.5),
                         base_hi=np.percentile(br, 97.5)))
    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(OUT, "ipd_summary.csv"), index=False)

    az_summ = az.summary(idata, var_names=["gamma", "mu_delta", "base_ref"],
                         hdi_prob=0.95)
    diverging = int(idata.sample_stats["diverging"].values.sum())
    max_rhat = float(az_summ["r_hat"].max())

    lines = ["=== IPD estimation-process evidence model ==="]
    lines.append(f"Trial counts + source-study Fisher-information measurement model; "
                 f"{CHAINS} chains; divergences={diverging}; max R-hat={max_rhat:.3f}")
    for _, r in summ.iterrows():
        lines.append(f"  {r['location']:<9} treatment OR {r['OR_treat']:.2f} "
                     f"[{r['OR_lo']:.2f}, {r['OR_hi']:.2f}]   "
                     f"reference baseline {r['base_risk_ref']*100:.1f}%")
    lines.append("Predictor ORs (posterior vs published point estimate):")
    for _, r in gamma_tab.iterrows():
        lines.append(f"  {r['predictor']:<22} {r['OR_posterior']:.2f} "
                     f"[{r['OR_lo']:.2f}, {r['OR_hi']:.2f}]  (pub {r['OR_published']:.2f})")
    lines.append("Strongest gamma correlations induced by the joint estimation model:")
    iu = np.triu_indices(K, 1)
    order = np.argsort(-np.abs(corr[iu]))
    for j in order[:4]:
        a, b = iu[0][j], iu[1][j]
        lines.append(f"  corr({PRED_ORDER[a]}, {PRED_ORDER[b]}) = {corr[a,b]:+.2f}")
    report = "\n".join(lines)
    print(report)
    with open(os.path.join(OUT, "ipd_report.txt"), "w") as f:
        f.write(report + "\n")

    _plot_or_comparison(gamma_tab, os.path.join(OUT, "ipd_or_comparison.png"))

    return dict(summary=summ, gamma=gamma_tab, corr=corr,
                diverging=diverging, max_rhat=max_rhat)


def _plot_or_comparison(gamma_tab, path):
    """Posterior OR (this model) vs published point estimate."""
    ink, c = "#1f2a44", "#2f6f8f"
    y = np.arange(len(gamma_tab))[::-1]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.errorbar(gamma_tab["OR_posterior"], y,
                xerr=[gamma_tab["OR_posterior"] - gamma_tab["OR_lo"],
                      gamma_tab["OR_hi"] - gamma_tab["OR_posterior"]],
                fmt="o", color=c, ecolor=ink, capsize=3, ms=7,
                label="posterior (Fisher-info measurement model)")
    ax.scatter(gamma_tab["OR_published"], y, marker="|", s=200, color="#b4451f",
               label="published point estimate")
    ax.axvline(1, ls="--", color="#888", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(gamma_tab["predictor"])
    ax.set_xscale("log")
    ax.set_xlabel("Odds ratio for nonunion")
    ax.set_title("Predictor ORs: modelling the estimation process", loc="left",
                 fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    run()
