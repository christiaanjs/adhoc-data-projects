"""
Hierarchical (multilevel) Bayesian meta-analysis of nonunion, operative vs
nonoperative treatment, covering BOTH midshaft and distal (lateral-third)
clavicle fractures. Fitted with PyMC using the NUTS sampler.

Structure (3-level normal-normal on the log risk ratio):

    y_i        ~ Normal(theta_i, s_i^2)      # observed effect | study true effect
    theta_i    ~ Normal(mu_{g[i]}, tau^2)     # study | fracture-location mean
    mu_g       = M + omega * z_g              # location mean | overall (non-centred)
    M          ~ Normal(0, 10)                # overall log-RR (vague)
    tau, omega ~ HalfNormal(1)                # weakly-informative SDs

s_i^2 (within-study sampling variance) is treated as known — the standard
meta-analytic assumption. Partial pooling lets the sparse DISTAL subgroup borrow
strength from the (better-evidenced) overall effect while still getting its own
estimate. With only two location groups, omega (between-location SD) is weakly
identified; the HalfNormal prior regularises it (reported and discussed).

Non-centred parameterisations are used for both random-effect levels so NUTS
does not fight the funnel geometry. Convergence is checked with R-hat and ESS.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

import pymc as pm
import arviz as az

from meta_analysis import _effect_per_study, dersimonian_laird

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "outputs")
os.makedirs(OUT, exist_ok=True)

MIDSHAFT = os.path.join(ROOT, "data", "meta_analysis_trials.csv")
DISTAL = os.path.join(ROOT, "data", "distal_trials.csv")

M_PRIOR_SD = 10.0        # vague prior on overall log-RR
HALFNORMAL_SCALE = 1.0   # weakly-informative prior scale for tau, omega
DRAWS = 3000
TUNE = 2000
CHAINS = 4
SEED = 20240702


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_all():
    mid = pd.read_csv(MIDSHAFT)
    mid["location"] = "midshaft"
    if "design" not in mid.columns:
        mid["design"] = "RCT"
    dist = pd.read_csv(DISTAL)
    dist["location"] = "distal"
    cols = ["study", "year", "n_op", "events_op", "n_nonop", "events_nonop",
            "location", "design"]
    return pd.concat([mid[cols], dist[cols]], ignore_index=True)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def build_model(y, s, groups, n_groups):
    coords = {"study": np.arange(len(y)), "location": ["midshaft", "distal"]}
    with pm.Model(coords=coords) as model:
        y_obs = pm.Data("y_obs", y, dims="study")
        s_obs = pm.Data("s_obs", s, dims="study")
        g_idx = pm.Data("g_idx", groups, dims="study")

        M = pm.Normal("M", 0.0, M_PRIOR_SD)                  # overall log-RR
        omega = pm.HalfNormal("omega", HALFNORMAL_SCALE)     # between-location SD
        tau = pm.HalfNormal("tau", HALFNORMAL_SCALE)         # between-study SD

        # location means (non-centred)
        z_loc = pm.Normal("z_loc", 0.0, 1.0, dims="location")
        mu = pm.Deterministic("mu", M + omega * z_loc, dims="location")

        # study true effects (non-centred)
        z_study = pm.Normal("z_study", 0.0, 1.0, dims="study")
        theta = pm.Deterministic("theta", mu[g_idx] + tau * z_study, dims="study")

        pm.Normal("lik", mu=theta, sigma=s_obs, observed=y_obs, dims="study")

        # useful contrasts
        pm.Deterministic("mu_midshaft", mu[0])
        pm.Deterministic("mu_distal", mu[1])
        pm.Deterministic("diff_mid_minus_distal", mu[0] - mu[1])
    return model


# --------------------------------------------------------------------------- #
# Run + summarise
# --------------------------------------------------------------------------- #
def _rr(draws):
    return (np.exp(np.median(draws)),
            np.exp(np.percentile(draws, 2.5)),
            np.exp(np.percentile(draws, 97.5)))


def run():
    df = load_all()
    y, v = _effect_per_study(df, "RR")
    y = np.asarray(y, float)
    s = np.sqrt(np.asarray(v, float))          # within-study SD (known)
    loc_levels = ["midshaft", "distal"]
    groups = df["location"].map({l: i for i, l in enumerate(loc_levels)}).to_numpy()

    model = build_model(y, s, groups, len(loc_levels))
    with model:
        idata = pm.sample(draws=DRAWS, tune=TUNE, chains=CHAINS, cores=CHAINS,
                          target_accept=0.99, random_seed=SEED,
                          progressbar=False)

    post = idata.posterior
    def flat(name, idx=None):
        arr = post[name]
        if idx is not None:
            arr = arr.isel({arr.dims[-1]: idx})
        return arr.values.reshape(-1)

    # Pooled estimates
    rows = []
    for lbl, draws in [("Overall (both locations)", flat("M")),
                       ("Midshaft", flat("mu_midshaft")),
                       ("Distal (lateral-third)", flat("mu_distal"))]:
        rr, lo, hi = _rr(draws)
        rows.append(dict(param=lbl, RR_median=rr, RR_lo=lo, RR_hi=hi,
                         logRR_mean=float(draws.mean()), logRR_sd=float(draws.std())))
    summ = pd.DataFrame(rows)

    tau_draws, omega_draws = flat("tau"), flat("omega")
    diff = flat("diff_mid_minus_distal")
    p_mid_lower = float(np.mean(diff < 0))

    # Convergence diagnostics via ArviZ
    az_summ = az.summary(idata, var_names=["M", "mu", "tau", "omega"],
                         hdi_prob=0.95)
    az_summ.to_csv(os.path.join(OUT, "hier_arviz_summary.csv"))
    diag_df = az_summ[["r_hat", "ess_bulk"]].reset_index().rename(
        columns={"index": "parameter", "r_hat": "R_hat", "ess_bulk": "ESS"})

    # Shrunken (partial-pooled) per-study estimates
    z = stats.norm.ppf(0.975)
    shrunk = []
    for i in range(len(df)):
        d = flat("theta", i)
        raw = y[i]
        rr, lo, hi = _rr(d)
        shrunk.append((df["study"].iloc[i], df["location"].iloc[i],
                       np.exp(raw), np.exp(raw - z * s[i]), np.exp(raw + z * s[i]),
                       rr, lo, hi))
    shrunk_df = pd.DataFrame(shrunk, columns=[
        "study", "location", "RR_raw", "RR_raw_lo", "RR_raw_hi",
        "RR_shrunk", "RR_shrunk_lo", "RR_shrunk_hi"])

    # Independent subgroup random-effects (no cross-location pooling) for contrast
    sub = {}
    for g, lab in enumerate(loc_levels):
        m = groups == g
        sub[lab] = dersimonian_laird(y[m], v[m])

    # Persist RR posterior draws by location so the treatment-benefit CLI can
    # propagate the full uncertainty of the meta-analytic effect.
    np.savez(os.path.join(OUT, "rr_posterior.npz"),
             midshaft=np.exp(flat("mu_midshaft")),
             distal=np.exp(flat("mu_distal")),
             overall=np.exp(flat("M")))

    # ----- write outputs -----
    summ.to_csv(os.path.join(OUT, "hier_pooled_estimates.csv"), index=False)
    shrunk_df.to_csv(os.path.join(OUT, "hier_shrunken_studies.csv"), index=False)
    diag_df.to_csv(os.path.join(OUT, "hier_diagnostics.csv"), index=False)

    _forest(df, shrunk_df, summ, groups, loc_levels,
            os.path.join(OUT, "hier_forest.png"))
    _shrinkage_plot(shrunk_df, os.path.join(OUT, "hier_shrinkage.png"))
    _variance_plot(tau_draws, omega_draws, os.path.join(OUT, "hier_variance.png"))

    diverging = int(idata.sample_stats["diverging"].values.sum())
    max_rhat = float(diag_df["R_hat"].max())

    # ----- console + text report -----
    lines = []
    lines.append("=== Hierarchical Bayesian meta-analysis (PyMC/NUTS): midshaft + distal ===")
    lines.append(f"Studies: {len(df)} ({int((groups==0).sum())} midshaft, "
                 f"{int((groups==1).sum())} distal); "
                 f"{CHAINS} chains x {DRAWS} draws (tune {TUNE}); "
                 f"divergences={diverging}; max R-hat={max_rhat:.3f}")
    for _, r in summ.iterrows():
        lines.append(f"  {r['param']:<28} RR {r['RR_median']:.2f} "
                     f"[{r['RR_lo']:.2f}, {r['RR_hi']:.2f}]")
    lines.append(f"  Between-study SD  tau  : median {np.median(tau_draws):.2f} "
                 f"[{np.percentile(tau_draws,2.5):.2f}, {np.percentile(tau_draws,97.5):.2f}]")
    lines.append(f"  Between-location SD omega: median {np.median(omega_draws):.2f} "
                 f"[{np.percentile(omega_draws,2.5):.2f}, {np.percentile(omega_draws,97.5):.2f}] "
                 f"(weakly identified: only 2 groups)")
    lines.append(f"  P(midshaft RR < distal RR) = {p_mid_lower:.2f}  "
                 f"(both strongly favour surgery; little evidence of a location difference)")
    lines.append("  Independent subgroup random-effects (no cross-location pooling):")
    for lab in loc_levels:
        st = sub[lab]
        lines.append(f"    {lab:<10} RR {st.estimate:.2f} [{st.ci_low:.2f}, {st.ci_high:.2f}]")
    lines.append("  Convergence (R-hat < 1.01 and healthy ESS expected):")
    for _, r in diag_df.iterrows():
        lines.append(f"    {r['parameter']:<14} R-hat {r['R_hat']:.3f}  ESS {r['ESS']:.0f}")
    report = "\n".join(lines)
    print(report)
    with open(os.path.join(OUT, "hier_report.txt"), "w") as f:
        f.write(report + "\n")

    return dict(summary=summ, shrunk=shrunk_df, diagnostics=diag_df,
                tau=tau_draws, omega=omega_draws, p_mid_lower=p_mid_lower,
                subgroups=sub, loc_levels=loc_levels, diverging=diverging,
                max_rhat=max_rhat)


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def _forest(df, shrunk_df, summ, groups, loc_levels, path):
    ink, raw_c, shr_c = "#1f2a44", "#9bb7c7", "#2f6f8f"
    diamond_c = "#b4451f"

    rows = []  # (kind, label, RR, lo, hi)
    for lab in loc_levels:
        rows.append(("header", lab.upper(), None, None, None))
        sub = shrunk_df[shrunk_df["location"] == lab]
        for _, r in sub.iterrows():
            rows.append(("study", r["study"], r["RR_shrunk"],
                         r["RR_shrunk_lo"], r["RR_shrunk_hi"]))
        key = "Distal" if lab == "distal" else "Midshaft"
        srow = summ[summ["param"].str.startswith(key)].iloc[0]
        rows.append(("subtotal", f"{lab} pooled", srow["RR_median"],
                     srow["RR_lo"], srow["RR_hi"]))
    orow = summ[summ["param"].str.startswith("Overall")].iloc[0]
    rows.append(("overall", "Overall pooled", orow["RR_median"],
                 orow["RR_lo"], orow["RR_hi"]))

    n = len(rows)
    fig, ax = plt.subplots(figsize=(9.5, 0.5 * n + 1.6))
    trans = ax.get_yaxis_transform()
    raw_map = {r["study"]: (r["RR_raw"], r["RR_raw_lo"], r["RR_raw_hi"])
               for _, r in shrunk_df.iterrows()}

    for j, (kind, label, rr, lo, hi) in enumerate(rows):
        y = n - j
        if kind == "header":
            ax.text(-0.02, y, label, transform=trans, ha="right", va="center",
                    fontsize=9.5, fontweight="bold", color=ink, clip_on=False)
            continue
        if kind == "study":
            r_raw, rl, rh = raw_map[label]
            ax.plot([rl, rh], [y + 0.16, y + 0.16], color=raw_c, lw=1.2, zorder=1)
            ax.scatter([r_raw], [y + 0.16], s=22, color=raw_c, zorder=2)
            ax.plot([lo, hi], [y, y], color=ink, lw=1.4, zorder=3)
            ax.scatter([rr], [y], s=46, color=shr_c, edgecolor=ink,
                       linewidth=0.7, zorder=4)
            ax.text(-0.02, y, label, transform=trans, ha="right", va="center",
                    fontsize=8.5, color=ink, clip_on=False)
            ax.text(1.03, y, f"{rr:.2f} [{lo:.2f}, {hi:.2f}]", transform=trans,
                    ha="left", va="center", fontsize=8, color=ink, clip_on=False)
        else:
            c = diamond_c if kind == "overall" else shr_c
            h = 0.30 if kind == "overall" else 0.24
            ax.fill([lo, rr, hi, rr], [y, y + h, y, y - h], color=c, zorder=5)
            fw = "bold" if kind == "overall" else "normal"
            ax.text(-0.02, y, label, transform=trans, ha="right", va="center",
                    fontsize=9, fontweight=fw, color=c, clip_on=False)
            ax.text(1.03, y, f"{rr:.2f} [{lo:.2f}, {hi:.2f}]", transform=trans,
                    ha="left", va="center", fontsize=8, fontweight=fw,
                    color=c, clip_on=False)

    ax.axvline(1.0, color="#888", lw=1, ls="--", zorder=0)
    ax.set_xscale("log")
    ax.set_xlim(0.02, 5)
    ax.set_xticks([0.05, 0.1, 0.25, 0.5, 1, 2, 4])
    ax.set_xticklabels(["0.05", "0.1", "0.25", "0.5", "1", "2", "4"])
    ax.set_ylim(0.2, n + 1)
    ax.set_yticks([])
    for sname in ["top", "right", "left"]:
        ax.spines[sname].set_visible(False)
    ax.set_xlabel("Risk ratio for nonunion  (< 1 favours operative fixation)",
                  fontsize=10, color=ink)
    ax.set_title("Hierarchical meta-analysis (PyMC/NUTS): nonunion by fracture location\n"
                 "faint = raw estimate, solid = partial-pooled (shrunken)",
                 fontsize=11.5, color=ink, loc="left", pad=16)
    fig.subplots_adjust(left=0.24, right=0.72, top=0.86, bottom=0.13)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _shrinkage_plot(shrunk_df, path):
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    colors = {"midshaft": "#2f6f8f", "distal": "#b4451f"}
    for _, r in shrunk_df.iterrows():
        c = colors[r["location"]]
        ax.plot([0, 1], [r["RR_raw"], r["RR_shrunk"]], color=c, alpha=0.5, lw=1)
        ax.scatter([0], [r["RR_raw"]], color=c, s=28)
        ax.scatter([1], [r["RR_shrunk"]], color=c, s=28)
    ax.set_yscale("log")
    ax.set_xlim(-0.2, 1.2)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["raw (no pooling)", "partial-pooled"])
    ax.set_ylabel("Risk ratio for nonunion (log scale)")
    ax.axhline(1, color="#999", ls="--", lw=1)
    handles = [plt.Line2D([], [], color=v, marker="o", ls="", label=k)
               for k, v in colors.items()]
    ax.legend(handles=handles, frameon=False, fontsize=9)
    ax.set_title("Shrinkage of study estimates toward location/overall means",
                 loc="left", fontsize=11)
    for sname in ["top", "right"]:
        ax.spines[sname].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _variance_plot(tau, omega, path):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.hist(tau, bins=40, density=True, alpha=0.6, color="#2f6f8f",
            label=r"$\tau$ (between-study)")
    ax.hist(omega, bins=40, density=True, alpha=0.6, color="#b4451f",
            label=r"$\omega$ (between-location)")
    ax.set_xlabel("SD on the log-RR scale")
    ax.set_ylabel("Posterior density")
    ax.set_title("Posterior of variance components", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    for sname in ["top", "right"]:
        ax.spines[sname].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    run()
