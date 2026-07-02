"""
Random-effects meta-analysis of nonunion after operative vs nonoperative
treatment of displaced midshaft clavicle fractures.

Method: DerSimonian-Laird random-effects model on the log risk ratio, with a
0.5 continuity correction for zero-event arms. Also reports the pooled odds
ratio, between-study heterogeneity (Cochran's Q, I^2, tau^2) and produces a
forest plot.

Everything here is implemented explicitly (no meta-analysis black box) so the
arithmetic is auditable.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data", "meta_analysis_trials.csv")
OUT = os.path.join(ROOT, "outputs")
os.makedirs(OUT, exist_ok=True)

CC = 0.5  # continuity correction for zero cells


@dataclass
class PooledResult:
    estimate: float          # pooled effect on the natural scale (e.g. RR)
    ci_low: float
    ci_high: float
    log_estimate: float
    se_log: float
    tau2: float
    Q: float
    df: int
    p_Q: float
    I2: float
    weights: np.ndarray      # random-effects weights (%)


def _effect_per_study(df: pd.DataFrame, measure: str):
    """Return log-effect and its variance per study, with continuity correction."""
    a = df["events_op"].to_numpy(float)      # events, operative
    b = df["n_op"].to_numpy(float) - a       # non-events, operative
    c = df["events_nonop"].to_numpy(float)   # events, nonoperative
    d = df["n_nonop"].to_numpy(float) - c    # non-events, nonoperative

    # Apply continuity correction only to studies with a zero cell.
    zero = (a == 0) | (b == 0) | (c == 0) | (d == 0)
    a = a + CC * zero
    b = b + CC * zero
    c = c + CC * zero
    d = d + CC * zero
    n_op = a + b
    n_nonop = c + d

    if measure == "RR":
        risk_op = a / n_op
        risk_nonop = c / n_nonop
        log_eff = np.log(risk_op / risk_nonop)
        var = (1 / a) - (1 / n_op) + (1 / c) - (1 / n_nonop)
    elif measure == "OR":
        log_eff = np.log((a * d) / (b * c))
        var = 1 / a + 1 / b + 1 / c + 1 / d
    else:
        raise ValueError(measure)
    return log_eff, var


def dersimonian_laird(log_eff: np.ndarray, var: np.ndarray) -> PooledResult:
    w_fixed = 1.0 / var
    fixed_mean = np.sum(w_fixed * log_eff) / np.sum(w_fixed)

    # Cochran's Q and DL tau^2
    Q = float(np.sum(w_fixed * (log_eff - fixed_mean) ** 2))
    k = len(log_eff)
    df = k - 1
    C = np.sum(w_fixed) - np.sum(w_fixed ** 2) / np.sum(w_fixed)
    tau2 = max(0.0, (Q - df) / C) if C > 0 else 0.0

    w_random = 1.0 / (var + tau2)
    mean = np.sum(w_random * log_eff) / np.sum(w_random)
    se = np.sqrt(1.0 / np.sum(w_random))
    z = stats.norm.ppf(0.975)
    lo, hi = mean - z * se, mean + z * se

    I2 = max(0.0, (Q - df) / Q) * 100 if Q > 0 else 0.0
    p_Q = float(stats.chi2.sf(Q, df)) if df > 0 else float("nan")
    weights_pct = 100 * w_random / np.sum(w_random)

    return PooledResult(
        estimate=float(np.exp(mean)),
        ci_low=float(np.exp(lo)),
        ci_high=float(np.exp(hi)),
        log_estimate=float(mean),
        se_log=float(se),
        tau2=float(tau2),
        Q=Q,
        df=df,
        p_Q=p_Q,
        I2=float(I2),
        weights=weights_pct,
    )


def number_needed_to_treat(df: pd.DataFrame):
    """Pooled absolute risk difference and NNT to prevent one nonunion."""
    risk_op = df["events_op"].sum() / df["n_op"].sum()
    risk_nonop = df["events_nonop"].sum() / df["n_nonop"].sum()
    ard = risk_nonop - risk_op
    return risk_op, risk_nonop, ard, (1.0 / ard if ard > 0 else np.inf)


def forest_plot(df: pd.DataFrame, log_eff, var, pooled: PooledResult, measure: str, path: str):
    z = stats.norm.ppf(0.975)
    eff = np.exp(log_eff)
    lo = np.exp(log_eff - z * np.sqrt(var))
    hi = np.exp(log_eff + z * np.sqrt(var))

    labels = [f"{s} ({y})" for s, y in zip(df["study"], df["year"])]
    n = len(df)
    ypos = np.arange(n)[::-1] + 1  # studies top-to-bottom

    fig, ax = plt.subplots(figsize=(9, 0.62 * n + 2.4))
    ink = "#1f2a44"
    accent = "#2f6f8f"
    diamond_c = "#b4451f"

    # Reserve gutters outside the plotting area for text (axes-fraction x).
    LBL_X = -0.02   # study labels: left of the axis
    STAT_X = 1.03   # per-study statistics: right of the axis
    trans = ax.get_yaxis_transform()

    for i in range(n):
        y = ypos[i]
        ax.plot([lo[i], hi[i]], [y, y], color=ink, lw=1.4, zorder=2)
        size = 40 + 8 * pooled.weights[i]
        ax.scatter([eff[i]], [y], s=size, color=accent, edgecolor=ink,
                   linewidth=0.8, zorder=3)
        txt = (f"{int(df['events_op'].iloc[i])}/{int(df['n_op'].iloc[i])}"
               f"  vs  {int(df['events_nonop'].iloc[i])}/{int(df['n_nonop'].iloc[i])}"
               f"   {measure} {eff[i]:.2f} [{lo[i]:.2f}, {hi[i]:.2f}]"
               f"   ({pooled.weights[i]:.0f}%)")
        ax.text(STAT_X, y, txt, va="center", ha="left", fontsize=8.5,
                color=ink, transform=trans, clip_on=False)
        ax.text(LBL_X, y, labels[i], va="center", ha="right", fontsize=9,
                color=ink, transform=trans, clip_on=False)

    # Pooled diamond
    yd = 0.2
    d_lo, d_c, d_hi = pooled.ci_low, pooled.estimate, pooled.ci_high
    ax.fill([d_lo, d_c, d_hi, d_c], [yd, yd + 0.28, yd, yd - 0.28],
            color=diamond_c, zorder=4)
    ax.text(LBL_X, yd, "Random-effects pooled", va="center", ha="right",
            fontsize=9, fontweight="bold", color=diamond_c,
            transform=trans, clip_on=False)
    ax.text(STAT_X, yd, f"{measure} {d_c:.2f} [{d_lo:.2f}, {d_hi:.2f}]",
            va="center", ha="left", fontsize=8.5, fontweight="bold",
            color=diamond_c, transform=trans, clip_on=False)

    ax.axvline(1.0, color="#888", lw=1, ls="--", zorder=1)
    ax.set_xscale("log")
    ax.set_xlim(0.02, 5)
    ax.set_xticks([0.05, 0.1, 0.25, 0.5, 1, 2, 4])
    ax.set_xticklabels(["0.05", "0.1", "0.25", "0.5", "1", "2", "4"])
    ax.set_ylim(-0.4, n + 1.2)
    ax.set_yticks([])
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.set_xlabel(f"{measure} for nonunion  (< 1 favours operative fixation)",
                  fontsize=10, color=ink)
    ax.set_title("Nonunion after operative vs nonoperative treatment of\n"
                 "displaced midshaft clavicle fractures",
                 fontsize=12, color=ink, loc="left", pad=24)
    ax.text(LBL_X, n + 0.9, "Study (year)", fontsize=8.5, style="italic",
            color="#555", ha="right", transform=trans, clip_on=False)
    ax.text(STAT_X, n + 0.9, "events (op vs nonop) · effect [95% CI] · weight",
            fontsize=8.5, style="italic", color="#555", ha="left",
            transform=trans, clip_on=False)
    # Wide margins so the gutters have room.
    fig.subplots_adjust(left=0.22, right=0.66, top=0.86, bottom=0.12)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run():
    df = pd.read_csv(DATA)
    results = {}
    for measure in ("RR", "OR"):
        log_eff, var = _effect_per_study(df, measure)
        pooled = dersimonian_laird(log_eff, var)
        results[measure] = (log_eff, var, pooled)
        forest_plot(df, log_eff, var, pooled, measure,
                    os.path.join(OUT, f"forest_{measure.lower()}.png"))

    risk_op, risk_nonop, ard, nnt = number_needed_to_treat(df)

    # Assemble a tidy summary table for the report.
    rr = results["RR"][2]
    orr = results["OR"][2]
    summary = pd.DataFrame({
        "measure": ["Risk ratio", "Odds ratio"],
        "pooled": [rr.estimate, orr.estimate],
        "ci_low": [rr.ci_low, orr.ci_low],
        "ci_high": [rr.ci_high, orr.ci_high],
        "I2_pct": [rr.I2, orr.I2],
        "tau2": [rr.tau2, orr.tau2],
        "Q": [rr.Q, orr.Q],
        "p_Q": [rr.p_Q, orr.p_Q],
    })
    summary.to_csv(os.path.join(OUT, "meta_analysis_summary.csv"), index=False)

    lines = []
    lines.append("=== Random-effects meta-analysis: nonunion, operative vs nonoperative ===")
    lines.append(f"Trials: {len(df)}   Patients: {int(df['n_op'].sum()+df['n_nonop'].sum())} "
                 f"(operative {int(df['n_op'].sum())}, nonoperative {int(df['n_nonop'].sum())})")
    lines.append(f"Crude nonunion risk: operative {risk_op*100:.1f}%  "
                 f"nonoperative {risk_nonop*100:.1f}%")
    lines.append(f"Absolute risk difference: {ard*100:.1f} percentage points   "
                 f"NNT (to prevent one nonunion): {nnt:.0f}")
    for measure in ("RR", "OR"):
        p = results[measure][2]
        sig = "favours operative" if p.ci_high < 1 else "not significant"
        lines.append(f"{measure}: {p.estimate:.3f}  95% CI [{p.ci_low:.3f}, {p.ci_high:.3f}]  "
                     f"({sig});  I^2={p.I2:.0f}%  tau^2={p.tau2:.3f}  "
                     f"Q={p.Q:.2f} (p={p.p_Q:.3f})")
    report = "\n".join(lines)
    print(report)
    with open(os.path.join(OUT, "meta_analysis_report.txt"), "w") as f:
        f.write(report + "\n")
    return results, summary


if __name__ == "__main__":
    run()
