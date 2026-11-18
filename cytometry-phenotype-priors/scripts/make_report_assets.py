"""Aggregate results/runs/*.jsonl into tables and figures.
PYTHONPATH=src python scripts/make_report_assets.py"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cytoprior.data import load_dataset  # noqa: E402
from cytoprior.metrics import ordering_violations  # noqa: E402
from cytoprior.priors import load_prior  # noqa: E402

RES = ROOT / "results"
FIG = RES / "figures"
FIG.mkdir(parents=True, exist_ok=True)
CFG = yaml.safe_load((ROOT / "configs" / "experiments.yaml").read_text())
ORDER = list(CFG["methods"])


def load_runs(split="test"):
    rows = []
    for f in sorted((RES / "runs").glob(f"*_{split}.jsonl")):
        for line in f.read_text().splitlines():
            j = json.loads(line)
            if "error" in j:
                print("ERROR", j["dataset"], j["method"], j["sample"], j["seed"], j["error"])
                continue
            rows.append(j)
    return pd.DataFrame(rows)


def fmt(m, s):
    return "—" if np.isnan(m) else f"{m:.3f} ± {s:.3f}"


def summary_table(df):
    """Per (dataset, method): average over samples within a seed, then mean ± sd over seeds."""
    out = []
    for (ds, me), g in df.groupby(["dataset", "method"]):
        cols = {}
        for c in ["hung_macro_f1", "id_macro_f1", "ari", "rare_f1", "model_order_violation", "runtime", "maxrss_mb"]:
            if c in g and g[c].notna().any():
                per_seed = g.groupby("seed")[c].mean()
                cols[c] = (per_seed.mean(), per_seed.std(ddof=0))
            else:
                cols[c] = (np.nan, np.nan)
        ece = g["calibration"].dropna().map(lambda c: c["ece"]) if "calibration" in g else pd.Series(dtype=float)
        cols["ece"] = (ece.mean(), ece.std(ddof=0)) if len(ece) else (np.nan, np.nan)
        out.append(dict(dataset=ds, method=me, n_runs=len(g), **{k: v[0] for k, v in cols.items()},
                        **{k + "_sd": v[1] for k, v in cols.items()}))
    t = pd.DataFrame(out)
    t["order"] = t.method.map(lambda m: ORDER.index(m) if m in ORDER else 99)
    return t.sort_values(["dataset", "order"]).drop(columns="order")


def write_markdown(t):
    lines = []
    for ds, g in t.groupby("dataset", sort=False):
        kind = CFG["datasets"][ds]["kind"]
        lines.append(f"\n### {ds} ({'held-out test samples ' + ','.join(CFG['datasets'][ds]['test']) if kind == 'multi' else 'single sample, in-sample'})\n")
        if kind == "multi":
            lines.append("| method | Hungarian macro F1 | identity macro F1 | ARI | ECE | model order viol. | runtime s | n |")
            lines.append("|---|---|---|---|---|---|---|---|")
            for _, r in g.iterrows():
                lines.append(f"| {r.method} | {fmt(r.hung_macro_f1, r.hung_macro_f1_sd)} | {fmt(r.id_macro_f1, r.id_macro_f1_sd)} | "
                             f"{fmt(r.ari, r.ari_sd)} | {'—' if np.isnan(r.ece) else f'{r.ece:.3f}'} | "
                             f"{'—' if np.isnan(r.model_order_violation) else f'{r.model_order_violation:.3f}'} | {r.runtime:.0f} | {r.n_runs} |")
        else:
            lines.append("| method | rare F1 (best cluster, W&R) | identity F1 (prior comp.) | ECE | runtime s | n |")
            lines.append("|---|---|---|---|---|---|")
            for _, r in g.iterrows():
                idf = fmt(r.id_macro_f1, r.id_macro_f1_sd)
                lines.append(f"| {r.method} | {fmt(r.rare_f1, r.rare_f1_sd)} | {idf} | "
                             f"{'—' if np.isnan(r.ece) else f'{r.ece:.3f}'} | {r.runtime:.0f} | {r.n_runs} |")
    (RES / "results_table.md").write_text("\n".join(lines) + "\n")


def per_pop_figure(df, ds, methods):
    d = load_dataset(ds)
    p = load_prior(ds, d.markers, d.label_names)
    sizes = np.bincount(d.y[d.y >= 0], minlength=p.K)
    order = np.argsort(-sizes)
    M = []
    labels = []
    for me in methods:
        g = df[(df.dataset == ds) & (df.method == me)]
        if not len(g):
            continue
        col = "id_f1" if g.identity.iloc[0] else "hung_f1"
        M.append(np.mean(np.stack(g[col].values), 0)[order])
        labels.append(f"{me} ({'identity' if col == 'id_f1' else 'Hungarian'})")
    if not M:
        return
    M = np.array(M)
    fig, ax = plt.subplots(figsize=(0.42 * p.K + 3, 0.38 * len(M) + 2))
    im = ax.imshow(M, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.set_xticks(range(p.K), [f"{p.populations[k]} ({sizes[k] / sizes.sum() * 100:.1f}%)" for k in order],
                  rotation=70, ha="right", fontsize=7)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}"[1:] if M[i, j] < 1 else "1", ha="center", va="center", fontsize=5.5,
                    color="w" if M[i, j] < 0.6 else "k")
    fig.colorbar(im, ax=ax, fraction=0.02, label="F1 (mean over test samples & seeds)")
    ax.set_title(f"{ds}: per-population F1, populations sorted by abundance")
    fig.tight_layout()
    fig.savefig(FIG / f"per_population_f1_{ds}.png", dpi=160)
    plt.close(fig)


def calibration_figure(df, methods):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, ds in zip(axes, ["Samusik_all", "Levine_32dim"]):
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        for me in methods:
            g = df[(df.dataset == ds) & (df.method == me)]
            if not len(g) or "calibration" not in g or g.calibration.isna().all():
                continue
            cs = g.calibration.dropna().tolist()
            cnt = np.sum([c["count"] for c in cs], 0)
            conf = np.sum([np.array(c["conf"]) * np.array(c["count"]) for c in cs], 0) / np.maximum(cnt, 1)
            acc = np.sum([np.array(c["acc"]) * np.array(c["count"]) for c in cs], 0) / np.maximum(cnt, 1)
            ok = cnt > 50
            ece = np.mean([c["ece"] for c in cs])
            ax.plot(conf[ok], acc[ok], "o-", ms=3, label=f"{me} (ECE {ece:.3f})")
        ax.set_xlabel("predicted probability of assigned population")
        ax.set_ylabel("fraction correct")
        ax.set_title(f"Reliability, {ds} (test, gated cells)")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "calibration.png", dpi=160)
    plt.close(fig)


def gates_vs_model_figure(ds, sample, pairs, methods):
    d = load_dataset(ds)
    s = d.by_sample(sample)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(s.y), min(30000, len(s.y)), replace=False)
    cmap = plt.get_cmap("tab20", 24)
    panels = [("manual gates", s.y)]
    for me in methods:
        f = RES / "cache" / f"{ds}_{me}_{sample}.npz"
        if f.exists():
            z = np.load(f)
            pp = z["pred_pop"] if z["pred_pop"].size else None
            if pp is None:  # Hungarian-map clusters to populations for display
                from cytoprior.metrics import f1_matrix
                from scipy.optimize import linear_sum_assignment
                cl = z["cluster"]
                m = s.y >= 0
                F, ids, _, _ = f1_matrix(cl[m], s.y[m], len(d.label_names))
                r, c = linear_sum_assignment(-F)
                lut = {ids[a]: b for a, b in zip(r, c)}
                pp = np.array([lut.get(x, -1) for x in cl])
            panels.append((me, pp))
    fig, axes = plt.subplots(len(pairs), len(panels), figsize=(3.2 * len(panels), 3.0 * len(pairs)), squeeze=False)
    for i, (mx, my) in enumerate(pairs):
        jx, jy = d.markers.index(mx), d.markers.index(my)
        for j, (title, lab) in enumerate(panels):
            ax = axes[i, j]
            l = lab[idx]
            un = l < 0
            ax.scatter(s.X[idx][un, jx], s.X[idx][un, jy], s=0.3, c="0.85", rasterized=True)
            ax.scatter(s.X[idx][~un, jx], s.X[idx][~un, jy], s=0.3, c=cmap(l[~un] % 24), rasterized=True)
            ax.set_xlabel(mx, fontsize=7)
            ax.set_ylabel(my, fontsize=7)
            ax.tick_params(labelsize=6)
            if i == 0:
                ax.set_title(title, fontsize=8)
    handles = [plt.Line2D([], [], marker="o", ls="", color=cmap(k % 24), label=n, ms=4) for k, n in enumerate(d.label_names)]
    handles.append(plt.Line2D([], [], marker="o", ls="", color="0.85", label="unassigned / unknown", ms=4))
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=6, frameon=False)
    fig.suptitle(f"{ds}, test sample {sample}: manual gates vs model assignments (seed 0)", fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 0.97))
    fig.savefig(FIG / f"gates_vs_models_{ds}_{sample}.png", dpi=150)
    plt.close(fig)


def empirical_violations():
    """Do HELD-OUT samples' manual gates violate the prior orderings? (tests the definitions)"""
    rows = []
    for ds in ["Samusik_all", "Levine_32dim"]:
        d = load_dataset(ds)
        p = load_prior(ds, d.markers, d.label_names)
        for s in CFG["datasets"][ds]["test"]:
            v = ordering_violations(d.by_sample(s).X, d.by_sample(s).y, p)
            rows.append(dict(dataset=ds, sample=s, rate=v["rate"], n_edges=v["n_edges"],
                             violated=[f"{m}: {a} ({x:.2f}) !< {b} ({y:.2f})" for m, a, b, x, y in v["violated"]]))
    (RES / "ordering_violations_heldout.json").write_text(json.dumps(rows, indent=1))
    return rows


if __name__ == "__main__":
    df = load_runs("test")
    t = summary_table(df)
    t.to_csv(RES / "results_table.csv", index=False)
    write_markdown(t)
    print((RES / "results_table.md").read_text())
    for ds in ["Samusik_all", "Levine_32dim"]:
        per_pop_figure(df, ds, ["R_rules_classifier", "L0_gmm", "L1_t", "L2_flowsom", "L3_order", "L4_rules",
                                "L4_both", "L5_both_ref1", "L5_both_ref3", "L5_neither_ref3", "L6_both"])
    calibration_figure(df, ["R_rules_classifier", "L3_order", "L4_both", "L5_both_ref3", "L5_neither_ref3", "L6_both"])
    gates_vs_model_figure("Samusik_all", "06", [("CD11b", "Ly6C"), ("CD19", "CD3"), ("Sca1", "cKit")],
                          ["R_rules_classifier", "L1_t", "L4_both", "L5_both_ref3", "L6_both"])
    gates_vs_model_figure("Levine_32dim", "H2", [("CD3", "CD19"), ("CD34", "CD38")],
                          ["R_rules_classifier", "L1_t", "L4_both", "L5_both_ref1", "L6_both"])
    print(json.dumps(empirical_violations(), indent=1)[:3000])
