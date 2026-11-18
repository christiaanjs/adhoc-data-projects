"""Run the ablation ladder on the test (or validation) samples. Resumable: results are appended as
JSON lines to results/runs/<dataset>.jsonl and finished (method, sample, seed) keys are skipped.

  PYTHONPATH=src python scripts/run_experiments.py --datasets Samusik_all --split test --workers 4
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
import traceback
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
CFG = yaml.safe_load((ROOT / "configs" / "experiments.yaml").read_text())
_DS = {}


def _data(name):
    if name not in _DS:
        from cytoprior.data import load_dataset
        from cytoprior.priors import load_prior
        d = load_dataset(name)
        _DS[name] = (d, load_prior(name, d.markers, d.label_names))
    return _DS[name]


def run_one(job):
    import torch
    torch.set_num_threads(1)
    import warnings
    warnings.filterwarnings("ignore")
    from cytoprior import metrics as M
    from cytoprior.models import fit_flowsom, fit_gmm, fit_prior_mixture
    name, method, sample, seed = job
    dcfg = CFG["datasets"][name]
    mcfg = {**CFG["common"], "n_unknown": dcfg["n_unknown"], **CFG["methods"][method]}
    d, prior = _data(name)
    ds = d.by_sample(sample)
    K = prior.K
    Ktot = K + mcfg["n_unknown"]
    rng = np.random.default_rng(seed)
    sub = rng.choice(len(ds.y), min(20000, len(ds.y)), replace=False)
    t0 = time.time()
    try:
        fam = mcfg["family"]
        if fam == "gmm":
            r = fit_gmm(ds.X[sub], ds.X, Ktot, seed)
        elif fam == "rules":
            from cytoprior.models import fit_rules_classifier
            r = fit_rules_classifier(ds.X, prior, {**mcfg}, seed)
        elif fam == "flowsom":
            r = fit_flowsom(ds.X, mcfg["k"], seed + 1)
        else:
            Xlab = ylab = None
            if mcfg.get("n_ref"):
                refs = dcfg["ref"][: mcfg["n_ref"]]
                parts = []
                for rs in refs:
                    dr = d.by_sample(rs)
                    ii = rng.choice(len(dr.y), min(20000, len(dr.y)), replace=False)
                    parts.append((dr.X[ii], dr.y[ii]))
                Xlab = np.concatenate([p[0] for p in parts])
                ylab = np.concatenate([p[1] for p in parts])
            # mixtures train on all cells of the sample by minibatch (init uses a subsample)
            r = fit_prior_mixture(ds.X, ds.X, prior, {**mcfg, "K": Ktot}, seed, Xlab=Xlab, ylab=ylab)
    except Exception as e:  # record failures honestly
        return dict(dataset=name, method=method, sample=sample, seed=seed, error=repr(e),
                    tb=traceback.format_exc()[-2000:])
    out = dict(dataset=name, method=method, sample=sample, seed=seed, runtime=time.time() - t0,
               maxrss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
               n_cells=int(len(ds.y)), identity=bool(r["identity"]))
    cl, y = r["cluster"], ds.y
    if dcfg["kind"] == "multi":
        hf = M.hungarian_f1(cl, y, K)
        out["hung_f1"] = hf.tolist()
        out["hung_macro_f1"] = float(hf.mean())
        out["ari"] = float(M.ari(cl, y))
        # ordering violations of the *matched* model components (data-space medians)
        m = y >= 0
        F, cl_ids, _, _ = M.f1_matrix(cl[m], y[m], K)
        from scipy.optimize import linear_sum_assignment
        rr, cc = linear_sum_assignment(-F)
        med = np.full((K, ds.X.shape[1]), np.nan)
        for a, b in zip(rr, cc):
            sel = cl == cl_ids[a]
            if sel.sum() >= 20:
                med[b] = np.median(ds.X[sel], 0)
        tot = v = 0
        for j, es in prior.edges.items():
            for a, b in es:
                if not (np.isnan(med[a, j]) or np.isnan(med[b, j])):
                    tot += 1
                    v += med[a, j] >= med[b, j]
        out["model_order_violation"] = v / max(tot, 1)
    out["rare_f1"] = float(M.rare_f1(cl, y)) if dcfg["kind"] == "rare" else None
    if r["identity"]:
        pp = r["pred_pop"]
        if dcfg["kind"] == "rare":   # 'other' cells are known negatives: score on ALL cells
            yy = np.where(y >= 0, y, K)
            idf = M.identity_f1(np.where(pp >= 0, pp, K), yy, K)
        else:
            idf = M.identity_f1(pp, y, K)
        out["id_f1"] = idf.tolist()
        out["id_macro_f1"] = float(idf.mean())
        lab = y >= 0
        if dcfg["kind"] == "rare":
            lab = np.ones(len(y), bool)
        pr = r["proba"][lab]
        pmax = pr.max(1)                       # confidence of the predicted component
        pred_lab = pp[lab]
        truth = y[lab] if dcfg["kind"] == "multi" else np.where(y[lab] >= 0, y[lab], -1)
        out["calibration"] = M.calibration(pmax, (pred_lab == truth).astype(float))
    if seed == 0 and sample in (dcfg["test"][0],):
        cache = ROOT / "results" / "cache"
        cache.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache / f"{name}_{method}_{sample}.npz", cluster=cl,
                            pred_pop=r["pred_pop"] if r["identity"] else np.array([]),
                            mu=r["mu"] if r["mu"] is not None else np.array([]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(CFG["datasets"]))
    ap.add_argument("--methods", nargs="+", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--seeds", nargs="+", type=int, default=CFG["seeds"])
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    jobs = []
    outdir = ROOT / "results" / "runs"
    outdir.mkdir(parents=True, exist_ok=True)
    for name in a.datasets:
        dcfg = CFG["datasets"][name]
        f = outdir / f"{name}_{a.split}.jsonl"
        done = set()
        if f.exists():
            for line in f.read_text().splitlines():
                j = json.loads(line)
                if "error" not in j:
                    done.add((j["method"], j["sample"], j["seed"]))
        for method in a.methods or CFG["methods"]:
            mc = CFG["methods"][method]
            if mc.get("n_ref") and (dcfg["kind"] != "multi" or len(dcfg["ref"]) < mc["n_ref"]):
                continue
            for s in dcfg[a.split]:
                for seed in a.seeds:
                    if (method, s, seed) not in done:
                        jobs.append((name, method, s, seed))
    print(f"{len(jobs)} jobs", flush=True)
    # heaviest first
    with Pool(a.workers, maxtasksperchild=8) as pool:
        for res in pool.imap_unordered(run_one, jobs):
            f = outdir / f"{res['dataset']}_{a.split}.jsonl"
            with open(f, "a") as fh:
                fh.write(json.dumps(res) + "\n")
            msg = res.get("error") or f"hung={res.get('hung_macro_f1')} id={res.get('id_macro_f1')} rare={res.get('rare_f1')} {res['runtime']:.0f}s"
            print(res["dataset"], res["method"], res["sample"], res["seed"], msg, flush=True)


if __name__ == "__main__":
    main()
