"""Hyper-parameter selection on VALIDATION samples only (Samusik 04,05; Levine H1).
PYTHONPATH=src python scripts/tune_val.py Samusik_all"""
import itertools, json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import run_experiments as R

def job(args):
    import torch, warnings; torch.set_num_threads(1); warnings.filterwarnings("ignore")
    from cytoprior.models import fit_prior_mixture
    from cytoprior.metrics import hungarian_f1, identity_f1
    name, s, hp = args
    d, p = R._data(name); ds = d.by_sample(s)
    cfg = {**R.CFG["common"], "use_order": True, "use_rules": True, "dist": "t", **hp}
    r = fit_prior_mixture(ds.X, ds.X, p, cfg, 0)
    return dict(name=name, sample=s, **hp, hung=float(hungarian_f1(r["cluster"], ds.y, p.K).mean()),
                idf=float(identity_f1(r["pred_pop"], ds.y, p.K).mean()))

if __name__ == "__main__":
    name = sys.argv[1]
    grid = [dict(lam=l, n_unknown=u, tau=t) for l, u, t in itertools.product([1, 3, 10], [0, 6, 12], [0.3, 0.6])]
    jobs = [(name, s, hp) for hp in grid for s in R.CFG["datasets"][name]["val"]]
    out = ROOT / "results" / f"tune_val_{name}.jsonl"
    with Pool(4) as pool, open(out, "w") as fh:
        for r in pool.imap_unordered(job, jobs):
            fh.write(json.dumps(r) + "\n"); fh.flush(); print(r, flush=True)
