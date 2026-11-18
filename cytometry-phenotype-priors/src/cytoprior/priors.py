"""Phenotype prior specification: marker levels, per-marker ordering DAGs, hierarchy, rules."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

LEVELS = ["neg", "dim", "pos", "bright"]
UNSPEC = -1
PRIOR_DIR = Path(__file__).resolve().parents[2] / "phenotype_priors"


@dataclass
class PriorSpec:
    dataset: str
    populations: list[str]          # known populations, order = component order
    markers: list[str]
    level: np.ndarray               # (K, D) int in {-1,0,1,2,3}
    inferred: np.ndarray            # (K,) bool
    citations: list[list[str]]
    parents: list[str | None]
    edges: dict[int, list[tuple[int, int]]]   # marker j -> [(lower k, higher k)] (transitive reduction)
    nodes: dict | None = None       # name -> (parent, {j: level}) own markers of every hierarchy node / population

    @property
    def K(self):
        return len(self.populations)

    def to_dag_yaml(self) -> dict:
        out = {}
        for j, es in self.edges.items():
            if es:
                out[self.markers[j]] = [[self.populations[a], self.populations[b]] for a, b in es]
        return out


def _resolve(node: str, table: dict, hier: dict) -> dict:
    """Marker levels of a node including everything inherited from its ancestors."""
    ent = table.get(node) or hier[node]
    parent = ent.get("parent")
    base = _resolve(parent, table, hier) if parent else {}
    base = dict(base)
    base.update(ent.get("markers", {}))
    return base


def load_prior(dataset: str, markers: list[str], label_names: list[str]) -> PriorSpec:
    cfg = yaml.safe_load((PRIOR_DIR / f"{dataset}.yaml").read_text())
    hier = cfg.get("hierarchy", {}) or {}
    pops = cfg["populations"]
    missing = set(pops) - set(label_names)
    assert not missing, f"prior populations not in labels: {missing}"
    order = [p for p in label_names if p in pops]      # same index as dataset labels
    assert order == list(label_names), "every labelled population needs a prior entry"
    K, D = len(order), len(markers)
    lev = np.full((K, D), UNSPEC, dtype=np.int64)
    inferred = np.zeros(K, bool)
    cites, parents = [], []
    for k, p in enumerate(order):
        ent = pops[p]
        mk = _resolve(p, pops, hier)
        for m, l in mk.items():
            if m not in markers:
                raise KeyError(f"{dataset}:{p}: marker {m} not in panel")
            lev[k, markers.index(m)] = LEVELS.index(l)
        # inferred if this node or any ancestor is inferred
        inf, node = bool(ent.get("inferred", False)), ent.get("parent")
        while node:
            inf |= bool(hier.get(node, pops.get(node, {})).get("inferred", False))
            node = hier.get(node, pops.get(node, {})).get("parent")
        inferred[k] = inf
        cites.append(list(ent.get("citation", [])))
        parents.append(ent.get("parent"))
    edges: dict[int, list[tuple[int, int]]] = {}
    for j in range(D):
        es = []
        present = [g for g in range(len(LEVELS)) if (lev[:, j] == g).any()]
        for g_lo, g_hi in zip(present[:-1], present[1:]):
            lo = np.where(lev[:, j] == g_lo)[0]
            hi = np.where(lev[:, j] == g_hi)[0]
            es += [(int(a), int(b)) for a in lo for b in hi]
        edges[j] = es
    for o in cfg.get("orderings", []) or []:
        j = markers.index(o["marker"])
        edges[j].append((order.index(o["lower"]), order.index(o["higher"])))
    nodes = {}
    for name, ent in {**hier, **pops}.items():
        nodes[name] = (ent.get("parent"), {markers.index(m): LEVELS.index(l) for m, l in ent.get("markers", {}).items()})
    return PriorSpec(dataset, order, list(markers), lev, inferred, cites, parents, edges, nodes)


# ---------------------------------------------------------------------------------------------
# Per-sample soft thresholds for rules. Data-driven per sample and per marker (no absolute
# thresholds are taken from the literature): a 2-component 1-D Gaussian mixture on the marker;
# t = decision boundary between the low and high component; h = boundary inside the high part
# (pos/bright split, from a 2-component fit on cells above t).
# ---------------------------------------------------------------------------------------------
def _gmm1d_boundary(x: np.ndarray, iters: int = 60) -> float:
    x = np.sort(x)
    m = np.array([np.quantile(x, 0.2), np.quantile(x, 0.9)])
    s = np.array([x.std(), x.std()]) / 2 + 1e-3
    w = np.array([0.5, 0.5])
    for _ in range(iters):
        lp = -0.5 * ((x[:, None] - m) / s) ** 2 - np.log(s) + np.log(w)
        lp -= lp.max(1, keepdims=True)
        r = np.exp(lp)
        r /= r.sum(1, keepdims=True)
        nk = r.sum(0) + 1e-9
        w = nk / nk.sum()
        m = (r * x[:, None]).sum(0) / nk
        s = np.sqrt((r * (x[:, None] - m) ** 2).sum(0) / nk) + 0.05
    lo, hi = np.argsort(m)
    grid = np.linspace(m[lo], m[hi], 200)
    dl = -0.5 * ((grid - m[lo]) / s[lo]) ** 2 - np.log(s[lo]) + np.log(w[lo])
    dh = -0.5 * ((grid - m[hi]) / s[hi]) ** 2 - np.log(s[hi]) + np.log(w[hi])
    idx = np.where(dh > dl)[0]
    return float(grid[idx[0]] if len(idx) else (m[lo] + m[hi]) / 2)


def marker_thresholds(X: np.ndarray, max_n: int = 20000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    Xs = X[rng.choice(len(X), min(max_n, len(X)), replace=False)]
    th = np.zeros((X.shape[1], 2), np.float32)
    for j in range(X.shape[1]):
        t = _gmm1d_boundary(Xs[:, j])
        above = Xs[Xs[:, j] > t, j]
        h = _gmm1d_boundary(above) if len(above) > 200 else t + 1.0
        th[j] = (t, max(h, t + 0.25))
    return th


def hierarchical_thresholds(X: np.ndarray, prior: PriorSpec, max_n: int = 30000, seed: int = 0,
                            min_cells: int = 200) -> np.ndarray:
    """Per-population thresholds (K, D, 2). A node's OWN markers are thresholded on the cells that pass
    all of its ancestors' gates (hard rules with the ancestors' thresholds), mirroring how manual gates
    are drawn inside the parent gate. Falls back to the parent's / global threshold if < min_cells."""
    rng = np.random.default_rng(seed)
    Xs = X[rng.choice(len(X), min(max_n, len(X)), replace=False)]
    glob = marker_thresholds(Xs, max_n=max_n, seed=seed)
    memo: dict = {}

    def passes(x, j, l, th):
        t, h = th
        return [x < t, (x > t - 0.5) & (x < h), x > t, x > h][l]

    def node(name):
        if name in memo:
            return memo[name]
        parent, own = prior.nodes[name]
        if parent is None:
            th_par, mask = glob.copy(), np.ones(len(Xs), bool)
        else:
            th_par, mask = node(parent)
        th = th_par.copy()
        cells = Xs[mask]
        for j in own:
            if len(cells) >= min_cells:
                t = _gmm1d_boundary(cells[:, j])
                above = cells[cells[:, j] > t, j]
                h = _gmm1d_boundary(above) if len(above) > min_cells else t + 1.0
                th[j] = (t, max(h, t + 0.25))
        m2 = mask.copy()
        for j, l in own.items():
            m2 &= passes(Xs[:, j], j, l, th[j])
        memo[name] = (th, m2)
        return memo[name]

    return np.stack([node(p)[0] for p in prior.populations]).astype(np.float32)
