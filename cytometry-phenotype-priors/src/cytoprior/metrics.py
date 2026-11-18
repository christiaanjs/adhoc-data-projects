"""Evaluation metrics. Hungarian F1 follows Weber & Robinson 2016 (helper_match_evaluate_multiple.R):
unassigned cells removed, F1 matrix clusters x populations, one-to-one Hungarian matching maximising
F1, unmatched populations get F1 = 0, mean over populations. Rare datasets: F1 of the best cluster."""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score


def f1_matrix(clusters: np.ndarray, y: np.ndarray, n_pop: int):
    cl_ids = np.unique(clusters)
    ci = np.searchsorted(cl_ids, clusters)
    C = np.zeros((len(cl_ids), n_pop))
    np.add.at(C, (ci, y), 1)
    tp = C
    psize = C.sum(0, keepdims=True)
    csize = C.sum(1, keepdims=True)
    prec = tp / np.maximum(csize, 1)
    rec = tp / np.maximum(psize, 1)
    F = np.where(tp > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0.0)
    return F, cl_ids, prec, rec


def hungarian_f1(clusters, y, n_pop):
    m = y >= 0
    F, cl_ids, P, R = f1_matrix(clusters[m], y[m], n_pop)
    r, c = linear_sum_assignment(-F)
    f1 = np.zeros(n_pop)
    f1[c] = F[r, c]
    return f1


def identity_f1(pred, y, n_pop):
    """pred: predicted population index, -1 = unknown component. Labelled cells only."""
    m = y >= 0
    p, t = pred[m], y[m]
    f1 = np.zeros(n_pop)
    for k in range(n_pop):
        tp = np.sum((p == k) & (t == k))
        fp = np.sum((p == k) & (t != k))
        fn = np.sum((p != k) & (t == k))
        f1[k] = 2 * tp / max(2 * tp + fp + fn, 1)
    return f1


def rare_f1(clusters, y):
    """W&R single rare population: F1 of the best single cluster vs. rare label (y==0 rare)."""
    target = (y == 0).astype(int)
    best = 0.0
    for c in np.unique(clusters):
        sel = clusters == c
        tp = np.sum(sel & (target == 1))
        if tp == 0:
            continue
        p, r = tp / sel.sum(), tp / target.sum()
        best = max(best, 2 * p * r / (p + r))
    return best


def ari(clusters, y):
    m = y >= 0
    return adjusted_rand_score(y[m], clusters[m])


def calibration(prob_max, correct, n_bins=10):
    """Reliability curve + expected calibration error for the predicted label's probability."""
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(prob_max, bins) - 1, 0, n_bins - 1)
    conf, acc, cnt = np.zeros(n_bins), np.zeros(n_bins), np.zeros(n_bins)
    for b in range(n_bins):
        s = idx == b
        cnt[b] = s.sum()
        if cnt[b]:
            conf[b], acc[b] = prob_max[s].mean(), correct[s].mean()
    ece = float(np.sum(cnt / max(cnt.sum(), 1) * np.abs(conf - acc)))
    return dict(ece=ece, conf=conf.tolist(), acc=acc.tolist(), count=cnt.tolist())


def ordering_violations(X, y, prior, min_cells=20):
    """Empirical check on held-out data: for every DAG edge (a<b on marker j) compare the medians of
    the manually gated populations a and b. Returns violation rate and list of violated edges.
    This evaluates the *phenotype definitions*, independent of any model."""
    med = {}
    for k in range(prior.K):
        s = y == k
        if s.sum() >= min_cells:
            med[k] = np.median(X[s], 0)
    viol, total = [], 0
    for j, es in prior.edges.items():
        for a, b in es:
            if a in med and b in med:
                total += 1
                if med[a][j] >= med[b][j]:
                    viol.append((prior.markers[j], prior.populations[a], prior.populations[b],
                                 float(med[a][j]), float(med[b][j])))
    return dict(rate=len(viol) / max(total, 1), n_edges=total, violated=viol)


def model_ordering_violations(mu, prior):
    """Fraction of DAG edges violated by a fitted model's component means (unconstrained models)."""
    total, v = 0, 0
    for j, es in prior.edges.items():
        for a, b in es:
            total += 1
            v += mu[a, j] >= mu[b, j]
    return v / max(total, 1)
