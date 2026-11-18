"""Prior-informed mixture models (PyTorch).

One class covers the ablation ladder L1, L3-L6 via switches:
  dist        : 'gauss' | 't'                      (component kernel, in latent space if flow)
  use_order   : ordering-constrained means from the per-marker DAGs (L3)
  use_rules   : rule-based gate p(k|x) from soft per-sample thresholds (L4)
  labels      : semi-supervised labelled reference cells (L5)
  flow        : 'none' | 'marginal' | 'marginal+coupling'   (L6)
  n_unknown   : extra components with free means for cells outside the known populations
L0 (sklearn GMM) and L2 (FlowSOM) are wrapped at the bottom.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .priors import PriorSpec, marker_thresholds


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def inv_softplus(x):
    x = torch.as_tensor(x, dtype=torch.float32)
    return torch.where(x > 20, x, torch.log(torch.expm1(torch.clamp(x, min=1e-4))))


# ------------------------------------------------------------------------------------- flows
class MarginalMonotone(nn.Module):
    """Per-marker strictly increasing map y = a x + b + sum_m w_m tanh((x - c_m)/s_m), a,w,s > 0.
    Preserves the order of every marker, so ordering constraints on the base means transfer to
    data space exactly. Acts as a learned per-sample monotone normalisation."""

    def __init__(self, D, M=6, x_ref=None):
        super().__init__()
        self.a = nn.Parameter(torch.full((D,), float(inv_softplus(1.0))))
        self.b = nn.Parameter(torch.zeros(D))
        c = torch.linspace(-1, 6, M)[None].repeat(D, 1) if x_ref is None else torch.as_tensor(x_ref)
        self.c = nn.Parameter(c.float())
        self.w = nn.Parameter(torch.full((D, M), -4.0))   # softplus(-4) ~ 0.018 -> near identity
        self.s = nn.Parameter(torch.zeros(D, M))

    def forward(self, x):
        a = F.softplus(self.a) + 1e-3
        w, s = F.softplus(self.w), F.softplus(self.s) + 0.05
        u = (x[..., None] - self.c) / s
        y = a * x + self.b + (w * torch.tanh(u)).sum(-1)
        dy = a + (w / s * (1 - torch.tanh(u) ** 2)).sum(-1)
        return y, torch.log(dy).sum(-1)


class AffineCoupling(nn.Module):
    def __init__(self, D, mask, hidden=64):
        super().__init__()
        self.register_buffer("mask", mask.float())
        self.net = nn.Sequential(nn.Linear(D, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(),
                                 nn.Linear(hidden, 2 * D))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        h = self.net(x * self.mask)
        s, t = h.chunk(2, -1)
        s = 0.5 * torch.tanh(s) * (1 - self.mask)
        t = t * (1 - self.mask)
        return x * torch.exp(s) + t, s.sum(-1)


# ------------------------------------------------------------------------------------- mixture
class PriorMixture(nn.Module):
    def __init__(self, D, K_known, n_unknown=0, dist="t", prior: PriorSpec | None = None,
                 use_order=False, use_rules=False, flow="none", n_coupling=2, gap_sigma=2.0,
                 margin=0.0, lam=1.0, tau=0.5, unk_score=math.log(0.5), seed=0):
        super().__init__()
        self.D, self.Kk, self.Ku = D, K_known, n_unknown
        self.K = K_known + n_unknown
        self.dist, self.prior = dist, prior
        self.use_order = use_order and prior is not None
        self.use_rules = use_rules and prior is not None
        self.gap_sigma, self.margin = gap_sigma, margin
        self.lam, self.tau, self.unk_score = lam, tau, unk_score
        K = self.K
        self.mu_free = nn.Parameter(torch.zeros(K, D))
        self.eta = nn.Parameter(torch.zeros(K, D))
        self.L_diag = nn.Parameter(torch.zeros(K, D))
        self.L_off = nn.Parameter(torch.zeros(K, D, D))
        self.dof_raw = nn.Parameter(torch.full((K,), float(inv_softplus(8.0))))
        self.logits = nn.Parameter(torch.zeros(K))
        lev = np.full((K, D), -1)
        if prior is not None:
            lev[:K_known] = prior.level
        self.register_buffer("level", torch.as_tensor(lev))
        self.register_buffer("tril_mask", torch.tril(torch.ones(D, D), -1))
        g = torch.Generator().manual_seed(seed)
        self.flows = nn.ModuleList()
        if flow.startswith("marginal"):
            self.flows.append(MarginalMonotone(D))
        if flow == "marginal+coupling":
            for i in range(n_coupling):
                mask = (torch.rand(D, generator=g) < 0.5)
                for m in (mask, ~mask):
                    self.flows.append(AffineCoupling(D, m))
        self.thresholds = None   # (D,2) per-sample, set in fit()

    # --- means with ordering reparameterisation -------------------------------------------
    def means(self):
        if not self.use_order:
            return self.mu_free
        mu = self.mu_free
        lev = self.level
        out = mu.clone()
        gaps = []
        neg_inf = torch.tensor(-1e9)
        for g in range(1, 4):            # level groups dim, pos, bright
            lower = (lev >= 0) & (lev < g)
            base = torch.where(lower, out, neg_inf).max(0).values     # (D,) max over lower-level comps
            has = lower.any(0)
            sel = (lev == g) & has[None]
            gap = F.softplus(self.eta) + self.margin
            out = torch.where(sel, base[None] + gap, out)
            gaps.append(gap[sel])
        self._gaps = torch.cat(gaps) if gaps else torch.zeros(0)
        return out

    def scale_tril(self):
        return torch.diag_embed(F.softplus(self.L_diag) + 1e-3) + self.L_off * self.tril_mask

    def transform(self, x):
        ld = torch.zeros(x.shape[0], device=x.device)
        for f in self.flows:
            x, l = f(x)
            ld = ld + l
        return x, ld

    def component_logpdf(self, z):
        mu, L = self.means(), self.scale_tril()
        diff = (z[None] - mu[:, None]).transpose(1, 2)              # (K,D,n)
        sol = torch.linalg.solve_triangular(L, diff, upper=False)  # (K,D,n)
        maha = (sol ** 2).sum(1)                                    # (K,n)
        logdet = torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1)[:, None]
        D = self.D
        if self.dist == "gauss":
            lp = -0.5 * maha - logdet - 0.5 * D * math.log(2 * math.pi)
        else:
            nu = (F.softplus(self.dof_raw) + 2.0)[:, None]
            lp = (torch.lgamma((nu + D) / 2) - torch.lgamma(nu / 2) - 0.5 * D * torch.log(nu * math.pi)
                  - logdet - 0.5 * (nu + D) * torch.log1p(maha / nu))
        return lp.T                                                 # (n,K)

    def rule_scores(self, x):
        """s_k(x) = mean over specified markers of log soft-membership in the stated level."""
        t, h = self.thresholds[:, 0], self.thresholds[:, 1]
        tau = self.tau
        ls = F.logsigmoid
        per = torch.stack([
            ls((t - x) / tau),                                    # neg
            ls((x - (t - tau)) / tau) + ls((h - x) / tau),        # dim
            ls((x - t) / tau),                                    # pos
            ls((x - h) / tau),                                    # bright
        ], -1)                                                    # (n,D,4)
        lev = self.level[: self.Kk]                               # (Kk,D)
        spec = (lev >= 0).float()
        idx = lev.clamp(min=0)
        # gather per (n,Kk,D)
        g = per[:, None, :, :].expand(-1, self.Kk, -1, -1).gather(3, idx[None, :, :, None].expand(x.shape[0], -1, -1, 1)).squeeze(-1)
        s_known = (g * spec).sum(-1) / spec.sum(-1).clamp(min=1)
        s_unk = torch.full((x.shape[0], self.Ku), self.unk_score, device=x.device)
        return torch.cat([s_known, s_unk], 1)

    def log_gate(self, x):
        logit = self.logits[None]
        if self.use_rules:
            logit = logit + self.lam * self.rule_scores(x)
        return F.log_softmax(logit, -1)

    def joint(self, x):
        z, ld = self.transform(x)
        return self.log_gate(x) + self.component_logpdf(z) + ld[:, None]    # (n,K)

    def log_prior(self):
        lp = torch.zeros(())
        if self.use_order and hasattr(self, "_gaps") and self._gaps.numel():
            lp = lp - (self._gaps ** 2).sum() / (2 * self.gap_sigma ** 2)
        return lp

    # --- fitting ---------------------------------------------------------------------------
    def init_params(self, X, mu0, cov0, w0):
        with torch.no_grad():
            self.mu_free.copy_(torch.as_tensor(mu0, dtype=torch.float32))
            Lc = torch.linalg.cholesky(torch.as_tensor(cov0, dtype=torch.float32))
            self.L_diag.copy_(inv_softplus(torch.diagonal(Lc, dim1=-2, dim2=-1)))
            self.L_off.copy_(Lc * self.tril_mask)
            self.logits.copy_(torch.log(torch.as_tensor(w0, dtype=torch.float32) + 1e-3))
        if self.use_order:   # fit eta so constrained means match the init means
            target = torch.as_tensor(mu0, dtype=torch.float32)
            opt = torch.optim.Adam([self.eta, self.mu_free], lr=0.05)
            for _ in range(300):
                opt.zero_grad()
                loss = ((self.means() - target) ** 2).sum()
                loss.backward()
                opt.step()

    def fit(self, X, Xlab=None, ylab=None, lab_weight=1.0, steps=600, lr=0.02, batch=None,
            verbose=False, seed=0):
        X = torch.as_tensor(X, dtype=torch.float32)
        if Xlab is not None:
            Xlab = torch.as_tensor(Xlab, dtype=torch.float32)
            ylab = torch.as_tensor(ylab)
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
        g = torch.Generator().manual_seed(seed)
        hist = []
        for it in range(steps):
            xb = X if batch is None or batch >= len(X) else X[torch.randint(len(X), (batch,), generator=g)]
            opt.zero_grad()
            ll = torch.logsumexp(self.joint(xb), 1).mean()
            loss = -ll - self.log_prior() / len(X)
            if Xlab is not None:
                jl = self.joint(Xlab)
                known = ylab >= 0
                lab_ll = jl[known].gather(1, ylab[known][:, None]).squeeze(1).sum()
                if self.Ku > 0 and (~known).any():
                    lab_ll = lab_ll + torch.logsumexp(jl[~known][:, self.Kk:], 1).sum()
                loss = loss - lab_weight * lab_ll / len(Xlab)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 50.0)
            opt.step()
            sched.step()
            hist.append(float(loss))
            if verbose and it % 100 == 0:
                print(it, hist[-1])
        return hist

    @torch.no_grad()
    def predict_proba(self, X, bs=50000):
        out = []
        for i in range(0, len(X), bs):
            xb = torch.as_tensor(X[i:i + bs], dtype=torch.float32)
            out.append(torch.softmax(self.joint(xb), 1).numpy())
        return np.concatenate(out)


# ------------------------------------------------------------------------------------- init
def kmeans_init(X, K, seed, reg=0.05):
    from sklearn.cluster import KMeans
    km = KMeans(K, n_init=1, random_state=seed, max_iter=100).fit(X)
    lab = km.labels_
    D = X.shape[1]
    cov = np.stack([np.cov(X[lab == k].T) + reg * np.eye(D) if (lab == k).sum() > D else np.eye(D) * 0.5
                    for k in range(K)])
    w = np.bincount(lab, minlength=K) / len(lab)
    return km.cluster_centers_, cov, w


LEVEL_Q = np.array([0.15, 0.55, 0.8, 0.95])   # target marker-quantile position per level


def prior_init(X, prior: PriorSpec, n_unknown, seed, reg=0.05, over=3):
    """Rank-based init: over-cluster with k-means, describe each cluster centroid by its quantile
    position u_cj = F_j(centroid_j) in the sample's marker distribution, and Hungarian-match clusters
    to known populations by sum_j |u_cj - LEVEL_Q[level_kj]| over specified markers. Uses only rank
    (ordering-type) information, no absolute thresholds. Remaining clusters (largest first) seed the
    unknown components."""
    from scipy.optimize import linear_sum_assignment
    Kk = prior.K
    C = max(over * (Kk + n_unknown), Kk + n_unknown + 5)
    cent, cov, w = kmeans_init(X, C, seed, reg)
    Xs = np.sort(X, 0)
    u = np.stack([np.searchsorted(Xs[:, j], cent[:, j]) / len(Xs) for j in range(X.shape[1])], 1)
    spec = prior.level >= 0
    tgt = LEVEL_Q[np.clip(prior.level, 0, 3)]
    cost = np.stack([(np.abs(u[c][None] - tgt) * spec).sum(1) / spec.sum(1).clip(1) for c in range(C)])  # (C,Kk)
    r, c = linear_sum_assignment(cost)
    order = np.argsort(c)
    known_cl = r[order]
    rest = np.setdiff1d(np.arange(C), known_cl)
    rest = rest[np.argsort(-w[rest])][:n_unknown]
    idx = np.concatenate([known_cl, rest]).astype(int)
    return cent[idx], cov[idx], w[idx] / w[idx].sum()


# ------------------------------------------------------------------------------------- runners
def fit_prior_mixture(X_fit, X_all, prior, cfg, seed, Xlab=None, ylab=None):
    """Returns dict(proba, pred_cluster, pred_pop, model, runtime)."""
    set_seed(seed)
    t0 = time.time()
    Kk = prior.K if prior is not None else 0
    Ku = cfg.get("n_unknown", 0)
    use_prior_identity = cfg.get("use_order") or cfg.get("use_rules") or cfg.get("prior_init", False) or Xlab is not None
    D = X_fit.shape[1]
    if prior is None or not use_prior_identity:
        K_tot = cfg.get("K", Kk + Ku)
        m = PriorMixture(D, K_tot, 0, dist=cfg.get("dist", "t"), prior=None, flow=cfg.get("flow", "none"),
                         seed=seed)
        mu0, cov0, w0 = kmeans_init(X_fit, K_tot, seed)
        identity = False
    else:
        m = PriorMixture(D, Kk, Ku, dist=cfg.get("dist", "t"), prior=prior,
                         use_order=cfg.get("use_order", False), use_rules=cfg.get("use_rules", False),
                         flow=cfg.get("flow", "none"), gap_sigma=cfg.get("gap_sigma", 2.0),
                         lam=cfg.get("lam", 1.0), tau=cfg.get("tau", 0.5), seed=seed)
        mu0, cov0, w0 = prior_init(X_fit, prior, Ku, seed)
        identity = True
    m.thresholds = torch.as_tensor(marker_thresholds(X_fit, seed=seed))
    m.init_params(X_fit, mu0, cov0, w0)
    hist = m.fit(X_fit, Xlab, ylab, steps=cfg.get("steps", 500), lr=cfg.get("lr", 0.02),
                 batch=cfg.get("batch"), seed=seed, lab_weight=cfg.get("lab_weight", 1.0))
    proba = m.predict_proba(X_all)
    cl = proba.argmax(1)
    pred_pop = np.where(cl < Kk, cl, -1) if identity else None
    with torch.no_grad():
        mu = m.means().numpy()
    return dict(proba=proba, cluster=cl, pred_pop=pred_pop, identity=identity, mu=mu,
                runtime=time.time() - t0, loss=hist[-1], model=m)


def fit_gmm(X_fit, X_all, K, seed):
    from sklearn.mixture import GaussianMixture
    t0 = time.time()
    g = GaussianMixture(K, covariance_type="full", reg_covar=1e-3, max_iter=200, random_state=seed,
                        init_params="k-means++").fit(X_fit)
    proba = g.predict_proba(X_all)
    return dict(proba=proba, cluster=proba.argmax(1), pred_pop=None, identity=False, mu=g.means_,
                runtime=time.time() - t0)


def fit_flowsom(X_all, k, seed, xdim=10, ydim=10):
    import anndata as ad
    import flowsom as fs
    import pandas as pd
    t0 = time.time()
    names = [f"m{j}" for j in range(X_all.shape[1])]
    var = pd.DataFrame(dict(channel=names, marker=names), index=names)
    a = ad.AnnData(np.asarray(X_all, dtype=np.float32), var=var)
    # flowsom 0.2.2 writes into a read-only array under pandas>=3 copy-on-write; equivalent no-op here
    a.var["pretty_colnames"] = names
    a.var["markers"] = names
    a.var["channels"] = names
    fs.FlowSOM.clean_anndata = lambda self: self.mudata
    f = fs.FlowSOM(a, cols_to_use=list(range(X_all.shape[1])), xdim=xdim, ydim=ydim, n_clusters=k, seed=seed)
    cl = np.asarray(f.get_cell_data().obs["metaclustering"]).astype(int)
    return dict(proba=None, cluster=cl, pred_pop=None, identity=False, mu=None, runtime=time.time() - t0)
