"""
Individualised treatment benefit: combine the interpretable *baseline* nonunion
risk model (risk after NONoperative treatment) with the *relative* effect of
surgery from the hierarchical meta-analysis, to get a per-patient absolute risk
reduction (ARR) and number-needed-to-treat (NNT).

    risk_nonop = baseline model prediction  (with coefficient uncertainty)
    risk_op    = risk_nonop x RR            (RR from the meta-analysis posterior)
    ARR        = risk_nonop - risk_op
    NNT        = 1 / ARR

Both sources of uncertainty are propagated by Monte Carlo:
  * logistic coefficients ~ MVN(mean, cov)  (from statsmodels fit)
  * RR ~ location-specific posterior draws   (from PyMC/NUTS)
These are independent (different data), so we pair independent draws.

Distal fractures are handled with a documented baseline-rate shift (their
nonoperative nonunion risk is far higher than midshaft); the predictor effects
are assumed to carry over, and the RR uses the distal posterior. See the note on
`DISTAL_LOGODDS_SHIFT`.

Requires artifacts produced by `run_all.py`:
  outputs/risk_model_logistic.npz   (mean + cov of the logistic coefficients)
  outputs/rr_posterior.npz          (RR posterior draws per location)
"""
from __future__ import annotations

import os
from dataclasses import dataclass, asdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "outputs")

LOGISTIC_NPZ = os.path.join(OUT, "risk_model_logistic.npz")
RR_NPZ = os.path.join(OUT, "rr_posterior.npz")

# Baseline nonoperative nonunion risk is ~15% for displaced midshaft and
# ~33% for displaced distal (Neer II) fractures. On the log-odds scale that is a
# shift of log(odds_distal / odds_midshaft) = log((.33/.67)/(.15/.85)) ~= 1.03.
# We apply this as an intercept offset for distal, keeping predictor slopes.
DISTAL_LOGODDS_SHIFT = np.log((0.33 / 0.67) / (0.15 / 0.85))

N_DRAWS = 12000
FEATURE_ORDER = ["const", "age_decades_c", "female", "smoking",
                 "complete_displacement", "comminution", "shortening_gt2cm"]


@dataclass
class Patient:
    age: float = 40.0
    female: int = 0
    smoking: int = 0
    complete_displacement: int = 0
    comminution: int = 0
    shortening_gt2cm: int = 0
    location: str = "midshaft"     # "midshaft" or "distal"

    def design_vector(self):
        return np.array([
            1.0,
            (self.age - 40) / 10.0,
            float(self.female),
            float(self.smoking),
            float(self.complete_displacement),
            float(self.comminution),
            float(self.shortening_gt2cm),
        ])


def _load_logistic():
    if not os.path.exists(LOGISTIC_NPZ):
        raise FileNotFoundError(
            f"{LOGISTIC_NPZ} not found - run `python run_all.py` first.")
    d = np.load(LOGISTIC_NPZ, allow_pickle=True)
    terms = list(d["terms"])
    mean, cov = d["mean"], d["cov"]
    # Reorder to FEATURE_ORDER (defensive; statsmodels preserves insertion order)
    idx = [terms.index(t) for t in FEATURE_ORDER]
    return mean[idx], cov[np.ix_(idx, idx)]


def _load_rr(location):
    if not os.path.exists(RR_NPZ):
        raise FileNotFoundError(
            f"{RR_NPZ} not found - run `python run_all.py` first.")
    d = np.load(RR_NPZ)
    key = "distal" if location == "distal" else "midshaft"
    return d[key]


def _summ(draws):
    return dict(median=float(np.median(draws)),
                lo=float(np.percentile(draws, 2.5)),
                hi=float(np.percentile(draws, 97.5)))


def treatment_benefit(patient: Patient, n_draws: int = N_DRAWS, seed: int = 0):
    """Return baseline risk, operative risk, ARR and NNT with 95% intervals."""
    rng = np.random.default_rng(seed)
    mean, cov = _load_logistic()
    beta = rng.multivariate_normal(mean, cov, size=n_draws)   # (n, 7)

    x = patient.design_vector()
    lin = beta @ x
    if patient.location == "distal":
        lin = lin + DISTAL_LOGODDS_SHIFT
    risk_nonop = 1.0 / (1.0 + np.exp(-lin))

    rr = _load_rr(patient.location)
    rr = rng.choice(rr, size=n_draws, replace=True)           # pair independently

    risk_op = np.clip(risk_nonop * rr, 0.0, 1.0)
    arr = risk_nonop - risk_op
    # NNT only defined where surgery helps (arr > 0); it always is here (RR<1).
    with np.errstate(divide="ignore"):
        nnt = np.where(arr > 1e-9, 1.0 / arr, np.inf)

    return dict(
        patient=asdict(patient),
        risk_nonoperative=_summ(risk_nonop),
        risk_operative=_summ(risk_op),
        absolute_risk_reduction=_summ(arr),
        relative_risk=_summ(rr),
        nnt=_summ(nnt[np.isfinite(nnt)]),
    )


UNIFIED_NPZ = os.path.join(OUT, "unified_posterior.npz")
# Predictor order in the unified model (age is per-decade, centred at 40).
UNIFIED_PRED_ORDER = ["age", "female", "smoking", "complete_displacement",
                      "comminution", "shortening_gt2cm"]


def treatment_benefit_unified(patient: Patient):
    """Per-patient risk/ARR/NNT read straight off the single joint posterior.

    Uses outputs/unified_posterior.npz (from src/unified_model.py): baseline,
    treatment effect and predictor slopes all come from ONE fit, so no
    independent-source pairing is needed - every draw is internally consistent.
    """
    if not os.path.exists(UNIFIED_NPZ):
        raise FileNotFoundError(
            f"{UNIFIED_NPZ} not found - run `python run_all.py` first.")
    d = np.load(UNIFIED_NPZ, allow_pickle=True)
    gamma = d["gamma"]                              # (n_draws, K)
    base_ref = d["base_ref"]                        # (n_draws, 2)
    mu_delta = d["mu_delta"]                        # (n_draws, 2)
    loc_levels = list(d["loc_levels"])
    li = loc_levels.index("distal" if patient.location == "distal" else "midshaft")

    x = np.array([
        (patient.age - 40) / 10.0, float(patient.female), float(patient.smoking),
        float(patient.complete_displacement), float(patient.comminution),
        float(patient.shortening_gt2cm)])
    lin_no = base_ref[:, li] + gamma @ x
    lin_op = lin_no + mu_delta[:, li]
    risk_nonop = 1.0 / (1.0 + np.exp(-lin_no))
    risk_op = 1.0 / (1.0 + np.exp(-lin_op))
    arr = risk_nonop - risk_op
    with np.errstate(divide="ignore"):
        nnt = np.where(arr > 1e-9, 1.0 / arr, np.inf)
    return dict(
        patient=asdict(patient),
        risk_nonoperative=_summ(risk_nonop),
        risk_operative=_summ(risk_op),
        absolute_risk_reduction=_summ(arr),
        relative_risk=_summ(np.exp(mu_delta[:, li])),   # OR (approx RR at low risk)
        nnt=_summ(nnt[np.isfinite(nnt)]),
    )


def benefit_band(nnt_median: float) -> str:
    """Illustrative (NOT prescriptive) interpretation of the NNT."""
    if nnt_median <= 5:
        return "large absolute benefit (low NNT)"
    if nnt_median <= 15:
        return "moderate absolute benefit"
    return "small absolute benefit (high NNT)"


if __name__ == "__main__":
    for p in [Patient(age=25),
              Patient(age=40, complete_displacement=1),
              Patient(age=60, female=1, smoking=1, complete_displacement=1,
                      comminution=1, shortening_gt2cm=1),
              Patient(age=55, complete_displacement=1, location="distal")]:
        r = treatment_benefit(p)
        print(p, "->",
              f"nonop {r['risk_nonoperative']['median']*100:.1f}% | "
              f"op {r['risk_operative']['median']*100:.1f}% | "
              f"NNT {r['nnt']['median']:.0f}")
