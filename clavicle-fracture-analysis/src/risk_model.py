"""
Interpretable predictive model for NONUNION after nonoperative treatment of a
displaced midshaft clavicle fracture.

Design goals (per the brief): interpretable, high predictive power.

We build three transparent models and compare them:
  1. Points-based risk score derived DIRECTLY from published odds ratios
     (no fitting; a paper-and-pencil calculator).
  2. Logistic regression fitted to a cohort.
  3. Shallow decision tree (depth 3) fitted to a cohort — human-readable rules.

Because clavicle IPD is not public, the cohort is SYNTHESISED from the same
published odds ratios (see data/REFERENCES.md). The fitted logistic model
therefore recovers those effect sizes, which validates the pipeline and gives
an estimate of achievable discrimination. To run on real data, replace
`build_cohort()` with a loader for a real CSV that has the same columns.

Outputs: coefficient table, points table, ROC curve, calibration plot,
model-comparison table, and a reusable `predict_nonunion_risk()` calculator.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, brier_score_loss, roc_curve

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PRED = os.path.join(ROOT, "data", "nonunion_predictors.csv")
OUT = os.path.join(ROOT, "outputs")
os.makedirs(OUT, exist_ok=True)

RNG = np.random.default_rng(20240702)
FEATURES = ["age_decades_c", "female", "smoking",
            "complete_displacement", "comminution", "shortening_gt2cm"]
BASELINE_NONUNION = 0.15  # displaced midshaft, nonoperative (literature anchor)
N_COHORT = 6000


# --------------------------------------------------------------------------- #
# Published effect sizes -> log-odds
# --------------------------------------------------------------------------- #
def load_betas():
    df = pd.read_csv(PRED)
    beta = {}
    for _, r in df.iterrows():
        beta[r["predictor"]] = np.log(r["odds_ratio"])
    return df, beta


# --------------------------------------------------------------------------- #
# Synthetic cohort generated from the published log-odds
# --------------------------------------------------------------------------- #
def build_cohort(n=N_COHORT):
    # Realistic marginal distributions for displaced midshaft fractures.
    age = np.clip(RNG.normal(40, 15, n), 16, 85)
    female = RNG.binomial(1, 0.30, n)
    smoking = RNG.binomial(1, 0.25, n)
    complete_displacement = RNG.binomial(1, 0.55, n)
    # comminution and shortening co-occur with displacement (mild correlation)
    p_com = 0.25 + 0.15 * complete_displacement
    comminution = RNG.binomial(1, p_com)
    p_short = 0.20 + 0.20 * complete_displacement
    shortening_gt2cm = RNG.binomial(1, p_short)

    df = pd.DataFrame({
        "age": age,
        "female": female,
        "smoking": smoking,
        "complete_displacement": complete_displacement,
        "comminution": comminution,
        "shortening_gt2cm": shortening_gt2cm,
    })
    df["age_decades_c"] = (df["age"] - 40) / 10.0  # centred, per-decade

    _, beta = load_betas()
    lin = (beta["age"] * df["age_decades_c"]
           + beta["female"] * df["female"]
           + beta["smoking"] * df["smoking"]
           + beta["complete_displacement"] * df["complete_displacement"]
           + beta["comminution"] * df["comminution"]
           + beta["shortening_gt2cm"] * df["shortening_gt2cm"])

    # Solve intercept so that mean P(nonunion) == BASELINE_NONUNION.
    def mean_p(b0):
        return np.mean(1 / (1 + np.exp(-(b0 + lin))))
    lo, hi = -12, 6
    for _ in range(80):
        mid = (lo + hi) / 2
        if mean_p(mid) < BASELINE_NONUNION:
            lo = mid
        else:
            hi = mid
    intercept = (lo + hi) / 2

    p = 1 / (1 + np.exp(-(intercept + lin)))
    df["nonunion"] = RNG.binomial(1, p)
    return df, intercept


# --------------------------------------------------------------------------- #
# 1. Points score from published ORs (reference-unit method, a la Framingham)
# --------------------------------------------------------------------------- #
def build_points_table(pred_df):
    """Convert each risk factor's log-OR into integer points.

    B = points per unit of log-odds. We pick B so the smallest binary factor is
    ~1 point, giving a compact, clinically usable score.
    """
    rows = []
    logor = {r["predictor"]: np.log(r["odds_ratio"]) for _, r in pred_df.iterrows()}
    # Sullivan/Framingham points method: fix the reference unit B to the
    # per-decade age effect, so age scores 1 point per decade over 40 and every
    # other factor is expressed relative to that. Keeps the age signal and
    # yields an interpretable integer score.
    B = logor["age"]  # 1 point == one decade of age
    for _, r in pred_df.iterrows():
        name = r["predictor"]
        if name == "age":
            # per decade above 40
            pts = round(np.log(r["odds_ratio"]) / B)
            rows.append((name, "per decade over 40", np.log(r["odds_ratio"]),
                         round(np.log(r["odds_ratio"]) / B)))
        else:
            rows.append((name, f"present (ref: {r['reference_level']})",
                         np.log(r["odds_ratio"]),
                         round(np.log(r["odds_ratio"]) / B)))
    tbl = pd.DataFrame(rows, columns=["factor", "condition", "log_or", "points"])
    return tbl, B


def points_for_patient(patient, points_tbl):
    pt_map = {r["factor"]: r["points"] for _, r in points_tbl.iterrows()}
    score = 0
    if patient.get("age", 40) > 40:
        score += pt_map["age"] * int((patient["age"] - 40) // 10)
    for f in ["female", "smoking", "complete_displacement",
              "comminution", "shortening_gt2cm"]:
        if patient.get(f, 0):
            score += pt_map[f]
    return int(score)


# --------------------------------------------------------------------------- #
# Fit + evaluate
# --------------------------------------------------------------------------- #
def fit_logistic_statsmodels(df):
    X = sm.add_constant(df[FEATURES])
    model = sm.Logit(df["nonunion"], X).fit(disp=0)
    return model


def coefficient_table(model, pred_df):
    params = model.params
    conf = model.conf_int()
    published = {r["predictor"]: r["odds_ratio"] for _, r in pred_df.iterrows()}
    name_map = {
        "age_decades_c": "age", "female": "female", "smoking": "smoking",
        "complete_displacement": "complete_displacement",
        "comminution": "comminution", "shortening_gt2cm": "shortening_gt2cm",
    }
    rows = []
    for term in params.index:
        or_hat = np.exp(params[term])
        lo, hi = np.exp(conf.loc[term, 0]), np.exp(conf.loc[term, 1])
        pub = published.get(name_map.get(term, ""), np.nan)
        rows.append((term, params[term], or_hat, lo, hi,
                     model.pvalues[term], pub))
    return pd.DataFrame(rows, columns=[
        "term", "coef", "OR_fitted", "OR_ci_low", "OR_ci_high",
        "p_value", "OR_published"])


def cv_metrics(df):
    X = df[FEATURES].to_numpy()
    y = df["nonunion"].to_numpy()
    cv = StratifiedKFold(5, shuffle=True, random_state=0)

    out = {}
    # Logistic regression
    lr = LogisticRegression(max_iter=2000)
    p_lr = cross_val_predict(lr, X, y, cv=cv, method="predict_proba")[:, 1]
    out["Logistic regression"] = (y, p_lr)

    # Shallow decision tree
    tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=100,
                                  random_state=0)
    p_tree = cross_val_predict(tree, X, y, cv=cv, method="predict_proba")[:, 1]
    out["Decision tree (depth 3)"] = (y, p_tree)

    metrics = []
    for name, (yy, pp) in out.items():
        metrics.append((name, roc_auc_score(yy, pp), brier_score_loss(yy, pp)))
    return pd.DataFrame(metrics, columns=["model", "AUC_cv", "Brier_cv"]), out


def plot_roc(curves, path):
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    colors = {"Logistic regression": "#2f6f8f",
              "Decision tree (depth 3)": "#b4451f"}
    for name, (y, p) in curves.items():
        fpr, tpr, _ = roc_curve(y, p)
        auc = roc_auc_score(y, p)
        ax.plot(fpr, tpr, lw=2, color=colors.get(name, "#555"),
                label=f"{name} (AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", color="#999", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Discrimination for nonunion (5-fold CV)", loc="left")
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_calibration(y, p, path, bins=10):
    order = np.argsort(p)
    y, p = y[order], p[order]
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    edges[-1] += 1e-9
    idx = np.digitize(p, edges[1:-1])
    xs, ys, ns = [], [], []
    for b in range(bins):
        m = idx == b
        if m.sum() > 0:
            xs.append(p[m].mean())
            ys.append(y[m].mean())
            ns.append(m.sum())
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    ax.plot([0, 1], [0, 1], ls="--", color="#999", lw=1, label="Perfect")
    ax.plot(xs, ys, "-o", color="#2f6f8f", lw=2, label="Logistic model")
    ax.set_xlabel("Predicted nonunion risk")
    ax.set_ylabel("Observed nonunion frequency")
    ax.set_title("Calibration (5-fold CV, decile bins)", loc="left")
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    lim = max(max(xs), max(ys)) * 1.1
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def score_to_risk(df, points_tbl):
    """Empirical risk per points score in the cohort -> lookup curve."""
    scores = df.apply(lambda r: points_for_patient(r, points_tbl), axis=1)
    tmp = pd.DataFrame({"score": scores, "nonunion": df["nonunion"]})
    curve = tmp.groupby("score")["nonunion"].agg(["mean", "count"]).reset_index()
    auc = roc_auc_score(df["nonunion"], scores)
    return curve, scores, auc


# --------------------------------------------------------------------------- #
# Reusable calculator
# --------------------------------------------------------------------------- #
def make_calculator(model):
    coef = model.params

    def predict_nonunion_risk(age=40, female=0, smoking=0,
                              complete_displacement=0, comminution=0,
                              shortening_gt2cm=0):
        x = {
            "const": 1.0,
            "age_decades_c": (age - 40) / 10.0,
            "female": female,
            "smoking": smoking,
            "complete_displacement": complete_displacement,
            "comminution": comminution,
            "shortening_gt2cm": shortening_gt2cm,
        }
        lin = sum(coef[k] * x[k] for k in coef.index)
        return float(1 / (1 + np.exp(-lin)))

    return predict_nonunion_risk


def run():
    pred_df, _ = load_betas()
    df, intercept = build_cohort()

    # 1. Points score
    points_tbl, B = build_points_table(pred_df)
    points_tbl.to_csv(os.path.join(OUT, "points_table.csv"), index=False)
    curve, scores, auc_points = score_to_risk(df, points_tbl)
    curve.to_csv(os.path.join(OUT, "score_risk_lookup.csv"), index=False)

    # 2. Logistic regression (statsmodels for inference)
    model = fit_logistic_statsmodels(df)
    coefs = coefficient_table(model, pred_df)
    coefs.to_csv(os.path.join(OUT, "logistic_coefficients.csv"), index=False)

    # Persist mean + covariance so the CLI can reconstruct the model (and
    # propagate coefficient uncertainty) without refitting.
    terms = list(model.params.index)
    np.savez(os.path.join(OUT, "risk_model_logistic.npz"),
             terms=np.array(terms, dtype=object),
             mean=model.params.to_numpy(float),
             cov=model.cov_params().to_numpy(float))

    # 3. Cross-validated comparison
    metrics, curves = cv_metrics(df)
    metrics = pd.concat([
        metrics,
        pd.DataFrame([("Points score (a priori, unfit)", auc_points, np.nan)],
                     columns=metrics.columns),
    ], ignore_index=True)
    metrics.to_csv(os.path.join(OUT, "model_comparison.csv"), index=False)

    plot_roc(curves, os.path.join(OUT, "roc.png"))
    y = curves["Logistic regression"][0]
    p = curves["Logistic regression"][1]
    plot_calibration(y, p, os.path.join(OUT, "calibration.png"))

    # Decision-tree rules (text)
    tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=100,
                                  random_state=0).fit(df[FEATURES], df["nonunion"])
    rules = export_text(tree, feature_names=FEATURES)
    with open(os.path.join(OUT, "decision_tree_rules.txt"), "w") as f:
        f.write(rules)

    # Example patients via calculator
    calc = make_calculator(model)
    examples = [
        dict(label="Low risk (young male, minimally displaced)",
             age=25, female=0, smoking=0, complete_displacement=0,
             comminution=0, shortening_gt2cm=0),
        dict(label="Typical displaced fracture",
             age=40, female=0, smoking=0, complete_displacement=1,
             comminution=0, shortening_gt2cm=0),
        dict(label="High risk (older female smoker, comminuted+shortened)",
             age=60, female=1, smoking=1, complete_displacement=1,
             comminution=1, shortening_gt2cm=1),
    ]
    ex_rows = []
    for e in examples:
        lab = e.pop("label")
        risk = calc(**e)
        pts = points_for_patient({**e}, points_tbl)
        ex_rows.append((lab, round(risk * 100, 1), pts))
    ex_df = pd.DataFrame(ex_rows, columns=["patient", "predicted_risk_pct", "points"])
    ex_df.to_csv(os.path.join(OUT, "example_patients.csv"), index=False)

    # Console report
    print("=== Interpretable nonunion risk model ===")
    print(f"Synthetic cohort n={len(df)}; overall nonunion "
          f"{df['nonunion'].mean()*100:.1f}% (target {BASELINE_NONUNION*100:.0f}%); "
          f"fitted intercept={intercept:.2f}")
    print("\nLogistic coefficients (fitted OR vs published OR):")
    print(coefs.to_string(index=False,
          formatters={"coef": "{:.3f}".format, "OR_fitted": "{:.2f}".format,
                      "OR_ci_low": "{:.2f}".format, "OR_ci_high": "{:.2f}".format,
                      "p_value": "{:.1e}".format, "OR_published": "{:.2f}".format}))
    print("\nModel comparison (5-fold cross-validated):")
    print(metrics.to_string(index=False,
          formatters={"AUC_cv": "{:.3f}".format, "Brier_cv": "{:.3f}".format}))
    print(f"\nPoints table (1 point = log-OR of {B:.3f}):")
    print(points_tbl.to_string(index=False,
          formatters={"log_or": "{:.3f}".format}))
    print("\nDecision-tree rules:\n" + rules)
    print("Example patients:")
    print(ex_df.to_string(index=False))

    return dict(model=model, coefs=coefs, metrics=metrics,
                points_tbl=points_tbl, examples=ex_df, curve=curve)


if __name__ == "__main__":
    run()
