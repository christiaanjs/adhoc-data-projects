#!/usr/bin/env python3
"""
Command-line clavicle-fracture outcome predictor.

Given patient and fracture factors, it reports:
  * baseline risk of NONUNION with nonoperative treatment,
  * predicted risk WITH operative fixation,
  * the absolute risk reduction (ARR) and number-needed-to-treat (NNT) from
    surgery, each with a 95% credible interval,
  * the interpretable points score.

All estimates come from the models built by `run_all.py`; run that once first.

Examples
--------
  python predict.py --age 28
  python predict.py --age 62 --female --smoking --displacement --comminution --shortening
  python predict.py --age 55 --displacement --location distal
  python predict.py --age 40 --smoking --displacement --json

This is a decision-support demonstration, NOT a validated clinical tool.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import pandas as pd  # noqa: E402
from treatment_benefit import (Patient, treatment_benefit,  # noqa: E402
                               treatment_benefit_unified, benefit_band)

ROOT = os.path.dirname(os.path.abspath(__file__))
POINTS_CSV = os.path.join(ROOT, "outputs", "points_table.csv")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Predict clavicle fracture nonunion risk and the benefit of surgery.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--age", type=float, default=40.0, help="patient age in years")
    p.add_argument("--female", action="store_true", help="patient is female")
    p.add_argument("--smoking", action="store_true", help="current smoker")
    p.add_argument("--displacement", action="store_true",
                   help="complete displacement (no cortical contact)")
    p.add_argument("--comminution", action="store_true",
                   help="comminuted fracture")
    p.add_argument("--shortening", action="store_true",
                   help="shortening > 2 cm")
    p.add_argument("--location", choices=["midshaft", "distal"],
                   default="midshaft", help="fracture location")
    p.add_argument("--model", choices=["unified", "two-stage"],
                   default="unified",
                   help="unified single-fit joint model, or the two-stage "
                        "(separate risk model x meta-analysis) combination")
    p.add_argument("--json", action="store_true",
                   help="emit machine-readable JSON instead of a report")
    p.add_argument("--seed", type=int, default=0, help="Monte-Carlo seed")
    return p.parse_args(argv)


def points_score(patient: Patient) -> int:
    if not os.path.exists(POINTS_CSV):
        return None
    tbl = pd.read_csv(POINTS_CSV)
    pts = {r["factor"]: r["points"] for _, r in tbl.iterrows()}
    score = 0
    if patient.age > 40:
        score += int(pts["age"]) * int((patient.age - 40) // 10)
    for f in ["female", "smoking", "complete_displacement",
              "comminution", "shortening_gt2cm"]:
        if getattr(patient, f):
            score += int(pts[f])
    return int(score)


def pct(d):
    return f"{d['median']*100:.1f}%  (95% CrI {d['lo']*100:.1f}-{d['hi']*100:.1f}%)"


def main(argv=None):
    args = parse_args(argv)
    patient = Patient(
        age=args.age, female=int(args.female), smoking=int(args.smoking),
        complete_displacement=int(args.displacement),
        comminution=int(args.comminution),
        shortening_gt2cm=int(args.shortening), location=args.location)

    try:
        if args.model == "unified":
            res = treatment_benefit_unified(patient)
        else:
            res = treatment_benefit(patient, seed=args.seed)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    res["model"] = args.model
    res["points_score"] = points_score(patient)

    if args.json:
        print(json.dumps(res, indent=2))
        return 0

    factors = [name for name, on in [
        ("female", args.female), ("smoker", args.smoking),
        ("complete displacement", args.displacement),
        ("comminution", args.comminution), ("shortening >2cm", args.shortening),
    ] if on] or ["none"]
    nnt = res["nnt"]
    arr = res["absolute_risk_reduction"]

    print("=" * 64)
    print(" Clavicle fracture nonunion — outcome prediction")
    print("=" * 64)
    print(f" Model               : {args.model}")
    print(f" Location            : {args.location}")
    print(f" Age                 : {args.age:.0f}")
    print(f" Risk factors        : {', '.join(factors)}")
    if res["points_score"] is not None:
        print(f" Points score        : {res['points_score']}")
    print("-" * 64)
    print(f" Nonunion risk if treated NONoperatively : {pct(res['risk_nonoperative'])}")
    print(f" Nonunion risk if treated operatively    : {pct(res['risk_operative'])}")
    eff = "odds ratio" if args.model == "unified" else "risk ratio"
    print(f" Relative effect of surgery ({eff:<10}) : "
          f"{res['relative_risk']['median']:.2f}  "
          f"(95% CrI {res['relative_risk']['lo']:.2f}-{res['relative_risk']['hi']:.2f})")
    print("-" * 64)
    print(f" Absolute risk reduction from surgery    : {pct(arr)}")
    print(f" Number needed to treat (NNT)            : "
          f"{nnt['median']:.0f}  (95% CrI {nnt['lo']:.0f}-{nnt['hi']:.0f})")
    print(f" Interpretation                          : {benefit_band(nnt['median'])}")
    print("=" * 64)
    print(" Decision support only — not a validated clinical tool.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
