"""Dataset loading. Raw files in data/raw are never modified.

Pipeline: data/raw/hdcytodata/<name>_SE.rda --(scripts/extract_hdcytodata.R)--> data/interim/<name>/
--(this module)--> data/processed/<name>.npz (transformed, clustering markers only).
"""
from __future__ import annotations

import gzip
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"

# Provenance, recorded per dataset (see also PLAN.md / REPORT.md).
DATASETS = {
    "Levine_32dim": dict(
        cofactor=5.0, multi_label=True, technology="mass cytometry (CyTOF)",
        publication="Levine JH et al. Cell 162:184-197 (2015) doi:10.1016/j.cell.2015.05.047",
        labels="manual gating by the original authors (14 populations); 'unassigned' cells not gated",
        source="FlowRepository FR-FCM-ZZPH via Bioconductor HDCytoData (ExperimentHub EH2240)",
    ),
    "Levine_13dim": dict(
        cofactor=5.0, multi_label=True, technology="mass cytometry (CyTOF)",
        publication="Bendall SC et al. Science 332:687-696 (2011) doi:10.1126/science.1198704 (benchmark set in Levine 2015)",
        labels="manual gating (24 populations) from Bendall 2011",
        source="FR-FCM-ZZPH via HDCytoData (EH2242)",
    ),
    "Samusik_all": dict(
        cofactor=5.0, multi_label=True, technology="mass cytometry (CyTOF)",
        publication="Samusik N et al. Nat Methods 13:493-496 (2016) doi:10.1038/nmeth.3863",
        labels="consensus of 3 experts' hand-gating, 24 populations, 10 replicate mouse BM samples",
        source="FR-FCM-ZZPH via HDCytoData (EH2246)",
    ),
    "Nilsson_rare": dict(
        cofactor=150.0, multi_label=False, technology="flow cytometry",
        publication="Rundberg Nilsson A et al. Cytometry A 83:721-727 (2013) doi:10.1002/cyto.a.22324",
        labels="HSC gate re-drawn by Weber & Robinson 2016 in Cytobank following Nilsson 2013 Fig 2",
        source="FR-FCM-ZZPH via HDCytoData (EH2248)",
    ),
    "Mosmann_rare": dict(
        cofactor=150.0, multi_label=False, technology="flow cytometry",
        publication="Mosmann TR et al. Cytometry A 85:422-433 (2014) doi:10.1002/cyto.a.22445",
        labels="activated memory CD4 T cells (IFNg+TNFa+) re-gated by Weber & Robinson 2016 following Mosmann 2014 Fig 4A",
        source="FR-FCM-ZZPH via HDCytoData (EH2250)",
    ),
}
UNASSIGNED = {"unassigned", "other"}


@dataclass
class Dataset:
    name: str
    X: np.ndarray                 # (n, d) float32, arcsinh-transformed, clustering markers
    markers: list[str]
    y: np.ndarray                 # (n,) int, -1 = unassigned / not gated
    label_names: list[str]
    sample: np.ndarray            # (n,) str sample id
    meta: dict = field(default_factory=dict)

    @property
    def samples(self) -> list[str]:
        return sorted(set(self.sample.tolist()))

    def subset(self, mask) -> "Dataset":
        return Dataset(self.name, self.X[mask], self.markers, self.y[mask], self.label_names,
                       self.sample[mask], self.meta)

    def by_sample(self, s: str) -> "Dataset":
        return self.subset(self.sample == s)

    def summary(self) -> dict:
        return dict(name=self.name, n_cells=int(len(self.y)), n_labelled=int((self.y >= 0).sum()),
                    n_samples=len(self.samples), n_markers=len(self.markers),
                    n_populations=len(self.label_names), **self.meta)


def _process(name: str) -> Path:
    src = DATA / "interim" / name
    n, p = map(int, (src / "dim.txt").read_text().split())
    m = np.fromfile(src / "exprs.f64", dtype=np.float64).reshape(p, n).T  # R is column-major
    col = pd.read_csv(src / "col_data.csv")
    row = pd.read_csv(src / "row_data.csv.gz", dtype=str)
    # clustering markers = all non-'none' markers (W&R: 'type' for multi-pop sets; type+state for Mosmann)
    use = (col.marker_class != "none").to_numpy()
    info = DATASETS[name]
    X = np.arcsinh(m[:, use] / info["cofactor"]).astype(np.float32)
    markers = col.marker_name[use].tolist()
    pops = row.population_id.to_numpy()
    names = sorted(set(pops) - UNASSIGNED)
    if not info["multi_label"]:
        names = [x for x in names]  # single rare pop
    lut = {nm: i for i, nm in enumerate(names)}
    y = np.array([lut.get(p_, -1) for p_ in pops], dtype=np.int64)
    scol = next((c for c in ("sample_id", "patient_id") if c in row), None)
    sample = row[scol].to_numpy() if scol else np.array(["S1"] * n)
    out = DATA / "processed" / f"{name}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, X=X, y=y, sample=sample.astype(str), markers=np.array(markers),
                        label_names=np.array(names))
    return out


def load_dataset(name: str) -> Dataset:
    f = DATA / "processed" / f"{name}.npz"
    if not f.exists():
        _process(name)
    z = np.load(f, allow_pickle=False)
    info = {k: v for k, v in DATASETS[name].items()}
    return Dataset(name, z["X"], z["markers"].tolist(), z["y"], z["label_names"].tolist(),
                   z["sample"], info)


if __name__ == "__main__":
    rows = []
    for nm in DATASETS:
        d = load_dataset(nm)
        s = d.summary()
        s["cells_per_sample"] = {k: int((d.sample == k).sum()) for k in d.samples}
        rows.append(s)
        print(json.dumps(s, indent=1))
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "dataset_summary.json").write_text(json.dumps(rows, indent=1))
