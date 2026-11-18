# Phenotype priors for mixture-model cytometry population identification

Does encoding published phenotype knowledge (marker levels, per-marker orderings, gating
hierarchy) as priors improve mixture-model identification of manually gated populations?
See `PLAN.md` (design), `REPORT.md` (results, limitations), `results/results_table.md`.

## Layout
```
configs/experiments.yaml     splits, methods (ablation ladder), validation-tuned hyper-parameters
phenotype_priors/*.yaml      per-dataset priors with per-population citations and `inferred` flags
src/cytoprior/data.py        loader -> Dataset(X, markers, y, label_names, sample, meta)
src/cytoprior/priors.py      YAML -> level matrix, per-marker ordering DAGs, per-sample soft thresholds
src/cytoprior/models.py      PriorMixture (t/Gaussian kernels, ordered means, rule gate, labels, flows),
                             GMM / FlowSOM / rules-only baselines
src/cytoprior/metrics.py     Hungarian F1 (Weber & Robinson), identity F1, ARI, rare F1, calibration,
                             ordering violations
scripts/extract_hdcytodata.R .rda (Bioconductor SummarizedExperiment) -> plain binary, no Bioc needed
scripts/tune_val.py          hyper-parameter grid on validation samples only
scripts/run_experiments.py   resumable test runs -> results/runs/*.jsonl
scripts/make_report_assets.py tables + figures -> results/
```

## Reproduce
```bash
pip install torch scikit-learn scipy pandas pyyaml matplotlib anndata flowsom   # + R (base only)
B=https://mghp.osn.xsede.org/bir190004-bucket01/ExperimentHub/HDCytoData
for d in Levine_32dim Levine_13dim Samusik_all Nilsson_rare Mosmann_rare; do
  curl -o data/raw/hdcytodata/${d}_SE.rda $B/$d/${d}_SE.rda
  Rscript scripts/extract_hdcytodata.R data/raw/hdcytodata/${d}_SE.rda data/interim/$d
done
sha256sum -c data/SHA256SUMS_raw.txt   # run inside data/raw/hdcytodata
PYTHONPATH=src python -m cytoprior.data
PYTHONPATH=src python scripts/tune_val.py Samusik_all
PYTHONPATH=src python scripts/run_experiments.py --workers 4
PYTHONPATH=src python scripts/run_experiments.py --split val --datasets Samusik_all Levine_32dim --methods L2_flowsom L4_both L5_both_ref3 L6_both L6_both_marginal X_marg_rules X_marg_both_ref3 X_marg_t_both   # validation check of post-hoc 3b variants
PYTHONPATH=src python scripts/make_report_assets.py
```
Seeds are fixed (0-4); device is chosen automatically (cuda > mps > cpu); results here were on CPU.
