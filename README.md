# dfa_compare

Aggregates and compare results from multiple df-analyze runs

## what it does
- collects results from multiple runs
- compares key metrics like `acc`, `auroc`, `f1`

## input
w each run folder should contain one of:
- `results/performance_long_table.csv`
- `results/final_performances.csv`
- `results/results_report.md`

## output
- `merged_performance_long.csv`
- `performance_summary.csv`
- `best_holdout_configs.csv`
- `warnings.csv`
- plots (`avg_holdout_<metric>.png`, `holdout_variability_<metric>.png`)
- `report.md`

## install
```bash
pip install -r requirements.txt
```

## usage
```bash
python -m cli --input <runs_dir> --output <out_dir> --metric <metric>
```

for example:
```bash
python -m cli --input test_runs --output output --metric auroc
```

## notes
- right now default metrics are `acc`, `auroc`, `f1`
- any metric present in the df analyze result_report.md can be used via `--metric`
- falls back to available metrics if missing
- has safety to skip bad runs with warnings
