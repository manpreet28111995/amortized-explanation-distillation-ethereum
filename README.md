# Amortized Explanation Distillation for Ultra-Low-Latency Fraud Alerting

Code, data, results, figures, and manuscript for the study **"Amortized
Explanation Distillation for Ultra-Low-Latency Fraud Alerting: Trading
Shapley Guarantees for Throughput on Real Ethereum Transactions"**
(Springer LNCS format).

KernelSHAP and LIME explanations of a gradient-boosting fraud model are
distilled into lightweight amortized MLP surrogates that map a transaction
directly to its attribution vector in a single forward pass. The study
quantifies the resulting latency–fidelity frontier, the amortization curve
over teacher budgets, the break-even point against direct explanation, and
the loss of the Shapley local-accuracy guarantee.

## Key results (10 seeds, mean ± std)

| Metric | SHAP surrogate | LIME surrogate |
|---|---|---|
| Teacher latency / explanation | 13.4 ± 0.6 ms | 22.0 ± 0.8 ms |
| Surrogate latency / explanation | < 5 µs | ≈ 5 µs |
| **Speed-up** | **2,769 ± 307×** | **4,262 ± 628×** |
| Fidelity R² (held-out alerts) | 0.870 ± 0.025 | 0.820 ± 0.038 |
| Top-3 overlap with teacher | 0.823 ± 0.044 | 0.791 ± 0.071 |
| Kendall τ | 0.688 ± 0.020 | 0.527 ± 0.049 |
| Local-accuracy (efficiency) gap | 0.013 ± 0.001 (teacher: 0.000) | — |

Alert model: gradient boosting, ROC-AUC 0.987 ± 0.002. Break-even against
direct KernelSHAP occurs at ≈ 420 served explanations.

## Repository structure

```
├── src/
│   ├── run_experiments.py     # Full 10-seed pipeline (model → teachers → surrogates → benchmarks)
│   └── generate_figures.py    # Regenerates all paper figures from results/
├── data/
│   └── transaction_dataset.csv  # Ethereum fraud dataset (9,841 accounts; committed, ~3 MB)
├── results/                   # Raw per-seed CSVs and arrays from the reported runs
├── figures/                   # All figures (300 dpi PNG + vector PDF)
├── paper/
│   ├── paper.tex              # Manuscript source (Springer LNCS)
│   ├── paper.pdf              # Compiled manuscript (12 pages)
│   ├── llncs.cls              # Official Springer LNCS class, v2.26
│   └── figures/               # Figure PDFs referenced by the manuscript
├── requirements.txt
└── LICENSE
```

## Dataset

**Ethereum Fraud Detection** — 9,841 real Ethereum accounts (2,179 illicit,
22.14%), described by aggregate transaction-behaviour features introduced by
Farrugia, Ellul & Azzopardi, *Expert Systems with Applications* 150 (2020).
The CSV is small and committed directly under `data/`.
Source: https://www.kaggle.com/datasets/vagifa/ethereum-frauddetection-dataset

## Reproducing the experiments

```bash
pip install -r requirements.txt
python src/run_experiments.py --csv data/transaction_dataset.csv --seeds 0 1 2 3 4 5 6 7 8 9
python src/generate_figures.py
```

Notes:
- `--seeds` accepts any subset, so runs are resumable; per-shard result files
  are merged by simple concatenation (the committed `results/` files are the
  merged outputs of seeds 0–9).
- CPU-only; the full 10-seed run completes in roughly five minutes.

## Compiling the manuscript

```bash
cd paper && pdflatex paper.tex && pdflatex paper.tex
```

Author and affiliation fields in `paper.tex` are placeholders to be completed
before submission.

## License

Code is released under the MIT License (see `LICENSE`). The dataset is
distributed by its authors on Kaggle under its own terms.
