# Intersectional fairness of federated CVD-risk prediction (BRFSS 2023, 52 state clients)

Code, cleaned data and per-seed results for the paper.

- `engine.py` – FedAvg, logistic regression, fairness penalties, training schedule and divergence guard
- `run_brfss.py`, `run2.py`, `run*.sh` – experiment suites and batch scripts
- `analysis*.py`, `make_*.py`, `nums*.py`, `dense.py`, `fig_dense.py` – tables and figures
- `checks3.py` – verifies the numbers quoted in the paper text
- `data/` – cleaned BRFSS 2023 extract and UCI Heart Disease table
- `results/` – per-seed outputs (prediction dumps omitted for size)
- `paper_parts/` – LaTeX sources and `build.py`

Requirements: Python 3, numpy, pandas, scipy, matplotlib.
Data source: CDC BRFSS 2023 (public).
