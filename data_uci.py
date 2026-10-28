"""
data_uci.py
------------
Loader for the cleaned UCI Heart Disease data (see prepare_uci.py).
Each of the 4 hospitals (cleveland, hungarian, switzerland, va) is one REAL
federated client.

LIMITATIONS TO DISCLOSE:
  * Only 851 patients in total; some site x (age,sex) cells hold 2-12 patients.
    Fairness metrics on single folds would be pure noise, so experiment_uci.py
    pools out-of-fold predictions over repeated cross-validation instead of
    using one small test split.
  * Strong label shift across sites: disease prevalence is 36% (Hungarian),
    46% (Cleveland), 79% (VA), 93% (Switzerland). Switzerland and VA are
    ~90-97% male. Real heterogeneity, but it also means site and sex are
    entangled with outcome.
  * Data is from the 1980s; no clinical claims -- method comparison only.
  * chol, fbs, slope, ca, thal were dropped (see prepare_uci.py).
"""
import numpy as np
import pandas as pd

UCI_FEATURE_COLS = ["age", "sex", "trestbps", "thalach", "exang", "oldpeak",
                    "cp_2", "cp_3", "cp_4", "restecg_1", "restecg_2"]
SITES = ["cleveland", "hungarian", "switzerland", "va"]
# FIXED group order. Every client must use the same column order, even if a
# client has no patient in some group (otherwise one-hot columns misalign
# across clients and the shrinkage pooling step would mix up groups).
GROUPS = ["old_female", "old_male", "young_female", "young_male"]


def load_uci(path: str = "uci_heart_clean.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = set(UCI_FEATURE_COLS + ["target", "site"]) - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    return df


def add_intersectional_group(df: pd.DataFrame, age_threshold: int = 55) -> pd.DataFrame:
    df = df.copy()
    age_bucket = np.where(df["age"] >= age_threshold, "old", "young")
    sex_label = np.where(df["sex"] == 1, "male", "female")
    df["intersect_group"] = age_bucket + "_" + sex_label
    return df
