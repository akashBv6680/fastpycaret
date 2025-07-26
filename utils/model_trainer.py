# utils/model_trainer.py
from pycaret.classification import setup as clf_setup, compare_models as compare_classifiers, pull as pull_clf
from pycaret.regression import setup as reg_setup, compare_models as compare_regressors, pull as pull_reg
import pandas as pd
import tempfile
import os

def train_model_and_report(df: pd.DataFrame, target_column: str) -> tuple[str, str]:
    """Train a model using PyCaret and export results as PDF. Returns (path, task_type)."""
    # Check if target is numeric (regression) or categorical (classification)
    if pd.api.types.is_numeric_dtype(df[target_column]):
        task_type = "regression"
        reg_setup(data=df, target=target_column, silent=True, verbose=False)
        compare_regressors()
        results_df = pull_reg()
    else:
        task_type = "classification"
        clf_setup(data=df, target=target_column, silent=True, verbose=False)
        compare_classifiers()
        results_df = pull_clf()

    # Save report to PDF
    temp_dir = tempfile.mkdtemp()
    pdf_path = os.path.join(temp_dir, f"{task_type}_report.pdf")
    results_df.to_string(buf=open(pdf_path, "w"))
    return pdf_path, task_type
