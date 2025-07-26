# utils/eda_generator.py
from ydata_profiling import ProfileReport
import pandas as pd
import tempfile
import os

def generate_eda_report(df: pd.DataFrame) -> str:
    """Generates a PDF EDA report and returns the file path."""
    profile = ProfileReport(df, title="📊 EDA Report", explorative=True)
    temp_dir = tempfile.mkdtemp()
    pdf_path = os.path.join(temp_dir, "eda_report.pdf")
    profile.to_file(pdf_path)
    return pdf_path
