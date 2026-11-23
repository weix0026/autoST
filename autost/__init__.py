"""AutoST fire-spread model: training and evaluation from the published crops."""
import os

# NumPy (MKL) and PyTorch can each bring their own OpenMP runtime on Windows
# conda builds; without this the process aborts the first time both are used.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

__version__ = "1.0.0"
