"""Paths and settings shared by every step.

Sensitivity reruns (Test 9) should only need to change values here.
Paths can be overridden with environment variables (used by smoke tests).
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_CSV = Path(os.environ.get("BN_DATA_CSV", ROOT.parent / "Survey_Results_UC.csv"))
OUTPUTS = Path(os.environ.get("BN_OUTPUTS", ROOT / "outputs"))
FIGURES = Path(os.environ.get("BN_FIGURES", ROOT / "figures"))

# --- Step 01: encoding -------------------------------------------------------
LIKERT = {
    "Strongly Disagree": 1,
    "Disagree": 2,
    "Neutral": 3,
    "Agree": 4,
    "Strongly Agree": 5,
}
NO_COMMENT = "No Comments"  # treated as missing, not as Neutral

THEMES = {"T": "Technology", "E": "Education", "S": "Society", "V": "Environment"}
THEME_COLORS = {"T": "#7B5EA7", "E": "#E09F3E", "S": "#2A9D8F", "V": "#6C9A3A"}

# --- Step 02: correlations ---------------------------------------------------
MIN_PAIRWISE_N = 3      # fewer shared respondents -> correlation undefined
EIG_FLOOR = 1e-4        # repair R only if its smallest eigenvalue is below this

# --- Step 03: graphical lasso + EBIC -----------------------------------------
EBIC_GAMMA = 0.5
N_LAMBDA = 100
LAMBDA_MIN_RATIO = 0.01
N_EBIC = None           # None -> median pairwise n
GLASSO_MAX_ITER = 500
GLASSO_TOL = 1e-4
ZERO_TOL = 1e-8         # |value| below this counts as no edge

# --- Step 04: baseline marginal graph ----------------------------------------
FDR_Q = 0.05

# --- Plotting -----------------------------------------------------------------
SEED = 42
POSITIVE_EDGE_COLOR = "#2F6DB5"
NEGATIVE_EDGE_COLOR = "#C4452F"


def ensure_dirs() -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
