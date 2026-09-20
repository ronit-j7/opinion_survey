"""Paths and settings shared by every step.

Sensitivity reruns (Test 9) should only need to change values here.
Paths can be overridden with environment variables (used by smoke tests).
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_CSV = Path(os.environ.get("SURVEY_CSV", ROOT.parent / "Survey_Results_UC.csv"))
OUTPUTS = Path(os.environ.get("OUTPUTS_DIR", ROOT / "outputs"))
FIGURES = Path(os.environ.get("FIGURES_DIR", ROOT / "figures"))

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
CORR_METHOD = "pearson"  # "pearson" on the 1-5 codes, or "spearman"
MIN_PAIRWISE_N = 3       # fewer shared respondents -> correlation undefined

# --- Step 03: correlation graph ------------------------------------------------
# Which correlations become edges:
#   "fdr"       -> keep a correlation only if it is statistically significant.
#                  We test all 1,770 pairs at once, so some would pass by pure luck.
#                  FDR ("false discovery rate") handles that: FDR_Q = 0.05 means that,
#                  on average, at most 5% of the edges we keep are flukes.
#   "threshold" -> keep a correlation if its size |r| is at least R_THRESHOLD.
EDGE_RULE = "threshold"
FDR_Q = 0.05
R_THRESHOLD = 0.3

# --- Optional: graphical lasso + EBIC (glasso_optional/) ----------------------
EBIC_GAMMA = 0.5
N_LAMBDA = 100
LAMBDA_MIN_RATIO = 0.01
N_EBIC = None           # None -> median pairwise n
GLASSO_MAX_ITER = 500
GLASSO_TOL = 1e-4
ZERO_TOL = 1e-8         # |value| below this counts as no edge

# --- Steps 05-08: analysis ----------------------------------------------------
CONSENSUS_SHARE = 0.85   # >= this share on one side of the scale -> "consensus" item
SPLIT_MIN_SHARE = 0.25   # at least this share agreeing AND disagreeing -> "split" item
LOUVAIN_SEEDS = 100      # Louvain is randomised; keep the best partition of this many runs
PERMUTATIONS = 10000     # random theme relabellings for the modularity test
BOOTSTRAP_B = 1000       # resamples of the respondents in step 08

# --- Plotting -----------------------------------------------------------------
LIKERT_COLORS = ["#B2182B", "#EF8A62", "#CFCFCF", "#7FADD4", "#2166AC"]  # SD, D, N, A, SA
COMMUNITY_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
                    "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"]
SEED = 42
POSITIVE_EDGE_COLOR = "#2F6DB5"
NEGATIVE_EDGE_COLOR = "#C4452F"


def ensure_dirs() -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
