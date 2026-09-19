"""Paths and settings shared by every step.

Every discretionary knob lives here so the robustness sweep (10_robust.py)
and sensitivity reruns only need to change values in one place.
Paths can be overridden with environment variables.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_CSV = Path(os.environ.get("RG_DATA_CSV", ROOT.parent / "Survey_Results_UC.csv"))
OUTPUTS = Path(os.environ.get("RG_OUTPUTS", ROOT / "outputs"))
FIGURES = Path(os.environ.get("RG_FIGURES", ROOT / "figures"))

# --- Step 01: encoding and filter cascade ------------------------------------
LIKERT = {
    "Strongly Disagree": 1,
    "Disagree": 2,
    "Neutral": 3,
    "Agree": 4,
    "Strongly Agree": 5,
}
NO_COMMENT = "No Comments"  # treated as missing, not as Neutral
MIN_ANSWERED = 40           # second filter tier
PRIMARY_FILTER = "complete"  # tiers: all (91) | ge40 (87) | complete (68)

DOMAINS = ("T", "E", "S", "V")  # V = Environment, matching the survey codes
THEMES = {"T": "Technology", "E": "Education", "S": "Society", "V": "Environment"}
THEME_COLORS = {  # 4 evenly spaced magma samples (0.20/0.42/0.64/0.86)
    "T": "#3B0F6F", "E": "#942B80", "S": "#EA5560", "V": "#FEBC82",
}

# --- Step 02: measures --------------------------------------------------------
HIST_BINS = 5          # answer categories 1..5
IPSATIVE_ZERO = True   # profiles are row-centered domain means (rows sum to 0)

# --- Step 03: OT metric -------------------------------------------------------
COST_KIND = "abs"      # abs -> |c - c'| ; sq -> (c - c')^2
AGG = "sum"            # sum | max | weighted (weighted = by answered items per domain)
EMD_TOL = 1e-12        # emd2 vs closed-form CDF-L1 agreement (abs cost only)
GATE_RHO = 0.85        # degeneration gate vs mean-priority distance
N_SPLIT = 100          # repeated stratified item half-splits
N_SPLIT_NULL = 200     # item-label permutation null for the split-half
NEGCTRL_SEED = 7       # random ground cost for the discarded 60-dim control

# --- Step 05: persistent homology ---------------------------------------------
MAXDIM = 1             # H0 and H1
N_PERM = 500           # column-permutation null ensemble
N_BOOT = 200           # respondent bootstrap
N_PERM_GRID = 100      # cheaper null inside the robustness grid

# --- Step 06: edges and the Louvain comparison --------------------------------
K_MUTUAL = 6           # mutual-kNN neighbours
K_SWEEP = range(3, 11)  # k stability sweep
N_NULL_GRAPH = 200     # null graphs for the modularity comparison
N_SUBSAMPLE = 100      # item half-splits for ARI / co-assignment
FACTION_MIN_SIZE_FRAC = 0.05  # a "faction" needs >= 5% of n on both sides
ALPHA = 0.05

# --- Step 09: priority validity -----------------------------------------------
MINORITY_SHARE = 0.10  # contested-item rule: min(agree, disagree) >= 0.10
CONTESTED_DEF_N = "ge40"   # contestedness defined on the n=87 subset
ENTROPY_TOPK = 10
BH_Q = 0.05
N_PERM_REG = 1000

# --- Step 10: robustness grid --------------------------------------------------
GRID_COSTS = ("abs", "sq")
GRID_AGGS = ("sum", "max", "weighted")
GRID_FILTERS = ("complete", "ge40", "all")
WITNESS_LANDMARK_FRAC = 0.2

SEED = 42


def ensure_dirs() -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
