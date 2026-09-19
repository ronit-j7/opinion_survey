#!/usr/bin/env bash
# Run the full respondent_graph pipeline in the dpcn conda environment.
#
# Usage:
#   ./run.sh              incremental rerun (outputs overwritten in place)
#   ./run.sh --fresh      wipe outputs/ and figures/ first (clean slate)
#   ./run.sh --skip-deps  skip the dependency probe/install
#
# Env overrides: CONDA_ENV (default dpcn). Each step logs to logs/NN_*.log.
set -euo pipefail

cd "$(dirname "$0")"

ENV_NAME="${CONDA_ENV:-dpcn}"
PY=(conda run --no-capture-output -n "$ENV_NAME" python)

FRESH=0
SKIP_DEPS=0
for arg in "$@"; do
  case "$arg" in
    --fresh)     FRESH=1 ;;
    --skip-deps) SKIP_DEPS=1 ;;
    *) echo "unknown flag: $arg (use --fresh or --skip-deps)"; exit 2 ;;
  esac
done

STEPS=(
  01_prepare
  02_measures
  03_ot_metric
  04_plans_barycenter
  05_persistence
  06_h0_factions
  07_outliers
  08_geometry
  09_priority_validity
  10_robust
  11_interactive
  12_filtration
)
N_STEPS=${#STEPS[@]}

echo "== respondent_graph pipeline | env: ${ENV_NAME} | ${N_STEPS} steps =="

if (( FRESH )); then
  echo "--fresh: removing outputs/ figures/"
  rm -rf outputs figures
fi

mkdir -p logs outputs figures

if (( ! SKIP_DEPS )); then
  echo "checking dependencies"
  if "${PY[@]}" -c "
import numpy, pandas, scipy, sklearn, networkx, matplotlib, seaborn, plotly
import pyarrow, ot, ripser, persim, gudhi" >/dev/null 2>&1; then
    echo "dependencies OK"
  else
    echo "installing requirements.txt into ${ENV_NAME}"
    conda run -n "$ENV_NAME" pip install -r requirements.txt
  fi
fi

run_step() {
  local name="$1" idx="$2" t0=$SECONDS
  printf '[%2d/%d] %-24s ' "$idx" "$N_STEPS" "$name"
  if "${PY[@]}" "${name}.py" >"logs/${name}.log" 2>&1; then
    printf 'ok  (%ds)\n' "$((SECONDS - t0))"
  else
    echo "FAILED - see logs/${name}.log"
    exit 1
  fi
}

idx=1
for name in "${STEPS[@]}"; do
  run_step "$name" "$idx"
  idx=$((idx + 1))

  if [[ "$name" == "03_ot_metric" ]]; then
    if ! "${PY[@]}" -c "
import json, sys
v = json.load(open('outputs/metric_validation.json'))
sys.exit(0 if v['gate_passed'] else 1)"; then
      rho=$("${PY[@]}" -c "
import json
print('%.3f' % json.load(open('outputs/metric_validation.json'))['rho_w1_vs_mean_priority'])")
      echo ""
      echo "=============================================================="
      echo " GATE FAILED: rho(W1, mean-priority) = ${rho} >= 0.85"
      echo " OT has degenerated - do NOT build steps 04-08 on this metric."
      echo " See outputs/metric_validation.json before proceeding."
      echo "=============================================================="
      exit 1
    fi
  fi
done

echo ""
echo "== summary =="
"${PY[@]}" - <<'EOF'
import json
import pandas as pd

mv = json.load(open("outputs/metric_validation.json"))
print(f"metric gate          rho = {mv['rho_w1_vs_mean_priority']:.3f} "
      f"(gate < {mv['gate_rho_max']}: {'PASS' if mv['gate_passed'] else 'FAIL'})")
print(f"split-half           mean r = {mv['split_half']['mean_r']:.3f}, "
      f"z = {mv['split_half']['z']:+.1f}")
fv = json.load(open("outputs/faction_verdict.json"))
print(f"factions             {fv['verdict']} "
      f"({fv['n_significant_gaps']} significant gaps, "
      f"max split {fv['significant_splits'][-1]['sizes'] if fv['significant_splits'] else '-'})")
try:
    out = pd.read_csv("outputs/outliers.csv")
    labels = [f"id {i} (row {r})" for i, r in
              zip(out["response_id"], out["row_position"])]
    print(f"outliers             {', '.join(labels)}")
except FileNotFoundError:
    print("outliers             none detected")
vi = json.load(open("outputs/validity_info.json"))
print(f"priority validity    split-half null z = {vi['split_half_null_z']:+.1f}; "
      f"BH survivors per set {vi['n_rejected_per_set']}")
rs = json.load(open("outputs/robustness_summary.json"))
print(f"robustness           {rs['n_cells']} grid cells; priority validity stable: "
      f"{rs['priority_validity_stable']}")
EOF
echo ""
echo "done: outputs/ artifacts, figures/ (incl. figures/interactive/), logs/"
