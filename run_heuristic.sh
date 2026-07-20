#!/usr/bin/env bash
# Stage 4 — macrophage-aware heuristic baseline (rule-based, no learning).
#
# The policy (PhysiCellModel.get_heuristic_action) injects a fixed dose at the
# centroid of the M2 (pro-tumoral) macrophages within HEURISTIC_RADIUS microns
# of any tumour cell — the cells drug_1 actually re-polarises per cell_rules.csv.
# It plugs into the same action interface as RAND/POMDP/SAC and is evaluated
# identically (same obs mode is irrelevant — the rule reads ground-truth
# positions, not the observation).
#
# Two phases:
#   PHASE A  radius tuning   — 1 seed over a few radii, pick best test return
#   PHASE B  full evaluation — 5 seeds at the chosen radius (set BEST_RADIUS)
# Run phase A first, inspect wandb test_return, set BEST_RADIUS, then phase B.
# Toggle the phases with RUN_PHASE_A / RUN_PHASE_B below.

set -euo pipefail

# ── Environment bootstrap (identical to run.sh) ─────────────────
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENVS_DIR="${PROJECT_ROOT}/custom_modules/physigym/physigym/envs"
export PYTHONPATH="${PROJECT_ROOT}:${ENVS_DIR}:${PYTHONPATH:-}"
export PHYSIGYM_QUIET=1

if [[ -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
  PYTHON="${PROJECT_ROOT}/.venv/bin/python"
else
  PYTHON="$(command -v python3)"
fi

# ── Shared configuration (match run.sh's baseline block) ────────
W_CELL=0.3
W_DOSE=2.0
W_SMOOTH=0.0
# Obs mode is irrelevant to the heuristic (it reads ground-truth cell positions),
# but run.py still needs a valid one to build the env; reuse the primary mode.
OBS_MODE=img_mc_cells_substrates
DOSE=0.5

RUN_PHASE_A=true       # radius tuning sweep
RUN_PHASE_B=false      # full 5-seed eval — enable after picking BEST_RADIUS
RADII=(5 10 20)        # phase A candidate radii (microns; domain is ~64 um)
TUNE_SEED=1
EVAL_SEEDS=(1 2 3 4 5)
BEST_RADIUS=10         # set this from phase A results before running phase B

run_heuristic () {
  local seed="$1" radius="$2" name="$3"
  echo "============================================================"
  echo "  HEURISTIC  seed=${seed}  radius=${radius}um  dose=${DOSE}"
  echo "============================================================"
  "$PYTHON" custom_modules/physigym/physigym/envs/run.py \
    --mode             heuristic       \
    --seed             "${seed}"       \
    --observation_mode "${OBS_MODE}"   \
    --action_mode      targeted        \
    --heuristic_radius "${radius}"     \
    --heuristic_dose   "${DOSE}"       \
    --w_cell           ${W_CELL}       \
    --w_dose           ${W_DOSE}       \
    --w_smooth         ${W_SMOOTH}     \
    --action_repeat    6               \
    --delta_x          0.25            \
    --delta_y          0.25            \
    --delta_radius     0.03            \
    --max_time_episode 7200            \
    --total_timesteps  100000          \
    --wandb            true            \
    --name             "${name}"
}

# ── PHASE A: radius tuning ──────────────────────────────────────
if [[ "${RUN_PHASE_A}" == "true" ]]; then
  for radius in "${RADII[@]}"; do
    run_heuristic "${TUNE_SEED}" "${radius}" \
      "heuristic_tuning_HEURISTIC_baseline_radius${radius}_w_cell=${W_CELL}_w_dose=${W_DOSE}_w_smooth=${W_SMOOTH}_seed${TUNE_SEED}"
  done
  echo "PHASE A done. Inspect wandb test_return per radius, set BEST_RADIUS, enable PHASE B."
fi

# ── PHASE B: full 5-seed evaluation at the chosen radius ────────
if [[ "${RUN_PHASE_B}" == "true" ]]; then
  for seed in "${EVAL_SEEDS[@]}"; do
    run_heuristic "${seed}" "${BEST_RADIUS}" \
      "best_hyperparameters_HEURISTIC_baseline_w_cell=${W_CELL}_w_dose=${W_DOSE}_w_smooth=${W_SMOOTH}_seed${seed}"
  done
fi

echo "All heuristic runs complete."
