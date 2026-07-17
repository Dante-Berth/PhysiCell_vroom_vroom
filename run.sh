#!/usr/bin/env bash
# Launch one SAC run per observation mode, sequentially.
# Uses best hyperparameters from reward_analysis: w_cell=0.3, w_dose=2.0, w_smooth=0.0
# Source: reward_analysis composite_score=2.972 (best overall)

set -euo pipefail

# ── Environment bootstrap ───────────────────────────────────────
# run.py mixes bare imports (from vectorized import ...) with full-package
# imports (from custom_modules.physigym.physigym.envs...), so BOTH the project
# root and the envs/ dir must be importable. Also there is no `python` on PATH,
# so use the project-local .venv explicitly.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENVS_DIR="${PROJECT_ROOT}/custom_modules/physigym/physigym/envs"
export PYTHONPATH="${PROJECT_ROOT}:${ENVS_DIR}:${PYTHONPATH:-}"

# Silence PhysiCell's C++ console output (the compiled physicell module mutes
# std::cout while PHYSIGYM_QUIET is set). run.py's own logs stay visible.
export PHYSIGYM_QUIET=1

if [[ -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
  PYTHON="${PROJECT_ROOT}/.venv/bin/python"
else
  PYTHON="$(command -v python3)"
fi

# ── 0. Shared configuration ─────────────────────────────────────
# Defined before section 1: the random baseline reads OBS_MODES/W_* too, and
# `set -u` aborts on any use before assignment.
SEEDS=(1 2 3)


 OBS_MODES=(
   img_mc_cells_substrates
   img_mc_cells
   img_mc_cells_substrates_m1m2
)

# Best hyperparameters from reward_analysis
W_CELL=0.3
W_DOSE=2.0
W_SMOOTH=0.0


# ── 1. SAC training runs (one per obs mode) ─────────────────────
for seed in "${SEEDS[@]}"; do
  for obs in "${OBS_MODES[@]}"; do
    echo "============================================================"
    echo "  SAC  seed=${seed}  observation_mode=${obs}"
    echo "  Hyperparameters: w_cell=${W_CELL} w_dose=${W_DOSE} w_smooth=${W_SMOOTH}"
    echo "============================================================"
    "$PYTHON" custom_modules/physigym/physigym/envs/run.py \
      --seed           "${seed}"  \
      --observation_mode "${obs}" \
      --action_mode    targeted   \
      --w_cell         ${W_CELL}  \
      --w_dose         ${W_DOSE}  \
      --w_smooth       ${W_SMOOTH} \
      --action_repeat  6          \
      --delta_x        0.25       \
      --delta_y        0.25       \
      --delta_radius   0.03       \
      --max_time_episode 7200     \
      --total_timesteps 100000    \
      --wandb          true       \
      --name           "best_hyperparameters_SAC_${obs}_w_cell=${W_CELL}_w_dose=${W_DOSE}_w_smooth=${W_SMOOTH}_seed${seed}"
  done
done

# ── 3. Compile videos once, at the very end (deferred compilation) ─
# Compiling inside the training loop re-scanned the whole data/ tree on
# every iteration (25x) for no benefit. Instead compile a single time here,
# and only for the last 10 runs per env/split (--last-n) so we don't render
# thousands of intermediate test episodes.
echo "============================================================"
echo "  Compiling videos (last 10 runs per env)..."
echo "============================================================"
"$PYTHON" video_maker.py --base-dir data/ --last-n 20

echo "============================================================"
echo "  All training and video compilation complete!"
echo "============================================================"


# ── 2. Random policy baseline — one run per seed (obs mode agnostic) ──
# Random baseline over its own seed list (independent of the training SEEDS).
# Runs after SAC so the training sweep isn't gated behind the baseline.
RANDOM_SEEDS=(1 2 3 4 5)
for seed in "${RANDOM_SEEDS[@]}"; do
  echo "============================================================"
  echo "  RANDOM BASELINE  seed=${seed}  observation_mode=${OBS_MODES[0]}"
  echo "  Hyperparameters: w_cell=${W_CELL} w_dose=${W_DOSE} w_smooth=${W_SMOOTH}"
  echo "============================================================"
  "$PYTHON" custom_modules/physigym/physigym/envs/run.py \
    --mode           random          \
    --seed           "${seed}"       \
    --observation_mode "${OBS_MODES[0]}" \
    --action_mode    targeted        \
    --w_cell         ${W_CELL}       \
    --w_dose         ${W_DOSE}       \
    --w_smooth       ${W_SMOOTH}     \
    --action_repeat  6               \
    --delta_x        0.25            \
    --delta_y        0.25            \
    --delta_radius   0.03            \
    --max_time_episode 7200          \
    --total_timesteps 100000         \
    --wandb          true            \
    --name           "best_hyperparameters_RANDOM_baseline_w_cell=${W_CELL}_w_dose=${W_DOSE}_w_smooth=${W_SMOOTH}_seed${seed}"
done



