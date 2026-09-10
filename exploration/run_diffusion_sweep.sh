#!/bin/bash
# As D grows, does the targeted action converge onto the uniform one?
# One dose (0.6, above the threshold), both arms, six seeds, network_field.
PY=/home/disc/a.bertin/Documents/Git/.venv/bin/python
for D in 0 0.03 0.3 3 30 300 3000; do
  $PY dose_driver.py --run-tag "diff_D${D}" --workers 12 --seeds 6 \
     --max-steps 160 --warmup 60 --no-terminate --split-seeds \
     --drug-diffusion "$D" --arms uniform,grid_m2 --families network_field \
     --only "d0.6__" >> output/diffusion_sweep.log 2>&1
done
echo "DIFFUSION SWEEP DONE" >> output/diffusion_sweep.log
