"""Capture the drug_1 field itself at a range of diffusion coefficients.

The sweep in output/diff_D* records only scalar summaries per step, which is
enough to show that placement collapses but not enough to SEE it. This runs one
short episode per D, injecting a targeted disc at a fixed point, and saves the
voxel field so the figure can show the field rather than a proxy for it.
"""
import json
import math
import os
import sys

import bootstrap

bootstrap.assert_all()

import common as C  # noqa: E402
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import physigym  # noqa: F401,E402
from extending import physicell  # noqa: E402
from init_conds import generate_initial_condition  # noqa: E402
from lxml import etree  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "drug_fields")
# The compiled physicell module is a per-process singleton (physicell_core.py:181
# refuses a second env), so one D per process; the driver below re-invokes this
# file once per value.
DS = [0.0, 0.3, 3.0, 30.0, 300.0]
# A POINT source of amplitude 1, injected once and read immediately, rather than
# a sustained disc. Three things this fixes, all of which the disc version made
# the reader work out:
#   * the peak is exactly 1.0 at D=0 by construction, so every panel reads as a
#     plain fraction of the injected amplitude with nothing to normalise by;
#   * no disc means no lattice discretisation, so the 241-to-256 voxel ambiguity
#     that made the floor a band rather than a value simply does not arise;
#   * a single pulse shows PURE spreading, where a sustained source shows
#     spreading and re-injection competing.
# Sustained injection of 1.0 would plateau at f/(1-f) = 0.895, not 1, since the
# survival factor over a 15-minute step is exp(-0.05*15) = 0.4724.
# Reverted to the disc 2026-09-10. A unit point source normalises exactly (peak
# 1.000 at D=0, no lattice ambiguity) but has no length scale of its own, so it
# goes from one voxel to the whole domain between D=0 and D=0.3: the peaks are
# then 1, 0.018, 0.002 and every panel after the first renders black on any
# shared scale. The disc has a radius to compete with the diffusion length,
# which is what puts the transition inside the range the figure shows.
POINT_SOURCE = False
# The field is read after the step completes and a step applies one round of
# decay, so injecting 1.0 is observed as 0.4724. Inject the reciprocal so the
# observed peak is exactly 1.0 and every other panel is a plain fraction of it.
POINT_AMPLITUDE = 1.0 / math.exp(-0.05 * 15.0)
WARMUP, TREAT, DOSE = 20, 20, 0.6
RADIUS_NORM = 0.20


def main():
    os.makedirs(OUT, exist_ok=True)
    saved = {}
    i = int(sys.argv[1])
    for D in [DS[i]]:
        run = os.path.join(OUT, f"w{i}")
        xml = os.path.join(run, "settings.xml")
        C.prepare_xml(xml, os.path.join(run, "pc"), max_time=(WARMUP + TREAT + 2) * C.DT_GYM)
        t = etree.parse(xml); r = t.getroot()
        r.xpath("//microenvironment_setup/variable[@name='drug_1']"
                "/physical_parameter_set/diffusion_coefficient")[0].text = repr(float(D))
        t.write(xml, pretty_print=True)

        env = gym.make(**{**C.BASE_ENV_KWARGS, "settingxml": xml})
        env.unwrapped.get_terminated = lambda: False
        u = env.unwrapped
        ic = os.path.join(run, "ic.csv")
        generate_initial_condition(csv_path=ic, mode="network_field",
                                   x_min=u.x_min * .9, x_max=u.x_max * .9,
                                   y_min=u.y_min * .9, y_max=u.y_max * .9,
                                   params=C.make_params(**dict(tumor=128, macrophage=32, t_cell=32)),
                                   seed=0)
        C.point_xml_at_ic(t, r, xml, ic, 0)
        env.reset(seed=-1)
        u.time_simulation = -1

        max_r = float(np.sqrt((u.width / 2) ** 2 + (u.height / 2) ** 2))
        # radius below one voxel half-width => exactly one voxel centre inside
        radius = 0.5 if POINT_SOURCE else RADIUS_NORM * max_r
        # a fixed injection point, so every panel differs ONLY in D
        cx, cy = u.x_min + u.width * 0.5, u.y_min + u.height * 0.5

        for step in range(WARMUP + TREAT):
            if POINT_SOURCE:
                # inject on the final step only, then read: peak is the injected
                # amplitude exactly, undecayed
                dose = POINT_AMPLITUDE if step == WARMUP + TREAT - 1 else 0.0
            else:
                dose = 0.0 if step < WARMUP else DOSE
            physicell.set_parameter("drug_1_x", cx)
            physicell.set_parameter("drug_1_y", cy)
            physicell.set_parameter("drug_1_radius", radius)
            env.step({"drug_1_dose": np.array([dose], dtype=np.float32)})

        field = np.asarray(physicell.get_microenv("drug_1"), float)
        np.savez_compressed(os.path.join(OUT, f"field_D{D}.npz"), field=field,
                            D=D, cx=cx, cy=cy, radius=radius,
                            x_min=u.x_min, x_max=u.x_max,
                            y_min=u.y_min, y_max=u.y_max)
        saved[str(D)] = dict(peak=float(field[:, -1].max()),
                             total=float(field[:, -1].sum()),
                             point_source=POINT_SOURCE)
        print(f"D={D:8.1f}  peak={saved[str(D)]['peak']:.4f}", flush=True)
        env.close()

    json.dump(dict(ds=DS, warmup=WARMUP, treat=TREAT, dose=DOSE,
                   radius_norm=RADIUS_NORM, summary=saved),
              open(os.path.join(OUT, f"manifest_{i}.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
