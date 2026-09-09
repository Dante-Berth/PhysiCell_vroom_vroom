"""Why do all macrophages read as M2 from step 1?

Tests the hypothesis that tumor_molecule saturates the polarisation rules.

cell_rules.csv, for the macrophage:
    tumor_molecule increases pro_tumoral_factor secretion  (max 10, half 0.5, hill 1)
    tumor_molecule decreases anti_tumoral_factor secretion (max  0, half 0.5, hill 1)
and drug_1 is the only signal that pushes either back the other way.

Prints, per step: the tumor_molecule distribution AT THE MACROPHAGES, the Hill
response that implies, and the fraction of the domain where pro > anti (which is
what get_macrophage_polarization_scalars actually thresholds on).
"""

import bootstrap

bootstrap.assert_all()

import common as C  # noqa: E402
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import physigym  # noqa: F401,E402
from extending import physicell  # noqa: E402
from init_conds import generate_initial_condition  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

HALF, HILL, MAXPRO = 0.5, 1.0, 10.0
RUN = "exploration/output/_polarisation"
XML = f"{RUN}/settings.xml"

tree, root = C.prepare_xml(XML, f"{RUN}/pc_scratch", max_time=1200.0)
env = gym.make(**{**C.BASE_ENV_KWARGS, "settingxml": XML})
u = env.unwrapped

ic = f"{RUN}/ic.csv"
generate_initial_condition(
    csv_path=ic, mode="network_field",
    x_min=u.x_min * 0.9, x_max=u.x_max * 0.9,
    y_min=u.y_min * 0.9, y_max=u.y_max * 0.9,
    params=C.make_params(**C.PAPER_COUNTS), seed=0,
)
C.point_xml_at_ic(tree, root, XML, ic, seed=0)
env.reset(seed=-1)
u.time_simulation = -1


def hill(s):
    return MAXPRO * (s ** HILL) / (s ** HILL + HALF ** HILL)


print(f"{'step':>4} {'TM@mac med':>11} {'TM@mac max':>11} {'proSec%max':>11} "
      f"{'frac vox pro>anti':>18} {'frac mac pro>anti':>18} {'anti_max':>9}")

for t in range(0, 13):
    if t:
        env.step(C.NOOP_ACTION)
    tm = np.asarray(physicell.get_microenv("tumor_molecule"))
    pro = np.asarray(physicell.get_microenv("pro_tumoral_factor"))
    anti = np.asarray(physicell.get_microenv("anti_tumoral_factor"))
    mac = u.df_alive[u.df_alive["type"] == "macrophage"]

    kd = cKDTree(tm[:, :2])
    _, idx = kd.query(mac[["x", "y"]].to_numpy())
    tm_at_mac = tm[:, -1][idx]

    frac_vox = float((pro[:, -1] > anti[:, -1]).mean())
    frac_mac = float((pro[:, -1][idx] > anti[:, -1][idx]).mean())
    resp = hill(np.median(tm_at_mac)) / MAXPRO * 100

    print(f"{t:>4} {np.median(tm_at_mac):>11.4f} {tm_at_mac.max():>11.4f} "
          f"{resp:>10.1f}% {frac_vox:>18.3f} {frac_mac:>18.3f} "
          f"{anti[:, -1].max():>9.5f}")

env.close()
