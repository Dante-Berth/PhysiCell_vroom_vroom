"""The parameters that were chosen rather than measured, and how to set them.

§4.6 of the thesis asks whether the qualitative untreated regime is robust to the
parameters nobody fitted. Those live in two places:

  * config/cell_rules.csv   -- max response, half max and Hill exponent of each
                               behaviour rule
  * the settings XML        -- growth rate, diffusion coefficients, secretion rates

Each entry says how to write one scalar into a copy of those files. Ranges are
multiplicative around the shipped value unless stated, because none of these has
a measured value to anchor an additive range to.

The drug_1 rules are deliberately absent: this is the untreated analysis, so they
never fire (drug_1 is identically zero), and including them would put two
provably inert parameters into the screening.
"""

# cell_rules.csv columns:
#   cell_type, signal, direction, behaviour, max_response, half_max, hill, dead
RULE_COLS = {"max_response": 4, "half_max": 5, "hill": 6}


def rule_key(cell_type, signal, behaviour):
    return (cell_type, signal, behaviour)


# name -> (kind, target, shipped value, (low, high))
# kind "rule": target = (cell_type, signal, behaviour, column)
# kind "xml" : target = xpath
PARAMS = {
    # ── the growth engine ────────────────────────────────────────────────
    # The real proliferation driver: the tumour's cycle phase transition rate.
    "tumor_cycle_rate": (
        "xml",
        "//cell_definition[@name='tumor']//cycle//phase_transition_rates"
        "/rate[@start_index='0']",
        3.0e-4, (1.0e-4, 6.0e-4),
    ),
    # NEGATIVE CONTROL, deliberately kept in the screening.
    # //user_parameters/growth_rate is read at physicell_model.py:112 to build
    # lambda_dt for the REWARD normalisation only. No C++ code reads it, so it
    # cannot move the untreated dynamics. A screening that works must return
    # mu_star = 0 for it, and the first run did exactly that. It is retained as
    # a check that the method detects a genuinely inert parameter.
    "growth_rate_reward_only": (
        "xml", "//user_parameters/growth_rate", 2.99e-4, (1.0e-4, 6.0e-4),
    ),
    # Named explicitly by the sec:tme:sensitivity gap: it sets how far the
    # recruitment signal reaches, which is a different quantity from its
    # amplitude (the secretion rate below).
    "tumor_molecule_decay": (
        "xml",
        "//microenvironment_setup/variable[@name='tumor_molecule']"
        "/physical_parameter_set/decay_rate",
        0.1, (0.01, 1.0),
    ),
    "tumor_molecule_secretion": (
        "xml",
        "//cell_definition[@name='tumor']//secretion/substrate"
        "[@name='tumor_molecule']/secretion_rate",
        1.0, (0.25, 4.0),
    ),
    # ── the only killing pathway: T cell -> cytokine -> tumour apoptosis ──
    "cytokine_secretion": (
        "xml",
        "//cell_definition[@name='t_cell']//secretion/substrate"
        "[@name='cytokine']/secretion_rate",
        1.0, (0.25, 4.0),
    ),
    "cytokine_diffusion": (
        "xml",
        "//microenvironment_setup/variable[@name='cytokine']"
        "/physical_parameter_set/diffusion_coefficient",
        0.003, (0.0003, 3.0),  # 'touching distance' up to factor 1000 wider
    ),
    "apoptosis_max": (
        "rule", ("tumor", "cytokine", "apoptosis", "max_response"),
        1.0, (0.1, 10.0),
    ),
    "apoptosis_half_max": (
        "rule", ("tumor", "cytokine", "apoptosis", "half_max"),
        0.5, (0.05, 2.0),
    ),
    "apoptosis_hill": (
        "rule", ("tumor", "cytokine", "apoptosis", "hill"),
        4.0, (1.0, 8.0),
    ),
    # ── the immunosuppression arm ────────────────────────────────────────
    "pro_secretion_max": (
        "rule",
        ("macrophage", "tumor_molecule", "pro_tumoral_factor secretion",
         "max_response"),
        10.0, (1.0, 40.0),
    ),
    "pro_secretion_half_max": (
        "rule",
        ("macrophage", "tumor_molecule", "pro_tumoral_factor secretion",
         "half_max"),
        0.5, (0.05, 2.0),
    ),
}

NAMES = list(PARAMS)


def salib_problem():
    return {
        "num_vars": len(NAMES),
        "names": NAMES,
        "bounds": [list(PARAMS[n][3]) for n in NAMES],
    }


def defaults():
    return {n: PARAMS[n][2] for n in NAMES}


def write_rules(src, dst, values):
    """Copy cell_rules.csv, overriding the sampled rule parameters."""
    out = []
    with open(src) as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("#"):
                out.append(line)
                continue
            f = s.split(",")
            for name in NAMES:
                kind, target, _, _ = PARAMS[name]
                if kind != "rule" or name not in values:
                    continue
                ct, sig, beh, col = target
                if f[0] == ct and f[1] == sig and f[3] == beh:
                    f[RULE_COLS[col]] = repr(float(values[name]))
            out.append(",".join(f) + "\n")
    with open(dst, "w") as fh:
        fh.writelines(out)


def apply_xml(root, values):
    """Write the sampled XML parameters into an already-parsed settings tree."""
    for name in NAMES:
        kind, xpath, _, _ = PARAMS[name]
        if kind != "xml" or name not in values:
            continue
        hits = root.xpath(xpath)
        if not hits:
            raise KeyError(f"xpath matched nothing for {name}: {xpath}")
        hits[0].text = repr(float(values[name]))
