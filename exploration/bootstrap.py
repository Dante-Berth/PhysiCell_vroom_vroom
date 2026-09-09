"""Import this FIRST, before numpy/gymnasium/anything else.

Fixes a genuinely broken import environment. There is no .venv in this tree, and
every venv on this machine resolves `extending.physicell` to a .so built from
~/Documents/Git/PhysiCell -- a different model that has no add_local_substrate at
all -- and resolves `physigym` to that tree's python as well. Running against
those would silently exercise the wrong C++ model.

The fix is to prepend this tree's own build directory and physigym package, then
assert loudly that the right ones won.
"""

import ctypes
import ctypes.util
import os
import sys

# 1. libgomp must be global BEFORE the physicell .so is dlopen'd.
#    (mirrors custom_modules/physigym/physigym/envs/test_obs_modes.py:6-13)
_gomp = ctypes.util.find_library("gomp")
if _gomp:
    ctypes.CDLL(_gomp, mode=ctypes.RTLD_GLOBAL)

# 2. Mute PhysiCell's std::cout, keep every worker single-threaded (we get our
#    parallelism from running one process per task instead), never open a display.
os.environ.setdefault("PHYSIGYM_QUIET", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

PROJECT_ROOT = "/home/disc/a.bertin/Documents/Git/PhysiCell_vroom_vroom"
BUILD = os.path.join(
    PROJECT_ROOT, "custom_modules/extending/build/lib.linux-x86_64-cpython-310"
)
PGYM = os.path.join(PROJECT_ROOT, "custom_modules/physigym")
ENVS = os.path.join(PROJECT_ROOT, "custom_modules/physigym/physigym/envs")

# 3. cwd must be the project root: //cell_rules/folder is "./config" and the
#    initial-condition paths in the settings XML are all relative.
os.chdir(PROJECT_ROOT)

# 4. Prepend. BUILD first: `extending` is a namespace package in both trees, so
#    path order decides. `physigym` is a regular package here, so prepending PGYM
#    beats the .pth-injected copy from the other tree.
for _p in (PROJECT_ROOT, ENVS, PGYM, BUILD):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

# The C++ symbol that distinguishes this tree's model from the other one.
_MANGLED = (
    "_Z19add_local_substrateNSt7__cxx1112basic_stringIcSt11"
    "char_traitsIcESaIcEEEdddd"
)


def assert_correct_so():
    """Fail loudly rather than silently simulate the wrong model."""
    from extending import physicell

    path = os.path.realpath(physicell.__file__)
    if not path.startswith(os.path.realpath(BUILD)):
        raise RuntimeError(
            f"WRONG physicell .so loaded:\n  {path}\nexpected under\n  {BUILD}"
        )
    lib = ctypes.CDLL(path)  # already resident; this only bumps the refcount
    try:
        getattr(lib, _MANGLED)
    except AttributeError:
        raise RuntimeError(
            f"{path} has no add_local_substrate, so it was built from the wrong "
            f"tree. Rebuild from {PROJECT_ROOT}."
        )
    return path


def assert_correct_physigym():
    import physigym
    import physigym.envs.physicell_model as model

    for mod in (physigym, model):
        p = os.path.realpath(mod.__file__)
        if not p.startswith(os.path.realpath(PGYM)):
            raise RuntimeError(f"WRONG physigym: {p}\nexpected under {PGYM}")
    return physigym.__file__


def assert_all():
    so = assert_correct_so()
    pg = assert_correct_physigym()
    return so, pg
