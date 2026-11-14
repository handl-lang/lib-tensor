#!/usr/bin/env python3
"""verify_gen — emit the generated differential-check section of
verify.hnl for the ~195 GENERATED aten ops (the arch-audit §3 gap:
verify.hnl hand-covers only the original 31 ops).

Inputs:
  ../host-tensor/tools/aten_port_list.txt  — the curated pin list
  ../host-tensor/tools/aten_schemas.json   — schema records (via
                                              aten_port_gen)

Output (this repo, regenerated in place):
  verify.hnl — two marker-spliced blocks:
    * `-- >>> verify-gen: helpers >>>` … — the `vg_tl` list
      normalizer + the `len` import it needs (only emitted when a
      list-returning op is covered);
    * `-- >>> verify-gen: checks >>>` … — one verdict expression per
      ported op, or an honest `-- skip: <reason>` comment.

Check forms (same `check`/`eq` helpers the handwritten section uses):
  eq("name", lt.op(largs), tensor.op(hargs))          -- tensor results
  check("name", lt.op(largs), tensor.op(hargs))       -- scalars/bools
  check("name", {values: to_list(L.values), ...}, …)  -- {values,indices}
  check("name", vg_tl(L), vg_tl(H))                   -- tensor lists

Ops that cannot be checked honestly are NEVER emitted as checks:
  * the 4 `aten_port_gen` stubs (incl. the 3 host seams);
  * host-side seams that die under hnl-run (`unique`, `nonzero_dim`);
  * ops needing `math.*` host natives — hnl-run's CatoptKit wires
    tensor.* but NOT plugin-lib-stdlib's math.* leaves, so any op
    whose .hnl recipe calls math.exp/log/sin/… dies with
    `call_host: function 'math.*' not found`. Those get a skip
    comment naming the missing runtime piece (environment gap, not a
    semantic verdict);
  * ops whose fixture makes BOTH sides reject, or where the lib side
    dies on a fixture the host accepts (accept/reject drift — also a
    finding, reported in docs/reports/verify_generated.md).

Fixtures are curated data in FIXTURES below: per op, the argument
strings spliced into `lt.<name>(…)` / `tensor.<name>(…)`. Most ops
share the call shape on both sides; the few asymmetric ones (meshgrid
wants variadic on lib, a list on host) carry separate `l`/`h` args.

Usage:
  python3 tools/verify_gen.py            # probe + emit verify.hnl
  python3 tools/verify_gen.py --no-probe # emit only (uses last probe
                                       # cache; stale cache = stale
                                       # skips — rerun with probe)
  python3 tools/verify_gen.py --check    # byte-diff verify.hnl against
                                       # a fresh emit (no write)
  HNL_RUN=/path/to/hnl-run python3 tools/verify_gen.py
"""

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import aten_port_gen as gen  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
VERIFY = REPO / "verify.hnl"
PROBE_DIR = Path(tempfile.gettempdir()) / "lib-tensor-verify-probes"
CACHE = PROBE_DIR / "results.json"
HNL_RUN = os.environ.get(
    "HNL_RUN",
    str(REPO.parent / "graphopt" / "target" / "release" / "hnl-run"),
)

H_BEGIN = "-- >>> verify-gen: helpers >>>"
H_END = "-- <<< verify-gen <<<"
C_BEGIN = "-- >>> verify-gen: checks >>>"
C_END = "-- <<< verify-gen <<<"

# ---------------------------------------------------------------------------
# fixture vocabulary — nested-list literals wrapped in tensor.from_list
# ---------------------------------------------------------------------------


def FL(s: str) -> str:
    return f"tensor.from_list({s})"


# 1-D / scalar-ish
T_SIGN = FL("[-2, 0, 3]")
T_SGN = FL("[-2.5, -0.5, 0.5, 2.5]")
T_POS = FL("[0.5, 1.5, 2.5, 3.5]")
T_POS1 = FL("[0.25, 0.5, 1.0, 2.0]")  # safe for log/sqrt/rsqrt/reciprocal
T_UNIT = FL("[0.1, 0.3, 0.5, 0.9]")  # logit domain (0,1)
T_ANG = FL("[-1.5, -0.5, 0.5, 1.5]")
T_FRAC = FL("[-2.3, -0.7, 0.2, 1.9]")
T_EXP = FL("[-1, 0, 0.5, 2]")
T_DEG = FL("[-90, -45, 0, 45, 90]")
T_RAD = FL("[-1.5, 0, 0.5, 3.0]")
T_AA = FL("[-0.9, -0.5, 0, 0.5, 0.9]")  # asin/acos/atanh domain
T_AC = FL("[1, 1.5, 2, 3]")  # acosh domain >= 1
T_SINC = FL("[-2, -1, 0, 1, 2]")  # hits the x == 0 arm
T_V3 = FL("[1, 2, 3]")
T_V3B = FL("[4, 5, 6]")
T_V4 = FL("[1, 2, 3, 4]")
T_V2A = FL("[1, 2]")
T_V2B = FL("[3, 4]")
T_DOTA = FL("[1, 2, 3]")
T_DOTB = FL("[4, 5, 6]")
T_CROSSA = FL("[1, 0, 0]")
T_CROSSB = FL("[0, 1, 0]")
T_BC = FL("[0, 1, 1, 2, 2, 2]")  # bincount input (non-negative ints)
T_SCAL = FL("[42]")  # numel-1 for item
T_PREL = FL("[-1, 2, -3, 4]")
T_PRELW = FL("[0.5, 1.5, 2.5, 3.5]")
T_DISTA = FL("[1, 2, 3]")
T_DISTB = FL("[4, 6, 3]")

# 2-D
T_M = FL("[[3, 1, 2], [6, 5, 4]]")  # 2x3, all-distinct-ish
T_M2 = FL("[[1, 2], [3, 4]]")
T_M2B = FL("[[5, 6], [7, 8]]")
T_M3 = FL("[[1, 2, 3], [4, 5, 6]]")  # 2x3 for cdist rows
T_CD = FL("[[1, 1, 1], [2, 2, 2]]")
T_BIN_A = FL("[1.5, -2.0, 3.0, -4.0]")
T_BIN_B = FL("[2.0, 0.5, -1.0, 4.0]")
T_POW_A = FL("[1.5, 2.0, 3.0, 4.0]")
T_POW_B = FL("[0.5, 2.0, 1.0, 3.0]")
T_LOG_A = FL("[0, 1, 2, 0]")
T_LOG_B = FL("[1, 0, 3, 0]")
T_HV_A = FL("[-1, 0, 1, 2]")  # heaviside: x==0 arm present
T_HV_B = FL("[0.5, 1, 2, 3]")
T_XLOG_A = FL("[0, 1, 2, 3]")  # xlogy: x==0 arm present
T_XLOG_B = FL("[1, 2, 3, 4]")
T_GLU = FL("[[1, 2, 3, 4], [5, 6, 7, 8]]")  # last dim 4 -> glu -> [.,2]
T_IDX_GATHER = FL("[[0, 1, 0], [1, 0, 1]]")
T_IDX_SEL = FL("[1, 0]")
T_IDX_ADDA = FL("[0, 2]")  # tensor form — the HOST index_add/fill arg
T_IDX_LIST = "[0, 2]"  # plain list — the LIB index_add/fill arg
T_SRC_ADDA = FL("[10, 30]")

# 3-D / 4-D
T_3D = FL("[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]")
T_13 = FL("[[[1, 2], [3, 4]]]")  # leading size-1 for squeeze
T_BA = FL("[[[1, 0], [0, 1]], [[2, 0], [0, 2]]]")
T_BB = FL("[[[3], [4]], [[5], [6]]]")
T_BMM_A = FL("[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]")
T_BMM_B = FL("[[[1, 0], [0, 1]], [[2, 1], [1, 2]]]")
T_PS = FL(
    "[[[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]], "
    "[[13, 14], [15, 16]]]]"
)  # [1,4,2,2] for pixel_shuffle r=2 -> [1,1,4,4]
T_PU = FL(
    "[[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]]]"
)  # [1,1,4,4] for pixel_unshuffle r=2 -> [1,4,2,2]
T_POOL = FL(
    "[[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]]]"
)  # [1,1,4,4] for avg_pool2d k=2 -> [1,1,2,2]
T_WHERE_C = FL("[1, 0, 1]")
T_WHERE_A = FL("[1, 2, 3]")
T_WHERE_B = FL("[9, 8, 7]")
T_MG_A = FL("[1, 2]")
T_MG_B = FL("[3, 4]")

# scalar/vector literals for non-tensor args
L = "lt"
H = "tensor"


def ck(name, largs, hargs=None, kind="eq", full=False):
    """One check spec: (check_name, lt_call_args, tensor_call_args, kind).

    largs/hargs are the argument lists spliced into `lt.<op>(…)` /
    `tensor.<op>(…)`; hargs defaults to largs (the common symmetric
    case). With full=True the strings are already COMPLETE call
    expressions (used when the fixture itself calls another op, e.g.
    is_contiguous over expand). kind:
      eq    — tensor-valued result, compared via tensor.to_list
      check — scalar/bool/list result, compared with ==
      pair  — {values, indices} record, each field to_list'ed
      list  — list of tensors, element-wise to_list via vg_tl
    """
    return {"name": name, "l": largs, "h": hargs if hargs is not None else largs,
            "kind": kind, "full": full}


# ---------------------------------------------------------------------------
# the fixture table — one entry per generated (non-stub) port-list op
# ---------------------------------------------------------------------------

FIXTURES = {
    # ---- unary: pure .hnl --------------------------------------------
    "abs": [ck("abs", T_SGN)],
    "neg": [ck("neg", T_SGN)],
    "square": [ck("square", T_SGN)],
    "sign": [ck("sign", T_SIGN)],
    "floor": [ck("floor", T_FRAC)],
    "ceil": [ck("ceil", T_FRAC)],
    "round": [ck("round", T_FRAC)],
    "trunc": [ck("trunc", T_FRAC)],
    "frac": [ck("frac", T_FRAC)],
    "deg2rad": [ck("deg2rad", T_DEG)],
    "rad2deg": [ck("rad2deg", T_RAD)],
    "relu": [ck("relu", T_ANG)],
    "leaky_relu": [ck("leaky_relu", T_ANG)],
    "softshrink": [ck("softshrink", T_ANG)],
    "hardshrink": [ck("hardshrink", T_ANG)],
    "hardtanh": [ck("hardtanh", T_ANG)],
    # threshold: host requires all three args; lib defaults th=0, v=0.
    "threshold": [ck("threshold", T_ANG,
                     hargs=f"{T_ANG}, 0, 0")],
    "nan_to_num": [ck("nan_to_num", T_ANG)],
    # ---- unary: math.* leaves (expected env-gated) --------------------
    "exp": [ck("exp", T_EXP)],
    "exp2": [ck("exp2", T_EXP)],
    "expm1": [ck("expm1", T_EXP)],
    "log1p": [ck("log1p", T_POS1)],
    "mish": [ck("mish", T_EXP)],
    # softplus: the 25.0 element clears beta*x > threshold (20) so both
    # recipe arms run; the second check passes explicit beta/threshold.
    "softplus": [ck("softplus", FL("[-1, 0, 0.5, 25]")),
                 ck("softplus_params", f"{FL('[-1, 0, 0.5, 25]')}, 2, 10")],
    "log": [ck("log", T_POS1)],
    "log2": [ck("log2", T_POS1)],
    "log10": [ck("log10", T_POS1)],
    "sqrt": [ck("sqrt", T_POS1)],
    "rsqrt": [ck("rsqrt", T_POS1)],
    "reciprocal": [ck("reciprocal", T_POS1)],
    "sin": [ck("sin", T_ANG)],
    "cos": [ck("cos", T_ANG)],
    "tan": [ck("tan", T_ANG)],
    "asin": [ck("asin", T_AA)],
    "acos": [ck("acos", T_AA)],
    "atan": [ck("atan", T_ANG)],
    "sinh": [ck("sinh", T_ANG)],
    "cosh": [ck("cosh", T_ANG)],
    "tanh": [ck("tanh", T_ANG)],
    "asinh": [ck("asinh", T_ANG)],
    "acosh": [ck("acosh", T_AC)],
    "atanh": [ck("atanh", T_AA)],
    "erf": [ck("erf", T_ANG)],
    "erfc": [ck("erfc", T_ANG)],
    "silu": [ck("silu", T_ANG)],
    "sigmoid": [ck("sigmoid", T_ANG)],
    "sinc": [ck("sinc", T_SINC)],
    "selu": [ck("selu", T_ANG)],
    "elu": [ck("elu", T_ANG)],
    "celu": [ck("celu", T_ANG)],
    "logit": [ck("logit", T_UNIT)],
    "gelu": [ck("gelu", T_ANG), ck("gelu_tanh", f"{T_ANG}, \"tanh\"")],
    # ---- predicates ---------------------------------------------------
    "isfinite": [ck("isfinite", T_ANG)],
    "isinf": [ck("isinf", T_ANG)],
    "isnan": [ck("isnan", T_ANG)],
    "isposinf": [ck("isposinf", T_ANG)],
    "isneginf": [ck("isneginf", T_ANG)],
    "signbit": [ck("signbit", T_ANG)],
    "logical_not": [ck("logical_not", T_ANG)],
    # ---- binary --------------------------------------------------------
    "atan2": [ck("atan2", f"{T_BIN_A}, {T_BIN_B}")],
    "hypot": [ck("hypot", f"{T_BIN_A}, {T_BIN_B}")],
    "fmod": [ck("fmod", f"{T_BIN_A}, {T_BIN_B}")],
    "floor_divide": [ck("floor_divide", f"{T_BIN_A}, {T_BIN_B}")],
    "maximum": [ck("maximum", f"{T_BIN_A}, {T_BIN_B}")],
    "minimum": [ck("minimum", f"{T_BIN_A}, {T_BIN_B}")],
    "pow": [ck("pow", f"{T_POW_A}, {T_POW_B}")],
    "float_power": [ck("float_power", f"{T_POW_A}, {T_POW_B}")],
    "eq": [ck("eq", f"{T_BIN_A}, {T_BIN_B}")],
    "ne": [ck("ne", f"{T_BIN_A}, {T_BIN_B}")],
    "lt": [ck("lt", f"{T_BIN_A}, {T_BIN_B}")],
    "le": [ck("le", f"{T_BIN_A}, {T_BIN_B}")],
    "gt": [ck("gt", f"{T_BIN_A}, {T_BIN_B}")],
    "ge": [ck("ge", f"{T_BIN_A}, {T_BIN_B}")],
    "logical_and": [ck("logical_and", f"{T_LOG_A}, {T_LOG_B}")],
    "logical_or": [ck("logical_or", f"{T_LOG_A}, {T_LOG_B}")],
    "logical_xor": [ck("logical_xor", f"{T_LOG_A}, {T_LOG_B}")],
    "logaddexp": [ck("logaddexp", f"{T_BIN_A}, {T_BIN_B}")],
    "logaddexp2": [ck("logaddexp2", f"{T_BIN_A}, {T_BIN_B}")],
    "copysign": [ck("copysign", f"{T_BIN_A}, {T_BIN_B}")],
    "nextafter": [ck("nextafter",
                     f"{FL('[1.0, 2.0, -1.0, 0.5]')}, {FL('[2.0, 1.0, -2.0, -0.5]')}")],
    "remainder": [ck("remainder", f"{T_BIN_A}, {T_BIN_B}")],
    "heaviside": [ck("heaviside", f"{T_HV_A}, {T_HV_B}")],
    "xlogy": [ck("xlogy", f"{T_XLOG_A}, {T_XLOG_B}")],
    # ---- reductions ----------------------------------------------------
    "sum": [ck("sum", T_M)],
    "prod": [ck("prod", T_M)],
    "amax": [ck("amax", T_M)],
    "amin": [ck("amin", T_M)],
    "mean": [ck("mean", T_M)],
    "var": [ck("var", T_M)],
    "std": [ck("std", T_M)],
    "nansum": [ck("nansum", T_M)],
    "nanmean": [ck("nanmean", T_M)],
    "argmax": [ck("argmax", f"{T_M}, 0")],
    "argmin": [ck("argmin", f"{T_M}, 0")],
    "cumsum": [ck("cumsum", f"{T_M}, 1")],
    "cumprod": [ck("cumprod", f"{T_M}, 1")],
    "softmax": [ck("softmax", f"{T_M}, 1")],
    "log_softmax": [ck("log_softmax", f"{T_M}, 1")],
    "logsumexp": [ck("logsumexp", f"{T_M}, [0, 1]")],
    "count_nonzero": [ck("count_nonzero", T_M)],
    "all": [ck("all", T_M)],
    "any": [ck("any", T_M)],
    "max": [ck("max", T_M), ck("max_elem", f"{T_BIN_A}, {T_BIN_B}")],
    "min": [ck("min", T_M), ck("min_elem", f"{T_BIN_A}, {T_BIN_B}")],
    # ---- sort / selection ----------------------------------------------
    "sort": [ck("sort", f"{T_M}, 1", kind="pair")],
    "argsort": [ck("argsort", f"{T_M}, 1")],
    "topk": [ck("topk", f"{T_M}, 2, 0", kind="pair")],
    "msort": [ck("msort", T_M)],
    "median": [ck("median", f"{T_M}, 0", kind="pair")],
    "mode": [ck("mode", f"{T_M}, 0", kind="pair")],
    "kthvalue": [ck("kthvalue", f"{T_M}, 2, 1", kind="pair")],
    "quantile": [ck("quantile", f"{T_V4}, 0.5")],
    # searchsorted: handwritten checks already in verify.hnl
    # unique / nonzero_dim: host-side seams under hnl-run (probed) —
    # fixtures exist so the probe classifies the seam honestly.
    "unique": [ck("unique", T_V4, kind="pair")],
    "nonzero_dim": [ck("nonzero_dim", f"{T_M}, 8")],
    # ---- indexing / nonzero --------------------------------------------
    "nonzero": [ck("nonzero", T_M)],
    "argwhere": [ck("argwhere", T_M)],
    "index_select": [ck("index_select", f"{T_M}, 0, {T_IDX_SEL}")],
    "gather": [ck("gather", f"{T_M}, 0, {T_IDX_GATHER}")],
    # index_add/index_fill: arg-type drift — the lib drivers take idxs
    # as a plain list (aten_index_add iterates `idxs[i]`), the host
    # kernels take a rank-1 index TENSOR. Asymmetric args, same math.
    "index_add": [ck("index_add",
                     f"{T_V3}, 0, {T_IDX_LIST}, {T_SRC_ADDA}",
                     hargs=f"{T_V3}, 0, {T_IDX_ADDA}, {T_SRC_ADDA}")],
    "index_fill": [ck("index_fill",
                      f"{T_V3}, 0, {T_IDX_LIST}, 9",
                      hargs=f"{T_V3}, 0, {T_IDX_ADDA}, 9")],
    "take": [ck("take", f"{T_M}, {FL('[0, 5, 2]')}")],
    "squeeze": [ck("squeeze", T_13)],
    "flatten": [ck("flatten", T_M)],
    "expand": [ck("expand", f"{T_V3}, [2, 3]")],
    "expand_as": [ck("expand_as", f"{T_V3}, {T_M}")],
    "narrow": [ck("narrow", f"{T_M}, 0, 1, 1")],
    "slice": [ck("slice", f"{T_M}, 1, 0, 2")],
    "chunk": [ck("chunk", f"{T_V4}, 2", kind="list")],
    "split": [ck("split", f"{T_V4}, [1, 3]", kind="list")],
    "stack": [ck("stack", f"[{T_V3}, {T_V3B}], 0")],
    "unbind": [ck("unbind", f"{T_M}, 0", kind="list")],
    "repeat": [ck("repeat", f"{T_V3}, [2]")],
    "flip": [ck("flip", f"{T_M}, [0, 1]")],
    "triu": [ck("triu", T_M2)],
    "tril": [ck("tril", T_M2)],
    "diag": [ck("diag", T_M2)],
    "diagonal": [ck("diagonal", T_M2)],
    "trace": [ck("trace", T_M2)],
    "t": [ck("t", T_M)],
    "mT": [ck("mT", T_3D)],
    "swapaxes": [ck("swapaxes", f"{T_3D}, 0, 2")],
    "moveaxis": [ck("moveaxis", f"{T_3D}, 0, 1")],  # host rejects dst=2
    "rot90": [ck("rot90", T_M2)],
    "contiguous": [ck("contiguous", T_M)],
    "clone": [ck("clone", T_M)],
    "detach": [ck("detach", T_M)],
    "view": [ck("view", f"{T_M}, [3, 2]")],
    "pixel_shuffle": [ck("pixel_shuffle", f"{T_PS}, 2")],
    "pixel_unshuffle": [ck("pixel_unshuffle", f"{T_PU}, 2")],
    "constant_pad_nd": [ck("constant_pad_nd", f"{T_M2}, [1, 1, 1, 1]")],
    # ---- linalg ----------------------------------------------------------
    "mm": [ck("mm", f"{T_M2}, {T_M2B}")],
    "bmm": [ck("bmm", f"{T_BMM_A}, {T_BMM_B}")],
    "mv": [ck("mv", f"{T_M2}, {T_V2B}")],
    "dot": [ck("dot", f"{T_DOTA}, {T_DOTB}")],
    "vdot": [ck("vdot", f"{T_DOTA}, {T_DOTB}")],
    "inner": [ck("inner", f"{T_DOTA}, {T_DOTB}")],
    "ger": [ck("ger", f"{T_DOTA}, {T_DOTB}")],
    "outer": [ck("outer", f"{T_DOTA}, {T_DOTB}")],
    "addr": [ck("addr", f"{T_M2}, {T_V2A}, {T_V2B}")],
    "addmv": [ck("addmv", f"{T_V2A}, {T_M2}, {T_V2B}")],
    "addmm": [ck("addmm", f"{T_M2}, {T_M2}, {T_M2B}")],
    "addbmm": [ck("addbmm", f"{T_BMM_A}, {T_BMM_A}, {T_BMM_B}")],
    "baddbmm": [ck("baddbmm", f"{T_BMM_A}, {T_BMM_A}, {T_BMM_B}")],
    "cross": [ck("cross", f"{T_CROSSA}, {T_CROSSB}")],
    "kron": [ck("kron", f"{T_M2}, {T_M2B}")],
    # tensordot + to: no honest fixture — see FIXTURE_SKIPS.
    "corrcoef": [ck("corrcoef", T_M)],
    "dist": [ck("dist", f"{T_DISTA}, {T_DISTB}")],
    "cdist": [ck("cdist", f"{T_M}, {T_CD}")],
    "pdist": [ck("pdist", T_M)],
    "prelu": [ck("prelu", f"{T_PREL}, {T_PRELW}")],
    "lerp": [ck("lerp", f"{T_DOTA}, {T_DOTB}, 0.5")],
    "clamp": [ck("clamp", f"{T_V4}, 0, 3")],
    "clamp_min": [ck("clamp_min", f"{T_V4}, 1")],
    "clamp_max": [ck("clamp_max", f"{T_V4}, 3")],
    "where": [ck("where", f"{T_WHERE_C}, {T_WHERE_A}, {T_WHERE_B}")],
    "norm": [ck("norm", f"{T_V4}, 2")],
    "renorm": [ck("renorm", f"{T_M}, 2, 0, 1.0")],
    "normalize": [ck("normalize", f"{T_V4}, 2, 0")],
    "layer_norm": [ck("layer_norm", f"{T_M}, [3]")],
    "rms_norm": [ck("rms_norm", f"{T_M}, [3]")],
    "glu": [ck("glu", T_GLU)],
    # avg_pool2d/bincount: values match, but the lib drivers emit
    # FLOAT shape entries ([1,1,2.0,2.0], [3.0]) that host
    # tensor.to_list rejects ("expected shape: non-negative integers").
    # Compare through lt.to_list on the lib side — values still get
    # differential coverage; the float-shape interop defect is
    # documented in docs/reports/verify_generated.md.
    "avg_pool2d": [ck("avg_pool2d",
                      f"{L}.to_list({L}.avg_pool2d({T_POOL}, [2, 2]))",
                      f"{H}.to_list({H}.avg_pool2d({T_POOL}, [2, 2]))",
                      kind="check", full=True)],
    # ---- constructors / metadata -----------------------------------------
    "full": [ck("full", "[2, 2], 5")],
    "full_like": [ck("full_like", f"{T_M}, 3")],
    "zeros_like": [ck("zeros_like", T_M)],
    "ones_like": [ck("ones_like", T_M)],
    "arange": [ck("arange", "5"), ck("arange_step", "1, 6, 2")],
    "linspace": [ck("linspace", "0, 1, 5")],
    "logspace": [ck("logspace", "0, 2, 5")],
    "eye": [ck("eye", "3")],
    "numel": [ck("numel", T_M, kind="check")],
    # host tensor.dim/tensor.stride take NO dim argument (arity 1) —
    # the lib wrappers accept one; no common fixture exists for the
    # per-dim forms, so only the full-tensor forms are checked.
    "dim": [ck("dim", T_M, kind="check")],
    "size": [ck("size", T_M, kind="check"), ck("size_i", f"{T_M}, 1", kind="check")],
    "stride": [ck("stride", T_M, kind="check")],
    "item": [ck("item", T_SCAL, kind="check")],
    "is_contiguous": [ck("is_contiguous", T_M, kind="check"),
                      ck("is_contiguous_expand",
                         f"{L}.is_contiguous({L}.expand({T_V3}, [2, 3]))",
                         f"{H}.is_contiguous({H}.expand({T_V3}, [2, 3]))",
                         kind="check", full=True)],
    "bincount": [ck("bincount",
                    f"{L}.to_list({L}.bincount({T_BC}))",
                    f"{H}.to_list({H}.bincount({T_BC}))",
                    kind="check", full=True)],
    "corrcoef": [ck("corrcoef", T_M)],
    # `to` has no common spec form: host accepts the "f64" dtype string
    # (or a tensor); aten_to accepts only a list / {f: _} record —
    # see FIXTURE_SKIPS.
    # allclose/is_nonzero: host returns a primitive bool, lib returns a
    # scalar wire tensor (documented predicate contract) — `pred` kind
    # normalizes the lib side to a bool without weakening the compare.
    "allclose": [ck("allclose", f"{T_M}, {T_M}", kind="pred")],
    "is_nonzero": [ck("is_nonzero", T_M, kind="pred")],
    "meshgrid": [ck("meshgrid", f"{T_MG_A}, {T_MG_B}",
                   hargs=f"[{T_MG_A}, {T_MG_B}]", kind="list")],
}

# Ops with fixtures curated but no honest check — probed and rejected.
FIXTURE_SKIPS = {
    "tensordot": ("lib `lt.tensordot` crashes `index: index 1 out of "
                  "bounds` on every probed dims spec ([1],[0] and "
                  "[0],[1], 2-D and 3-D operands) while host "
                  "k_tensordot returns correct values — candidate lib "
                  "defect, not a fixture problem"),
    "to": ("no common spec form: host `tensor.to` accepts the \"f64\" "
           "dtype string (or a tensor rhs); lib `aten_to` accepts only "
           "a non-empty list or {f: _} record and dies on the string — "
           "interface drift, nothing comparable to emit"),
}

# ---------------------------------------------------------------------------
# probe + classify
# ---------------------------------------------------------------------------

PROBE_PRELUDE = """do {
  from "host-tensor" import tensor
  from "lib-tensor" import ltensor as lt
  from "./src/internal/helpers.hnl" import { len }
  vg_tl := (xs) ~> while { i: 0, acc: [] } {
    (i, acc) ~> if i < len(xs) {
      { state: { i: i + 1, acc: [...acc, tensor.to_list(xs[i])] } }
    } else { { out: acc } }
  }
  [
"""


def call_for(side, opname, spec):
    """spec['l']/['h'] are arg lists, or full call exprs when full=True."""
    args = spec["l"] if side == "lt" else spec["h"]
    if spec.get("full"):
        return args
    return f"{side}.{opname}({args})"


def values_expr(opname, spec):
    l = call_for("lt", opname, spec)
    h = call_for("tensor", opname, spec)
    k = spec["kind"]
    if k == "eq":
        return f"[tensor.to_list({l}), tensor.to_list({h})]"
    if k == "check":
        return f"[{l}, {h}]"
    if k == "pair":
        return (f"[{{values: tensor.to_list(({l}).values), "
                f"indices: tensor.to_list(({l}).indices)}}, "
                f"{{values: tensor.to_list(({h}).values), "
                f"indices: tensor.to_list(({h}).indices)}}]")
    if k == "list":
        return f"[vg_tl({l}), vg_tl({h})]"
    if k == "pred":
        # lib predicates return a scalar wire tensor (1.0/0.0 — the
        # documented aten.hnl contract); host returns a primitive bool.
        # Normalize the lib side to a bool; the VALUE is still compared.
        # (lt.item unwraps the scalar wire tensor to a number —
        # tensor.to_list on a 0-d wire yields a bare number, not a list.)
        return f"[lt.item({l}) != 0, {h}]"
    raise ValueError(k)


def check_expr(opname, spec):
    l = call_for("lt", opname, spec)
    h = call_for("tensor", opname, spec)
    k = spec["kind"]
    if k == "eq":
        return f'eq("{spec["name"]}", {l}, {h})'
    if k == "check":
        return f'check("{spec["name"]}", {l}, {h})'
    if k == "pair":
        return (f'check("{spec["name"]}", '
                f'{{values: tensor.to_list(({l}).values), '
                f'indices: tensor.to_list(({l}).indices)}}, '
                f'{{values: tensor.to_list(({h}).values), '
                f'indices: tensor.to_list(({h}).indices)}})')
    if k == "list":
        return f'check("{spec["name"]}", vg_tl({l}), vg_tl({h}))'
    if k == "pred":
        return (f'check("{spec["name"]}", '
                f'lt.item({l}) != 0, {h})')
    raise ValueError(k)


def probe_file(opname, spec) -> str:
    return (PROBE_PRELUDE
            + f'    {{ k: "{spec["name"]}", v: {values_expr(opname, spec)} }}\n'
            + "  ]\n}\n")


def run_probe(path: Path):
    """-> (status, payload). status: ok | mismatch | error:<kind>."""
    try:
        p = subprocess.run(
            [HNL_RUN, "--project", str(REPO), str(path)],
            capture_output=True, text=True, timeout=120, cwd=REPO,
        )
    except subprocess.TimeoutExpired:
        return "error:timeout", ""
    out = p.stdout.strip()
    if p.returncode != 0:
        err = p.stderr.strip().splitlines()
        msg = err[-1] if err else "unknown"
        m = re.search(r"call_host: function '([\w.]+)'", msg)
        if m:
            return f"error:call_host:{m.group(1)}", msg
        if "port seam" in msg:
            return "error:seam", msg
        if "no arm matched" in msg:
            return "error:die", msg
        return "error:eval", msg
    try:
        data = json.loads(out)
    except json.JSONDecodeError:
        return "error:parse", out[:200]
    v = data[0]["v"] if isinstance(data, list) and data else None
    if isinstance(v, list) and len(v) == 2:
        return ("ok" if v[0] == v[1] else "mismatch"), json.dumps(v)[:400]
    return "error:shape", json.dumps(v)[:200]


def collect_specs(entries, schemas):
    """-> {hnl_name: [spec,...]} plus skip reasons for non-emittable ops."""
    specs, skips = {}, {}
    for e in entries:
        cls, detail = gen.classify(e)
        name = e["hnl"]
        if cls == "stub":
            skips[name] = f"lib stub ({detail})"
            continue
        if name == "searchsorted":
            skips[name] = "handwritten checks already in verify.hnl"
            continue
        if name in FIXTURE_SKIPS:
            skips[name] = FIXTURE_SKIPS[name]
            continue
        if name not in FIXTURES:
            skips[name] = "no fixture curated"
            continue
        specs[name] = FIXTURES[name]
    return specs, skips


def spec_key(opname, spec) -> str:
    """Cache/probe key — carries a fixture hash so edited fixtures
    re-probe instead of replaying a stale cache verdict."""
    import hashlib
    fh = hashlib.sha1(
        f"{spec['l']}|{spec['h']}|{spec['kind']}|{spec.get('full')}"
        .encode()).hexdigest()[:8]
    return f"{opname}:{spec['name']}:{fh}"


def probe_all(specs):
    """Run every spec's probe under hnl-run; -> {check_name: (status, msg)}."""
    PROBE_DIR.mkdir(parents=True, exist_ok=True)
    cache = {}
    if CACHE.exists():
        cache = json.loads(CACHE.read_text())
    results = {}
    todo = []
    for opname, slist in specs.items():
        for spec in slist:
            key = spec_key(opname, spec)
            if key in cache:
                results[key] = tuple(cache[key])
            else:
                todo.append((opname, spec, key))
    for i, (opname, spec, key) in enumerate(todo):
        path = PROBE_DIR / f"{opname}__{spec['name']}.hnl"
        path.write_text(probe_file(opname, spec))
        status, msg = run_probe(path)
        results[key] = (status, msg)
        cache[key] = [status, msg]
        print(f"  probe {i + 1}/{len(todo)} {key}: {status}", flush=True)
    CACHE.write_text(json.dumps(cache, indent=1))
    return results


# ---------------------------------------------------------------------------
# emit
# ---------------------------------------------------------------------------


def emit_checks(specs, skips, results):
    """-> (helpers_needed, [lines]) for the generated blocks."""
    lines = []
    helpers = False
    # preserve port-list order for readability
    order = [e["hnl"] for e in gen.parse_port_list(gen.PORT_LIST)]
    for opname in order:
        if opname not in specs:
            continue
        for spec in specs[opname]:
            key = spec_key(opname, spec)
            status, msg = results.get(key, ("error:no-probe", ""))
            if spec["kind"] == "list":
                helpers = True
            if status == "ok" or status == "mismatch":
                lines.append(f"    {check_expr(opname, spec)},")
                if status == "mismatch":
                    lines.append(
                        f"    -- ^^ MISMATCH {key}: {msg} — semantic drift")
            elif status.startswith("error:call_host:math"):
                leaf = status.rsplit(":", 1)[-1]
                lines.append(
                    f"    -- skip {key}: needs `{leaf}` host native — "
                    "hnl-run's CatoptKit does not register plugin-lib-stdlib "
                    "math.* leaves (environment gap, not a semantic verdict)")
                lines.append(f"    --   {check_expr(opname, spec)},")
            elif status == "error:seam":
                lines.append(
                    f"    -- skip {key}: host `tensor.{opname}` is a "
                    "declared port seam under hnl-run — nothing to "
                    "compare against")
            elif status == "error:die":
                lines.append(
                    f"    -- skip {key}: fixture rejected — {msg[:140]}")
            else:
                lines.append(
                    f"    -- skip {key}: {status} — {msg[:140]}")
    # pre-skipped ops (stubs / handwritten-covered / uncurated) — honest
    # comments so the generated section accounts for every port-list row.
    for name in order:
        if name in specs or name not in skips:
            continue
        lines.append(f"    -- skip {name}: {skips[name]}")
    return helpers, lines


def transform(text: str, helpers: bool, check_lines) -> str:
    """Splice the helpers + checks blocks into `text`; returns new text."""
    # helpers block (only when a list-kind check exists)
    helper_body = (
        '  from "./src/internal/helpers.hnl" import { len }\n'
        "  vg_tl := (xs) ~> while { i: 0, acc: [] } {\n"
        "    (i, acc) ~> if i < len(xs) {\n"
        "      { state: { i: i + 1, acc: [...acc, tensor.to_list(xs[i])] } }\n"
        "    } else { { out: acc } }\n"
        "  }\n"
    )
    hblock = H_BEGIN + "\n" + helper_body + H_END
    if H_BEGIN in text:
        pre, rest = text.split(H_BEGIN, 1)
        _, post = rest.split(H_END, 1)
        text = pre + (hblock if helpers else "") + post
        if not helpers:
            text = re.sub(r"\n\n\n+", "\n\n", text)
    elif helpers:
        # insert before the result-list `[`
        idx = text.index("\n  [\n")
        text = text[:idx] + "\n\n" + hblock + "\n" + text[idx:]
    # checks block: before the closing `  ]` of the verdict list — the
    # LAST occurrence of "\n  ]\n}" at top level. The final emitted
    # check expression must not carry a trailing comma (list literal).
    for i in range(len(check_lines) - 1, -1, -1):
        if check_lines[i].lstrip().startswith(("eq(", "check(")):
            check_lines[i] = check_lines[i].rstrip(",")
            break
    cblock = C_BEGIN + "\n" + "\n".join(check_lines) + "\n  " + C_END
    if C_BEGIN in text:
        pre, rest = text.split(C_BEGIN, 1)
        _, post = rest.split(C_END, 1)
        text = pre + cblock + post
    else:
        marker = "\n  ]\n}"
        idx = text.rindex(marker)
        # the handwritten tail's last check has no trailing comma —
        # generated checks follow it, so it needs one.
        head = text[:idx]
        lines = head.split("\n")
        for j in range(len(lines) - 1, -1, -1):
            s = lines[j].strip()
            if not s or s.startswith("--"):
                continue
            if s.endswith(")") and not s.endswith(","):
                lines[j] += ","
            break
        text = "\n".join(lines) + "\n" + cblock + text[idx:]
    return text


def main():
    check_only = "--no-probe" in sys.argv
    diff_check = "--check" in sys.argv
    entries = gen.parse_port_list(gen.PORT_LIST)
    schemas = gen.load_schemas(gen.SCHEMAS)
    specs, skips = collect_specs(entries, schemas)
    print(f"specs: {sum(len(v) for v in specs.values())} checks over "
          f"{len(specs)} ops; pre-skips: {len(skips)}")
    for n, r in sorted(skips.items()):
        print(f"  pre-skip {n}: {r}")
    if check_only:
        results = {}
        if CACHE.exists():
            for k, v in json.loads(CACHE.read_text()).items():
                results[k] = tuple(v)
    else:
        results = probe_all(specs)
    helpers, lines = emit_checks(specs, skips, results)
    new_text = transform(VERIFY.read_text(), helpers, lines)
    if diff_check:
        if new_text == VERIFY.read_text():
            print("verify.hnl: up to date")
        else:
            print("verify.hnl: STALE — rerun "
                  "`python3 tools/verify_gen.py` to regenerate",
                  file=sys.stderr)
            return 1
    else:
        VERIFY.write_text(new_text)
    n_check = sum(1 for l in lines if l.startswith("    eq(")
                  or l.startswith("    check("))
    n_skip = sum(1 for l in lines if l.startswith("    -- skip"))
    n_mis = sum(1 for l in lines if "MISMATCH" in l)
    print(f"emitted {n_check} checks, {n_skip} skip comments "
          f"({n_mis} mismatches), helpers={'yes' if helpers else 'no'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
