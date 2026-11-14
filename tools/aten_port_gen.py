#!/usr/bin/env python3
"""aten_port_gen — emit lib-tensor `.hnl` op specs from the host-tensor
ATen port catalog.

Inputs (host-tensor is READ-ONLY reference):
  ../host-tensor/tools/aten_schemas.json   — torch._C._jit_get_all_schemas()
  ../host-tensor/tools/aten_port_list.txt  — the curated 207-pin list

Output (this repo, regenerated in place):
  src/aten/*.hnl        — generated op modules, one per family
  src/index.hnl         — a generated block spliced between markers

Design: every curated pin maps to a *recipe* calling the shared
drivers in src/internal/aten.hnl (the .hnl port of
host-tensor/src/ops/aten_kernels.rs). Ops with no honest .hnl recipe
emit `aten_stub`, which dies loudly — the same "declared seam"
contract the host keeps. The three remaining host seams (empty, rand,
normal) plus einsum are stubs by construction; `unique` and
`nonzero_dim` have handwritten drivers in `src/internal/aten.hnl`.

Usage:  python3 tools/aten_port_gen.py [--check]
"""

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
HOST = REPO.parent / "host-tensor"
SCHEMAS = HOST / "tools" / "aten_schemas.json"
PORT_LIST = HOST / "tools" / "aten_port_list.txt"
OUT_DIR = REPO / "src" / "aten"
INDEX = REPO / "src" / "index.hnl"

BEGIN = "-- >>> aten-port-gen: generated surface (do not edit) >>>"
END = "-- <<< aten-port-gen <<<"

HANDWRITTEN = {"reshape", "unsqueeze"}  # already in the 31-op surface

# ---------------------------------------------------------------------------
# port-list parsing
# ---------------------------------------------------------------------------

def parse_port_list(path: Path):
    entries = []
    for raw in path.read_text().splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        m = re.match(r"tensor\.(\w+)\s*=\s*aten::([\w.]+)(.*)", s)
        if not m:
            continue
        hnl, aten, rest = m.group(1), m.group(2), m.group(3)
        entries.append({
            "hnl": hnl, "aten": aten,
            "seam": "seam:" in s,
            "covers": "covers" in s,
        })
    return entries


def load_schemas(path: Path):
    data = json.loads(path.read_text())
    recs = data if isinstance(data, list) else data.get("schemas", [])
    by_name = {}
    for rec in recs:
        by_name.setdefault(rec["name"], []).append(rec)
    return by_name


def resolve_schema(schemas, aten: str):
    name = aten.split(".")[0]
    cands = schemas.get(name, [])
    if not cands:
        return None, "no schema"
    if "." in aten:
        ov = aten.split(".", 1)[1]
        for rec in cands:
            if rec.get("overload") == ov:
                return rec, None
        return cands[0], f"overload {ov} not found; using first"
    return cands[0], None


# ---------------------------------------------------------------------------
# recipes
# ---------------------------------------------------------------------------

MATH_UNARY = {
    "abs": "aten_abs(x)", "neg": "0 - x", "exp": "math.exp(x)",
    "log": "math.log(x)", "log10": "math.log10(x)", "log2": "math.log2(x)",
    "sqrt": "math.sqrt(x)", "rsqrt": "1 / math.sqrt(x)",
    "reciprocal": "1 / x", "square": "x * x",
    "sin": "math.sin(x)", "cos": "math.cos(x)", "tan": "math.tan(x)",
    "asin": "math.asin(x)", "acos": "math.acos(x)", "atan": "math.atan(x)",
    "sinh": "math.sinh(x)", "cosh": "math.cosh(x)", "tanh": "math.tanh(x)",
    "asinh": "math.asinh(x)", "acosh": "math.acosh(x)", "atanh": "math.atanh(x)",
    "floor": "aten_floor(x)", "ceil": "aten_ceil(x)", "round": "aten_round(x)",
    "trunc": "aten_trunc(x)", "frac": "aten_fract(x)",
    "deg2rad": "x * 3.141592653589793 / 180", "rad2deg": "x * 180 / 3.141592653589793",
    "erf": "aten_erf(x)", "erfc": "1 - aten_erf(x)",
    "silu": "x / (1 + math.exp(0 - x))",
    "sigmoid": "1 / (1 + math.exp(0 - x))",
    "exp2": "math.exp2(x)", "expm1": "math.expm1(x)", "log1p": "math.log1p(x)",
    "mish": "x * math.tanh(math.log1p(math.exp(x)))",
    "sinc": "if x == 0 { 1 } else { math.sin(3.141592653589793 * x) / (3.141592653589793 * x) }",
    "relu": "aten_fmax(x, 0)",
    "selu": "if x > 0 { 1.0507009873554805 * x } else { 1.0507009873554805 * 1.6732632423543772 * (math.exp(x) - 1) }",
    "sign": "if x > 0 { 1 } else { if x < 0 { 0 - 1 } else { x } }",
}

PRED = {
    "isfinite": "aten_is_finite(x)", "isinf": "aten_is_infinite(x)",
    "isnan": "aten_is_nan(x)",
    "isposinf": "aten_is_infinite(x) && x > 0",
    "isneginf": "aten_is_infinite(x) && x < 0",
    "signbit": "aten_sign(x) < 0",
    "logical_not": "x == 0",
}

BIN = {
    "atan2": "math.atan2(x, y)", "hypot": "math.hypot(x, y)",
    "fmod": "x % y",
    "floor_divide": "aten_floor(x / y)",
    "maximum": "aten_fmax(x, y)", "minimum": "aten_fmin(x, y)",
    "pow": "math.pow(x, y)", "float_power": "math.pow(x, y)",
    "eq": "if x == y { 1 } else { 0 }",
    "ne": "if x != y { 1 } else { 0 }",
    "lt": "if x < y { 1 } else { 0 }",
    "le": "if x <= y { 1 } else { 0 }",
    "gt": "if x > y { 1 } else { 0 }",
    "ge": "if x >= y { 1 } else { 0 }",
    "logical_and": "if (x != 0) && (y != 0) { 1 } else { 0 }",
    "logical_or": "if (x != 0) || (y != 0) { 1 } else { 0 }",
    "logical_xor": "if (x != 0) != (y != 0) { 1 } else { 0 }",
    "logaddexp": "do {\n  m :: aten_fmax(x, y)\n  m + math.log(math.exp(x - m) + math.exp(y - m))\n}",
    "logaddexp2": "do {\n  m :: aten_fmax(x, y)\n  m + math.log2(math.exp2(x - m) + math.exp2(y - m))\n}",
    "copysign": "math.copysign(x, y)",
    "nextafter": "math.nextafter(x, y)",
    "remainder": "do {\n  rr :: x % y\n  if rr != 0 && (rr < 0) != (y < 0) { rr + y } else { rr }\n}",
    "heaviside": "if x > 0 { 1 } else { if x < 0 { 0 } else { 0.5 * y } }",
    "xlogy": "if x == 0 { 0 } else { x * math.log(y) }",
}

STUBS = {
    "empty": "uninitialized fill cannot be represented on the wire",
    "rand": "RNG state is not part of the wire contract; nondeterminism breaks verification",
    "normal": "RNG state is not part of the wire contract",
    "einsum": "equation-string arg + general contraction engine — handwritten seam (host has a dedicated parser)",
}

# ---------------------------------------------------------------------------
# driver recipes: hnl → (driver, [ (param, optional, default) ... ], ret)
# ret: sd={s,d}->aten_ow, wire, pair, rec, list, meta
# ---------------------------------------------------------------------------

D = "aten_"
# shorthand for arg tuples: (name, has_default, default_expr)
ND = ("dims", True, None)
NK = ("keepdim", True, False)

DRIVER = {
    # reductions (custom _-drivers: dim-spec normalization + fold args)
    "sum": ("_reduce", [ND, NK], "sd", "0", "(a, b) ~> a + b"),
    "prod": ("_reduce", [ND, NK], "sd", "1", "(a, b) ~> a * b"),
    "amax": ("_reduce", [ND, NK], "sd", "0 - aten_f64_max", "(a, b) ~> aten_fmax(a, b)"),
    "amin": ("_reduce", [ND, NK], "sd", "aten_f64_max", "(a, b) ~> aten_fmin(a, b)"),
    "argmax": ("_argred", [("dim", True, None), NK], "sd", None, "true"),
    "argmin": ("_argred", [("dim", True, None), NK], "sd", None, "false"),
    "mean": ("aten_mean", [ND, NK], "sd", None, None),
    "var": ("aten_var_std", [ND, ("unbiased", True, True), NK], "sd", None, "false"),
    "std": ("aten_var_std", [ND, ("unbiased", True, True), NK], "sd", None, "true"),
    "logsumexp": ("aten_logsumexp", [("dims", False, None), NK], "sd", None, None),
    "cumsum": ("_cum", [("dim", False, None)], "sd", None, "false"),
    "cumprod": ("_cum", [("dim", False, None)], "sd", None, "true"),
    "softmax": ("_softmax", [("dim", False, None)], "sd", None, "false"),
    "log_softmax": ("_softmax", [("dim", False, None)], "sd", None, "true"),
    "count_nonzero": ("aten_count_nonzero", [("dim", True, None)], "wire", None, None),
    "nansum": ("aten_nansum", [ND, NK], "wire", None, None),
    "nanmean": ("aten_nanmean", [ND, NK], "wire", None, None),
    # sorting / selection
    "sort": ("_sort_pair", [("dim", False, None), ("_desc", False, "false")], "pair", None, None),
    "argsort": ("_argsort", [("dim", False, None), ("_desc", False, "false")], "pair", None, None),
    "topk": (D+"topk", [("k", False, None), ("dim", False, None), ("largest", True, "true")], "pairv", None, None),
    "msort": (D+"msort", [], "sd", None, None),
    "allclose": (D+"allclose", [("other", False, None), ("rtol", True, 0.00001), ("atol", True, 0.00000001), ("equal_nan", True, False)], "sd", None, None),
    "is_nonzero": (D+"is_nonzero", [], "sd", None, None),
    "median": (D+"median", [("dim", True, None)], "rec_or_sd", None, None),
    "mode": (D+"mode", [("dim", False, None), NK], "rec", None, None),
    "kthvalue": (D+"kthvalue", [("k", False, None), ("dim", True, 0), NK], "rec", None, None),
    "quantile": (D+"quantile", [("q", False, None)], "sd_rest", None, None),
    "nonzero": (D+"nonzero", [], "sd", None, None),
    "argwhere": (D+"nonzero", [], "sd", None, None),
    "unique": (D+"unique", [("sorted", True, True), ("return_inverse", True, False)], "rec", None, None),
    "nonzero_dim": (D+"nonzero_static", [("size", False, None), ("fill_value", True, -1)], "sd", None, None),
    # indexing / shape
    "index_select": (D+"index_select", [("dim", False, None), ("idxs", False, None)], "sd", None, None),
    "gather": (D+"gather", [("dim", False, None), ("idx", False, None)], "sd", None, None),
    "index_add": (D+"index_add", [("dim", False, None), ("idxs", False, None), ("src", False, None), ("alpha", True, "1")], "sd", None, None),
    "index_fill": (D+"index_fill", [("dim", False, None), ("idxs", False, None), ("value", False, None)], "sd", None, None),
    "take": (D+"take", [("idx", False, None)], "sd", None, None),
    "squeeze": (D+"squeeze", [ND], "sd", None, None),
    "flatten": (D+"flatten", [("start", True, 0), ("end", True, -1)], "sd", None, None),
    "expand": (D+"expand", [("shape", False, None)], "sd", None, None),
    "expand_as": (D+"expand_as", [("other", False, None)], "sd", None, None),
    "narrow": (D+"slice_dim", [("dim", False, None), ("start", False, None), ("_len", False, None)], "sd", None, None),
    "slice": (D+"slice", [("dim", False, None), ("start", True, None), ("end", True, None), ("step", True, 1)], "sd", None, None),
    "chunk": (D+"chunk", [("chunks", False, None), ("dim", True, 0)], "sdlist", None, None),
    "split": (D+"split", [("sizes", False, None), ("dim", True, 0)], "sdlist", None, None),
    "stack": (D+"stack", [("_list", False, None), ("dim", False, None)], "sd", None, None),
    "unbind": (D+"unbind", [("dim", False, None)], "sdlist", None, None),
    "repeat": (D+"repeat", [("reps", False, None)], "sd", None, None),
    "flip": (D+"flip", [ND], "sd", None, None),
    "triu": (D+"triu", [("diagonal", True, 0), ("_upper", False, True)], "sd", None, None),
    "tril": (D+"triu", [("diagonal", True, 0), ("_upper", False, False)], "sd", None, None),
    "diag": (D+"diag", [("diagonal", True, 0)], "sd", None, None),
    "diagonal": (D+"diagonal", [("offset", True, 0), ("dim1", True, 0), ("dim2", True, 1)], "sd", None, None),
    "trace": (D+"trace", [], "sd", None, None),
    "t": (D+"t", [], "wire", None, None),
    "mT": (D+"t", [], "wire", None, None),
    "swapaxes": (D+"swapaxes", [("d0", False, None), ("d1", False, None)], "sd", None, None),
    "moveaxis": (D+"moveaxis", [("src", False, None), ("dst", False, None)], "sd", None, None),
    "rot90": (D+"rot90", [("k", True, 1), ("dims", True, None)], "sd", None, None),
    "contiguous": (D+"id", [], "wire", None, None),
    "clone": (D+"id", [], "wire", None, None),
    "detach": (D+"id", [], "wire", None, None),
    "view": (D+"reshape_view", [("shape", False, None)], "sd", None, None),
    "pixel_shuffle": (D+"pixel_space_depth", [("r", False, None), ("_shuffle", False, True)], "sd", None, None),
    "pixel_unshuffle": (D+"pixel_space_depth", [("r", False, None), ("_shuffle", False, False)], "sd", None, None),
    "constant_pad_nd": (D+"constant_pad_nd", [("pad", False, None), ("value", True, 0)], "sd", None, None),
    # linalg
    "mm": (D+"mm2", [("other", False, None)], "sd", None, None),
    "bmm": (D+"bmm", [("other", False, None)], "sd", None, None),
    "mv": (D+"mv", [("vec", False, None)], "sd", None, None),
    "dot": (D+"dot_wrap", [("other", False, None)], "sd", None, None),
    "vdot": (D+"dot_wrap", [("other", False, None)], "sd", None, None),
    "inner": (D+"dot_wrap", [("other", False, None)], "sd", None, None),
    "ger": (D+"ger", [("other", False, None)], "sd", None, None),
    "outer": (D+"ger", [("other", False, None)], "sd", None, None),
    "addr": (D+"addr", [("vec1", False, None), ("vec2", False, None), ("beta", True, 1), ("alpha", True, 1)], "wire", None, None),
    "addmv": (D+"addmv", [("mat", False, None), ("vec", False, None), ("beta", True, 1), ("alpha", True, 1)], "sd", None, None),
    "addmm": (D+"addmm", [("mat1", False, None), ("mat2", False, None), ("beta", True, 1), ("alpha", True, 1), ("_batched", False, False)], "wire", None, None),
    "addbmm": (D+"addmm", [("mat1", False, None), ("mat2", False, None), ("beta", True, 1), ("alpha", True, 1), ("_batched", False, True)], "wire", None, None),
    "baddbmm": (D+"addmm", [("mat1", False, None), ("mat2", False, None), ("beta", True, 1), ("alpha", True, 1), ("_batched", False, True)], "wire", None, None),
    # constructors / metadata
    "full": (D+"full", [("shape", False, None), ("fill", True, 0)], "sd", None, None),
    "full_like": (D+"full_like", [("fill", True, 0)], "sd", None, None),
    "zeros_like": (D+"zeros_like", [], "sd", None, None),
    "ones_like": (D+"ones_like", [], "sd", None, None),
    "arange": ("_arange", [("_args", False, None)], "sd", None, None),
    "linspace": (D+"linspace", [("start", False, None), ("end", False, None), ("steps", False, None)], "sd_ls", None, None),
    "logspace": (D+"linspace", [("start", False, None), ("end", False, None), ("steps", False, None), ("base", True, "10")], "sd_ls_log", None, None),
    "eye": (D+"eye", [("n", False, None), ("m", True, "null")], "sd", None, None),
    "numel": ("_numel", [], "meta", None, None),
    "dim": ("_dim", [("_d", False, None)], "meta", None, None),
    "size": ("_size", [("_d", True, "null")], "meta", None, None),
    "stride": ("_stride", [("_d", True, "null")], "meta", None, None),
    "item": ("_item", [], "meta", None, None),
    "is_contiguous": ("_is_contig", [], "meta", None, None),
    # norms / activations-ish
    "layer_norm": (D+"layer_norm", [("shape", False, None), ("weight", True, "null"), ("bias", True, "null"), ("eps", True, 0.00001)], "wire", None, None),
    "rms_norm": (D+"rms_norm", [("shape", False, None), ("weight", True, "null"), ("eps", True, 0.00001)], "wire", None, None),
    "glu": (D+"glu", [("dim", True, -1)], "wire", None, None),
    "avg_pool2d": (D+"avg_pool2d", [("kernel", False, None), ("stride", True, "null"), ("padding", True, "0"), ("ceil_mode", True, "false"), ("count_include_pad", True, "true"), ("divisor_override", True, "null")], "sd", None, None),
    "norm": (D+"norm", [("p", False, None)], "sd", None, None),
    "renorm": (D+"renorm", [("p", False, None), ("dim", False, None), ("maxnorm", False, None)], "sd", None, None),
    "normalize": (D+"normalize", [("ord", True, "null"), ND, NK], "sd", None, None),
    "cross": (D+"cross", [("dim", True, -1)], "sd", None, None),
    "tensordot": (D+"tensordot", [("dims_self", False, None), ("dims_other", False, None)], "wire", None, None),
    "kron": (D+"kron", [("other", False, None)], "sd", None, None),
    "dist": (D+"dist", [("other", False, None), ("p", True, 2)], "sd", None, None),
    "cdist": (D+"cdist", [("other", False, None), ("p", True, 2)], "sd", None, None),
    "pdist": (D+"pdist", [("p", True, 2)], "sd", None, None),
    "bincount": (D+"bincount", [("weights", True, "null"), ("minlength", True, 0)], "sd", None, None),
    "corrcoef": (D+"corrcoef", [], "sd", None, None),
    "to": (D+"to", [("spec", False, None)], "sd", None, None),
    # rest = [self, out_int32, right, side, sorter] (t = sorted_sequence);
    # the special-case call below binds them — out_int32 is ignored
    # (f64 wire), side overrides right, sorter dies in the driver.
    "searchsorted": (D+"searchsorted", [("self_", False, None), ("_out_int32", True, False), ("right", True, False), ("_side", True, None), ("_sorter", True, None)], "sd", None, None),
    "meshgrid": (D+"meshgrid", [("_list", False, None)], "listwire", None, None),
    # multi-arg dispatch
    "max": (D+"maxmin", [("_rest", False, None)], "maxmin", "true", None),
    "min": (D+"maxmin", [("_rest", False, None)], "maxmin", "false", None),
    "all": (D+"allany", [("_rest", False, None)], "wire", "true", None),
    "any": (D+"allany", [("_rest", False, None)], "wire", "false", None),
    "prelu": (D+"prelu", [("weight", False, None)], "sd", None, None),
    "where": (D+"where_op", [("_rest", False, None)], "wire", None, None),
    "clamp": (D+"clamp_op", [("_rest", False, None)], "wire", None, None),
    "clamp_min": (D+"clamp_minmax", [("_rest", False, None)], "wire", "true", None),
    "clamp_max": (D+"clamp_minmax", [("_rest", False, None)], "wire", "false", None),
    "lerp": (D+"lerp", [("end", False, None), ("weight", False, None)], "sd", None, None),
}

# parametric unary/binary compose recipes: name → (kind, prelude_lines, expr)
COMPOSE = {
    "elu": ("unary_p", "a :: aten_opt_num(rest, 0, 1)",
            "if x > 0 { x } else { a * (math.exp(x) - 1) }"),
    "celu": ("unary_p", "a :: aten_opt_num(rest, 0, 1)",
             "aten_fmax(x, 0) + a * (math.exp(aten_fmin(x / a, 0)) - 1)"),
    "leaky_relu": ("unary_p", "ns :: aten_opt_num(rest, 0, 0.01)",
                   "if x >= 0 { x } else { ns * x }"),
    "softshrink": ("unary_p", "lam :: aten_opt_num(rest, 0, 0.5)",
                   "if x > lam { x - lam } else { if x < 0 - lam { x + lam } else { 0 } }"),
    "hardshrink": ("unary_p", "lam :: aten_opt_num(rest, 0, 0.5)",
                   "if aten_abs(x) > lam { x } else { 0 }"),
    "hardtanh": ("unary_p2", "lo :: aten_opt_num(rest, 0, 0 - 1)\nhi :: aten_opt_num(rest, 1, 1)",
                 "aten_fmin(aten_fmax(x, lo), hi)"),
    "threshold": ("unary_p2", "th :: aten_opt_num(rest, 0, 0)\nv :: aten_opt_num(rest, 1, 0)",
                  "if x > th { x } else { v }"),
    "logit": ("unary_p", "eps :: aten_opt(rest, 0, null)",
              "do {\n  xc :: if eps == null { x } else { aten_fmin(aten_fmax(x, eps), 1 - eps) }\n  math.log(xc / (1 - xc))\n}"),
    "nan_to_num": ("unary_p3", "nanv :: aten_opt_num(rest, 0, 0)\nposv :: aten_opt(rest, 1, null)\nnegv :: aten_opt(rest, 2, null)",
                   "do {\n  xv :: if aten_is_nan(x) { nanv } else { if aten_is_infinite(x) { if x > 0 { if posv == null { x } else { posv } } else { if negv == null { x } else { negv } } } else { x } }\n  xv\n}"),
    "gelu": ("unary_p", "approx :: aten_opt(rest, 0, null)",
             "do {\n  is_tanh :: approx == \"tanh\"\n  x3 :: x * x * x\n  v :: if is_tanh {\n    0.5 * x * (1 + math.tanh(math.sqrt(1.5707963267948966 / 3.141592653589793) * (x + 0.044715 * x3)))\n  } else {\n    0.5 * x * (1 + aten_erf(x / math.sqrt(2)))\n  }\n  v\n}"),
    "softplus": ("unary_p2", "beta :: aten_opt_num(rest, 0, 1)\nthr :: aten_opt_num(rest, 1, 20)",
                 "if beta * x > thr { x } else { math.log1p(math.exp(beta * x)) / beta }"),
    "is_contiguous": (None, None, None),  # handled as meta driver
}

STUB_EXTRA = set()

FAMILIES = {
    "unary": ["abs","neg","exp","log","log10","log2","sqrt","rsqrt","reciprocal",
              "square","sin","cos","tan","asin","acos","atan","sinh","cosh",
              "tanh","asinh","acosh","atanh","floor","ceil","round","trunc",
              "frac","deg2rad","rad2deg","erf","erfc","silu","sigmoid","sinc",
              "relu","selu","sign","elu","celu","leaky_relu","softshrink",
              "hardshrink","hardtanh","threshold","logit","nan_to_num","gelu",
              "exp2","expm1","log1p","mish","softplus"],
    "pred": ["isfinite","isinf","isnan","isposinf","isneginf","signbit",
             "logical_not"],
    "binary": ["atan2","hypot","fmod","floor_divide","maximum","minimum",
               "pow","float_power","eq","ne","lt","le","gt","ge",
               "logical_and","logical_or","logical_xor","logaddexp",
               "logaddexp2","copysign","nextafter","remainder","heaviside",
               "xlogy"],
    "reduce": ["sum","prod","amax","amin","argmax","argmin","mean","var",
               "std","logsumexp","cumsum","cumprod","softmax",
               "log_softmax","count_nonzero","nansum","nanmean","all","any",
               "max","min"],
    "sortsel": ["sort","argsort","topk","msort","median","mode","kthvalue",
                "quantile","searchsorted"],
    "index": ["nonzero","argwhere","unique","nonzero_dim","index_select","gather","index_add",
              "index_fill","take","squeeze","flatten","expand","expand_as",
              "narrow","slice","chunk","split","stack","unbind","repeat",
              "flip","triu","tril","diag","diagonal","trace","t","mT",
              "swapaxes","moveaxis","rot90","contiguous","clone","detach",
              "view","pixel_shuffle","pixel_unshuffle","constant_pad_nd"],
    "linalg": ["mm","bmm","mv","dot","vdot","inner","ger","outer","addr",
               "addmv","addmm","addbmm","baddbmm","cross","tensordot",
               "kron","dist","cdist","pdist","prelu","lerp","clamp",
               "clamp_min","clamp_max","where","norm","renorm","normalize"],
    "nn": ["layer_norm","rms_norm","glu","avg_pool2d"],
    "construct": ["full","full_like","zeros_like","ones_like","arange",
                   "linspace","logspace","eye","numel","dim","size",
                   "stride","item","is_contiguous","bincount","corrcoef",
                   "to","meshgrid"],
    "stub": ["empty","rand","normal","einsum"],
}

# ---------------------------------------------------------------------------
# codegen
# ---------------------------------------------------------------------------

IMPORTS = """from "../internal/aten.hnl" import {
  aten_T, aten_W, aten_ow, aten_owl, aten_is_num, aten_at, aten_opt,
  aten_opt_num, aten_req, aten_arg_w, aten_wrap_dim, aten_mem,
  aten_dim_spec, aten_bump, aten_vmap, aten_map1, aten_bin_ss, aten_map3,
  aten_where, aten_clamp_side, aten_reduce, aten_arg_reduce, aten_cum,
  aten_softmax, aten_sort_dim, aten_select_dim_index, aten_map_pos,
  aten_permute, aten_flip, aten_repeat, aten_expand, aten_triu,
  aten_slice_dim, aten_index_select, aten_unbind, aten_chunk, aten_split,
  aten_stack, aten_nonzero, aten_mm2, aten_bmm, aten_mv, aten_dot_data,
  aten_ger, aten_addr, aten_addmv, aten_gather, aten_index_add,
  aten_take, aten_index_fill, aten_squeeze, aten_flatten, aten_slice,
  aten_diag, aten_diagonal, aten_trace, aten_t, aten_swapaxes,
  aten_moveaxis, aten_rot90, aten_reduce_dims_list, aten_p_norm,
  aten_norm, aten_renorm, aten_normalize, aten_cross, aten_reshape,
  aten_tensordot, aten_kron, aten_p_dist_rows, aten_cdist, aten_pdist,
  aten_dist, aten_erf, aten_logsumexp, aten_var_std, aten_layer_norm,
  aten_rms_norm, aten_glu, aten_avg_pool2d, aten_constant_pad_nd,
  aten_pixel_space_depth, aten_bincount, aten_corrcoef, aten_to,
  aten_topk, aten_searchsorted, aten_meshgrid, aten_count_nonzero,
  null_dims, aten_allclose, aten_is_nonzero, aten_msort, aten_nansum,
  aten_nanmean, aten_full, aten_full_like, aten_zeros_like,
  aten_ones_like, aten_arange, aten_linspace, aten_eye, aten_sort_select,
  aten_id, aten_dot_wrap, aten_expand_as, aten_reshape_view, aten_mean,
  aten_maxmin, aten_allany, aten_prelu, aten_where_op, aten_clamp_op,
  aten_floor, aten_ceil, aten_trunc, aten_fract, aten_round, aten_abs,
  aten_fmax, aten_fmin, aten_is_nan, aten_is_finite, aten_is_infinite,
  aten_sign, aten_f64_max,
  aten_clamp_minmax, aten_lerp, aten_addmm, aten_isort, aten_median,
  aten_mode, aten_kthvalue, aten_quantile, aten_unique,
  aten_nonzero_static
}
from "../internal/helpers.hnl" import {
  die, len, product, replicate, set, slice, strides, unravel, dot,
  take, drop, rem_euclid
}
from "../codec.hnl" import { wire, as_wire, wire_shape, wire_data }
from "../elementwise.hnl" import elementwise
from "math" import * as math
"""

SAFE_COMMENT = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,;:()[]{}<>!=+-*/_%\"|@#$&~ ")
def clean_comment(s):
    return "".join(c for c in s if c in SAFE_COMMENT or ord(c) < 128 and c.isalnum())

def hnl_lit(v):
    if v is None: return "null"
    if v is True: return "true"
    if v is False: return "false"
    if isinstance(v, str):
        # DRIVER tables spell literal defaults as strings for
        # readability ("null", "true", "0", "10"); emit them as the
        # literal they name. Genuine string defaults stay quoted.
        if v in ("null", "true", "false"):
            return v
        try:
            return str(int(v)) if v.lstrip("-").isdigit() else repr(float(v))
        except ValueError:
            return json.dumps(v)
    if isinstance(v, float):
        r = repr(v)
        if "e" in r or "E" in r:
            return format(v, ".12f").rstrip("0").rstrip(".")
        return r
    if isinstance(v, int): return str(v)
    return "null"


def gen_unary(expr):
    return f"""(t, ...rest) ~> do {{
  w :: as_wire(t)
  aten_ow(aten_map1(w, (x) ~> {expr}))
}}"""

def gen_pred(cond):
    return f"""(t, ...rest) ~> do {{
  w :: as_wire(t)
  aten_ow(aten_map1(w, (x) ~> if {cond} {{ 1 }} else {{ 0 }}))
}}"""

def gen_bin(expr):
    return f"""(a, b, ...rest) ~> do {{
  aten_ow(aten_bin_ss(a, b, (x, y) ~> {expr}))
}}"""

def arg_expr(param, idx, optional, default):
    if param.startswith("_"):
        # literal/flag params: not a rest arg
        return None
    if optional:
        return f"aten_opt(rest, {idx}, {hnl_lit(default)})"
    return f"aten_req(rest, {idx})"


def gen_driver(op, spec):
    driver, params, ret = spec[0], spec[1], spec[2]
    extra = spec[3:]
    # rest indices count only non-_-prefixed params (literal/flag
    # params are not rest args)
    call_args = []
    ridx = 0
    for p in params:
        name, opt, default = p
        if name.startswith("_"):
            call_args.append(hnl_lit(default))
        else:
            call_args.append(arg_expr(name, ridx, opt, default))
            ridx += 1
    call = f"{driver}(t, {', '.join(call_args)})" if call_args else f"{driver}(t)"

    # underscore drivers: custom arg plumbing
    if driver == "_reduce":
        init, f = extra[0], extra[1]
        call = ("aten_reduce(w, aten_dim_spec(aten_opt(rest, 0, null), "
                "len(w.__tensor__.shape)), aten_opt(rest, 1, false), "
                f"{init}, {f})")
        body = f"aten_ow({call})"
        return f"""(t, ...rest) ~> do {{
  w :: as_wire(t)
  {body}
}}"""
    if driver == "_argred":
        want = extra[-1]
        call = ("aten_arg_reduce(t, aten_opt(rest, 0, null), "
                "aten_opt(rest, 1, false), " + want + ")")
        body = f"aten_ow({call})"
        return f"""(t, ...rest) ~> do {{
  w :: as_wire(t)
  {body}
}}"""
    if driver == "_cum":
        prod = extra[-1]
        call = f"aten_cum(t, aten_req(rest, 0), {prod})"
        body = f"aten_ow({call})"
        return f"""(t, ...rest) ~> do {{
  w :: as_wire(t)
  {body}
}}"""
    if driver == "_softmax":
        log = extra[-1]
        call = f"aten_softmax(t, aten_req(rest, 0), {log})"
        body = f"aten_ow({call})"
        return f"""(t, ...rest) ~> do {{
  w :: as_wire(t)
  {body}
}}"""

    # special first-arg shapes (t is not always `self`)
    if op == "full":
        call = "aten_full(t, aten_opt(rest, 0, 0))"
    if op == "eye":
        call = "aten_eye(t, aten_opt(rest, 0, null))"
    if op == "arange":
        call = "aten_arange([t, ...rest])"
    if op == "linspace" or op == "logspace":
        pass  # handled by ret below
    if op == "stack":
        call = "aten_stack(t, aten_opt(rest, 0, 0))"

    # special drivers needing custom first args
    if op == "meshgrid":
        call = "aten_meshgrid([t, ...rest])"
    if op == "narrow":
        # aten_slice_dim(t, dim, start, len)
        call = f"aten_slice_dim(t, aten_req(rest, 0), aten_req(rest, 1), aten_req(rest, 2))"
    if op == "searchsorted":
        # schema: searchsorted(sorted_sequence, self, *, out_int32,
        # right, side, sorter) — t is sorted_sequence; rest =
        # [self, out_int32, right, side, sorter]. out_int32 is
        # accepted-and-ignored; side/sorter are forwarded to the
        # driver (side overrides right; non-null sorter dies).
        call = ("aten_searchsorted(t, aten_req(rest, 0), "
                "aten_opt(rest, 2, false), aten_opt(rest, 3, null), "
                "aten_opt(rest, 4, null))")
    if op == "cross":
        call = "aten_cross(t, aten_req(rest, 0), aten_opt(rest, 1, -1))"
    if op == "where":
        call = "aten_where(t, aten_req(rest, 0), aten_req(rest, 1))"
    if op == "clamp":
        call = "aten_clamp_op(t, rest)"
    if op == "var" or op == "std":
        call = ("aten_var_std(t, aten_opt(rest, 0, null), "
                "aten_opt(rest, 1, true), aten_opt(rest, 2, false), "
                + extra[-1] + ")")
    if op == "max" or op == "min":
        call = f"aten_maxmin(t, rest, {extra[0]})"
    if op == "all" or op == "any":
        call = f"aten_allany(t, rest, {extra[0]})"
    if op == "clamp_min" or op == "clamp_max":
        call = f"aten_clamp_minmax(t, rest, {extra[0]})"

    # return shaping
    if ret == "sd":
        body = f"aten_ow({call})"
    elif ret == "wire":
        body = f"aten_ow({call})"
    elif ret == "sdlist":
        body = f"aten_owl({call})"
    elif ret == "listwire":
        body = call  # aten_meshgrid already returns wires
    elif ret == "pair":
        # sort/argsort → {values, indices}
        if op == "sort":
            body = ("do {\n  s :: aten_sort_dim(t, aten_opt(rest, 0, -1), "
                    "aten_opt(rest, 1, false))\n  ({ values: aten_W(s.v), "
                    "indices: aten_W(s.i) })\n}")
        elif op == "argsort":
            body = ("do {\n  s :: aten_sort_dim(t, aten_opt(rest, 0, -1), "
                    "aten_opt(rest, 1, false))\n  aten_W(s.i)\n}")
    elif ret == "pairv":
        body = ("do {\n  r :: aten_topk(t, aten_req(rest, 0), "
                "aten_opt(rest, 1, -1), aten_opt(rest, 2, true))\n"
                "  ({ values: aten_W(r.v), indices: aten_W(r.i) })\n}")
    elif ret == "rec":
        body = call  # median/mode/kthvalue already return wire records
    elif ret == "rec_or_sd":
        body = ("do {\n  r :: aten_median(t, rest)\n"
                "  if aten_is_num(r) || r.s != null { aten_ow(r) } else { r }\n}")
    elif ret == "sd_rest":
        # quantile: rest = [q, dim?, keepdim?, interp?]
        body = ("aten_ow(aten_quantile(t, rest))")
    elif ret == "sd_ls":
        body = ("aten_ow(aten_linspace(t, aten_req(rest, 0), aten_req(rest, 1), "
                "false, 0))")
    elif ret == "sd_ls_log":
        body = ("aten_ow(aten_linspace(t, aten_req(rest, 0), aten_req(rest, 1), "
                "true, aten_opt_num(rest, 2, 10)))")
    elif ret == "maxmin":
        body = call
    elif ret == "meta":
        body = META_BODIES[op]
    else:
        body = call

    return f"""(t, ...rest) ~> {body if '\n' in body else body}"""


META_BODIES = {
    "numel": """do {
  w :: as_wire(t)
  len(w.__tensor__.data) * 1.0
}""",
    "dim": """do {
  w :: as_wire(t)
  len(w.__tensor__.shape) * 1.0
}""",
    "size": """do {
  w :: as_wire(t)
  shp :: w.__tensor__.shape
  if len(rest) == 0 || rest[0] == null {
    shp
  } else {
    d :: rem_euclid(rest[0], len(shp))
    shp[d] * 1.0
  }
}""",
    "stride": """do {
  w :: as_wire(t)
  shp :: w.__tensor__.shape
  st :: strides(shp)
  if len(rest) == 0 || rest[0] == null {
    st
  } else {
    d :: rem_euclid(rest[0], len(shp))
    st[d] * 1.0
  }
}""",
    "item": """do {
  w :: as_wire(t)
  d :: w.__tensor__.data
  if len(d) == 1 {
    d[0]
  } else {
    die("aten.item: numel != 1")
  }
}""",
    "is_contiguous": "true",
}


def gen_compose(op):
    kind, pre, expr = COMPOSE[op]
    if kind == "unary_p":
        return f"""(t, ...rest) ~> do {{
  {pre}
  w :: as_wire(t)
  aten_ow(aten_map1(w, (x) ~> {expr}))
}}"""
    if kind == "unary_p2":
        return f"""(t, ...rest) ~> do {{
  {pre}
  w :: as_wire(t)
  aten_ow(aten_map1(w, (x) ~> {expr}))
}}"""
    if kind == "unary_p3":
        return f"""(t, ...rest) ~> do {{
  {pre}
  w :: as_wire(t)
  aten_ow(aten_map1(w, (x) ~> {expr}))
}}"""
    return None


def gen_stub(op, reason):
    r = "".join(c for c in reason if ord(c) < 128 and c not in "'\\`")
    return f"""(t, ...rest) ~> do {{
  die("aten.{op}: {r}")
}}"""


def classify(entry):
    hnl = entry["hnl"]
    if entry["seam"] or hnl in STUBS:
        return ("stub", STUBS.get(hnl, "explicit host seam"))
    if hnl in HANDWRITTEN:
        return ("skip", "handwritten op already in surface")
    if hnl in MATH_UNARY:
        return ("unary", MATH_UNARY[hnl])
    if hnl in PRED:
        return ("pred", PRED[hnl])
    if hnl in BIN:
        return ("bin", BIN[hnl])
    if hnl in COMPOSE and COMPOSE[hnl][0]:
        return ("compose", None)
    if hnl in DRIVER:
        return ("driver", DRIVER[hnl])
    return ("stub", "no honest .hnl recipe — schema features not representable")


def main():
    check = "--check" in sys.argv
    entries = parse_port_list(PORT_LIST)
    schemas = load_schemas(SCHEMAS)

    # resolve + classify
    plans = []
    report = {"clean": 0, "manual": 0, "unsupported": 0, "skip": 0,
              "seam": 0, "notes": []}
    for e in entries:
        rec, warn = resolve_schema(schemas, e["aten"])
        if warn:
            report["notes"].append(f"{e['hnl']}: {warn}")
        cls, detail = classify(e)
        plans.append((e, cls, detail, rec))
        if cls == "skip":
            report["skip"] += 1
        elif cls == "stub":
            report["unsupported"] += 1
            if e["seam"]:
                report["seam"] += 1
        else:
            report["clean"] += 1

    # emit: build every output in memory first so --check can diff
    # without touching the tree.
    fam_ops = {f: [] for f in FAMILIES}
    for e, cls, detail, rec in plans:
        if cls == "skip":
            continue
        fam = next((f for f, names in FAMILIES.items() if e["hnl"] in names), "misc")
        fam_ops.setdefault(fam, []).append((e, cls, detail, rec))

    outputs = {}  # Path -> full file text
    for fam, ops in sorted(fam_ops.items()):
        if not ops:
            continue
        path = OUT_DIR / f"{fam}.hnl"
        body = [IMPORTS, f"--- aten-port-gen: {fam} family ---\n"]
        for e, cls, detail, rec in ops:
            schema_str = ""
            if rec:
                args = ", ".join(
                    f"{a.get('name')}:{a.get('type')}" for a in rec.get("args", []))
                schema_str = "-- aten: " + clean_comment(rec.get("schema", e["aten"])) + "\n"
            seam_note = "  -- SEAM (host-declared)\n" if e["seam"] else ""
            ename = e["hnl"] + ("_" if e["hnl"] in ("sort", "where") else "")
            body.append(f"{seam_note}{schema_str}export {ename} := ")
            if cls == "unary":
                body.append(gen_unary(detail))
            elif cls == "pred":
                body.append(gen_pred(detail))
            elif cls == "bin":
                body.append(gen_bin(detail))
            elif cls == "compose":
                body.append(gen_compose(e["hnl"]))
            elif cls == "driver":
                body.append(gen_driver(e["hnl"], detail))
            elif cls == "stub":
                body.append(gen_stub(e["hnl"], detail))
            body.append("\n\n")
        outputs[path] = "".join(body)

    # index.hnl splice
    idx = INDEX.read_text()
    imports, exports, fields = [], [], []
    for e, cls, detail, rec in plans:
        if cls == "skip":
            continue
        fam = next((f for f, names in FAMILIES.items() if e["hnl"] in names), "misc")
        ename = e["hnl"] + ("_" if e["hnl"] in ("sort", "where") else "")
        imports.append(f'from "./aten/{fam}.hnl" import {ename} as __aten_{e["hnl"]}')
        if e["hnl"] not in ("sort", "where"):
            exports.append(f"export {e['hnl']} := __aten_{e['hnl']}")
        fields.append(f"  {e['hnl']}: __aten_{e['hnl']},")
    # block ends at END (no trailing newline): `post` already carries
    # the newline that ended the END line, so appending one here used
    # to accumulate a blank line per regen.
    block = (BEGIN + "\n" + "\n".join(imports) + "\n\n"
             + "\n".join(exports) + "\n" + END)
    ltensor_fields = "\n".join(fields) + "\n"

    if BEGIN in idx:
        pre, rest = idx.split(BEGIN, 1)
        _, post = rest.split(END, 1)
        # collapse the blank run after END (bug-era regens accumulated
        # one per run) to a single blank line — keeps emit idempotent
        if post.startswith("\n"):
            post = "\n\n" + post.lstrip("\n")
        idx = pre + block + post
        # replace ltensor fields region: regenerate whole ltensor record
        m = re.search(r"export ltensor := \{.*?\n\}", idx, re.S)
        if m:
            inner = m.group(0)
            inner = re.sub(r"\n  \w+: __aten_\w+,", "", inner)
            inner = inner.rstrip()[:-1].rstrip() + "\n" + ltensor_fields + "}"
            idx = idx[:m.start()] + inner + idx[m.end():]
    else:
        # append import/export block before `export ltensor`, fields inside
        lt = idx.index("export ltensor := {")
        idx = idx[:lt] + block + "\n\n" + idx[lt:]
        m = re.search(r"export ltensor := \{.*?\n\}", idx, re.S)
        inner = m.group(0)
        inner = re.sub(r"\n  \w+: __aten_\w+,", "", inner)
        inner = inner.rstrip()[:-1].rstrip() + "\n" + ltensor_fields + "}"
        idx = idx[:m.start()] + inner + idx[m.end():]
    outputs[INDEX] = idx

    # summary
    print(f"entries: {len(entries)}  clean: {report['clean']}  "
          f"unsupported/stub: {report['unsupported']}  "
          f"handwritten-skip: {report['skip']}  seams: {report['seam']}")
    for n in report["notes"]:
        print("  note:", n)

    if check:
        # real check: byte-diff every generated file (plus flag any
        # stale src/aten/*.hnl the emit no longer produces)
        drift = []
        for path, text in sorted(outputs.items()):
            if not path.exists() or path.read_text() != text:
                drift.append(path)
        for stale in sorted(OUT_DIR.glob("*.hnl")):
            if stale not in outputs:
                drift.append(stale)
        if drift:
            for p in drift:
                print("DRIFT:", p.relative_to(REPO))
            print(f"{len(drift)} file(s) differ from generated output — "
                  "rerun without --check to regenerate")
            return 1
        print(f"no drift: {len(outputs)} files match generated output")
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for path, text in outputs.items():
        path.write_text(text)
        print("wrote", path.relative_to(REPO))
    return 0


if __name__ == "__main__":
    sys.exit(main())
