# lib-tensor generated-op differential verification — report

Date: 2026-07 (work in progress, uncommitted)

Closes the gap flagged in `graphopt/docs/reports/arch_audit_tensor_libs.md`
§3: `verify.hnl` differentially verified only the ~31 handwritten ops;
the ~195 generated ATen wrappers (`src/aten/*.hnl`, emitted by
`tools/aten_port_gen.py`) had zero coverage against host-tensor.

## What this is

`tools/verify_gen.py` — a separate generator (same inputs and marker
conventions as `aten_port_gen.py`) that owns two marker-spliced blocks
in `verify.hnl`:

- `-- >>> verify-gen: helpers >>>` — the `vg_tl` list-of-tensors
  normalizer plus its `len` import (emitted only when a list-returning
  op is covered: `chunk`, `split`, `unbind`, `meshgrid`).
- `-- >>> verify-gen: checks >>>` — one verdict expression per ported
  op, or an honest `-- skip <key>: <reason>` comment. Every one of the
  207 port-list rows is accounted for: emitted check, or skip comment
  with a reason.

Check forms reuse the handwritten section's `check`/`eq` helpers:

| kind | emitted form | used for |
|---|---|---|
| `eq` | `eq(name, lt.op(L), tensor.op(H))` | tensor results (`tensor.to_list` both sides) |
| `check` | `check(name, lt.op(L), tensor.op(H))` | scalars/ints/bools/lists |
| `pair` | `check(name, {values:…, indices:…}, {…})` | `{values, indices}` tuple records (`mode`, `kthvalue`-style) |
| `list` | `check(name, vg_tl(L), vg_tl(H))` | lists of tensors |
| `pred` | `check(name, lt.item(L) != 0, tensor.op(H))` | lib scalar-wire-tensor predicate vs host primitive bool |

Fixtures are curated data (`FIXTURES` in `tools/verify_gen.py`): small
deterministic tensors plus per-op argument strings; asymmetric
`l`/`h` argument lists handle the few ops whose two surfaces take
different argument *shapes* (see drift §D2–D5). `FIXTURE_SKIPS` holds
ops that were probed and rejected — never emitted as fake checks.

## Regenerate / verify

```sh
python3 tools/verify_gen.py              # probe every fixture under
                                         # hnl-run, then emit
python3 tools/verify_gen.py --no-probe   # emit from the probe cache
                                         # ($TMPDIR/lib-tensor-verify-probes)
python3 tools/verify_gen.py --check      # byte-diff verify.hnl, no write
HNL_RUN=../graphopt/target/release/hnl-run python3 tools/verify_gen.py
../graphopt/target/release/hnl-run --project . verify.hnl
```

The probe cache keys carry a fixture hash, so editing a fixture
re-probes just that check; `--check` fails if `verify.hnl` is stale.

## Coverage (207 port-list rows)

| outcome | count |
|---|---|
| emitted differential checks | 146 check-sites over 192 ops |
| — of which currently `ok` | 145 |
| — of which `ok: false` (real drift, §D1) | 1 (`hardshrink`) |
| skip comments emitted | 67 sites (50 math-gated + 2 host-seam + fixture/stub/pre-skip lines) |
| pre-skipped ops (no fixture attempted) | 15 |

Pre-skip breakdown: 12 `aten_port_gen` stubs (3 host-declared seams —
`empty`, `rand`, `normal`; 9 lib stubs for wire-unrepresentable math —
`exp2`, `expm1`, `log1p`, `mish`, `softplus`, `logaddexp2`, `nextafter`,
`copysign`), `searchsorted` (already covered by handwritten checks),
`tensordot` and `to` (§D3, §D4 — no honest fixture exists).

Skip taxonomy for the 67 comment lines:

- **50 sites / 49 ops — `math.*` environment gap.** The lib `.hnl`
  recipes call `math.exp/pow/sqrt/sin/…`, but hnl-run's CatoptKit
  registers `tensor.*` hosts only, not plugin-lib-stdlib's `math.*`
  leaves → `call_host: function 'math.X' not found`. Environment gap,
  not a semantic verdict; each skip comment names the missing leaf.
- **2 — host seams under hnl-run.** `unique`, `nonzero_dim`: the host
  `tensor.*` surface dies with "ATen port seam — TODO" even though the
  Rust kernels exist; nothing to compare against.
- **15 — stub/pre-skip comments** as tabulated above.

## Semantic drift found (findings, not hidden)

- **D1 — `hardshrink` is a real lib defect (suite reports `ok:false`).**
  `src/aten/unary.hnl:277-282` emits a body identical to `softshrink`
  (`x - λ·sign(x)` outside the threshold). Host `k_hardshrink` keeps
  `x` when `|x| > λ`. Probed: input `[-2.5,-0.5,0.5,2.5]`, lib default
  → `[-2.0,0,0,2.0]`, host → `[-2.5,0,0,2.5]`; explicit `λ=1` → lib
  `[-1.5,0,0,1.5]`, host `[-2.5,0,0,2.5]`. The check is emitted and
  fails on purpose — it is the suite's one red row until the wrapper
  is fixed.
- **D2 — `tensordot` crashes lib-side.** `lt.tensordot` dies
  `index: index 1 out of bounds` on every probed dims spec while host
  `k_tensordot` returns correct contractions. Skipped with the reason
  inline; candidate defect in the HNL driver/helper.
- **D3 — `to` has no common spec form.** Host `tensor.to` accepts the
  `"f64"` dtype string (or tensor rhs); lib `aten_to`
  (`src/internal/aten.hnl:2910`) accepts only a non-empty list or
  `{f: _}` record and dies on the string. Interface drift — nothing
  comparable to emit.
- **D4 — `index_add`/`index_fill` arg-type drift (checked anyway).**
  Lib drivers take `idxs` as a plain list; host kernels take a rank-1
  index tensor. Asymmetric fixtures (`l`/`h` args) verify the shared
  math — both pass.
- **D5 — optional/merged-arity drift (checked on the common form).**
  Host `tensor.dim`/`tensor.stride` take no dim argument (lib accepts
  one); host `tensor.threshold` requires all three args (lib defaults
  `th`/`v`); host `moveaxis` rejects the probed list-dst form. Only
  the forms both sides accept are checked; the divergent overloads are
  noted in fixture comments.
- **D6 — float shape entries (values verified, interop defect noted).**
  `bincount` and `avg_pool2d` return correct values but lib drivers
  emit FLOAT shape entries (`[3.0]`, `[1,1,2.0,2.0]`) that the host
  codec rejects ("expected shape: non-negative integers"). Compared
  via `lt.to_list` on the lib side so values still get coverage; the
  shape-typing defect is recorded here.
- **D7 — predicate representation (normalized, documented contract).**
  `allclose`/`is_nonzero` return a scalar wire tensor (`1.0`/`0.0`) on
  lib — the documented `src/internal/aten.hnl` predicate contract —
  vs a primitive bool on host. The `pred` kind unwraps via `lt.item`
  and compares `!= 0`; a wrong value still fails.

## Result

`hnl-run --project . verify.hnl` → 204 verdicts: 203 `ok`, 1 `ok:false`
(`hardshrink`, D1). Deterministic: fixtures are literals, probes are
cached by fixture hash, and `--check` byte-verifies the emitted file.
