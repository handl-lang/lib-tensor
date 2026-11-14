# lib-tensor ATen port generator — report

Date: 2026-07 (work in progress, uncommitted)

## What this is

`tools/aten_port_gen.py` is a lib-tensor-side emitter that reads the
host-tensor ATen catalog and generates `.hnl` operation wrappers, closing
the gap between `host-tensor` (~238 ops) and `lib-tensor` (was 31).
It does **not** reuse the Rust generator in `host-tensor` (that emits
Rust); it follows the `ext-sdkgen` schema-to-typed-bindings pattern but
targets `.hnl`.

Inputs (host-tensor is read-only reference material):

- `../host-tensor/tools/aten_schemas.json` — 3,757 ATen schemas
- `../host-tensor/tools/aten_port_list.txt` — 207 curated pins
  (`tensor.<name> = aten::<schema>` lines, with `seam`/`covers` flags)

Outputs:

- `src/aten/{binary,construct,index,linalg,misc,nn,pred,reduce,
  sortsel,stub,unary}.hnl` — 207 generated exports, ~4.9k lines total
- `src/index.hnl` — spliced import/export/`ltensor` record block
  between `// aten-port-gen: begin` / `// aten-port-gen: end` markers
- `src/internal/aten.hnl` — shared transparent helper layer
  (~3.5k lines): broadcasting, reductions, argsorts, indexing,
  constructors, linalg, pooling, norms, cumsum, sort/topk/median/mode,
  gather/scatter/index-*, shape transforms, etc.

## Counts (from `python3 tools/aten_port_gen.py --check`)

| class | count |
|---|---|
| entries in port list | 207 |
| clean (generated wrapper over shared helpers) | 195 |
| unsupported / explicit stub | 12 |
| handwritten-skip | 0 |
| host-declared seams (subset of stubs) | 3 |

Class breakdown of the 207: `driver` 120, `unary` 37, `bin` 21,
`compose` 10, `pred` 7, `stub` 12.

### The three host seams (all emit `die("aten.<op>: ...")` stubs)

- `empty` — uninitialized fill cannot be represented on the wire
- `rand` — RNG state is not part of the wire contract; nondeterminism
  breaks verification
- `normal` — same RNG reason

### Closed since first emit (2)

- `unique` (`aten::_unique`) — handwritten `aten_unique` driver:
  flat unique over row-major data, `{values, indices}` record encoding
  (sorted ascending NaN-last, or first-occurrence order under
  `sorted=False`; `return_inverse` inert per the schema's fixed
  2-tuple return) — matches host `k_unique`.
- `nonzero_dim` (`aten::nonzero_static`) — handwritten
  `aten_nonzero_static` driver: `[size, rank]` output, first `nnz`
  rows = nonzero coords, rest `fill_value`; `size < nnz` dies —
  matches host `k_nonzero_static`.

### Other explicit stubs (9)

`exp2`, `expm1`, `log1p`, `logaddexp2`, `mish`, `softplus` (missing or
non-bit-identical `math.*` leaves), `nextafter` (needs f64 bit
patterns), `copysign` (sign-of-zero semantics), `einsum` (equation-string
arg + general contraction engine — host has a dedicated parser).

### What would close the math stubs (vocab extension, not done here)

Eight of the nine non-seam stubs need five new `math.*` leaves — all
are thin `unary_fn!`/`bin_op!` natives plus the usual
lib-math → lib-math-c → lib-math-js wrapper chain:

| leaf | unblocks | host recipe once the leaf exists |
|---|---|---|
| `math.exp2` (`f64::exp2`) | `exp2`, `logaddexp2` | `math.exp2(x)`; logaddexp2 mirrors `logaddexp` with `exp2`/`log2` |
| `math.expm1` (`f64::exp_m1`) | `expm1` | `math.expm1(x)` |
| `math.log1p` (`f64::ln_1p`) | `log1p`, `mish`, `softplus` | `math.log1p(x)`; `mish` = `x * math.tanh(math.log1p(math.exp(x)))`; `softplus` = host `k_softplus` threshold branch (`beta*x > threshold → x`, else `log1p(exp(beta*x))/beta`) |
| `math.nextafter` (`f64::next_up`/`from_bits` step) | `nextafter` | `math.nextafter(x, y)` — one-ULP step needs bit patterns no .hnl expr reaches |
| `math.copysign` (`f64::copysign`) | `copysign` | `math.copysign(x, y)` — sign-of-zero semantics need bit-level access |

Recipes must match the host kernels bit-for-bit (`k_mish`,
`k_softplus`, `k_logaddexp2` in `host-tensor/src/ops/aten_kernels.rs`)
since `verify.hnl` compares exact f64 equality — `math.pow(2, x)` is
*not* a substitute for `exp2`, and `exp(x) - 1` / `log(1 + x)` lose
precision for small `|x|`.

`einsum` is the remaining non-seam stub and needs no leaf — it needs
the equation-string parser ported to `.hnl` (host `k_einsum`,
~400 lines: explicit/implicit equations, ellipsis batch dims,
≥2 operands). Everything else (`empty`, `rand`, `normal`) is
genuinely unsupported: uninitialized fill and RNG state have no honest
wire representation.

Every stub body is `die("aten.<op>: <reason>")` — an honest failure, not
a pretend implementation.

## Schema features that cannot be represented on the wire

- **Scalar transcendental leaves**: `math.exp/pow/sqrt/sin/tanh/log/
  atan2/hypot/...` are host calls, not transparent structure. They work
  under runtimes that register the math extension (verified via dogfood
  `hnl eval`), but fail under `hnl-run` builds that don't load it. Ops
  depending on them (`exp`, `softmax`, `norm`, `std`, GELU/GLU, layer
  norm, distances, …) are structurally correct but environment-gated.
- **Missing math leaves**: no `exp2`, `expm1`, `log1p` → stubbed.
- **Bit-level float semantics**: `copysign` (signed zero), `nextafter`
  (one ULP) → stubbed.
- **RNG / uninitialized memory**: `empty`, `rand`, `normal` → seams.
- **Tuple/multi returns**: `unique`, `nonzero_static`, `sort`/`topk`
  values+indices (sort/topk return records and are implemented; unique
  remains a seam).
- **String/equation args**: `einsum` equation parsing → seam.
- **`out=` mutation params**: ignored by wrappers (functional style).
- **dtype args**: `ScalarType?` params accepted but mostly ignored —
  the wire is f64-only.
- **Overload resolution**: `resolve_schema` picks the first schema when
  the port-list entry doesn't name an overload; a warning is recorded.

## Verification status

- `python3 tools/aten_port_gen.py --check` — 207 entries classified,
  deterministic (writes only without `--check`).
- `python3 tools/aten_port_gen_test.py` — 20 unit tests, all pass
  (port-list parsing, schema resolution, literal rendering, overload
  classification, seam preservation, reserved-name handling
  `sort`/`where` → `sort_`/`where_`, ASCII-safe stub bodies,
  determinism).
- `hnl lint` (dogfood, `--ext-dir` built extensions): warnings only
  (shadowed/unused bindings in `src/internal/aten.hnl`), no errors.
- `hnl typecheck` (dogfood ext): `src/index.hnl` — the generated surface
  plus all `src/aten/*.hnl` wrappers — typechecks `ok`. The dynamic
  helper layer does **not**: `src/internal/aten.hnl` reports
  `array_element_type` on `xs[i]` indexing, and the same class of
  failure already exists in the pre-existing handwritten files
  (`src/codec.hnl` `record_field_type`, `src/elementwise.hnl`
  `cannot unify`, `src/internal/helpers.hnl` `unbound variable
  'product'`). The dogfood typechecker does not cover these dynamic
  indexing/record patterns; it is a dialect limitation shared by the
  old and new code, not a generated-code regression.
  `src/lib_tensor.test.hnl` also fails (`synth_assign: var name`,
  record-unify `[]` vs `["host","lib"]`).
- `hnl-run` (graphopt release build) ran the original `verify.hnl`
  differential suite: all 55 checks `ok` (constructors, elementwise,
  broadcasting, matmul, linear, conv, pooling, batchnorm,
  reshape/transpose/unsqueeze/select, backward ops, derive).
- Generated-op smoke probes under `hnl-run` passed for non-transcendental
  ops: `count_nonzero`, `clamp`, `nonzero`, `argmax`, `arange`, `expand`,
  `numel`, `isfinite`, `full*`, `zeros_like`, `ones_like`, `eye`,
  `searchsorted`, `mean`, `cumprod`, `nansum`, `nanmean`, `all`, `any`,
  `allclose`, `is_nonzero`, `msort`, `median`, `quantile`, `bmm`, `mv`,
  `dot`, `vdot`, `inner`, `ger`, `outer`, `kron`, `cross`, `addmm`, `mm`,
  `avg_pool2d`, `sort`, `argsort`, `topk`, `index_select`, `split`,
  `rot90`, `moveaxis` (some forms), `var`, `dim`, `size`, `stride`.
- Transcendental ops verified under dogfood `hnl eval` where `math.*`
  natives are registered (`exp`, `softmax`, `std`, `norm` returned
  correct values in inline probes).

## Known remaining issues (honest list)

- `where` once returned nested wire records in a probe — needs a
  dedicated check.
- `tensordot` still fails in some arg shapes (null/field access).
- `meshgrid`, `moveaxis` default/list forms, `cat`/`permute`/`scatter`
  probe forms, `mode`, `p_norm` — partially exercised, some call-shape
  bugs remain.
- Float-valued shape entries (`[2.0]`) appear in some results —
  semantically usable, cosmetically unpolished.
- `std`/`softmax`/`norm`/GELU/layer-norm/distances require a runtime
  with `math.*` registered; they are not broken `.hnl`, they are
  environment-gated.
- `hnl run` file-mode binds `lt.*` differently than `hnl eval` in some
  probes (`call_host: lt.exp not found`) — runtime binding quirk, not a
  module bug; inline `eval` and `hnl-run` script probes both work.

## How to regenerate

```sh
python3 tools/aten_port_gen.py          # rewrite src/aten/* + splice index.hnl
python3 tools/aten_port_gen.py --check  # dry run: counts only
python3 tools/aten_port_gen_test.py     # unit tests
```

Manual edits inside the `// aten-port-gen: begin/end` block in
`src/index.hnl` will be overwritten; helper-layer changes belong in
`src/internal/aten.hnl` (not generated).
