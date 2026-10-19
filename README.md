# lib-tensor

`.hnl` source implementations of the tensor ops that
[`host-tensor`](https://github.com/handl-lang/host-tensor) provides
natively — the **transparent spec** to host-tensor's **opaque atoms**.

## The design pair

| | host-tensor | lib-tensor |
|---|---|---|
| role | fast eval | the spec |
| ops are | Rust `Function`s behind `op`/`handle` declarations — `call_host` leaves | `.hnl` arrows — `while` loops, `match`, index arithmetic |
| a rewrite sees | `tensor.conv2d(x, w, …)` as an atomic leaf | the broadcast odometer, the NCHW accumulation loops, the per-channel affine — structure it can look *inside* |
| values | `TaggedValue::Tensor` kernel variant | the `{"__tensor__": …}` wire record |

The two expose the same 24-op surface — the `*_backward` adjoints
mirror host-tensor 1:1 (`elementwise.hnl`/`matmul.hnl`/`conv.hnl`), and
`verify.hnl` checks all of them against the host under a composed
runtime. (`host-tensor/src/index.hnl` is
the interface contract; `src/ops/` the reference impl this package
mirrors line-for-line — same file map on both sides:
`codec`/`elementwise`/`matmul`/`conv`/`norm`/`shape_ops`, with `derive`
split out of `norm` in Rust since `DERIV_PAIRS` spans domains — and the
exact f64 expression order, so `batch_norm`'s `x·g + (β − μ·g)` and the
backward pass's `dγ += g·x̂` are bit-identical, not just mathematically
equal).

```hnl
from "lib-tensor" import ltensor as lt
lt.conv2d(x, w, null, [2, 2], 1)          -- same signature as tensor.conv2d
```

## The wire representation

lib-tensor operates on the **`Function`-boundary form** — the tagged
plain object every host op actually consumes:

```json
{"__tensor__": {"shape": [2, 3], "data": [1, 2, 3, 4, 5, 6], "dtype": "f64"}}
```

Every op **accepts both representations**: a wire record passes
through; an opaque `Tensor` (a host op's result) is projected through
`"tensor.shape"`/`"tensor.to_list"` at op entry — the only host calls
in the package, made as literal string callees so no import binding is
required on that path (only a runtime that has the `tensor` host
module composed). Every op **returns the wire record** — transparent
structure the e-graph can read — which any host op will decode back to
a `Tensor` on the next `tensor.*` call.

## The surface (24 ops, mirroring host-tensor 1:1)

`zeros`, `ones`, `from_list`, `to_list`, `shape` — codec/boundary ops
(pure over wire records; the two projections delegate to the host on
opaque inputs, and `zeros`/`ones`/`from_list` construct the record
directly — `data` may hold ints where the host emits `0.0`, an
int/float-insensitive difference that never survives the wire).
`add`, `mul`, `sub`, `div` — elementwise with numpy broadcasting.
`matmul` — `(m×k)·(k×n)`, rank-3 batch broadcasting.
`conv2d`, `max_pool` — NCHW spatial loops with the positional-optional
`null`-means-default convention and int-or-`[h,w]` pair decoding.
`batch_norm` — eval-mode per-channel affine.
`reshape`, `transpose`, `concat` — relabeling ops.
`batch_norm_backward` — the `{input, gamma, beta}` grads record.
`derive` — the registered-pairs table (`"batch_norm"` →
`"tensor.batch_norm_backward"`, else `null`).

## Expressibility findings

Writing the surface in the language surfaced its exact boundary —
these are the findings, not defects to be patched over:

1. **A `Tensor` is opaque to `.hnl`.** There is no eval-vocabulary
   destructure of a `TaggedValue::Tensor` — no field access, no match
   arm, no literal constructor. The whole spec could still be written
   because the *wire record* is a plain object: the boundary ops
   (`to_list`/`shape`) are the only host calls, used solely to adapt
   opaque inputs. Construction/projection were already known to be
   atoms (host-tensor's header says so); the finding here is that
   *everything above them* — all 14 algorithmic ops — is pure `.hnl`.

2. **The `while` step boundary reserializes values through the wire
   codec.** Loop state and captured env round-trip via
   `serde_json::Value → TaggedValue` each iteration, so a
   `{"__tensor__": …}` record held in `while` state (or captured from
   enclosing scope) *decodes to an opaque `Tensor`* — `.__tensor__`
   then fails on it. Plain lists and untagged records pass through
   untouched. The discipline this forces: destructure tensors to
   shape/data lists *before* entering loop state, and build the wire
   record only at the arrow's tail. Recursive arrow calls do **not**
   reserialize — `concat`'s per-element projections
   (`shapes_of`/`datas_of` in `src/shape_ops.hnl`) are recursive for
   exactly this reason.

3. **Module arrows capture only earlier bindings.** An `export f`
   defined before `g` cannot call `g` ("a name missing from that list
   did not survive into it") — helpers must precede use sites, and
   `import` bindings do not travel inside re-exported arrows (a helper
   that calls `tensor.*` through an imported record silently loses it
   at the caller's boundary — hence the literal `"tensor.*"` string
   callees in `src/codec.hnl`).

4. **Reserved names.** `data` and `infer` are keywords/native names —
   a binding spelled `data :=` fails to parse (`token.rparen` at the
   param list), and `export infer :=` is shadowed by the builtin
   (calls to it return their argument unchanged — a silent identity).
   Internal helpers are named `ds`/`infer_sh` accordingly. A bare
   `~> {` parses as a block, not a record literal — `wire` returns
   `({ ... })`.

5. **Error *kinds* do not port.** The Rust impls raise `EvalError`s
   (`TypeMismatch`, `Other` with messages); the spec raises the
   engine's `match` failure (`die` in `src/internal/helpers.hnl`).
   Accept/reject parity is asserted by construction; messages are not
   — `derive`'s non-string case returns `null` where the host raises a
   `TypeMismatch`.

## Layout

```text
src/index.hnl            facade: every op + the `ltensor` record
src/codec.hnl            the wire form, `as_wire`, from_list/to_list/shape/zeros/ones
src/elementwise.hnl      broadcast_shape/strides + add/mul/sub/div
src/matmul.hnl           matmul
src/conv.hnl             conv2d + max_pool (optional-param decoding)
src/norm.hnl             batch_norm + batch_norm_backward + derive
src/shape_ops.hnl        reshape + transpose + concat
src/internal/helpers.hnl list/stride/odometer machinery, `die`
src/lib_tensor.test.hnl  the .test.hnl spec suite (host-dependent)
verify.hnl               the same suite as a runnable verdict program
pkg.hnl                  { modules: ["lib-tensor"] }
.handl.toml              [package] + host-tensor git dep (subdir = "src") + [patch]
```

## Verification

`src/lib_tensor.test.hnl` is the spec-test form (`test_* := assert`);
like host-tensor's own suite it is host-dependent — `hnl test` does
not compose the `tensor` module (and its dialect detector skips the
arrow dialect), so the executable path is a host that does. catopt's
`CatoptKit` is the first consumer; its `hnl-run` resolves this repo's
`.handl.toml` deps:

```bash
<catopt>/target/debug/hnl-run --project . verify.hnl
# → [{"name":"zeros","ok":true}, …] — 32 checks, all ok
```

`ci.sh` runs `scripts/hnl-gate.sh` (the shared `hnl` typecheck/lint/
test gate); the sources here are all arrow-dialect, so the gate's
`is_canonical` detector skips them by design — `verify.hnl` is the
honest executable check until a tensor-composing `hnl test` exists.
