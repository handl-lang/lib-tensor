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

The two expose the same op surface — the `*_backward` adjoints
mirror host-tensor 1:1 (`elementwise.hnl`/`matmul.hnl`/`conv.hnl`),
and `verify.hnl` checks all of them against the host under a composed
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

Every op **accepts both representations** and **returns the wire
record** — transparent structure the e-graph can read — which any host
op will decode back to a `Tensor` on the next `tensor.*` call. The two
representations are *observationally identical* in the language since
the tagged-object view seam landed (kernel
`TaggedValue::object_view`): an opaque `Tensor` projects its canonical
wire record for field access, `has_field`, and record destructuring,
so `t.__tensor__.shape` and `match t { { "__tensor__": inner }: … }`
read either form. The package makes **no host calls** — the old
literal-string `"tensor.shape"`/`"tensor.to_list"` projection arms
are gone.

## The surface (31 ops — host-tensor's full surface)

`zeros`, `ones`, `from_list`, `to_list`, `shape` — codec/boundary ops
(pure over wire records; `zeros`/`ones`/`from_list` construct the
record
directly — `data` may hold ints where the host emits `0.0`, an
int/float-insensitive difference that never survives the wire).
`add`, `mul`, `sub`, `div` — elementwise with numpy broadcasting.
`matmul` — `(m×k)·(k×n)`, rank-3 batch broadcasting.
`linear` — `F.linear`: `x @ wᵀ + b`, the `(out, in)` weight rank-2
checked up front, the bias broadcast by the elementwise `add`.
`conv2d`, `max_pool` — NCHW spatial loops with the positional-optional
`null`-means-default convention and int-or-`[h,w]` pair decoding.
`batch_norm` — eval-mode per-channel affine.
`reshape`, `transpose`, `concat` — relabeling ops.
`unsqueeze`, `select` — axis insert/drop with torch's
`rem_euclid`-wrapped dims (a new `rem_euclid` helper in
`internal/helpers.hnl` — `%` alone is the C-style remainder).
`batch_norm_backward` — the `{input, gamma, beta}` grads record.
`derive` — the registered-pairs table (`"batch_norm"` →
`"tensor.batch_norm_backward"`, else `null`).

## Expressibility findings

Writing the surface in the language surfaced its exact boundary. The
first two findings below describe the state *before* the tagged-object
view seam landed (kernel `TaggedValue::object_view`); the seam is the
answer to both.

1. ~~**A `Tensor` is opaque to `.hnl`.**~~ **Resolved by the view
   seam.** A `TaggedValue::Tensor` still has no literal constructor and
   no variant-specific ops in the language, but the *tagged-object
   view* projects it through its canonical wire record: `t.__tensor__`
   field access, `has_field`, `is_obj`, `index`, object spread, and
   record destructuring (`match t { { "__tensor__": inner }: … }`)
   all read an opaque `Tensor` exactly as they read a `.hnl`-authored
   `{"__tensor__": …}` record — the two are the same *value* on the
   wire, and now the same *observation* in the language. The
   projection is generic, not tensor-specific: `Bytes`/`Sym`/
   `Resource` project `{"__bytes__": …}`/`{"__sym__": …}`/
   `{"__resource__": …}` the same way; `Pair`/`Either`/`Arrow` keep
   their existing `.hnl` projections and do not present a record
   view. With the seam, the whole spec — all 31 ops — is pure `.hnl`
   with zero host calls (construction stays in-language too: the wire
   record *is* the tensor's encoding).

2. ~~**The `while` step boundary reserializes values through the wire
   codec.**~~ **Characterized and made harmless by the seam.** The
   promotion is real but was misattributed: `while` *state* passed
   through step-arrow parameters stays a wire record across iterations
   (verified by probe); what decodes is the **captured env** — the
   `closure` mint bakes arrow-free captures through `tagged_to_value`
   into `body_def` `lit` nodes, and every `TaggedValue::from` decode
   (baked captures, static `{"__tensor__": …}` literals, module
   exports, the `call_host` boundary) promotes a well-formed tagged
   record to its opaque variant. That decode is the wire contract
   itself — a `{"__tensor__": …}` record *is* the tensor's
   serialization — so it is a spec property, not a bug, and not
   loop-specific. Under the view it is no longer observable: a
   promoted value still destructures `.__tensor__` identically, so the
   old "destructure before `while` state" discipline is unnecessary.
   (`concat`'s recursive `shapes_of`/`datas_of` shape is kept — it is
   correct `.hnl` regardless; the residual difference is only
   `type_tag`/`typeof`, which still reports `"tensor"` vs `"object"`,
   and `=`/`assert`-style *variant* equality — `==` compares through
   `tagged_to_value` and already equates the two forms.)

3. **Module arrows capture only earlier bindings.** An `export f`
   defined before `g` cannot call `g` ("a name missing from that list
   did not survive into it") — helpers must precede use sites.
   (The earlier claim that *import* bindings do not travel inside
   re-exported arrows was wrong — imported records bake into
   `body_def` like any other capture and survive re-export; the
   literal `"tensor.*"` string callees this file used to carry are
   gone for a different reason: the view made them unnecessary.)

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
src/matmul.hnl           matmul + linear
src/conv.hnl             conv2d + max_pool (optional-param decoding)
src/norm.hnl             batch_norm + batch_norm_backward + derive
src/shape_ops.hnl        reshape + transpose + unsqueeze + select + concat
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
# → [{"name":"zeros","ok":true}, …] — 51 checks, all ok
```

`ci.sh` runs `scripts/hnl-gate.sh` (the shared `hnl` typecheck/lint/
test gate); the sources here are all arrow-dialect, so the gate's
`is_canonical` detector skips them by design — `verify.hnl` is the
honest executable check until a tensor-composing `hnl test` exists.

## License

MIT.
