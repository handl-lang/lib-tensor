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

Both expose the same op surface — the `*_backward` adjoints mirror
host-tensor 1:1 (`elementwise`/`matmul`/`conv`), and `verify.hnl`
checks all of them against the host under a composed runtime.
`host-tensor/src/index.hnl` is the interface contract; `src/ops/` the
reference impl this package mirrors line-for-line — same file map on
both sides (`codec`/`elementwise`/`matmul`/`conv`/`norm`/`shape_ops`,
with `derive` split out of `norm` in Rust since `DERIV_PAIRS` spans
domains) — and the exact f64 expression order, so results are
bit-identical, not just mathematically equal.

## Layout

| path | role |
| --- | --- |
| `src/index.hnl` | `from "lib-tensor" import ltensor` binds the whole surface |
| `src/{codec,elementwise,matmul,conv,norm,shape_ops}.hnl` | the op mirror (host-tensor file map) |
| `src/aten/` | the ATen-style op surface (`binary`/`linalg`/`nn`/`reduce`/`unary`/…) |
| `src/internal/` | `aten.hnl`, `helpers.hnl` |
| `verify.hnl` | composed-runtime check of all ops vs the host |
| `tools/` | `aten_port_gen.py`, `verify_gen.py` generators |
| `docs/reports/` | verification reports |
| `pkg.hnl`, `scripts/hnl-gate.sh`, `ci.sh` | manifest / gate / bupha entry |

## Contract

- Values are the *wire form* `{"__tensor__": {"shape": [d, …],
  "data": [x, …], "dtype": "f64"}}`; opaque `TaggedValue::Tensor`
  inputs read the same record through `TaggedValue::object_view`, so
  the package makes **no host calls at all**. See README/`index.hnl`
  header for expressibility findings (capture-boundary decode,
  reserved names).
- The dep on host-tensor is declared in `.handl.toml` (`git` +
  `subdir`) — `pkg.hnl` deps resolve to `../lib-<name>` checkouts,
  which `host-tensor` is not.

## Verification

```bash
HNL=../host-cli/target/debug/hnl scripts/hnl-gate.sh
./verify.hnl   # composed-runtime parity check vs host-tensor (see README)
```

## License

MIT.

<!-- myco:agent-awareness -->

## Agent awareness

- Session start / any work discussion: run `myco status --prompt`
  before proposing next steps — know the board first.
- User describes new work → offer to capture it (`myco todo add`,
  `issue new`, `proposal new`). Don't wait to be asked.
- Picking up a task: `myco resume <ID>` is the briefing.
- Wrapping up: `myco handoff <ID>` anything resumable; keep statuses
  honest (claimed → in-progress → blocked/done).

<!-- /myco:agent-awareness -->
