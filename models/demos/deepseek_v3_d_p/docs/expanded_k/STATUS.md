# Expanded K on arrival: status

Milestone: Phase 1 of [PLAN.md](PLAN.md). Every core expands its head's K itself, with no exchange. Branch
`pjosipovic/unrolled-v-tp-sharded-kv`. Last updated 2026-10-18. Everything was measured on one Blackhole Galaxy 8×4
(TorusXY, high-power host). The feature is off by default: `PREFILL_MLA_EXPANDED_K=1` turns it on.

## What it does

`ring_joint` SDPA takes a new optional `k_expand_weight`, which is W_UK per head (`wkv_b1` as stored). K stays the
shared 576-wide latent on every data-movement path: DRAM, the SP ring gather and the row multicast. For each K chunk,
the compute kernel builds the per-head K^T chunk the QK reads (`sdpa_expand_k_chunk`, `compute_streaming.hpp`):
- **No-PE rows:** 4 tile rows of c_kv @ W_UK^T. These are per-tile matmuls, in1 is transposed in-tile (the QK's own
  flag), and the result is packed in K^T tile order.
- **PE rows:** the 2 PE tile rows of the latent, copied.

The QK then runs unchanged at 192 wide (6 tiles), against 576 (18 tiles) absorbed.

The rest of the change:
- **Q:** un-absorbed, `[q_nope | q_pe]`. `_q_stem` skips `wkv_b1`.
- **W_UK delivery:** the reader pushes the core's head's W_UK with every Q push, and compute pops it with Q.
- **Validation:** `k_expand_weight` requires causal self-attention, one shared K head, a separate per-head V,
  streaming compute, batch 1, and no joint, sink, sliding, circular, split or balanced attention.
- **Hash:** the weight's spec is hashed; its address is a runtime argument.
- **Chunks:** K3 runs q320 / k256 in 2 K-split bands (k512 does not fit L1 beside the expanded K^T CB); K2.7 runs
  q320 / k256 in 3 bands (unchanged).

## Exit criteria (PLAN.md §0)

| | Criterion | Status | Evidence |
|---|---|---|---|
| E1 | Op correctness vs un-absorbed reference | ◐ | Checked through MLA (E2). A dedicated op-level test is not written yet |
| E2 | MLA PCC vs CPU reference ≥ absorbed − 0.001; `tp_sharded` == `sp_only` bitwise | ◐ | `test_mla_expanded_k_matches_absorbed` (rot-multislab, rot_partial, maxedge-1u × K2.7 / K3 × scalar / metadata × `sp_only` / `tp_sharded`): 24/24 pass. Each passes the harness's CPU-reference check, and expanded vs absorbed is PCC ≥ 0.99979 (gate 0.999). `tp_sharded` == `sp_only` bitwise with expanded K: 56/56 functional scenarios. The full CPU matrix and the K3 golden trace are not run yet |
| E3 | ≥ 15% faster at K3 512k / K2.7 256k, ≥ 8% at 50k | ◐ | K2.7 256k −16.3%, K2.7 50k −14.1%, K3 50k −12.5%, K3 512k −14.6% (just under 15%: Phase 2) |
| E4 | Multi-layer trace with distinct weights | ☐ | — |
| E5 | Absorbed path unchanged, selectable | ◐ | Off by default. Absorbed X2 subset (rotated scenarios, both variants and paths): 24/24 bitwise. The Phase 3 prefetch test passes on the same build; X4 was not re-run |
| E6 | Two reviews per commit | ☐ | Pending for the Phase 1 commit |

## Accuracy (E2)

PCC vs the CPU reference, from the harness's own check:

| Scenario | Absorbed | Expanded |
|---|---|---|
| K2.7 rot-multislab, scalar, `sp_only` | 0.99613 | 0.99616 |
| K2.7 rot_partial, metadata, `tp_sharded` | 0.99660 | 0.99661 |
| K3 rot-multislab, scalar, `sp_only` (k256) | 0.99584 | 0.99586 |
| K3 rot-multislab, scalar, `sp_only` (q320 / 2 bands) | 0.99584 | 0.99587 |

## Performance (E3)

Traced replays of a steady-state layer (layer 1 of 3), the parent plan's X5 method, `sp_only` layout. One run each.

| Point | Absorbed (unrolled-V) | Expanded K, Phase 1 | Change | main `ring_mla` |
|---|---|---|---|---|
| K3 50k | 7.582 ms | 6.630 ms | −12.5% | 10.33 ms |
| K3 512k | 61.779 ms | 52.745 ms | −14.6% | 89.61 ms |
| K2.7 50k | 5.467 ms | 4.689 ms | −14.1% | 6.83 ms |
| K2.7 256k | 22.127 ms | 18.519 ms | −16.3% | 29.46 ms |

**`tp_sharded`** with the prefetch: K3 512k 52.822 ms, K2.7 256k 18.329 ms.

**Notes on these numbers:**
- **Not yet expanded once per device.** In Phase 1 each core expands its head's whole K chunk. With K-split bands,
  each band's 2 cores per head both do it, so expansion is still 2× redundant.
- **K3's schedule.** K3 at its absorbed schedule (q160, 4 cores per head, k256) was +11.5% slower (8.457 ms at 50k).
  The q320 / 2-band schedule is what made Phase 1 a gain for K3.

## Next steps

1. Review the Phase 1 commit (E6).
2. Op-level test (E1): torch reference, K-split bands, chunk 0, cache-hit address patching.
3. E4: a 3-layer trace with distinct weights, prefetch on and off.
4. Phase 2, the head-group exchange (PLAN.md §3.4): removes the remaining 2× redundant expansion, which is the path
   to ≥ 15% at K3 512k. Then a chunk retune.
5. In-situ expansion throughput (MAC/s) from the realtime profiler, for the stop criterion.
