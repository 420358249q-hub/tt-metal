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
| E1 | Op correctness vs un-absorbed reference | ◐ | `test_ring_joint_expanded_k_per_head_vs_reference`: one chunk at position 0, full or valid-rows-only (2560 tokens; at position 0 this adds the unwritten pad rows, not partial-chunk masking) × `sp_only` / `tp_sharded` (single-slot K) × scalar / metadata × K2.7 / K3 × absorbed / expanded. The SDPA output per (query chip, head) is compared with fp32 attention on its own inputs: 32/32, min PCC 0.99951 (gate 0.999), max abs error ≤ 0.053, finite. Rotated starts and cache-hit sequences are checked only against absorbed on the device: per (device, head) PCC ≥ 0.99962 over rotated, multi-user and partial scenarios and forced schedules. Weight address on cache hits: `test_mla_expanded_k_distinct_layer_weights` |
| E2 | MLA PCC vs CPU reference ≥ absorbed − 0.001; `tp_sharded` == `sp_only` bitwise | ◐ | Asserted directly (the harness's `return_pccs`); over 40 user-runs, expanded is at worst 3e-6 below absorbed and at best 3.4e-5 above. `test_mla_expanded_k_matches_absorbed` (rot-multislab, rot_partial, maxedge-1u × K2.7 / K3 × scalar / metadata × `sp_only` / `tp_sharded`): 24/24 pass. With `return_outputs`, the harness runs its per-iteration 0.98 checks but not the full-region and KV-cache comparisons; the full-region PCC is returned (`return_pccs`) and compared. Expanded vs absorbed: output PCC ≥ 0.99979 (gate 0.999), and per (device, head) SDPA output PCC ≥ 0.999. `tp_sharded` == `sp_only` bitwise with expanded K: 56/56 functional scenarios. The full CPU matrix and the K3 golden trace are not run yet |
| E3 | ≥ 15% faster at K3 512k / K2.7 256k, ≥ 8% at 50k | ◐ | K2.7 256k −16.3%, K2.7 50k −14.1%, K3 50k −12.5%, K3 512k −14.6% (just under 15%: Phase 2) |
| E4 | Multi-layer trace with distinct weights | ◐ | X4 `expanded_k` (K2.7, K3): 3 layers with distinct per-layer `kv_b`, 2 users, a segmented capture with the prefetch, allocation tracking. Traced == eager == `sp_only`, bitwise: 4/4 with absorbed. All three paths are expanded, so X4 would not see a stale weight address; `test_mla_expanded_k_distinct_layer_weights` checks that. Prefetch off is not traced yet |
| E5 | Absorbed path unchanged, selectable | ◐ | Off by default. Absorbed X2 subset (rotated scenarios, both variants and paths): 24/24 bitwise. The Phase 3 prefetch test passes on the same build; X4 `absorbed` passes (with per-layer weight scaling added for E4) |
| E6 | Two reviews per commit | ◐ | Phase 1 (`c19695fd7b5`) reviewed by both, no blocker ([log](reviews/c19695fd7b5.md)); the fix commit is pending review |

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

## Expansion throughput (stop criterion)

Realtime profiler, SDPA program, steady-state K2.7 256k layer (`sp_only`), all on the same q320 / k256 / 3-band
schedule:

| Variant | SDPA device time |
|---|---|
| Absorbed | 19.705 ms |
| Expanded | 16.543 ms |
| Expanded, 1 of every 16 expansion matmuls kept (a local kernel edit, not committed) | 12.744 ms |

- **The expansion takes ≈ 4.05 ms** (16/15 × 3.80 ms), about 24% of the expanded SDPA.
- **Throughput, not resolved:** it is 16.8M MACs per core per K chunk. Going from 576 to 192 wide saves 31.5M QK
  MACs, worth 19.71 − 12.49 = 7.2 ms. That puts the expansion at ≲ 95% of QK's MAC throughput, but only as an upper
  bound: the 7.2 ms also includes the narrower Q read and other non-MAC savings.
  - **Lower end:** a peak-based estimate (HiFi2, about 2 k MACs per cycle per core, 200–340 K chunks per core) puts
    the expansion at roughly 30–50% of FPU peak, against QK's ≈ 70% math utilisation. That ratio is roughly
    0.43–0.71.
  - **So the 60% floor is not established.** What it guards, a net gain, is measured directly (−12.5% to −16.3% at all four
    points, with the expansion still 2× redundant). A blocked (multi-tile) expansion matmul is the lever if
    throughput matters after Phase 2.
- **What's left:** the expansion is still 2× redundant (both Q-chunk cores of a head in a band expand it). Phase 2's
  exchange would save ≈ 2 ms more here, for ≈ −26% SDPA vs absorbed.

**Decision (2026-10-18): Phase 1's stop criterion (expansion ≥ 60% of QK's MAC throughput) is waived, not passed.**
- Its purpose is to stop when the expansion cannot pay for itself. The measured gains already show that it does.
- Phase 2 is the step that tests the remaining question directly: halving the expansion should move the SDPA by about
  as much as the expansion's measured cost.
- This is recorded here for the author to confirm.

## Next steps

1. Review the Phase 1 fix commit (E6).
2. E4: trace the prefetch-off case too.
3. Phase 2, the head-group exchange (PLAN.md §3.4): removes the remaining 2× redundant expansion, which is the path
   to ≥ 15% at K3 512k. Then a chunk retune.
4. Model-level runs with weights (golden trace, transformer tests) with `PREFILL_MLA_EXPANDED_K=1`.
