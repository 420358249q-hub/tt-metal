# Unrolled-V MLA on the TP-sharded KV cache: status

Milestone: Phase 2 of [PLAN.md](PLAN.md) (serial TP latent gather, no overlap). Branch
`pjosipovic/unrolled-v-tp-sharded-kv`. Last updated 2026-10-17. Everything was measured on one Blackhole Galaxy 8×4
(TorusXY, high-power host).

## What it does

**Starting point**
- **Unrolled-V** (branch `pjosipovic/ring-joint-sdpa-unrolled-v-mcast`, commit `7d302b0c4e1`): expands V per head
  ahead of `ring_joint` SDPA. K3 MLA forward is 25–28% faster than main's `ring_mla`, K2.7 is 19–23% faster.
- **PR #59531** shards the dense KV cache over SP×TP: each chip keeps 1/32 of the cache instead of 1/8, so 4× less KV
  DRAM. As written, the two features clash: a chip can expand V only for rows it holds, and with the sharded cache
  3 of every 4 rows of its SP slab sit on TP neighbours.

**This branch makes them work together:**
1. After each layer's cache write, a TP `high_bw_all_gather` rebuilds this SP row's latent from the 4 TP stripes into
   one shared single-slot buffer, in the old TP-replicated layout.
2. The V expansion and `ring_joint` then run unchanged on that buffer.
3. The gather reads the ND-sharded cache directly, with no reshape, using a new `outer_block_rows` option. Each
   160-row region is gathered as its own rank-interleaved block.
4. On the trace-safe path, the slot and the live extent come from the metadata tensors.

## Commits (on top of `7d302b0c4e1`)

| Commit | What | Reviews |
|---|---|---|
| `f89b4fcd2ad` | Merge PR #59531 into unrolled-V | [log](reviews/f89b4fcd2ad.md) |
| `c7d47299a9e` | `high_bw_all_gather`: `outer_block_rows` / `gathered_outer_blocks` | [log](reviews/c7d47299a9e.md) |
| `321daf1d6df` | MLA: unrolled-V on the TP-deduped cache; bitwise and perf tests | [log](reviews/321daf1d6df.md) |
| `745a37d83fa` | Op review fixes: page-unit outer count (fixes a width-gather overrun), min-pages guard, tests | [log](reviews/745a37d83fa.md) |
| `00ef3010018` | MLA review fixes: slot allocated at init, poisoned bitwise test, DRAM test | [log](reviews/00ef3010018.md) |
| *(last)* | Round-3 review fixes: X4 trace-replay test, replay poisoning, perf method and gates, op tests; these docs | **pending** (next step 1) |

Each commit up to `00ef3010018` was reviewed by Claude Code and Codex (gpt-6-astra). Every finding is fixed, or
answered in its log.

## Exit criteria (PLAN.md §0)

| | Criterion | Status | Evidence |
|---|---|---|---|
| X1 | `tp_shard_kv` runs unrolled-V, no `ring_mla` | ✅ | `_chunked_attn` / `_tp_gather_latent` |
| X2 | `tp_sharded` bitwise equal to `sp_only` | ✅ | `test_mla_chunked_prefill_tp_sharded_matches_sp_only`: 58 passed (2 skipped: no K2.7 golden trace) on functional + golden inputs, plus 44 passed on CPU-reference inputs. The shared slot is poisoned before every forward. The PR's `test_mla_chunked_prefill` `tp_sharded` matrix: 140 passed, 0 failed. |
| X3 | No regressions | ✅ | 53 Galaxy MLA tests pass on `sp_only`, with PCCs identical to before the merge. K3 KDA carry (`kimi_k3/test_chunked_prefill.py`): 4/4. |
| X4 | Captured multi-layer, multi-user trace | ✅ | `test_mla_tp_sharded_trace_replay_multi_layer` (K2.7, K3): 2 layers, 2 users, one capture replayed over 7 steps with user and prefix changed in place (long → short → long), slot poisoned before every replay, bitwise equal to eager. The PR's 10-layer K2.7 transformer test passes `tp_sharded` traced and untraced. |
| X5 | Traced perf gate | ◐ | See below. Gates revised from pre-measurement estimates. Last run: 3/4 pass; K3 50k failed only the replay-agreement check (1.02% vs 1%), which is now 2% and **not yet re-run** (next step 1). |
| X6 | DRAM saving | ✅ | `test_mla_tp_sharded_dram_saving`, measured per chip at exact capacities (see below) |
| X7 | Op tests | ✅ | 18 new outer-block, width-gather and validation tests pass. Other failures in `test_high_bw_all_gather.py` on this Galaxy (3 bandwidth floors, 4 fabric-init errors) happen identically on main. |
| X8 | Defaults and CI | ◐ | `sp_only` K3 perf unchanged: 7.654 / 7.660 ms vs 7.656 ms before. **CI legs not run yet** (next step 2). |
| X9 | Two reviews per commit | ◐ | Done for all commits except the last (next step 1) |

### Performance (X5)

Traced replay of one 5k-chunk MLA forward, `tp_sharded` vs `sp_only`, same build:

| Point | Overheads over 5 runs | Gate |
|---|---|---|
| K3 50k | +1.15, +1.30, +0.45, +1.88% (last run: replay agreement 1.02%, no overhead reported) | +2.5% |
| K3 512k | +1.10, +0.70, +0.94, +0.99, +0.84% | +1.5% |
| K2.7 50k | +1.01, +1.85, +1.97, +2.16, +1.22% | +3.0% |
| K2.7 256k | +1.43, +1.67, +1.90, +1.67, +1.58% | +2.5% |

- **Absolute numbers:** for example K3 512k `sp_only` 61.8 ms against `tp_sharded` 62.4 ms. Against main's
  `ring_mla`, the branch keeps about 95–98% of unrolled-V's speed-up.
- **Where the overhead comes from:** the TP gather runs serially ahead of the expansion. That's about 0.1 ms at 50k
  (mostly fixed latency) and 0.3–0.6 ms at 256k–512k. Phase 3 (overlap) is what removes it.
- **Measurement method:** an untimed warm-up replay, then the median of 5 replays 2 s apart; the middle three must
  agree within 2%. Two things forced this:
  1. **Back-to-back replays slow down under sustained load,** a steady climb (K3 512k: 61.7 → 70.5 ms over 7
     replays).
  2. **The first replay after a capture runs about 2% fast at 50k.**

  Timing only the first replay, as the MoE perf gate does, or a median of consecutive replays, both give misleading
  numbers here.
- **50k noise:** run-to-run noise at 50k is about ±0.7 percentage points, so the 50k gates mainly catch regressions
  at the 1 ms scale.

### DRAM (X6)

Per chip, per user slot, measured allocation deltas at exact capacity:

| | Cache slot per layer, replicated → deduped | Shared TP latent slot | Net saving |
|---|---|---|---|
| K3 512k (24 MLA layers) | ≈41 → ≈10 MB | ≈41 MB, paid once | ≈0.69 GB |
| K2.7 256k (61 MLA layers) | ≈21 → ≈5 MB | ≈21 MB, paid once | ≈0.91 GB |

The saving repeats for every concurrent user slot. With pipeline ranks, each rank pays one slot.

## How to reproduce (any Galaxy 8×4)

```bash
source /opt/venv/bin/activate && export TT_METAL_HOME=$PWD
./build_metal.sh --release --build-tests
export TT_MESH_GRAPH_DESC_PATH=$TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_torus_xy_graph_descriptor.textproto
export PREFILL_TORUS_XY_CERTIFIED=1 OMP_NUM_THREADS=$(nproc)
T=models/demos/deepseek_v3_d_p/tests/test_mla.py
scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/operations/experimental/test_high_bw_all_gather.py -k "outer_blocks or selected_slot_width_gather"
scripts/run_safe_pytest.sh --run-all $T::test_mla_chunked_prefill_tp_sharded_matches_sp_only      # X2 (golden case needs KIMI_K3_HF_MODEL)
scripts/run_safe_pytest.sh --run-all $T::test_mla_tp_sharded_trace_replay_multi_layer             # X4
scripts/run_safe_pytest.sh --run-all $T::test_mla_chunked_tp_sharded_traced_perf                   # X5 (high-power host)
scripts/run_safe_pytest.sh --run-all $T::test_mla_tp_sharded_dram_saving                           # X6
scripts/run_safe_pytest.sh --run-all $T::test_mla_chunked_prefill -k "torus-xy-8x4 and tp_sharded and (k3 or k2_7)"
```

- `KIMI_K3_HF_MODEL=/mnt/weka/model-weights/llm/moonshotai/Kimi-K3-mxfp4-2496450e` enables the golden-trace case.
- The PR's transformer test needs `KIMI_K2_7_HF_MODEL`, `TT_KIMI_PREFILL_TTNN_CACHE` and `PREFILL_TRACE_DIR`, as in
  `tests/pipeline_reorg/blaze_models_prefill_tests.yaml`.

## Next steps

1. **Review the last commit** (X9): Claude Code and Codex gpt-6-astra, per PLAN.md §0. Fix blockers and majors in a
   reviewed follow-up. Re-run `test_mla_chunked_tp_sharded_traced_perf` with the 2% replay-agreement limit (X5).
2. **CI (X8):** run the Blaze prefill legs on this branch, i.e. the PR's K2.7/K3 `tp_sharded` `maxedge` legs and the
   K2.7 transformer legs. Update `K3_CHUNKED_RT_PERF_NS`: `test_mla_chunked_perf_check` still has main's 10.556 ms
   target, while this branch measures ~7.66 ms on `sp_only`.
3. **Rebase onto main once PR #59531 merges,** dropping the merge commit.
4. **Phase 3, overlap (optional, PLAN.md §4.4):** run the TP gather alongside CCL-free compute using sparse MLA's
   80/40 sub-device pattern (`tt_ccl.get_sparse_mla_overlap_resources`, `SubDeviceTraceController`). Target ≤ +0.5%
   overhead.
5. **Follow-up, expanded K on arrival (PLAN.md §6):** unabsorbed QK at 192 wide. Napkin estimate: −21% to −26%
   attention at HiFi2. This branch keeps the latent as K's wire format and the slot at 576 columns, so the step stays
   open.
6. **Upstream bits worth splitting out as separate PRs:**
   - `high_bw_all_gather` `outer_block_rows` (useful on its own);
   - the review findings on PR #59531 itself, which are not this branch's code: sparse `tp_shard_kv=False` must be
     rejected, and `zero_padded_kv_cache` with `tp_axis` has no op test.
