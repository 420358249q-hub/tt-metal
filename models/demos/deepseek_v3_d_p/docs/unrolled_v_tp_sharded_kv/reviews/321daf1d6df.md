# Review log: 321daf1d6df (unrolled-V on the TP-deduped KV cache)

Reviewers: Claude Code, Codex gpt-6-astra (high), both read-only, 2026-10-17. No production-path bug found by
either. Fixes are in the follow-up commit.

| # | Finding | Reviewer | Severity | Resolution |
|---|---|---|---|---|
| 1 | Bitwise cases can't detect a gather that misses older blocks: the shared slot still holds the previous chunk's (correct) data | Claude | major | Fixed: the bitwise test poisons the shared TP latent slot with finite garbage before every forward; the full scenario matrix (multi-user, padoverflow, rot-*, golden trace) is run |
| 2 | Multi-layer slot selection and X4 (captured multi-layer multi-user trace) untested | both | major | X4 is covered by the PR's multi-layer `test_prefill_transformer_chunked` `tp_sharded` traced/notrace runs (layer-indexed slots, real captured trace) |
| 3 | Traced replay output never checked; perf gate single-sample with no repeatability or power guard | both | major | Fixed: replay output must equal the eager output bitwise; two alternating runs per layout must agree within 1%; `is_high_power` guard |
| 4 | Bitwise matrix lacks CPU-reference / pretrained inputs | Codex | major | The golden-trace case uses pretrained K3 inputs; the PR's `tp_sharded` cpu/trace matrix gives PCC equal to sp_only for every case |
| 5 | X6 (DRAM saving) has no test | both | minor | Fixed: `test_mla_tp_sharded_dram_saving` measures per-chip allocation deltas and scales to 24 / 61 MLA layers |
| 6 | Slot allocated lazily inside the forward from a 1.2 GB fp32 host tensor | Claude | minor | Fixed: allocated in `set_tp_shard_kv` (outside any forward/capture), from bf16 host zeros |
| 7 | Stale `_chunked_attn` comment about metadata[0]; plan/doc divergence | Claude | minor | Fixed |
| 8 | Slot buffers per distinct cache size are never freed | Claude | minor | Same pattern as the existing chunked buffers (per mesh, per shape); one size in production |
| 9 | Timer includes host enqueue; no input-refresh trace | Claude | minor | Same method as PR #60928's MoE gate; MLA's forward does not free its input, so no refresh is needed |
