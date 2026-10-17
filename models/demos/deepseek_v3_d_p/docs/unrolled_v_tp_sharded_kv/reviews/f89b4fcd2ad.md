# Review log: f89b4fcd2ad (merge PR #59531 into unrolled-V)

Reviewers: Claude Code (fresh agent), Codex gpt-6-astra (high), both read-only, 2026-10-17.

| # | Finding | Reviewer | Severity | Resolution |
|---|---|---|---|---|
| 1 | Kept dense `assert not self.tp_shard_kv` makes the PR's `tp_sharded` CI legs (blaze yaml :99, :208, :391) and `PREFILL_TP_SHARD_KV=1` runner runs fail on the first dense chunk, after the weight load | both | major | Fixed by the Phase 2 commit: the assert is removed and dense `tp_shard_kv` runs through the TP latent gather |
| 2 | Galaxy Mistral4 perf selector `_CMD_MISTRAL4_CHUNKED_8X4` not pinned to `sp_only` | Claude | major | Fixed in the follow-up commit (`and sp_only`) |
| 3 | `set_tp_shard_kv(False)` is accepted for sparse (DSA) layers, whose gather always treats the cache as SP×TP striped | Codex | major | Fixed in the follow-up commit: sparse on a TP mesh rejects `False` |
| 4 | The merge re-adds `--wrapper-invocation` that #60746 removed on purpose | Claude | minor | Fixed: removed from the K2.7/K3 commands, matching main |
| 5 | Stale comments describe dense dedup as split-KV `ring_mla`; `_declare_q_sequence_topology` has no callers | Claude | minor | Fixed: messages updated, dead helper removed |
| 6 | Phase 1 exit "bitwise equal to pre-rebase" was not shown | Claude | minor | Checked: all 53 Galaxy MLA tests pass on `sp_only`, and every logged output and KV-cache PCC (K3, K3 golden 55k, K2.7, Mistral4 chunked) is identical to the pre-merge run. The bitwise `tp_sharded` vs `sp_only` test is added in Phase 2 |
| 7 | X3 needs the K3 KDA-carry test on `tp_sharded`, which can't pass before Phase 2 | Claude | minor | Plan: the `tp_sharded` part of X3 is checked at Phase 2 |
| 8 | TP-aware `zero_padded_kv_cache` has no direct op test (`tp_axis` never passed in its unit test) | Codex | minor | Belongs to PR #59531's op change. Covered indirectly by the `tp_sharded` padoverflow model tests in Phase 2; flagged to the PR |
