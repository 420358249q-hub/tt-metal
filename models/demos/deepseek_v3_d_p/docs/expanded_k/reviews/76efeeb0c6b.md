# Review log: 76efeeb0c6b (expansion matmul in full DST blocks at the SDPA's matmul fidelity)

Reviewers: Claude Code, Codex gpt-6-astra (high), both read-only, 2026-10-18. No blocker or major. Claude checked both
block shapes against the LLK (pairs ct 2 / rt 4 → reuse_a false, in0 rows stepped by TILE_SIZE_B × KT_DIM with kt_dim
1; no pairs ct 4 / rt 2 → reuse_a true, in1 columns stepped by TILE_SIZE_A; DST row-major t × ct + c in both), the
in-tile transpose still on W_UK, the fidelity selection (HiFi2 PCCs identical to every digit), and init order: every PV
setup follows a fresh QK `sdpa_mm_init` in the same step, so no PV reinit reuses the expansion's replay image. Fixes are
in the next commit.

| # | Finding | Reviewer | Severity | Resolution |
|---|---|---|---|---|
| 1 | No committed test runs the expansion at LoFi (pairs 4 × 2 or un-paired 2 × 4) | Codex | minor | Added `test_mla_expanded_k_lofi`: K2.7 and K3, 5k pairs, 5k forced q160 (2 × 4 blocks), 2k pairs; expanded vs absorbed and CPU-reference checks at LoFi; asserts the SDPA program's fidelity is LoFi in both modes. All 6 pass (per-head ≥ 0.99947). Traced LoFi replay is covered by the matrix runs, not a committed test |
| 2 | STATUS 2k LoFi range says −1.4%; K3 55k 2k is −1.3% | Claude | minor | −1.3 to −3.5% |
| 3 | STATUS ceiling "1.0–1.4% at 5k" holds at KV 512k only | Claude | minor | Split by KV length (512k: 1.0–1.4% at 5k, 0.6–2.6% at 2k; 55k: 0.5–2.6%) |
| 4 | "−5 to −7% at 55k" rounds K2.7's −4.8% | Claude | nit | −4.8 to −6.6% |
| 5 | "8-tile DST" assumes bf16 dest | Claude | nit | Says so; the code sizes blocks from `DEST_AUTO_LIMIT` |
| 6 | Comment line breaks mid-sentence (compute_streaming.hpp header and `sdpa_expand_k_chunk` doc) | Claude | nit | Reflowed |
| 7 | Reader hard-codes 16 latent columns where compute uses `latent_rows` | Claude | nit | Kept: pre-existing; the reader's static_assert pins W_UK at 4 × 16 tiles |
