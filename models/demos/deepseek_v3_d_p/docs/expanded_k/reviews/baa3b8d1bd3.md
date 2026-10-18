# Review log: baa3b8d1bd3 (per-head expanded vs absorbed SDPA check on real-token rows only)

Reviewers: Claude Code, Codex gpt-6-astra (high), both read-only, 2026-10-18. No blocker or major. Both confirmed the
mask's row order matches the SDPA output (flat chip-major positions, SP sharding on dim -2, devices in mesh order
sp * tp_size + tp), that every caller keeps one SDPA call per forward, and that only discarded padding rows are
excluded. Fixes are in the next commit.

| # | Finding | Reviewer | Severity | Resolution |
|---|---|---|---|---|
| 1 | No test proves the mask still catches errors on real rows, per TP rank | Codex | minor | Added host-only `test_mla_per_head_pccs_real_rows` (pad-overflow mask; corrupt real rows on each device fails, corrupt padding passes) |
| 2 | `tp_size = len(a) // sp` does not check divisibility | Claude | minor | Assert added |
| 3 | All devices skipped would raise a bare `ValueError` from `min()` | Claude | minor | Not changed: every iteration has at least one real token (asserted in the harness) |
