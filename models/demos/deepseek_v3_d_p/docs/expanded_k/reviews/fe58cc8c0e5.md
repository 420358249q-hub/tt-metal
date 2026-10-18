# Review log: fe58cc8c0e5 (review fixes, model-level results, utilisation probe)

Reviewers: Claude Code, Codex gpt-6-astra (high), both read-only, 2026-10-18. No blocker from either.
- Claude confirmed all 10 resolutions in `reviews/5e55ab86188.md` are applied and every number matches the logs.
- `git apply --check` of the probe patch succeeds on the parent, and pytest does not collect the probe files.

Fixes are in the next commit.

| # | Finding | Reviewer | Severity | Resolution |
|---|---|---|---|---|
| 1 | Probe `.py` files lack the SPDX header (`check-spdx-licenses` would fail) | Claude | major | Added |
| 2 | `expk_util_analyze.py` fails pre-commit: black, autoflake (unused numpy), isort | Claude | major | Fixed; pre-commit passes on both files |
| 3 | "Copy the harness to `tests/`" breaks fixture discovery (`variant` lives in the model's `tests/conftest.py`) | Codex | minor | Header gives the exact commands, into `models/demos/deepseek_v3_d_p/tests/` |
| 4 | Analyser mixes workloads if one CSV holds both cases | Codex | minor | Exits if any core has more than one SDPA launch; one case per CSV documented. No synthetic-CSV test (a scratch probe); rerun on the K3 CSV reproduces STATUS |
| 5 | Golden row lacks the PLAN 0.001 criterion | Claude | minor | Stated |
| 6 | Over-wide line in the Phase 1 throughput notes | Claude | minor | Rewrapped |
| 7 | "X2 golden bitwise" unverified | Claude | minor | Kept: X2 is `test_mla_chunked_prefill_tp_sharded_matches_sp_only`, which asserts bitwise equality |
