# Review log: 44efae7f0a1 (V expansion: fp32 accumulation at HiFi2)

Reviewers: Claude Code, Codex gpt-6-astra (high), both read-only, 2026-10-18. No blocker or major. Both checked:
- fp32 accumulation is kept, and the subblocks still fit the fp32 destination;
- the fidelity is part of `minimal_matmul`'s program hash;
- no scratch lines remain.

Codex notes the broad E5 regression run on the new defaults is still pending (it runs after the row-0 decision).
Fixes are in the next commit.

| # | Finding | Reviewer | Severity | Resolution |
|---|---|---|---|---|
| 1 | Parent PLAN G5 still says "+1% of the forward" | Claude | minor | +0.2% at HiFi2 |
| 2 | STATUS step 6 still lists the HiFi4 costs; one line over-wide | Claude | minor | Rewritten with the HiFi2 delta against bf16 accumulation (K3 +0.2%, K2.7 +0.03%) |
| 3 | "HiFi2 is exact for the bf8 operands" overstates | Claude | minor | "sufficient for the bf8_b operands (7-bit mantissa)" |
| 4 | K3 fp32 PCC quoted as 0.99828 and 0.99826 in different places | Claude | minor | The floor test comment uses the expanded-KV HiFi2 values (0.99828 / 0.99751) |
