# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# Expansion copy / peer-wait analysis (docs/expanded_k/STATUS.md). Apply expk_copy_probe.patch instead of
# expk_util_probe.patch, run expk_util_harness.py the same way (one case per run), then from the repo root:
#   python models/demos/deepseek_v3_d_p/docs/expanded_k/probes/expk_copy_analyze.py \
#       generated/profiler/.logs/profile_log_device.csv
# EXPK-CP is the k_pe copy plus the peer-row copy; EXPK-WAIT is the wait for the peer's rows. Shares are of the SDPA
# kernel on the same core and launch. Valid readings: EXPK-CP on TRISC_1 (math) and EXPK-WAIT on TRISC_0 (unpack).
# The probe's sum slots are shared with the built-in CB-COMPUTE-WAIT-FRONT (slot 0, unpack) and
# CB-COMPUTE-RESERVE-BACK (slot 1, pack) zones, so EXPK-CP on unpack and EXPK-WAIT on pack are merged into those and
# not reported.
import sys

import pandas as pd

df = pd.read_csv(sys.argv[1], skiprows=1, skipinitialspace=True, usecols=range(14), low_memory=False)
df.columns = [c.strip() for c in df.columns]
c = ["PCIe slot", "core_x", "core_y", "RISC processor type", "run host ID"]
# The valid readings (see above) must be present, or the profile was not taken with this probe and the sum zones on.
VALID = [("TRISC_1", "EXPK-CP"), ("TRISC_0", "EXPK-WAIT")]
for risc, zone in VALID:
    if not ((df["RISC processor type"] == risc) & (df["zone name"] == zone)).any():
        sys.exit(f"no {zone} on {risc}: apply expk_copy_probe.patch and set TT_METAL_PROFILER_SUM=1")
for risc, col in VALID:
    d = df[df["RISC processor type"] == risc]
    z = d[(d["zone name"] == col) & (d["time[cycles since reset]"] > 0)].pivot_table(
        index=c, columns="zone name", values="data", aggfunc="max"
    )
    k = d[d["zone name"] == "TRISC-KERNEL"].pivot_table(
        index=c, columns="type", values="time[cycles since reset]", aggfunc="max"
    )
    k = (k["ZONE_END"] - k["ZONE_START"]).rename("kernel")
    m = z.join(k, how="inner")
    share = m[col] / m["kernel"]
    print(
        f"{col} on {risc}: {len(m)} cores, kernel {m['kernel'].median()/1350:.0f} us, "
        f"share median {share.median():.2%} (p90 {share.quantile(.9):.2%})"
    )
