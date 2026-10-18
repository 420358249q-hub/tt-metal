# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# Expansion copy / peer-wait analysis (docs/expanded_k/STATUS.md). Apply expk_copy_probe.patch instead of
# expk_util_probe.patch, run expk_util_harness.py the same way (one case per run), then:
#   python expk_copy_analyze.py profile_log_device.csv
# EXPK-CP is the k_pe copy plus the peer-row copy; EXPK-WAIT is the wait for the peer's rows. Shares are of the SDPA
# kernel on the same core and launch.
import sys
import pandas as pd

df = pd.read_csv(sys.argv[1], skiprows=1, skipinitialspace=True, usecols=range(14), low_memory=False)
df.columns = [c.strip() for c in df.columns]
c = ["PCIe slot", "core_x", "core_y", "RISC processor type", "run host ID"]
for risc in ["TRISC_0", "TRISC_1", "TRISC_2"]:
    d = df[df["RISC processor type"] == risc]
    z = d[d["zone name"].isin(["EXPK-CP", "EXPK-WAIT"]) & (d["time[cycles since reset]"] > 0)].pivot_table(
        index=c, columns="zone name", values="data", aggfunc="max"
    )
    k = d[d["zone name"] == "TRISC-KERNEL"].pivot_table(
        index=c, columns="type", values="time[cycles since reset]", aggfunc="max"
    )
    k = (k["ZONE_END"] - k["ZONE_START"]).rename("kernel")
    m = z.join(k, how="inner")
    if m.empty:
        continue
    out = [f"{risc}: {len(m)} cores, kernel {m['kernel'].median()/1350:.0f} us"]
    for col in ["EXPK-CP", "EXPK-WAIT"]:
        if col in m:
            out.append(f"{col} {(m[col]/m['kernel']).median():.2%} (p90 {(m[col]/m['kernel']).quantile(.9):.2%})")
    print(" | ".join(out))
