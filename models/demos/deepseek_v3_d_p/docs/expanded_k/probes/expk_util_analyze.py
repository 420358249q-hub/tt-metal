# Expansion-utilisation analysis (docs/expanded_k/STATUS.md). Usage: python expk_util_analyze.py profile_log_device.csv
import pandas as pd, sys, numpy as np
df = pd.read_csv(sys.argv[1], skiprows=1, skipinitialspace=True, usecols=range(14), low_memory=False)
df.columns = [c.strip() for c in df.columns]
df = df[df["RISC processor type"].isin(["TRISC_0","TRISC_1","TRISC_2"])]
c = ["PCIe slot","core_x","core_y","RISC processor type"]
n = df[df["zone name"].isin(["EXPK-N","QK-N"])].pivot_table(index=c+["run host ID"], columns="zone name", values="data", aggfunc="sum").reset_index()
t = df[df["zone name"].isin(["EXPK-MM","QK-MM"]) & (df["time[cycles since reset]"]>0)].pivot_table(index=c+["run host ID"], columns="zone name", values="data", aggfunc="max").reset_index()
k = df[(df["zone name"]=="TRISC-KERNEL")].pivot_table(index=c+["run host ID"], columns="type", values="time[cycles since reset]", aggfunc="max").reset_index()
k["kernel"] = k["ZONE_END"]-k["ZONE_START"]
m = n.merge(t, on=c+["run host ID"], how="left").merge(k[c+["run host ID","kernel"]], on=c+["run host ID"], how="left").fillna(0)
for risc in ["TRISC_0","TRISC_1","TRISC_2"]:
    r = m[(m["RISC processor type"]==risc) & (m["EXPK-N"]>0)]
    if r.empty: continue
    e = r["EXPK-MM"]/r["EXPK-N"]; q = r["QK-MM"]/r["QK-N"]
    print(f"{risc}: {len(r)} cores | cyc/tile-mm EXPK mean {e.mean():.1f} (p10 {e.quantile(.1):.1f} p90 {e.quantile(.9):.1f}) "
          f"QK mean {q.mean():.1f} (p10 {q.quantile(.1):.1f} p90 {q.quantile(.9):.1f}) | aggregate EXPK {r['EXPK-MM'].sum()/r['EXPK-N'].sum():.1f} QK {r['QK-MM'].sum()/r['QK-N'].sum():.1f}")
    print(f"   kernel {r['kernel'].median()/1350:.0f} us median; share EXPK-MM {(r['EXPK-MM']/r['kernel']).median():.1%}, QK-MM {(r['QK-MM']/r['kernel']).median():.1%}; tiles/core EXPK {r['EXPK-N'].median():.0f} QK {r['QK-N'].median():.0f}; max-kernel core share EXPK {(r.loc[r['kernel'].idxmax(),'EXPK-MM']/r['kernel'].max()):.1%}")
