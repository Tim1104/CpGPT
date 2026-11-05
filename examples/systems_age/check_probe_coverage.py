"""Systems Age (methylCIPHER) 所需 CpG 在 935k 样本中的覆盖率。

用法: python check_probe_coverage.py PROBES.rda SAMPLE.csv [SAMPLE.csv ...]
样本格式: 行=样本, 列=CpG ID (首列为样本名)。
"""
import sys
import pandas as pd
import pyreadr

probes = pyreadr.read_r(sys.argv[1])["SystemsAge_CpGs"].iloc[:, 0]
need = set(probes)
print(f"Systems Age 需要探针: {len(probes)} (唯一 {len(need)})")

for f in sys.argv[2:]:
    d = pd.read_csv(f, index_col=0)
    have = set(d.columns)
    hit = need & have
    miss = sorted(need - have)
    sub = d[sorted(hit)]
    print(f"\n== {f}: {d.shape[0]} 样本, {d.shape[1]} 列")
    print(f"  覆盖 {len(hit)}/{len(need)} = {len(hit)/len(need):.2%}")
    print(f"  命中探针中 NaN 比例: {sub.isna().mean().mean():.4%}")
    print(f"  缺失探针前 5 个: {miss[:5]}")
    if miss:
        pd.Series(miss).to_csv(f.rsplit('/',1)[-1] + ".missing_probes.txt", index=False, header=False)
