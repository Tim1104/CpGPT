"""Systems Age 结果分析：对照实际年龄做偏差检查，并计算各系统相对同龄人的偏差。

用法:
  python analyze_systems_age.py systems_age_results.csv ages.csv [OUT_DIR]
ages.csv 需要列: sample,age   (sample 形如 000366 或 000366.AVG_Beta; 缺实际年龄的样本可不列)

输出 (OUT_DIR, 默认 ./sa_analysis):
  age_check.csv       每个样本的 Age_prediction 偏差
  system_residuals.csv  各系统分数对实际年龄回归后的残差 (仅当有年龄的样本 >= 30)
  age_check.png       预测年龄 vs 实际年龄
  样本量不足时只输出偏差，并明确提示不做统计推断。
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

MIN_N_REGRESSION = 30
SYSTEMS = ["Blood", "Brain", "Inflammation", "Heart", "Hormone", "Immune",
           "Kidney", "Liver", "Metabolic", "Lung", "MusculoSkeletal"]


def norm_id(s):
    return s.astype(str).str.replace(r"\.AVG_Beta$", "", regex=True)


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    out = Path(sys.argv[3] if len(sys.argv) > 3 else "sa_analysis")
    out.mkdir(parents=True, exist_ok=True)

    res = pd.read_csv(sys.argv[1])
    res["sample"] = norm_id(res["sample"])
    ages = pd.read_csv(sys.argv[2], dtype={"sample": str})
    ages["sample"] = norm_id(ages["sample"])
    df = res.merge(ages[["sample", "age"]], on="sample", how="left")
    known = df.dropna(subset=["age"]).copy()
    print(f"样本 {len(df)} 个，其中有实际年龄 {len(known)} 个")
    if known.empty:
        sys.exit("没有可用的实际年龄，无法分析")

    known["bias"] = known["Age_prediction"] - known["age"]
    known[["sample", "age", "Age_prediction", "bias", "SystemsAge"]].to_csv(out / "age_check.csv", index=False)
    print(known[["sample", "age", "Age_prediction", "bias"]].round(2).to_string(index=False))
    print(f"平均偏差 {known['bias'].mean():+.2f} 岁, MAE {known['bias'].abs().mean():.2f} 岁")

    n = len(known)
    if n >= 3:
        r = np.corrcoef(known["age"], known["Age_prediction"])[0, 1]
        print(f"Pearson r = {r:.3f} (n={n})")

    if n < MIN_N_REGRESSION:
        print(f"\n[提示] 有实际年龄的样本仅 {n} 个 (< {MIN_N_REGRESSION})，"
              "不做按年龄回归的系统残差，也不应据此做统计结论。")
    else:
        resid = known[["sample", "age"]].copy()
        for c in SYSTEMS + ["SystemsAge"]:
            slope, icpt = np.polyfit(known["age"], known[c], 1)
            resid[c + "_resid"] = known[c] - (slope * known["age"] + icpt)
        resid.to_csv(out / "system_residuals.csv", index=False)
        print(f"系统残差已写入 {out/'system_residuals.csv'}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(known["age"], known["Age_prediction"])
        lim = [min(known["age"].min(), known["Age_prediction"].min()) - 5,
               max(known["age"].max(), known["Age_prediction"].max()) + 5]
        ax.plot(lim, lim, "--", color="gray")
        ax.set(xlim=lim, ylim=lim, xlabel="Chronological age", ylabel="Predicted age",
               title=f"Age check (n={n})")
        fig.tight_layout()
        fig.savefig(out / "age_check.png", dpi=150)
    except ImportError:
        print("未安装 matplotlib，跳过绘图")


if __name__ == "__main__":
    main()
