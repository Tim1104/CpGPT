"""
验证935k甲基化数据格式
Validate 935k methylation data format

使用此脚本检查您的CSV数据是否符合CpGPT的输入要求。
Use this script to check if your CSV data meets CpGPT's input requirements.
"""

import sys
from pathlib import Path

import pandas as pd

# ============================================================================
# 配置
# ============================================================================

DATA_PATH = "./data/935k_samples.csv"  # 修改为您的数据路径

# ============================================================================
# 验证函数
# ============================================================================


def validate_935k_data(data_path):
    """验证935k数据格式"""

    print("=" * 80)
    print("935k甲基化数据格式验证")
    print("=" * 80)

    # 检查文件是否存在
    if not Path(data_path).exists():
        print(f"\n❌ 错误: 文件不存在: {data_path}")
        print("\n请确保数据文件路径正确。")
        return False

    print(f"\n✓ 找到数据文件: {data_path}")

    # 读取数据
    print("\n正在读取数据...")
    try:
        if data_path.endswith(".csv"):
            df = pd.read_csv(data_path)
        elif data_path.endswith((".arrow", ".feather")):
            df = pd.read_feather(data_path)
        else:
            print(f"❌ 不支持的文件格式: {data_path}")
            print("支持的格式: .csv, .arrow, .feather")
            return False

        print(f"✓ 成功读取数据")
    except Exception as e:
        print(f"❌ 读取数据失败: {e}")
        return False

    # 基本信息
    print("\n" + "=" * 80)
    print("数据基本信息")
    print("=" * 80)
    print(f"样本数（行数）: {len(df)}")
    print(f"总列数: {len(df.columns)}")
    print(f"数据大小: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    # 检查列名
    print("\n" + "=" * 80)
    print("列名检查")
    print("=" * 80)
    print(f"前10列: {list(df.columns[:10])}")

    # 识别CpG列
    cpg_cols = [col for col in df.columns if col.startswith("cg")]
    ch_cols = [col for col in df.columns if col.startswith("ch")]
    other_cols = [
        col for col in df.columns if not col.startswith("cg") and not col.startswith("ch")
    ]

    print(f"\nCpG位点列数: {len(cpg_cols)}")
    print(f"CH位点列数: {len(ch_cols)}")
    print(f"其他列数: {len(other_cols)}")

    if len(cpg_cols) == 0:
        print("\n⚠️  警告: 未找到以'cg'开头的CpG位点列！")
        print("请确保列名格式正确（如 cg00000029, cg00000108 等）")
        print(f"\n当前列名示例: {list(df.columns[:5])}")
        return False

    print(f"\n✓ 找到 {len(cpg_cols)} 个CpG位点")
    print(f"示例CpG位点: {cpg_cols[:5]}")

    if len(other_cols) > 0:
        print(f"\n其他列（可能是样本ID或元数据）: {other_cols}")

    # 检查Beta值范围
    print("\n" + "=" * 80)
    print("Beta值检查")
    print("=" * 80)

    # 只检查CpG列
    if len(cpg_cols) > 0:
        cpg_data = df[cpg_cols]

        # 统计信息
        min_val = cpg_data.min().min()
        max_val = cpg_data.max().max()
        mean_val = cpg_data.mean().mean()
        median_val = cpg_data.median().median()

        print(f"最小值: {min_val:.6f}")
        print(f"最大值: {max_val:.6f}")
        print(f"平均值: {mean_val:.6f}")
        print(f"中位数: {median_val:.6f}")

        # 检查范围
        if min_val < 0 or max_val > 1:
            print("\n⚠️  警告: Beta值超出正常范围 [0, 1]！")
            if min_val < -10 and max_val > 10:
                print("数据可能是M值（log2比值），需要转换为Beta值。")
                print("\n转换方法:")
                print("  beta = 2^M / (1 + 2^M)")
            else:
                print("请检查数据预处理步骤。")
        else:
            print("\n✓ Beta值范围正常 [0, 1]")

        # 检查缺失值
        print("\n" + "=" * 80)
        print("缺失值检查")
        print("=" * 80)

        total_values = cpg_data.size
        missing_values = cpg_data.isna().sum().sum()
        missing_rate = (missing_values / total_values) * 100

        print(f"总数据点: {total_values:,}")
        print(f"缺失值数量: {missing_values:,}")
        print(f"缺失率: {missing_rate:.2f}%")

        if missing_rate > 20:
            print("\n⚠️  警告: 缺失率较高（>20%），可能影响预测质量")
            print("建议:")
            print("  1. 检查数据质量控制步骤")
            print("  2. 考虑填补缺失值或移除缺失率高的探针")
        elif missing_rate > 5:
            print("\n⚠️  注意: 缺失率中等（5-20%），建议进行填补")
        else:
            print("\n✓ 缺失率较低")

        # 每个样本的缺失率
        sample_missing = cpg_data.isna().mean(axis=1) * 100
        print(f"\n样本缺失率统计:")
        print(f"  平均: {sample_missing.mean():.2f}%")
        print(f"  最小: {sample_missing.min():.2f}%")
        print(f"  最大: {sample_missing.max():.2f}%")

        high_missing_samples = (sample_missing > 20).sum()
        if high_missing_samples > 0:
            print(f"\n⚠️  {high_missing_samples} 个样本的缺失率 > 20%")

    # 数据预览
    print("\n" + "=" * 80)
    print("数据预览")
    print("=" * 80)
    print("\n前5行，前10列:")
    print(df.iloc[:5, :10])

    # 总结
    print("\n" + "=" * 80)
    print("验证总结")
    print("=" * 80)

    issues = []
    warnings = []

    if len(cpg_cols) == 0:
        issues.append("❌ 未找到CpG位点列")
    elif len(cpg_cols) < 1000:
        warnings.append(f"⚠️  CpG位点数量较少（{len(cpg_cols)}），可能影响预测准确性")

    if len(cpg_cols) > 0:
        if min_val < 0 or max_val > 1:
            issues.append("❌ Beta值超出正常范围 [0, 1]")

        if missing_rate > 20:
            warnings.append(f"⚠️  缺失率较高（{missing_rate:.2f}%）")

    if len(df) < 10:
        warnings.append(f"⚠️  样本数量较少（{len(df)}）")

    # 打印问题
    if len(issues) > 0:
        print("\n发现以下问题:")
        for issue in issues:
            print(f"  {issue}")
        print("\n请修复这些问题后再运行推理。")
        return False

    if len(warnings) > 0:
        print("\n注意事项:")
        for warning in warnings:
            print(f"  {warning}")

    if len(issues) == 0 and len(warnings) == 0:
        print("\n✅ 数据格式验证通过！")
        print("\n您可以运行以下命令进行推理:")
        print("  python examples/935k_zero_shot_inference.py")
        return True
    elif len(issues) == 0:
        print("\n✅ 数据格式基本符合要求，但有一些注意事项。")
        print("\n您可以继续运行推理，但建议先处理上述警告。")
        return True

    return False


# ============================================================================
# 主程序
# ============================================================================

if __name__ == "__main__":
    # 检查命令行参数
    if len(sys.argv) > 1:
        DATA_PATH = sys.argv[1]

    # 运行验证
    success = validate_935k_data(DATA_PATH)

    # 退出码
    sys.exit(0 if success else 1)

