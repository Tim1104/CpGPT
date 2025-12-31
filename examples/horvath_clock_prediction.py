#!/usr/bin/env python3
"""
使用Horvath Clock预测DNA甲基化年龄

这个脚本使用pyaging包中的Horvath Clock来预测年龄，
并与CpGPT的预测结果进行对比。

安装依赖：
    pip install pyaging

使用方法：
    python horvath_clock_prediction.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# 配置
# ============================================================================
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
# 数据路径
DATA_FILE = SCRIPT_DIR / "data" / "Sample1107.arrow"  # 你的935k数据文件
METADATA_FILE =  SCRIPT_DIR / "data" / "sample_metadata.csv"  # 元数据（包含实际年龄）
CPGPT_PREDICTIONS = SCRIPT_DIR /"results"/"935k_enhanced_predictions"/"age_predictions.csv"  # CpGPT预测结果

# 输出路径
OUTPUT_DIR = SCRIPT_DIR / "results" / "horvath_clock_predictions"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# 主函数
# ============================================================================

def main():
    print("=" * 80)
    print("Horvath Clock 年龄预测")
    print("=" * 80)
    
    # 步骤1：检查pyaging是否安装
    print("\n[1/6] 检查依赖...")
    try:
        import pyaging as pya
        print(f"  ✓ pyaging 已安装 (版本: {pya.__version__})")
    except ImportError:
        print("  ✗ pyaging 未安装")
        print("\n请运行以下命令安装：")
        print("  pip install pyaging")
        print("\n或者：")
        print("  pip install pyaging torch")
        return
    
    # 步骤2：读取数据
    print("\n[2/6] 读取935k数据...")
    df = pd.read_feather(DATA_FILE)
    print(f"  ✓ 读取了 {len(df)} 个样本")
    print(f"  ✓ 包含 {len(df.columns)-1} 个CpG位点")
    
    # 步骤3：准备数据格式
    print("\n[3/6] 准备数据格式...")
    # pyaging需要的格式：行=CpG位点，列=样本
    # 当前格式：行=样本，列=CpG位点
    
    # 提取样本ID
    sample_ids = df['sample_id'].values
    
    # 转置数据：CpG位点 x 样本
    cpg_data = df.drop(columns=['sample_id']).T
    cpg_data.columns = sample_ids
    
    print(f"  ✓ 数据形状: {cpg_data.shape} (CpG位点 x 样本)")
    print(f"  ✓ 样本数: {len(sample_ids)}")
    
    # 步骤4：运行多个甲基化时钟
    print("\n[4/6] 运行多个甲基化时钟...")
    print("  ℹ️ 这可能需要几分钟...")

    # 定义要运行的时钟
    clocks_to_run = {
        'horvath2013': 'Horvath Clock (2013)',
        'hannum2013': 'Hannum Clock (2013)',
        'phenoage': 'PhenoAge',
        'grimage': 'GrimAge',
        'skinandblood': 'Skin & Blood Clock'
    }

    all_results = pd.DataFrame({'sample_id': sample_ids})

    for clock_name, clock_display in clocks_to_run.items():
        try:
            print(f"\n  运行 {clock_display}...")
            ages = pya.pred.predict_age(cpg_data, clock=clock_name)
            all_results[clock_name] = ages.values
            print(f"    ✓ {clock_display} 完成")
            print(f"    范围: {ages.min():.1f} - {ages.max():.1f} 岁")
            print(f"    均值: {ages.mean():.1f} 岁")
        except Exception as e:
            print(f"    ✗ {clock_display} 失败: {e}")
            all_results[clock_name] = np.nan

    # 保存所有时钟的结果
    all_results.to_csv(f"{OUTPUT_DIR}/all_clocks_predictions.csv", index=False)
    print(f"\n  ✓ 所有结果已保存到: {OUTPUT_DIR}/all_clocks_predictions.csv")

    # 为了兼容性，也保存单独的horvath结果
    horvath_results = all_results[['sample_id', 'horvath2013']].copy()
    horvath_results.columns = ['sample_id', 'horvath_age']
    horvath_results.to_csv(f"{OUTPUT_DIR}/horvath_predictions.csv", index=False)
    
    # 步骤5：对比CpGPT结果
    print("\n[5/6] 对比CpGPT预测结果...")

    try:
        cpgpt_results = pd.read_csv(CPGPT_PREDICTIONS)

        # 合并所有结果
        comparison = all_results.merge(cpgpt_results, on='sample_id')

        # 如果有实际年龄，也加入
        if Path(METADATA_FILE).exists():
            metadata = pd.read_csv(METADATA_FILE)
            if 'actual_age' in metadata.columns:
                comparison = comparison.merge(
                    metadata[['sample_id', 'actual_age']],
                    on='sample_id',
                    how='left'
                )

        # 保存对比结果
        comparison.to_csv(f"{OUTPUT_DIR}/comparison.csv", index=False)
        print(f"  ✓ 对比结果已保存")

        # 打印统计
        print("\n" + "=" * 80)
        print("预测结果对比")
        print("=" * 80)

        # 显示所有时钟的统计
        for clock_name, clock_display in clocks_to_run.items():
            if clock_name in comparison.columns and not comparison[clock_name].isna().all():
                print(f"\n{clock_display}:")
                print(f"  均值: {comparison[clock_name].mean():.1f} 岁")
                print(f"  标准差: {comparison[clock_name].std():.1f} 岁")
                print(f"  范围: {comparison[clock_name].min():.1f} - {comparison[clock_name].max():.1f} 岁")

        print("\nCpGPT:")
        print(f"  均值: {comparison['predicted_age'].mean():.1f} 岁")
        print(f"  标准差: {comparison['predicted_age'].std():.1f} 岁")
        print(f"  范围: {comparison['predicted_age'].min():.1f} - {comparison['predicted_age'].max():.1f} 岁")

        if 'actual_age' in comparison.columns:
            print("\n实际年龄:")
            print(f"  均值: {comparison['actual_age'].mean():.1f} 岁")
            print(f"  标准差: {comparison['actual_age'].std():.1f} 岁")
            print(f"  范围: {comparison['actual_age'].min():.1f} - {comparison['actual_age'].max():.1f} 岁")

            # 计算所有时钟的误差
            print("\n预测误差（MAE）:")
            for clock_name, clock_display in clocks_to_run.items():
                if clock_name in comparison.columns and not comparison[clock_name].isna().all():
                    comparison[f'{clock_name}_error'] = comparison[clock_name] - comparison['actual_age']
                    mae = abs(comparison[f'{clock_name}_error']).mean()
                    print(f"  {clock_display}: {mae:.1f} 岁")

            comparison['cpgpt_error'] = comparison['predicted_age'] - comparison['actual_age']
            print(f"  CpGPT: {abs(comparison['cpgpt_error']).mean():.1f} 岁")

    except Exception as e:
        print(f"  ⚠️ 无法对比结果: {e}")

    # 步骤6：生成可视化
    print("\n[6/6] 生成可视化...")

    try:
        create_visualizations(comparison)
        print(f"  ✓ 可视化已保存到: {OUTPUT_DIR}/")
    except Exception as e:
        print(f"  ⚠️ 可视化失败: {e}")

    print("\n" + "=" * 80)
    print("完成！")
    print("=" * 80)


def create_visualizations(df):
    """生成对比可视化"""

    # 设置样式
    sns.set_style("whitegrid")
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 定义时钟列表和颜色
    clock_columns = {
        'horvath2013': ('Horvath 2013', 'blue'),
        'hannum2013': ('Hannum 2013', 'green'),
        'phenoage': ('PhenoAge', 'orange'),
        'grimage': ('GrimAge', 'purple'),
        'skinandblood': ('Skin & Blood', 'brown')
    }

    # 图1：所有时钟 vs 实际年龄（如果有）
    if 'actual_age' in df.columns:
        fig, ax = plt.subplots(figsize=(12, 8))

        # 绘制所有时钟
        for clock_col, (clock_name, color) in clock_columns.items():
            if clock_col in df.columns and not df[clock_col].isna().all():
                ax.scatter(df['actual_age'], df[clock_col],
                          alpha=0.6, s=100, label=clock_name, color=color)

        # 添加CpGPT
        ax.scatter(df['actual_age'], df['predicted_age'],
                  alpha=0.6, s=100, label='CpGPT', color='red', marker='s')

        # 添加对角线
        min_age = df['actual_age'].min()
        max_age = df['actual_age'].max()
        ax.plot([min_age, max_age], [min_age, max_age], 'k--', alpha=0.3,
                label='Perfect prediction', linewidth=2)

        ax.set_xlabel('Actual Age (years)', fontsize=12)
        ax.set_ylabel('Predicted Age (years)', fontsize=12)
        ax.set_title('All Clocks vs Actual Age', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/all_clocks_vs_actual.png", dpi=300, bbox_inches='tight')
        plt.close()

    # 图2：所有时钟的预测范围对比
    fig, ax = plt.subplots(figsize=(12, 6))

    clock_data = []
    clock_labels = []

    for clock_col, (clock_name, color) in clock_columns.items():
        if clock_col in df.columns and not df[clock_col].isna().all():
            clock_data.append(df[clock_col].dropna().values)
            clock_labels.append(clock_name)

    # 添加CpGPT
    clock_data.append(df['predicted_age'].dropna().values)
    clock_labels.append('CpGPT')

    # 如果有实际年龄，也添加
    if 'actual_age' in df.columns:
        clock_data.append(df['actual_age'].dropna().values)
        clock_labels.append('Actual Age')

    bp = ax.boxplot(clock_data, labels=clock_labels, patch_artist=True)

    # 设置颜色
    colors = [clock_columns[col][1] for col in clock_columns.keys() if col in df.columns and not df[col].isna().all()]
    colors.append('red')  # CpGPT
    if 'actual_age' in df.columns:
        colors.append('gray')  # Actual age

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel('Age (years)', fontsize=12)
    ax.set_title('Age Prediction Distribution by Clock', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/clocks_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 图3：误差对比（如果有实际年龄）
    if 'actual_age' in df.columns:
        fig, ax = plt.subplots(figsize=(14, 6))

        x = np.arange(len(df))
        n_clocks = sum(1 for col in clock_columns.keys() if col in df.columns and not df[col].isna().all()) + 1  # +1 for CpGPT
        width = 0.8 / n_clocks

        offset = -(n_clocks - 1) * width / 2

        # 绘制所有时钟的误差
        for clock_col, (clock_name, color) in clock_columns.items():
            if clock_col in df.columns and not df[clock_col].isna().all():
                error_col = f'{clock_col}_error'
                if error_col in df.columns:
                    ax.bar(x + offset, df[error_col], width,
                           label=clock_name, alpha=0.8, color=color)
                    offset += width

        # 添加CpGPT误差
        ax.bar(x + offset, df['cpgpt_error'], width,
               label='CpGPT', alpha=0.8, color='red')

        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=2)
        ax.set_xlabel('Sample', fontsize=12)
        ax.set_ylabel('Prediction Error (years)', fontsize=12)
        ax.set_title('Prediction Errors by Sample and Clock', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(df['sample_id'], rotation=45, ha='right')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/error_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

        # 图4：误差绝对值对比（MAE）
        fig, ax = plt.subplots(figsize=(10, 6))

        mae_data = []
        mae_labels = []
        mae_colors = []

        for clock_col, (clock_name, color) in clock_columns.items():
            if clock_col in df.columns and not df[clock_col].isna().all():
                error_col = f'{clock_col}_error'
                if error_col in df.columns:
                    mae_data.append(abs(df[error_col]).mean())
                    mae_labels.append(clock_name)
                    mae_colors.append(color)

        # 添加CpGPT
        mae_data.append(abs(df['cpgpt_error']).mean())
        mae_labels.append('CpGPT')
        mae_colors.append('red')

        bars = ax.bar(mae_labels, mae_data, color=mae_colors, alpha=0.8)

        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax.set_ylabel('Mean Absolute Error (years)', fontsize=12)
        ax.set_title('Average Prediction Error by Clock', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45, ha='right')

        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/mae_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    main()

