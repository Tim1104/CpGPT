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
import pyaging as pya
import anndata as ad

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
    print("\n[3/6] 准备数据格式为AnnData...")
    # pyaging需要AnnData对象
    # AnnData格式：行=样本，列=CpG位点

    # 提取样本ID
    sample_ids = df['sample_id'].astype(str).values

    # 准备数据矩阵（样本 x CpG位点）
    cpg_matrix = df.drop(columns=['sample_id']).values
    cpg_names = df.drop(columns=['sample_id']).columns.values

    # 创建AnnData对象
    adata = ad.AnnData(
        X=cpg_matrix,
        obs=pd.DataFrame({'sample_id': sample_ids}, index=sample_ids),
        var=pd.DataFrame(index=cpg_names)
    )

    print(f"  ✓ 数据形状: {adata.shape} (样本 x CpG位点)")
    print(f"  ✓ 样本数: {adata.n_obs}")
    print(f"  ✓ CpG位点数: {adata.n_vars}")
    
    # 步骤4：运行多个甲基化时钟
    print("\n[4/6] 运行多个甲基化时钟...")
    print("  ℹ️ 这可能需要几分钟...")

    # 定义要运行的时钟（只使用基于DNA甲基化的时钟）
    # 注意：phenoage和grimage需要临床数据，不能只用甲基化数据
    clocks_to_run = ['horvath2013', 'skinandblood']
    clock_display_names = {
        'horvath2013': 'Horvath Clock (2013)',
        'skinandblood': 'Skin & Blood Clock'
    }

    try:
        # 一次性运行所有时钟
        print(f"\n  运行所有时钟: {', '.join([clock_display_names[c] for c in clocks_to_run])}")
        # pyaging会直接修改adata对象，不返回新对象
        pya.pred.predict_age(adata, clock_names=clocks_to_run, verbose=True)
        print(f"\n  ✓ 所有时钟预测完成！")

        # 提取结果
        all_results = pd.DataFrame({'sample_id': sample_ids})

        for clock_name in clocks_to_run:
            if clock_name in adata.obs.columns:
                ages = adata.obs[clock_name].values
                all_results[clock_name] = ages
                print(f"\n  {clock_display_names[clock_name]}:")
                print(f"    范围: {ages.min():.1f} - {ages.max():.1f} 岁")
                print(f"    均值: {ages.mean():.1f} 岁")
            else:
                print(f"\n  ⚠️ {clock_display_names[clock_name]} 未找到结果")
                all_results[clock_name] = np.nan

        # 保存所有时钟的结果
        all_results.to_csv(f"{OUTPUT_DIR}/all_clocks_predictions.csv", index=False)
        print(f"\n  ✓ 所有结果已保存到: {OUTPUT_DIR}/all_clocks_predictions.csv")

        # 为了兼容性，也保存单独的horvath结果
        horvath_results = all_results[['sample_id', 'horvath2013']].copy()
        horvath_results.columns = ['sample_id', 'horvath_age']
        horvath_results.to_csv(f"{OUTPUT_DIR}/horvath_predictions.csv", index=False)

    except Exception as e:
        print(f"\n  ✗ 预测失败: {e}")
        print("\n可能的原因：")
        print("  1. 数据格式不匹配")
        print("  2. 缺少必需的CpG位点")
        print("  3. 数据值范围不正确（应该是0-1的Beta值）")
        import traceback
        traceback.print_exc()
        return
    
    # 步骤5：对比CpGPT结果
    print("\n[5/6] 对比CpGPT预测结果...")

    comparison = None  # 初始化变量

    try:
        cpgpt_results = pd.read_csv(CPGPT_PREDICTIONS)

        # 确保sample_id类型一致，并移除前导零以便匹配
        all_results['sample_id'] = all_results['sample_id'].astype(str).str.lstrip('0')
        cpgpt_results['sample_id'] = cpgpt_results['sample_id'].astype(str).str.lstrip('0')

        # 合并所有结果
        comparison = all_results.merge(cpgpt_results, on='sample_id', how='outer')

        # 如果有实际年龄，也加入
        if Path(METADATA_FILE).exists():
            metadata = pd.read_csv(METADATA_FILE)
            if 'actual_age' in metadata.columns:
                metadata['sample_id'] = metadata['sample_id'].astype(str).str.lstrip('0')
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
        for clock_name in clocks_to_run:
            if clock_name in comparison.columns and not comparison[clock_name].isna().all():
                print(f"\n{clock_display_names[clock_name]}:")
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
            for clock_name in clocks_to_run:
                if clock_name in comparison.columns and not comparison[clock_name].isna().all():
                    comparison[f'{clock_name}_error'] = comparison[clock_name] - comparison['actual_age']
                    mae = abs(comparison[f'{clock_name}_error']).mean()
                    print(f"  {clock_display_names[clock_name]}: {mae:.1f} 岁")

            comparison['cpgpt_error'] = comparison['predicted_age'] - comparison['actual_age']
            print(f"  CpGPT: {abs(comparison['cpgpt_error']).mean():.1f} 岁")

    except Exception as e:
        print(f"  ⚠️ 无法对比结果: {e}")
        import traceback
        traceback.print_exc()

    # 步骤6：生成可视化
    print("\n[6/6] 生成可视化...")

    if comparison is not None and len(comparison) > 0:
        try:
            create_visualizations(comparison, clocks_to_run, clock_display_names)
            print(f"  ✓ 可视化已保存到: {OUTPUT_DIR}/")
        except Exception as e:
            print(f"  ⚠️ 可视化失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"  ⚠️ 没有对比数据，跳过可视化")

    print("\n" + "=" * 80)
    print("完成！")
    print("=" * 80)


def create_visualizations(df, clocks_to_run, clock_display_names):
    """生成对比可视化"""

    # 设置样式
    sns.set_style("whitegrid")
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 定义时钟列表和颜色
    clock_colors = {
        'horvath2013': 'blue',
        'skinandblood': 'brown'
    }

    # 图1：所有时钟 vs 实际年龄（如果有）
    if 'actual_age' in df.columns:
        fig, ax = plt.subplots(figsize=(12, 8))

        # 绘制所有时钟
        for clock_col in clocks_to_run:
            if clock_col in df.columns and not df[clock_col].isna().all():
                ax.scatter(df['actual_age'], df[clock_col],
                          alpha=0.6, s=100, label=clock_display_names[clock_col],
                          color=clock_colors[clock_col])

        # 添加CpGPT
        if 'predicted_age' in df.columns:
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
    colors = []

    for clock_col in clocks_to_run:
        if clock_col in df.columns and not df[clock_col].isna().all():
            clock_data.append(df[clock_col].dropna().values)
            clock_labels.append(clock_display_names[clock_col])
            colors.append(clock_colors[clock_col])

    # 添加CpGPT
    if 'predicted_age' in df.columns:
        clock_data.append(df['predicted_age'].dropna().values)
        clock_labels.append('CpGPT')
        colors.append('red')

    # 如果有实际年龄，也添加
    if 'actual_age' in df.columns:
        clock_data.append(df['actual_age'].dropna().values)
        clock_labels.append('Actual Age')
        colors.append('gray')

    if len(clock_data) == 0:
        print("  ⚠️ 没有数据可以绘制箱线图")
        return

    bp = ax.boxplot(clock_data, labels=clock_labels, patch_artist=True)

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

        # 计算有多少个时钟有误差数据
        n_clocks = sum(1 for col in clocks_to_run if f'{col}_error' in df.columns)
        if 'cpgpt_error' in df.columns:
            n_clocks += 1

        if n_clocks == 0:
            print("  ⚠️ 没有误差数据可以绘制")
            return

        width = 0.8 / n_clocks
        offset = -(n_clocks - 1) * width / 2

        # 绘制所有时钟的误差
        for clock_col in clocks_to_run:
            error_col = f'{clock_col}_error'
            if error_col in df.columns:
                ax.bar(x + offset, df[error_col], width,
                       label=clock_display_names[clock_col], alpha=0.8,
                       color=clock_colors[clock_col])
                offset += width

        # 添加CpGPT误差
        if 'cpgpt_error' in df.columns:
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

        for clock_col in clocks_to_run:
            error_col = f'{clock_col}_error'
            if error_col in df.columns:
                mae_data.append(abs(df[error_col]).mean())
                mae_labels.append(clock_display_names[clock_col])
                mae_colors.append(clock_colors[clock_col])

        # 添加CpGPT
        if 'cpgpt_error' in df.columns:
            mae_data.append(abs(df['cpgpt_error']).mean())
            mae_labels.append('CpGPT')
            mae_colors.append('red')

        if len(mae_data) == 0:
            print("  ⚠️ 没有MAE数据可以绘制")
            return

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

