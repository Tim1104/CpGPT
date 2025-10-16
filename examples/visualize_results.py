"""
可视化已有预测结果
Visualize existing prediction results

如果您已经运行过935k_zero_shot_inference.py并获得了预测结果，
可以使用此脚本快速生成可视化图表和分析报告。

If you have already run 935k_zero_shot_inference.py and obtained prediction results,
you can use this script to quickly generate visualization charts and analysis reports.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# 设置中文字体支持
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_style("whitegrid")
sns.set_palette("husl")

# ============================================================================
# 配置参数
# ============================================================================

RESULTS_DIR = "./results/935k_predictions"
COMBINED_RESULTS_PATH = f"{RESULTS_DIR}/combined_predictions.csv"
FIGURES_DIR = f"{RESULTS_DIR}/figures"
REPORT_PATH = f"{RESULTS_DIR}/analysis_report.html"

# 创建图表目录
Path(FIGURES_DIR).mkdir(parents=True, exist_ok=True)

# ============================================================================
# 检查数据文件
# ============================================================================

print("=" * 80)
print("可视化935k预测结果")
print("=" * 80)

if not Path(COMBINED_RESULTS_PATH).exists():
    print(f"\n❌ 错误: 未找到预测结果文件: {COMBINED_RESULTS_PATH}")
    print("\n请先运行以下命令生成预测结果:")
    print("  python examples/935k_zero_shot_inference.py")
    sys.exit(1)

print(f"\n✓ 找到预测结果文件: {COMBINED_RESULTS_PATH}")

# ============================================================================
# 加载数据
# ============================================================================

print("\n加载预测结果...")
combined_results = pd.read_csv(COMBINED_RESULTS_PATH)
print(f"  ✓ 加载了 {len(combined_results)} 个样本的预测结果")
print(f"\n数据预览:")
print(combined_results.head())

# ============================================================================
# 生成可视化图表
# ============================================================================

print("\n" + "=" * 80)
print("生成可视化图表")
print("=" * 80)


def create_age_distribution_plot(age_results, save_path):
    """创建年龄分布图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 直方图
    axes[0].hist(
        age_results["predicted_age"], bins=30, color="skyblue", edgecolor="black", alpha=0.7
    )
    axes[0].axvline(
        age_results["predicted_age"].mean(),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f'平均: {age_results["predicted_age"].mean():.1f}岁',
    )
    axes[0].set_xlabel("预测年龄 (岁)", fontsize=12)
    axes[0].set_ylabel("样本数量", fontsize=12)
    axes[0].set_title("年龄分布直方图", fontsize=14, fontweight="bold")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 箱线图
    box = axes[1].boxplot(
        age_results["predicted_age"],
        vert=True,
        patch_artist=True,
        labels=["预测年龄"],
        boxprops=dict(facecolor="lightblue", alpha=0.7),
        medianprops=dict(color="red", linewidth=2),
    )
    axes[1].set_ylabel("年龄 (岁)", fontsize=12)
    axes[1].set_title("年龄分布箱线图", fontsize=14, fontweight="bold")
    axes[1].grid(True, alpha=0.3, axis="y")

    # 添加统计信息
    stats_text = f"中位数: {age_results['predicted_age'].median():.1f}岁\n"
    stats_text += f"范围: {age_results['predicted_age'].min():.1f} - {age_results['predicted_age'].max():.1f}岁"
    axes[1].text(
        1.15, age_results["predicted_age"].median(), stats_text, fontsize=10, va="center"
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ 年龄分布图: {save_path}")


def create_cancer_distribution_plot(cancer_results, save_path):
    """创建癌症预测分布图"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 概率分布直方图
    axes[0].hist(
        cancer_results["cancer_probability"], bins=30, color="coral", edgecolor="black", alpha=0.7
    )
    axes[0].axvline(0.5, color="red", linestyle="--", linewidth=2, label="阈值 (0.5)")
    axes[0].set_xlabel("癌症概率", fontsize=12)
    axes[0].set_ylabel("样本数量", fontsize=12)
    axes[0].set_title("癌症概率分布", fontsize=14, fontweight="bold")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 预测结果饼图
    cancer_counts = cancer_results["cancer_prediction"].value_counts()
    colors = ["lightgreen", "lightcoral"]
    labels = [f'正常 ({cancer_counts.get(0, 0)})', f'癌症 ({cancer_counts.get(1, 0)})']
    explode = (0.05, 0.05) if len(cancer_counts) == 2 else (0.05,)
    axes[1].pie(
        cancer_counts,
        labels=labels,
        autopct="%1.1f%%",
        colors=colors[: len(cancer_counts)],
        explode=explode,
        startangle=90,
        textprops={"fontsize": 12},
    )
    axes[1].set_title("癌症预测分类", fontsize=14, fontweight="bold")

    # 风险分层
    risk_categories = pd.cut(
        cancer_results["cancer_probability"],
        bins=[0, 0.2, 0.5, 0.8, 1.0],
        labels=["低风险", "中低风险", "中高风险", "高风险"],
    )
    risk_counts = risk_categories.value_counts().sort_index()

    colors_risk = ["green", "yellowgreen", "orange", "red"]
    bars = axes[2].barh(range(len(risk_counts)), risk_counts.values, color=colors_risk, alpha=0.7)
    axes[2].set_yticks(range(len(risk_counts)))
    axes[2].set_yticklabels(risk_counts.index)
    axes[2].set_xlabel("样本数", fontsize=12)
    axes[2].set_title("癌症风险分层", fontsize=14, fontweight="bold")
    axes[2].grid(True, alpha=0.3, axis="x")

    # 添加数值标签
    for i, v in enumerate(risk_counts.values):
        axes[2].text(v + 0.5, i, str(v), va="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ 癌症分布图: {save_path}")


def create_correlation_plot(combined_results, save_path):
    """创建年龄与癌症相关性图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 散点图
    colors = ["green" if x == 0 else "red" for x in combined_results["cancer_prediction"]]
    axes[0].scatter(
        combined_results["predicted_age"],
        combined_results["cancer_probability"],
        c=colors,
        alpha=0.6,
        s=50,
        edgecolors="black",
        linewidth=0.5,
    )
    axes[0].axhline(0.5, color="red", linestyle="--", linewidth=1.5, alpha=0.5)
    axes[0].set_xlabel("预测年龄 (岁)", fontsize=12)
    axes[0].set_ylabel("癌症概率", fontsize=12)
    axes[0].set_title("年龄 vs 癌症概率", fontsize=14, fontweight="bold")
    axes[0].grid(True, alpha=0.3)

    # 年龄分组的癌症率
    age_bins = [0, 30, 40, 50, 60, 70, 100]
    age_labels = ["<30", "30-40", "40-50", "50-60", "60-70", "70+"]
    combined_results["age_group"] = pd.cut(
        combined_results["predicted_age"], bins=age_bins, labels=age_labels
    )

    cancer_rate_by_age = combined_results.groupby("age_group")["cancer_prediction"].mean() * 100
    sample_count_by_age = combined_results.groupby("age_group").size()

    x_pos = np.arange(len(age_labels))
    bars = axes[1].bar(x_pos, cancer_rate_by_age, color="steelblue", alpha=0.7, edgecolor="black")

    for i, (bar, count) in enumerate(zip(bars, sample_count_by_age)):
        height = bar.get_height()
        axes[1].text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 1,
            f"n={count}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    axes[1].set_xlabel("年龄组", fontsize=12)
    axes[1].set_ylabel("癌症预测率 (%)", fontsize=12)
    axes[1].set_title("不同年龄组的癌症预测率", fontsize=14, fontweight="bold")
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(age_labels)
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ 相关性分析图: {save_path}")


# 生成图表
print("\n1. 生成年龄分布图...")
create_age_distribution_plot(combined_results, f"{FIGURES_DIR}/age_distribution.png")

print("\n2. 生成癌症分布图...")
create_cancer_distribution_plot(combined_results, f"{FIGURES_DIR}/cancer_distribution.png")

print("\n3. 生成相关性分析图...")
create_correlation_plot(combined_results, f"{FIGURES_DIR}/age_cancer_correlation.png")

# ============================================================================
# 生成简化版HTML报告
# ============================================================================

print("\n" + "=" * 80)
print("生成HTML分析报告")
print("=" * 80)

# 计算统计数据
age_stats = combined_results["predicted_age"].describe()
cancer_count = combined_results["cancer_prediction"].sum()
normal_count = len(combined_results) - cancer_count
cancer_rate = (cancer_count / len(combined_results)) * 100

# 高风险样本
high_risk = combined_results[combined_results["cancer_probability"] > 0.8].sort_values(
    "cancer_probability", ascending=False
)

html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>935k甲基化数据分析报告</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; background: #f5f5f5; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px; margin-bottom: 30px; }}
        .section {{ background: white; padding: 25px; margin-bottom: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ margin: 0; font-size: 2.5em; }}
        h2 {{ color: #667eea; border-bottom: 3px solid #667eea; padding-bottom: 10px; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }}
        .stat-card {{ background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); padding: 20px; border-radius: 8px; }}
        .stat-card .value {{ font-size: 2em; font-weight: bold; color: #667eea; }}
        .figure {{ margin: 30px 0; text-align: center; }}
        .figure img {{ max-width: 100%; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #667eea; color: white; }}
        .alert {{ padding: 15px; margin: 20px 0; border-radius: 5px; border-left: 4px solid; }}
        .alert-info {{ background-color: #e3f2fd; border-color: #2196f3; }}
        .alert-warning {{ background-color: #fff3e0; border-color: #ff9800; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🧬 935k甲基化数据分析报告</h1>
        <p>基于CpGPT的零样本推理分析</p>
    </div>

    <div class="section">
        <h2>📊 执行摘要</h2>
        <div class="stats-grid">
            <div class="stat-card">
                <h4>总样本数</h4>
                <div class="value">{len(combined_results)}</div>
            </div>
            <div class="stat-card">
                <h4>平均年龄</h4>
                <div class="value">{age_stats['mean']:.1f}</div>
                <div>岁</div>
            </div>
            <div class="stat-card">
                <h4>癌症预测率</h4>
                <div class="value">{cancer_rate:.1f}%</div>
            </div>
            <div class="stat-card">
                <h4>高风险样本</h4>
                <div class="value">{len(high_risk)}</div>
            </div>
        </div>
    </div>

    <div class="section">
        <h2>🎂 年龄预测分析</h2>
        <div class="figure">
            <img src="figures/age_distribution.png" alt="年龄分布">
        </div>
        <p><strong>统计摘要：</strong> 平均年龄 {age_stats['mean']:.1f} 岁，范围 {age_stats['min']:.1f} - {age_stats['max']:.1f} 岁</p>
    </div>

    <div class="section">
        <h2>🏥 癌症预测分析</h2>
        <div class="figure">
            <img src="figures/cancer_distribution.png" alt="癌症分布">
        </div>
        <p><strong>预测结果：</strong> {normal_count} 个正常样本，{cancer_count} 个癌症样本</p>
        {f'<div class="alert alert-warning"><strong>⚠️ 发现 {len(high_risk)} 个高风险样本（概率>0.8）</strong></div>' if len(high_risk) > 0 else ''}
    </div>

    <div class="section">
        <h2>🔗 年龄与癌症相关性</h2>
        <div class="figure">
            <img src="figures/age_cancer_correlation.png" alt="相关性分析">
        </div>
    </div>

    <div class="section">
        <h2>💡 说明</h2>
        <div class="alert alert-info">
            <p>本报告使用CpGPT预训练模型进行零样本推理，无需微调即可预测年龄和癌症风险。</p>
            <p><strong>注意：</strong> 预测结果仅供科研参考，不能作为临床诊断依据。</p>
        </div>
    </div>
</body>
</html>
"""

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"  ✓ HTML报告已生成: {REPORT_PATH}")

# ============================================================================
# 完成
# ============================================================================

print("\n" + "=" * 80)
print("✅ 可视化完成！")
print("=" * 80)
print(f"\n📁 结果目录: {RESULTS_DIR}")
print(f"\n📈 生成的图表:")
print(f"  - {FIGURES_DIR}/age_distribution.png")
print(f"  - {FIGURES_DIR}/cancer_distribution.png")
print(f"  - {FIGURES_DIR}/age_cancer_correlation.png")
print(f"\n📄 分析报告:")
print(f"  - {REPORT_PATH}")
print(f"\n💡 在浏览器中打开 {REPORT_PATH} 查看完整报告")
print("=" * 80)

