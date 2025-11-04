"""
报告生成器 - 生成HTML可视化报告
"""

import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# 设置中文字体支持
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_style("whitegrid")
sns.set_palette("husl")


def create_age_distribution_plot(age_results, save_path):
    """创建年龄分布图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 直方图
    axes[0].hist(age_results["predicted_age"], bins=30, color="skyblue", edgecolor="black", alpha=0.7)
    axes[0].axvline(
        age_results["predicted_age"].mean(), color="red", linestyle="--", linewidth=2, label="平均年龄"
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
        whiskerprops=dict(color="blue", linewidth=1.5),
        capprops=dict(color="blue", linewidth=1.5),
    )
    axes[1].set_ylabel("年龄 (岁)", fontsize=12)
    axes[1].set_title("年龄分布箱线图", fontsize=14, fontweight="bold")
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


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
    labels = ["正常", "癌症"]
    explode = (0.05, 0.05) if len(cancer_counts) == 2 else (0.05,)
    axes[1].pie(
        cancer_counts,
        labels=labels[:len(cancer_counts)],
        autopct="%1.1f%%",
        colors=colors[:len(cancer_counts)],
        explode=explode[:len(cancer_counts)],
        startangle=90,
        textprops={"fontsize": 12},
    )
    axes[1].set_title("癌症预测分类", fontsize=14, fontweight="bold")

    # 概率箱线图（按预测分类）
    normal_probs = cancer_results[cancer_results["cancer_prediction"] == 0]["cancer_probability"]
    cancer_probs = cancer_results[cancer_results["cancer_prediction"] == 1]["cancer_probability"]

    box_data = [normal_probs, cancer_probs] if len(cancer_probs) > 0 else [normal_probs]
    box_labels = ["预测正常", "预测癌症"] if len(cancer_probs) > 0 else ["预测正常"]
    
    box = axes[2].boxplot(
        box_data,
        labels=box_labels,
        patch_artist=True,
        boxprops=dict(alpha=0.7),
        medianprops=dict(color="red", linewidth=2),
    )
    if len(box_data) == 2:
        box["boxes"][0].set_facecolor("lightgreen")
        box["boxes"][1].set_facecolor("lightcoral")
    else:
        box["boxes"][0].set_facecolor("lightgreen")
    
    axes[2].axhline(0.5, color="red", linestyle="--", linewidth=1.5, alpha=0.5, label="阈值")
    axes[2].set_ylabel("癌症概率", fontsize=12)
    axes[2].set_title("概率分布（按分类）", fontsize=14, fontweight="bold")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def create_age_cancer_correlation_plot(combined_results, save_path):
    """创建年龄与癌症概率相关性图"""
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
    axes[0].axhline(0.5, color="red", linestyle="--", linewidth=1.5, alpha=0.5, label="癌症阈值")
    axes[0].set_xlabel("预测年龄 (岁)", fontsize=12)
    axes[0].set_ylabel("癌症概率", fontsize=12)
    axes[0].set_title("年龄 vs 癌症概率", fontsize=14, fontweight="bold")
    axes[0].legend(["癌症阈值", "正常", "癌症"])
    axes[0].grid(True, alpha=0.3)

    # 年龄分组的癌症率
    age_bins = [0, 30, 40, 50, 60, 70, 100]
    age_labels = ["<30", "30-40", "40-50", "50-60", "60-70", "70+"]
    combined_results["age_group"] = pd.cut(
        combined_results["predicted_age"], bins=age_bins, labels=age_labels
    )

    cancer_rate_by_age = (
        combined_results.groupby("age_group")["cancer_prediction"].mean() * 100
    )
    sample_count_by_age = combined_results.groupby("age_group").size()

    x_pos = np.arange(len(age_labels))
    bars = axes[1].bar(x_pos, cancer_rate_by_age, color="steelblue", alpha=0.7, edgecolor="black")

    # 在柱子上标注样本数
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


def create_summary_statistics_plot(combined_results, save_path):
    """创建统计摘要图"""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. 年龄统计
    ax1 = fig.add_subplot(gs[0, 0])
    age_stats = combined_results["predicted_age"].describe()
    stats_text = f"""
    样本数: {int(age_stats['count'])}
    平均值: {age_stats['mean']:.1f} 岁
    标准差: {age_stats['std']:.1f} 岁
    最小值: {age_stats['min']:.1f} 岁
    25%分位: {age_stats['25%']:.1f} 岁
    中位数: {age_stats['50%']:.1f} 岁
    75%分位: {age_stats['75%']:.1f} 岁
    最大值: {age_stats['max']:.1f} 岁
    """
    ax1.text(0.1, 0.5, stats_text, fontsize=11, verticalalignment="center", family="monospace")
    ax1.set_title("年龄预测统计", fontsize=12, fontweight="bold")
    ax1.axis("off")

    # 2. 癌症统计
    ax2 = fig.add_subplot(gs[0, 1])
    cancer_stats = combined_results["cancer_probability"].describe()
    cancer_count = combined_results["cancer_prediction"].sum()
    normal_count = len(combined_results) - cancer_count
    cancer_rate = (cancer_count / len(combined_results)) * 100

    cancer_text = f"""
    总样本数: {len(combined_results)}
    预测正常: {normal_count} ({100-cancer_rate:.1f}%)
    预测癌症: {cancer_count} ({cancer_rate:.1f}%)

    癌症概率统计:
    平均值: {cancer_stats['mean']:.3f}
    标准差: {cancer_stats['std']:.3f}
    最小值: {cancer_stats['min']:.3f}
    中位数: {cancer_stats['50%']:.3f}
    最大值: {cancer_stats['max']:.3f}
    """
    ax2.text(0.1, 0.5, cancer_text, fontsize=11, verticalalignment="center", family="monospace")
    ax2.set_title("癌症预测统计", fontsize=12, fontweight="bold")
    ax2.axis("off")

    # 3. 风险分层
    ax3 = fig.add_subplot(gs[0, 2])
    risk_categories = pd.cut(
        combined_results["cancer_probability"],
        bins=[0, 0.2, 0.5, 0.8, 1.0],
        labels=["低风险", "中低风险", "中高风险", "高风险"],
    )
    risk_counts = risk_categories.value_counts().sort_index()

    colors_risk = ["green", "yellowgreen", "orange", "red"]
    ax3.barh(range(len(risk_counts)), risk_counts.values, color=colors_risk[:len(risk_counts)], alpha=0.7)
    ax3.set_yticks(range(len(risk_counts)))
    ax3.set_yticklabels(risk_counts.index)
    ax3.set_xlabel("样本数", fontsize=10)
    ax3.set_title("癌症风险分层", fontsize=12, fontweight="bold")
    ax3.grid(True, alpha=0.3, axis="x")

    # 添加数值标签
    for i, v in enumerate(risk_counts.values):
        ax3.text(v + 0.5, i, str(v), va="center", fontsize=10)

    # 4. 年龄分布
    ax4 = fig.add_subplot(gs[1, :])
    ax4.hist(
        combined_results["predicted_age"],
        bins=50,
        color="skyblue",
        edgecolor="black",
        alpha=0.7,
        density=True,
    )
    ax4.set_xlabel("预测年龄 (岁)", fontsize=11)
    ax4.set_ylabel("密度", fontsize=11)
    ax4.set_title("年龄分布密度图", fontsize=12, fontweight="bold")
    ax4.grid(True, alpha=0.3)

    # 5. 癌症概率分布
    ax5 = fig.add_subplot(gs[2, :2])
    scatter = ax5.scatter(
        range(len(combined_results)),
        combined_results["cancer_probability"],
        c=combined_results["predicted_age"],
        cmap="coolwarm",
        alpha=0.6,
        s=30,
        edgecolors="black",
        linewidth=0.3,
    )
    ax5.axhline(0.5, color="red", linestyle="--", linewidth=1.5, alpha=0.5)
    ax5.set_xlabel("样本索引", fontsize=11)
    ax5.set_ylabel("癌症概率", fontsize=11)
    ax5.set_title("癌症概率分布（颜色表示年龄）", fontsize=12, fontweight="bold")
    ax5.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax5)
    cbar.set_label("预测年龄 (岁)", fontsize=10)

    # 6. 高风险样本表
    ax6 = fig.add_subplot(gs[2, 2])
    high_risk = combined_results[combined_results["cancer_probability"] > 0.8].sort_values(
        "cancer_probability", ascending=False
    )
    if len(high_risk) > 0:
        top_5 = high_risk.head(5)
        table_data = []
        for idx, row in top_5.iterrows():
            table_data.append(
                [
                    str(row["sample_id"])[:10],
                    f"{row['predicted_age']:.1f}",
                    f"{row['cancer_probability']:.3f}",
                ]
            )

        table = ax6.table(
            cellText=table_data,
            colLabels=["样本ID", "年龄", "癌症概率"],
            cellLoc="center",
            loc="center",
            colWidths=[0.4, 0.3, 0.3],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        ax6.set_title(f"高风险样本 (Top 5)", fontsize=12, fontweight="bold")
    else:
        ax6.text(
            0.5, 0.5, "无高风险样本\n(概率>0.8)", ha="center", va="center", fontsize=12
        )
    ax6.axis("off")

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def create_clocks_distribution_plot(clocks_results, save_path):
    """创建表观遗传时钟分布图"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("表观遗传时钟分析", fontsize=20, fontweight="bold", y=0.995)

    clock_names = ["altumage", "dunedinpace", "grimage2", "hrsinchphenoage", "pchorvath2013"]
    clock_labels = {
        "altumage": "AltumAge",
        "dunedinpace": "DunedinPACE (×100)",
        "grimage2": "GrimAge2",
        "hrsinchphenoage": "HRS InCHPhenoAge",
        "pchorvath2013": "PC Horvath 2013"
    }

    for idx, clock_name in enumerate(clock_names):
        ax = axes[idx // 3, idx % 3]
        if clock_name in clocks_results.columns:
            values = clocks_results[clock_name]

            # 直方图
            ax.hist(values, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
            ax.axvline(values.mean(), color='red', linestyle='--', linewidth=2, label=f'均值: {values.mean():.2f}')
            ax.axvline(values.median(), color='green', linestyle='--', linewidth=2, label=f'中位数: {values.median():.2f}')

            ax.set_xlabel(clock_labels[clock_name], fontsize=12)
            ax.set_ylabel("样本数", fontsize=12)
            ax.set_title(f"{clock_labels[clock_name]} 分布", fontsize=14, fontweight="bold")
            ax.legend()
            ax.grid(True, alpha=0.3)

    # 隐藏最后一个子图
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def create_proteins_heatmap(proteins_results, save_path):
    """创建蛋白质水平热图"""
    # 获取蛋白质列
    protein_cols = [col for col in proteins_results.columns if col.startswith('protein_')]

    if len(protein_cols) == 0:
        # 如果没有蛋白质数据，创建空图
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.text(0.5, 0.5, '无蛋白质数据', ha='center', va='center', fontsize=20)
        ax.axis('off')
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        return

    # 限制显示的蛋白质数量
    protein_cols = protein_cols[:20]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle("血浆蛋白质水平分析", fontsize=20, fontweight="bold")

    # 热图
    protein_data = proteins_results[protein_cols].T
    im = ax1.imshow(protein_data, aspect='auto', cmap='RdYlBu_r')
    ax1.set_xlabel("样本", fontsize=12)
    ax1.set_ylabel("蛋白质", fontsize=12)
    ax1.set_title("蛋白质水平热图", fontsize=14, fontweight="bold")
    ax1.set_yticks(range(len(protein_cols)))
    ax1.set_yticklabels([f"P{i+1}" for i in range(len(protein_cols))])
    plt.colorbar(im, ax=ax1, label="标准化水平")

    # 箱线图
    protein_data_list = [proteins_results[col].values for col in protein_cols]
    bp = ax2.boxplot(protein_data_list, labels=[f"P{i+1}" for i in range(len(protein_cols))],
                     patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    ax2.set_xlabel("蛋白质", fontsize=12)
    ax2.set_ylabel("标准化水平", fontsize=12)
    ax2.set_title("蛋白质水平分布", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def create_visualizations(combined_results, age_results, cancer_results,
                         clocks_results, proteins_results, figures_dir):
    """创建所有可视化图表"""
    create_age_distribution_plot(age_results, f"{figures_dir}/age_distribution.png")
    create_cancer_distribution_plot(cancer_results, f"{figures_dir}/cancer_distribution.png")
    create_age_cancer_correlation_plot(combined_results, f"{figures_dir}/age_cancer_correlation.png")
    create_summary_statistics_plot(combined_results, f"{figures_dir}/summary_statistics.png")
    create_clocks_distribution_plot(clocks_results, f"{figures_dir}/clocks_distribution.png")
    create_proteins_heatmap(proteins_results, f"{figures_dir}/proteins_heatmap.png")


def generate_html_report(combined_results, report_path, figures_dir):
    """生成HTML分析报告"""

    # 计算统计数据
    age_stats = combined_results["predicted_age"].describe()
    cancer_stats = combined_results["cancer_probability"].describe()
    cancer_count = combined_results["cancer_prediction"].sum()
    normal_count = len(combined_results) - cancer_count
    cancer_rate = (cancer_count / len(combined_results)) * 100

    # 风险分层
    risk_categories = pd.cut(
        combined_results["cancer_probability"],
        bins=[0, 0.2, 0.5, 0.8, 1.0],
        labels=["低风险", "中低风险", "中高风险", "高风险"],
    )
    risk_counts = risk_categories.value_counts().sort_index()

    # 高风险样本
    high_risk = combined_results[combined_results["cancer_probability"] > 0.8].sort_values(
        "cancer_probability", ascending=False
    )

    # 年龄组癌症率
    age_bins = [0, 30, 40, 50, 60, 70, 100]
    age_labels = ["<30", "30-40", "40-50", "50-60", "60-70", "70+"]
    combined_results["age_group"] = pd.cut(
        combined_results["predicted_age"], bins=age_bins, labels=age_labels
    )
    cancer_rate_by_age = combined_results.groupby("age_group")["cancer_prediction"].mean() * 100

    # 生成HTML
    html_content = f"""
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>CpGPT 935k甲基化数据分析报告</title>
        <style>
            body {{
                font-family: 'Arial', 'Microsoft YaHei', sans-serif;
                line-height: 1.6;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                border-radius: 10px;
                margin-bottom: 30px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }}
            .header h1 {{
                margin: 0;
                font-size: 2.5em;
            }}
            .header p {{
                margin: 10px 0 0 0;
                font-size: 1.1em;
                opacity: 0.9;
            }}
            .section {{
                background: white;
                padding: 25px;
                margin-bottom: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .section h2 {{
                color: #667eea;
                border-bottom: 3px solid #667eea;
                padding-bottom: 10px;
                margin-top: 0;
            }}
            .section h3 {{
                color: #764ba2;
                margin-top: 20px;
            }}
            .stats-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }}
            .stat-card {{
                background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                padding: 20px;
                border-radius: 8px;
                border-left: 4px solid #667eea;
            }}
            .stat-card h4 {{
                margin: 0 0 10px 0;
                color: #333;
                font-size: 0.9em;
                text-transform: uppercase;
                letter-spacing: 1px;
            }}
            .stat-card .value {{
                font-size: 2em;
                font-weight: bold;
                color: #667eea;
                margin: 5px 0;
            }}
            .stat-card .subtitle {{
                font-size: 0.9em;
                color: #666;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            th, td {{
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background-color: #667eea;
                color: white;
                font-weight: bold;
            }}
            tr:hover {{
                background-color: #f5f5f5;
            }}
            .figure {{
                margin: 30px 0;
                text-align: center;
            }}
            .figure img {{
                max-width: 100%;
                border-radius: 8px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            }}
            .figure-caption {{
                margin-top: 10px;
                font-style: italic;
                color: #666;
            }}
            .alert {{
                padding: 15px;
                margin: 20px 0;
                border-radius: 5px;
                border-left: 4px solid;
            }}
            .alert-info {{
                background-color: #e3f2fd;
                border-color: #2196f3;
                color: #0d47a1;
            }}
            .alert-warning {{
                background-color: #fff3e0;
                border-color: #ff9800;
                color: #e65100;
            }}
            .alert-success {{
                background-color: #e8f5e9;
                border-color: #4caf50;
                color: #1b5e20;
            }}
            .interpretation {{
                background-color: #f9f9f9;
                padding: 15px;
                border-left: 4px solid #764ba2;
                margin: 15px 0;
                border-radius: 4px;
            }}
            .interpretation h4 {{
                margin-top: 0;
                color: #764ba2;
            }}
            .footer {{
                text-align: center;
                padding: 20px;
                color: #666;
                font-size: 0.9em;
                margin-top: 30px;
            }}
            .risk-badge {{
                display: inline-block;
                padding: 5px 10px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 0.9em;
            }}
            .risk-low {{ background-color: #4caf50; color: white; }}
            .risk-medium-low {{ background-color: #8bc34a; color: white; }}
            .risk-medium-high {{ background-color: #ff9800; color: white; }}
            .risk-high {{ background-color: #f44336; color: white; }}
            .download-btn {{
                background-color: #667eea;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-size: 1em;
                margin: 10px 5px;
                text-decoration: none;
                display: inline-block;
            }}
            .download-btn:hover {{
                background-color: #5568d3;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🧬 CpGPT 935k甲基化数据分析报告</h1>
            <p>基于CpGPT预训练模型的年龄与癌症预测分析</p>
            <p>生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>

        <div class="section">
            <h2>📊 执行摘要</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <h4>总样本数</h4>
                    <div class="value">{len(combined_results)}</div>
                    <div class="subtitle">935k平台样本</div>
                </div>
                <div class="stat-card">
                    <h4>平均预测年龄</h4>
                    <div class="value">{age_stats['mean']:.1f}</div>
                    <div class="subtitle">岁 (范围: {age_stats['min']:.1f} - {age_stats['max']:.1f})</div>
                </div>
                <div class="stat-card">
                    <h4>癌症预测率</h4>
                    <div class="value">{cancer_rate:.1f}%</div>
                    <div class="subtitle">{cancer_count} / {len(combined_results)} 样本</div>
                </div>
                <div class="stat-card">
                    <h4>平均癌症概率</h4>
                    <div class="value">{cancer_stats['mean']:.3f}</div>
                    <div class="subtitle">范围: {cancer_stats['min']:.3f} - {cancer_stats['max']:.3f}</div>
                </div>
            </div>

            <div class="alert alert-info">
                <strong>ℹ️ 说明：</strong> 本报告使用CpGPT预训练模型进行零样本推理，无需微调即可对935k甲基化数据进行年龄和癌症预测。
                预测结果基于模型在大规模甲基化数据上学习到的表观遗传模式。
            </div>
        </div>"""

    # 继续HTML内容 - 年龄分析部分
    html_content += f"""
        <div class="section">
            <h2>🎂 年龄预测分析</h2>
            <h3>统计摘要</h3>
            <table>
                <tr><th>统计指标</th><th>数值</th><th>说明</th></tr>
                <tr><td>样本数</td><td>{int(age_stats['count'])}</td><td>参与年龄预测的总样本数</td></tr>
                <tr><td>平均年龄</td><td>{age_stats['mean']:.2f} 岁</td><td>所有样本的平均预测年龄</td></tr>
                <tr><td>标准差</td><td>{age_stats['std']:.2f} 岁</td><td>年龄分布的离散程度</td></tr>
                <tr><td>最小值</td><td>{age_stats['min']:.2f} 岁</td><td>最年轻的预测年龄</td></tr>
                <tr><td>中位数</td><td>{age_stats['50%']:.2f} 岁</td><td>年龄分布的中间值</td></tr>
                <tr><td>最大值</td><td>{age_stats['max']:.2f} 岁</td><td>最年长的预测年龄</td></tr>
            </table>
            <div class="figure">
                <img src="figures/age_distribution.png" alt="年龄分布图">
                <div class="figure-caption">图1: 年龄分布直方图和箱线图</div>
            </div>
            <div class="interpretation">
                <h4>📖 结果解读</h4>
                <p><strong>年龄分布特征：</strong></p>
                <ul>
                    <li>样本年龄范围从 {age_stats['min']:.1f} 岁到 {age_stats['max']:.1f} 岁，跨度 {age_stats['max']-age_stats['min']:.1f} 年</li>
                    <li>平均年龄为 {age_stats['mean']:.1f} 岁，中位数为 {age_stats['50%']:.1f} 岁</li>
                    <li>标准差为 {age_stats['std']:.1f} 岁，表明年龄分布{'较为集中' if age_stats['std'] < 15 else '较为分散'}</li>
                </ul>
            </div>
        </div>

        <div class="section">
            <h2>🏥 癌症预测分析</h2>
            <h3>预测结果分布</h3>
            <div class="stats-grid">
                <div class="stat-card">
                    <h4>预测正常</h4>
                    <div class="value">{normal_count}</div>
                    <div class="subtitle">{100-cancer_rate:.1f}% 的样本</div>
                </div>
                <div class="stat-card">
                    <h4>预测癌症</h4>
                    <div class="value">{cancer_count}</div>
                    <div class="subtitle">{cancer_rate:.1f}% 的样本</div>
                </div>
            </div>

            <h3>风险分层统计</h3>
            <table>
                <tr><th>风险等级</th><th>概率范围</th><th>样本数</th><th>占比</th><th>建议</th></tr>
                <tr>
                    <td><span class="risk-badge risk-low">低风险</span></td>
                    <td>0.0 - 0.2</td>
                    <td>{risk_counts.get('低风险', 0)}</td>
                    <td>{risk_counts.get('低风险', 0)/len(combined_results)*100:.1f}%</td>
                    <td>常规监测</td>
                </tr>
                <tr>
                    <td><span class="risk-badge risk-medium-low">中低风险</span></td>
                    <td>0.2 - 0.5</td>
                    <td>{risk_counts.get('中低风险', 0)}</td>
                    <td>{risk_counts.get('中低风险', 0)/len(combined_results)*100:.1f}%</td>
                    <td>定期复查</td>
                </tr>
                <tr>
                    <td><span class="risk-badge risk-medium-high">中高风险</span></td>
                    <td>0.5 - 0.8</td>
                    <td>{risk_counts.get('中高风险', 0)}</td>
                    <td>{risk_counts.get('中高风险', 0)/len(combined_results)*100:.1f}%</td>
                    <td>密切关注，建议进一步检查</td>
                </tr>
                <tr>
                    <td><span class="risk-badge risk-high">高风险</span></td>
                    <td>0.8 - 1.0</td>
                    <td>{risk_counts.get('高风险', 0)}</td>
                    <td>{risk_counts.get('高风险', 0)/len(combined_results)*100:.1f}%</td>
                    <td>强烈建议临床诊断</td>
                </tr>
            </table>

            <div class="figure">
                <img src="figures/cancer_distribution.png" alt="癌症预测分布图">
                <div class="figure-caption">图2: 癌症概率分布、预测分类和概率箱线图</div>
            </div>

            {"<h3>⚠️ 高风险样本列表</h3>" if len(high_risk) > 0 else ""}
            {f'''
            <div class="alert alert-warning">
                <strong>警告：</strong> 发现 {len(high_risk)} 个高风险样本（癌症概率 > 0.8），建议优先关注。
            </div>
            <table>
                <tr><th>样本ID</th><th>预测年龄</th><th>癌症概率</th><th>风险等级</th></tr>
                {"".join([f'''
                <tr>
                    <td>{row["sample_id"]}</td>
                    <td>{row["predicted_age"]:.1f} 岁</td>
                    <td>{row["cancer_probability"]:.4f}</td>
                    <td><span class="risk-badge risk-high">高风险</span></td>
                </tr>
                ''' for _, row in high_risk.head(20).iterrows()])}
            </table>
            ''' if len(high_risk) > 0 else '<div class="alert alert-success"><strong>✓ 好消息：</strong> 未发现高风险样本（癌症概率 > 0.8）。</div>'}
        </div>

        <div class="section">
            <h2>🔗 年龄与癌症相关性分析</h2>
            <div class="figure">
                <img src="figures/age_cancer_correlation.png" alt="年龄与癌症相关性">
                <div class="figure-caption">图3: 年龄与癌症概率的相关性分析</div>
            </div>
        </div>

        <div class="section">
            <h2>📈 综合统计分析</h2>
            <div class="figure">
                <img src="figures/summary_statistics.png" alt="统计摘要">
                <div class="figure-caption">图4: 综合统计分析图表</div>
            </div>
        </div>

        <div class="section">
            <h2>⏰ 表观遗传时钟分析</h2>
            <p>表观遗传时钟是基于DNA甲基化模式预测生物学年龄和健康状态的重要指标。</p>
            <div class="figure">
                <img src="figures/clocks_distribution.png" alt="表观遗传时钟">
                <div class="figure-caption">图5: 五种表观遗传时钟的分布</div>
            </div>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-label">AltumAge</div>
                    <div class="stat-value">{combined_results.get('altumage', pd.Series([0])).mean():.2f}</div>
                    <div class="stat-desc">多组织生物学年龄</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">DunedinPACE</div>
                    <div class="stat-value">{combined_results.get('dunedinpace', pd.Series([0])).mean():.2f}</div>
                    <div class="stat-desc">衰老速度指标 (×100)</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">GrimAge2</div>
                    <div class="stat-value">{combined_results.get('grimage2', pd.Series([0])).mean():.2f}</div>
                    <div class="stat-desc">死亡率预测</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">HRS InCHPhenoAge</div>
                    <div class="stat-value">{combined_results.get('hrsinchphenoage', pd.Series([0])).mean():.2f}</div>
                    <div class="stat-desc">表型年龄</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">PC Horvath 2013</div>
                    <div class="stat-value">{combined_results.get('pchorvath2013', pd.Series([0])).mean():.2f}</div>
                    <div class="stat-desc">经典表观遗传时钟</div>
                </div>
            </div>
            <div class="alert alert-info">
                <p><strong>时钟解读：</strong></p>
                <ul>
                    <li><strong>AltumAge:</strong> 综合多组织的生物学年龄估计</li>
                    <li><strong>DunedinPACE:</strong> 衰老速度，值越高表示衰老越快（正常约100）</li>
                    <li><strong>GrimAge2:</strong> 与死亡率相关的表观遗传年龄</li>
                    <li><strong>HRS InCHPhenoAge:</strong> 基于健康和退休研究的表型年龄</li>
                    <li><strong>PC Horvath 2013:</strong> 最早的多组织表观遗传时钟</li>
                </ul>
            </div>
        </div>

        <div class="section">
            <h2>🧬 血浆蛋白质水平分析</h2>
            <p>血浆蛋白质水平可用于评估健康状态和预测疾病风险（如GrimAge3死亡率预测）。</p>
            <div class="figure">
                <img src="figures/proteins_heatmap.png" alt="蛋白质水平">
                <div class="figure-caption">图6: 血浆蛋白质水平热图和分布</div>
            </div>
            <div class="alert alert-info">
                <p><strong>蛋白质分析说明：</strong></p>
                <ul>
                    <li>显示的是标准化蛋白质水平（均值0，方差1）</li>
                    <li>可用于GrimAge3等高级表观遗传时钟的计算</li>
                    <li>蛋白质水平异常可能提示健康风险</li>
                </ul>
            </div>
        </div>

        <div class="section">
            <h2>💡 建议与注意事项</h2>
            <div class="alert alert-info">
                <p><strong>零样本推理特点：</strong></p>
                <ul>
                    <li>✅ <strong>优势：</strong> 无需训练数据，快速部署，可处理未见过的CpG位点</li>
                    <li>⚠️ <strong>限制：</strong> 准确性可能略低于微调模型，受平台特异性影响</li>
                    <li>📊 <strong>适用场景：</strong> 初步筛查、大规模分析、探索性研究</li>
                </ul>
            </div>
            <div class="alert alert-warning">
                <p><strong>⚠️ 免责声明：</strong></p>
                <ul>
                    <li>本报告仅供科研参考，不能作为临床诊断依据</li>
                    <li>预测结果基于DNA甲基化模式，可能受样本质量、技术偏差等因素影响</li>
                    <li>任何临床决策应由专业医疗人员基于综合信息做出</li>
                </ul>
            </div>
        </div>

        <div class="footer">
            <p>报告由 CpGPT 自动生成 | 生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>CpGPT: 首个具有链式思维推理能力的DNA甲基化基础模型</p>
            <p>论文: <a href="https://www.biorxiv.org/content/10.1101/2024.10.24.619766v1" target="_blank">bioRxiv 2024.10.24.619766</a></p>
        </div>
    </body>
    </html>
    """

    # 保存完整的HTML文件
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_content)

