"""
935k平台零样本推理示例（带可视化分析）
Zero-shot inference example for 935k methylation platform with visualization

此脚本演示如何在不微调的情况下，使用预训练的CpGPT模型对935k甲基化数据进行：
1. 年龄预测
2. 癌症预测
3. 其他表型预测
4. 生成可视化分析图谱
5. 输出详细分析报告

This script demonstrates how to perform zero-shot inference on 935k methylation data for:
1. Age prediction
2. Cancer prediction
3. Other phenotype predictions
4. Generate visualization charts
5. Output detailed analysis reports
"""

import gc
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from lightning import seed_everything

# 内存优化：禁用 MPS 回退到 CPU（如果需要）
# 取消下面的注释可以完全禁用 MPS，强制使用 CPU
# os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from cpgpt.data.components.cpgpt_datasaver import CpGPTDataSaver
from cpgpt.data.components.dna_llm_embedder import DNALLMEmbedder
from cpgpt.data.components.illumina_methylation_prober import IlluminaMethylationProber
from cpgpt.data.cpgpt_datamodule import CpGPTDataModule
from cpgpt.infer.cpgpt_inferencer import CpGPTInferencer
from cpgpt.trainer.cpgpt_trainer import CpGPTTrainer

# 设置中文字体支持
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
sns.set_style("whitegrid")
sns.set_palette("husl")

# ============================================================================
# 配置参数 / Configuration
# ============================================================================

# 路径配置
DEPENDENCIES_ROOT = "./dependencies"  # 依赖根目录（用于inferencer）
DEPENDENCIES_DIR = "./dependencies/human"  # DNA嵌入和基因组数据在human子目录下
MODEL_DIR = "./dependencies/model"  # 模型配置和权重在model子目录下
DATA_DIR = "./data"
RAW_935K_DATA_PATH = "./data/Sample-251107-Rico-1.csv"  # 您的935k数据路径（支持CSV或Arrow格式）
PROCESSED_DIR = "./data/935k_processed"
RESULTS_DIR = "./results/935k_predictions"
FIGURES_DIR = "./results/935k_predictions/figures"  # 图表保存目录
REPORT_PATH = "./results/935k_predictions/analysis_report.html"  # 分析报告路径

# 模型配置
RANDOM_SEED = 42
MAX_INPUT_LENGTH = 15000  # 减小以适应内存限制（从30000降低）
USE_CPU = True  # 设置为True使用CPU（稳定），False使用MPS GPU（快但可能内存溢出）

# 创建结果目录
Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
Path(FIGURES_DIR).mkdir(parents=True, exist_ok=True)
Path(DATA_DIR).mkdir(parents=True, exist_ok=True)


# ============================================================================
# 可视化和报告生成函数
# ============================================================================


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
    print(f"  ✓ 年龄分布图已保存: {save_path}")


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

    # 确保包含所有类别（即使计数为0）
    all_labels = {0: "正常", 1: "癌症"}
    all_colors = {0: "lightgreen", 1: "lightcoral"}

    # 构建完整的数据（包括0计数的类别）
    plot_data = []
    plot_labels = []
    plot_colors = []
    plot_explode = []

    for category in [0, 1]:
        count = cancer_counts.get(category, 0)
        plot_data.append(count)
        plot_labels.append(all_labels[category])
        plot_colors.append(all_colors[category])
        plot_explode.append(0.05)

    axes[1].pie(
        plot_data,
        labels=plot_labels,
        autopct="%1.1f%%",
        colors=plot_colors,
        explode=plot_explode,
        startangle=90,
        textprops={"fontsize": 12},
    )
    axes[1].set_title("癌症预测分类", fontsize=14, fontweight="bold")

    # 概率箱线图（按预测分类）
    normal_probs = cancer_results[cancer_results["cancer_prediction"] == 0]["cancer_probability"]
    cancer_probs = cancer_results[cancer_results["cancer_prediction"] == 1]["cancer_probability"]

    # 只绘制有数据的类别
    box_data = []
    box_labels = []
    box_colors = []

    if len(normal_probs) > 0:
        box_data.append(normal_probs)
        box_labels.append("预测正常")
        box_colors.append("lightgreen")

    if len(cancer_probs) > 0:
        box_data.append(cancer_probs)
        box_labels.append("预测癌症")
        box_colors.append("lightcoral")

    if len(box_data) > 0:
        box = axes[2].boxplot(
            box_data,
            labels=box_labels,
            patch_artist=True,
            boxprops=dict(alpha=0.7),
            medianprops=dict(color="red", linewidth=2),
        )
        # 设置颜色
        for i, color in enumerate(box_colors):
            box["boxes"][i].set_facecolor(color)
    else:
        axes[2].text(0.5, 0.5, "无数据", ha="center", va="center", fontsize=14)

    axes[2].axhline(0.5, color="red", linestyle="--", linewidth=1.5, alpha=0.5, label="阈值")
    axes[2].set_ylabel("癌症概率", fontsize=12)
    axes[2].set_title("概率分布（按分类）", fontsize=14, fontweight="bold")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ 癌症分布图已保存: {save_path}")


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
    print(f"  ✓ 年龄-癌症相关性图已保存: {save_path}")


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
    ax3.barh(range(len(risk_counts)), risk_counts.values, color=colors_risk, alpha=0.7)
    ax3.set_yticks(range(len(risk_counts)))
    ax3.set_yticklabels(risk_counts.index)
    ax3.set_xlabel("样本数", fontsize=10)
    ax3.set_title("癌症风险分层", fontsize=12, fontweight="bold")
    ax3.grid(True, alpha=0.3, axis="x")

    # 添加数值标签
    for i, v in enumerate(risk_counts.values):
        ax3.text(v + 0.5, i, str(v), va="center", fontsize=10)

    # 4-6. 年龄分布的不同视图
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

    # 7. 癌症概率分布（按年龄着色）
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

    # 8. 高风险样本表
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
    print(f"  ✓ 统计摘要图已保存: {save_path}")

# ============================================================================
# 步骤1: 环境设置
# ============================================================================

print("=" * 80)
print("步骤1: 环境设置")
print("=" * 80)

# 检测可用设备
print("\n🖥️ 设备检测:")
print(f"  - CUDA 可用: {torch.cuda.is_available()}")
print(f"  - MPS 可用: {torch.backends.mps.is_available()}")
print(f"  - CPU 核心数: {os.cpu_count()}")
if USE_CPU:
    print(f"  ✓ 配置使用: CPU (稳定模式)")
else:
    print(f"  ✓ 配置使用: MPS GPU (高性能模式)")
    print(f"  ⚠️ 注意: 如果遇到内存溢出，请设置 USE_CPU=True")

# 设置随机种子
seed_everything(RANDOM_SEED, workers=True)
try:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
except Exception:
    pass

# 初始化inferencer（使用根目录）
inferencer = CpGPTInferencer(dependencies_dir=DEPENDENCIES_ROOT, data_dir=DATA_DIR)

# ============================================================================
# 步骤2: 下载依赖和模型（首次运行需要）
# ============================================================================

print("\n" + "=" * 80)
print("步骤2: 下载依赖和模型")
print("=" * 80)

# 下载人类物种的依赖（DNA嵌入等）
print("下载DNA嵌入依赖...")
inferencer.download_dependencies(species="human", overwrite=False)

# 下载预训练模型
# 可选模型：
# - "small": 轻量级模型，快速推理
# - "large": 完整模型，更高准确性
# - "age_cot": 年龄预测专用模型
# - "cancer": 癌症预测专用模型
# - "clock_proxies": 表观遗传时钟代理

models_to_download = ["small", "age_cot", "cancer","clock_proxies"]

for model_name in models_to_download:
    print(f"下载模型: {model_name}...")
    inferencer.download_model(model_name=model_name, overwrite=False)

# ============================================================================
# 步骤3: 数据预处理
# ============================================================================

print("\n" + "=" * 80)
print("步骤3: 数据预处理")
print("=" * 80)

# 初始化组件
print("初始化DNA嵌入器和探针映射器...")
embedder = DNALLMEmbedder(dependencies_dir=DEPENDENCIES_DIR)
prober = IlluminaMethylationProber(dependencies_dir=DEPENDENCIES_DIR, embedder=embedder)

# 检查数据格式并转换为Arrow格式（如果是CSV）
print(f"检查数据格式: {RAW_935K_DATA_PATH}")
if RAW_935K_DATA_PATH.endswith(".csv"):
    print("检测到CSV格式，正在转换为Arrow格式...")
    df_csv = pd.read_csv(RAW_935K_DATA_PATH)
    print(f"  ✓ 加载CSV数据: {df_csv.shape[0]} 行 x {df_csv.shape[1]} 列")

    # 转换为Arrow格式
    arrow_path = RAW_935K_DATA_PATH.replace(".csv", ".arrow")
    df_csv.to_feather(arrow_path)
    print(f"  ✓ 已转换为Arrow格式: {arrow_path}")

    # 更新数据路径
    DATA_PATH_FOR_PROCESSING = arrow_path
else:
    DATA_PATH_FOR_PROCESSING = RAW_935K_DATA_PATH

# 创建数据保存器
print(f"处理935k数据: {DATA_PATH_FOR_PROCESSING}")
datasaver = CpGPTDataSaver(
    data_paths=DATA_PATH_FOR_PROCESSING,
    processed_dir=PROCESSED_DIR,
    metadata_cols=None,  # 如果有元数据列（如真实年龄），可以在这里指定
)

# 处理文件（探针ID -> 基因组位置）
print("转换探针ID到基因组位置...")
datasaver.process_files(prober=prober, embedder=embedder, check_methylation_pattern=False)

# 获取所有基因组位置
all_genomic_locations = datasaver.all_genomic_locations.get("homo_sapiens", set())
print(f"总共识别到 {len(all_genomic_locations)} 个基因组位置")

# 生成DNA嵌入（如果尚未生成）
print("生成DNA序列嵌入...")
embedder.parse_dna_embeddings(
    genomic_locations=sorted(all_genomic_locations),
    species="homo_sapiens",
    dna_llm="nucleotide-transformer-v2-500m-multi-species",
    dna_context_len=2001,
    batch_size=8,  # 根据GPU内存调整
    num_workers=0,  # 修复: macOS + Python 3.13 多进程序列化问题
)

print("数据预处理完成！")

# ============================================================================
# 步骤4: 年龄预测（零样本）
# ============================================================================

print("\n" + "=" * 80)
print("步骤4: 年龄预测（零样本推理）")
print("=" * 80)

# 加载年龄预测模型
MODEL_NAME = "age_cot"
MODEL_CONFIG_PATH = f"{MODEL_DIR}/config/{MODEL_NAME}.yaml"
MODEL_CHECKPOINT_PATH = f"{MODEL_DIR}/weights/{MODEL_NAME}.ckpt"
MODEL_VOCAB_PATH = f"{MODEL_DIR}/vocab/{MODEL_NAME}.json"

print(f"加载模型配置: {MODEL_CONFIG_PATH}")
config_age = inferencer.load_cpgpt_config(MODEL_CONFIG_PATH)

print(f"加载模型权重: {MODEL_CHECKPOINT_PATH}")
model_age = inferencer.load_cpgpt_model(
    config_age, model_ckpt_path=MODEL_CHECKPOINT_PATH, strict_load=True
)

# 过滤特征（使用模型训练时的词汇表）
print("过滤特征以匹配模型词汇表...")
# 读取数据（支持CSV和Arrow格式）
if RAW_935K_DATA_PATH.endswith(".csv"):
    df_935k = pd.read_feather(DATA_PATH_FOR_PROCESSING)  # 使用之前转换的Arrow文件
else:
    df_935k = pd.read_feather(RAW_935K_DATA_PATH)

vocab_age = json.load(open(MODEL_VOCAB_PATH, "r"))

# 保存样本ID
sample_ids = df_935k.iloc[:, 0] if "GSM_ID" not in df_935k.columns else df_935k["GSM_ID"]

# 过滤列
available_features = [col for col in df_935k.columns if col in vocab_age["input"]]
print(f"935k数据中有 {len(available_features)} 个特征在模型词汇表中")

if len(available_features) == 0:
    print("\n❌ 错误: 没有找到匹配的特征！")
    print("请检查:")
    print("  1. 数据列名是否为CpG位点ID（如cg00000029）")
    print("  2. 数据是否已经过预处理")
    print(f"\n数据前5列: {list(df_935k.columns[:5])}")
    sys.exit(1)

df_filtered = df_935k[available_features]
filtered_path = f"{DATA_DIR}/935k_filtered_age.arrow"
df_filtered.to_feather(filtered_path)

# 重新处理过滤后的数据
datasaver_age = CpGPTDataSaver(
    data_paths=filtered_path, processed_dir=f"{PROCESSED_DIR}_age", metadata_cols=None
)
datasaver_age.process_files(prober=prober, embedder=embedder)

# 创建数据模块
datamodule_age = CpGPTDataModule(
    predict_dir=f"{PROCESSED_DIR}_age",
    dependencies_dir=DEPENDENCIES_DIR,
    batch_size=1,
    num_workers=0,
    max_length=MAX_INPUT_LENGTH,
    dna_llm=config_age.data.dna_llm,
    dna_context_len=config_age.data.dna_context_len,
    sorting_strategy=config_age.data.sorting_strategy,
    pin_memory=False,
)

# 创建训练器并进行预测
print("执行年龄预测...")
if USE_CPU:
    print("⚙️ 使用 CPU 进行推理（稳定但较慢）")
    trainer = CpGPTTrainer(accelerator="cpu", precision="32")  # CPU 使用 float32
else:
    print("⚙️ 使用 MPS GPU 进行推理（快但可能内存溢出）")
    trainer = CpGPTTrainer(precision="16-mixed")  # GPU 使用混合精度

age_predictions = trainer.predict(
    model=model_age,
    datamodule=datamodule_age,
    predict_mode="forward",
    return_keys=["pred_conditions"],
)

# 保存年龄预测结果
age_results = pd.DataFrame(
    {"sample_id": sample_ids, "predicted_age": age_predictions["pred_conditions"].flatten()}
)
age_results_path = f"{RESULTS_DIR}/age_predictions.csv"
age_results.to_csv(age_results_path, index=False)
print(f"年龄预测结果已保存到: {age_results_path}")
print(age_results.head())

# 释放年龄模型内存
print("\n释放年龄模型内存...")
del model_age
del datamodule_age
del age_predictions
if torch.backends.mps.is_available():
    torch.mps.empty_cache()
elif torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()
print("内存已释放")

# ============================================================================
# 步骤5: 癌症预测（零样本）
# ============================================================================

print("\n" + "=" * 80)
print("步骤5: 癌症预测（零样本推理）")
print("=" * 80)

# 加载癌症预测模型
MODEL_NAME = "cancer"
MODEL_CONFIG_PATH = f"{MODEL_DIR}/config/{MODEL_NAME}.yaml"
MODEL_CHECKPOINT_PATH = f"{MODEL_DIR}/weights/{MODEL_NAME}.ckpt"
MODEL_VOCAB_PATH = f"{MODEL_DIR}/vocab/{MODEL_NAME}.json"

print(f"加载模型配置: {MODEL_CONFIG_PATH}")
config_cancer = inferencer.load_cpgpt_config(MODEL_CONFIG_PATH)

print(f"加载模型权重: {MODEL_CHECKPOINT_PATH}")
model_cancer = inferencer.load_cpgpt_model(
    config_cancer, model_ckpt_path=MODEL_CHECKPOINT_PATH, strict_load=True
)

# 过滤特征
print("过滤特征以匹配癌症模型词汇表...")
vocab_cancer = json.load(open(MODEL_VOCAB_PATH, "r"))
available_features_cancer = [col for col in df_935k.columns if col in vocab_cancer["input"]]
print(f"935k数据中有 {len(available_features_cancer)} 个特征在癌症模型词汇表中")

df_filtered_cancer = df_935k[available_features_cancer]
filtered_path_cancer = f"{DATA_DIR}/935k_filtered_cancer.arrow"
df_filtered_cancer.to_feather(filtered_path_cancer)

# 重新处理过滤后的数据
datasaver_cancer = CpGPTDataSaver(
    data_paths=filtered_path_cancer, processed_dir=f"{PROCESSED_DIR}_cancer", metadata_cols=None
)
datasaver_cancer.process_files(prober=prober, embedder=embedder)

# 创建数据模块
datamodule_cancer = CpGPTDataModule(
    predict_dir=f"{PROCESSED_DIR}_cancer",
    dependencies_dir=DEPENDENCIES_DIR,
    batch_size=1,
    num_workers=0,
    max_length=MAX_INPUT_LENGTH,
    dna_llm=config_cancer.data.dna_llm,
    dna_context_len=config_cancer.data.dna_context_len,
    sorting_strategy=config_cancer.data.sorting_strategy,
    pin_memory=False,
)

# 执行癌症预测
print("执行癌症预测...")
cancer_predictions = trainer.predict(
    model=model_cancer,
    datamodule=datamodule_cancer,
    predict_mode="forward",
    return_keys=["pred_conditions"],
)

# 将logits转换为概率（使用sigmoid）
cancer_logits = cancer_predictions["pred_conditions"].flatten()
cancer_probabilities = torch.sigmoid(torch.tensor(cancer_logits)).numpy()

# 保存癌症预测结果
cancer_results = pd.DataFrame(
    {
        "sample_id": sample_ids,
        "cancer_logit": cancer_logits,
        "cancer_probability": cancer_probabilities,
        "cancer_prediction": (cancer_probabilities > 0.5).astype(int),
    }
)
cancer_results_path = f"{RESULTS_DIR}/cancer_predictions.csv"
cancer_results.to_csv(cancer_results_path, index=False)
print(f"癌症预测结果已保存到: {cancer_results_path}")
print(cancer_results.head())

# 释放癌症模型内存
print("\n释放癌症模型内存...")
del model_cancer
del datamodule_cancer
del cancer_predictions
if torch.backends.mps.is_available():
    torch.mps.empty_cache()
elif torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()
print("内存已释放")

# ============================================================================
# 步骤5.5: 表观遗传时钟预测（零样本）
# ============================================================================

print("\n" + "=" * 80)
print("步骤5.5: 表观遗传时钟预测（零样本推理）")
print("=" * 80)

# 加载表观遗传时钟模型
MODEL_NAME = "clock_proxies"
MODEL_CONFIG_PATH = f"{MODEL_DIR}/config/{MODEL_NAME}.yaml"
MODEL_CHECKPOINT_PATH = f"{MODEL_DIR}/weights/{MODEL_NAME}.ckpt"
MODEL_VOCAB_PATH = f"{MODEL_DIR}/vocab/{MODEL_NAME}.json"

print(f"加载模型配置: {MODEL_CONFIG_PATH}")
config_clocks = inferencer.load_cpgpt_config(MODEL_CONFIG_PATH)

print(f"加载模型权重: {MODEL_CHECKPOINT_PATH}")
model_clocks = inferencer.load_cpgpt_model(
    config_clocks, model_ckpt_path=MODEL_CHECKPOINT_PATH, strict_load=True
)

# 过滤特征
print("过滤特征以匹配表观遗传时钟模型词汇表...")
vocab_clocks = json.load(open(MODEL_VOCAB_PATH, "r"))
available_features_clocks = [col for col in df_935k.columns if col in vocab_clocks["input"]]
print(f"935k数据中有 {len(available_features_clocks)} 个特征在时钟模型词汇表中")

df_filtered_clocks = df_935k[available_features_clocks]
filtered_path_clocks = f"{DATA_DIR}/935k_filtered_clocks.arrow"
df_filtered_clocks.to_feather(filtered_path_clocks)

# 重新处理过滤后的数据
datasaver_clocks = CpGPTDataSaver(
    data_paths=filtered_path_clocks, processed_dir=f"{PROCESSED_DIR}_clocks", metadata_cols=None
)
datasaver_clocks.process_files(prober=prober, embedder=embedder)

# 创建数据模块
datamodule_clocks = CpGPTDataModule(
    predict_dir=f"{PROCESSED_DIR}_clocks",
    dependencies_dir=DEPENDENCIES_DIR,
    batch_size=1,
    num_workers=0,
    max_length=MAX_INPUT_LENGTH,
    dna_llm=config_clocks.data.dna_llm,
    dna_context_len=config_clocks.data.dna_context_len,
    sorting_strategy=config_clocks.data.sorting_strategy,
    pin_memory=False,
)

# 执行表观遗传时钟预测
print("执行表观遗传时钟预测...")
clocks_predictions = trainer.predict(
    model=model_clocks,
    datamodule=datamodule_clocks,
    predict_mode="forward",
    return_keys=["pred_conditions"],
)

# 保存表观遗传时钟预测结果
# clock_proxies模型预测5个时钟：Horvath, Hannum, PhenoAge, GrimAge, DunedinPACE
clock_names = ["Horvath", "Hannum", "PhenoAge", "GrimAge", "DunedinPACE"]
clocks_data = {"sample_id": sample_ids}

# 检查预测结果的维度
pred_clocks = clocks_predictions["pred_conditions"]
print(f"时钟预测结果形状: {pred_clocks.shape}")

# 如果是多维输出，每一列对应一个时钟
if len(pred_clocks.shape) > 1 and pred_clocks.shape[1] >= len(clock_names):
    for i, clock_name in enumerate(clock_names):
        clocks_data[clock_name] = pred_clocks[:, i]
else:
    # 如果只有一维输出，可能需要调整
    print("⚠️ 警告: 时钟预测输出维度不符合预期")
    clocks_data["clock_prediction"] = pred_clocks.flatten()

clocks_results = pd.DataFrame(clocks_data)
clocks_results_path = f"{RESULTS_DIR}/clocks_predictions.csv"
clocks_results.to_csv(clocks_results_path, index=False)
print(f"表观遗传时钟预测结果已保存到: {clocks_results_path}")
print(clocks_results.head())

# 释放时钟模型内存
print("\n释放时钟模型内存...")
del model_clocks
del datamodule_clocks
del clocks_predictions
if torch.backends.mps.is_available():
    torch.mps.empty_cache()
elif torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()
print("内存已释放")

# ============================================================================
# 步骤6: 综合结果
# ============================================================================

print("\n" + "=" * 80)
print("步骤6: 综合结果")
print("=" * 80)

# 合并所有预测结果
combined_results = pd.merge(age_results, cancer_results, on="sample_id")
combined_results = pd.merge(combined_results, clocks_results, on="sample_id")
combined_results_path = f"{RESULTS_DIR}/combined_predictions.csv"
combined_results.to_csv(combined_results_path, index=False)

print(f"综合预测结果已保存到: {combined_results_path}")
print("\n预测结果摘要:")
print(combined_results.describe())

# 检查年龄预测的合理性
print("\n年龄预测质量检查:")
print(f"  - age_cot模型预测范围: {age_results['predicted_age'].min():.2f} - {age_results['predicted_age'].max():.2f} 岁")
if "Horvath" in clocks_results.columns:
    print(f"  - Horvath时钟预测范围: {clocks_results['Horvath'].min():.2f} - {clocks_results['Horvath'].max():.2f} 岁")
if "Hannum" in clocks_results.columns:
    print(f"  - Hannum时钟预测范围: {clocks_results['Hannum'].min():.2f} - {clocks_results['Hannum'].max():.2f} 岁")

# 如果年龄预测异常（如负数或过大），给出警告
if age_results['predicted_age'].min() < 0:
    print("\n⚠️ 警告: 检测到负数年龄预测！")
    print("   可能原因:")
    print("   1. 数据质量问题（缺失值、异常值）")
    print("   2. 特征匹配不足（可用特征太少）")
    print("   3. 平台差异（935k vs 训练数据平台）")
    print(f"   建议: 检查数据质量，当前可用特征数: {len(available_features)}")

if age_results['predicted_age'].max() > 120:
    print("\n⚠️ 警告: 检测到异常高的年龄预测（>120岁）！")
    print("   建议检查数据预处理和特征匹配")

# ============================================================================
# 步骤7: 生成可视化图表
# ============================================================================

print("\n" + "=" * 80)
print("步骤7: 生成可视化分析图表")
print("=" * 80)

# 年龄分布图
print("生成年龄分布图...")
age_dist_path = f"{FIGURES_DIR}/age_distribution.png"
create_age_distribution_plot(age_results, age_dist_path)

# 癌症分布图
print("生成癌症分布图...")
cancer_dist_path = f"{FIGURES_DIR}/cancer_distribution.png"
create_cancer_distribution_plot(cancer_results, cancer_dist_path)

# 年龄-癌症相关性图
print("生成年龄-癌症相关性图...")
correlation_path = f"{FIGURES_DIR}/age_cancer_correlation.png"
create_age_cancer_correlation_plot(combined_results, correlation_path)

# 统计摘要图
print("生成统计摘要图...")
summary_path = f"{FIGURES_DIR}/summary_statistics.png"
create_summary_statistics_plot(combined_results, summary_path)

print("\n所有图表生成完成！")

# ============================================================================
# 步骤8: 生成HTML分析报告
# ============================================================================

print("\n" + "=" * 80)
print("步骤8: 生成HTML分析报告")
print("=" * 80)


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
        <title>935k甲基化数据零样本推理分析报告</title>
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
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🧬 935k甲基化数据零样本推理分析报告</h1>
            <p>基于CpGPT预训练模型的多维度表观遗传分析</p>
            <p>生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>

        <div class="section">
            <h2>📊 执行摘要</h2>

            <div class="alert alert-info">
                <strong>🤖 使用的AI模型：</strong>
                <ul style="margin: 10px 0 0 0;">
                    <li><strong>age_cot：</strong> 多组织年龄预测模型（Chain-of-Thought）</li>
                    <li><strong>cancer：</strong> 多组织癌症预测模型</li>
                    <li><strong>clock_proxies：</strong> 五大表观遗传时钟集成模型（Horvath, Hannum, PhenoAge, GrimAge, DunedinPACE）</li>
                </ul>
            </div>
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
        </div>

        <div class="section">
            <h2>🎂 年龄预测分析</h2>

            <h3>统计摘要</h3>
            <table>
                <tr>
                    <th>统计指标</th>
                    <th>数值</th>
                    <th>说明</th>
                </tr>
                <tr>
                    <td>样本数</td>
                    <td>{int(age_stats['count'])}</td>
                    <td>参与年龄预测的总样本数</td>
                </tr>
                <tr>
                    <td>平均年龄</td>
                    <td>{age_stats['mean']:.2f} 岁</td>
                    <td>所有样本的平均预测年龄</td>
                </tr>
                <tr>
                    <td>标准差</td>
                    <td>{age_stats['std']:.2f} 岁</td>
                    <td>年龄分布的离散程度</td>
                </tr>
                <tr>
                    <td>最小值</td>
                    <td>{age_stats['min']:.2f} 岁</td>
                    <td>最年轻的预测年龄</td>
                </tr>
                <tr>
                    <td>25%分位数</td>
                    <td>{age_stats['25%']:.2f} 岁</td>
                    <td>25%的样本年龄低于此值</td>
                </tr>
                <tr>
                    <td>中位数</td>
                    <td>{age_stats['50%']:.2f} 岁</td>
                    <td>年龄分布的中间值</td>
                </tr>
                <tr>
                    <td>75%分位数</td>
                    <td>{age_stats['75%']:.2f} 岁</td>
                    <td>75%的样本年龄低于此值</td>
                </tr>
                <tr>
                    <td>最大值</td>
                    <td>{age_stats['max']:.2f} 岁</td>
                    <td>最年长的预测年龄</td>
                </tr>
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
                <p><strong>临床意义：</strong></p>
                <ul>
                    <li>表观遗传年龄（DNA甲基化年龄）可能与实际年龄存在差异，这种差异称为"年龄加速"</li>
                    <li>年龄加速与多种健康状况相关，包括死亡率、慢性疾病风险等</li>
                    <li>建议将预测年龄与实际年龄对比，评估表观遗传年龄加速情况</li>
                </ul>
            </div>
        </div>

        <div class="section">
            <h2>⏰ 表观遗传时钟分析</h2>

            <div class="alert alert-info">
                <strong>ℹ️ 关于表观遗传时钟：</strong> 表观遗传时钟是基于DNA甲基化模式预测生物学年龄的算法。
                不同的时钟模型关注不同的生物学特征和健康结局。
            </div>

            <h3>五大表观遗传时钟预测结果</h3>
            <table>
                <tr>
                    <th>时钟模型</th>
                    <th>平均年龄</th>
                    <th>范围</th>
                    <th>标准差</th>
                    <th>特点</th>
                </tr>
                {f'''
                <tr>
                    <td><strong>Horvath时钟</strong></td>
                    <td>{combined_results["Horvath"].mean():.2f} 岁</td>
                    <td>{combined_results["Horvath"].min():.2f} - {combined_results["Horvath"].max():.2f}</td>
                    <td>{combined_results["Horvath"].std():.2f}</td>
                    <td>多组织通用，最早的表观遗传时钟</td>
                </tr>
                ''' if "Horvath" in combined_results.columns else ''}
                {f'''
                <tr>
                    <td><strong>Hannum时钟</strong></td>
                    <td>{combined_results["Hannum"].mean():.2f} 岁</td>
                    <td>{combined_results["Hannum"].min():.2f} - {combined_results["Hannum"].max():.2f}</td>
                    <td>{combined_results["Hannum"].std():.2f}</td>
                    <td>血液特异性，预测实际年龄</td>
                </tr>
                ''' if "Hannum" in combined_results.columns else ''}
                {f'''
                <tr>
                    <td><strong>PhenoAge时钟</strong></td>
                    <td>{combined_results["PhenoAge"].mean():.2f} 岁</td>
                    <td>{combined_results["PhenoAge"].min():.2f} - {combined_results["PhenoAge"].max():.2f}</td>
                    <td>{combined_results["PhenoAge"].std():.2f}</td>
                    <td>预测表型年龄，与死亡率相关</td>
                </tr>
                ''' if "PhenoAge" in combined_results.columns else ''}
                {f'''
                <tr>
                    <td><strong>GrimAge时钟</strong></td>
                    <td>{combined_results["GrimAge"].mean():.2f} 岁</td>
                    <td>{combined_results["GrimAge"].min():.2f} - {combined_results["GrimAge"].max():.2f}</td>
                    <td>{combined_results["GrimAge"].std():.2f}</td>
                    <td>预测寿命，与多种疾病风险相关</td>
                </tr>
                ''' if "GrimAge" in combined_results.columns else ''}
                {f'''
                <tr>
                    <td><strong>DunedinPACE</strong></td>
                    <td>{combined_results["DunedinPACE"].mean():.2f}</td>
                    <td>{combined_results["DunedinPACE"].min():.2f} - {combined_results["DunedinPACE"].max():.2f}</td>
                    <td>{combined_results["DunedinPACE"].std():.2f}</td>
                    <td>衰老速度指标（非年龄，1.0=正常速度）</td>
                </tr>
                ''' if "DunedinPACE" in combined_results.columns else ''}
            </table>

            <div class="interpretation">
                <h4>📖 时钟模型解读</h4>
                <p><strong>各时钟的临床意义：</strong></p>
                <ul>
                    <li><strong>Horvath时钟：</strong> 最早开发的表观遗传时钟，适用于多种组织类型，预测实际年龄准确度高</li>
                    <li><strong>Hannum时钟：</strong> 专门针对血液样本开发，与免疫系统衰老密切相关</li>
                    <li><strong>PhenoAge：</strong> 预测"表型年龄"，比实际年龄更能反映健康状态和死亡风险</li>
                    <li><strong>GrimAge：</strong> 目前预测寿命最准确的时钟，与吸烟、BMI、疾病史等因素相关</li>
                    <li><strong>DunedinPACE：</strong> 衡量衰老速度而非年龄，1.0表示正常衰老速度，>1.0表示加速衰老</li>
                </ul>
                <p><strong>年龄加速的意义：</strong></p>
                <ul>
                    <li>表观遗传年龄 > 实际年龄：年龄加速，可能提示健康风险增加</li>
                    <li>表观遗传年龄 < 实际年龄：年龄减速，可能提示较好的健康状态</li>
                    <li>建议将预测年龄与实际年龄对比，评估个体化健康风险</li>
                </ul>
            </div>

            <h3>时钟模型对比</h3>
            <div class="interpretation">
                <p><strong>模型一致性分析：</strong></p>
                <ul>
                    {f'<li>age_cot模型预测: {combined_results["predicted_age"].mean():.2f} ± {combined_results["predicted_age"].std():.2f} 岁</li>' if "predicted_age" in combined_results.columns else ''}
                    {f'<li>Horvath时钟预测: {combined_results["Horvath"].mean():.2f} ± {combined_results["Horvath"].std():.2f} 岁</li>' if "Horvath" in combined_results.columns else ''}
                    {f'<li>Hannum时钟预测: {combined_results["Hannum"].mean():.2f} ± {combined_results["Hannum"].std():.2f} 岁</li>' if "Hannum" in combined_results.columns else ''}
                    {f'<li>PhenoAge时钟预测: {combined_results["PhenoAge"].mean():.2f} ± {combined_results["PhenoAge"].std():.2f} 岁</li>' if "PhenoAge" in combined_results.columns else ''}
                    {f'<li>GrimAge时钟预测: {combined_results["GrimAge"].mean():.2f} ± {combined_results["GrimAge"].std():.2f} 岁</li>' if "GrimAge" in combined_results.columns else ''}
                </ul>
                <p><strong>建议：</strong></p>
                <ul>
                    <li>如果多个时钟预测结果一致，说明年龄预测较为可靠</li>
                    <li>如果不同时钟差异较大，可能反映了不同的生物学衰老维度</li>
                    <li>GrimAge和PhenoAge更关注健康结局，可能与实际年龄差异更大</li>
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
                <tr>
                    <th>风险等级</th>
                    <th>概率范围</th>
                    <th>样本数</th>
                    <th>占比</th>
                    <th>建议</th>
                </tr>
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

            <div class="interpretation">
                <h4>📖 结果解读</h4>
                <p><strong>癌症预测特征：</strong></p>
                <ul>
                    <li>使用阈值0.5进行二分类：概率>0.5预测为癌症，≤0.5预测为正常</li>
                    <li>共有 {cancer_count} 个样本（{cancer_rate:.1f}%）被预测为癌症</li>
                    <li>平均癌症概率为 {cancer_stats['mean']:.3f}</li>
                </ul>
                <p><strong>临床意义：</strong></p>
                <ul>
                    <li>DNA甲基化模式可以反映癌症相关的表观遗传改变</li>
                    <li>高风险样本（概率>0.8）建议进行临床验证和进一步诊断</li>
                    <li>此预测为辅助工具，不能替代临床诊断</li>
                </ul>
            </div>

            {"<h3>⚠️ 高风险样本列表</h3>" if len(high_risk) > 0 else ""}
            {f'''
            <div class="alert alert-warning">
                <strong>警告：</strong> 发现 {len(high_risk)} 个高风险样本（癌症概率 > 0.8），建议优先关注。
            </div>
            <table>
                <tr>
                    <th>样本ID</th>
                    <th>预测年龄</th>
                    <th>癌症概率</th>
                    <th>风险等级</th>
                </tr>
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

            <h3>不同年龄组的癌症预测率</h3>
            <table>
                <tr>
                    <th>年龄组</th>
                    <th>癌症预测率</th>
                    <th>样本数</th>
                </tr>
                {"".join([f'''
                <tr>
                    <td>{age_group}</td>
                    <td>{rate:.1f}%</td>
                    <td>{combined_results[combined_results["age_group"]==age_group].shape[0]}</td>
                </tr>
                ''' for age_group, rate in cancer_rate_by_age.items()])}
            </table>

            <div class="figure">
                <img src="figures/age_cancer_correlation.png" alt="年龄与癌症相关性">
                <div class="figure-caption">图3: 年龄与癌症概率的相关性分析</div>
            </div>

            <div class="interpretation">
                <h4>📖 结果解读</h4>
                <p><strong>年龄-癌症关联：</strong></p>
                <ul>
                    <li>散点图显示了每个样本的年龄和癌症概率分布</li>
                    <li>柱状图展示了不同年龄组的癌症预测率趋势</li>
                    <li>{'年龄与癌症风险呈正相关' if cancer_rate_by_age.corr(pd.Series(range(len(cancer_rate_by_age)))) > 0.3 else '年龄与癌症风险相关性较弱'}</li>
                </ul>
                <p><strong>生物学意义：</strong></p>
                <ul>
                    <li>癌症风险通常随年龄增加而上升，这与DNA损伤累积和免疫功能下降有关</li>
                    <li>表观遗传改变（如DNA甲基化）在癌症发生发展中起重要作用</li>
                </ul>
            </div>
        </div>

        <div class="section">
            <h2>📈 综合统计分析</h2>

            <div class="figure">
                <img src="figures/summary_statistics.png" alt="统计摘要">
                <div class="figure-caption">图4: 综合统计分析图表</div>
            </div>

            <div class="interpretation">
                <h4>📖 图表说明</h4>
                <p>综合统计图包含以下内容：</p>
                <ul>
                    <li><strong>左上：</strong> 年龄预测的详细统计数据</li>
                    <li><strong>中上：</strong> 癌症预测的详细统计数据</li>
                    <li><strong>右上：</strong> 癌症风险分层柱状图</li>
                    <li><strong>中间：</strong> 年龄分布密度图</li>
                    <li><strong>左下：</strong> 癌症概率分布（按年龄着色）</li>
                    <li><strong>右下：</strong> 高风险样本列表（如有）</li>
                </ul>
            </div>
        </div>

        <div class="section">
            <h2>💡 建议与注意事项</h2>

            <h3>模型性能说明</h3>
            <div class="alert alert-info">
                <p><strong>零样本推理特点：</strong></p>
                <ul>
                    <li>✅ <strong>优势：</strong> 无需训练数据，快速部署，可处理未见过的CpG位点</li>
                    <li>⚠️ <strong>限制：</strong> 准确性可能略低于微调模型，受平台特异性影响</li>
                    <li>📊 <strong>适用场景：</strong> 初步筛查、大规模分析、探索性研究</li>
                </ul>
            </div>

            <h3>后续建议</h3>
            <ol>
                <li><strong>验证准确性：</strong> 如有真实标签，计算预测误差和相关性</li>
                <li><strong>高风险样本：</strong> 对高风险样本进行临床验证和进一步检查</li>
                <li><strong>模型微调：</strong> 如零样本性能不满意，考虑收集50-100个带标签样本进行微调</li>
                <li><strong>多模型验证：</strong> 尝试其他预训练模型（如relative_age, clock_proxies）进行交叉验证</li>
                <li><strong>纵向研究：</strong> 对同一样本进行多时间点检测，追踪表观遗传变化</li>
            </ol>

            <h3>重要声明</h3>
            <div class="alert alert-warning">
                <p><strong>⚠️ 免责声明：</strong></p>
                <ul>
                    <li>本报告仅供科研参考，不能作为临床诊断依据</li>
                    <li>预测结果基于DNA甲基化模式，可能受样本质量、技术偏差等因素影响</li>
                    <li>任何临床决策应由专业医疗人员基于综合信息做出</li>
                    <li>建议结合其他临床检查和生物标志物进行综合评估</li>
                </ul>
            </div>
        </div>

        <div class="section">
            <h2>📚 技术细节</h2>

            <h3>模型信息</h3>
            <table>
                <tr>
                    <th>项目</th>
                    <th>详情</th>
                </tr>
                <tr>
                    <td>年龄预测模型</td>
                    <td>CpGPT-2M-Age (age_cot)</td>
                </tr>
                <tr>
                    <td>癌症预测模型</td>
                    <td>CpGPT-2M-Cancer (cancer)</td>
                </tr>
                <tr>
                    <td>DNA嵌入模型</td>
                    <td>Nucleotide Transformer v2 500M Multi-Species</td>
                </tr>
                <tr>
                    <td>推理模式</td>
                    <td>零样本推理（Zero-shot Inference）</td>
                </tr>
                <tr>
                    <td>精度设置</td>
                    <td>16-bit混合精度</td>
                </tr>
            </table>

            <h3>数据处理流程</h3>
            <ol>
                <li>探针ID → 基因组位置转换</li>
                <li>DNA序列提取与嵌入生成</li>
                <li>特征过滤（匹配模型词汇表）</li>
                <li>模型前向传播预测</li>
                <li>结果后处理与可视化</li>
            </ol>
        </div>

        <div class="footer">
            <p>报告由元能基因GPT平台自动生成 | 生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>元能基因GPT平台: 首个具有链式思维推理能力的DNA甲基化基础模型</p>
        </div>
    </body>
    </html>
    """

    # 保存HTML报告
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"  ✓ HTML分析报告已生成: {report_path}")


# 生成报告
generate_html_report(combined_results, REPORT_PATH, FIGURES_DIR)

print("\n" + "=" * 80)
print("✅ 零样本推理完成！")
print("=" * 80)
print(f"\n📁 所有结果已保存到: {RESULTS_DIR}")
print("\n📊 数据文件:")
print(f"  - {age_results_path}")
print(f"  - {cancer_results_path}")
print(f"  - {combined_results_path}")
print("\n📈 可视化图表:")
print(f"  - {age_dist_path}")
print(f"  - {cancer_dist_path}")
print(f"  - {correlation_path}")
print(f"  - {summary_path}")
print("\n📄 分析报告:")
print(f"  - {REPORT_PATH}")
print("\n💡 提示: 在浏览器中打开 {REPORT_PATH} 查看完整的交互式分析报告")
print("=" * 80)

