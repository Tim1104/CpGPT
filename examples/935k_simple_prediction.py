"""
935k/EPICv2 简化预测脚本
Simplified prediction script for 935k/EPICv2 methylation data

这是一个简化版本的预测脚本，用于快速运行935k数据的所有预测功能。
This is a simplified script for quick predictions on 935k data.

支持的预测 / Supported predictions:
1. 多组织器官年龄预测 (Multi-tissue age prediction)
2. 癌症预测 (Cancer prediction)
3. 五种表观遗传时钟 (Five epigenetic clocks)
4. 血浆蛋白质预测 (Plasma protein prediction)
"""

import sys
from pathlib import Path
import pandas as pd
import torch
from lightning import seed_everything

from cpgpt.data.components.cpgpt_datasaver import CpGPTDataSaver
from cpgpt.data.components.dna_llm_embedder import DNALLMEmbedder
from cpgpt.data.components.illumina_methylation_prober import IlluminaMethylationProber
from cpgpt.data.cpgpt_datamodule import CpGPTDataModule
from cpgpt.infer.cpgpt_inferencer import CpGPTInferencer
from cpgpt.trainer.cpgpt_trainer import CpGPTTrainer

# ============================================================================
# 配置参数 - 请根据您的需求修改
# Configuration - Please modify according to your needs
# ============================================================================

# 数据路径 (支持 CSV 或 Arrow 格式)
# Data path (supports CSV or Arrow format)
RAW_DATA_PATH = "./data/Sample251212.arrow"

# 输出目录
# Output directory
RESULTS_DIR = "./results/935k_predictions"

# 依赖目录 (模型和DNA嵌入)
# Dependencies directory (models and DNA embeddings)
DEPENDENCIES_DIR = "./dependencies"

# 要运行的预测 (设置为 True 启用)
# Predictions to run (set to True to enable)
PREDICT_AGE = True          # 年龄预测
PREDICT_CANCER = True       # 癌症预测
PREDICT_CLOCKS = True       # 表观遗传时钟
PREDICT_PROTEINS = True     # 蛋白质预测

# 其他配置
RANDOM_SEED = 42
MAX_INPUT_LENGTH = 30000    # 如果内存不足，降低此值 (如 15000)
USE_CPU = False             # 设置为 True 使用 CPU (更稳定但更慢)

# ============================================================================
# 主程序
# Main Program
# ============================================================================

def main():
    """主函数"""
    
    # 设置随机种子
    seed_everything(RANDOM_SEED)
    
    # 创建输出目录
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    processed_dir = Path(RESULTS_DIR) / "processed"
    
    print("=" * 80)
    print("935k/EPICv2 甲基化数据预测")
    print("935k/EPICv2 Methylation Data Prediction")
    print("=" * 80)
    
    # ========================================================================
    # 步骤 1: 初始化组件
    # Step 1: Initialize components
    # ========================================================================
    print("\n[1/6] 初始化组件...")
    print("[1/6] Initializing components...")
    
    inferencer = CpGPTInferencer(dependencies_dir=DEPENDENCIES_DIR)
    embedder = DNALLMEmbedder(dependencies_dir=DEPENDENCIES_DIR)
    prober = IlluminaMethylationProber(dependencies_dir=DEPENDENCIES_DIR, embedder=embedder)
    
    # ========================================================================
    # 步骤 2: 下载依赖和模型 (首次运行)
    # Step 2: Download dependencies and models (first run only)
    # ========================================================================
    print("\n[2/6] 检查并下载依赖和模型...")
    print("[2/6] Checking and downloading dependencies and models...")

    # 下载 DNA 嵌入等依赖
    print("  - 下载 DNA 嵌入依赖...")
    print("  - Downloading DNA embedding dependencies...")
    inferencer.download_dependencies(species="human", overwrite=False)

    # 下载所需的模型
    models_to_download = []
    if PREDICT_AGE:
        models_to_download.append("age_cot")
    if PREDICT_CANCER:
        models_to_download.append("cancer")
    if PREDICT_CLOCKS:
        models_to_download.append("clock_proxies")
    if PREDICT_PROTEINS:
        models_to_download.append("proteins")

    if models_to_download:
        print(f"  - 下载 {len(models_to_download)} 个模型...")
        print(f"  - Downloading {len(models_to_download)} models...")
        for model_name in models_to_download:
            print(f"    • {model_name}")
            inferencer.download_model(model_name, overwrite=False)
    
    # ========================================================================
    # 步骤 3: 准备数据
    # Step 3: Prepare data
    # ========================================================================
    print("\n[3/6] 准备数据...")
    print("[3/6] Preparing data...")

    # 如果是CSV格式，检查并转换
    if RAW_DATA_PATH.endswith('.csv'):
        print("  - 检测到 CSV 格式...")
        print("  - Detected CSV format...")

        # 读取CSV检查格式
        df_check = pd.read_csv(RAW_DATA_PATH, nrows=5)
        first_col = df_check.columns[0]

        # 检查是否是厂商格式（第一列是探针ID）
        is_manufacturer_format = False
        if first_col.lower() in ['targetid', 'probe_id', 'probeid']:
            is_manufacturer_format = True
        elif df_check.iloc[0, 0].startswith('cg') and '_' in str(df_check.iloc[0, 0]):
            is_manufacturer_format = True

        if is_manufacturer_format:
            print("  ⚠️  检测到厂商格式（行=探针，列=样本）")
            print("  ⚠️  Detected manufacturer format (rows=probes, columns=samples)")
            print("  - 正在转换格式...")
            print("  - Converting format...")

            # 使用转换函数
            from convert_935k_format import convert_935k_format
            arrow_path = RAW_DATA_PATH.replace('.csv', '_converted.arrow')
            convert_935k_format(RAW_DATA_PATH, arrow_path, verbose=False)
            data_path = arrow_path

            print("  ✓ 格式转换完成")
            print("  ✓ Format conversion completed")
        else:
            # 标准格式，直接转换为Arrow
            print("  - 标准格式，转换为 Arrow...")
            print("  - Standard format, converting to Arrow...")
            df = pd.read_csv(RAW_DATA_PATH, index_col=0)
            arrow_path = RAW_DATA_PATH.replace('.csv', '.arrow')
            df.reset_index().to_feather(arrow_path)
            data_path = arrow_path
    else:
        data_path = RAW_DATA_PATH

    # 读取样本ID
    df_raw = pd.read_feather(data_path)
    sample_ids = df_raw.iloc[:, 0].tolist()
    print(f"  - 检测到 {len(sample_ids)} 个样本")
    print(f"  - Detected {len(sample_ids)} samples")
    
    # ========================================================================
    # 步骤 4: 数据预处理
    # Step 4: Data preprocessing
    # ========================================================================
    print("\n[4/6] 数据预处理 (探针ID → 基因组位置 → DNA嵌入)...")
    print("[4/6] Data preprocessing (Probe ID → Genomic location → DNA embedding)...")
    
    datasaver = CpGPTDataSaver(
        data_paths=data_path,
        processed_dir=str(processed_dir)
    )
    datasaver.process_files(prober=prober, embedder=embedder)
    
    # ========================================================================
    # 步骤 5: 运行预测
    # Step 5: Run predictions
    # ========================================================================
    print("\n[5/6] 运行预测...")
    print("[5/6] Running predictions...")
    
    # 配置训练器
    if USE_CPU:
        print("  - 使用 CPU 进行推理")
        print("  - Using CPU for inference")
        trainer = CpGPTTrainer(accelerator="cpu", precision="32")
    else:
        print("  - 使用 GPU 进行推理")
        print("  - Using GPU for inference")
        trainer = CpGPTTrainer(precision="16-mixed")
    
    all_results = {}
    
    # 5.1 年龄预测
    if PREDICT_AGE:
        print("\n  [5.1] 年龄预测...")
        print("  [5.1] Age prediction...")
        age_results = predict_age(inferencer, str(processed_dir), sample_ids, trainer)
        all_results['age'] = age_results
        age_results.to_csv(f"{RESULTS_DIR}/age_predictions.csv", index=False)
        print(f"    ✓ 保存到: {RESULTS_DIR}/age_predictions.csv")

    # 5.2 癌症预测
    if PREDICT_CANCER:
        print("\n  [5.2] 癌症预测...")
        print("  [5.2] Cancer prediction...")
        cancer_results = predict_cancer(inferencer, str(processed_dir), sample_ids, trainer)
        all_results['cancer'] = cancer_results
        cancer_results.to_csv(f"{RESULTS_DIR}/cancer_predictions.csv", index=False)
        print(f"    ✓ 保存到: {RESULTS_DIR}/cancer_predictions.csv")

    # 5.3 表观遗传时钟
    if PREDICT_CLOCKS:
        print("\n  [5.3] 表观遗传时钟预测...")
        print("  [5.3] Epigenetic clocks prediction...")
        clocks_results = predict_clocks(inferencer, str(processed_dir), sample_ids, trainer)
        all_results['clocks'] = clocks_results
        clocks_results.to_csv(f"{RESULTS_DIR}/clocks_predictions.csv", index=False)
        print(f"    ✓ 保存到: {RESULTS_DIR}/clocks_predictions.csv")

    # 5.4 蛋白质预测
    if PREDICT_PROTEINS:
        print("\n  [5.4] 蛋白质预测...")
        print("  [5.4] Protein prediction...")
        proteins_results = predict_proteins(inferencer, str(processed_dir), sample_ids, trainer)
        all_results['proteins'] = proteins_results
        proteins_results.to_csv(f"{RESULTS_DIR}/proteins_predictions.csv", index=False)
        print(f"    ✓ 保存到: {RESULTS_DIR}/proteins_predictions.csv")

    # ========================================================================
    # 步骤 6: 合并结果
    # Step 6: Combine results
    # ========================================================================
    print("\n[6/6] 合并结果...")
    print("[6/6] Combining results...")

    # 合并所有结果
    combined = pd.DataFrame({'sample_id': sample_ids})
    for key, df in all_results.items():
        combined = combined.merge(df, on='sample_id', how='left')

    combined.to_csv(f"{RESULTS_DIR}/combined_predictions.csv", index=False)
    print(f"  ✓ 所有结果已保存到: {RESULTS_DIR}/combined_predictions.csv")
    print(f"  ✓ All results saved to: {RESULTS_DIR}/combined_predictions.csv")

    # ========================================================================
    # 完成
    # Done
    # ========================================================================
    print("\n" + "=" * 80)
    print("预测完成！")
    print("Prediction completed!")
    print("=" * 80)
    print(f"\n结果文件:")
    print(f"Result files:")
    if PREDICT_AGE:
        print(f"  - 年龄预测: {RESULTS_DIR}/age_predictions.csv")
    if PREDICT_CANCER:
        print(f"  - 癌症预测: {RESULTS_DIR}/cancer_predictions.csv")
    if PREDICT_CLOCKS:
        print(f"  - 表观遗传时钟: {RESULTS_DIR}/clocks_predictions.csv")
    if PREDICT_PROTEINS:
        print(f"  - 蛋白质预测: {RESULTS_DIR}/proteins_predictions.csv")
    print(f"  - 合并结果: {RESULTS_DIR}/combined_predictions.csv")
    print()


# ============================================================================
# 预测函数
# Prediction Functions
# ============================================================================

def predict_age(inferencer, processed_dir, sample_ids, trainer):
    """年龄预测"""
    # 加载配置和模型
    config = inferencer.load_cpgpt_config(f"{DEPENDENCIES_DIR}/model/config/age_cot.yaml")
    model = inferencer.load_cpgpt_model(
        config,
        model_ckpt_path=f"{DEPENDENCIES_DIR}/model/weights/age_cot.ckpt",
        strict_load=True
    )

    # 创建数据模块
    datamodule = CpGPTDataModule(
        predict_dir=processed_dir,
        dependencies_dir=DEPENDENCIES_DIR,
        batch_size=1,
        num_workers=0,
        max_length=MAX_INPUT_LENGTH,
        dna_llm=config.data.dna_llm,
        dna_context_len=config.data.dna_context_len,
        sorting_strategy=config.data.sorting_strategy,
        pin_memory=False
    )

    # 执行预测
    predictions = trainer.predict(
        model=model,
        datamodule=datamodule,
        predict_mode="forward",
        return_keys=["pred_conditions"]
    )

    return pd.DataFrame({
        'sample_id': sample_ids,
        'predicted_age': predictions["pred_conditions"].flatten()
    })


def predict_cancer(inferencer, processed_dir, sample_ids, trainer):
    """癌症预测"""
    # 加载配置和模型
    config = inferencer.load_cpgpt_config(f"{DEPENDENCIES_DIR}/model/config/cancer.yaml")
    model = inferencer.load_cpgpt_model(
        config,
        model_ckpt_path=f"{DEPENDENCIES_DIR}/model/weights/cancer.ckpt",
        strict_load=True
    )

    # 创建数据模块
    datamodule = CpGPTDataModule(
        predict_dir=processed_dir,
        dependencies_dir=DEPENDENCIES_DIR,
        batch_size=1,
        num_workers=0,
        max_length=MAX_INPUT_LENGTH,
        dna_llm=config.data.dna_llm,
        dna_context_len=config.data.dna_context_len,
        sorting_strategy=config.data.sorting_strategy,
        pin_memory=False
    )

    # 执行预测
    predictions = trainer.predict(
        model=model,
        datamodule=datamodule,
        predict_mode="forward",
        return_keys=["pred_conditions"]
    )

    # 转换为概率
    cancer_logits = predictions["pred_conditions"].flatten()
    cancer_probabilities = torch.sigmoid(torch.tensor(cancer_logits)).numpy()

    return pd.DataFrame({
        'sample_id': sample_ids,
        'cancer_logit': cancer_logits,
        'cancer_probability': cancer_probabilities,
        'cancer_prediction': (cancer_probabilities > 0.5).astype(int)
    })


def predict_clocks(inferencer, processed_dir, sample_ids, trainer):
    """表观遗传时钟预测 - 5种时钟"""
    # 加载配置和模型
    config = inferencer.load_cpgpt_config(f"{DEPENDENCIES_DIR}/model/config/clock_proxies.yaml")
    model = inferencer.load_cpgpt_model(
        config,
        model_ckpt_path=f"{DEPENDENCIES_DIR}/model/weights/clock_proxies.ckpt",
        strict_load=True
    )

    # 创建数据模块
    datamodule = CpGPTDataModule(
        predict_dir=processed_dir,
        dependencies_dir=DEPENDENCIES_DIR,
        batch_size=1,
        num_workers=0,
        max_length=MAX_INPUT_LENGTH,
        dna_llm=config.data.dna_llm,
        dna_context_len=config.data.dna_context_len,
        sorting_strategy=config.data.sorting_strategy,
        pin_memory=False
    )

    # 执行预测
    predictions = trainer.predict(
        model=model,
        datamodule=datamodule,
        predict_mode="forward",
        return_keys=["pred_conditions"]
    )

    # 5种表观遗传时钟
    clock_names = ['altumage', 'dunedinpace', 'grimage2', 'hrsinchphenoage', 'pchorvath2013']
    clock_values = predictions["pred_conditions"]

    result_dict = {'sample_id': sample_ids}
    for i, clock_name in enumerate(clock_names):
        result_dict[clock_name] = clock_values[:, i]

    return pd.DataFrame(result_dict)


def predict_proteins(inferencer, processed_dir, sample_ids, trainer):
    """蛋白质预测"""
    # 加载配置和模型
    config = inferencer.load_cpgpt_config(f"{DEPENDENCIES_DIR}/model/config/proteins.yaml")
    model = inferencer.load_cpgpt_model(
        config,
        model_ckpt_path=f"{DEPENDENCIES_DIR}/model/weights/proteins.ckpt",
        strict_load=True
    )

    # 创建数据模块
    datamodule = CpGPTDataModule(
        predict_dir=processed_dir,
        dependencies_dir=DEPENDENCIES_DIR,
        batch_size=1,
        num_workers=0,
        max_length=MAX_INPUT_LENGTH,
        dna_llm=config.data.dna_llm,
        dna_context_len=config.data.dna_context_len,
        sorting_strategy=config.data.sorting_strategy,
        pin_memory=False
    )

    # 执行预测
    predictions = trainer.predict(
        model=model,
        datamodule=datamodule,
        predict_mode="forward",
        return_keys=["pred_conditions"]
    )

    # 蛋白质预测结果 (标准化值)
    protein_values = predictions["pred_conditions"]

    # 创建结果DataFrame (假设有多个蛋白质)
    result_dict = {'sample_id': sample_ids}

    # 如果是单个蛋白质
    if len(protein_values.shape) == 1:
        result_dict['protein_level'] = protein_values
    else:
        # 如果是多个蛋白质
        for i in range(protein_values.shape[1]):
            result_dict[f'protein_{i+1}'] = protein_values[:, i]

    return pd.DataFrame(result_dict)


# ============================================================================
# 运行主程序
# Run main program
# ============================================================================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断执行")
        print("Execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n错误: {str(e)}")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

