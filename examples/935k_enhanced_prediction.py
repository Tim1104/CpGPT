"""
935k/EPICv2 增强版预测脚本 - 包含死亡率预测、年龄加速、疾病风险分层和器官健康评分
Enhanced prediction script with mortality, age acceleration, disease risk stratification, and organ health scores

新增功能 / New Features:
1. CpGPTGrimAge3 死亡率预测 (Mortality prediction)
2. 年龄加速指标 (Age acceleration metrics)
3. CVD/癌症相关蛋白风险分层 (Disease risk stratification)
4. 器官健康评分 (Organ health scores) - 6大器官系统评估 ⭐新增
5. 详细PDF报告生成 (Comprehensive PDF report with organ health radar chart)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from lightning import seed_everything

from cpgpt.data.components.cpgpt_datasaver import CpGPTDataSaver
from cpgpt.data.components.dna_llm_embedder import DNALLMEmbedder
from cpgpt.data.components.illumina_methylation_prober import IlluminaMethylationProber
from cpgpt.data.cpgpt_datamodule import CpGPTDataModule
from cpgpt.infer.cpgpt_inferencer import CpGPTInferencer
from cpgpt.trainer.cpgpt_trainer import CpGPTTrainer

# ============================================================================
# 配置参数
# ============================================================================

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent

# 数据路径
RAW_DATA_PATH = SCRIPT_DIR / "data" / "Sample1107.arrow"

# 输出目录
RESULTS_DIR = SCRIPT_DIR / "results" / "935k_enhanced_predictions"

# 依赖目录
DEPENDENCIES_DIR = SCRIPT_DIR / "dependencies"
if not DEPENDENCIES_DIR.exists():
    DEPENDENCIES_DIR = PROJECT_ROOT / "dependencies"

# 预测开关
PREDICT_AGE = True
PREDICT_CANCER = True
PREDICT_CLOCKS = True
PREDICT_PROTEINS = True
PREDICT_MORTALITY = True  # 新增：死亡率预测

# Systems Age 结果（examples/systems_age/run.sh 生成）；文件不存在则跳过
SYSTEMS_AGE_CSV = SCRIPT_DIR / "systems_age" / "systems_age_results.csv"

# 其他配置
RANDOM_SEED = 42
# ⚠️ 重要：MAX_INPUT_LENGTH 应该与模型训练时的配置一致
# age_cot 模型训练时使用 20000
# clock_proxies 模型训练时使用 10000
# proteins 模型训练时使用 10000
# 建议使用 20000 作为通用值，或者为每个模型单独设置
MAX_INPUT_LENGTH = 20000  # 修改为与 age_cot 模型一致
USE_CPU = False

# 年龄加速计算配置
CHRONOLOGICAL_AGE_COLUMN = None  # 如果数据中有实际年龄，设置列名，如 "age"

# ⚠️ 重要：模型输出标准化参数
# 这些参数用于将模型的标准化输出转换回实际值
# 如果你有训练数据的统计信息，请在这里设置
# 格式：{'mean': 平均值, 'std': 标准差}
# 如果设置为 None，将直接使用模型输出（可能导致预测值偏差）
NORMALIZATION_PARAMS = {
    'age': None,  # 暂时设为None，等计算正确的参数
    'clocks': None,
    'proteins': None,
}

# ============================================================================
# 主程序
# ============================================================================

def main():
    """主函数"""
    seed_everything(RANDOM_SEED)
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    processed_dir = Path(RESULTS_DIR) / "processed"
    
    print("=" * 80)
    print("935k/EPICv2 增强版甲基化数据预测")
    print("Enhanced 935k/EPICv2 Methylation Data Prediction")
    print("=" * 80)
    
    # 步骤 1: 初始化组件
    print("\n[1/7] 初始化组件...")
    inferencer = CpGPTInferencer(dependencies_dir=str(DEPENDENCIES_DIR))
    embedder = DNALLMEmbedder(dependencies_dir=str(DEPENDENCIES_DIR))
    prober = IlluminaMethylationProber(dependencies_dir=str(DEPENDENCIES_DIR), embedder=embedder)
    
    # 步骤 2: 下载依赖和模型
    print("\n[2/7] 检查并下载依赖和模型...")
    inferencer.download_dependencies(species="human", overwrite=False)
    
    # 修复目录结构
    import os
    dna_embeddings_dir = Path(DEPENDENCIES_DIR) / "dna_embeddings"
    dna_embeddings_dir.mkdir(parents=True, exist_ok=True)
    homo_sapiens_link = dna_embeddings_dir / "homo_sapiens"
    human_source = Path(DEPENDENCIES_DIR) / "human" / "dna_embeddings" / "homo_sapiens"
    
    if human_source.exists() and not homo_sapiens_link.exists():
        try:
            homo_sapiens_link.symlink_to(human_source.resolve(), target_is_directory=True)
        except OSError:
            import shutil
            shutil.copytree(human_source, homo_sapiens_link, dirs_exist_ok=True)
    
    # 下载所需模型
    models_to_download = []
    if PREDICT_AGE:
        models_to_download.append("age_cot")
    if PREDICT_CANCER:
        models_to_download.append("cancer")
    if PREDICT_CLOCKS:
        models_to_download.append("clock_proxies")
    if PREDICT_PROTEINS:
        models_to_download.append("proteins")
    
    for model_name in models_to_download:
        print(f"  - 下载模型: {model_name}")
        inferencer.download_model(model_name, overwrite=False)
    
    # 步骤 3: 准备数据
    print("\n[3/7] 准备数据...")
    raw_data_path_str = str(RAW_DATA_PATH)
    
    # CSV格式转换逻辑（与原脚本相同）
    if raw_data_path_str.endswith('.csv'):
        df_check = pd.read_csv(raw_data_path_str, nrows=5)
        first_col = df_check.columns[0]
        is_manufacturer_format = first_col.lower() in ['targetid', 'probe_id', 'probeid']
        
        if is_manufacturer_format:
            from convert_935k_format import convert_935k_format
            arrow_path = raw_data_path_str.replace('.csv', '_converted.arrow')
            convert_935k_format(raw_data_path_str, arrow_path, verbose=False)
            data_path = arrow_path
        else:
            df = pd.read_csv(raw_data_path_str, index_col=0)
            arrow_path = raw_data_path_str.replace('.csv', '.arrow')
            df.reset_index().to_feather(arrow_path)
            data_path = arrow_path
    else:
        data_path = raw_data_path_str
    
    # 读取样本ID和实际年龄（如果有）
    df_raw = pd.read_feather(data_path)
    sample_ids = df_raw.iloc[:, 0].tolist()
    chronological_ages = None
    if CHRONOLOGICAL_AGE_COLUMN and CHRONOLOGICAL_AGE_COLUMN in df_raw.columns:
        chronological_ages = df_raw[CHRONOLOGICAL_AGE_COLUMN].values
    
    print(f"  - 检测到 {len(sample_ids)} 个样本")

    # 步骤 4: 数据预处理
    print("\n[4/7] 数据预处理...")
    if processed_dir.exists():
        import shutil
        shutil.rmtree(processed_dir)

    datasaver = CpGPTDataSaver(data_paths=data_path, processed_dir=str(processed_dir))
    datasaver.process_files(prober=prober, embedder=embedder)

    # 读取实际处理的样本ID
    processed_sample_ids = []
    for data_path_key in datasaver.dataset_metrics.keys():
        dataset_name = str(Path(data_path_key).with_suffix("")).replace("/", "_").replace("\\", "_")
        dataset_dir = processed_dir / dataset_name
        obs_names_file = dataset_dir / "obs_names.npy"
        if obs_names_file.exists():
            obs_names = np.load(obs_names_file, allow_pickle=True)
            processed_sample_ids.extend(obs_names.tolist())

    sample_ids = processed_sample_ids
    print(f"  - 实际处理的样本数: {len(sample_ids)}")

    # 生成DNA嵌入索引
    all_genomic_locations = datasaver.all_genomic_locations.get("homo_sapiens", set())
    embedder.parse_dna_embeddings(
        genomic_locations=sorted(all_genomic_locations),
        species="homo_sapiens",
        dna_llm="nucleotide-transformer-v2-500m-multi-species",
        dna_context_len=2001,
        batch_size=8,
        num_workers=1,
    )

    # 步骤 5: 运行预测
    print("\n[5/7] 运行预测...")
    trainer = CpGPTTrainer(precision="16-mixed") if not USE_CPU else CpGPTTrainer(accelerator="cpu", precision="32")

    all_results = {}

    # 5.1 年龄预测
    if PREDICT_AGE:
        print("\n  [5.1] 年龄预测...")
        age_results = predict_age(inferencer, str(processed_dir), sample_ids, trainer)
        all_results['age'] = age_results
        age_results.to_csv(f"{str(RESULTS_DIR)}/age_predictions.csv", index=False)

    # 5.2 癌症预测
    if PREDICT_CANCER:
        print("\n  [5.2] 癌症预测...")
        cancer_results = predict_cancer(inferencer, str(processed_dir), sample_ids, trainer)
        all_results['cancer'] = cancer_results
        cancer_results.to_csv(f"{str(RESULTS_DIR)}/cancer_predictions.csv", index=False)

    # 5.3 表观遗传时钟
    if PREDICT_CLOCKS:
        print("\n  [5.3] 表观遗传时钟预测...")
        clocks_results = predict_clocks(inferencer, str(processed_dir), sample_ids, trainer)
        all_results['clocks'] = clocks_results
        clocks_results.to_csv(f"{str(RESULTS_DIR)}/clocks_predictions.csv", index=False)

    # 5.4 蛋白质预测
    if PREDICT_PROTEINS:
        print("\n  [5.4] 蛋白质预测...")
        proteins_results = predict_proteins(inferencer, str(processed_dir), sample_ids, trainer)
        all_results['proteins'] = proteins_results
        proteins_results.to_csv(f"{str(RESULTS_DIR)}/proteins_predictions.csv", index=False)

    # 步骤 6: 高级分析
    print("\n[6/7] 高级分析...")

    # 合并所有结果
    combined = pd.DataFrame({'sample_id': sample_ids})
    for key, df in all_results.items():
        combined = combined.merge(df, on='sample_id', how='left')

    # 6.1 计算年龄加速指标
    if PREDICT_AGE and PREDICT_CLOCKS:
        print("\n  [6.1] 计算年龄加速指标...")
        age_acceleration = calculate_age_acceleration(combined, chronological_ages)
        combined = combined.merge(age_acceleration, on='sample_id', how='left')
        age_acceleration.to_csv(f"{str(RESULTS_DIR)}/age_acceleration.csv", index=False)

    # 6.2 CpGPTGrimAge3 死亡率预测
    if PREDICT_MORTALITY and PREDICT_PROTEINS and PREDICT_CLOCKS:
        print("\n  [6.2] CpGPTGrimAge3 死亡率预测...")
        mortality_results = calculate_grimage3_mortality(combined, proteins_results)
        combined = combined.merge(mortality_results, on='sample_id', how='left')
        mortality_results.to_csv(f"{str(RESULTS_DIR)}/mortality_predictions.csv", index=False)

    # 6.3 疾病风险分层
    if PREDICT_PROTEINS and PREDICT_CANCER:
        print("\n  [6.3] CVD/癌症风险分层...")
        risk_stratification = calculate_disease_risk(combined, proteins_results, cancer_results)
        combined = combined.merge(risk_stratification, on='sample_id', how='left')
        risk_stratification.to_csv(f"{str(RESULTS_DIR)}/risk_stratification.csv", index=False)

    # 6.4 器官健康评分 ⭐新增
    if PREDICT_PROTEINS:
        print("\n  [6.4] 器官健康评分（基于蛋白质生物标志物）...")
        organ_health_scores = calculate_organ_health_scores(proteins_results)
        combined = combined.merge(organ_health_scores, on='sample_id', how='left')
        organ_health_scores.to_csv(f"{str(RESULTS_DIR)}/organ_health_scores.csv", index=False)

    # 6.5 Systems Age（methylCIPHER，外部 R 流程生成）
    if SYSTEMS_AGE_CSV.exists():
        print("\n  [6.5] 合并 Systems Age 结果...")
        sa = pd.read_csv(SYSTEMS_AGE_CSV).drop(columns=['file'], errors='ignore')
        sa['sample_id'] = sa['sample'].astype(str).str.replace(r'\.AVG_Beta$', '', regex=True)
        sa = sa.drop(columns=['sample']).add_prefix('sysage_').rename(columns={'sysage_sample_id': 'sample_id'})
        combined['sample_id'] = combined['sample_id'].astype(str)
        combined = combined.merge(sa, on='sample_id', how='left')
        sa.to_csv(f"{str(RESULTS_DIR)}/systems_age_scores.csv", index=False)

    # 保存合并结果
    combined.to_csv(f"{str(RESULTS_DIR)}/combined_predictions.csv", index=False)

    # 步骤 7: 生成PDF报告
    print("\n[7/7] 生成PDF报告...")
    # 为每个样本生成独立的PDF报告
    for idx, row in combined.iterrows():
        sample_id = row['sample_id']
        print(f"  生成样本 {sample_id} 的PDF报告...")
        generate_individual_pdf_report(row, str(RESULTS_DIR), sample_id)

    print("\n" + "=" * 80)
    print("预测完成！")
    print("=" * 80)
    print(f"\n结果文件:")
    print(f"  - 合并结果: {RESULTS_DIR}/combined_predictions.csv")
    print(f"  - 年龄加速: {RESULTS_DIR}/age_acceleration.csv")
    print(f"  - 死亡率预测: {RESULTS_DIR}/mortality_predictions.csv")
    print(f"  - 风险分层: {RESULTS_DIR}/risk_stratification.csv")
    print(f"  - 器官健康评分: {RESULTS_DIR}/organ_health_scores.csv")
    print(f"  - PDF报告: {RESULTS_DIR}/comprehensive_report.pdf")
    print()


# ============================================================================
# 预测函数
# ============================================================================

def denormalize_predictions(values, mean=None, std=None):
    """
    反标准化预测值

    Args:
        values: 标准化的预测值（numpy array 或 float）
        mean: 训练数据的均值（如果为 None，则不进行反标准化）
        std: 训练数据的标准差（如果为 None，则不进行反标准化）

    Returns:
        反标准化后的值
    """
    if mean is None or std is None:
        print("  ⚠️ 警告：未提供标准化参数，直接使用模型输出值")
        print("  💡 提示：如果预测值看起来不合理，请设置 NORMALIZATION_PARAMS")
        return values

    return values * std + mean


def predict_age(inferencer, processed_dir, sample_ids, trainer):
    """年龄预测"""
    config = inferencer.load_cpgpt_config(f"{str(DEPENDENCIES_DIR)}/model/config/age_cot.yaml")
    model = inferencer.load_cpgpt_model(
        config,
        model_ckpt_path=f"{str(DEPENDENCIES_DIR)}/model/weights/age_cot.ckpt",
        strict_load=True
    )

    # ⚠️ 使用与训练时一致的 max_length
    model_max_length = config.data.get('max_length', 20000)
    if model_max_length != MAX_INPUT_LENGTH:
        print(f"  ℹ️ 注意：自动使用 age_cot 模型的 max_length={model_max_length}（全局设置为 {MAX_INPUT_LENGTH}）")
    else:
        print(f"  ✓ 使用 max_length={model_max_length}")

    datamodule = CpGPTDataModule(
        predict_dir=processed_dir,
        dependencies_dir=str(DEPENDENCIES_DIR),
        batch_size=1,
        num_workers=0,
        max_length=model_max_length,  # 使用模型配置的值
        dna_llm=config.data.dna_llm,
        dna_context_len=config.data.dna_context_len,
        sorting_strategy=config.data.sorting_strategy,
        pin_memory=False
    )

    predictions = trainer.predict(
        model=model,
        datamodule=datamodule,
        predict_mode="forward",
        return_keys=["pred_conditions"]
    )

    pred_values = predictions["pred_conditions"].flatten().cpu().numpy()

    # 反标准化预测值
    if NORMALIZATION_PARAMS['age'] is not None:
        pred_values = denormalize_predictions(
            pred_values,
            mean=NORMALIZATION_PARAMS['age'].get('mean'),
            std=NORMALIZATION_PARAMS['age'].get('std')
        )

    return pd.DataFrame({'sample_id': sample_ids, 'predicted_age': pred_values})


def predict_cancer(inferencer, processed_dir, sample_ids, trainer):
    """癌症预测"""
    config = inferencer.load_cpgpt_config(f"{str(DEPENDENCIES_DIR)}/model/config/cancer.yaml")
    model = inferencer.load_cpgpt_model(
        config,
        model_ckpt_path=f"{str(DEPENDENCIES_DIR)}/model/weights/cancer.ckpt",
        strict_load=True
    )

    # ⚠️ 使用与训练时一致的 max_length
    model_max_length = config.data.get('max_length', 10000)
    if model_max_length != MAX_INPUT_LENGTH:
        print(f"  ℹ️ 注意：自动使用 cancer 模型的 max_length={model_max_length}（全局设置为 {MAX_INPUT_LENGTH}）")
    else:
        print(f"  ✓ 使用 max_length={model_max_length}")

    datamodule = CpGPTDataModule(
        predict_dir=processed_dir,
        dependencies_dir=str(DEPENDENCIES_DIR),
        batch_size=1,
        num_workers=0,
        max_length=model_max_length,  # 使用模型配置的值
        dna_llm=config.data.dna_llm,
        dna_context_len=config.data.dna_context_len,
        sorting_strategy=config.data.sorting_strategy,
        pin_memory=False
    )

    predictions = trainer.predict(
        model=model,
        datamodule=datamodule,
        predict_mode="forward",
        return_keys=["pred_conditions"]
    )

    cancer_logits = predictions["pred_conditions"].flatten().cpu().numpy()
    cancer_probabilities = torch.sigmoid(torch.tensor(cancer_logits)).numpy()

    return pd.DataFrame({
        'sample_id': sample_ids,
        'cancer_logit': cancer_logits,
        'cancer_probability': cancer_probabilities,
        'cancer_prediction': (cancer_probabilities > 0.5).astype(int)
    })


def predict_clocks(inferencer, processed_dir, sample_ids, trainer):
    """表观遗传时钟预测"""
    config = inferencer.load_cpgpt_config(f"{str(DEPENDENCIES_DIR)}/model/config/clock_proxies.yaml")
    model = inferencer.load_cpgpt_model(
        config,
        model_ckpt_path=f"{str(DEPENDENCIES_DIR)}/model/weights/clock_proxies.ckpt",
        strict_load=True
    )

    # ⚠️ 使用与训练时一致的 max_length
    model_max_length = config.data.get('max_length', 10000)
    if model_max_length != MAX_INPUT_LENGTH:
        print(f"  ℹ️ 注意：自动使用 clock_proxies 模型的 max_length={model_max_length}（全局设置为 {MAX_INPUT_LENGTH}）")
    else:
        print(f"  ✓ 使用 max_length={model_max_length}")

    datamodule = CpGPTDataModule(
        predict_dir=processed_dir,
        dependencies_dir=str(DEPENDENCIES_DIR),
        batch_size=1,
        num_workers=0,
        max_length=model_max_length,  # 使用模型配置的值
        dna_llm=config.data.dna_llm,
        dna_context_len=config.data.dna_context_len,
        sorting_strategy=config.data.sorting_strategy,
        pin_memory=False
    )

    predictions = trainer.predict(
        model=model,
        datamodule=datamodule,
        predict_mode="forward",
        return_keys=["pred_conditions"]
    )

    clock_names = ['altumage', 'dunedinpace', 'grimage2', 'hrsinchphenoage', 'pchorvath2013']
    clock_values = predictions["pred_conditions"].cpu().numpy()

    result_dict = {'sample_id': sample_ids}
    for i, clock_name in enumerate(clock_names):
        # 反标准化每个时钟的预测值
        values = clock_values[:, i]
        if NORMALIZATION_PARAMS['clocks'] is not None and clock_name in NORMALIZATION_PARAMS['clocks']:
            values = denormalize_predictions(
                values,
                mean=NORMALIZATION_PARAMS['clocks'][clock_name].get('mean'),
                std=NORMALIZATION_PARAMS['clocks'][clock_name].get('std')
            )
        result_dict[clock_name] = values

    return pd.DataFrame(result_dict)


def predict_proteins(inferencer, processed_dir, sample_ids, trainer):
    """
    蛋白质预测

    注意：模型预测 322 种血浆蛋白质的标准化值（均值=0，标准差=1）
    - 负值：低于平均水平（通常表示更健康）
    - 0：平均水平
    - 正值：高于平均水平（可能表示炎症或疾病风险）

    这些标准化值可以直接用于风险评估和器官健康评分。
    """
    config = inferencer.load_cpgpt_config(f"{str(DEPENDENCIES_DIR)}/model/config/proteins.yaml")
    model = inferencer.load_cpgpt_model(
        config,
        model_ckpt_path=f"{str(DEPENDENCIES_DIR)}/model/weights/proteins.ckpt",
        strict_load=True
    )

    # ⚠️ 使用与训练时一致的 max_length
    model_max_length = config.data.get('max_length', 10000)
    if model_max_length != MAX_INPUT_LENGTH:
        print(f"  ℹ️ 注意：自动使用 proteins 模型的 max_length={model_max_length}（全局设置为 {MAX_INPUT_LENGTH}）")
    else:
        print(f"  ✓ 使用 max_length={model_max_length}")

    datamodule = CpGPTDataModule(
        predict_dir=processed_dir,
        dependencies_dir=str(DEPENDENCIES_DIR),
        batch_size=1,
        num_workers=0,
        max_length=model_max_length,  # 使用模型配置的值
        dna_llm=config.data.dna_llm,
        dna_context_len=config.data.dna_context_len,
        sorting_strategy=config.data.sorting_strategy,
        pin_memory=False
    )

    predictions = trainer.predict(
        model=model,
        datamodule=datamodule,
        predict_mode="forward",
        return_keys=["pred_conditions"]
    )

    protein_values = predictions["pred_conditions"].cpu().numpy()

    # 蛋白质名称（基于GrimAge3相关蛋白）
    protein_names = get_protein_names()

    result_dict = {'sample_id': sample_ids}
    num_proteins = min(protein_values.shape[1], len(protein_names))

    print(f"  ℹ️ 预测了 {num_proteins} 种蛋白质的标准化值")
    print(f"  ℹ️ 标准化值范围：均值=0，标准差=1")
    print(f"  ℹ️ 典型范围：[-3, +3]（99.7% 的值在此范围内）")

    for i in range(num_proteins):
        # 蛋白质值保持标准化（不需要反标准化）
        # 标准化值可以直接用于风险评估
        result_dict[protein_names[i]] = protein_values[:, i]

    return pd.DataFrame(result_dict)


# ============================================================================
# 高级分析函数
# ============================================================================

def calculate_age_acceleration(combined_df, chronological_ages=None):
    """
    计算年龄加速指标
    Age Acceleration = Epigenetic Age - Chronological Age
    """
    results = {'sample_id': combined_df['sample_id'].values}

    # 如果有实际年龄，计算年龄加速
    if chronological_ages is not None:
        if 'predicted_age' in combined_df.columns:
            results['age_acceleration_cot'] = combined_df['predicted_age'].values - chronological_ages

        # 对每个时钟计算年龄加速
        clock_names = ['altumage', 'grimage2', 'hrsinchphenoage', 'pchorvath2013']
        for clock in clock_names:
            if clock in combined_df.columns:
                results[f'age_acceleration_{clock}'] = combined_df[clock].values - chronological_ages

        # DunedinPACE 是速度指标，不是年龄
        if 'dunedinpace' in combined_df.columns:
            results['aging_pace_dunedinpace'] = combined_df['dunedinpace'].values
    else:
        # 没有实际年龄时，使用predicted_age作为参考
        if 'predicted_age' in combined_df.columns:
            ref_age = combined_df['predicted_age'].values

            clock_names = ['altumage', 'grimage2', 'hrsinchphenoage', 'pchorvath2013']
            for clock in clock_names:
                if clock in combined_df.columns:
                    results[f'age_diff_{clock}_vs_cot'] = combined_df[clock].values - ref_age

    return pd.DataFrame(results)


def calculate_grimage3_mortality(combined_df, proteins_df):
    """
    计算 CpGPTGrimAge3 死亡率风险
    基于蛋白质水平和表观遗传时钟
    """
    results = {'sample_id': combined_df['sample_id'].values}

    # GrimAge2 作为基础死亡率指标
    if 'grimage2' in combined_df.columns:
        grimage2 = combined_df['grimage2'].values
        results['grimage2_age'] = grimage2

        # 计算相对于年龄的风险
        if 'predicted_age' in combined_df.columns:
            age_diff = grimage2 - combined_df['predicted_age'].values
            results['grimage2_age_acceleration'] = age_diff

            # 风险分层：基于年龄加速
            risk_categories = []
            for diff in age_diff:
                if diff < -5:
                    risk_categories.append('低风险')
                elif diff < 0:
                    risk_categories.append('中低风险')
                elif diff < 5:
                    risk_categories.append('中高风险')
                else:
                    risk_categories.append('高风险')
            results['mortality_risk_category'] = risk_categories

    # 基于关键蛋白质计算综合风险评分
    cvd_proteins = get_cvd_related_proteins()
    cancer_proteins = get_cancer_related_proteins()

    # CVD风险评分
    cvd_score = np.zeros(len(combined_df))
    cvd_count = 0
    for protein in cvd_proteins:
        if protein in proteins_df.columns:
            cvd_score += proteins_df[protein].values
            cvd_count += 1
    if cvd_count > 0:
        results['cvd_protein_score'] = cvd_score / cvd_count

    # 癌症风险评分
    cancer_score = np.zeros(len(combined_df))
    cancer_count = 0
    for protein in cancer_proteins:
        if protein in proteins_df.columns:
            cancer_score += proteins_df[protein].values
            cancer_count += 1
    if cancer_count > 0:
        results['cancer_protein_score'] = cancer_score / cancer_count

    return pd.DataFrame(results)


def calculate_disease_risk(combined_df, proteins_df, cancer_df):
    """
    疾病风险分层
    结合蛋白质、癌症预测和表观遗传时钟
    """
    results = {'sample_id': combined_df['sample_id'].values}

    # 癌症风险分层
    if 'cancer_probability' in cancer_df.columns:
        cancer_prob = cancer_df['cancer_probability'].values
        risk_levels = []
        for prob in cancer_prob:
            if prob < 0.2:
                risk_levels.append('低风险')
            elif prob < 0.4:
                risk_levels.append('中低风险')
            elif prob < 0.6:
                risk_levels.append('中高风险')
            else:
                risk_levels.append('高风险')
        results['cancer_risk_level'] = risk_levels

    # CVD风险分层（基于蛋白质）
    cvd_proteins = get_cvd_related_proteins()
    cvd_scores = []
    for idx in range(len(combined_df)):
        score = 0
        count = 0
        for protein in cvd_proteins:
            if protein in proteins_df.columns:
                score += proteins_df[protein].iloc[idx]
                count += 1
        cvd_scores.append(score / count if count > 0 else 0)

    results['cvd_risk_score'] = cvd_scores

    # CVD风险等级
    cvd_risk_levels = []
    for score in cvd_scores:
        if score < -0.5:
            cvd_risk_levels.append('低风险')
        elif score < 0:
            cvd_risk_levels.append('中低风险')
        elif score < 0.5:
            cvd_risk_levels.append('中高风险')
        else:
            cvd_risk_levels.append('高风险')
    results['cvd_risk_level'] = cvd_risk_levels

    return pd.DataFrame(results)


def calculate_organ_health_scores(proteins_df):
    """
    计算器官健康评分
    基于器官特异性蛋白质生物标志物

    评分范围：0-100
    - 90-100: 优秀 (Excellent)
    - 75-89: 良好 (Good)
    - 60-74: 一般 (Fair)
    - 40-59: 较差 (Poor)
    - 0-39: 差 (Very Poor)
    """
    results = {'sample_id': proteins_df['sample_id'].values}

    organ_proteins = get_organ_specific_proteins()

    # 对每个器官系统计算健康评分
    for organ_key, organ_info in organ_proteins.items():
        organ_name = organ_info['name']
        protein_list = organ_info['proteins']

        # 计算该器官的蛋白质平均值
        organ_scores = []
        organ_protein_values = []

        for idx in range(len(proteins_df)):
            protein_values = []
            for protein in protein_list:
                if protein in proteins_df.columns:
                    value = proteins_df[protein].iloc[idx]
                    if not pd.isna(value):
                        protein_values.append(value)

            if len(protein_values) > 0:
                # 计算平均值（标准化的蛋白质值）
                avg_value = np.mean(protein_values)
                organ_protein_values.append(avg_value)

                # 转换为健康评分 (0-100)
                # 假设蛋白质值已经标准化（均值0，标准差1）
                # 负值表示低于平均水平（更健康），正值表示高于平均水平（风险更高）
                # 转换公式：score = 100 - (value + 3) * 100 / 6
                # 这样 value=-3 -> score=100, value=0 -> score=50, value=3 -> score=0
                health_score = max(0, min(100, 100 - (avg_value + 3) * 100 / 6))
                organ_scores.append(health_score)
            else:
                organ_scores.append(np.nan)
                organ_protein_values.append(np.nan)

        # 保存评分和原始蛋白质值
        results[f'{organ_key}_score'] = organ_scores
        results[f'{organ_key}_protein_avg'] = organ_protein_values

        # 健康等级分类
        health_levels = []
        for score in organ_scores:
            if pd.isna(score):
                health_levels.append('未知')
            elif score >= 90:
                health_levels.append('优秀')
            elif score >= 75:
                health_levels.append('良好')
            elif score >= 60:
                health_levels.append('一般')
            elif score >= 40:
                health_levels.append('较差')
            else:
                health_levels.append('差')
        results[f'{organ_key}_level'] = health_levels

    # 计算综合健康评分（所有器官的平均）
    all_organ_scores = []
    for idx in range(len(proteins_df)):
        scores = []
        for organ_key in organ_proteins.keys():
            score = results[f'{organ_key}_score'][idx]
            if not pd.isna(score):
                scores.append(score)
        if len(scores) > 0:
            all_organ_scores.append(np.mean(scores))
        else:
            all_organ_scores.append(np.nan)

    results['overall_health_score'] = all_organ_scores

    # 综合健康等级
    overall_levels = []
    for score in all_organ_scores:
        if pd.isna(score):
            overall_levels.append('未知')
        elif score >= 90:
            overall_levels.append('优秀')
        elif score >= 75:
            overall_levels.append('良好')
        elif score >= 60:
            overall_levels.append('一般')
        elif score >= 40:
            overall_levels.append('较差')
        else:
            overall_levels.append('差')
    results['overall_health_level'] = overall_levels

    return pd.DataFrame(results)


# ============================================================================
# 辅助函数
# ============================================================================

def get_protein_names():
    """获取蛋白质名称列表（基于GrimAge相关蛋白）"""
    # 这些是GrimAge3中使用的关键蛋白质
    return [
        'ADM', 'B2M', 'Cystatin_C', 'GDF15', 'Leptin', 'PAI1', 'TIMP1',
        'CRP', 'IL6', 'TNF_alpha', 'MMP1', 'MMP9', 'VEGF', 'ICAM1',
        'VCAM1', 'E_selectin', 'P_selectin', 'Fibrinogen', 'vWF', 'D_dimer',
        # 更多蛋白质...
    ] + [f'protein_{i}' for i in range(21, 323)]  # 总共322个蛋白质


def get_cvd_related_proteins():
    """获取CVD相关蛋白质"""
    return [
        'ADM',  # Adrenomedullin - 心血管调节
        'CRP',  # C-reactive protein - 炎症标志物
        'IL6',  # Interleukin-6 - 炎症
        'TNF_alpha',  # Tumor necrosis factor alpha - 炎症
        'ICAM1',  # Intercellular adhesion molecule 1 - 内皮功能
        'VCAM1',  # Vascular cell adhesion molecule 1 - 内皮功能
        'E_selectin',  # 内皮激活
        'P_selectin',  # 血小板激活
        'Fibrinogen',  # 凝血因子
        'vWF',  # von Willebrand factor - 凝血
        'D_dimer',  # 凝血激活
        'PAI1',  # Plasminogen activator inhibitor-1 - 纤溶抑制
        'MMP1',  # Matrix metalloproteinase-1 - 血管重塑
        'MMP9',  # Matrix metalloproteinase-9 - 血管重塑
    ]


def get_cancer_related_proteins():
    """获取癌症相关蛋白质"""
    return [
        'GDF15',  # Growth differentiation factor 15 - 肿瘤标志物
        'VEGF',  # Vascular endothelial growth factor - 血管生成
        'IL6',  # Interleukin-6 - 炎症和肿瘤
        'TNF_alpha',  # Tumor necrosis factor alpha
        'MMP1',  # Matrix metalloproteinase-1 - 肿瘤侵袭
        'MMP9',  # Matrix metalloproteinase-9 - 肿瘤侵袭
        'Leptin',  # 与肥胖相关癌症
        'CRP',  # C-reactive protein - 炎症
        'B2M',  # Beta-2-microglobulin - 免疫功能
    ]


def get_organ_specific_proteins():
    """
    获取器官特异性蛋白质标志物
    基于最新研究（Nature 2023, Lancet Digital Health 2025）
    """
    return {
        'heart': {
            'name': '心脏 (Heart)',
            'proteins': [
                'ADM',  # Adrenomedullin - 心血管调节
                'CRP',  # C-reactive protein - 心血管炎症
                'IL6',  # Interleukin-6 - 心脏炎症
                'TNF_alpha',  # TNF-α - 心肌损伤
                'ICAM1',  # ICAM-1 - 内皮功能
                'VCAM1',  # VCAM-1 - 内皮功能
                'E_selectin',  # E-selectin - 内皮激活
                'P_selectin',  # P-selectin - 血小板激活
                'Fibrinogen',  # 凝血因子
                'vWF',  # von Willebrand factor
                'PAI1',  # PAI-1 - 纤溶抑制
                'MMP1',  # MMP-1 - 心脏重塑
                'MMP9',  # MMP-9 - 心脏重塑
            ],
            'description': '心血管系统健康指标，包括内皮功能、炎症和凝血状态'
        },
        'kidney': {
            'name': '肾脏 (Kidney)',
            'proteins': [
                'Cystatin_C',  # Cystatin C - 肾功能金标准
                'B2M',  # β2-微球蛋白 - 肾小球滤过
                'CRP',  # CRP - 肾脏炎症
                'IL6',  # IL-6 - 肾脏炎症
                'TNF_alpha',  # TNF-α - 肾损伤
                'VEGF',  # VEGF - 肾血管
                'PAI1',  # PAI-1 - 肾纤维化
            ],
            'description': '肾脏功能和炎症状态评估'
        },
        'liver': {
            'name': '肝脏 (Liver)',
            'proteins': [
                'CRP',  # CRP - 肝脏合成
                'Fibrinogen',  # 纤维蛋白原 - 肝脏合成
                'PAI1',  # PAI-1 - 肝纤维化
                'MMP1',  # MMP-1 - 肝纤维化
                'MMP9',  # MMP-9 - 肝纤维化
                'IL6',  # IL-6 - 肝脏炎症
                'TNF_alpha',  # TNF-α - 肝损伤
                'GDF15',  # GDF-15 - 肝脏应激
            ],
            'description': '肝脏合成功能和纤维化风险'
        },
        'immune': {
            'name': '免疫系统 (Immune System)',
            'proteins': [
                'IL6',  # IL-6 - 促炎细胞因子
                'TNF_alpha',  # TNF-α - 促炎细胞因子
                'CRP',  # CRP - 急性期反应
                'B2M',  # β2-微球蛋白 - 免疫激活
                'ICAM1',  # ICAM-1 - 免疫细胞粘附
                'VCAM1',  # VCAM-1 - 免疫细胞粘附
                'E_selectin',  # E-selectin - 免疫细胞募集
                'P_selectin',  # P-selectin - 免疫细胞募集
            ],
            'description': '免疫系统激活和炎症状态'
        },
        'metabolic': {
            'name': '代谢系统 (Metabolic System)',
            'proteins': [
                'Leptin',  # Leptin - 能量代谢
                'GDF15',  # GDF-15 - 代谢应激
                'PAI1',  # PAI-1 - 代谢综合征
                'CRP',  # CRP - 代谢炎症
                'IL6',  # IL-6 - 代谢炎症
                'TNF_alpha',  # TNF-α - 胰岛素抵抗
                'ADM',  # ADM - 代谢调节
            ],
            'description': '代谢健康和能量平衡'
        },
        'vascular': {
            'name': '血管系统 (Vascular System)',
            'proteins': [
                'ICAM1',  # ICAM-1 - 内皮功能
                'VCAM1',  # VCAM-1 - 内皮功能
                'E_selectin',  # E-selectin - 内皮激活
                'P_selectin',  # P-selectin - 内皮激活
                'vWF',  # vWF - 内皮损伤
                'MMP1',  # MMP-1 - 血管重塑
                'MMP9',  # MMP-9 - 血管重塑
                'VEGF',  # VEGF - 血管生成
                'ADM',  # ADM - 血管张力
            ],
            'description': '血管内皮功能和血管健康'
        },
    }


# 全局字体配置（只执行一次）
_FONT_CONFIGURED = False
_CHINESE_FONT_PATH = None

def configure_chinese_font():
    """配置中文字体（全局执行一次）"""
    global _FONT_CONFIGURED, _CHINESE_FONT_PATH

    if _FONT_CONFIGURED:
        return _CHINESE_FONT_PATH

    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    from pathlib import Path

    print("  配置中文字体...")

    # 尝试多个中文字体路径
    font_paths = [
        '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
        '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/truetype/arphic/uming.ttc',
        '/System/Library/Fonts/STHeiti Light.ttc',
        'C:\\Windows\\Fonts\\simhei.ttf',
    ]

    chinese_font_found = False

    # 方法1：直接使用字体文件路径
    for font_path in font_paths:
        if Path(font_path).exists():
            try:
                from matplotlib.font_manager import FontProperties
                font_prop = FontProperties(fname=font_path)
                font_name = font_prop.get_name()

                # 设置全局字体
                plt.rcParams['font.family'] = 'sans-serif'
                plt.rcParams['font.sans-serif'] = [font_name, 'DejaVu Sans']
                plt.rcParams['axes.unicode_minus'] = False

                # 测试字体是否真的可用
                fig, ax = plt.subplots(figsize=(1, 1))
                ax.text(0.5, 0.5, '测试', fontsize=10)
                plt.close(fig)

                chinese_font_found = True
                _CHINESE_FONT_PATH = font_path
                print(f"  ✓ 使用中文字体文件: {font_path}")
                print(f"  ✓ 字体名称: {font_name}")
                break
            except Exception as e:
                print(f"  ✗ 字体 {font_path} 加载失败: {e}")
                continue

    # 方法2：使用字体名称
    if not chinese_font_found:
        chinese_fonts = [
            'WenQuanYi Micro Hei',
            'WenQuanYi Zen Hei',
            'Noto Sans CJK SC',
            'Noto Sans CJK',
            'AR PL UMing CN',
            'SimHei',
            'STHeiti',
        ]

        # 重建字体列表
        try:
            fm._rebuild()
        except:
            pass

        available_fonts = set(f.name for f in fm.fontManager.ttflist)

        for font in chinese_fonts:
            if font in available_fonts:
                try:
                    plt.rcParams['font.family'] = 'sans-serif'
                    plt.rcParams['font.sans-serif'] = [font, 'DejaVu Sans']
                    plt.rcParams['axes.unicode_minus'] = False

                    # 测试字体
                    fig, ax = plt.subplots(figsize=(1, 1))
                    ax.text(0.5, 0.5, '测试', fontsize=10)
                    plt.close(fig)

                    chinese_font_found = True
                    print(f"  ✓ 使用中文字体名称: {font}")
                    break
                except Exception as e:
                    print(f"  ✗ 字体 {font} 加载失败: {e}")
                    continue

    if not chinese_font_found:
        print("  ⚠ 未找到可用的中文字体！")
        print("  💡 建议执行: sudo apt-get install fonts-wqy-microhei")
        print("  💡 然后清除缓存: rm -rf ~/.cache/matplotlib")
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

    _FONT_CONFIGURED = True
    return _CHINESE_FONT_PATH


def generate_individual_pdf_report(sample_data, output_dir, sample_id):
    """
    为单个样本生成详细的PDF报告
    包含所有预测结果和可视化图表
    """
    try:
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
        from reportlab.lib import colors
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')

        # 配置中文字体（只在第一次调用时执行）
        configure_chinese_font()

        # 注册PDF中文字体
        try:
            font_paths = [
                '/System/Library/Fonts/STHeiti Light.ttc',  # macOS
                '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',  # Linux WenQuanYi
                '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf',  # Linux Droid
                'C:\\Windows\\Fonts\\simhei.ttf',  # Windows
            ]

            chinese_font = 'Helvetica'
            for font_path in font_paths:
                try:
                    from pathlib import Path
                    if Path(font_path).exists():
                        pdfmetrics.registerFont(TTFont('ChineseFont', font_path))
                        chinese_font = 'ChineseFont'
                        break
                except:
                    continue
        except:
            chinese_font = 'Helvetica'

        # 创建PDF
        pdf_path = f"{output_dir}/report_{sample_id}.pdf"
        doc = SimpleDocTemplate(pdf_path, pagesize=A4)
        story = []
        styles = getSampleStyleSheet()

        # 自定义样式（使用中文字体）
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontName=chinese_font,
            fontSize=24,
            textColor=colors.HexColor('#2C3E50'),
            spaceAfter=30,
            alignment=1  # 居中
        )

        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontName=chinese_font,
            fontSize=16,
            textColor=colors.HexColor('#34495E'),
            spaceAfter=12
        )

        body_style = ParagraphStyle(
            'CustomBody',
            parent=styles['BodyText'],
            fontName=chinese_font,
            fontSize=10,
            leading=14
        )

        # 标题
        story.append(Paragraph("DNA Methylation Analysis Report", title_style))
        story.append(Paragraph("DNA甲基化分析报告", title_style))
        story.append(Spacer(1, 0.2*inch))

        # 样本ID样式
        sample_id_style = ParagraphStyle(
            'SampleID',
            parent=styles['Heading3'],
            fontName=chinese_font,
            fontSize=14,
            textColor=colors.HexColor('#2C3E50')
        )
        story.append(Paragraph(f"Sample ID: {sample_id}", sample_id_style))
        story.append(Spacer(1, 0.3*inch))

        # 第1章：执行摘要
        story.append(Paragraph("1. Executive Summary / 执行摘要", heading_style))

        summary_data = [['Metric / 指标', 'Value / 值']]
        summary_data.append(['Sample ID / 样本ID', str(sample_id)])

        if 'predicted_age' in sample_data.index and pd.notna(sample_data['predicted_age']):
            summary_data.append(['Predicted Age / 预测年龄', f"{sample_data['predicted_age']:.1f} years"])

        if 'cancer_prediction' in sample_data.index and pd.notna(sample_data['cancer_prediction']):
            cancer_status = "Positive / 阳性" if sample_data['cancer_prediction'] == 1 else "Negative / 阴性"
            summary_data.append(['Cancer Prediction / 癌症预测', cancer_status])

        if 'cancer_probability' in sample_data.index and pd.notna(sample_data['cancer_probability']):
            summary_data.append(['Cancer Probability / 癌症概率', f"{sample_data['cancer_probability']:.2%}"])

        if 'mortality_risk_category' in sample_data.index and pd.notna(sample_data['mortality_risk_category']):
            summary_data.append(['Mortality Risk / 死亡风险', str(sample_data['mortality_risk_category'])])

        summary_table = Table(summary_data, colWidths=[3.5*inch, 2.5*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498DB')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), chinese_font),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 0.3*inch))

        # ========================================================================
        # 第2章：器官健康评分（基于蛋白质生物标志物）
        # ========================================================================

        story.append(PageBreak())
        story.append(Paragraph("2. Organ Health Scores / 器官健康评分", heading_style))
        story.append(Paragraph("Based on organ-specific protein biomarkers / 基于器官特异性蛋白质生物标志物", body_style))
        story.append(Spacer(1, 0.2*inch))

        # 2.1 器官健康评分表格
        organ_systems = get_organ_specific_proteins()
        organ_health_data = [['Organ System / 器官系统', 'Health Score / 健康评分', 'Level / 等级']]

        has_organ_health = False
        for organ_key, organ_info in organ_systems.items():
            organ_name = organ_info['name']
            score_col = f'{organ_key}_score'
            level_col = f'{organ_key}_level'

            if score_col in sample_data.index and pd.notna(sample_data[score_col]):
                has_organ_health = True
                score = sample_data[score_col]
                level = sample_data[level_col] if level_col in sample_data.index else "N/A"
                organ_health_data.append([organ_name, f"{score:.1f}", str(level)])

        if has_organ_health:
            organ_health_table = Table(organ_health_data, colWidths=[3*inch, 1.5*inch, 2*inch])
            organ_health_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#E67E22')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, -1), chinese_font),
                ('FONTSIZE', (0, 0), (-1, 0), 11),
                ('FONTSIZE', (0, 1), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.bisque),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(organ_health_table)
            story.append(Spacer(1, 0.2*inch))

            # 2.2 器官健康雷达图
            try:
                organ_scores = []
                organ_labels = []
                for organ_key, organ_info in organ_systems.items():
                    score_col = f'{organ_key}_score'
                    if score_col in sample_data.index and pd.notna(sample_data[score_col]):
                        organ_scores.append(sample_data[score_col])
                        organ_labels.append(organ_info['name'].split('(')[0].strip())

                if len(organ_scores) >= 3:
                    import numpy as np
                    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))

                    # 计算角度
                    angles = np.linspace(0, 2 * np.pi, len(organ_scores), endpoint=False).tolist()
                    organ_scores_plot = organ_scores + [organ_scores[0]]
                    angles += angles[:1]

                    # 绘制雷达图
                    ax.plot(angles, organ_scores_plot, 'o-', linewidth=2, color='#E67E22', label='Health Score')
                    ax.fill(angles, organ_scores_plot, alpha=0.25, color='#E67E22')

                    # 添加参考线（健康阈值）
                    healthy_threshold = [75] * len(angles)  # 75分为良好阈值
                    ax.plot(angles, healthy_threshold, '--', linewidth=2, color='green', label='Healthy Threshold (75)')

                    ax.set_xticks(angles[:-1])
                    ax.set_xticklabels(organ_labels, fontsize=10)
                    ax.set_ylim(0, 100)
                    ax.set_title('Organ Health Scores / 器官健康评分', fontsize=14, fontweight='bold', pad=20)
                    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
                    ax.grid(True)

                    organ_radar_path = f"{output_dir}/organ_health_radar_{sample_id}.png"
                    plt.tight_layout()
                    plt.savefig(organ_radar_path, dpi=150, bbox_inches='tight')
                    plt.close()

                    story.append(Image(organ_radar_path, width=5*inch, height=5*inch))
                    story.append(Spacer(1, 0.2*inch))
            except Exception as e:
                print(f"    ⚠ 器官健康雷达图生成失败: {e}")
                plt.close()
        else:
            story.append(Paragraph("No organ health data available / 无器官健康数据", body_style))
            story.append(Spacer(1, 0.2*inch))

        # ========================================================================
        # 第3章：癌症预测
        # ========================================================================

        story.append(PageBreak())
        story.append(Paragraph("3. Cancer Prediction / 癌症预测", heading_style))

        # 3.1 癌症预测结果
        cancer_result_data = [['Metric / 指标', 'Value / 值']]

        if 'cancer_prediction' in sample_data.index and pd.notna(sample_data['cancer_prediction']):
            cancer_status = "Positive / 阳性" if sample_data['cancer_prediction'] == 1 else "Negative / 阴性"
            cancer_result_data.append(['Prediction / 预测结果', cancer_status])

        if 'cancer_probability' in sample_data.index and pd.notna(sample_data['cancer_probability']):
            cancer_result_data.append(['Probability / 概率', f"{sample_data['cancer_probability']:.2%}"])

            # 风险等级
            prob = sample_data['cancer_probability']
            if prob < 0.3:
                risk_level = "Low / 低风险"
            elif prob < 0.7:
                risk_level = "Medium / 中风险"
            else:
                risk_level = "High / 高风险"
            cancer_result_data.append(['Risk Level / 风险等级', risk_level])

        cancer_table = Table(cancer_result_data, colWidths=[3*inch, 3*inch])
        cancer_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#9B59B6')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), chinese_font),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lavender),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(cancer_table)
        story.append(Spacer(1, 0.3*inch))

        # 3.2 癌症概率可视化
        if 'cancer_probability' in sample_data.index and pd.notna(sample_data['cancer_probability']):
            try:
                fig, ax = plt.subplots(figsize=(8, 4))

                prob = sample_data['cancer_probability']
                categories = ['Negative\n阴性', 'Positive\n阳性']
                values = [1 - prob, prob]
                colors_bar = ['#2ECC71', '#E74C3C']

                bars = ax.barh(categories, values, color=colors_bar, edgecolor='black', linewidth=2)
                ax.set_xlim(0, 1)
                ax.set_xlabel('Probability / 概率', fontsize=12)
                ax.set_title('Cancer Prediction Probability / 癌症预测概率', fontsize=14, fontweight='bold')
                ax.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Threshold / 阈值')

                # 添加数值标签
                for bar, val in zip(bars, values):
                    ax.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                           f'{val:.1%}', va='center', fontsize=12, fontweight='bold')

                ax.legend()
                ax.grid(True, alpha=0.3, axis='x')

                cancer_prob_path = f"{output_dir}/cancer_probability_{sample_id}.png"
                plt.tight_layout()
                plt.savefig(cancer_prob_path, dpi=150, bbox_inches='tight')
                plt.close()

                story.append(Image(cancer_prob_path, width=5*inch, height=2.5*inch))
                story.append(Spacer(1, 0.2*inch))
            except Exception as e:
                print(f"    ⚠ 癌症概率图生成失败: {e}")
                plt.close()

        # ========================================================================
        # 第4章：五种表观遗传时钟
        # ========================================================================

        story.append(PageBreak())
        story.append(Paragraph("4. Epigenetic Clocks / 五种表观遗传时钟", heading_style))

        # 4.1 时钟结果表格
        clock_data = [['Clock / 时钟', 'Age / 年龄', 'Acceleration / 加速']]

        # 实际预测的5种时钟
        clock_columns = {
            'altumage': 'AltumAge / Altum年龄',
            'dunedinpace': 'DunedinPACE / 衰老速度',
            'grimage2': 'GrimAge2 / Grim年龄2',
            'hrsinchphenoage': 'PhenoAge / 表型年龄',
            'pchorvath2013': 'Horvath2013 / Horvath时钟',
        }

        has_clock_data = False
        for col, name in clock_columns.items():
            if col in sample_data.index and pd.notna(sample_data[col]):
                has_clock_data = True
                clock_value = sample_data[col]

                # DunedinPACE是速度指标（正常值约100），不是年龄
                if col == 'dunedinpace':
                    clock_data.append([name, f"{clock_value:.2f}", "Pace of Aging / 衰老速度"])
                else:
                    # 其他时钟是年龄，计算加速
                    if 'predicted_age' in sample_data.index and pd.notna(sample_data['predicted_age']):
                        acceleration = clock_value - sample_data['predicted_age']
                        accel_str = f"{acceleration:+.1f} years"
                    else:
                        accel_str = "N/A"
                    clock_data.append([name, f"{clock_value:.1f} years", accel_str])

        if has_clock_data:
            clock_table = Table(clock_data, colWidths=[3*inch, 1.5*inch, 2*inch])
            clock_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498DB')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, -1), chinese_font),
                ('FONTSIZE', (0, 0), (-1, 0), 11),
                ('FONTSIZE', (0, 1), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(clock_table)
            story.append(Spacer(1, 0.2*inch))

            # 4.2 时钟对比图
            try:
                import numpy as np
                clock_ages = []
                clock_labels = []
                for col, name in clock_columns.items():
                    if col in sample_data.index and pd.notna(sample_data[col]):
                        clock_ages.append(sample_data[col])
                        clock_labels.append(name.split('/')[0].strip())

                if len(clock_ages) > 0:
                    fig, ax = plt.subplots(figsize=(8, 5))

                    x_pos = np.arange(len(clock_labels))
                    bars = ax.bar(x_pos, clock_ages, color='#3498DB', edgecolor='black', linewidth=1.5)

                    # 添加参考线（实际年龄）
                    if 'predicted_age' in sample_data.index and pd.notna(sample_data['predicted_age']):
                        ax.axhline(y=sample_data['predicted_age'], color='red', linestyle='--',
                                  linewidth=2, label=f"Predicted Age / 预测年龄: {sample_data['predicted_age']:.1f}")

                    ax.set_xticks(x_pos)
                    ax.set_xticklabels(clock_labels, rotation=45, ha='right')
                    ax.set_ylabel('Age (years) / 年龄（岁）', fontsize=12)
                    ax.set_title('Epigenetic Clocks Comparison / 表观遗传时钟对比', fontsize=14, fontweight='bold')
                    ax.legend()
                    ax.grid(True, alpha=0.3, axis='y')

                    # 添加数值标签
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:.1f}', ha='center', va='bottom', fontsize=10)

                    clock_comp_path = f"{output_dir}/clock_comparison_{sample_id}.png"
                    plt.tight_layout()
                    plt.savefig(clock_comp_path, dpi=150, bbox_inches='tight')
                    plt.close()

                    story.append(Image(clock_comp_path, width=5.5*inch, height=3.5*inch))
                    story.append(Spacer(1, 0.2*inch))
            except Exception as e:
                print(f"    ⚠ 时钟对比图生成失败: {e}")
                plt.close()
        else:
            story.append(Paragraph("No epigenetic clock data available / 无表观遗传时钟数据", body_style))
            story.append(Spacer(1, 0.2*inch))

        # ========================================================================
        # 第5章：血浆蛋白质预测
        # ========================================================================

        story.append(PageBreak())
        story.append(Paragraph("5. Plasma Protein Prediction / 血浆蛋白质预测", heading_style))

        # 统计蛋白质数据
        protein_columns = [col for col in sample_data.index if col.startswith('protein_')]
        has_protein_data = len(protein_columns) > 0

        if has_protein_data:
            # 5.1 蛋白质统计
            protein_count = len(protein_columns)
            story.append(Paragraph(f"Total Proteins Predicted / 预测蛋白质总数: {protein_count}", body_style))
            story.append(Spacer(1, 0.1*inch))

            # 5.2 Top 10 蛋白质表格
            protein_values = {}
            for col in protein_columns:
                if pd.notna(sample_data[col]):
                    protein_name = col.replace('protein_', '')
                    protein_values[protein_name] = sample_data[col]

            if len(protein_values) > 0:
                # 按值排序，取前10
                sorted_proteins = sorted(protein_values.items(), key=lambda x: abs(x[1]), reverse=True)[:10]

                # 创建子标题样式
                subheading_style = ParagraphStyle(
                    'SubHeading',
                    parent=styles['Heading3'],
                    fontName=chinese_font,
                    fontSize=12,
                    textColor=colors.HexColor('#16A085'),
                    spaceAfter=6
                )

                story.append(Paragraph("Top 10 Proteins by Absolute Value / 绝对值前10的蛋白质", subheading_style))
                story.append(Spacer(1, 0.1*inch))

                protein_table_data = [['Protein / 蛋白质', 'Predicted Value / 预测值']]
                for protein, value in sorted_proteins:
                    protein_table_data.append([protein, f"{value:.3f}"])

                protein_table = Table(protein_table_data, colWidths=[3*inch, 2*inch])
                protein_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#16A085')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, -1), chinese_font),
                    ('FONTSIZE', (0, 0), (-1, 0), 11),
                    ('FONTSIZE', (0, 1), (-1, -1), 9),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.lightgreen),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                story.append(protein_table)
                story.append(Spacer(1, 0.2*inch))

                # 5.3 蛋白质分布图
                try:
                    import numpy as np
                    fig, ax = plt.subplots(figsize=(8, 5))

                    proteins = [p[0] for p in sorted_proteins]
                    values = [p[1] for p in sorted_proteins]
                    colors_bar = ['#E74C3C' if v < 0 else '#2ECC71' for v in values]

                    y_pos = np.arange(len(proteins))
                    bars = ax.barh(y_pos, values, color=colors_bar, edgecolor='black', linewidth=1)

                    ax.set_yticks(y_pos)
                    ax.set_yticklabels(proteins, fontsize=9)
                    ax.set_xlabel('Predicted Value / 预测值', fontsize=12)
                    ax.set_title('Top 10 Plasma Proteins / 前10血浆蛋白质', fontsize=14, fontweight='bold')
                    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
                    ax.grid(True, alpha=0.3, axis='x')

                    protein_dist_path = f"{output_dir}/protein_distribution_{sample_id}.png"
                    plt.tight_layout()
                    plt.savefig(protein_dist_path, dpi=150, bbox_inches='tight')
                    plt.close()

                    story.append(Image(protein_dist_path, width=5.5*inch, height=3.5*inch))
                    story.append(Spacer(1, 0.2*inch))
                except Exception as e:
                    print(f"    ⚠ 蛋白质分布图生成失败: {e}")
                    plt.close()
        else:
            story.append(Paragraph("No protein data available / 无蛋白质数据", body_style))
            story.append(Spacer(1, 0.2*inch))



        # 构建PDF
        doc.build(story)
        print(f"    ✓ PDF报告已生成: {pdf_path}")

    except Exception as e:
        print(f"    ✗ PDF报告生成失败: {e}")
        import traceback
        traceback.print_exc()


def generate_pdf_report(combined_df, output_dir):
    """
    生成详细的PDF报告
    包含预测原理图和分析结果
    """
    try:
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
        from reportlab.lib import colors
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')

        # 使用全局配置的中文字体
        chinese_font_path = configure_chinese_font()

        # 注册PDF中文字体
        chinese_font = 'Helvetica'  # 默认字体
        if chinese_font_path:
            try:
                pdfmetrics.registerFont(TTFont('ChineseFont', chinese_font_path))
                chinese_font = 'ChineseFont'
                print(f"  ✓ PDF使用中文字体: {chinese_font_path}")
            except Exception as e:
                print(f"  ⚠ PDF字体注册失败: {e}")
                # 尝试其他字体路径
                font_paths = [
                    '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
                    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
                    '/System/Library/Fonts/STHeiti Light.ttc',
                    'C:\\Windows\\Fonts\\simhei.ttf',
                ]
                for font_path in font_paths:
                    try:
                        from pathlib import Path
                        if Path(font_path).exists():
                            pdfmetrics.registerFont(TTFont('ChineseFont', font_path))
                            chinese_font = 'ChineseFont'
                            print(f"  ✓ PDF使用备用中文字体: {font_path}")
                            break
                    except:
                        continue
        else:
            print("  ⚠ 未配置中文字体，PDF中文可能显示为方框")

        # 创建PDF
        pdf_path = f"{output_dir}/comprehensive_report.pdf"
        doc = SimpleDocTemplate(pdf_path, pagesize=A4)
        story = []
        styles = getSampleStyleSheet()

        # 自定义样式（使用中文字体）
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontName=chinese_font,
            fontSize=24,
            textColor=colors.HexColor('#2C3E50'),
            spaceAfter=30,
            alignment=1  # 居中
        )

        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontName=chinese_font,
            fontSize=16,
            textColor=colors.HexColor('#34495E'),
            spaceAfter=12
        )

        # 正文样式
        body_style = ParagraphStyle(
            'CustomBody',
            parent=styles['BodyText'],
            fontName=chinese_font,
            fontSize=10,
            leading=14
        )

        # 标题
        story.append(Paragraph("DNA Methylation Analysis Report", title_style))
        story.append(Paragraph("DNA甲基化分析报告", title_style))
        story.append(Spacer(1, 0.3*inch))

        # 1. 执行摘要
        story.append(Paragraph("1. Executive Summary / 执行摘要", heading_style))
        summary_data = [
            ['Metric / 指标', 'Value / 值'],
            ['Total Samples / 样本总数', str(len(combined_df))],
            ['Average Predicted Age / 平均预测年龄', f"{combined_df['predicted_age'].mean():.1f} years"],
            ['Cancer Detection Rate / 癌症检出率', f"{(combined_df['cancer_prediction'].sum() / len(combined_df) * 100):.1f}%"],
        ]

        if 'grimage2' in combined_df.columns:
            summary_data.append(['Average GrimAge2 / 平均GrimAge2', f"{combined_df['grimage2'].mean():.1f} years"])

        summary_table = Table(summary_data, colWidths=[3.5*inch, 2*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498DB')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), chinese_font),  # 所有单元格使用中文字体
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 0.3*inch))

        # 2. 预测原理图
        story.append(PageBreak())
        story.append(Paragraph("2. Prediction Methodology / 预测方法学", heading_style))

        # 创建流程图
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.axis('off')

        # 绘制流程图
        boxes = [
            (0.5, 0.9, 'DNA Methylation Data\nDNA甲基化数据'),
            (0.5, 0.75, 'Probe ID → Genomic Location\n探针ID → 基因组位置'),
            (0.5, 0.6, 'DNA Sequence Embedding\nDNA序列嵌入'),
            (0.5, 0.45, 'CpGPT Transformer Model\nCpGPT转换器模型'),
            (0.2, 0.25, 'Age\n年龄'),
            (0.4, 0.25, 'Cancer\n癌症'),
            (0.6, 0.25, 'Clocks\n时钟'),
            (0.8, 0.25, 'Proteins\n蛋白质'),
            (0.5, 0.05, 'Integrated Risk Assessment\n综合风险评估'),
        ]

        for x, y, text in boxes:
            bbox = dict(boxstyle='round,pad=0.5', facecolor='lightblue', edgecolor='black', linewidth=2)
            ax.text(x, y, text, ha='center', va='center', fontsize=10, bbox=bbox, wrap=True)

        # 绘制箭头
        arrows = [
            (0.5, 0.87, 0.5, 0.78),
            (0.5, 0.72, 0.5, 0.63),
            (0.5, 0.57, 0.5, 0.48),
            (0.5, 0.42, 0.2, 0.28),
            (0.5, 0.42, 0.4, 0.28),
            (0.5, 0.42, 0.6, 0.28),
            (0.5, 0.42, 0.8, 0.28),
            (0.2, 0.22, 0.5, 0.08),
            (0.4, 0.22, 0.5, 0.08),
            (0.6, 0.22, 0.5, 0.08),
            (0.8, 0.22, 0.5, 0.08),
        ]

        for x1, y1, x2, y2 in arrows:
            ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                       arrowprops=dict(arrowstyle='->', lw=2, color='darkblue'))

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # 保存流程图
        flowchart_path = f"{output_dir}/methodology_flowchart.png"
        plt.tight_layout()
        plt.savefig(flowchart_path, dpi=150, bbox_inches='tight')
        plt.close()

        # 添加到PDF
        story.append(Image(flowchart_path, width=6*inch, height=4.8*inch))
        story.append(Spacer(1, 0.3*inch))

        # 3. 详细结果
        story.append(PageBreak())
        story.append(Paragraph("3. Detailed Results / 详细结果", heading_style))

        # 3.1 年龄预测分布
        if 'predicted_age' in combined_df.columns:
            try:
                fig, ax = plt.subplots(figsize=(8, 5))

                # 自动调整bins数量，避免数据范围太小的错误
                age_data = combined_df['predicted_age'].dropna()
                n_samples = len(age_data)

                if n_samples > 0:
                    data_range = age_data.max() - age_data.min()

                    # 根据样本数和数据范围自动调整bins
                    if n_samples < 10:
                        bins = min(5, n_samples)
                    elif data_range < 1:
                        bins = 5
                    elif data_range < 10:
                        bins = min(10, n_samples)
                    else:
                        bins = min(30, n_samples)

                    ax.hist(age_data, bins=bins, color='skyblue', edgecolor='black')
                    ax.set_xlabel('Predicted Age (years)', fontsize=12)
                    ax.set_ylabel('Frequency', fontsize=12)
                    ax.set_title('Distribution of Predicted Ages', fontsize=14, fontweight='bold')
                    ax.grid(True, alpha=0.3)

                    age_dist_path = f"{output_dir}/age_distribution.png"
                    plt.tight_layout()
                    plt.savefig(age_dist_path, dpi=150, bbox_inches='tight')
                    plt.close()

                    # 子标题样式
                    subheading_style = ParagraphStyle(
                        'SubHeading',
                        parent=styles['Heading3'],
                        fontName=chinese_font,
                        fontSize=12,
                        textColor=colors.HexColor('#2C3E50')
                    )
                    story.append(Paragraph("3.1 Age Distribution / 年龄分布", subheading_style))
                    story.append(Image(age_dist_path, width=5*inch, height=3.125*inch))
                    story.append(Spacer(1, 0.2*inch))
            except Exception as e:
                print(f"  ⚠ 年龄分布图生成失败: {e}")
                plt.close()

        # 3.2 癌症风险分布
        if 'cancer_probability' in combined_df.columns:
            try:
                fig, ax = plt.subplots(figsize=(8, 5))

                # 自动调整bins数量
                cancer_data = combined_df['cancer_probability'].dropna()
                n_samples = len(cancer_data)

                if n_samples > 0:
                    data_range = cancer_data.max() - cancer_data.min()

                    if n_samples < 10:
                        bins = min(5, n_samples)
                    elif data_range < 0.1:
                        bins = 5
                    elif data_range < 0.5:
                        bins = min(10, n_samples)
                    else:
                        bins = min(30, n_samples)

                    ax.hist(cancer_data, bins=bins, color='salmon', edgecolor='black')
                    ax.set_xlabel('Cancer Probability', fontsize=12)
                    ax.set_ylabel('Frequency', fontsize=12)
                    ax.set_title('Distribution of Cancer Risk', fontsize=14, fontweight='bold')
                    ax.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Threshold')
                    ax.legend()
                    ax.grid(True, alpha=0.3)

                    cancer_dist_path = f"{output_dir}/cancer_distribution.png"
                    plt.tight_layout()
                    plt.savefig(cancer_dist_path, dpi=150, bbox_inches='tight')
                    plt.close()

                    story.append(Paragraph("3.2 Cancer Risk Distribution / 癌症风险分布", subheading_style))
                    story.append(Image(cancer_dist_path, width=5*inch, height=3.125*inch))
                    story.append(Spacer(1, 0.2*inch))
            except Exception as e:
                print(f"  ⚠ 癌症风险分布图生成失败: {e}")
                plt.close()

        # 4. 风险分层总结
        story.append(PageBreak())
        story.append(Paragraph("4. Risk Stratification Summary / 风险分层总结", heading_style))

        if 'mortality_risk_category' in combined_df.columns:
            risk_counts = combined_df['mortality_risk_category'].value_counts()
            risk_data = [['Risk Category / 风险类别', 'Count / 数量', 'Percentage / 百分比']]
            for category, count in risk_counts.items():
                percentage = f"{count / len(combined_df) * 100:.1f}%"
                risk_data.append([category, str(count), percentage])

            risk_table = Table(risk_data, colWidths=[2.5*inch, 1.5*inch, 1.5*inch])
            risk_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#E74C3C')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, -1), chinese_font),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('FONTSIZE', (0, 1), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(risk_table)
            story.append(Spacer(1, 0.3*inch))

        # 5. 器官健康评分 ⭐新增
        if 'overall_health_score' in combined_df.columns:
            story.append(PageBreak())
            story.append(Paragraph("5. Organ Health Scores / 器官健康评分", heading_style))

            # 5.1 器官健康评分表格
            organ_systems = get_organ_specific_proteins()
            organ_score_data = [['Organ System / 器官系统', 'Average Score / 平均评分', 'Health Level / 健康等级']]

            for organ_key, organ_info in organ_systems.items():
                organ_name = organ_info['name']
                score_col = f'{organ_key}_score'
                level_col = f'{organ_key}_level'

                if score_col in combined_df.columns:
                    avg_score = combined_df[score_col].mean()
                    # 获取最常见的健康等级
                    if level_col in combined_df.columns:
                        most_common_level = combined_df[level_col].mode()[0] if len(combined_df[level_col].mode()) > 0 else '未知'
                    else:
                        most_common_level = '未知'

                    organ_score_data.append([
                        organ_name,
                        f"{avg_score:.1f}" if not pd.isna(avg_score) else "N/A",
                        most_common_level
                    ])

            # 添加综合评分
            if 'overall_health_score' in combined_df.columns:
                overall_avg = combined_df['overall_health_score'].mean()
                overall_level = combined_df['overall_health_level'].mode()[0] if len(combined_df['overall_health_level'].mode()) > 0 else '未知'
                organ_score_data.append([
                    '综合健康 (Overall)',
                    f"{overall_avg:.1f}" if not pd.isna(overall_avg) else "N/A",
                    overall_level
                ])

            organ_table = Table(organ_score_data, colWidths=[2.5*inch, 1.5*inch, 1.5*inch])
            organ_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#27AE60')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, -1), chinese_font),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('FONTSIZE', (0, 1), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                # 高亮综合评分行
                ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor('#D5F4E6')),
            ]))
            story.append(organ_table)
            story.append(Spacer(1, 0.3*inch))

            # 5.2 器官健康雷达图
            try:
                fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))

                # 准备数据
                organ_names = []
                organ_scores = []
                for organ_key, organ_info in organ_systems.items():
                    score_col = f'{organ_key}_score'
                    if score_col in combined_df.columns:
                        avg_score = combined_df[score_col].mean()
                        if not pd.isna(avg_score):
                            organ_names.append(organ_info['name'].split('(')[0].strip())
                            organ_scores.append(avg_score)

                if len(organ_scores) > 0:
                    # 计算角度
                    angles = np.linspace(0, 2 * np.pi, len(organ_names), endpoint=False).tolist()
                    organ_scores_plot = organ_scores + [organ_scores[0]]  # 闭合图形
                    angles += angles[:1]

                    # 绘制雷达图
                    ax.plot(angles, organ_scores_plot, 'o-', linewidth=2, color='#27AE60', label='Organ Health')
                    ax.fill(angles, organ_scores_plot, alpha=0.25, color='#27AE60')

                    # 设置刻度和标签
                    ax.set_xticks(angles[:-1])
                    ax.set_xticklabels(organ_names, fontsize=10)
                    ax.set_ylim(0, 100)
                    ax.set_yticks([20, 40, 60, 80, 100])
                    ax.set_yticklabels(['20', '40', '60', '80', '100'], fontsize=8)
                    ax.set_title('Organ Health Radar Chart\n器官健康雷达图',
                                fontsize=14, fontweight='bold', pad=20)

                    # 添加参考线
                    ax.plot(angles, [90]*len(angles), '--', linewidth=1, color='green', alpha=0.5, label='Excellent (90)')
                    ax.plot(angles, [75]*len(angles), '--', linewidth=1, color='blue', alpha=0.5, label='Good (75)')
                    ax.plot(angles, [60]*len(angles), '--', linewidth=1, color='orange', alpha=0.5, label='Fair (60)')
                    ax.plot(angles, [40]*len(angles), '--', linewidth=1, color='red', alpha=0.5, label='Poor (40)')

                    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)
                    ax.grid(True)

                    # 保存雷达图
                    radar_path = f"{output_dir}/organ_health_radar.png"
                    plt.tight_layout()
                    plt.savefig(radar_path, dpi=150, bbox_inches='tight')
                    plt.close()

                    # 添加到PDF
                    story.append(Image(radar_path, width=5*inch, height=5*inch))
                    story.append(Spacer(1, 0.3*inch))
            except Exception as e:
                print(f"  ⚠ 雷达图生成失败: {e}")

            # 5.3 评分说明
            score_explanation = """
            <b>Organ Health Score Interpretation / 器官健康评分解读：</b><br/>
            - <b>90-100 (优秀/Excellent):</b> Optimal organ function / 器官功能最佳<br/>
            - <b>75-89 (良好/Good):</b> Good organ health / 器官健康良好<br/>
            - <b>60-74 (一般/Fair):</b> Moderate concerns / 需要适度关注<br/>
            - <b>40-59 (较差/Poor):</b> Significant concerns / 需要重点关注<br/>
            - <b>0-39 (差/Very Poor):</b> Serious concerns / 需要紧急关注<br/><br/>

            <b>Note:</b> Scores are based on protein biomarkers predicted from DNA methylation data.
            Lower protein levels generally indicate better health (less inflammation, better function).<br/><br/>

            <b>注意：</b> 评分基于从DNA甲基化数据预测的蛋白质生物标志物。
            较低的蛋白质水平通常表示更好的健康状态（炎症更少，功能更好）。
            """
            story.append(Paragraph(score_explanation, styles['BodyText']))
            story.append(Spacer(1, 0.3*inch))

        # 6. 方法学说明
        story.append(PageBreak())
        story.append(Paragraph("6. Methodology Notes / 方法学说明", heading_style))

        methodology_text = """
        <b>CpGPT Model:</b> A transformer-based deep learning model trained on DNA methylation data
        to predict biological age, disease risk, and protein levels.<br/><br/>

        <b>CpGPT模型：</b> 基于Transformer的深度学习模型，在DNA甲基化数据上训练，
        用于预测生物学年龄、疾病风险和蛋白质水平。<br/><br/>

        <b>Epigenetic Clocks:</b><br/>
        - <b>GrimAge2:</b> Mortality predictor based on DNAm surrogates of plasma proteins<br/>
        - <b>DunedinPACE:</b> Pace of aging measure<br/>
        - <b>PhenoAge:</b> Phenotypic age predictor<br/>
        - <b>Horvath:</b> Pan-tissue age predictor<br/><br/>

        <b>表观遗传时钟：</b><br/>
        - <b>GrimAge2:</b> 基于血浆蛋白DNAm替代物的死亡率预测器<br/>
        - <b>DunedinPACE:</b> 衰老速度测量<br/>
        - <b>PhenoAge:</b> 表型年龄预测器<br/>
        - <b>Horvath:</b> 泛组织年龄预测器<br/><br/>

        <b>Organ Health Scores:</b> Based on organ-specific protein biomarkers predicted from
        DNA methylation. Scores reflect the functional status of major organ systems including
        heart, kidney, liver, immune system, metabolic system, and vascular system.<br/><br/>

        <b>器官健康评分：</b> 基于从DNA甲基化预测的器官特异性蛋白质生物标志物。
        评分反映主要器官系统的功能状态，包括心脏、肾脏、肝脏、免疫系统、代谢系统和血管系统。<br/><br/>

        <b>Risk Stratification:</b> Based on protein biomarkers, epigenetic clocks,
        and cancer prediction models.<br/><br/>

        <b>风险分层：</b> 基于蛋白质生物标志物、表观遗传时钟和癌症预测模型。<br/><br/>

        <b>References:</b><br/>
        - Nature 2023: Organ aging signatures in the plasma proteome<br/>
        - Lancet Digital Health 2025: Proteomic organ-specific ageing signatures
        """

        story.append(Paragraph(methodology_text, styles['BodyText']))

        # 生成PDF
        doc.build(story)
        print(f"  ✓ PDF报告已生成: {pdf_path}")

    except ImportError as e:
        print(f"  ⚠ 无法生成PDF报告: 缺少依赖库 ({e})")
        print("  提示: 安装 reportlab 和 matplotlib: pip install reportlab matplotlib")
    except Exception as e:
        print(f"  ⚠ PDF报告生成失败: {e}")


if __name__ == "__main__":
    main()

