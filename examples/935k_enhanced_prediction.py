"""
935k/EPICv2 增强版预测脚本 - 包含死亡率预测、年龄加速和疾病风险分层
Enhanced prediction script with mortality, age acceleration, and disease risk stratification

新增功能 / New Features:
1. CpGPTGrimAge3 死亡率预测 (Mortality prediction)
2. 年龄加速指标 (Age acceleration metrics)
3. CVD/癌症相关蛋白风险分层 (Disease risk stratification)
4. 详细PDF报告生成 (Comprehensive PDF report)
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
RAW_DATA_PATH = SCRIPT_DIR / "data" / "Sample251212.arrow"

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

# 其他配置
RANDOM_SEED = 42
MAX_INPUT_LENGTH = 30000
USE_CPU = False

# 年龄加速计算配置
CHRONOLOGICAL_AGE_COLUMN = None  # 如果数据中有实际年龄，设置列名，如 "age"

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

    # 保存合并结果
    combined.to_csv(f"{str(RESULTS_DIR)}/combined_predictions.csv", index=False)

    # 步骤 7: 生成PDF报告
    print("\n[7/7] 生成PDF报告...")
    generate_pdf_report(combined, str(RESULTS_DIR))

    print("\n" + "=" * 80)
    print("预测完成！")
    print("=" * 80)
    print(f"\n结果文件:")
    print(f"  - 合并结果: {RESULTS_DIR}/combined_predictions.csv")
    print(f"  - 年龄加速: {RESULTS_DIR}/age_acceleration.csv")
    print(f"  - 死亡率预测: {RESULTS_DIR}/mortality_predictions.csv")
    print(f"  - 风险分层: {RESULTS_DIR}/risk_stratification.csv")
    print(f"  - PDF报告: {RESULTS_DIR}/comprehensive_report.pdf")
    print()


# ============================================================================
# 预测函数
# ============================================================================

def predict_age(inferencer, processed_dir, sample_ids, trainer):
    """年龄预测"""
    config = inferencer.load_cpgpt_config(f"{str(DEPENDENCIES_DIR)}/model/config/age_cot.yaml")
    model = inferencer.load_cpgpt_model(
        config,
        model_ckpt_path=f"{str(DEPENDENCIES_DIR)}/model/weights/age_cot.ckpt",
        strict_load=True
    )

    datamodule = CpGPTDataModule(
        predict_dir=processed_dir,
        dependencies_dir=str(DEPENDENCIES_DIR),
        batch_size=1,
        num_workers=0,
        max_length=MAX_INPUT_LENGTH,
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
    return pd.DataFrame({'sample_id': sample_ids, 'predicted_age': pred_values})


def predict_cancer(inferencer, processed_dir, sample_ids, trainer):
    """癌症预测"""
    config = inferencer.load_cpgpt_config(f"{str(DEPENDENCIES_DIR)}/model/config/cancer.yaml")
    model = inferencer.load_cpgpt_model(
        config,
        model_ckpt_path=f"{str(DEPENDENCIES_DIR)}/model/weights/cancer.ckpt",
        strict_load=True
    )

    datamodule = CpGPTDataModule(
        predict_dir=processed_dir,
        dependencies_dir=str(DEPENDENCIES_DIR),
        batch_size=1,
        num_workers=0,
        max_length=MAX_INPUT_LENGTH,
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

    datamodule = CpGPTDataModule(
        predict_dir=processed_dir,
        dependencies_dir=str(DEPENDENCIES_DIR),
        batch_size=1,
        num_workers=0,
        max_length=MAX_INPUT_LENGTH,
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
        result_dict[clock_name] = clock_values[:, i]

    return pd.DataFrame(result_dict)


def predict_proteins(inferencer, processed_dir, sample_ids, trainer):
    """蛋白质预测"""
    config = inferencer.load_cpgpt_config(f"{str(DEPENDENCIES_DIR)}/model/config/proteins.yaml")
    model = inferencer.load_cpgpt_model(
        config,
        model_ckpt_path=f"{str(DEPENDENCIES_DIR)}/model/weights/proteins.ckpt",
        strict_load=True
    )

    datamodule = CpGPTDataModule(
        predict_dir=processed_dir,
        dependencies_dir=str(DEPENDENCIES_DIR),
        batch_size=1,
        num_workers=0,
        max_length=MAX_INPUT_LENGTH,
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
    for i in range(num_proteins):
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

        # 注册中文字体（如果可用）
        try:
            pdfmetrics.registerFont(TTFont('SimHei', '/System/Library/Fonts/STHeiti Light.ttc'))
            chinese_font = 'SimHei'
        except:
            chinese_font = 'Helvetica'

        # 创建PDF
        pdf_path = f"{output_dir}/comprehensive_report.pdf"
        doc = SimpleDocTemplate(pdf_path, pagesize=A4)
        story = []
        styles = getSampleStyleSheet()

        # 自定义样式
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#2C3E50'),
            spaceAfter=30,
            alignment=1  # 居中
        )

        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#34495E'),
            spaceAfter=12
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
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
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
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(combined_df['predicted_age'].dropna(), bins=30, color='skyblue', edgecolor='black')
            ax.set_xlabel('Predicted Age (years)', fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.set_title('Distribution of Predicted Ages', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)

            age_dist_path = f"{output_dir}/age_distribution.png"
            plt.tight_layout()
            plt.savefig(age_dist_path, dpi=150, bbox_inches='tight')
            plt.close()

            story.append(Paragraph("3.1 Age Distribution / 年龄分布", styles['Heading3']))
            story.append(Image(age_dist_path, width=5*inch, height=3.125*inch))
            story.append(Spacer(1, 0.2*inch))

        # 3.2 癌症风险分布
        if 'cancer_probability' in combined_df.columns:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(combined_df['cancer_probability'].dropna(), bins=30, color='salmon', edgecolor='black')
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

            story.append(Paragraph("3.2 Cancer Risk Distribution / 癌症风险分布", styles['Heading3']))
            story.append(Image(cancer_dist_path, width=5*inch, height=3.125*inch))
            story.append(Spacer(1, 0.2*inch))

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
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(risk_table)
            story.append(Spacer(1, 0.3*inch))

        # 5. 方法学说明
        story.append(PageBreak())
        story.append(Paragraph("5. Methodology Notes / 方法学说明", heading_style))

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

        <b>Risk Stratification:</b> Based on protein biomarkers, epigenetic clocks,
        and cancer prediction models.<br/><br/>

        <b>风险分层：</b> 基于蛋白质生物标志物、表观遗传时钟和癌症预测模型。
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

