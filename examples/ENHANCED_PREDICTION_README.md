# 935k/EPICv2 增强版预测脚本使用指南

## 📋 概述

`935k_enhanced_prediction.py` 是一个功能增强的DNA甲基化数据预测脚本，在基础预测功能之上新增了：

### 🆕 新增功能

1. **CpGPTGrimAge3 死亡率预测** - 基于蛋白质和表观遗传时钟的死亡率风险评估
2. **年龄加速指标** - 计算表观遗传年龄与实际年龄的差异
3. **CVD/癌症风险分层** - 基于蛋白质生物标志物的心血管疾病和癌症风险评估
4. **器官健康评分** ⭐ - 评估6大器官系统健康状态（心脏、肾脏、肝脏、免疫、代谢、血管）
5. **详细PDF报告** - 包含可视化图表和方法学说明的综合报告

### 📊 预测内容

- ✅ 生物学年龄 (Biological Age)
- ✅ 癌症风险 (Cancer Risk)
- ✅ 表观遗传时钟 (Epigenetic Clocks)
  - GrimAge2 (死亡率预测)
  - DunedinPACE (衰老速度)
  - PhenoAge (表型年龄)
  - Horvath (泛组织年龄)
  - AltumAge
- ✅ 蛋白质水平 (Protein Levels) - 322种蛋白质
- ✅ 年龄加速 (Age Acceleration)
- ✅ 死亡率风险 (Mortality Risk)
- ✅ 疾病风险分层 (Disease Risk Stratification)
- ✅ 器官健康评分 (Organ Health Scores) - 6大器官系统

## 🚀 快速开始

### 1. 安装依赖

```bash
# 基础依赖（已在 CpGPT 中）
pip install torch lightning pandas numpy pyarrow

# PDF报告生成依赖（可选）
pip install reportlab matplotlib
```

### 2. 准备数据

将您的 935k/EPICv2 甲基化数据放在 `examples/data/` 目录下：

```
examples/
├── data/
│   └── Sample251212.arrow  # 您的数据文件
└── 935k_enhanced_prediction.py
```

支持的数据格式：
- `.arrow` (推荐)
- `.csv` (自动转换)

### 3. 运行脚本

```bash
cd examples
python 935k_enhanced_prediction.py
```

## 📁 输出文件

脚本会在 `examples/results/935k_enhanced_predictions/` 目录下生成以下文件：

```
results/935k_enhanced_predictions/
├── combined_predictions.csv          # 所有预测结果合并
├── age_predictions.csv                # 年龄预测
├── cancer_predictions.csv             # 癌症预测
├── clocks_predictions.csv             # 表观遗传时钟
├── proteins_predictions.csv           # 蛋白质预测
├── age_acceleration.csv               # 年龄加速指标 ⭐新增
├── mortality_predictions.csv          # 死亡率预测 ⭐新增
├── risk_stratification.csv            # 风险分层 ⭐新增
├── organ_health_scores.csv            # 器官健康评分 ⭐新增
├── comprehensive_report.pdf           # PDF综合报告 ⭐新增
├── methodology_flowchart.png          # 方法学流程图
├── age_distribution.png               # 年龄分布图
├── cancer_distribution.png            # 癌症风险分布图
└── organ_health_radar.png             # 器官健康雷达图 ⭐新增
```

## ⚙️ 配置选项

在脚本顶部可以修改以下配置：

```python
# 数据路径
RAW_DATA_PATH = SCRIPT_DIR / "data" / "Sample251212.arrow"

# 预测开关
PREDICT_AGE = True
PREDICT_CANCER = True
PREDICT_CLOCKS = True
PREDICT_PROTEINS = True
PREDICT_MORTALITY = True  # 新增：死亡率预测

# 年龄加速计算（如果数据中有实际年龄）
CHRONOLOGICAL_AGE_COLUMN = None  # 设置为实际年龄列名，如 "age"

# 其他配置
RANDOM_SEED = 42
MAX_INPUT_LENGTH = 30000
USE_CPU = False  # 设置为 True 使用 CPU
```

## 📊 结果解读

### 1. 年龄加速 (Age Acceleration)

- **正值**: 表观遗传年龄 > 实际年龄，衰老加速
- **负值**: 表观遗传年龄 < 实际年龄，衰老减缓
- **阈值**: 
  - < -5 年: 显著年轻
  - -5 到 0 年: 轻微年轻
  - 0 到 5 年: 轻微衰老
  - > 5 年: 显著衰老

### 2. 死亡率风险 (Mortality Risk)

基于 GrimAge2 的风险分层：
- **低风险**: GrimAge2 比预测年龄小 > 5 年
- **中低风险**: GrimAge2 比预测年龄小 0-5 年
- **中高风险**: GrimAge2 比预测年龄大 0-5 年
- **高风险**: GrimAge2 比预测年龄大 > 5 年

### 3. 疾病风险分层

#### 癌症风险
- **低风险**: 概率 < 0.2
- **中低风险**: 概率 0.2-0.4
- **中高风险**: 概率 0.4-0.6
- **高风险**: 概率 > 0.6

#### CVD风险
基于心血管相关蛋白质评分（ADM, CRP, IL6, TNF-α, ICAM1, VCAM1等）

### 4. 器官健康评分 ⭐新增

基于器官特异性蛋白质生物标志物，评估6大器官系统的健康状态：

#### 评分范围：0-100
- **90-100 (优秀)**: 器官功能最佳
- **75-89 (良好)**: 器官健康良好
- **60-74 (一般)**: 需要适度关注
- **40-59 (较差)**: 需要重点关注
- **0-39 (差)**: 需要紧急关注

#### 6大器官系统

| 器官系统 | 关键蛋白质标志物 | 评估内容 |
|---------|----------------|---------|
| **心脏** | ADM, CRP, IL6, TNF-α, ICAM1, VCAM1, Fibrinogen, vWF, PAI1, MMP1/9 | 心血管系统健康、内皮功能、炎症和凝血状态 |
| **肾脏** | Cystatin C, B2M, CRP, IL6, TNF-α, VEGF, PAI1 | 肾功能和炎症状态 |
| **肝脏** | CRP, Fibrinogen, PAI1, MMP1/9, IL6, TNF-α, GDF15 | 肝脏合成功能和纤维化风险 |
| **免疫系统** | IL6, TNF-α, CRP, B2M, ICAM1, VCAM1, E-selectin, P-selectin | 免疫系统激活和炎症状态 |
| **代谢系统** | Leptin, GDF15, PAI1, CRP, IL6, TNF-α, ADM | 代谢健康和能量平衡 |
| **血管系统** | ICAM1, VCAM1, E-selectin, P-selectin, vWF, MMP1/9, VEGF, ADM | 血管内皮功能和血管健康 |

#### 科学依据

基于最新研究成果：
- **Nature 2023**: "Organ aging signatures in the plasma proteome track health and disease"
- **Lancet Digital Health 2025**: "Proteomic organ-specific ageing signatures and 20-year risk of age-related diseases"

这些研究证明：**血液中的蛋白质生物标志物可以反映器官特异性衰老**

## 🔬 方法学

### CpGPT 模型
基于 Transformer 的深度学习模型，在大规模 DNA 甲基化数据上训练

### 关键蛋白质

**CVD相关**: ADM, CRP, IL6, TNF-α, ICAM1, VCAM1, Fibrinogen, vWF, D-dimer, PAI1, MMP1, MMP9

**癌症相关**: GDF15, VEGF, IL6, TNF-α, MMP1, MMP9, Leptin, CRP, B2M

## 📝 注意事项

1. **首次运行**: 会自动下载模型和依赖文件（约 2-3 GB）
2. **内存需求**: 建议至少 16GB RAM
3. **GPU加速**: 推荐使用 GPU，可显著加快预测速度
4. **实际年龄**: 如果数据中包含实际年龄，设置 `CHRONOLOGICAL_AGE_COLUMN` 可获得更准确的年龄加速指标

## 🆚 与简单版本的区别

| 功能 | 简单版 | 增强版 |
|------|--------|--------|
| 年龄预测 | ✅ | ✅ |
| 癌症预测 | ✅ | ✅ |
| 表观遗传时钟 | ✅ | ✅ |
| 蛋白质预测 | ✅ | ✅ |
| 年龄加速指标 | ❌ | ✅ |
| 死亡率预测 | ❌ | ✅ |
| 疾病风险分层 | ❌ | ✅ |
| 器官健康评分 | ❌ | ✅ ⭐ |
| PDF报告 | ❌ | ✅ |
| 可视化图表 | ❌ | ✅ |
| 器官健康雷达图 | ❌ | ✅ ⭐ |

## 🐛 故障排除

### 问题1: 内存不足
```python
# 减少批处理大小
MAX_INPUT_LENGTH = 20000  # 默认 30000
```

### 问题2: PDF生成失败
```bash
# 安装缺失的依赖
pip install reportlab matplotlib

# 如果中文显示有问题，脚本会自动降级到英文
```

### 问题3: 模型下载失败
```bash
# 手动下载模型到 examples/dependencies/model/
# 或检查网络连接
```

## 📧 支持

如有问题，请参考主项目文档或提交 Issue。

