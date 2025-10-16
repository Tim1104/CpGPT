# 935k平台零样本推理指南

## 概述

**好消息：您可以在不微调的情况下直接使用CpGPT预测年龄和疾病！**

CpGPT的设计允许它泛化到未见过的基因组位点和新的甲基化平台，包括935k。这是因为：

1. **基于基因组位置**：模型不依赖特定的探针ID，而是基于染色体位置
2. **DNA语言模型嵌入**：使用预训练的DNA模型处理任意基因组序列
3. **预训练的表型解码器**：模型已经学习了年龄、癌症等表型的甲基化模式

## 可用的预训练模型

### 年龄相关模型
- **`age_cot`**: 多组织年龄预测器（推荐用于年龄预测）
- **`relative_age`**: 相对年龄预测（0-1范围）
- **`clock_proxies`**: 5种表观遗传时钟代理

### 疾病相关模型
- **`cancer`**: 多组织癌症预测器
- **`proteins`**: 血浆蛋白预测器（可用于GrimAge3死亡率预测）

### 其他模型
- **`small`**: 通用轻量级模型（可用于样本嵌入）
- **`large`**: 通用完整模型（更高准确性）

## 零样本推理的优势和限制

### ✅ 优势
1. **无需训练数据**：不需要带标签的935k数据
2. **快速部署**：下载模型后即可使用
3. **泛化能力强**：可以处理未见过的CpG位点
4. **多任务支持**：一次数据处理，多个模型预测

### ⚠️ 限制
1. **准确性可能略低**：相比微调模型，零样本推理的准确性可能稍低
2. **平台偏差**：不同平台的技术特性可能影响预测
3. **特征覆盖**：935k的CpG位点需要与模型词汇表有足够重叠

## 数据准备要求

### 输入数据格式

您的935k数据需要是**Arrow或Feather格式**，结构如下：

```python
# 示例数据结构
import pandas as pd

df = pd.DataFrame({
    'sample_id': ['sample1', 'sample2', 'sample3'],
    'species': ['homo_sapiens', 'homo_sapiens', 'homo_sapiens'],  # 必需列
    'cg00000029': [0.85, 0.82, 0.88],  # 探针ID作为列名，值为Beta值(0-1)
    'cg00000165': [0.12, 0.15, 0.10],
    # ... 更多探针
})

# 保存为Arrow格式
df.to_feather('data/935k_samples.arrow')
```

### 必需列
- **`species`**: 必须包含，值为 `"homo_sapiens"`

### 可选列
- 真实年龄、癌症状态等（用于验证预测准确性）

## 快速开始

### 步骤1: 安装依赖

```bash
# 克隆仓库
git clone https://github.com/lcamillo/CpGPT.git
cd CpGPT

# 安装依赖
pip install poetry
poetry install
poetry shell
```

### 步骤2: 配置AWS CLI

```bash
# 安装AWS CLI
# macOS:
brew install awscli

# Linux:
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# 配置（使用任意有效的AWS凭证）
aws configure
# Access Key ID: 输入您的key
# Secret Access Key: 输入您的secret
# Region: us-east-1
# Output format: json
```

### 步骤3: 准备935k数据

确保您的数据符合上述格式要求，保存为Arrow文件。

### 步骤4: 运行零样本推理

```bash
# 使用提供的示例脚本
python examples/935k_zero_shot_inference.py
```

或者使用Python交互式环境：

```python
from cpgpt.infer.cpgpt_inferencer import CpGPTInferencer
from cpgpt.trainer.cpgpt_trainer import CpGPTTrainer
from cpgpt.data.cpgpt_datamodule import CpGPTDataModule
from cpgpt.data.components.cpgpt_datasaver import CpGPTDataSaver
from cpgpt.data.components.dna_llm_embedder import DNALLMEmbedder
from cpgpt.data.components.illumina_methylation_prober import IlluminaMethylationProber
import pandas as pd
import json

# 1. 初始化
inferencer = CpGPTInferencer(
    dependencies_dir="./dependencies",
    data_dir="./data"
)

# 2. 下载依赖和模型
inferencer.download_dependencies(species="human")
inferencer.download_model(model_name="age_cot")

# 3. 数据预处理
embedder = DNALLMEmbedder(dependencies_dir="./dependencies")
prober = IlluminaMethylationProber(
    dependencies_dir="./dependencies",
    embedder=embedder
)

datasaver = CpGPTDataSaver(
    data_paths="./data/935k_samples.arrow",
    processed_dir="./data/935k_processed"
)
datasaver.process_files(prober=prober, embedder=embedder)

# 4. 加载模型
config = inferencer.load_cpgpt_config(
    "./dependencies/model/configs/age_cot.yaml"
)
model = inferencer.load_cpgpt_model(
    config,
    model_ckpt_path="./dependencies/model/weights/age_cot.ckpt",
    strict_load=True
)

# 5. 过滤特征（使用模型词汇表）
df = pd.read_feather("./data/935k_samples.arrow")
vocab = json.load(open("./dependencies/model/vocabs/age_cot.json"))
df_filtered = df[[col for col in df.columns if col in vocab["input"]]]
df_filtered.to_feather("./data/935k_filtered.arrow")

# 重新处理过滤后的数据
datasaver_filtered = CpGPTDataSaver(
    data_paths="./data/935k_filtered.arrow",
    processed_dir="./data/935k_processed_filtered"
)
datasaver_filtered.process_files(prober=prober, embedder=embedder)

# 6. 创建数据模块
datamodule = CpGPTDataModule(
    predict_dir="./data/935k_processed_filtered",
    dependencies_dir="./dependencies",
    batch_size=1,
    num_workers=0,
    max_length=30000,
    dna_llm=config.data.dna_llm,
    dna_context_len=config.data.dna_context_len,
    sorting_strategy=config.data.sorting_strategy,
    pin_memory=False
)

# 7. 执行预测
trainer = CpGPTTrainer(precision="16-mixed")  # 重要！
predictions = trainer.predict(
    model=model,
    datamodule=datamodule,
    predict_mode="forward",
    return_keys=["pred_conditions"]
)

# 8. 查看结果
predicted_ages = predictions["pred_conditions"].flatten()
print("预测年龄:", predicted_ages)
```

## 预测结果解释

### 年龄预测 (`age_cot`)
- **输出**: 年龄（单位：年）
- **范围**: 通常0-120岁
- **解释**: 直接使用预测值

### 癌症预测 (`cancer`)
- **输出**: Logits（需要转换为概率）
- **转换**: `probability = sigmoid(logit)`
- **解释**: 
  - 概率 > 0.5: 预测为癌症
  - 概率 < 0.5: 预测为正常

```python
import torch

cancer_logits = predictions["pred_conditions"].flatten()
cancer_probs = torch.sigmoid(torch.tensor(cancer_logits)).numpy()
```

### 表观遗传时钟 (`clock_proxies`)
- **输出**: 5个时钟代理值
  - altumage
  - dunedinpace (需要乘以100)
  - grimage2
  - hrsinchphenoage
  - pchorvath2013

## 性能优化建议

### 1. 特征过滤（重要！）

使用模型的词汇表过滤特征可以显著提升性能：

```python
import json

# 加载模型词汇表
vocab = json.load(open("./dependencies/model/vocabs/age_cot.json"))

# 过滤数据
df_filtered = df[[col for col in df.columns if col in vocab["input"]]]
```

### 2. 批处理大小

根据GPU内存调整：
- **8GB GPU**: batch_size=1-2
- **16GB GPU**: batch_size=4-8
- **24GB+ GPU**: batch_size=8-16

### 3. 序列长度

935k平台CpG位点较多，建议：
- `max_length=30000` 或更大
- 如果内存不足，可以减小到20000

### 4. 使用GPU

确保使用GPU加速：
```python
import torch
print(f"GPU可用: {torch.cuda.is_available()}")
print(f"GPU名称: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
```

## 常见问题

### Q1: 预测结果准确吗？

**答**: 零样本推理的准确性取决于：
- 935k与训练数据的CpG位点重叠程度
- 样本组织类型（血液样本通常效果最好）
- 选择的模型（专用模型如`age_cot`通常优于通用模型）

建议：如果有少量带标签的935k数据，可以验证预测准确性。

### Q2: 需要多少935k特征与模型词汇表重叠？

**答**: 
- **最少**: 1000个重叠特征可以进行预测
- **推荐**: 5000+个重叠特征以获得较好性能
- **理想**: 10000+个重叠特征

检查重叠：
```python
vocab = json.load(open("./dependencies/model/vocabs/age_cot.json"))
overlap = set(df.columns) & set(vocab["input"])
print(f"重叠特征数: {len(overlap)}")
```

### Q3: 如果准确性不够怎么办？

**答**: 考虑微调模型：
1. 收集50-100个带标签的935k样本
2. 参考 `docs/935k_platform_preparation_guide.md`
3. 使用预训练模型作为起点进行微调

### Q4: 可以同时预测多个表型吗？

**答**: 可以，但需要分别加载不同的模型：

```python
# 年龄预测
model_age = inferencer.load_cpgpt_model(config_age, ckpt_age)
age_pred = trainer.predict(model_age, datamodule, ...)

# 癌症预测
model_cancer = inferencer.load_cpgpt_model(config_cancer, ckpt_cancer)
cancer_pred = trainer.predict(model_cancer, datamodule, ...)
```

### Q5: 数据预处理需要多长时间？

**答**: 取决于样本数和CpG位点数：
- **DNA嵌入生成**: 首次运行较慢（可能数小时），之后会缓存
- **探针转换**: 通常几分钟
- **数据加载**: 实时

## 下一步

1. **验证准确性**: 如果有真实标签，计算预测误差
2. **尝试不同模型**: 比较`age_cot`、`relative_age`等
3. **考虑微调**: 如果零样本性能不满意，准备微调数据

## 相关文档

- [935k平台微调准备指南](./935k_platform_preparation_guide.md)
- [快速设置教程](../tutorials/quick_setup.ipynb)
- [死亡率预测教程](../tutorials/predict_mortality.ipynb)

## 技术支持

如有问题，请参考：
- GitHub Issues: https://github.com/lcamillo/CpGPT/issues
- 论文: https://www.biorxiv.org/content/10.1101/2024.10.24.619766v1

