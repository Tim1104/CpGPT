# 935k (GPL33022/EPICv2) 快速使用指南

## 🎉 重要说明

**935k 就是 GPL33022 (EPICv2) 平台！**

CpGPT 已经原生支持 EPICv2 平台，因此您可以直接使用现有功能来分析 935k 数据，无需任何额外配置或代码修改。

## 📋 支持的预测功能

使用 935k/EPICv2 数据，您可以直接运行以下所有预测：

### 1. 多组织器官年龄预测
- **`age_cot`**: 多组织年龄预测器（推荐）
- **`relative_age`**: 相对年龄预测（0-1范围）

### 2. 癌症预测
- **`cancer`**: 多组织癌症预测器（输出癌症概率）

### 3. 五种表观遗传时钟
- **`clock_proxies`**: 一次预测5种表观遗传时钟
  - altumage
  - dunedinpace
  - grimage2
  - hrsinchphenoage
  - pchorvath2013

### 4. 血浆蛋白质预测
- **`proteins`**: 血浆蛋白质预测因子（可用于死亡率预测）

### 5. 其他预测模型
- **`average_adultweight`**: 平均成年体重预测
- **`maximum_lifespan`**: 最大寿命预测

## 🚀 快速开始

### 步骤 1: 环境准备

```bash
# 克隆仓库
git clone https://github.com/lcamillo/CpGPT.git
cd CpGPT

# 安装依赖
poetry install

# 激活环境
poetry shell
```

### 步骤 2: 配置 AWS CLI（用于下载模型）

```bash
# 安装 AWS CLI
# macOS: brew install awscli
# Linux: sudo apt install awscli

# 配置 AWS
aws configure
# 输入您的 AWS Access Key ID
# 输入您的 AWS Secret Access Key
# Region: us-east-1
# Output format: json
```

### 步骤 3: 准备数据

您的 935k 数据应该是以下格式之一：

**CSV 格式**:
```
sample_id,cg00000029,cg00000108,cg00000109,...
sample1,0.85,0.23,0.67,...
sample2,0.91,0.19,0.72,...
```

**Arrow/Feather 格式**:
- 行：样本
- 列：探针ID（如 cg00000029）
- 值：Beta值（0-1之间）

### 步骤 4: 运行预测

使用提供的示例脚本：

```bash
# 编辑配置
# 修改 examples/935k_zero_shot_inference.py 中的数据路径
# RAW_935K_DATA_PATH = "./data/your_935k_data.csv"

# 运行预测（包含所有功能）
python examples/935k_zero_shot_inference.py
```

或者使用 Web 界面：

```bash
cd webapp
python app.py

# 在浏览器中打开 http://localhost:8000
# 上传您的 935k CSV 文件
# 选择要运行的预测类型
```

## 📊 输出结果

运行完成后，您将获得：

### 1. 预测结果文件
- `age_predictions.csv`: 年龄预测结果
- `cancer_predictions.csv`: 癌症预测结果（包含概率和分类）
- `clocks_predictions.csv`: 5种表观遗传时钟结果
- `proteins_predictions.csv`: 蛋白质预测结果
- `combined_predictions.csv`: 所有预测的汇总

### 2. 可视化图表
- 年龄分布图
- 癌症概率分布图
- 表观遗传时钟对比图
- 样本质量评估图

### 3. HTML 分析报告
- 完整的分析报告，包含所有图表和统计信息
- 数据质量评估
- 异常值检测

## 🔧 自定义预测

如果您只想运行特定的预测，可以使用以下代码模板：

```python
from cpgpt.infer.cpgpt_inferencer import CpGPTInferencer
from cpgpt.data.components.dna_llm_embedder import DNALLMEmbedder
from cpgpt.data.components.illumina_methylation_prober import IlluminaMethylationProber
from cpgpt.data.components.cpgpt_datasaver import CpGPTDataSaver
from cpgpt.data.cpgpt_datamodule import CpGPTDataModule
from cpgpt.trainer.cpgpt_trainer import CpGPTTrainer
import pandas as pd

# 1. 初始化
inferencer = CpGPTInferencer(dependencies_dir="./dependencies")

# 2. 下载依赖和模型（首次运行）
inferencer.download_dependencies(species="human")
inferencer.download_model(model_name="age_cot")  # 或其他模型

# 3. 准备数据
embedder = DNALLMEmbedder(dependencies_dir="./dependencies")
prober = IlluminaMethylationProber(dependencies_dir="./dependencies", embedder=embedder)

# 4. 转换数据格式（如果是CSV）
df = pd.read_csv("your_935k_data.csv", index_col=0)
df.reset_index().to_feather("./data/935k_data.arrow")

# 5. 处理数据
datasaver = CpGPTDataSaver(
    data_paths="./data/935k_data.arrow",
    processed_dir="./data/processed"
)
datasaver.process_files(prober=prober, embedder=embedder)

# 6. 加载模型并预测
config = inferencer.load_cpgpt_config("./dependencies/model/configs/age_cot.yaml")
model = inferencer.load_cpgpt_model(config, 
    model_ckpt_path="./dependencies/model/weights/age_cot.ckpt")

# 7. 创建数据模块
datamodule = CpGPTDataModule(
    predict_dir="./data/processed",
    dependencies_dir="./dependencies",
    batch_size=1,
    max_length=30000
)

# 8. 执行预测
trainer = CpGPTTrainer(precision="16-mixed")
predictions = trainer.predict(model=model, datamodule=datamodule, 
    predict_mode="forward", return_keys=["pred_conditions"])

# 9. 保存结果
results = pd.DataFrame({
    "predicted_age": predictions["pred_conditions"].flatten()
})
results.to_csv("age_predictions.csv", index=False)
```

## ⚙️ 可用的预训练模型列表

| 模型名称 | 功能 | 输出 | 下载命令 |
|---------|------|------|---------|
| `age_cot` | 多组织年龄预测 | 年龄（岁） | `inferencer.download_model("age_cot")` |
| `cancer` | 癌症预测 | 癌症概率 (0-1) | `inferencer.download_model("cancer")` |
| `clock_proxies` | 5种表观遗传时钟 | 5个时钟值 | `inferencer.download_model("clock_proxies")` |
| `proteins` | 血浆蛋白质预测 | 标准化蛋白质水平 | `inferencer.download_model("proteins")` |
| `relative_age` | 相对年龄 | 相对年龄 (0-1) | `inferencer.download_model("relative_age")` |
| `average_adultweight` | 平均成年体重 | log1p(体重kg) | `inferencer.download_model("average_adultweight")` |
| `maximum_lifespan` | 最大寿命 | log1p(寿命年) | `inferencer.download_model("maximum_lifespan")` |

## 💡 技术说明

### 为什么 935k 可以直接使用？

1. **平台兼容性**: 935k 芯片使用的是 GPL33022 平台ID，这就是 Illumina EPICv2 平台
2. **探针映射**: CpGPT 已经包含了 EPICv2 的完整探针到基因组位置的映射
3. **DNA 嵌入**: 模型使用基因组位置而非探针ID，因此可以泛化到任何平台
4. **预训练模型**: 所有模型都在多平台数据上训练，包括 EPICv2

### 数据处理流程

```
935k CSV 数据
    ↓
转换为 Arrow 格式
    ↓
探针ID → 基因组位置 (使用 EPICv2 manifest)
    ↓
基因组位置 → DNA 序列嵌入
    ↓
过滤匹配模型词汇表的位点
    ↓
CpGPT 模型推理
    ↓
输出预测结果
```

## 🔍 常见问题

### Q1: 我的数据需要预处理吗？
**A**: 如果是 CSV 格式，需要转换为 Arrow 格式。脚本会自动处理探针ID到基因组位置的转换。

### Q2: 需要多少样本才能运行？
**A**: 零样本推理不需要训练数据，1个样本就可以运行预测。

### Q3: 预测准确吗？
**A**: 模型在多个平台上训练，对 EPICv2/935k 数据有很好的泛化能力。但具体准确性取决于：
- 数据质量
- 样本类型（血液、组织等）
- 预处理方法

### Q4: 可以同时运行多个预测吗？
**A**: 可以！使用 `examples/935k_zero_shot_inference.py` 脚本可以一次运行所有预测。

### Q5: 内存不足怎么办？
**A**:
- 减小 `MAX_INPUT_LENGTH` 参数（如从 30000 降到 15000）
- 设置 `USE_CPU = True` 使用 CPU 而非 GPU
- 减小 `batch_size` 参数

### Q6: 如何解读癌症预测结果？
**A**:
- `cancer_logit`: 原始输出值
- `cancer_probability`: 经过 sigmoid 转换的概率（0-1）
- `cancer_prediction`: 二分类结果（概率 > 0.5 为癌症）

### Q7: 表观遗传时钟的单位是什么？
**A**:
- `altumage`, `grimage2`, `hrsinchphenoage`, `pchorvath2013`: 年龄（岁）
- `dunedinpace`: 衰老速度（已乘以100，正常值约100）

## 📚 相关资源

- **完整示例脚本**: `examples/935k_zero_shot_inference.py`
- **Web 应用**: `webapp/app.py`
- **数据格式指南**: `docs/935k_data_format_guide.md`
- **原始论文**: [CpGPT bioRxiv](https://www.biorxiv.org/content/10.1101/2024.10.24.619766v1)

## 🆘 获取帮助

如果遇到问题：
1. 查看日志文件：`logs/cpgptinferencer.log`
2. 检查数据格式是否正确
3. 确认所有依赖已下载
4. 联系：lucas_camillo@alumni.brown.edu

## 📝 引用

如果使用 CpGPT 进行研究，请引用：

```bibtex
@article{camillo2024cpgpt,
  title={CpGPT: A Foundation Model for DNA Methylation},
  author={de Lima Camillo, Lucas Paulo et al.},
  journal={bioRxiv},
  year={2024},
  doi={10.1101/2024.10.24.619766}
}
```


