# 935k 快速开始示例

## 🎯 目标

使用 CpGPT 对 935k 甲基化数据进行：
- ✅ 年龄预测
- ✅ 癌症预测  
- ✅ 五种表观遗传时钟
- ✅ 血浆蛋白质预测

## 📋 前提条件

1. 已安装 CpGPT
2. 已配置 AWS CLI
3. 有 935k CSV 数据文件

## 🚀 方法一：使用简化脚本（推荐）

### 步骤 1: 准备数据

确保您的数据是 CSV 格式：

```csv
sample_id,cg00000029,cg00000108,cg00000109,...
sample1,0.85,0.23,0.67,...
sample2,0.91,0.19,0.72,...
```

### 步骤 2: 修改配置

编辑 `examples/935k_simple_prediction.py`：

```python
# 第 30 行：修改数据路径
RAW_DATA_PATH = "./data/你的数据.csv"

# 第 33 行：修改输出目录（可选）
RESULTS_DIR = "./results/935k_predictions"

# 第 39-42 行：选择要运行的预测
PREDICT_AGE = True          # 年龄预测
PREDICT_CANCER = True       # 癌症预测
PREDICT_CLOCKS = True       # 表观遗传时钟
PREDICT_PROTEINS = True     # 蛋白质预测
```

### 步骤 3: 运行预测

```bash
python examples/935k_simple_prediction.py
```

### 步骤 4: 查看结果

结果保存在 `results/935k_predictions/` 目录：

```
results/935k_predictions/
├── age_predictions.csv          # 年龄预测
├── cancer_predictions.csv       # 癌症预测
├── clocks_predictions.csv       # 5种时钟
├── proteins_predictions.csv     # 蛋白质预测
└── combined_predictions.csv     # 所有结果汇总
```

## 🌐 方法二：使用 Web 界面

### 步骤 1: 启动 Web 服务

```bash
cd webapp
python app.py
```

### 步骤 2: 打开浏览器

访问 `http://localhost:8000`

### 步骤 3: 上传数据

1. 点击"选择文件"
2. 选择您的 935k CSV 文件
3. 点击"上传"

### 步骤 4: 选择预测

勾选您需要的预测类型：
- [ ] 年龄预测
- [ ] 癌症预测
- [ ] 表观遗传时钟
- [ ] 蛋白质预测

### 步骤 5: 开始分析

点击"开始分析"按钮，等待完成。

### 步骤 6: 下载结果

- 查看可视化图表
- 下载 HTML 报告
- 下载 CSV 结果文件

## 💻 方法三：使用 Python 代码

### 最小化示例

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
embedder = DNALLMEmbedder(dependencies_dir="./dependencies")
prober = IlluminaMethylationProber(dependencies_dir="./dependencies", embedder=embedder)

# 2. 下载依赖（首次运行）
inferencer.download_dependencies(species="human")
inferencer.download_model(model_name="age_cot")

# 3. 转换数据格式
df = pd.read_csv("your_data.csv", index_col=0)
df.reset_index().to_feather("./data/935k_data.arrow")

# 4. 处理数据
datasaver = CpGPTDataSaver(
    data_paths="./data/935k_data.arrow",
    processed_dir="./data/processed"
)
datasaver.process_files(prober=prober, embedder=embedder)

# 5. 加载模型
config = inferencer.load_cpgpt_config("./dependencies/model/configs/age_cot.yaml")
model = inferencer.load_cpgpt_model(
    config,
    model_ckpt_path="./dependencies/model/weights/age_cot.ckpt"
)

# 6. 创建数据模块
datamodule = CpGPTDataModule(
    predict_dir="./data/processed",
    dependencies_dir="./dependencies",
    batch_size=1,
    max_length=30000
)

# 7. 执行预测
trainer = CpGPTTrainer(precision="16-mixed")
predictions = trainer.predict(
    model=model,
    datamodule=datamodule,
    predict_mode="forward",
    return_keys=["pred_conditions"]
)

# 8. 保存结果
results = pd.DataFrame({
    "predicted_age": predictions["pred_conditions"].flatten()
})
results.to_csv("age_predictions.csv", index=False)
print("预测完成！")
```

## 📊 结果解读

### 年龄预测结果

```csv
sample_id,predicted_age
sample1,45.2
sample2,38.7
```

- `predicted_age`: 预测的生物学年龄（岁）

### 癌症预测结果

```csv
sample_id,cancer_logit,cancer_probability,cancer_prediction
sample1,-2.3,0.09,0
sample2,1.8,0.86,1
```

- `cancer_logit`: 原始输出值
- `cancer_probability`: 癌症概率（0-1）
- `cancer_prediction`: 二分类结果（0=正常，1=癌症）

**解读**：
- 概率 > 0.5：预测为癌症
- 概率 < 0.5：预测为正常
- 概率越接近 0 或 1，预测越确定

### 表观遗传时钟结果

```csv
sample_id,altumage,dunedinpace,grimage2,hrsinchphenoage,pchorvath2013
sample1,45.1,98.5,47.2,44.8,46.3
sample2,39.2,102.3,40.1,38.5,39.8
```

**各时钟含义**：
- `altumage`: 年龄预测（岁）
- `dunedinpace`: 衰老速度（正常值约100）
  - < 100: 衰老较慢
  - = 100: 正常衰老
  - > 100: 衰老加速
- `grimage2`: GrimAge2 死亡率预测时钟（岁）
- `hrsinchphenoage`: PhenoAge 表型年龄（岁）
- `pchorvath2013`: Horvath 2013 经典时钟（岁）

## ⚙️ 常见配置

### 内存不足

```python
# 在脚本中修改
MAX_INPUT_LENGTH = 15000  # 从 30000 降低
USE_CPU = True            # 使用 CPU 而非 GPU
```

### 只运行部分预测

```python
PREDICT_AGE = True        # 只运行年龄预测
PREDICT_CANCER = False    # 跳过癌症预测
PREDICT_CLOCKS = False    # 跳过时钟预测
PREDICT_PROTEINS = False  # 跳过蛋白质预测
```

### 批量处理多个文件

```python
import glob

# 获取所有 CSV 文件
csv_files = glob.glob("./data/*.csv")

for csv_file in csv_files:
    print(f"处理: {csv_file}")
    # 运行预测...
```

## 🆘 故障排除

### 问题 1: 找不到模型文件

**解决方案**：
```bash
# 重新下载模型
python -c "
from cpgpt.infer.cpgpt_inferencer import CpGPTInferencer
inf = CpGPTInferencer()
inf.download_model('age_cot', overwrite=True)
"
```

### 问题 2: 内存溢出

**解决方案**：
- 降低 `MAX_INPUT_LENGTH`
- 设置 `USE_CPU = True`
- 减小 `batch_size`

### 问题 3: 数据格式错误

**解决方案**：
```bash
# 验证数据格式
python examples/validate_935k_data.py --input your_data.csv
```

## 📚 更多资源

- 📖 [完整文档](../docs/935k_README_CN.md)
- 🔬 [技术指南](../docs/935k_EPICv2_QUICKSTART.md)
- 💻 [完整脚本](935k_zero_shot_inference.py)
- 🌐 [Web 应用](../webapp/README.md)

## ✅ 检查清单

开始之前，确保：

- [ ] 已安装 CpGPT (`poetry install`)
- [ ] 已配置 AWS CLI
- [ ] 数据是 CSV 格式
- [ ] 第一列是样本ID
- [ ] 其他列是探针ID（如 cg00000029）
- [ ] 值是 Beta 值（0-1之间）
- [ ] 已修改脚本中的数据路径

**准备好了？开始预测吧！** 🚀

