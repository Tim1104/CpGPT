# 935k甲基化数据格式指南

## 📋 支持的数据格式

CpGPT支持以下两种数据格式：
- **CSV格式** (`.csv`) - 推荐用于初始数据准备
- **Arrow/Feather格式** (`.arrow` / `.feather`) - 高性能二进制格式

脚本会自动检测CSV格式并转换为Arrow格式进行处理。

---

## 📊 CSV数据格式要求

### 1. 基本结构

您的CSV文件应该是一个**样本×特征**的矩阵：

```
sample_id,cg00000029,cg00000108,cg00000109,...,cg26999970
sample_001,0.234,0.567,0.891,...,0.123
sample_002,0.345,0.678,0.912,...,0.234
sample_003,0.456,0.789,0.123,...,0.345
...
```

### 2. 必需列

#### **第一列：样本ID**
- 列名可以是：`sample_id`, `GSM_ID`, `Sample_Name` 或任意名称
- 每行代表一个样本
- 样本ID应该是唯一的

#### **其他列：CpG位点甲基化值**
- 列名必须是**CpG位点ID**（如 `cg00000029`, `cg00000108` 等）
- 值为**Beta值**，范围 0-1
  - `0` = 完全未甲基化
  - `1` = 完全甲基化
  - `0.5` = 50%甲基化
- 缺失值可以用 `NA`, `NaN` 或空白表示

### 3. 可选列（元数据）

如果您有真实标签，可以添加额外的列：

```
sample_id,age,cancer_status,tissue,cg00000029,cg00000108,...
sample_001,45,0,blood,0.234,0.567,...
sample_002,62,1,blood,0.345,0.678,...
sample_003,38,0,blood,0.456,0.789,...
```

常见的元数据列：
- `age` - 实际年龄（用于验证预测准确性）
- `cancer_status` - 癌症状态（0=正常，1=癌症）
- `tissue` - 组织类型（如 `blood`, `saliva`, `tumor`）
- `sex` - 性别（`M` / `F`）
- `disease` - 疾病类型

---

## 📝 数据准备示例

### 示例1：最简CSV格式

```csv
sample_id,cg00000029,cg00000108,cg00000109,cg00000165,cg00000236
S001,0.8234,0.1567,0.9891,0.2345,0.6789
S002,0.7345,0.2678,0.8912,0.3456,0.5678
S003,0.6456,0.3789,0.7123,0.4567,0.4567
```

### 示例2：带元数据的CSV格式

```csv
sample_id,age,cancer,cg00000029,cg00000108,cg00000109,cg00000165
S001,45,0,0.8234,0.1567,0.9891,0.2345
S002,62,1,0.7345,0.2678,0.8912,0.3456
S003,38,0,0.6456,0.3789,0.7123,0.4567
```

---

## 🔧 数据预处理建议

### 1. 从原始IDAT文件处理

如果您有原始的IDAT文件，推荐使用以下R包进行预处理：

```r
# 使用minfi包
library(minfi)

# 读取IDAT文件
rgSet <- read.metharray.exp("path/to/idat/files")

# 预处理（Noob背景校正 + 归一化）
mSet <- preprocessNoob(rgSet)

# 获取Beta值
beta <- getBeta(mSet)

# 保存为CSV
write.csv(beta, "935k_samples.csv", row.names = TRUE)
```

或使用 `sesame` 包：

```r
library(sesame)

# 读取935k数据
betas <- openSesame("path/to/idat/files", platform = "MSA")

# 保存为CSV
write.csv(betas, "935k_samples.csv", row.names = TRUE)
```

### 2. 质量控制

在导出CSV之前，建议进行质量控制：

```r
# 移除低质量探针
detection_p <- detectionP(rgSet)
failed_probes <- rowSums(detection_p > 0.01) > 0.1 * ncol(detection_p)
beta_filtered <- beta[!failed_probes, ]

# 移除性染色体探针（可选）
library(IlluminaHumanMethylationEPICanno.ilm10b4.hg19)
anno <- getAnnotation(IlluminaHumanMethylationEPICanno.ilm10b4.hg19)
autosomal <- anno$chr %in% paste0("chr", 1:22)
beta_filtered <- beta_filtered[autosomal, ]
```

### 3. 转置数据（如果需要）

确保数据是**样本×特征**格式（行=样本，列=CpG位点）：

```python
import pandas as pd

# 如果数据是 特征×样本 格式，需要转置
df = pd.read_csv("935k_samples.csv", index_col=0)
df_transposed = df.T  # 转置
df_transposed.to_csv("935k_samples_transposed.csv")
```

---

## ✅ 数据验证

在运行推理之前，验证您的数据：

```python
import pandas as pd

# 读取数据
df = pd.read_csv("935k_samples.csv")

print(f"样本数: {len(df)}")
print(f"特征数: {len(df.columns) - 1}")  # 减去sample_id列
print(f"\n前5列: {list(df.columns[:5])}")
print(f"\nBeta值范围: {df.iloc[:, 1:].min().min():.3f} - {df.iloc[:, 1:].max().max():.3f}")
print(f"缺失值比例: {df.iloc[:, 1:].isna().sum().sum() / df.iloc[:, 1:].size * 100:.2f}%")

# 检查CpG列名格式
cpg_cols = [col for col in df.columns if col.startswith('cg')]
print(f"\nCpG位点数: {len(cpg_cols)}")
print(f"示例CpG位点: {cpg_cols[:5]}")
```

**预期输出：**
```
样本数: 100
特征数: 935000
前5列: ['sample_id', 'cg00000029', 'cg00000108', 'cg00000109', 'cg00000165']
Beta值范围: 0.000 - 1.000
缺失值比例: 2.34%
CpG位点数: 935000
示例CpG位点: ['cg00000029', 'cg00000108', 'cg00000109', 'cg00000165', 'cg00000236']
```

---

## 🚀 使用示例

### 1. 准备CSV数据

```bash
# 将您的数据放在 data 目录
cp /path/to/your/935k_data.csv ./data/935k_samples.csv
```

### 2. 修改脚本配置

编辑 `examples/935k_zero_shot_inference.py`：

```python
RAW_935K_DATA_PATH = "./data/935k_samples.csv"  # 您的CSV文件路径
```

### 3. 运行推理

```bash
python examples/935k_zero_shot_inference.py
```

脚本会自动：
1. ✅ 检测CSV格式
2. ✅ 转换为Arrow格式（保存为 `935k_samples.arrow`）
3. ✅ 处理数据并执行预测
4. ✅ 生成可视化图表和HTML报告

---

## ⚠️ 常见问题

### Q1: 数据列名不是CpG ID怎么办？

**A:** 您需要将列名映射到CpG ID。例如：

```python
import pandas as pd

df = pd.read_csv("935k_samples.csv")

# 如果有manifest文件，可以映射
manifest = pd.read_csv("935k_manifest.csv")
probe_to_cpg = dict(zip(manifest['Probe_ID'], manifest['CpG_ID']))

# 重命名列
df.rename(columns=probe_to_cpg, inplace=True)
df.to_csv("935k_samples_renamed.csv", index=False)
```

### Q2: Beta值不在0-1范围怎么办？

**A:** 如果您的数据是M值（log2比值），需要转换为Beta值：

```python
import numpy as np

# M值转Beta值
def m_to_beta(m_values):
    return 2**m_values / (1 + 2**m_values)

df.iloc[:, 1:] = df.iloc[:, 1:].apply(m_to_beta)
```

### Q3: 数据太大，内存不足怎么办？

**A:** 可以分批处理：

```python
# 分批读取大型CSV
chunk_size = 1000
for chunk in pd.read_csv("935k_samples.csv", chunksize=chunk_size):
    # 处理每个批次
    process_chunk(chunk)
```

### Q4: 如何处理缺失值？

**A:** CpGPT可以处理缺失值，但建议先进行填补：

```python
# 方法1: 用中位数填补
df.fillna(df.median(), inplace=True)

# 方法2: 用均值填补
df.fillna(df.mean(), inplace=True)

# 方法3: 删除缺失值过多的探针
threshold = 0.2  # 20%缺失率
df = df.loc[:, df.isna().mean() < threshold]
```

---

## 📚 参考资源

- **Illumina 935k (MSA) 官方文档**: [链接](https://www.illumina.com/products/by-type/microarray-kits/infinium-methylation-screening.html)
- **minfi R包**: [Bioconductor](https://bioconductor.org/packages/release/bioc/html/minfi.html)
- **sesame R包**: [GitHub](https://github.com/zwdzwd/sesame)
- **CpGPT论文**: [bioRxiv](https://www.biorxiv.org/content/10.1101/2024.10.24.619766v1)

---

## 💡 最佳实践

1. **质量控制优先**: 在导出CSV前进行充分的质量控制
2. **保留原始数据**: 保存一份未处理的原始数据备份
3. **记录预处理步骤**: 文档化所有预处理和归一化步骤
4. **验证数据格式**: 使用上述验证脚本检查数据
5. **小规模测试**: 先用少量样本测试流程，确认无误后再处理全部数据

---

如有任何问题，请参考 `docs/935k_zero_shot_inference_guide.md` 或提交Issue。

