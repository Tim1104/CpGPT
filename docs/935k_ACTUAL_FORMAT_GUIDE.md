# 935k 实际数据格式说明和转换指南

## 📋 实际数据格式

根据实际的935k芯片输出，数据格式如下：

### 原始CSV格式（厂商输出）

```csv
TargetID,000536.AVG_Beta,000537.AVG_Beta
cg00000029_TC21,0.4630385,0.4062999
cg00000109_TC21,0.8233373,0.8394986
cg00000155_BC21,0.8882958,0.8735513
cg00000158_BC21,0.9374912,0.9106941
```

**特点**：
- **第一列**：`TargetID` - 探针ID（带后缀，如 `_TC21`, `_BC21`）
- **其他列**：样本数据，列名格式为 `样本ID.AVG_Beta`
- **数据方向**：行=探针，列=样本（**转置的**）
- **探针后缀**：`_TC21`, `_BC21`, `_BC11` 等

## 🔍 为什么需要转换？

### 1. 探针ID后缀问题

**你的数据**：`cg00000029_TC21`  
**CpGPT处理**：自动去除后缀变成 `cg00000029`

✅ **好消息**：CpGPT 已经在代码中自动处理这个问题！

在 `illumina_methylation_prober.py` 第332行：
```python
# Collapase probe IDs that map to the same locus in EPICv2
manifest_df["Probe_ID"] = manifest_df["Probe_ID"].str.split("_").str[0]
```

**但是**，你的数据需要先转换成正确的格式，CpGPT才能处理。

### 2. 数据方向问题

**你的数据**：
- 行 = 探针（如 cg00000029_TC21）
- 列 = 样本（如 000536.AVG_Beta）

**CpGPT需要**：
- 行 = 样本（如 000536）
- 列 = 探针（如 cg00000029）

### 3. 列名后缀问题

**你的列名**：`000536.AVG_Beta`  
**CpGPT需要**：`000536`

## 🚀 转换步骤（2步）

### 步骤 1: 运行转换脚本

```bash
# 转换你的数据
python examples/convert_935k_format.py "Sample Methylation Profile.csv"

# 或者指定输出文件名
python examples/convert_935k_format.py "Sample Methylation Profile.csv" -o my_data.arrow
```

**转换脚本会自动**：
1. ✅ 去除探针ID后缀（`cg00000029_TC21` → `cg00000029`）
2. ✅ 去除样本ID后缀（`000536.AVG_Beta` → `000536`）
3. ✅ 转置数据（行列互换）
4. ✅ 处理重复探针（取平均值）
5. ✅ 保存为Arrow格式

### 步骤 2: 运行预测

```bash
# 使用转换后的数据运行预测
python examples/935k_simple_prediction.py
```

记得在脚本中修改数据路径：
```python
RAW_DATA_PATH = "./data/Sample Methylation Profile.arrow"  # 转换后的文件
```

## 📊 转换前后对比

### 转换前（你的原始数据）

```csv
TargetID,000536.AVG_Beta,000537.AVG_Beta
cg00000029_TC21,0.4630385,0.4062999
cg00000109_TC21,0.8233373,0.8394986
```

- 形状：(探针数, 样本数+1)
- 第一列：探针ID（带后缀）
- 其他列：样本（列名带后缀）

### 转换后（CpGPT格式）

```csv
sample_id,cg00000029,cg00000109
000536,0.4630385,0.8233373
000537,0.4062999,0.8394986
```

- 形状：(样本数, 探针数+1)
- 第一列：样本ID（无后缀）
- 其他列：探针（无后缀）

## 🔧 技术细节

### EPICv2 探针设计

EPICv2 芯片对某些CpG位点使用了多个探针，后缀格式多样：
- `cg00000029_TC21` - Type I 探针，红色通道
- `cg00000029_BC21` - Type I 探针，绿色通道
- `cg00000029_BC11` - Type II 探针
- 还有其他各种后缀格式（`_TC11`, `_BC31` 等）

**转换工具的处理方式**：
1. **保留下划线前的所有内容**，去除下划线及之后的所有字符
   - `cg00000029_TC21` → `cg00000029`
   - `cg00000029_BC21` → `cg00000029`
   - `cg00000029_BC11` → `cg00000029`
   - `cg12345678_ANY_SUFFIX` → `cg12345678`
2. 如果有重复，取平均值
3. 使用基因组位置而非探针ID进行后续分析

### 空值处理

**你提到的空值问题**：

✅ **CpGPT 可以处理空值**！

在数据处理过程中：
1. 空值会被保留为 NaN
2. 模型会自动忽略这些位置
3. 不会影响预测结果

**建议**：
- 如果某个探针在所有样本中都是空值，可以删除
- 如果某个样本的大部分探针都是空值，需要检查数据质量

## 📝 完整使用流程

### 方法1：使用转换脚本（推荐）

```bash
# 1. 转换数据格式
python examples/convert_935k_format.py "Sample Methylation Profile.csv"

# 2. 运行预测
python examples/935k_simple_prediction.py
```

### 方法2：手动转换（Python代码）

```python
import pandas as pd

# 1. 读取数据
df = pd.read_csv("Sample Methylation Profile.csv")

# 2. 清理探针ID（去除后缀）
probe_col = df.columns[0]
df[probe_col] = df[probe_col].str.split('_').str[0]

# 3. 处理重复探针（取平均值）
df = df.groupby(probe_col, as_index=False).mean()

# 4. 清理样本ID（去除 .AVG_Beta）
new_columns = [probe_col] + [col.split('.')[0] for col in df.columns[1:]]
df.columns = new_columns

# 5. 转置数据
df = df.set_index(probe_col).T.reset_index()
df = df.rename(columns={'index': 'sample_id'})

# 6. 保存为Arrow格式
df.to_feather("converted_data.arrow")

print("转换完成！")
```

## ❓ 常见问题

### Q1: 为什么探针ID有后缀？
**A**: EPICv2 芯片设计使用了多个探针测量同一个CpG位点，以提高准确性。后缀表示探针类型和颜色通道。

### Q2: 重复的探针如何处理？
**A**: 转换脚本会自动对重复探针取平均值。这是标准做法，可以提高数据质量。

### Q3: 空值会影响预测吗？
**A**: 不会。CpGPT 会自动忽略空值位置，使用其他可用的探针进行预测。

### Q4: 转换后数据会丢失吗？
**A**: 不会。转换只是改变数据格式，所有的Beta值都会保留。

### Q5: 可以直接用原始CSV吗？
**A**: 不可以。必须先转换格式，因为：
- 数据方向不对（需要转置）
- 列名有后缀（需要清理）
- 需要转换为Arrow格式

## ✅ 验证转换结果

转换后，检查数据：

```python
import pandas as pd

# 读取转换后的数据
df = pd.read_feather("Sample Methylation Profile.arrow")

print("数据形状:", df.shape)
print("样本数:", len(df))
print("探针数:", len(df.columns) - 1)
print("\n前几行:")
print(df.head())
print("\n前几列:")
print(df.iloc[:, :5])
```

**期望输出**：
- 第一列是 `sample_id`
- 其他列是探针ID（如 `cg00000029`，无后缀）
- 值在 0-1 之间
- 行数 = 样本数

## 🎯 总结

1. ✅ **CpGPT 支持 EPICv2/935k**，但需要正确的数据格式
2. ✅ **探针后缀会自动处理**，但需要先转换数据格式
3. ✅ **空值可以处理**，不影响预测
4. ✅ **使用转换脚本**，一键完成所有转换

**下一步**：
```bash
# 转换数据
python examples/convert_935k_format.py "你的文件.csv"

# 运行预测
python examples/935k_simple_prediction.py
```

就这么简单！🚀

