# 使用元数据进行预测的完整指南

## 📋 概述

这个方法允许你将样本的实际信息（年龄、癌症状态等）放在一个 CSV 文件中，脚本会自动：
1. 读取元数据
2. 计算标准化参数
3. 生成对比报告（预测 vs 实际）

## 🚀 快速开始

### 步骤 1：创建元数据文件

编辑 `examples/data/sample_metadata.csv`：

```csv
sample_id,actual_age,has_cancer,notes
007012,45,0,健康对照
000383,55,1,乳腺癌患者
000457,50,0,
000399,60,0,
000698,48,1,肺癌患者
000699,52,0,
000700,58,0,
```

**列说明**：
- `sample_id`：样本ID（必需）
- `actual_age`：实际年龄（可选，但强烈推荐）
- `has_cancer`：是否有癌症，0=否，1=是（可选）
- `notes`：备注（可选）

### 步骤 2：运行预测

```bash
cd examples
python 935k_enhanced_prediction.py
```

### 步骤 3：生成对比报告

```bash
python predict_with_metadata.py
```

脚本会：
- ✅ 自动计算标准化参数
- ✅ 生成对比报告
- ✅ 显示预测准确性统计

### 步骤 4：更新配置并重新预测

复制脚本输出的 `NORMALIZATION_PARAMS` 到 `935k_enhanced_prediction.py`，然后重新运行预测。

---

## 📊 元数据文件格式

### 最小配置（仅样本ID）

```csv
sample_id
007012
000383
000457
```

### 推荐配置（包含年龄）

```csv
sample_id,actual_age
007012,45
000383,55
000457,50
```

### 完整配置（包含所有信息）

```csv
sample_id,actual_age,has_cancer,gender,ethnicity,notes
007012,45,0,F,Asian,健康对照
000383,55,1,F,Caucasian,乳腺癌患者
000457,50,0,M,Asian,
```

**支持的列**：
- `sample_id` - 样本ID（必需）
- `actual_age` - 实际年龄（推荐）
- `has_cancer` - 癌症状态：0/1 或 False/True
- `gender` - 性别：M/F 或 Male/Female
- `ethnicity` - 种族
- `notes` - 备注

---

## 🔧 工作流程

### 方案 A：有实际年龄（推荐）⭐⭐⭐

```bash
# 1. 填写元数据
vim data/sample_metadata.csv

# 2. 运行预测
python 935k_enhanced_prediction.py

# 3. 生成对比报告并计算参数
python predict_with_metadata.py

# 4. 复制输出的 NORMALIZATION_PARAMS 到脚本

# 5. 重新运行预测
python 935k_enhanced_prediction.py

# 6. 再次生成对比报告
python predict_with_metadata.py
```

### 方案 B：没有实际年龄

```bash
# 1. 只填写 sample_id
vim data/sample_metadata.csv

# 2. 使用默认参数运行预测
python 935k_enhanced_prediction.py

# 3. 查看结果
python predict_with_metadata.py
```

---

## 📈 对比报告

运行 `predict_with_metadata.py` 后，会生成：

### `comparison_report.csv`

包含以下列：
- `sample_id` - 样本ID
- `actual_age` - 实际年龄
- `predicted_age` - 预测年龄
- `age_error` - 误差（预测 - 实际）
- `age_abs_error` - 绝对误差
- `has_cancer` - 实际癌症状态
- `cancer_probability` - 预测的癌症概率
- `cancer_prediction` - 预测的癌症状态（0/1）

### 统计信息

```
年龄预测准确性：
  平均绝对误差: 3.45 岁
  最大误差: 6.20 岁
  相关系数: 0.923

癌症预测准确性：
  准确率: 85.7%
```

---

## 💡 实际示例

### 示例 1：研究队列

你有一个研究队列，包含 100 个样本，年龄 30-70 岁：

```csv
sample_id,actual_age,has_cancer,cohort
S001,45,0,control
S002,52,1,case
S003,38,0,control
...
```

运行预测后，你会得到：
- 每个样本的预测年龄 vs 实际年龄
- 癌症预测准确率
- 整体统计信息

### 示例 2：临床样本

你有一些临床样本，但只知道部分样本的年龄：

```csv
sample_id,actual_age,has_cancer
P001,55,1
P002,,1
P003,48,0
P004,,0
```

脚本会：
- 使用有年龄的样本（P001, P003）计算参数
- 对所有样本进行预测
- 生成对比报告

---

## 🎯 最佳实践

### 1. 至少提供 2-3 个已知年龄的样本

这样可以计算准确的标准化参数。

### 2. 年龄范围要有代表性

如果你的样本年龄范围是 40-60 岁，标准化参数会针对这个范围优化。

### 3. 定期更新元数据

随着你获得更多信息，更新 `sample_metadata.csv` 并重新运行。

### 4. 保存对比报告

每次运行都会生成 `comparison_report.csv`，可以用于追踪预测准确性。

---

## 🔍 故障排查

### 问题 1：找不到元数据文件

```
⚠️ 元数据文件不存在: examples/data/sample_metadata.csv
```

**解决**：创建文件
```bash
cp examples/data/sample_metadata.csv.template examples/data/sample_metadata.csv
```

### 问题 2：无法计算标准化参数

```
⚠️ 无法计算标准化参数：至少需要 2 个有实际年龄的样本
```

**解决**：在 `sample_metadata.csv` 中至少填写 2 个样本的实际年龄

### 问题 3：预测结果不存在

```
❌ 预测结果不存在
```

**解决**：先运行预测脚本
```bash
python 935k_enhanced_prediction.py
```

---

## 📚 相关文档

- `predict_with_metadata.py` - 带元数据的预测脚本
- `PREDICTION_FIX_SUMMARY.md` - 预测修复总结
- `DIAGNOSIS_REPORT.md` - 诊断报告

---

## 🎓 总结

使用元数据文件的优势：
1. ✅ 数据和元数据在一起，易于管理
2. ✅ 自动计算标准化参数
3. ✅ 自动生成对比报告
4. ✅ 可以追踪预测准确性
5. ✅ 支持批量处理

**推荐工作流程**：
1. 创建 `sample_metadata.csv`
2. 填写实际年龄（至少 2-3 个样本）
3. 运行 `935k_enhanced_prediction.py`
4. 运行 `predict_with_metadata.py`
5. 复制标准化参数
6. 重新运行预测
7. 查看对比报告

