# 预测结果诊断报告

## 📊 当前预测结果分析

### 基本信息
- **样本数量**：7 个
- **预测日期**：2023-12-30
- **数据来源**：935k/EPICv2 甲基化芯片

---

## 🔍 发现的问题

### 问题 1：年龄预测标准差过小 ⚠️⚠️⚠️ **最严重**

**当前状态**：
```
年龄预测统计：
  均值: 59.73 岁
  标准差: 2.30 岁  ← 异常！
  范围: 56.72 - 62.80 岁
```

**问题分析**：
- ❌ 标准差只有 2.30 岁，远低于正常值（应该 10-20 岁）
- ❌ 7 个样本的年龄都挤在 56-63 岁之间
- ❌ 这表明模型输出的是**标准化值**，但缺少反标准化

**正常情况应该是**：
```
年龄预测统计：
  均值: 50-60 岁
  标准差: 10-20 岁  ← 正常
  范围: 30-80 岁（取决于样本）
```

**根本原因**：
```python
# 当前配置（935k_enhanced_prediction.py 第 70 行）
NORMALIZATION_PARAMS = {
    'age': None,  # ← 这里是 None，导致没有反标准化
    'clocks': None,
    'proteins': None,
}
```

---

### 问题 2：DunedinPACE 值异常 ⚠️⚠️

**当前状态**：
```
dunedinpace (衰老速度):
  均值: 52.17  ← 异常！
  范围: 46.60 - 54.72
```

**问题分析**：
- ❌ DunedinPACE 应该是衰老速度指标，正常值约 1.0
- ❌ 当前值 52.17 明显是标准化值
- ❌ 需要反标准化：`actual_value = predicted_value * 0.1 + 1.0`

**正常情况应该是**：
```
dunedinpace (衰老速度):
  均值: 1.0  ← 正常衰老速度
  范围: 0.8 - 1.2
```

---

### 问题 3：年龄与时钟相关性低 ⚠️

**当前状态**：
```
predicted_age vs altumage: -0.264  ← 负相关！
predicted_age vs grimage2: -0.081
predicted_age vs hrsinchphenoage: -0.157
predicted_age vs pchorvath2013: -0.128
```

**问题分析**：
- ❌ 年龄和表观遗传时钟应该高度正相关（> 0.8）
- ❌ 当前是负相关或低相关
- ❌ 原因：都是标准化值，缺少反标准化

**正常情况应该是**：
```
predicted_age vs altumage: 0.85  ← 高度正相关
predicted_age vs grimage2: 0.82
predicted_age vs hrsinchphenoage: 0.88
```

---

### 问题 4：炎症标志物偏高 ⚠️

**当前状态**：
```
关键炎症标志物：
  CRP: 1.080 ⚠️ (高于平均，需要关注)
  IL6: 0.537 ⚠️ (高于平均，需要关注)
  GDF15: 1.371 ⚠️ (高于平均，需要关注)
```

**问题分析**：
- ⚠️ 这些是标准化值（Z-score）
- ⚠️ 正值表示高于人群平均水平
- ✅ 这是正常的标准化值，**不需要修复**

**解释**：
- 蛋白质预测输出的是标准化值，这是设计行为
- 正值表示炎症水平高于平均，可能需要关注
- 详见 `PROTEIN_PREDICTION_GUIDE.md`

---

## 💡 解决方案

### 方案 1：使用已知年龄（推荐）⭐⭐⭐

如果你知道样本的实际年龄：

**步骤 1**：编辑 `fix_predictions_with_actual_ages.py`

```python
KNOWN_AGES = {
    '007012': 45,  # 填入实际年龄
    '000383': 55,
    '000457': 50,
    '000399': 60,
    '000698': 48,
    '000699': 52,
    '000700': 58,
}
```

**步骤 2**：运行脚本

```bash
cd examples
python fix_predictions_with_actual_ages.py
```

**步骤 3**：复制输出的配置代码到 `935k_enhanced_prediction.py`

**步骤 4**：重新运行预测

```bash
python 935k_enhanced_prediction.py
```

---

### 方案 2：使用其他实验室结果 ⭐⭐

如果你有其他实验室的年龄预测结果：

```python
OTHER_LAB_RESULTS = {
    '007012': 47,  # 其他实验室的结果
    '000383': 53,
    # ...
}
```

然后运行 `fix_predictions_with_actual_ages.py`

---

### 方案 3：使用年龄范围估计 ⭐

如果你只知道样本的大致年龄范围：

```python
EXPECTED_AGE_RANGE = {
    'min': 40,  # 最小年龄
    'max': 65,  # 最大年龄
    'mean': 52,  # 平均年龄（可选）
}
```

然后运行 `fix_predictions_with_actual_ages.py`

---

### 方案 4：使用经验值 ⭐

如果完全没有参考信息，使用人群统计值：

直接在 `935k_enhanced_prediction.py` 中设置：

```python
NORMALIZATION_PARAMS = {
    'age': {'mean': 50.0, 'std': 15.0},  # 假设人群平均50岁
    'clocks': {
        'altumage': {'mean': 50.0, 'std': 15.0},
        'dunedinpace': {'mean': 1.0, 'std': 0.1},
        'grimage2': {'mean': 50.0, 'std': 15.0},
        'hrsinchphenoage': {'mean': 50.0, 'std': 15.0},
        'pchorvath2013': {'mean': 50.0, 'std': 15.0},
    },
    'proteins': None,
}
```

---

## 📋 修复后的预期结果

### 年龄预测
```
年龄预测统计：
  均值: 50-60 岁
  标准差: 10-20 岁  ✅
  范围: 30-80 岁
```

### 表观遗传时钟
```
predicted_age vs altumage: 0.85  ✅ (高度相关)
predicted_age vs grimage2: 0.82  ✅
```

### DunedinPACE
```
dunedinpace:
  均值: 1.0  ✅ (正常衰老速度)
  范围: 0.8 - 1.2
```

---

## 🎯 总结

### 核心问题
✅ **模型输出的是标准化值，需要反标准化才能得到实际年龄**

### 解决步骤
1. 提供参考年龄（实际年龄、其他实验室结果或年龄范围）
2. 运行 `fix_predictions_with_actual_ages.py` 计算参数
3. 更新 `NORMALIZATION_PARAMS` 配置
4. 重新运行预测

### 蛋白质预测
✅ **蛋白质预测结果正常，不需要修复**
- 输出的是标准化值（Z-score）
- 这是设计行为
- 详见 `PROTEIN_PREDICTION_GUIDE.md`

---

## 📞 需要帮助？

如果你：
1. ✅ 知道样本的实际年龄 → 使用方案 1
2. ✅ 有其他实验室的结果 → 使用方案 2
3. ✅ 知道年龄范围 → 使用方案 3
4. ❌ 完全没有参考信息 → 使用方案 4（精度较低）

**相关文档**：
- `fix_predictions_with_actual_ages.py` - 修复脚本
- `PREDICTION_FIX_SUMMARY.md` - 修复总结
- `PROTEIN_PREDICTION_GUIDE.md` - 蛋白质解读

