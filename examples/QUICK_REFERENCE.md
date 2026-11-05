# 935k 预测快速参考卡

## 🚀 快速开始

```bash
cd examples
python 935k_enhanced_prediction.py
```

## 📊 预测结果解读

### 1️⃣ 年龄预测

**文件**：`results/935k_enhanced_predictions/age_predictions.csv`

**正常范围**：20-90 岁

**如果结果不对**：
- 📖 阅读：`PREDICTION_FIX_SUMMARY.md`
- 🔧 使用：`calculate_normalization_params.py`

---

### 2️⃣ 蛋白质预测 ⭐ 重要

**文件**：`results/935k_enhanced_predictions/proteins_predictions.csv`

**⚠️ 关键信息**：
- ✅ 输出的是**标准化值**（Z-score），不是实际浓度
- ✅ 范围：通常在 [-3, +3]
- ✅ **不需要反标准化**

**如何解读**：

| 值 | 含义 | 临床意义 |
|---|------|---------|
| **< -1** | 低于平均 | ✅ 通常更健康 |
| **0** | 平均水平 | ✓ 正常 |
| **> +1** | 高于平均 | ⚠️ 需要关注 |
| **> +2** | 明显升高 | ⚠️⚠️ 风险较高 |

**详细解读**：
- 📖 **必读**：`PROTEIN_PREDICTION_GUIDE.md`
- 📖 快速总结：`PROTEIN_FIX_SUMMARY.md`
- 🔧 检查工具：`check_protein_predictions.py`

---

### 3️⃣ 表观遗传时钟

**文件**：`results/935k_enhanced_predictions/clocks_predictions.csv`

**包含**：
- GrimAge2（死亡率预测）
- DunedinPACE（衰老速度，1.0 = 正常）
- PhenoAge（表型年龄）
- Horvath（泛组织年龄）
- AltumAge

**解读**：与年龄预测类似，需要反标准化

---

### 4️⃣ 癌症风险

**文件**：`results/935k_enhanced_predictions/cancer_predictions.csv`

**包含**：
- `cancer_probability`：0-1 之间的概率
- `cancer_prediction`：0（低风险）或 1（高风险）

---

### 5️⃣ 器官健康评分

**文件**：`results/935k_enhanced_predictions/organ_health_scores.csv`

**评分范围**：0-100
- **80-100**：优秀 💚
- **60-79**：良好 ✅
- **40-59**：一般 ⚠️
- **< 40**：需要关注 ⚠️⚠️

**包含器官**：
- 心脏（Heart）
- 肾脏（Kidney）
- 肝脏（Liver）
- 免疫系统（Immune）
- 代谢系统（Metabolic）
- 血管系统（Vascular）

---

## 🔧 常见问题

### Q1: 年龄预测不准确

**解决方案**：
1. 阅读 `PREDICTION_FIX_SUMMARY.md`
2. 使用 `calculate_normalization_params.py` 计算参数
3. 更新 `NORMALIZATION_PARAMS` 配置

### Q2: 蛋白质值看起来很小

**这是正常的！**
- 蛋白质输出的是标准化值（Z-score）
- 不是实际浓度（mg/L）
- 详见 `PROTEIN_PREDICTION_GUIDE.md`

### Q3: max_length 警告

**已自动修复**：
- 脚本会自动使用每个模型训练时的 max_length
- 警告信息已改为提示信息
- 不影响预测准确性

---

## 📚 完整文档索引

### 核心文档

| 文档 | 用途 | 优先级 |
|------|------|--------|
| `QUICK_REFERENCE.md` | 快速参考（本文档） | ⭐⭐⭐ |
| `PROTEIN_PREDICTION_GUIDE.md` | 蛋白质预测详细解读 | ⭐⭐⭐ 必读 |
| `PREDICTION_FIX_SUMMARY.md` | 年龄预测修复总结 | ⭐⭐⭐ |
| `PROTEIN_FIX_SUMMARY.md` | 蛋白质预测修复总结 | ⭐⭐ |

### 工具脚本

| 脚本 | 用途 |
|------|------|
| `935k_enhanced_prediction.py` | 主预测脚本 |
| `calculate_normalization_params.py` | 计算年龄标准化参数 |
| `check_protein_predictions.py` | 检查蛋白质预测结果 |

### 参考文档

| 文档 | 用途 |
|------|------|
| `935k_prediction_troubleshooting.md` | 完整排查指南 |
| `ENHANCED_PREDICTION_README.md` | 功能说明 |
| `ORGAN_HEALTH_SCORES_GUIDE.md` | 器官健康评分指南 |

---

## 💡 最佳实践

### 1. 首次使用

```bash
# 1. 运行预测
python 935k_enhanced_prediction.py

# 2. 检查蛋白质结果
python check_protein_predictions.py

# 3. 查看 PDF 报告
# 位于 results/935k_enhanced_predictions/report.pdf
```

### 2. 如果年龄预测不准

```bash
# 1. 准备已知年龄数据
# 编辑 calculate_normalization_params.py

# 2. 计算参数
python calculate_normalization_params.py

# 3. 更新配置并重新运行
python 935k_enhanced_prediction.py
```

### 3. 解读蛋白质结果

```python
import pandas as pd

# 读取结果
proteins = pd.read_csv('results/935k_enhanced_predictions/proteins_predictions.csv')

# 查看关键炎症标志物
print(proteins[['sample_id', 'CRP', 'IL6', 'TNF_alpha']])

# 负值 = 健康 ✅
# 正值 = 需要关注 ⚠️
```

---

## 🎯 记住这些要点

1. ✅ **蛋白质预测输出标准化值是正常的**
2. ✅ **不需要反标准化蛋白质值**
3. ✅ **负值通常表示更健康**
4. ✅ **脚本会自动使用正确的 max_length**
5. ✅ **年龄预测需要设置标准化参数**

---

**需要帮助？** 查看对应的详细文档或运行检查工具！

