# 935k 预测结果偏差修复总结

## 📋 问题诊断

你遇到的"预测结果与实际相差较远"的问题，主要由以下原因造成：

### 1. **输入长度不匹配** ⚠️
- **原配置**：`MAX_INPUT_LENGTH = 30000`
- **模型训练配置**：
  - `age_cot`: 20000
  - `clock_proxies`: 10000
  - `proteins`: 10000

**影响**：输入长度不一致会导致模型看到的数据分布与训练时不同。

### 2. **缺少反标准化** ⚠️⚠️⚠️ 最关键
- 模型输出的是**标准化值**（均值≈0，标准差≈1）
- 需要反标准化：`actual_value = predicted_value * std + mean`
- 原脚本直接使用了标准化值，导致预测结果不合理

## ✅ 已修复内容

### 1. 自动使用正确的 max_length
```python
# 修改前
max_length=MAX_INPUT_LENGTH,  # 固定使用 30000

# 修改后
model_max_length = config.data.get('max_length', 20000)
max_length=model_max_length,  # 使用模型训练时的配置
```

### 2. 添加反标准化函数
```python
def denormalize_predictions(values, mean=None, std=None):
    """反标准化预测值"""
    if mean is None or std is None:
        print("⚠️ 警告：未提供标准化参数")
        return values
    return values * std + mean
```

### 3. 添加配置参数
```python
NORMALIZATION_PARAMS = {
    'age': None,  # 需要设置
    'clocks': None,  # 需要设置
    'proteins': None,
}
```

### 4. 添加警告提示
- 当 max_length 不匹配时会警告
- 当缺少标准化参数时会提示

## 🛠️ 使用步骤

### 方法 A：使用已知样本反推参数（推荐）

**步骤 1**：运行预测（使用默认配置）
```bash
cd examples
python 935k_enhanced_prediction.py
```

**步骤 2**：准备已知年龄数据

编辑 `calculate_normalization_params.py`：
```python
KNOWN_AGES = {
    'Sample1': 45,  # 样本1的实际年龄
    'Sample2': 55,  # 样本2的实际年龄
    'Sample3': 50,  # 样本3的实际年龄
    # ... 添加更多已知样本
}
```

**步骤 3**：计算标准化参数
```bash
python calculate_normalization_params.py
```

**步骤 4**：复制输出的配置代码到 `935k_enhanced_prediction.py`

**步骤 5**：重新运行预测
```bash
python 935k_enhanced_prediction.py
```

### 方法 B：使用经验值

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
    'proteins': None,  # 蛋白质保持标准化值
}
```

然后运行：
```bash
python 935k_enhanced_prediction.py
```

## 📊 预期结果

正确配置后：

### 年龄预测
- **合理范围**：20-90 岁
- **误差**：±5-10 岁（取决于样本质量）
- **相关性**：与实际年龄 R² > 0.8

### 表观遗传时钟
- **altumage, grimage2, hrsinchphenoage, pchorvath2013**：应该接近实际年龄
- **dunedinpace**：速度指标，正常值约 1.0（1.0 = 正常衰老速度）

### 蛋白质预测 ⭐ 重要
- **范围**：标准化值（Z-score），通常在 [-3, 3] 之间
- **解释**：
  - **0** = 人群平均水平
  - **负值（< 0）** = 低于平均（通常表示更健康）
  - **正值（> 0）** = 高于平均（可能表示风险）
- **注意**：
  - ✅ **不需要反标准化** - 这是设计行为
  - ✅ 标准化值可以直接用于风险评估
  - ✅ 详见 `PROTEIN_PREDICTION_GUIDE.md`

## 🔍 验证方法

### 1. 检查年龄预测是否合理
```python
import pandas as pd

results = pd.read_csv('results/935k_enhanced_predictions/age_predictions.csv')
print(results['predicted_age'].describe())

# 应该看到：
# mean    约 40-60
# std     约 10-20
# min     约 20-30
# max     约 70-90
```

### 1b. 检查蛋白质预测是否合理
```python
proteins = pd.read_csv('results/935k_enhanced_predictions/proteins_predictions.csv')

# 检查关键炎症标志物
print(proteins[['sample_id', 'CRP', 'IL6', 'TNF_alpha']].describe())

# 应该看到：
# mean    约 -0.5 到 0.5（接近 0）
# std     约 0.5 到 1.5
# min     约 -3 到 -1
# max     约 1 到 3

# 如果值超出 [-5, +5]，可能是数据质量问题
```

### 2. 对比已知样本（年龄）
```python
# 如果有实际年龄
results['actual_age'] = [45, 55, 50, ...]
results['error'] = results['predicted_age'] - results['actual_age']
print(results[['sample_id', 'predicted_age', 'actual_age', 'error']])
```

### 3. 检查时钟一致性
```python
clocks = pd.read_csv('results/935k_enhanced_predictions/clocks_predictions.csv')
age = pd.read_csv('results/935k_enhanced_predictions/age_predictions.csv')

merged = clocks.merge(age, on='sample_id')
print(merged[['predicted_age', 'altumage', 'grimage2']].corr())

# 相关性应该 > 0.8
```

### 4. 检查蛋白质异常值
```python
# 找出异常升高的蛋白质（> +2 标准差）
proteins_numeric = proteins.select_dtypes(include='number')
high_proteins = proteins_numeric.columns[(proteins_numeric > 2).any()]
print(f"异常升高的蛋白质: {len(high_proteins)} 个")

# 找出异常降低的蛋白质（< -2 标准差）
low_proteins = proteins_numeric.columns[(proteins_numeric < -2).any()]
print(f"异常降低的蛋白质: {len(low_proteins)} 个")

# 通常每个样本应该有 5-10% 的蛋白质在 ±2 标准差之外
```

## 📁 相关文件

- `935k_enhanced_prediction.py` - 主预测脚本（已修复）
- `calculate_normalization_params.py` - 年龄参数计算工具
- `935k_prediction_troubleshooting.md` - 详细排查指南
- `PROTEIN_PREDICTION_GUIDE.md` - **蛋白质预测解读指南** ⭐ 必读

## ⚠️ 注意事项

1. **不同人群可能需要不同的标准化参数**
   - 欧美人群 vs 亚洲人群
   - 健康人群 vs 疾病人群

2. **数据质量很重要**
   - 缺失值 < 20%
   - Beta 值在 [0, 1] 范围内
   - 使用正确的芯片类型（935k/EPICv2）

3. **模型局限性**
   - 模型在特定数据集上训练
   - 极端年龄（<20 或 >90）可能不准确
   - 某些疾病状态可能影响预测

## 🆘 仍然有问题？

如果修复后结果仍不理想，请检查：

1. ✅ 是否正确设置了 `NORMALIZATION_PARAMS`
2. ✅ 输入数据格式是否正确
3. ✅ 是否有大量缺失值
4. ✅ 样本是否来自人类（homo sapiens）
5. ✅ 是否使用了正确的芯片类型

提供以下信息以便进一步诊断：
- 样本的实际年龄范围
- 预测的年龄范围（修复前后）
- 数据来源（EPIC/EPICv2/935k）
- 样本数量和缺失值比例

## 📚 参考资料

- [935k Zero-Shot Inference Guide](../docs/935k_zero_shot_inference_guide.md)
- [935k Platform Preparation Guide](../docs/935k_platform_preparation_guide.md)

