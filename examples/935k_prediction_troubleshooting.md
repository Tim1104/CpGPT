# 935k 预测结果偏差问题排查指南

## 🔍 问题诊断

如果你发现 `935k_enhanced_prediction.py` 的预测结果与实际值相差较远，可能是以下原因：

### 1. **输入长度不匹配** ⚠️ 最常见

**问题**：
- 脚本中的 `MAX_INPUT_LENGTH` 与模型训练时的配置不一致
- 不同模型训练时使用了不同的 `max_length`：
  - `age_cot`: 20000
  - `clock_proxies`: 10000
  - `proteins`: 10000
  - `cancer`: 10000

**解决方案**：
```python
# 在 935k_enhanced_prediction.py 中修改
MAX_INPUT_LENGTH = 20000  # 改为与 age_cot 一致
```

或者为每个模型单独设置（已在修复版本中实现）。

### 2. **缺少反标准化** ⚠️ 最关键

**问题**：
- 模型输出的是**标准化后的值**（均值≈0，标准差≈1）
- 需要反标准化才能得到实际的年龄/蛋白质浓度等值

**解决方案**：

#### 方法 1：使用训练数据的统计信息（推荐）

如果你有训练数据，计算其均值和标准差：

```python
import pandas as pd
import numpy as np

# 读取训练数据
train_data = pd.read_csv('your_training_data.csv')

# 计算年龄的均值和标准差
age_mean = train_data['age'].mean()
age_std = train_data['age'].std()

print(f"Age - Mean: {age_mean:.2f}, Std: {age_std:.2f}")

# 在脚本中设置
NORMALIZATION_PARAMS = {
    'age': {'mean': age_mean, 'std': age_std},
    'clocks': {
        'altumage': {'mean': age_mean, 'std': age_std},
        'grimage2': {'mean': age_mean, 'std': age_std},
        # ... 其他时钟
    },
    'proteins': None,  # 蛋白质通常已经是标准化值
}
```

#### 方法 2：使用经验值

如果没有训练数据，可以使用常见的人群统计值：

```python
NORMALIZATION_PARAMS = {
    'age': {'mean': 50.0, 'std': 15.0},  # 假设人群平均年龄50岁，标准差15岁
    'clocks': {
        'altumage': {'mean': 50.0, 'std': 15.0},
        'dunedinpace': {'mean': 1.0, 'std': 0.1},  # DunedinPACE 是速度指标
        'grimage2': {'mean': 50.0, 'std': 15.0},
        'hrsinchphenoage': {'mean': 50.0, 'std': 15.0},
        'pchorvath2013': {'mean': 50.0, 'std': 15.0},
    },
    'proteins': None,
}
```

#### 方法 3：从已知样本反推

如果你有一些已知年龄的样本：

```python
# 运行预测后
predicted_ages = [25.3, 30.1, 28.7]  # 模型输出（标准化值）
actual_ages = [45, 55, 50]  # 实际年龄

# 反推标准化参数
import numpy as np
from scipy.optimize import minimize

def loss(params):
    mean, std = params
    denorm = np.array(predicted_ages) * std + mean
    return np.mean((denorm - np.array(actual_ages))**2)

result = minimize(loss, x0=[50, 15], method='Nelder-Mead')
mean, std = result.x
print(f"推测的标准化参数 - Mean: {mean:.2f}, Std: {std:.2f}")
```

### 3. **数据质量问题**

**检查点**：
- ✅ 输入数据格式是否正确（935k/EPICv2 格式）
- ✅ 是否有大量缺失值（NA > 20%）
- ✅ Beta 值是否在 [0, 1] 范围内
- ✅ 样本是否来自人类（homo sapiens）

### 4. **模型适用性**

**注意**：
- 模型是在特定人群上训练的（可能是欧美人群）
- 如果你的样本来自不同人群，可能需要重新校准
- 某些疾病状态可能影响预测准确性

## 🛠️ 快速修复步骤

### 步骤 1：更新脚本

已修复的版本包含：
- ✅ 自动使用模型训练时的 `max_length`
- ✅ 添加反标准化函数
- ✅ 警告提示配置不匹配

### 步骤 2：设置标准化参数

在 `935k_enhanced_prediction.py` 中找到：

```python
NORMALIZATION_PARAMS = {
    'age': None,  # 👈 在这里设置
    'clocks': None,
    'proteins': None,
}
```

修改为：

```python
NORMALIZATION_PARAMS = {
    'age': {'mean': 50.0, 'std': 15.0},  # 根据你的数据调整
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

### 步骤 3：重新运行预测

```bash
cd examples
python 935k_enhanced_prediction.py
```

### 步骤 4：验证结果

检查输出的 CSV 文件：
- `age_predictions.csv` - 年龄应该在合理范围内（例如 20-90 岁）
- `clocks_predictions.csv` - 时钟值应该接近实际年龄
- 如果还是不合理，调整 `NORMALIZATION_PARAMS`

## 📊 调试技巧

### 查看原始模型输出

临时禁用反标准化，查看模型的原始输出：

```python
# 在 predict_age 函数中
pred_values = predictions["pred_conditions"].flatten().cpu().numpy()
print(f"原始模型输出: {pred_values}")  # 应该接近 0，范围大约 [-3, 3]
```

### 对比已知样本

如果有已知年龄的样本：

```python
# 在结果中添加实际年龄
results = pd.read_csv('results/935k_enhanced_predictions/age_predictions.csv')
results['actual_age'] = [45, 55, 50, ...]  # 你的实际年龄
results['difference'] = results['predicted_age'] - results['actual_age']
print(results[['sample_id', 'predicted_age', 'actual_age', 'difference']])
```

## 🎯 预期结果

正确配置后，你应该看到：
- **年龄预测**：误差 ±5-10 岁（取决于样本质量）
- **表观遗传时钟**：与实际年龄相关性 > 0.8
- **蛋白质预测**：标准化值，范围大约 [-3, 3]

## 📞 需要帮助？

如果问题仍然存在，请提供：
1. 样本的实际年龄范围
2. 预测的年龄范围
3. 数据来源（EPIC/EPICv2/935k）
4. 样本数量和缺失值比例

