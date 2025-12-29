# PDF生成问题修复说明

## 🐛 问题描述

根据日志 `log2025-12-24-143010.txt`，PDF生成遇到两个问题：

### 问题1: 中文字体缺失
```
UserWarning: Glyph 30002 (\N{CJK UNIFIED IDEOGRAPH-7532}) missing from font(s) DejaVu Sans.
```
**影响**：图表中的中文会显示为方框

### 问题2: 直方图bins数量错误
```
Too many bins for data range. Cannot create 30 finite-sized bins.
```
**影响**：PDF生成失败

---

## ✅ 已修复

### 修复1: 自动调整直方图bins数量

**问题原因**：当数据范围很小或样本数很少时，固定使用30个bins会导致错误。

**解决方案**：根据数据范围和样本数自动调整bins数量

```python
# 修复前
ax.hist(data, bins=30, ...)

# 修复后
age_data = combined_df['predicted_age'].dropna()
n_samples = len(age_data)
data_range = age_data.max() - age_data.min()

# 根据样本数和数据范围自动调整bins
if n_samples < 10:
    bins = min(5, n_samples)
elif data_range < 1:
    bins = 5
elif data_range < 10:
    bins = min(10, n_samples)
else:
    bins = min(30, n_samples)

ax.hist(age_data, bins=bins, ...)
```

### 修复2: 配置中文字体支持

**问题原因**：Linux系统默认使用DejaVu Sans字体，不支持中文。

**解决方案**：自动检测并使用系统中文字体

```python
# 尝试多个常见的中文字体
chinese_fonts = [
    'SimHei',  # Windows
    'WenQuanYi Micro Hei',  # Linux
    'Noto Sans CJK SC',  # Linux
    'Droid Sans Fallback',  # Linux
    'STHeiti',  # macOS
    'Arial Unicode MS',  # macOS
]

import matplotlib.font_manager as fm
available_fonts = [f.name for f in fm.fontManager.ttflist]

for font in chinese_fonts:
    if font in available_fonts:
        plt.rcParams['font.sans-serif'] = [font]
        plt.rcParams['axes.unicode_minus'] = False
        break
```

### 修复3: 添加错误处理

**解决方案**：即使某个图表生成失败，也能继续生成其他部分

```python
try:
    # 生成图表
    fig, ax = plt.subplots(figsize=(8, 5))
    # ... 绘图代码 ...
    plt.savefig(path)
    plt.close()
except Exception as e:
    print(f"  ⚠ 图表生成失败: {e}")
    plt.close()
```

---

## 🚀 如何使用修复后的版本

### 1. 直接运行（已自动修复）

```bash
cd examples
python 935k_enhanced_prediction.py
```

修复后的脚本会：
- ✅ 自动调整bins数量，避免数据范围错误
- ✅ 自动检测并使用系统中文字体
- ✅ 即使某个图表失败，也能生成PDF

### 2. 安装中文字体（可选，改善中文显示）

#### Linux (Ubuntu/Debian)

```bash
# 安装文泉驿微米黑字体
sudo apt-get install fonts-wqy-microhei

# 或安装Noto CJK字体
sudo apt-get install fonts-noto-cjk

# 清除matplotlib字体缓存
rm -rf ~/.cache/matplotlib
```

#### Linux (CentOS/RHEL)

```bash
# 安装文泉驿微米黑字体
sudo yum install wqy-microhei-fonts

# 清除matplotlib字体缓存
rm -rf ~/.cache/matplotlib
```

#### macOS

macOS已自带中文字体（STHeiti），无需额外安装。

#### Windows

Windows已自带中文字体（SimHei），无需额外安装。

### 3. 验证中文字体

运行脚本时，会显示使用的字体：

```
[7/7] 生成PDF报告...
  ✓ 使用中文字体: WenQuanYi Micro Hei
```

如果显示：
```
  ⚠ 未找到中文字体，图表中文可能显示为方框
```

说明系统没有中文字体，建议安装（见上方安装说明）。

---

## 📊 修复效果

### 修复前
- ❌ PDF生成失败：`Too many bins for data range`
- ❌ 中文显示为方框

### 修复后
- ✅ PDF成功生成
- ✅ 自动调整bins数量
- ✅ 自动使用系统中文字体（如果有）
- ✅ 即使某个图表失败，也能生成其他部分

---

## 🔍 日志解读

### 正常日志（修复后）

```
[7/7] 生成PDF报告...
  ✓ 使用中文字体: WenQuanYi Micro Hei
  ✓ PDF报告已生成: results/935k_enhanced_predictions/comprehensive_report.pdf
```

### 警告日志（可忽略）

```
  ⚠ 未找到中文字体，图表中文可能显示为方框
```
**说明**：PDF仍会生成，但图表中文显示为方框。建议安装中文字体。

```
  ⚠ 年龄分布图生成失败: ...
```
**说明**：某个图表生成失败，但PDF会继续生成其他部分。

---

## 🛠️ 故障排除

### 问题1: 仍然报bins错误

**可能原因**：数据全部相同（方差为0）

**解决方案**：检查数据是否正常
```bash
# 查看预测结果
head results/935k_enhanced_predictions/age_predictions.csv
```

### 问题2: 中文仍显示为方框

**解决方案**：
1. 安装中文字体（见上方安装说明）
2. 清除matplotlib缓存：`rm -rf ~/.cache/matplotlib`
3. 重新运行脚本

### 问题3: PDF完全无法生成

**可能原因**：缺少依赖库

**解决方案**：
```bash
pip install reportlab matplotlib
```

---

## 📝 技术细节

### bins数量计算逻辑

| 条件 | bins数量 | 说明 |
|------|---------|------|
| 样本数 < 10 | min(5, n_samples) | 样本太少，减少bins |
| 数据范围 < 1 | 5 | 范围太小，固定5个bins |
| 数据范围 < 10 | min(10, n_samples) | 中等范围 |
| 其他 | min(30, n_samples) | 正常范围 |

### 中文字体优先级

1. **Windows**: SimHei
2. **Linux**: WenQuanYi Micro Hei → Noto Sans CJK SC → Droid Sans Fallback
3. **macOS**: STHeiti → Arial Unicode MS

---

## ✅ 总结

所有问题已修复！现在可以正常生成包含器官健康评分的PDF报告了。

如果遇到其他问题，请查看完整日志或提交Issue。

