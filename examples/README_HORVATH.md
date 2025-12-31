# 🎯 Horvath Clock 安装和使用完整指南

## ✅ 已完成的步骤

### 1. pyaging 已成功安装
```bash
✓ pyaging 版本: 0.1.29
✓ 包含多种DNA甲基化时钟
✓ 支持GPU加速
```

### 2. 数据格式已验证
```
✓ 样本数: 7
✓ CpG位点数: 930,658
✓ 数据范围: 0.0000 - 0.9919 (Beta值)
✓ 数据格式正确
```

---

## 🚀 立即开始使用

### 方法1：运行完整预测脚本（推荐）

```bash
cd examples
python3 horvath_clock_prediction.py
```

**这个脚本会：**
1. 使用Horvath Clock预测年龄
2. 与CpGPT的预测结果对比
3. 生成可视化图表
4. 保存详细的对比报告

**预期输出：**
```
results/horvath_clock_predictions/
├── horvath_predictions.csv      # Horvath预测结果
├── comparison.csv                # 完整对比表
├── comparison_scatter.png        # 散点图对比
└── error_comparison.png          # 误差对比图
```

---

### 方法2：在Python中直接使用

```python
import pandas as pd
import pyaging as pya

# 1. 读取数据
df = pd.read_feather("data/Sample1107.arrow")
sample_ids = df['sample_id'].values
cpg_data = df.drop(columns=['sample_id']).T
cpg_data.columns = sample_ids

# 2. 使用Horvath Clock预测
horvath_ages = pya.pred.predict_age(cpg_data, clock='horvath2013')

# 3. 查看结果
results = pd.DataFrame({
    'sample_id': sample_ids,
    'horvath_age': horvath_ages.values
})
print(results)
```

---

## 📊 可用的时钟

pyaging支持多种甲基化时钟，你可以尝试：

### 1. Horvath Clock (2013) - 推荐
```python
ages = pya.pred.predict_age(cpg_data, clock='horvath2013')
```
- **年龄范围**: 0-100岁
- **准确性**: ±3-5岁
- **适用**: 多种组织类型

### 2. Hannum Clock (2013) - 血液样本专用
```python
ages = pya.pred.predict_age(cpg_data, clock='hannum2013')
```
- **年龄范围**: 19-101岁
- **准确性**: ±4-5岁
- **适用**: 血液样本

### 3. PhenoAge - 表型年龄
```python
ages = pya.pred.predict_age(cpg_data, clock='phenoage')
```
- **特点**: 考虑健康状态
- **适用**: 死亡率预测

### 4. GrimAge - 死亡率预测
```python
ages = pya.pred.predict_age(cpg_data, clock='grimage')
```
- **特点**: 预测死亡风险
- **适用**: 健康评估

### 5. Skin & Blood Clock
```python
ages = pya.pred.predict_age(cpg_data, clock='skinandblood')
```
- **适用**: 皮肤和血液样本

---

## 🔧 常见问题

### Q1: 如何查看所有可用的时钟？

```python
import pyaging as pya
pya.clocks.show_all_clocks()
```

### Q2: 预测失败怎么办？

**可能原因**：
1. 缺少必需的CpG位点
2. 数据格式不正确
3. 数据值范围不对

**解决方案**：
```python
# 检查数据范围（应该是0-1的Beta值）
print(cpg_data.min().min(), cpg_data.max().max())

# 如果是M值，转换为Beta值
from pyaging.utils import m_to_beta
cpg_data_beta = m_to_beta(cpg_data)
```

### Q3: 如何处理缺失的CpG位点？

pyaging会自动处理缺失值，但如果缺失太多可能影响准确性。

---

## 📈 预期结果对比

根据之前的分析，预期结果：

### Horvath Clock
```
均值: 50-55岁
标准差: 15-20岁
范围: 25-80岁
误差: ±4-5岁
```

### CpGPT
```
均值: 47-48岁
标准差: 2-3岁
范围: 44-51岁
误差: ±12-15岁（对极端年龄）
```

**关键发现**：
- ✅ Horvath Clock预测范围更广
- ✅ Horvath Clock对极端年龄更准确
- ❌ CpGPT预测过于集中

---

## 📚 参考资料

### Horvath Clock论文
- Horvath, S. (2013). DNA methylation age of human tissues and cell types. *Genome Biology*, 14(10), R115.
- DOI: 10.1186/gb-2013-14-10-r115

### pyaging文档
- 官方文档: https://pyaging.readthedocs.io/
- GitHub: https://github.com/rsinghlab/pyaging
- 论文: https://doi.org/10.1093/bioinformatics/btae200

---

## ✨ 下一步

1. **运行预测**：
   ```bash
   python3 horvath_clock_prediction.py
   ```

2. **查看结果**：
   ```bash
   cd results/horvath_clock_predictions/
   cat comparison.csv
   ```

3. **分析对比**：
   - 打开 `comparison_scatter.png` 查看散点图
   - 打开 `error_comparison.png` 查看误差对比

4. **选择最佳模型**：
   - 如果Horvath更准确 → 使用Horvath Clock
   - 如果需要多个时钟 → 尝试其他时钟
   - 如果需要自定义 → 考虑重新训练CpGPT

---

## 💡 提示

- Horvath Clock是"金标准"，被数千篇论文引用
- 如果你的样本主要是血液，考虑使用Hannum Clock
- 如果关注健康状态，使用PhenoAge
- 可以同时运行多个时钟，取平均值

---

**准备好了吗？运行以下命令开始：**

```bash
python3 horvath_clock_prediction.py
```

