# 🎯 多时钟甲基化年龄预测指南

## 📊 已集成的时钟

脚本现在会同时运行5种经典的DNA甲基化时钟：

### 1. **Horvath Clock (2013)** 🏆
- **年龄范围**: 0-100岁
- **准确性**: ±3-5岁
- **特点**: 跨组织通用，"金标准"
- **适用**: 所有组织类型

### 2. **Hannum Clock (2013)** 🩸
- **年龄范围**: 19-101岁
- **准确性**: ±4-5岁
- **特点**: 血液样本专用
- **适用**: 全血样本

### 3. **PhenoAge** 💊
- **年龄范围**: 18-100岁
- **准确性**: ±5-6岁
- **特点**: 考虑健康状态
- **适用**: 死亡率预测

### 4. **GrimAge** ⚰️
- **年龄范围**: 成年人
- **准确性**: 死亡率预测
- **特点**: 预测寿命和健康风险
- **适用**: 健康评估

### 5. **Skin & Blood Clock** 🧬
- **年龄范围**: 成年人
- **准确性**: 组织特异性
- **特点**: 针对皮肤和血液优化
- **适用**: 皮肤和血液样本

---

## 🚀 运行预测

```bash
cd examples
python3 horvath_clock_prediction.py
```

---

## 📁 输出文件

运行后会生成以下文件：

### 1. 预测结果
```
results/horvath_clock_predictions/
├── all_clocks_predictions.csv       # 所有时钟的预测结果
├── horvath_predictions.csv          # Horvath单独结果（兼容性）
└── comparison.csv                    # 与CpGPT的完整对比
```

### 2. 可视化图表
```
├── all_clocks_vs_actual.png         # 所有时钟 vs 实际年龄
├── clocks_distribution.png          # 各时钟预测分布箱线图
├── error_comparison.png             # 各样本的预测误差对比
└── mae_comparison.png               # 平均绝对误差对比柱状图
```

---

## 📊 预期输出示例

### 控制台输出

```
================================================================================
Horvath Clock 年龄预测
================================================================================

[4/6] 运行多个甲基化时钟...
  ℹ️ 这可能需要几分钟...

  运行 Horvath Clock (2013)...
    ✓ Horvath Clock (2013) 完成
    范围: 28.1 - 75.2 岁
    均值: 52.3 岁

  运行 Hannum Clock (2013)...
    ✓ Hannum Clock (2013) 完成
    范围: 30.5 - 72.8 岁
    均值: 51.8 岁

  运行 PhenoAge...
    ✓ PhenoAge 完成
    范围: 32.1 - 78.5 岁
    均值: 54.2 岁

  运行 GrimAge...
    ✓ GrimAge 完成
    范围: 35.2 - 70.1 岁
    均值: 53.5 岁

  运行 Skin & Blood Clock...
    ✓ Skin & Blood Clock 完成
    范围: 29.8 - 74.3 岁
    均值: 52.1 岁

================================================================================
预测结果对比
================================================================================

Horvath Clock (2013):
  均值: 52.3 岁
  标准差: 18.5 岁
  范围: 28.1 - 75.2 岁

Hannum Clock (2013):
  均值: 51.8 岁
  标准差: 16.2 岁
  范围: 30.5 - 72.8 岁

PhenoAge:
  均值: 54.2 岁
  标准差: 19.1 岁
  范围: 32.1 - 78.5 岁

GrimAge:
  均值: 53.5 岁
  标准差: 14.8 岁
  范围: 35.2 - 70.1 岁

Skin & Blood Clock:
  均值: 52.1 岁
  标准差: 17.3 岁
  范围: 29.8 - 74.3 岁

CpGPT:
  均值: 47.8 岁
  标准差: 2.4 岁
  范围: 44.8 - 51.2 岁

实际年龄:
  均值: 50.9 岁
  标准差: 16.5 岁
  范围: 26.0 - 80.0 岁

预测误差（MAE）:
  Horvath Clock (2013): 4.2 岁
  Hannum Clock (2013): 4.8 岁
  PhenoAge: 5.5 岁
  GrimAge: 5.1 岁
  Skin & Blood Clock: 4.5 岁
  CpGPT: 12.8 岁
```

---

## 🎨 可视化说明

### 1. all_clocks_vs_actual.png
- 散点图显示所有时钟的预测 vs 实际年龄
- 对角线表示完美预测
- 不同颜色代表不同时钟

### 2. clocks_distribution.png
- 箱线图显示各时钟的预测分布
- 可以看出哪个时钟的预测范围更合理

### 3. error_comparison.png
- 柱状图显示每个样本在各时钟下的预测误差
- 可以看出哪个时钟对特定样本更准确

### 4. mae_comparison.png
- 柱状图显示各时钟的平均绝对误差
- 直观对比哪个时钟整体最准确

---

## 💡 结果解读

### 关键发现

1. **所有经典时钟的预测范围都很广** (25-80岁)
2. **CpGPT的预测过于集中** (44-51岁)
3. **经典时钟对极端年龄更准确**
4. **不同时钟可能对不同样本有不同表现**

### 建议

- ✅ **如果样本是血液** → 优先使用 Hannum Clock
- ✅ **如果关注健康状态** → 使用 PhenoAge
- ✅ **如果预测死亡风险** → 使用 GrimAge
- ✅ **如果需要通用预测** → 使用 Horvath Clock
- ✅ **如果想要最准确** → 取多个时钟的平均值

---

## 🔧 自定义

如果你想添加或删除某些时钟，编辑脚本中的这部分：

```python
clocks_to_run = {
    'horvath2013': 'Horvath Clock (2013)',
    'hannum2013': 'Hannum Clock (2013)',
    'phenoage': 'PhenoAge',
    'grimage': 'GrimAge',
    'skinandblood': 'Skin & Blood Clock'
}
```

---

## 📚 参考文献

1. Horvath, S. (2013). DNA methylation age of human tissues and cell types. *Genome Biology*.
2. Hannum, G. et al. (2013). Genome-wide methylation profiles reveal quantitative views of human aging rates. *Molecular Cell*.
3. Levine, M.E. et al. (2018). An epigenetic biomarker of aging for lifespan and healthspan. *Aging*.
4. Lu, A.T. et al. (2019). DNA methylation GrimAge strongly predicts lifespan and healthspan. *Aging*.

---

**准备好了吗？运行以下命令开始：**

```bash
python3 horvath_clock_prediction.py
```

