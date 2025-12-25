# 🏥 器官健康评分功能 - 快速开始

## 🎯 功能概述

基于 DNA 甲基化预测的蛋白质生物标志物，评估 **6大器官系统** 的健康状态：

| 器官系统 | 蛋白质数量 | 评估内容 |
|---------|-----------|---------|
| ❤️ **心脏** | 13种 | 心血管健康、内皮功能、炎症、凝血 |
| 🫘 **肾脏** | 7种 | 肾功能、炎症状态 |
| 🫀 **肝脏** | 8种 | 肝脏合成功能、纤维化风险 |
| 🛡️ **免疫系统** | 8种 | 免疫激活、炎症状态 |
| ⚡ **代谢系统** | 7种 | 代谢健康、能量平衡 |
| 🩸 **血管系统** | 9种 | 内皮功能、血管健康 |

---

## 🚀 快速开始

### 1. 运行预测

```bash
cd examples
python 935k_enhanced_prediction.py
```

### 2. 查看结果

```bash
# 器官健康评分CSV
cat results/935k_enhanced_predictions/organ_health_scores.csv

# PDF报告（包含雷达图）
open results/935k_enhanced_predictions/comprehensive_report.pdf
```

---

## 📊 输出文件

### organ_health_scores.csv

包含以下列：

```csv
sample_id,
heart_score, heart_level, heart_protein_avg,
kidney_score, kidney_level, kidney_protein_avg,
liver_score, liver_level, liver_protein_avg,
immune_score, immune_level, immune_protein_avg,
metabolic_score, metabolic_level, metabolic_protein_avg,
vascular_score, vascular_level, vascular_protein_avg,
overall_health_score, overall_health_level
```

**示例**：
```csv
sample_id,heart_score,heart_level,kidney_score,kidney_level,...,overall_health_score,overall_health_level
Sample1,85.3,良好,78.2,良好,...,81.5,良好
Sample2,92.1,优秀,88.7,良好,...,89.3,良好
```

---

## 📈 评分解读

### 评分范围：0-100

| 分数 | 等级 | 含义 | 建议 |
|------|------|------|------|
| **90-100** | 优秀 | 器官功能最佳 | 保持健康生活方式 |
| **75-89** | 良好 | 器官健康良好 | 继续维持，定期监测 |
| **60-74** | 一般 | 需要适度关注 | 改善生活方式，定期检查 |
| **40-59** | 较差 | 需要重点关注 | 建议医学检查和干预 |
| **0-39** | 差 | 需要紧急关注 | 强烈建议就医检查 |

---

## 🔬 科学依据

### 最新研究支持

1. **Nature 2023**: "Organ aging signatures in the plasma proteome track health and disease"
   - 血浆蛋白质可追踪11个器官的衰老
   - 器官特异性蛋白质与器官功能高度相关

2. **Lancet Digital Health 2025**: "Proteomic organ-specific ageing signatures"
   - 器官特异性生物学年龄可预测疾病风险
   - 血液蛋白质可反映器官健康状态

### 关键蛋白质标志物

#### 心脏 (Heart)
- **炎症**: CRP, IL6, TNF-α
- **内皮功能**: ICAM1, VCAM1, E-selectin, P-selectin
- **凝血**: Fibrinogen, vWF, D-dimer, PAI1
- **重塑**: MMP1, MMP9
- **调节**: ADM

#### 肾脏 (Kidney)
- **功能**: Cystatin C (金标准), B2M
- **炎症**: CRP, IL6, TNF-α
- **血管**: VEGF
- **纤维化**: PAI1

#### 肝脏 (Liver)
- **合成**: CRP, Fibrinogen
- **纤维化**: PAI1, MMP1, MMP9
- **炎症**: IL6, TNF-α
- **应激**: GDF15

---

## 📄 PDF报告内容

### 新增章节：器官健康评分

1. **器官健康评分表格**
   - 所有器官系统的平均评分
   - 健康等级分类
   - 中英文双语

2. **器官健康雷达图** 🎯
   - 6个器官系统的可视化展示
   - 参考线（优秀90、良好75、一般60、较差40）
   - 直观对比各器官健康状态

3. **评分说明**
   - 详细的解读指南
   - 评分计算方法
   - 注意事项

---

## 🧪 测试功能

运行测试脚本验证功能：

```bash
python test_organ_health_scores.py
```

这将：
- 创建模拟蛋白质数据
- 计算器官健康评分
- 显示详细结果
- 保存测试输出

---

## 📚 详细文档

- **ORGAN_HEALTH_SCORES_GUIDE.md** - 详细的器官健康评分指南
- **ORGAN_HEALTH_UPDATE_SUMMARY.md** - 功能更新总结
- **ENHANCED_PREDICTION_README.md** - 完整使用文档

---

## ⚠️ 重要提示

1. **辅助工具**：器官健康评分基于生物标志物预测，仅供参考，不能替代医学诊断
2. **数据依赖**：需要运行蛋白质预测（PREDICT_PROTEINS = True）
3. **可选依赖**：需要安装 matplotlib 和 reportlab 生成雷达图和PDF
4. **个体差异**：评分受年龄、性别、生活方式等多种因素影响
5. **动态监测**：建议定期检测，观察趋势变化
6. **医学咨询**：如评分异常，请咨询专业医生

---

## 💡 使用场景

- ✅ 健康体检和预防医学
- ✅ 衰老研究和抗衰老干预
- ✅ 疾病风险评估
- ✅ 健康管理和监测
- ✅ 生活方式干预效果评估

---

## 🎨 可视化示例

### 器官健康雷达图

雷达图展示6个器官系统的健康评分，可以直观地看到：
- 哪些器官系统健康状态良好
- 哪些器官系统需要关注
- 整体健康平衡状况

---

## 📧 支持

如有问题或需要更多信息，请参考：
- 主项目文档
- ORGAN_HEALTH_SCORES_GUIDE.md
- ENHANCED_PREDICTION_README.md

