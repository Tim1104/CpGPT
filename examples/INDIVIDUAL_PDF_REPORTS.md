# 个人PDF报告生成说明

## ✅ 已完成更新

现在脚本会**为每个样本生成独立的详细PDF报告**，而不是将所有样本合并在一个PDF中。

---

## 📄 PDF报告结构

每个样本的PDF报告包含以下6个章节：

### 第1章：执行摘要 (Executive Summary)
- 样本ID
- 预测年龄
- 癌症预测结果
- 癌症概率
- 死亡风险等级

### 第2章：多组织器官年龄预测 (Multi-Tissue Organ Age Prediction)
- **器官年龄表格**：11个器官的预测年龄和状态（加速/正常/减缓）
  - 脑、肝脏、心脏、肺、肾脏、肌肉、脂肪、血液、免疫、皮肤、骨骼
- **器官年龄雷达图**：可视化展示各器官年龄与预测年龄的对比

### 第3章：癌症预测 (Cancer Prediction)
- **癌症预测结果表格**：
  - 预测结果（阳性/阴性）
  - 癌症概率
  - 风险等级（低/中/高）
- **癌症概率可视化**：横向条形图展示阴性/阳性概率

### 第4章：五种表观遗传时钟 (Epigenetic Clocks)
- **时钟结果表格**：
  - Horvath Clock
  - Hannum Clock
  - PhenoAge
  - GrimAge
  - GrimAge2
  - 每个时钟的年龄加速值
- **时钟对比图**：柱状图对比5种时钟与预测年龄

### 第5章：血浆蛋白质预测 (Plasma Protein Prediction)
- **蛋白质统计**：预测的蛋白质总数
- **Top 10蛋白质表格**：按绝对值排序的前10个蛋白质
- **蛋白质分布图**：横向条形图展示Top 10蛋白质的预测值

### 第6章：器官健康评分 (Organ Health Scores)
- **器官健康评分表格**：
  - 心脏、肾脏、肝脏、免疫系统、代谢系统、血管系统
  - 每个系统的评分（0-100）和等级（优秀/良好/一般/较差/差）
- **器官健康雷达图**：
  - 6个器官系统的健康评分可视化
  - 参考线：优秀(90)、良好(75)、一般(60)、较差(40)

---

## 📁 输出文件

### PDF报告文件
```
results/935k_enhanced_predictions/
├── report_Sample1.pdf          # 样本1的完整报告
├── report_Sample2.pdf          # 样本2的完整报告
└── ...
```

### 图表文件（每个样本）
```
results/935k_enhanced_predictions/
├── organ_age_radar_Sample1.png           # 器官年龄雷达图
├── cancer_probability_Sample1.png        # 癌症概率图
├── clock_comparison_Sample1.png          # 时钟对比图
├── protein_distribution_Sample1.png      # 蛋白质分布图
├── organ_health_radar_Sample1.png        # 器官健康雷达图
└── ...（Sample2, Sample3等）
```

---

## 🚀 使用方法

### 运行脚本
```bash
cd examples
python 935k_enhanced_prediction.py
```

### 查看输出
```bash
# 查看生成的PDF报告
ls -lh results/935k_enhanced_predictions/report_*.pdf

# 打开某个样本的报告
open results/935k_enhanced_predictions/report_Sample1.pdf
```

---

## 📊 报告特点

### ✅ 优点
1. **独立报告**：每个样本一个PDF，便于分发和存档
2. **全面详细**：包含所有预测结果和可视化
3. **中英双语**：所有标题和标签都有中英文
4. **专业美观**：使用颜色编码和专业图表
5. **易于理解**：雷达图、柱状图、条形图等多种可视化

### 📈 可视化图表
- **雷达图**：器官年龄、器官健康评分
- **柱状图**：表观遗传时钟对比
- **横向条形图**：癌症概率、蛋白质分布

### 🎨 颜色编码
- **器官年龄**：红色 (#E74C3C)
- **癌症预测**：紫色 (#9B59B6)
- **表观遗传时钟**：蓝色 (#3498DB)
- **血浆蛋白质**：绿色 (#16A085)
- **器官健康**：橙色 (#E67E22)

---

## 🔍 示例输出

### 控制台输出
```
[7/7] 生成PDF报告...
  生成样本 Sample1 的PDF报告...
    ✓ PDF报告已生成: results/935k_enhanced_predictions/report_Sample1.pdf
  生成样本 Sample2 的PDF报告...
    ✓ PDF报告已生成: results/935k_enhanced_predictions/report_Sample2.pdf
```

---

## ⚠️ 注意事项

1. **数据依赖**：
   - 需要运行完整的预测流程（年龄、癌症、时钟、蛋白质、器官健康）
   - 如果某些数据缺失，对应章节会显示"No data available"

2. **图表生成**：
   - 雷达图至少需要3个数据点
   - 如果数据不足，会跳过图表生成

3. **中文字体**：
   - 如果系统没有中文字体，图表中文可能显示为方框
   - 建议安装中文字体（见 PDF_GENERATION_FIX.md）

4. **文件大小**：
   - 每个PDF约2-5MB（取决于图表数量）
   - 包含高分辨率图表（150 DPI）

---

## 🛠️ 自定义选项

如果需要修改PDF报告，可以编辑 `generate_individual_pdf_report()` 函数：

- **修改颜色**：搜索 `colors.HexColor` 修改颜色代码
- **调整图表大小**：修改 `figsize` 参数
- **添加新章节**：在构建PDF前添加新的 `story.append()` 语句
- **修改表格样式**：调整 `TableStyle` 参数

---

## 📚 相关文档

- **ENHANCED_PREDICTION_README.md** - 完整使用文档
- **ORGAN_HEALTH_SCORES_GUIDE.md** - 器官健康评分指南
- **PDF_GENERATION_FIX.md** - PDF生成问题修复

---

**现在每个样本都有自己的详细PDF报告了！** 🎉

