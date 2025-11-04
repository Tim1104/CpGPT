# CpGPT Web 应用更新日志

## 版本 2.1 - 2024-11-04

### 🚀 GPU兼容性增强

#### ✨ 新增功能

1. **跨平台GPU支持**
   - ✅ **NVIDIA CUDA**: 完整支持，使用16-bit混合精度
   - ✅ **Apple Silicon MPS**: 支持M1/M2/M3系列，使用32-bit精度
   - ✅ **CPU Fallback**: 自动降级到CPU（无GPU时）

2. **智能设备检测**
   - 🔍 自动检测最佳可用设备（CUDA > MPS > CPU）
   - 📊 启动时显示详细设备信息
   - ⚙️ 根据设备类型自动优化精度和性能

3. **新增GPU工具模块** (`webapp/gpu_utils.py`)
   - 统一的设备检测和管理
   - 设备特定的优化设置
   - MPS兼容性检查和警告

#### 🔧 改进

1. **动态精度选择**
   - CUDA: 16-bit混合精度（最快）
   - MPS: 32-bit精度（稳定性优先）
   - CPU: 32-bit精度

2. **健康检查增强**
   - 显示设备类型和名称
   - 显示平台信息（macOS/Linux/Windows）
   - 显示GPU内存使用（CUDA）

3. **启动脚本改进**
   - 检测CUDA和MPS
   - 显示PyTorch版本
   - 显示推荐精度设置

#### 📚 文档更新

- README: 添加GPU支持详细说明
- QUICKSTART: 添加GPU检测信息
- 所有预测函数: 使用动态精度设置

---

## 版本 2.0 - 2024-11-04

### 🎉 重大更新：新增表观遗传时钟和蛋白质分析

#### ✨ 新增功能

1. **表观遗传时钟预测** (`clock_proxies` 模型)
   - ⏰ **AltumAge**: 多组织生物学年龄估计
   - 📈 **DunedinPACE**: 衰老速度指标（×100）
   - 💀 **GrimAge2**: 与死亡率相关的表观遗传年龄
   - 🏥 **HRS InCHPhenoAge**: 基于健康和退休研究的表型年龄
   - 🕐 **PC Horvath 2013**: 经典多组织表观遗传时钟

2. **血浆蛋白质水平预测** (`proteins` 模型)
   - 🧬 预测血浆蛋白质标准化水平
   - 📊 可用于GrimAge3等高级表观遗传时钟计算
   - 🔬 提供蛋白质水平热图和分布分析

#### 📊 新增可视化

1. **表观遗传时钟分布图**
   - 5种时钟的直方图分布
   - 均值和中位数标注
   - 统计卡片展示

2. **蛋白质水平热图**
   - 样本×蛋白质热图
   - 蛋白质水平箱线图
   - 标准化水平可视化

#### 🔧 技术改进

1. **后端增强** (`webapp/app.py`)
   - 新增 `predict_clocks()` 函数
   - 新增 `predict_proteins()` 函数
   - 优化进度追踪（40% → 50% → 60% → 70% → 80%）
   - 改进结果合并逻辑

2. **报告生成器增强** (`webapp/report_generator.py`)
   - 新增 `create_clocks_distribution_plot()` 函数
   - 新增 `create_proteins_heatmap()` 函数
   - 扩展 HTML 报告，包含时钟和蛋白质部分
   - 添加详细的时钟解读说明

3. **文档更新**
   - 更新 README.md 功能列表
   - 更新 QUICKSTART.md 模型下载说明
   - 更新 start_server.sh 自动下载脚本

#### 📈 分析流程更新

**旧流程（v1.0）:**
```
上传 → 数据处理 → 年龄预测 → 癌症预测 → 生成报告
```

**新流程（v2.0）:**
```
上传 → 数据处理 → 年龄预测 → 癌症预测 → 
表观遗传时钟预测 → 蛋白质预测 → 生成报告
```

#### 📊 报告内容对比

| 分析项目 | v1.0 | v2.0 |
|---------|------|------|
| 年龄预测 | ✅ | ✅ |
| 癌症预测 | ✅ | ✅ |
| 表观遗传时钟 | ❌ | ✅ (5种) |
| 蛋白质水平 | ❌ | ✅ |
| 可视化图表 | 4个 | 6个 |

#### 🎯 使用的模型

| 模型名称 | 功能 | 输出 |
|---------|------|------|
| `age_cot` | 年龄预测 | 年龄（岁） |
| `cancer` | 癌症预测 | 癌症概率 |
| `clock_proxies` | 表观遗传时钟 | 5种时钟值 |
| `proteins` | 血浆蛋白 | 蛋白质水平 |

#### 📦 依赖更新

需要下载的模型从 2 个增加到 4 个：
- `age_cot` (已有)
- `cancer` (已有)
- `clock_proxies` (新增)
- `proteins` (新增)

#### 🚀 升级指南

如果您已经安装了 v1.0，请按以下步骤升级：

1. **下载新模型**
```bash
python -c "
from cpgpt.infer.cpgpt_inferencer import CpGPTInferencer
inferencer = CpGPTInferencer(dependencies_dir='./dependencies')
inferencer.download_model('clock_proxies')
inferencer.download_model('proteins')
"
```

2. **更新代码**
```bash
git pull origin main
```

3. **重启服务器**
```bash
bash webapp/start_server.sh
```

#### ⚠️ 注意事项

1. **存储空间**: 新模型需要额外约 2GB 存储空间
2. **分析时间**: 完整分析时间增加约 30-50%
3. **内存需求**: 建议至少 16GB RAM
4. **兼容性**: 与 v1.0 生成的数据不兼容

#### 🐛 已知问题

- 蛋白质名称使用通用命名（protein_1, protein_2...），未来版本将添加具体名称
- 某些时钟值可能需要进一步校准

#### 📚 参考资料

- **表观遗传时钟**: 
  - Horvath S. (2013). DNA methylation age of human tissues and cell types.
  - Belsky DW et al. (2022). DunedinPACE, a DNA methylation biomarker of the pace of aging.
  - Lu AT et al. (2019). DNA methylation GrimAge strongly predicts lifespan and healthspan.

- **蛋白质预测**:
  - Hillary RF et al. (2020). Epigenetic measures of ageing predict the prevalence and incidence of leading causes of death and disease burden.

---

## 版本 1.0 - 2024-11-03

### 初始版本

#### ✨ 功能

- 文件上传（CSV/Arrow）
- 年龄预测
- 癌症预测
- HTML 报告生成
- PDF 导出
- 实时进度追踪

#### 📊 可视化

- 年龄分布图
- 癌症分布图
- 年龄-癌症相关性图
- 综合统计图

---

**完整文档**: 请参阅 README.md 和 QUICKSTART.md

