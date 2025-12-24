# 935k/EPICv2 资源索引

## 📚 文档资源

### 🌟 推荐入门文档

| 文档 | 语言 | 适合人群 | 描述 |
|------|------|---------|------|
| [935k_README_CN.md](935k_README_CN.md) | 中文 | 新手 | 最简单的入门指南，3步开始使用 |
| [935k_EPICv2_QUICKSTART.md](935k_EPICv2_QUICKSTART.md) | 英文 | 新手 | 完整的快速开始指南，包含技术细节 |
| [935k_QUICKSTART_EXAMPLE.md](../examples/935k_QUICKSTART_EXAMPLE.md) | 中文 | 新手 | 实用示例和代码片段 |

### 📖 详细文档

| 文档 | 用途 | 描述 |
|------|------|------|
| [935k_UPDATE_SUMMARY.md](935k_UPDATE_SUMMARY.md) | 了解更新 | 本次更新的完整说明 |
| [935k_data_format_guide.md](935k_data_format_guide.md) | 数据准备 | 数据格式要求和转换方法 |
| [935k_zero_shot_inference_guide.md](935k_zero_shot_inference_guide.md) | 深入理解 | 零样本推理的原理和应用 |

### 🔧 技术文档（可选）

| 文档 | 用途 | 描述 |
|------|------|------|
| [935k_platform_preparation_guide.md](935k_platform_preparation_guide.md) | 参考 | 平台准备指南（现已不需要） |
| [935k_CSV_SUPPORT_README.md](935k_CSV_SUPPORT_README.md) | 参考 | CSV 格式支持说明 |

## 💻 代码资源

### 🌟 推荐脚本

| 脚本 | 难度 | 功能 | 适合场景 |
|------|------|------|---------|
| [935k_simple_prediction.py](../examples/935k_simple_prediction.py) | ⭐ 简单 | 所有核心预测 | 快速开始，日常使用 |
| [935k_zero_shot_inference.py](../examples/935k_zero_shot_inference.py) | ⭐⭐ 中等 | 预测+可视化+报告 | 需要详细分析和图表 |

### 🛠️ 辅助工具

| 工具 | 用途 | 描述 |
|------|------|------|
| [validate_935k_data.py](../examples/validate_935k_data.py) | 数据验证 | 检查数据格式是否正确 |
| [visualize_results.py](../examples/visualize_results.py) | 结果可视化 | 生成图表和报告 |

### 🌐 Web 应用

| 应用 | 特点 | 适合人群 |
|------|------|---------|
| [webapp/app.py](../webapp/app.py) | 图形界面，无需编程 | 不熟悉编程的用户 |

## 🎯 使用场景指南

### 场景 1: 我是新手，想快速开始

**推荐路径**：
1. 📖 阅读 [935k_README_CN.md](935k_README_CN.md)
2. 💻 使用 [935k_simple_prediction.py](../examples/935k_simple_prediction.py)
3. 📊 查看结果文件

**预计时间**: 30分钟

### 场景 2: 我需要详细的分析报告

**推荐路径**：
1. 📖 阅读 [935k_QUICKSTART_EXAMPLE.md](../examples/935k_QUICKSTART_EXAMPLE.md)
2. 💻 使用 [935k_zero_shot_inference.py](../examples/935k_zero_shot_inference.py)
3. 📊 查看 HTML 报告和图表

**预计时间**: 1小时

### 场景 3: 我不会编程，想用图形界面

**推荐路径**：
1. 📖 阅读 [935k_README_CN.md](935k_README_CN.md) 的 "Web 界面" 部分
2. 🌐 启动 [webapp/app.py](../webapp/app.py)
3. 📤 上传数据，点击分析

**预计时间**: 15分钟

### 场景 4: 我想深入了解技术原理

**推荐路径**：
1. 📖 阅读 [935k_EPICv2_QUICKSTART.md](935k_EPICv2_QUICKSTART.md)
2. 📖 阅读 [935k_zero_shot_inference_guide.md](935k_zero_shot_inference_guide.md)
3. 💻 研究 [935k_zero_shot_inference.py](../examples/935k_zero_shot_inference.py) 源码

**预计时间**: 2-3小时

### 场景 5: 我想自定义预测流程

**推荐路径**：
1. 📖 阅读 [935k_QUICKSTART_EXAMPLE.md](../examples/935k_QUICKSTART_EXAMPLE.md) 的 "方法三"
2. 💻 参考 [935k_simple_prediction.py](../examples/935k_simple_prediction.py)
3. 🔧 修改代码以满足需求

**预计时间**: 1-2小时

## 🚀 快速开始（3步）

### 步骤 1: 选择方法

- **简单快速**: 使用 `935k_simple_prediction.py`
- **详细分析**: 使用 `935k_zero_shot_inference.py`
- **图形界面**: 使用 `webapp/app.py`

### 步骤 2: 准备数据

确保数据是 CSV 格式：
```csv
sample_id,cg00000029,cg00000108,...
sample1,0.85,0.23,...
sample2,0.91,0.19,...
```

### 步骤 3: 运行预测

```bash
# 方法 1: 简单脚本
python examples/935k_simple_prediction.py

# 方法 2: 完整脚本
python examples/935k_zero_shot_inference.py

# 方法 3: Web 界面
cd webapp && python app.py
```

## 📊 支持的预测功能

| 功能 | 模型 | 输出 | 文档 |
|------|------|------|------|
| 多组织年龄预测 | `age_cot` | 年龄（岁） | 所有文档 |
| 癌症预测 | `cancer` | 概率 (0-1) | 所有文档 |
| 表观遗传时钟 | `clock_proxies` | 5个时钟值 | 所有文档 |
| 蛋白质预测 | `proteins` | 标准化值 | 所有文档 |

### 五种表观遗传时钟

1. **altumage** - 年龄预测
2. **dunedinpace** - 衰老速度（正常值约100）
3. **grimage2** - GrimAge2 死亡率预测
4. **hrsinchphenoage** - PhenoAge 表型年龄
5. **pchorvath2013** - Horvath 2013 经典时钟

## 🔍 常见问题快速查找

| 问题 | 查看文档 | 章节 |
|------|---------|------|
| 如何安装？ | [935k_README_CN.md](935k_README_CN.md) | 快速开始 |
| 数据格式要求？ | [935k_data_format_guide.md](935k_data_format_guide.md) | 全文 |
| 内存不足？ | [935k_README_CN.md](935k_README_CN.md) | 常见问题 Q3 |
| 如何解读结果？ | [935k_QUICKSTART_EXAMPLE.md](../examples/935k_QUICKSTART_EXAMPLE.md) | 结果解读 |
| 为什么可以直接使用？ | [935k_EPICv2_QUICKSTART.md](935k_EPICv2_QUICKSTART.md) | 技术说明 |
| 预测准确吗？ | [935k_README_CN.md](935k_README_CN.md) | 常见问题 Q2 |

## 📁 文件结构

```
CpGPT/
├── docs/
│   ├── 935k_README_CN.md                    # ⭐ 中文入门指南
│   ├── 935k_EPICv2_QUICKSTART.md            # ⭐ 英文快速指南
│   ├── 935k_UPDATE_SUMMARY.md               # 更新说明
│   ├── 935k_RESOURCES_INDEX.md              # 本文档
│   ├── 935k_data_format_guide.md            # 数据格式指南
│   ├── 935k_zero_shot_inference_guide.md    # 零样本推理指南
│   └── 935k_platform_preparation_guide.md   # 平台准备（已不需要）
│
├── examples/
│   ├── 935k_simple_prediction.py            # ⭐ 简化预测脚本
│   ├── 935k_zero_shot_inference.py          # ⭐ 完整功能脚本
│   ├── 935k_QUICKSTART_EXAMPLE.md           # ⭐ 快速示例
│   ├── validate_935k_data.py                # 数据验证工具
│   └── visualize_results.py                 # 可视化工具
│
└── webapp/
    └── app.py                                # ⭐ Web 应用
```

## 🆘 获取帮助

### 文档帮助
- 📖 查看相关文档（见上方索引）
- 💡 查看示例代码
- 📝 查看常见问题

### 技术支持
- 📧 邮件：lucas_camillo@alumni.brown.edu
- 📄 论文：[CpGPT bioRxiv](https://www.biorxiv.org/content/10.1101/2024.10.24.619766v1)
- 💻 GitHub：[CpGPT Repository](https://github.com/lcamillo/CpGPT)

## ✅ 开始前检查清单

- [ ] 已阅读 [935k_README_CN.md](935k_README_CN.md) 或 [935k_EPICv2_QUICKSTART.md](935k_EPICv2_QUICKSTART.md)
- [ ] 已安装 CpGPT (`poetry install`)
- [ ] 已配置 AWS CLI
- [ ] 数据格式正确（CSV，第一列为样本ID）
- [ ] 已选择合适的脚本或工具
- [ ] 已修改脚本中的数据路径

**准备好了？开始使用 CpGPT 分析您的 935k 数据吧！** 🎉

---

**最后更新**: 2024-12-24
**版本**: 1.0

