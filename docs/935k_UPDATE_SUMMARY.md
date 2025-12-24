# 935k/EPICv2 支持更新说明

## 📅 更新日期
2024-12-24

## 🎯 更新目的

经过与935k厂商沟通确认，**935k 芯片就是 GPL33022 (EPICv2) 平台**。因此，我们无需添加新的平台支持，而是直接使用 CpGPT 现有的 EPICv2 支持来运行 935k 数据的所有预测功能。

## ✨ 主要更新内容

### 1. 新增文档

#### 📄 `docs/935k_EPICv2_QUICKSTART.md`
- 935k/EPICv2 快速使用指南（英文）
- 详细说明为什么 935k 可以直接使用
- 完整的使用步骤和代码示例
- 所有可用预测模型的列表
- 常见问题解答

#### 📄 `docs/935k_README_CN.md`
- 935k 使用指南（中文）
- 简化的3步快速开始流程
- 输出结果说明
- Web 界面使用说明
- 技术原理解释

#### 📄 `docs/935k_UPDATE_SUMMARY.md`（本文档）
- 更新内容总结
- 文件清单
- 使用建议

### 2. 新增示例脚本

#### 🐍 `examples/935k_simple_prediction.py`
一个简化的预测脚本，专门用于 935k 数据的快速预测。

**特点**：
- 简洁易用，配置清晰
- 支持所有预测功能的开关
- 自动处理 CSV 到 Arrow 的转换
- 详细的进度提示（中英文）
- 自动合并所有预测结果

**支持的预测**：
1. 多组织器官年龄预测 (`age_cot`)
2. 癌症预测 (`cancer`)
3. 五种表观遗传时钟 (`clock_proxies`)
4. 血浆蛋白质预测 (`proteins`)

**使用方法**：
```bash
# 1. 编辑脚本中的数据路径
# RAW_DATA_PATH = "./data/你的数据.csv"

# 2. 选择要运行的预测
# PREDICT_AGE = True
# PREDICT_CANCER = True
# PREDICT_CLOCKS = True
# PREDICT_PROTEINS = True

# 3. 运行
python examples/935k_simple_prediction.py
```

### 3. 更新主 README

#### 📝 `README.md`
- 在目录中添加了 "935k/EPICv2 Platform Support" 章节
- 在 Overview 后添加了 935k 快速开始说明
- 提供了三个文档的链接：
  - 快速开始指南
  - 简化预测脚本
  - 完整功能脚本（带可视化）

## 📂 文件清单

### 新增文件
```
docs/
├── 935k_EPICv2_QUICKSTART.md      # 英文快速指南
├── 935k_README_CN.md              # 中文使用指南
└── 935k_UPDATE_SUMMARY.md         # 本文档

examples/
└── 935k_simple_prediction.py      # 简化预测脚本
```

### 修改文件
```
README.md                           # 添加 935k 支持说明
```

### 保留的现有文件
```
examples/
├── 935k_zero_shot_inference.py    # 完整功能脚本（带可视化）
└── validate_935k_data.py          # 数据验证脚本

docs/
├── 935k_data_format_guide.md      # 数据格式指南
├── 935k_platform_preparation_guide.md  # 平台准备指南（现已不需要）
└── 935k_zero_shot_inference_guide.md   # 零样本推理指南

webapp/
└── app.py                          # Web 应用（支持 935k）
```

## 🎯 使用建议

### 对于新用户

**推荐使用**：`examples/935k_simple_prediction.py`

**原因**：
- 代码简洁，易于理解
- 配置清晰，容易修改
- 自动处理数据格式转换
- 包含所有核心预测功能

**步骤**：
1. 阅读 `docs/935k_README_CN.md`（中文）或 `docs/935k_EPICv2_QUICKSTART.md`（英文）
2. 准备 CSV 格式的 935k 数据
3. 修改脚本中的数据路径
4. 运行脚本

### 对于需要可视化的用户

**推荐使用**：`examples/935k_zero_shot_inference.py`

**原因**：
- 包含完整的可视化功能
- 生成 HTML 分析报告
- 数据质量检查
- 异常值检测

### 对于喜欢图形界面的用户

**推荐使用**：Web 应用

**步骤**：
```bash
cd webapp
python app.py
# 在浏览器中打开 http://localhost:8000
```

## 🔧 技术说明

### 为什么不需要修改代码？

1. **平台已支持**: CpGPT 已经包含 EPICv2 (GPL33022) 的完整支持
2. **Manifest 已存在**: EPICv2 的 manifest 文件已在依赖中
3. **模型已训练**: 所有预训练模型都支持 EPICv2 平台
4. **探针映射完整**: 探针ID到基因组位置的映射已完成

### 数据流程

```
用户的 935k CSV 数据
    ↓
自动识别为 EPICv2 平台
    ↓
使用现有的 EPICv2 manifest
    ↓
探针ID → 基因组位置
    ↓
基因组位置 → DNA 嵌入
    ↓
CpGPT 模型推理
    ↓
输出预测结果
```

## 📊 支持的预测功能

| 功能 | 模型名称 | 输出 | 用途 |
|------|---------|------|------|
| 多组织年龄预测 | `age_cot` | 年龄（岁） | 生物学年龄评估 |
| 癌症预测 | `cancer` | 概率 (0-1) | 癌症检测 |
| 表观遗传时钟 | `clock_proxies` | 5个时钟值 | 衰老评估 |
| 蛋白质预测 | `proteins` | 标准化值 | 死亡率风险 |

### 五种表观遗传时钟详解

1. **altumage**: 年龄预测时钟
2. **dunedinpace**: 衰老速度（正常值约100，>100表示衰老加速）
3. **grimage2**: GrimAge2 死亡率预测时钟
4. **hrsinchphenoage**: PhenoAge 表型年龄时钟
5. **pchorvath2013**: Horvath 2013 经典时钟

## 🚀 下一步

### 立即开始使用

```bash
# 1. 克隆仓库（如果还没有）
git clone https://github.com/lcamillo/CpGPT.git
cd CpGPT

# 2. 安装依赖
poetry install
poetry shell

# 3. 运行简化脚本
python examples/935k_simple_prediction.py
```

### 获取帮助

- 📖 阅读文档：`docs/935k_README_CN.md`
- 💬 查看示例：`examples/935k_simple_prediction.py`
- 🌐 使用 Web 界面：`webapp/app.py`
- 📧 联系作者：lucas_camillo@alumni.brown.edu

## ✅ 总结

通过这次更新，我们：

1. ✅ 确认了 935k 就是 EPICv2 平台
2. ✅ 创建了简化的使用文档和脚本
3. ✅ 提供了中英文使用指南
4. ✅ 支持所有核心预测功能
5. ✅ 无需修改任何核心代码

用户现在可以直接使用 CpGPT 对 935k 数据进行：
- 多组织器官年龄预测
- 癌症预测
- 五种表观遗传时钟
- 血浆蛋白质预测

**一切都已准备就绪，开始使用吧！** 🎉

