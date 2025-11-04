# CpGPT 935k 甲基化数据分析 Web 应用

这是一个基于 FastAPI 和 CpGPT 模型的 Web 应用，提供友好的界面用于上传 935k 甲基化数据并生成分析报告。

## 🌟 功能特性

- **📁 文件上传**: 支持 CSV 和 Arrow/Feather 格式的 935k 甲基化数据
- **🔬 智能分析**: 使用 CpGPT 预训练模型进行零样本推理
  - 年龄预测 (Age Prediction)
  - 癌症风险评估 (Cancer Risk Assessment)
  - 表观遗传时钟 (Epigenetic Clocks) - 5种时钟代理
  - 血浆蛋白质水平 (Plasma Proteins)
- **📊 可视化报告**: 自动生成包含多种图表的 HTML 报告
  - 年龄分布分析
  - 癌症概率分布
  - 年龄与癌症相关性分析
  - 表观遗传时钟分布
  - 蛋白质水平热图
  - 风险分层统计
- **📄 PDF 导出**: 一键下载 PDF 格式的分析报告
- **⚡ 实时进度**: 实时显示分析进度和状态

## 📋 系统要求

### 必需
- Python 3.8 或更高版本
- 8GB+ RAM (推荐 16GB+)
- 10GB+ 可用磁盘空间

### GPU 支持（可选但推荐）

应用支持多种GPU加速方案：

#### ✅ NVIDIA GPU (CUDA)
- **支持**: CUDA 11.0+
- **精度**: 16-bit 混合精度
- **性能**: 最佳
- **推荐**: RTX 3060 或更高

#### ✅ Apple Silicon (M1/M2/M3)
- **支持**: macOS 12.3+ with PyTorch 2.0+
- **后端**: Metal Performance Shaders (MPS)
- **精度**: 32-bit (为稳定性)
- **性能**: 优秀
- **推荐**: M1 Pro/Max, M2 Pro/Max, M3 系列

#### ⚠️ CPU Only
- **性能**: 较慢（约为GPU的5-10倍时间）
- **适用**: 小规模测试或无GPU环境

### 其他推荐
- 16GB+ RAM
- SSD 存储

## 🚀 快速开始

### 1. 安装依赖

```bash
# 安装 Web 应用依赖
pip install -r webapp/requirements.txt

# 确保已安装 CpGPT 核心依赖
pip install -r requirements.txt
```

### 2. 下载模型和依赖

```bash
# 使用 Python 脚本下载
python -c "
from cpgpt.infer.cpgpt_inferencer import CpGPTInferencer
inferencer = CpGPTInferencer(dependencies_dir='./dependencies')
inferencer.download_dependencies()
inferencer.download_model('age_cot')
inferencer.download_model('cancer')
inferencer.download_model('clock_proxies')
inferencer.download_model('proteins')
"
```

或者运行示例脚本（会自动下载）：
```bash
python examples/935k_zero_shot_inference.py
```

### 3. 启动服务器

```bash
# 使用启动脚本（推荐）
bash webapp/start_server.sh

# 或手动启动
cd webapp
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### 4. 访问应用

打开浏览器访问：
- **主页**: http://localhost:8000
- **API 文档**: http://localhost:8000/docs
- **健康检查**: http://localhost:8000/health

## 📖 使用指南

### 数据格式要求

#### CSV 格式
```csv
sample_id,species,cg00000029,cg00000108,cg00000109,...
Sample1,homo_sapiens,0.123,0.456,0.789,...
Sample2,homo_sapiens,0.234,0.567,0.890,...
```

#### Arrow/Feather 格式
使用 pandas 转换：
```python
import pandas as pd
df = pd.read_csv('your_data.csv')
df.to_feather('your_data.arrow')
```

### 数据要求
- **必需列**: `species` (物种名称，如 "homo_sapiens")
- **CpG 列**: 列名为 CpG 位点 ID (如 `cg00000029`)
- **Beta 值**: 0-1 之间的甲基化值
- **样本数**: 建议 1-1000 个样本

### 使用流程

1. **上传文件**
   - 点击"选择文件"或拖拽文件到上传区域
   - 支持的格式: `.csv`, `.arrow`, `.feather`
   - 最大文件大小: 500MB

2. **开始分析**
   - 点击"开始分析"按钮
   - 系统会显示实时进度和状态信息
   - 分析时间取决于样本数量（通常 5-30 分钟）

3. **查看报告**
   - 分析完成后，点击"查看报告"在浏览器中查看
   - 报告包含：
     - 执行摘要
     - 年龄预测分析
     - 癌症风险评估
     - 可视化图表
     - 统计分析

4. **下载 PDF**
   - 点击"下载 PDF"按钮
   - 系统会自动生成并下载 PDF 格式的报告

## 🏗️ 架构说明

### 后端 (FastAPI)
- `webapp/app.py`: 主应用和 API 端点
- `webapp/report_generator.py`: HTML 报告和可视化生成
- `webapp/pdf_generator.py`: PDF 转换功能

### 前端
- `webapp/static/index.html`: 主页面
- `webapp/static/app.js`: 前端交互逻辑

### API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 主页 |
| `/api/upload` | POST | 上传文件并创建分析任务 |
| `/api/task/{task_id}` | GET | 获取任务状态 |
| `/api/tasks` | GET | 列出所有任务 |
| `/results/{task_id}/analysis_report.html` | GET | 获取 HTML 报告 |
| `/api/download/{task_id}/pdf` | GET | 下载 PDF 报告 |
| `/api/task/{task_id}` | DELETE | 删除任务 |
| `/health` | GET | 健康检查 |

## 🔧 配置选项

在 `webapp/app.py` 中可以修改以下配置：

```python
DEPENDENCIES_DIR = "./dependencies"  # 模型和依赖目录
UPLOAD_DIR = "./webapp/uploads"      # 上传文件目录
RESULTS_DIR = "./webapp/results"     # 结果存储目录
RANDOM_SEED = 42                     # 随机种子
MAX_INPUT_LENGTH = 30000             # 最大输入长度
```

## 📊 分析内容说明

### 年龄预测
- 使用 `age_cot` 模型（链式思维推理）
- 输出：预测年龄（岁）
- 基于 DNA 甲基化时钟原理

### 癌症预测
- 使用 `cancer` 模型
- 输出：
  - 癌症概率 (0-1)
  - 癌症预测 (0=正常, 1=癌症)
  - 癌症 logit 值

### 风险分层
- **低风险**: 概率 0.0-0.2
- **中低风险**: 概率 0.2-0.5
- **中高风险**: 概率 0.5-0.8
- **高风险**: 概率 0.8-1.0

## 🐛 故障排除

### 问题：GPU 不可用
**解决方案**:
- 检查 CUDA 安装: `nvidia-smi`
- 检查 PyTorch GPU 支持: `python -c "import torch; print(torch.cuda.is_available())"`
- CPU 模式仍可工作，但速度较慢

### 问题：模型下载失败
**解决方案**:
- 检查网络连接
- 手动下载模型到 `dependencies/model/` 目录
- 使用代理: `export HTTP_PROXY=...`

### 问题：PDF 生成失败
**解决方案**:
```bash
# 安装 weasyprint (推荐)
pip install weasyprint

# 或安装 wkhtmltopdf
# macOS
brew install wkhtmltopdf

# Ubuntu/Debian
sudo apt-get install wkhtmltopdf
```

### 问题：内存不足
**解决方案**:
- 减少批处理大小
- 减少样本数量
- 使用更小的模型 (`small` 而不是 `large`)

## 📝 注意事项

1. **数据隐私**: 上传的数据存储在本地服务器，请确保数据安全
2. **计算资源**: 大规模数据分析需要较多计算资源
3. **结果解读**: 预测结果仅供科研参考，不能作为临床诊断依据
4. **模型限制**: 零样本推理的准确性可能略低于微调模型

## 🔗 相关资源

- **CpGPT 论文**: [bioRxiv 2024.10.24.619766](https://www.biorxiv.org/content/10.1101/2024.10.24.619766v1)
- **GitHub**: [CpGPT Repository](https://github.com/yourusername/CpGPT)
- **文档**: 查看 `docs/` 目录获取更多信息

## 📄 许可证

本项目遵循 CpGPT 主项目的许可证。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📧 联系方式

如有问题或建议，请通过以下方式联系：
- 提交 GitHub Issue
- 发送邮件至项目维护者

---

**免责声明**: 本工具仅供科研使用，预测结果不能作为临床诊断依据。任何医疗决策应由专业医疗人员基于综合信息做出。

