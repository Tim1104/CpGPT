# CpGPT Web 应用快速入门指南

## 🚀 5分钟快速开始

### 步骤 1: 安装依赖

```bash
# 确保在CpGPT项目根目录
cd /path/to/CpGPT

# 安装Web应用依赖
pip install -r webapp/requirements.txt
```

### 步骤 2: 下载模型

```bash
# 方法1: 使用Python脚本
python -c "
from cpgpt.infer.cpgpt_inferencer import CpGPTInferencer
inferencer = CpGPTInferencer(dependencies_dir='./dependencies')
print('Downloading dependencies...')
inferencer.download_dependencies()
print('Downloading age_cot model...')
inferencer.download_model('age_cot')
print('Downloading cancer model...')
inferencer.download_model('cancer')
print('Downloading clock_proxies model...')
inferencer.download_model('clock_proxies')
print('Downloading proteins model...')
inferencer.download_model('proteins')
print('Done!')
"

# 方法2: 运行示例脚本（会自动下载）
python examples/935k_zero_shot_inference.py
```

### 步骤 3: 启动服务器

```bash
# 使用启动脚本（推荐）
bash webapp/start_server.sh

# 或手动启动
cd webapp
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

**启动时会自动检测GPU:**
- ✅ **NVIDIA GPU**: 使用CUDA加速 + 16-bit混合精度
- ✅ **Apple Silicon**: 使用MPS加速 + 32-bit精度
- ⚠️ **CPU Only**: 使用CPU（较慢）

### 步骤 4: 访问应用

打开浏览器访问: **http://localhost:8000**

## 📝 使用流程

### 1. 准备数据

确保您的数据符合以下格式：

**CSV格式示例:**
```csv
sample_id,species,cg00000029,cg00000108,cg00000109,...
Sample1,homo_sapiens,0.123,0.456,0.789,...
Sample2,homo_sapiens,0.234,0.567,0.890,...
```

**必需要求:**
- ✅ 包含 `species` 列（值为 "homo_sapiens"）
- ✅ CpG位点列名格式为 `cgXXXXXXXX`
- ✅ Beta值范围: 0-1
- ✅ 文件大小 < 500MB

### 2. 上传文件

1. 在Web界面点击"选择文件"或拖拽文件到上传区域
2. 支持格式: `.csv`, `.arrow`, `.feather`
3. 点击"开始分析"

### 3. 等待分析

- 系统会显示实时进度
- 分析时间取决于样本数量（通常5-30分钟）
- 进度条会显示当前步骤

### 4. 查看报告

分析完成后：
- 点击"查看报告"在浏览器中查看HTML报告
- 点击"下载PDF"保存PDF版本
- 报告包含年龄预测、癌症风险评估和可视化图表

## 🧪 测试应用

```bash
# 基础测试（不上传文件）
python webapp/test_webapp.py

# 完整测试（包含文件上传）
python webapp/test_webapp.py /path/to/your/test_data.csv
```

## 🔧 常见问题

### Q: 服务器启动失败

**A:** 检查以下几点：
```bash
# 1. 检查端口是否被占用
lsof -i :8000

# 2. 检查Python版本
python --version  # 需要 3.8+

# 3. 检查依赖是否安装
pip list | grep fastapi
```

### Q: 模型下载失败

**A:** 尝试以下方法：
```bash
# 1. 检查网络连接
ping s3.amazonaws.com

# 2. 使用代理（如果需要）
export HTTP_PROXY=http://your-proxy:port
export HTTPS_PROXY=http://your-proxy:port

# 3. 手动下载模型文件到 dependencies/model/ 目录
```

### Q: GPU不可用

**A:** 
- GPU不是必需的，CPU也可以运行（速度较慢）
- 检查CUDA安装: `nvidia-smi`
- 检查PyTorch GPU支持: `python -c "import torch; print(torch.cuda.is_available())"`

### Q: 分析失败

**A:** 检查日志文件：
```bash
# 查看最新日志
tail -f webapp/logs/cpgpt_web_*.log

# 常见问题：
# - 数据格式不正确（缺少species列）
# - CpG位点ID格式错误
# - Beta值超出0-1范围
# - 内存不足
```

### Q: PDF生成失败

**A:** 安装PDF生成工具：
```bash
# 方法1: 安装weasyprint（推荐）
pip install weasyprint

# 方法2: 安装wkhtmltopdf
# macOS
brew install wkhtmltopdf

# Ubuntu/Debian
sudo apt-get install wkhtmltopdf
```

## 📊 API使用示例

### Python示例

```python
import requests
import time

# 上传文件
with open('your_data.csv', 'rb') as f:
    files = {'file': ('data.csv', f)}
    response = requests.post('http://localhost:8000/api/upload', files=files)
    task_id = response.json()['task_id']

# 监控进度
while True:
    response = requests.get(f'http://localhost:8000/api/task/{task_id}')
    task = response.json()
    print(f"Progress: {task['progress']}% - {task['message']}")
    
    if task['status'] == 'completed':
        print(f"Report URL: {task['report_url']}")
        break
    elif task['status'] == 'failed':
        print(f"Error: {task['error']}")
        break
    
    time.sleep(2)

# 下载PDF
response = requests.get(f'http://localhost:8000/api/download/{task_id}/pdf')
with open('report.pdf', 'wb') as f:
    f.write(response.content)
```

### cURL示例

```bash
# 上传文件
curl -X POST "http://localhost:8000/api/upload" \
  -F "file=@your_data.csv"

# 获取任务状态
curl "http://localhost:8000/api/task/{task_id}"

# 下载PDF
curl "http://localhost:8000/api/download/{task_id}/pdf" \
  -o report.pdf
```

## 🎯 性能优化建议

### 1. 使用GPU加速
```bash
# 确保CUDA可用
export CUDA_VISIBLE_DEVICES=0
```

### 2. 调整批处理大小
在 `webapp/app.py` 中修改：
```python
# 如果内存充足，可以增加batch_size
batch_size=2  # 默认为1
```

### 3. 减少样本数量
- 对于初步测试，建议使用10-100个样本
- 大规模分析（>1000样本）可能需要较长时间

### 4. 使用Arrow格式
- Arrow格式比CSV加载更快
- 预先转换数据：
```python
import pandas as pd
df = pd.read_csv('data.csv')
df.to_feather('data.arrow')
```

## 📚 更多资源

- **完整文档**: `webapp/README.md`
- **API文档**: http://localhost:8000/docs
- **示例脚本**: `examples/935k_zero_shot_inference.py`
- **CpGPT论文**: https://www.biorxiv.org/content/10.1101/2024.10.24.619766v1

## 🆘 获取帮助

如果遇到问题：

1. **查看日志**: `webapp/logs/cpgpt_web_*.log`
2. **运行测试**: `python webapp/test_webapp.py`
3. **检查健康状态**: http://localhost:8000/health
4. **提交Issue**: GitHub Issues

## 📄 许可证

本项目遵循CpGPT主项目的许可证。

---

**祝您使用愉快！** 🎉

如有问题，请查看完整文档或提交Issue。

