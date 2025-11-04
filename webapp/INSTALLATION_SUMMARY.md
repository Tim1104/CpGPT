# CpGPT Web应用 - 安装总结

## ✅ 已完成的工作

### 1. 依赖安装状态

#### ✅ 核心依赖（已安装）
- **FastAPI** 0.114.2 - Web框架
- **Uvicorn** 0.38.0 - ASGI服务器
- **python-multipart** - 文件上传支持
- **Matplotlib** 3.10.7 - 数据可视化
- **Seaborn** 0.13.2 - 统计可视化
- **Pandas** 2.3.3 - 数据处理
- **NumPy** 2.3.4 - 数值计算

#### ⚠️ 可选依赖（部分可用）
- **WeasyPrint** 66.0 - ⚠️ 已安装但缺少系统库
  - 状态：需要额外的系统依赖
  - 影响：PDF导出功能暂时不可用
  - 解决方案：见下文"PDF导出功能"部分

- **PyArrow** - ❌ 未安装
  - 状态：未安装（下载太慢）
  - 影响：不支持Arrow/Feather格式
  - 解决方案：只使用CSV格式，或稍后手动安装

### 2. GPU支持状态

#### ✅ 完全支持
- **Apple Silicon (MPS)** - ✅ 已检测并配置
  - 设备类型：mps
  - 设备名称：Apple Silicon (arm64)
  - 精度：32-bit（为稳定性优化）
  - PyTorch版本：2.6.0
  - 状态：完全正常工作

#### ✅ 跨平台兼容性
- NVIDIA CUDA - ✅ 代码已支持（自动检测）
- Apple MPS - ✅ 已测试并工作正常
- CPU Fallback - ✅ 自动降级支持

### 3. 功能状态

#### ✅ 完全可用
1. **文件上传** - CSV格式
2. **年龄预测** - age_cot模型
3. **癌症预测** - cancer模型
4. **表观遗传时钟** - clock_proxies模型（5种时钟）
5. **蛋白质预测** - proteins模型
6. **HTML报告生成** - 包含4种可视化图表
7. **GPU加速** - Apple Silicon MPS支持

#### ⚠️ 部分可用
8. **Arrow格式支持** - 需要安装PyArrow
9. **PDF导出** - 需要安装系统依赖

---

## 🚀 快速开始

### 方法1：直接启动（推荐）

```bash
# 启动服务器
bash webapp/start_server.sh

# 或者直接使用uvicorn
cd webapp
python3 -m uvicorn app:app --host 0.0.0.0 --port 8000
```

### 方法2：使用安装脚本

```bash
# 运行完整的安装和验证
bash webapp/install_dependencies.sh
```

### 访问应用

打开浏览器访问：http://localhost:8000

---

## 📋 当前限制和解决方案

### 限制1：只支持CSV格式

**原因**: PyArrow未安装（下载太慢）

**解决方案**:
```bash
# 稍后安装（可能需要几分钟）
pip3 install pyarrow

# 或使用国内镜像加速
pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple pyarrow
```

**临时方案**: 只上传CSV格式的935k数据文件

---

### 限制2：PDF导出不可用

**原因**: WeasyPrint缺少系统库（libgobject-2.0-0等）

**解决方案**:
```bash
# 1. 安装系统依赖（使用Homebrew）
brew install pango cairo gdk-pixbuf libffi glib

# 2. 重新安装weasyprint
pip3 install --force-reinstall weasyprint

# 3. 验证安装
python3 -c "from webapp.pdf_generator import generate_pdf_report; print('✅ PDF功能可用')"
```

**临时方案**: 
- HTML报告仍然可以正常查看
- 可以使用浏览器的"打印为PDF"功能
- PDF下载按钮会显示友好的错误提示

---

## 🔍 验证安装

### 检查核心依赖

```bash
python3 -c "
import fastapi, uvicorn, matplotlib, seaborn, pandas, numpy
print('✅ 所有核心依赖已安装')
print(f'FastAPI: {fastapi.__version__}')
print(f'Pandas: {pandas.__version__}')
print(f'NumPy: {numpy.__version__}')
"
```

### 检查GPU支持

```bash
python3 webapp/test_gpu_detection.py
```

应该看到：
```
✅ Device initialized: MPS
   Device: mps
   Precision: 32-bit
```

### 检查Web服务

```bash
# 启动服务器
bash webapp/start_server.sh

# 在另一个终端检查健康状态
curl http://localhost:8000/health | python3 -m json.tool
```

应该看到：
```json
{
  "status": "healthy",
  "device_type": "mps",
  "device_name": "Apple Silicon (arm64)",
  "gpu_available": true,
  "mps_available": true,
  "precision": "32-bit",
  "pdf_export_available": false
}
```

---

## 📊 使用流程

### 1. 准备数据

确保您的935k甲基化数据是CSV格式：
- 行：CpG位点（probe IDs）
- 列：样本ID
- 值：Beta值（0-1之间）

### 2. 启动服务

```bash
bash webapp/start_server.sh
```

### 3. 上传文件

1. 打开浏览器：http://localhost:8000
2. 拖拽或选择CSV文件
3. 点击"开始分析"

### 4. 查看结果

- 实时进度显示
- 完成后自动显示HTML报告
- 包含4个预测模型的结果：
  - 年龄预测
  - 癌症风险
  - 表观遗传时钟（5种）
  - 蛋白质水平

### 5. 导出报告（可选）

- 如果PDF功能可用：点击"下载PDF"
- 如果PDF功能不可用：使用浏览器"打印为PDF"

---

## 🛠️ 故障排除

### 问题1：服务器启动失败

**检查**:
```bash
# 检查端口是否被占用
lsof -i :8000

# 如果被占用，杀死进程或使用其他端口
python3 -m uvicorn webapp.app:app --host 0.0.0.0 --port 8001
```

### 问题2：GPU未被检测

**检查**:
```bash
python3 -c "
import torch
print(f'MPS Available: {torch.backends.mps.is_available()}')
print(f'PyTorch Version: {torch.__version__}')
"
```

### 问题3：分析失败

**检查日志**:
```bash
# 查看最新日志
tail -f webapp/logs/cpgpt_web_*.log
```

---

## 📚 相关文档

- **README.md** - 完整的项目文档
- **QUICKSTART.md** - 快速入门指南
- **INSTALL_MACOS.md** - macOS详细安装指南
- **GPU_COMPATIBILITY.md** - GPU兼容性指南
- **CHANGELOG.md** - 版本更新记录

---

## ✨ 下一步

### 立即可用
1. ✅ 启动服务器测试基本功能
2. ✅ 上传CSV格式的935k数据
3. ✅ 查看HTML报告

### 可选增强
1. ⚠️ 安装PyArrow支持Arrow格式
2. ⚠️ 安装系统库启用PDF导出
3. 💡 配置生产环境部署

---

**最后更新**: 2024-11-04  
**状态**: 核心功能完全可用，可选功能待完善  
**平台**: macOS (Apple Silicon) + Python 3.13

