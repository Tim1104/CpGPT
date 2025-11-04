# macOS安装指南（Python 3.13 + Apple Silicon）

## 问题说明

在macOS上使用Python 3.13时，某些包（如pandas 2.1.4）需要从源码编译，这会导致安装失败或非常缓慢。

## 解决方案

### 方案1：分步安装（推荐）

由于您的系统已经有了兼容的pandas和numpy，只需安装其他依赖：

```bash
# 1. 安装Web框架
pip3 install fastapi==0.104.1 uvicorn[standard]==0.24.0 python-multipart==0.0.6

# 2. 安装可视化库
pip3 install matplotlib==3.8.2 seaborn==0.13.0

# 3. 安装PDF生成（可选，如果需要PDF导出）
pip3 install weasyprint==60.1

# 4. 安装pyarrow（可能需要较长时间）
pip3 install pyarrow>=14.0.0
```

### 方案2：跳过pyarrow（临时方案）

如果pyarrow安装太慢，可以暂时跳过，使用CSV格式：

```bash
# 安装除pyarrow外的所有依赖
pip3 install fastapi==0.104.1 uvicorn[standard]==0.24.0 python-multipart==0.0.6
pip3 install matplotlib==3.8.2 seaborn==0.13.0
pip3 install weasyprint==60.1
```

然后修改代码，只支持CSV格式（不支持Arrow格式）。

### 方案3：使用conda（如果已安装）

```bash
conda install -c conda-forge fastapi uvicorn python-multipart
conda install -c conda-forge matplotlib seaborn
conda install -c conda-forge pyarrow
conda install -c conda-forge weasyprint
```

## 验证安装

```bash
python3 -c "
import fastapi
import uvicorn
import matplotlib
import seaborn
import pandas
import numpy
print('✅ All core dependencies installed!')
print(f'FastAPI: {fastapi.__version__}')
print(f'Pandas: {pandas.__version__}')
print(f'NumPy: {numpy.__version__}')
"
```

## 检查pyarrow（可选）

```bash
python3 -c "
try:
    import pyarrow
    print(f'✅ PyArrow installed: {pyarrow.__version__}')
except ImportError:
    print('⚠️  PyArrow not installed (Arrow format not supported)')
"
```

## 启动服务器

```bash
# 方法1：使用启动脚本
bash webapp/start_server.sh

# 方法2：直接启动
cd webapp
python3 -m uvicorn app:app --host 0.0.0.0 --port 8000
```

## 常见问题

### Q1: pandas编译错误

**错误**: `error: subprocess-exited-with-error` 在安装pandas时

**解决**: 您的系统已经有pandas 2.3.3，无需重新安装。跳过pandas安装。

### Q2: pyarrow下载太慢

**解决方案A**: 使用国内镜像
```bash
pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple pyarrow
```

**解决方案B**: 暂时跳过，只使用CSV格式

### Q3: weasyprint安装失败

**错误**: `cannot load library 'libgobject-2.0-0'`

**解决**: 使用Homebrew安装系统依赖
```bash
# 安装必需的系统库
brew install pango cairo gdk-pixbuf libffi glib

# 重新安装weasyprint
pip3 install --force-reinstall weasyprint
```

**临时方案**: 如果仍然失败，可以跳过PDF导出功能
- Web应用会自动检测weasyprint是否可用
- 如果不可用，PDF下载按钮将被禁用
- HTML报告仍然可以正常查看

### Q4: 缺少pkg-config

**解决**:
```bash
brew install pkg-config
```

## 最小化安装（快速测试）

如果只想快速测试，可以只安装核心依赖：

```bash
pip3 install fastapi uvicorn python-multipart matplotlib seaborn
```

然后：
1. 只上传CSV文件（不支持Arrow）
2. 跳过PDF导出功能

## 完整依赖列表

### 必需
- ✅ fastapi (Web框架)
- ✅ uvicorn (ASGI服务器)
- ✅ python-multipart (文件上传)
- ✅ matplotlib (绘图)
- ✅ seaborn (统计可视化)
- ✅ pandas (已安装 2.3.3)
- ✅ numpy (已安装 2.3.4)

### 可选
- ⚠️ pyarrow (Arrow格式支持，可选)
- ⚠️ weasyprint (PDF导出，可选)

### CpGPT核心依赖（应该已安装）
- torch >= 2.0.0
- lightning >= 2.0.0
- transformers >= 4.30.0

## 下一步

安装完成后，运行GPU检测测试：

```bash
python3 webapp/test_gpu_detection.py
```

应该看到：
```
✅ Apple Silicon GPU (MPS) available
   Device: arm64
   Will use 32-bit precision for stability
```

然后启动服务器：

```bash
bash webapp/start_server.sh
```

访问 http://localhost:8000

---

**最后更新**: 2024-11-04  
**适用于**: macOS (Apple Silicon) + Python 3.13

