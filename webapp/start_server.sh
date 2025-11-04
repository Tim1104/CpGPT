#!/bin/bash

# CpGPT Web Application Startup Script

echo "🧬 Starting CpGPT 935k Methylation Analysis Web Server..."
echo ""

# 检查Python环境
if command -v python3 &> /dev/null; then
    PYTHON=python3
elif command -v python &> /dev/null; then
    PYTHON=python
else
    echo "❌ Python not found. Please install Python 3.8 or higher."
    exit 1
fi

echo "Using Python: $PYTHON ($($PYTHON --version))"
echo ""

# 检查是否在项目根目录
if [ ! -f "webapp/app.py" ]; then
    echo "❌ Please run this script from the CpGPT project root directory."
    exit 1
fi

# 创建必要的目录
echo "📁 Creating necessary directories..."
mkdir -p webapp/uploads
mkdir -p webapp/results
mkdir -p webapp/static
mkdir -p dependencies

# 检查依赖
echo "📦 Checking dependencies..."
if ! $PYTHON -c "import fastapi" &> /dev/null; then
    echo "⚠️  FastAPI not found. Installing web dependencies..."
    $PYTHON -m pip install -r webapp/requirements.txt
fi

# 检查CpGPT模型和依赖
echo "🔍 Checking CpGPT dependencies..."
if [ ! -d "dependencies/model" ]; then
    echo "⚠️  CpGPT models not found. Please download them first:"
    echo "   python examples/935k_zero_shot_inference.py --download-only"
    echo ""
    read -p "Do you want to download now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        $PYTHON -c "
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
print('✅ Download complete!')
"
    else
        echo "⚠️  Warning: Models not downloaded. The server may not work properly."
    fi
fi

# 检查GPU（支持CUDA和MPS）
echo "🖥️  Checking GPU availability..."
$PYTHON -c "
import torch
import platform

print(f'Platform: {platform.system()} ({platform.machine()})')
print(f'PyTorch version: {torch.__version__}')

if torch.cuda.is_available():
    print(f'✅ NVIDIA GPU available: {torch.cuda.get_device_name(0)}')
    print(f'   CUDA version: {torch.version.cuda}')
    print(f'   Will use 16-bit mixed precision')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print(f'✅ Apple Silicon GPU (MPS) available')
    print(f'   Device: {platform.machine()}')
    print(f'   Will use 32-bit precision for stability')
else:
    print('⚠️  No GPU detected. Analysis will use CPU (slower).')
    print('   Recommendation: Use a machine with GPU for better performance')
"

echo ""
echo "🚀 Starting server..."
echo "📍 Server will be available at: http://localhost:8000"
echo "📊 API documentation: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# 启动服务器
cd webapp
$PYTHON -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload

