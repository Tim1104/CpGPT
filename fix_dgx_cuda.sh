#!/bin/bash
# DGX CUDA 环境快速修复脚本

set -e

echo "================================================================================"
echo "🔧 DGX CUDA 环境快速修复脚本"
echo "================================================================================"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 1. 检查是否在 DGX 上
echo -e "\n${YELLOW}1️⃣ 检查系统信息...${NC}"
echo "主机名: $(hostname)"
echo "操作系统: $(uname -a)"

# 2. 检查 NVIDIA 驱动
echo -e "\n${YELLOW}2️⃣ 检查 NVIDIA 驱动...${NC}"
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✓ nvidia-smi 可用${NC}"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
    
    # 获取 CUDA 版本
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
    echo -e "${GREEN}✓ CUDA 版本: $CUDA_VERSION${NC}"
else
    echo -e "${RED}✗ nvidia-smi 不可用${NC}"
    echo -e "${RED}请联系管理员检查 NVIDIA 驱动安装${NC}"
    exit 1
fi

# 3. 检查 Python 和 PyTorch
echo -e "\n${YELLOW}3️⃣ 检查 Python 和 PyTorch...${NC}"
PYTHON_CMD="python3"
if ! command -v $PYTHON_CMD &> /dev/null; then
    PYTHON_CMD="python"
fi

echo "Python 路径: $(which $PYTHON_CMD)"
echo "Python 版本: $($PYTHON_CMD --version)"

# 检查 PyTorch
echo -e "\n检查 PyTorch CUDA 支持..."
CUDA_AVAILABLE=$($PYTHON_CMD -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")

if [ "$CUDA_AVAILABLE" = "True" ]; then
    echo -e "${GREEN}✓ PyTorch CUDA 已可用！${NC}"
    $PYTHON_CMD -c "import torch; print(f'PyTorch 版本: {torch.__version__}'); print(f'CUDA 版本: {torch.version.cuda}'); print(f'GPU 数量: {torch.cuda.device_count()}')"
    
    echo -e "\n${GREEN}✅ 环境正常，无需修复！${NC}"
    echo -e "\n${YELLOW}建议配置:${NC}"
    echo "  - 在 935k_zero_shot_inference.py 中设置: USE_CPU = False"
    echo "  - 可以使用: MAX_INPUT_LENGTH = 30000 或更大"
    exit 0
else
    echo -e "${RED}✗ PyTorch CUDA 不可用${NC}"
    
    # 检查是否安装了 PyTorch
    if $PYTHON_CMD -c "import torch" 2>/dev/null; then
        echo -e "${YELLOW}⚠️ PyTorch 已安装但是 CPU 版本${NC}"
        PYTORCH_VERSION=$($PYTHON_CMD -c "import torch; print(torch.__version__)")
        echo "当前 PyTorch 版本: $PYTORCH_VERSION"
    else
        echo -e "${RED}✗ PyTorch 未安装${NC}"
    fi
fi

# 4. 提供修复方案
echo -e "\n${YELLOW}4️⃣ 修复方案${NC}"
echo "================================================================================"

# 确定推荐的 CUDA 版本
if [[ $CUDA_VERSION == 12.* ]]; then
    RECOMMENDED_CUDA="cu121"
    CONDA_CUDA="12.1"
elif [[ $CUDA_VERSION == 11.* ]]; then
    RECOMMENDED_CUDA="cu118"
    CONDA_CUDA="11.8"
else
    RECOMMENDED_CUDA="cu118"
    CONDA_CUDA="11.8"
    echo -e "${YELLOW}⚠️ 无法确定 CUDA 版本，使用默认 11.8${NC}"
fi

echo -e "\n${GREEN}推荐安装命令（基于检测到的 CUDA $CUDA_VERSION）:${NC}"
echo ""
echo "方法 1: 使用 pip (推荐)"
echo "----------------------------------------"
echo "# 卸载当前版本"
echo "pip uninstall torch torchvision torchaudio -y"
echo ""
echo "# 安装 GPU 版本"
echo "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/$RECOMMENDED_CUDA"
echo ""
echo "方法 2: 使用 conda"
echo "----------------------------------------"
echo "conda install pytorch torchvision torchaudio pytorch-cuda=$CONDA_CUDA -c pytorch -c nvidia"
echo ""
echo "验证安装:"
echo "----------------------------------------"
echo "python -c \"import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')\""
echo ""

# 5. 询问是否自动修复
echo -e "\n${YELLOW}是否自动执行修复？(y/n)${NC}"
read -r response

if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    echo -e "\n${GREEN}开始自动修复...${NC}"
    
    # 卸载
    echo -e "\n${YELLOW}卸载当前 PyTorch...${NC}"
    pip uninstall torch torchvision torchaudio -y || true
    
    # 安装
    echo -e "\n${YELLOW}安装 GPU 版本 PyTorch...${NC}"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/$RECOMMENDED_CUDA
    
    # 验证
    echo -e "\n${YELLOW}验证安装...${NC}"
    CUDA_AVAILABLE_NEW=$($PYTHON_CMD -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")
    
    if [ "$CUDA_AVAILABLE_NEW" = "True" ]; then
        echo -e "\n${GREEN}✅ 修复成功！${NC}"
        $PYTHON_CMD -c "import torch; print(f'✓ PyTorch 版本: {torch.__version__}'); print(f'✓ CUDA 版本: {torch.version.cuda}'); print(f'✓ GPU 数量: {torch.cuda.device_count()}')"
        
        echo -e "\n${GREEN}下一步:${NC}"
        echo "1. 修改 examples/935k_zero_shot_inference.py"
        echo "   - 设置: USE_CPU = False"
        echo "   - 设置: MAX_INPUT_LENGTH = 30000 (或更大)"
        echo "2. 运行: python examples/935k_zero_shot_inference.py"
    else
        echo -e "\n${RED}✗ 修复失败，请手动执行上述命令${NC}"
        exit 1
    fi
else
    echo -e "\n${YELLOW}跳过自动修复，请手动执行上述命令${NC}"
fi

echo ""
echo "================================================================================"
echo "🎉 完成！"
echo "================================================================================"

