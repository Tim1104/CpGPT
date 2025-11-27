# 🚀 DGX 服务器 CUDA 环境设置指南

## 问题描述

在 NVIDIA DGX Spark 机器上运行时，检测到 `CUDA 可用: False`，这是不正常的。

## 📋 诊断步骤

### 1. SSH 登录到 DGX 服务器

```bash
# 替换为你的 DGX 服务器地址和用户名
ssh username@dgx-server-address
```

### 2. 上传诊断脚本到 DGX

**方法 A: 使用 scp 上传**
```bash
# 在你的 Mac 上运行
cd /Users/wulianghua/Documents/GitHub/CpGPT
scp check_cuda_environment.py username@dgx-server-address:~/
```

**方法 B: 在 DGX 上直接创建**
```bash
# SSH 到 DGX 后，创建文件
cat > check_cuda_environment.py << 'EOF'
# 复制 check_cuda_environment.py 的内容到这里
EOF
```

**方法 C: 使用 git clone（推荐）**
```bash
# 在 DGX 上运行
cd ~
git clone https://github.com/yourusername/CpGPT.git
cd CpGPT
```

### 3. 在 DGX 上运行诊断

```bash
# 在 DGX 服务器上运行
python check_cuda_environment.py
# 或
python3 check_cuda_environment.py
```

## 🔍 预期输出分析

### 情况 1: CUDA 可用（正常）

```
✅ 状态: CUDA 环境正常
✅ 可以使用 8 个 GPU 进行训练/推理
```

**解决方案:** 
- 修改 `935k_zero_shot_inference.py` 中的 `USE_CPU = False`
- 直接运行即可

### 情况 2: PyTorch 是 CPU 版本（最常见）

```
❌ PyTorch 没有编译 CUDA 支持（CPU 版本）
```

**解决方案:** 重新安装 GPU 版本的 PyTorch

#### 步骤 A: 检查 DGX 的 CUDA 版本

```bash
# 在 DGX 上运行
nvidia-smi
```

查看输出顶部的 `CUDA Version`，例如：
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.125.06   Driver Version: 525.125.06   CUDA Version: 12.0   |
+-----------------------------------------------------------------------------+
```

#### 步骤 B: 卸载当前 PyTorch

```bash
pip uninstall torch torchvision torchaudio -y
```

#### 步骤 C: 安装匹配的 GPU 版本

**如果 CUDA 版本是 11.x (如 11.7, 11.8):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**如果 CUDA 版本是 12.x (如 12.0, 12.1):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**使用 conda (推荐，如果 DGX 使用 conda):**
```bash
# CUDA 11.8
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# CUDA 12.1
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

#### 步骤 D: 验证安装

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
```

预期输出：
```
CUDA available: True
GPU count: 8
```

### 情况 3: NVIDIA 驱动问题

```
❌ nvidia-smi 未找到 - NVIDIA 驱动可能未安装
```

**解决方案:** 联系 DGX 管理员，这需要系统级别的修复

## 🎯 修改 935k 脚本以使用 GPU

修复 CUDA 后，编辑 `examples/935k_zero_shot_inference.py`:

```python
# 第63行，改为:
USE_CPU = False  # 在 DGX 上使用 GPU

# 第62行，可以增加（DGX 内存充足）:
MAX_INPUT_LENGTH = 30000  # 或更大
```

## 📊 DGX 优化配置

在 DGX 上，你可以使用更激进的配置：

```python
# 在 935k_zero_shot_inference.py 中

# 使用 GPU
USE_CPU = False

# 更大的输入长度
MAX_INPUT_LENGTH = 50000  # DGX 内存充足

# 数据加载优化
datamodule_age = CpGPTDataModule(
    predict_dir=f"{PROCESSED_DIR}_age",
    dependencies_dir=DEPENDENCIES_DIR,
    batch_size=4,  # 可以增加 batch size
    num_workers=8,  # 使用多个 worker
    max_length=MAX_INPUT_LENGTH,
    dna_llm=config_age.data.dna_llm,
    dna_context_len=config_age.data.dna_context_len,
    sorting_strategy=config_age.data.sorting_strategy,
    pin_memory=True,  # GPU 上启用
)

# Trainer 配置
trainer = CpGPTTrainer(
    accelerator="gpu",
    devices=1,  # 使用 1 个 GPU，或 [0,1,2,3] 使用多个
    precision="16-mixed",  # 混合精度训练
)
```

## 🚀 完整运行流程（DGX）

```bash
# 1. SSH 到 DGX
ssh username@dgx-server

# 2. 激活环境（如果使用 conda/venv）
conda activate your_env
# 或
source venv/bin/activate

# 3. 进入项目目录
cd ~/CpGPT

# 4. 检查 CUDA
python check_cuda_environment.py

# 5. 如果 CUDA 不可用，重新安装 PyTorch（见上文）

# 6. 修改配置
vim examples/935k_zero_shot_inference.py
# 设置 USE_CPU = False

# 7. 运行推理
python examples/935k_zero_shot_inference.py
```

## ⚡ 性能对比

| 环境 | 速度 | 推荐配置 |
|------|------|----------|
| Mac (CPU) | 基准 | `USE_CPU=True`, `MAX_INPUT_LENGTH=15000` |
| Mac (MPS) | 5-10x | `USE_CPU=False`, `MAX_INPUT_LENGTH=8000` |
| **DGX (GPU)** | **50-100x** | `USE_CPU=False`, `MAX_INPUT_LENGTH=50000` |

## 🐛 常见问题

### Q1: 我在 DGX 上但是 CUDA 还是不可用

**检查清单:**
1. ✅ 确认你真的在 DGX 上（运行 `hostname`）
2. ✅ 运行 `nvidia-smi` 确认驱动正常
3. ✅ 检查 PyTorch 版本：`python -c "import torch; print(torch.__version__)"`
4. ✅ 确认不是在 CPU-only 的 Docker 容器中

### Q2: 安装 GPU 版本 PyTorch 后还是不行

**可能原因:**
1. CUDA 版本不匹配
2. 环境变量问题
3. 多个 Python 环境混淆

**解决方案:**
```bash
# 完全清理
pip uninstall torch torchvision torchaudio -y
pip cache purge

# 重新安装
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 验证
python -c "import torch; print(torch.cuda.is_available())"
```

### Q3: 如何使用多个 GPU？

修改 Trainer 配置：
```python
trainer = CpGPTTrainer(
    accelerator="gpu",
    devices=4,  # 使用 4 个 GPU
    strategy="ddp",  # 分布式数据并行
    precision="16-mixed",
)
```

### Q4: 内存溢出怎么办？

即使在 DGX 上也可能遇到，解决方案：
```python
# 减小 batch_size
batch_size=1

# 减小 max_length
MAX_INPUT_LENGTH=20000

# 使用梯度累积
trainer = CpGPTTrainer(
    accumulate_grad_batches=4,  # 累积 4 个 batch
)
```

## 📞 获取帮助

如果问题仍然存在：
1. 运行 `check_cuda_environment.py` 并保存完整输出
2. 运行 `nvidia-smi` 并保存输出
3. 运行 `pip list | grep torch` 查看 PyTorch 相关包
4. 联系 DGX 管理员或提供以上信息寻求帮助

## 🎓 学习资源

- [PyTorch 安装指南](https://pytorch.org/get-started/locally/)
- [NVIDIA DGX 用户指南](https://docs.nvidia.com/dgx/)
- [PyTorch Lightning GPU 训练](https://lightning.ai/docs/pytorch/stable/accelerators/gpu.html)

