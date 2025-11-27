#!/usr/bin/env python3
"""
CUDA 环境诊断脚本
用于检查 PyTorch 和 CUDA 配置是否正确
"""

import sys
import subprocess

print("=" * 80)
print("🔍 CUDA 环境诊断")
print("=" * 80)

# 1. Python 版本
print("\n1️⃣ Python 版本:")
print(f"   {sys.version}")
print(f"   路径: {sys.executable}")

# 2. PyTorch 版本和 CUDA 支持
print("\n2️⃣ PyTorch 配置:")
try:
    import torch
    print(f"   ✓ PyTorch 版本: {torch.__version__}")
    print(f"   ✓ PyTorch 安装路径: {torch.__file__}")
    print(f"   ✓ CUDA 可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"   ✓ CUDA 版本 (PyTorch): {torch.version.cuda}")
        print(f"   ✓ cuDNN 版本: {torch.backends.cudnn.version()}")
        print(f"   ✓ GPU 数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"   ✓ GPU {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"      - 显存: {props.total_memory / 1024**3:.2f} GB")
            print(f"      - 计算能力: {props.major}.{props.minor}")
    else:
        print(f"   ❌ CUDA 不可用!")
        print(f"   ⚠️ PyTorch 可能是 CPU 版本")
        
        # 检查是否编译了 CUDA 支持
        print(f"\n   检查 PyTorch 编译配置:")
        print(f"   - CUDA 编译支持: {torch.version.cuda is not None}")
        if torch.version.cuda is None:
            print(f"   ❌ PyTorch 没有编译 CUDA 支持（CPU 版本）")
        
except ImportError as e:
    print(f"   ❌ PyTorch 未安装: {e}")

# 3. NVIDIA 驱动和 CUDA Toolkit
print("\n3️⃣ NVIDIA 驱动和 CUDA Toolkit:")
try:
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        print("   ✓ nvidia-smi 可用:")
        # 提取关键信息
        lines = result.stdout.split('\n')
        for line in lines[:15]:  # 只显示前15行
            if line.strip():
                print(f"   {line}")
    else:
        print(f"   ❌ nvidia-smi 失败: {result.stderr}")
except FileNotFoundError:
    print("   ❌ nvidia-smi 未找到 - NVIDIA 驱动可能未安装")
except subprocess.TimeoutExpired:
    print("   ❌ nvidia-smi 超时")
except Exception as e:
    print(f"   ❌ nvidia-smi 错误: {e}")

# 4. CUDA Toolkit 版本
print("\n4️⃣ CUDA Toolkit 版本:")
try:
    result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        print("   ✓ nvcc 可用:")
        print(f"   {result.stdout.strip()}")
    else:
        print(f"   ⚠️ nvcc 不可用（可能未安装 CUDA Toolkit）")
except FileNotFoundError:
    print("   ⚠️ nvcc 未找到（CUDA Toolkit 可能未安装或未在 PATH 中）")
except Exception as e:
    print(f"   ⚠️ nvcc 检查失败: {e}")

# 5. 环境变量
print("\n5️⃣ 相关环境变量:")
import os
cuda_vars = ['CUDA_HOME', 'CUDA_PATH', 'LD_LIBRARY_PATH', 'PATH']
for var in cuda_vars:
    value = os.environ.get(var, '未设置')
    if var in ['LD_LIBRARY_PATH', 'PATH'] and value != '未设置':
        # 只显示 CUDA 相关的路径
        cuda_paths = [p for p in value.split(':') if 'cuda' in p.lower()]
        if cuda_paths:
            print(f"   {var} (CUDA 相关):")
            for p in cuda_paths[:3]:  # 只显示前3个
                print(f"      - {p}")
        else:
            print(f"   {var}: (无 CUDA 相关路径)")
    else:
        print(f"   {var}: {value}")

# 6. 其他深度学习库
print("\n6️⃣ 其他深度学习库:")
libs = [
    ('lightning', 'PyTorch Lightning'),
    ('transformers', 'Hugging Face Transformers'),
    ('numpy', 'NumPy'),
]

for module_name, display_name in libs:
    try:
        module = __import__(module_name)
        version = getattr(module, '__version__', '未知版本')
        print(f"   ✓ {display_name}: {version}")
    except ImportError:
        print(f"   ⚠️ {display_name}: 未安装")

# 7. 诊断总结
print("\n" + "=" * 80)
print("📋 诊断总结")
print("=" * 80)

try:
    import torch
    if torch.cuda.is_available():
        print("✅ 状态: CUDA 环境正常")
        print(f"✅ 可以使用 {torch.cuda.device_count()} 个 GPU 进行训练/推理")
        print("\n建议配置:")
        print("   - 在 935k_zero_shot_inference.py 中设置: USE_CPU = False")
        print("   - 可以使用更大的 batch_size 和 max_length")
    else:
        print("❌ 状态: CUDA 不可用")
        print("\n可能的原因:")
        
        if torch.version.cuda is None:
            print("   1. ❌ PyTorch 是 CPU 版本（最可能）")
            print("      解决方案: 重新安装 GPU 版本的 PyTorch")
        else:
            print("   1. ⚠️ NVIDIA 驱动问题")
            print("   2. ⚠️ CUDA Toolkit 版本不匹配")
            print("   3. ⚠️ 环境变量配置问题")
        
        print("\n修复步骤:")
        print("   1. 检查 NVIDIA 驱动: nvidia-smi")
        print("   2. 卸载当前 PyTorch: pip uninstall torch torchvision torchaudio")
        print("   3. 安装 GPU 版本 PyTorch (见下方命令)")
        
except ImportError:
    print("❌ 状态: PyTorch 未安装")
    print("\n修复步骤:")
    print("   安装 GPU 版本的 PyTorch (见下方命令)")

# 8. 推荐的安装命令
print("\n" + "=" * 80)
print("🔧 推荐的 PyTorch 安装命令")
print("=" * 80)

print("\n根据你的 CUDA 版本选择合适的命令:")
print("\n# CUDA 11.8 (推荐)")
print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")

print("\n# CUDA 12.1")
print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")

print("\n# 或使用 conda (推荐)")
print("conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia")

print("\n# 安装后验证:")
print("python -c \"import torch; print(f'CUDA available: {torch.cuda.is_available()}')\"")

print("\n" + "=" * 80)
print("💡 提示:")
print("   - DGX 机器通常预装了 CUDA，检查 nvidia-smi 输出的 CUDA 版本")
print("   - 安装与系统 CUDA 版本匹配的 PyTorch")
print("   - 如果不确定，CUDA 11.8 通常兼容性最好")
print("=" * 80)

