#!/usr/bin/env python3
"""
GPU检测测试脚本
测试CUDA和MPS的检测功能
"""

import sys
import platform
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from webapp.gpu_utils import (
    get_device_info,
    get_optimal_precision,
    check_mps_compatibility,
    get_device_summary,
    log_device_info,
    initialize_device,
)


def print_section(title):
    """打印分节标题"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def test_basic_torch_detection():
    """测试基础PyTorch检测"""
    print_section("基础PyTorch检测")
    
    print(f"Platform: {platform.system()} ({platform.machine()})")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if hasattr(torch.backends, "mps"):
        print(f"MPS Available: {torch.backends.mps.is_available()}")
    else:
        print("MPS Available: False (PyTorch version too old)")
    
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Count: {torch.cuda.device_count()}")
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")


def test_device_info():
    """测试设备信息获取"""
    print_section("设备信息获取")
    
    device_info = get_device_info()
    
    print(f"Device Type: {device_info['device_type']}")
    print(f"Device Name: {device_info['device_name']}")
    print(f"Device: {device_info['device']}")
    print(f"GPU Available: {device_info['gpu_available']}")
    print(f"CUDA Available: {device_info['cuda_available']}")
    print(f"MPS Available: {device_info['mps_available']}")
    print(f"Recommended Precision: {device_info['precision']}")
    
    if device_info['cuda_available']:
        print(f"GPU Count: {device_info['gpu_count']}")
        print(f"GPU Memory: {device_info['gpu_memory']}")


def test_optimal_precision():
    """测试精度选择"""
    print_section("精度选择测试")
    
    device_types = ["cuda", "mps", "cpu"]
    
    for device_type in device_types:
        precision = get_optimal_precision(device_type)
        print(f"{device_type.upper()}: {precision}")


def test_mps_compatibility():
    """测试MPS兼容性检查"""
    print_section("MPS兼容性检查")
    
    compat = check_mps_compatibility()
    
    print(f"MPS Available: {compat['mps_available']}")
    print(f"PyTorch Version: {compat['pytorch_version']}")
    
    if compat['warnings']:
        print("\nWarnings:")
        for warning in compat['warnings']:
            print(f"  ⚠️  {warning}")
    
    if compat['recommendations']:
        print("\nRecommendations:")
        for rec in compat['recommendations']:
            print(f"  💡 {rec}")


def test_device_summary():
    """测试设备摘要"""
    print_section("设备摘要")
    
    summary = get_device_summary()
    
    for key, value in summary.items():
        print(f"{key}: {value}")


def test_device_initialization():
    """测试设备初始化"""
    print_section("设备初始化")
    
    device_info = initialize_device()
    
    print(f"✅ Device initialized: {device_info['device_type'].upper()}")
    print(f"   Device: {device_info['device']}")
    print(f"   Precision: {device_info['precision']}")


def test_tensor_operations():
    """测试张量操作"""
    print_section("张量操作测试")
    
    device_info = get_device_info()
    device = device_info['device']
    
    print(f"Creating tensor on {device}...")
    
    try:
        # 创建测试张量
        x = torch.randn(100, 100).to(device)
        y = torch.randn(100, 100).to(device)
        
        # 矩阵乘法
        z = torch.matmul(x, y)
        
        print(f"✅ Tensor operations successful on {device}")
        print(f"   Tensor shape: {z.shape}")
        print(f"   Tensor device: {z.device}")
        print(f"   Mean value: {z.mean().item():.4f}")
        
    except Exception as e:
        print(f"❌ Tensor operations failed: {str(e)}")


def main():
    """主函数"""
    print("\n" + "🔍" * 40)
    print("GPU Detection Test Script")
    print("🔍" * 40)
    
    try:
        # 运行所有测试
        test_basic_torch_detection()
        test_device_info()
        test_optimal_precision()
        test_mps_compatibility()
        test_device_summary()
        test_device_initialization()
        test_tensor_operations()
        
        print_section("测试完成")
        print("✅ All tests completed successfully!")
        
    except Exception as e:
        print_section("测试失败")
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

