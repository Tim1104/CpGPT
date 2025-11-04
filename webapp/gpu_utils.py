"""
GPU工具模块 - 跨平台GPU检测和设备管理
支持 NVIDIA CUDA 和 Apple Silicon MPS
"""

import torch
import platform
import logging

logger = logging.getLogger("cpgpt_web")


def get_device_info():
    """
    获取设备信息，支持 CUDA (NVIDIA) 和 MPS (Apple Silicon)
    
    Returns:
        dict: 包含设备类型、名称、可用性等信息
    """
    device_info = {
        "platform": platform.system(),
        "machine": platform.machine(),
        "device_type": "cpu",
        "device_name": "CPU",
        "device": torch.device("cpu"),
        "cuda_available": False,
        "mps_available": False,
        "gpu_available": False,
        "precision": "32-bit",  # 默认精度
    }
    
    # 检查 CUDA (NVIDIA GPU)
    if torch.cuda.is_available():
        device_info["cuda_available"] = True
        device_info["gpu_available"] = True
        device_info["device_type"] = "cuda"
        device_info["device_name"] = torch.cuda.get_device_name(0)
        device_info["device"] = torch.device("cuda:0")
        device_info["gpu_count"] = torch.cuda.device_count()
        device_info["gpu_memory"] = f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        device_info["precision"] = "16-mixed"  # CUDA支持混合精度
        logger.info(f"✅ NVIDIA GPU detected: {device_info['device_name']}")
        logger.info(f"   Memory: {device_info['gpu_memory']}")
    
    # 检查 MPS (Apple Silicon)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device_info["mps_available"] = True
        device_info["gpu_available"] = True
        device_info["device_type"] = "mps"
        device_info["device_name"] = f"Apple Silicon ({platform.machine()})"
        device_info["device"] = torch.device("mps")
        device_info["precision"] = "32-bit"  # MPS目前推荐使用32-bit
        logger.info(f"✅ Apple Silicon GPU (MPS) detected: {device_info['device_name']}")
        logger.info(f"   Note: Using 32-bit precision for MPS compatibility")
    
    else:
        logger.warning("⚠️  No GPU detected. Using CPU (slower performance)")
        logger.info(f"   Platform: {device_info['platform']} ({device_info['machine']})")
    
    return device_info


def get_optimal_precision(device_type):
    """
    根据设备类型获取最优精度设置
    
    Args:
        device_type: 设备类型 ("cuda", "mps", "cpu")
    
    Returns:
        str: 精度设置 ("16-mixed", "32-bit", "bf16-mixed")
    """
    if device_type == "cuda":
        # NVIDIA GPU支持混合精度训练
        return "16-mixed"
    elif device_type == "mps":
        # Apple Silicon MPS 目前推荐使用32-bit
        # 注意: PyTorch 2.0+ 的MPS可能支持部分混合精度，但稳定性需要测试
        return "32-bit"
    else:
        # CPU使用32-bit
        return "32-bit"


def move_to_device(model, device_info):
    """
    将模型移动到指定设备
    
    Args:
        model: PyTorch模型
        device_info: 设备信息字典
    
    Returns:
        model: 移动后的模型
    """
    device = device_info["device"]
    device_type = device_info["device_type"]
    
    try:
        model = model.to(device)
        logger.info(f"✅ Model moved to {device_type.upper()}")
        return model
    except Exception as e:
        logger.warning(f"⚠️  Failed to move model to {device_type}: {str(e)}")
        logger.info("   Falling back to CPU")
        return model.to("cpu")


def get_dataloader_kwargs(device_type):
    """
    根据设备类型获取DataLoader的最优参数
    
    Args:
        device_type: 设备类型
    
    Returns:
        dict: DataLoader参数
    """
    if device_type == "cuda":
        return {
            "pin_memory": True,
            "num_workers": 4,
        }
    elif device_type == "mps":
        # MPS不支持pin_memory
        return {
            "pin_memory": False,
            "num_workers": 0,  # MPS在多进程时可能有问题
        }
    else:
        return {
            "pin_memory": False,
            "num_workers": 4,
        }


def optimize_for_device(device_type):
    """
    根据设备类型进行优化设置
    
    Args:
        device_type: 设备类型
    """
    if device_type == "cuda":
        # CUDA优化
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        logger.info("✅ CUDA optimizations enabled")
    
    elif device_type == "mps":
        # MPS优化
        # 设置MPS后备到CPU（如果某些操作不支持）
        if hasattr(torch.backends.mps, "fallback_to_cpu"):
            torch.backends.mps.fallback_to_cpu = True
        logger.info("✅ MPS optimizations enabled")
    
    else:
        # CPU优化
        torch.set_num_threads(4)
        logger.info("✅ CPU optimizations enabled")


def check_mps_compatibility():
    """
    检查MPS兼容性和已知问题
    
    Returns:
        dict: 兼容性信息
    """
    compat_info = {
        "mps_available": False,
        "pytorch_version": torch.__version__,
        "warnings": [],
        "recommendations": [],
    }
    
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        compat_info["mps_available"] = True
        
        # 检查PyTorch版本
        pytorch_version = tuple(map(int, torch.__version__.split('.')[:2]))
        
        if pytorch_version < (2, 0):
            compat_info["warnings"].append(
                "PyTorch version < 2.0. MPS support may be limited."
            )
            compat_info["recommendations"].append(
                "Upgrade to PyTorch 2.0+ for better MPS support"
            )
        
        # 已知的MPS限制
        compat_info["warnings"].append(
            "MPS may not support all PyTorch operations. Fallback to CPU enabled."
        )
        compat_info["recommendations"].append(
            "Use 32-bit precision for best stability on MPS"
        )
    
    return compat_info


def get_device_summary():
    """
    获取设备摘要信息（用于日志和健康检查）
    
    Returns:
        dict: 设备摘要
    """
    device_info = get_device_info()
    
    summary = {
        "platform": device_info["platform"],
        "machine": device_info["machine"],
        "device_type": device_info["device_type"],
        "device_name": device_info["device_name"],
        "gpu_available": device_info["gpu_available"],
        "precision": device_info["precision"],
    }
    
    if device_info["cuda_available"]:
        summary["gpu_memory"] = device_info["gpu_memory"]
        summary["gpu_count"] = device_info["gpu_count"]
    
    if device_info["mps_available"]:
        compat = check_mps_compatibility()
        summary["pytorch_version"] = compat["pytorch_version"]
        summary["mps_warnings"] = compat["warnings"]
    
    return summary


def log_device_info():
    """
    记录详细的设备信息到日志
    """
    logger.info("=" * 80)
    logger.info("Device Information")
    logger.info("=" * 80)
    
    device_info = get_device_info()
    
    logger.info(f"Platform: {device_info['platform']} ({device_info['machine']})")
    logger.info(f"Device Type: {device_info['device_type'].upper()}")
    logger.info(f"Device Name: {device_info['device_name']}")
    logger.info(f"GPU Available: {device_info['gpu_available']}")
    logger.info(f"Recommended Precision: {device_info['precision']}")
    
    if device_info["cuda_available"]:
        logger.info(f"CUDA Version: {torch.version.cuda}")
        logger.info(f"GPU Count: {device_info['gpu_count']}")
        logger.info(f"GPU Memory: {device_info['gpu_memory']}")
    
    if device_info["mps_available"]:
        compat = check_mps_compatibility()
        logger.info(f"PyTorch Version: {compat['pytorch_version']}")
        if compat["warnings"]:
            logger.warning("MPS Warnings:")
            for warning in compat["warnings"]:
                logger.warning(f"  - {warning}")
        if compat["recommendations"]:
            logger.info("MPS Recommendations:")
            for rec in compat["recommendations"]:
                logger.info(f"  - {rec}")
    
    logger.info("=" * 80)


# 全局设备信息（在应用启动时初始化）
DEVICE_INFO = None


def initialize_device():
    """
    初始化设备（在应用启动时调用）
    
    Returns:
        dict: 设备信息
    """
    global DEVICE_INFO
    DEVICE_INFO = get_device_info()
    optimize_for_device(DEVICE_INFO["device_type"])
    log_device_info()
    return DEVICE_INFO


def get_current_device():
    """
    获取当前设备信息
    
    Returns:
        dict: 设备信息
    """
    global DEVICE_INFO
    if DEVICE_INFO is None:
        DEVICE_INFO = initialize_device()
    return DEVICE_INFO

