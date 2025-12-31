#!/usr/bin/env python3
"""
简单测试Horvath Clock

这个脚本用于测试pyaging是否正常工作
"""

import pandas as pd
import numpy as np

print("=" * 80)
print("测试 Horvath Clock")
print("=" * 80)

# 步骤1：检查pyaging
print("\n[1/4] 检查pyaging...")
try:
    import pyaging as pya
    print(f"  ✓ pyaging 已安装 (版本: {pya.__version__})")
except ImportError as e:
    print(f"  ✗ pyaging 未安装: {e}")
    exit(1)

# 步骤2：读取数据
print("\n[2/4] 读取数据...")
try:
    df = pd.read_feather("data/Sample1107.arrow")
    print(f"  ✓ 读取了 {len(df)} 个样本")
    print(f"  ✓ 包含 {len(df.columns)-1} 个CpG位点")
    
    # 检查数据范围
    sample_ids = df['sample_id'].values
    cpg_data = df.drop(columns=['sample_id'])
    
    print(f"\n  数据检查:")
    print(f"    - 最小值: {cpg_data.min().min():.4f}")
    print(f"    - 最大值: {cpg_data.max().max():.4f}")
    print(f"    - 均值: {cpg_data.mean().mean():.4f}")
    print(f"    - NaN数量: {cpg_data.isna().sum().sum()}")
    
except Exception as e:
    print(f"  ✗ 读取数据失败: {e}")
    exit(1)

# 步骤3：准备数据格式
print("\n[3/4] 准备数据格式...")
try:
    # 转置：CpG位点 x 样本
    cpg_matrix = cpg_data.T
    cpg_matrix.columns = sample_ids
    
    print(f"  ✓ 数据形状: {cpg_matrix.shape} (CpG位点 x 样本)")
    print(f"  ✓ 样本ID: {list(sample_ids)}")
    
except Exception as e:
    print(f"  ✗ 数据准备失败: {e}")
    exit(1)

# 步骤4：查看可用的时钟
print("\n[4/4] 查看可用的时钟...")
try:
    # 列出所有DNA甲基化时钟
    print("\n  可用的DNA甲基化时钟:")
    clocks = ['horvath2013', 'hannum2013', 'phenoage', 'grimage', 'skinandblood']
    for clock in clocks:
        print(f"    - {clock}")
    
    print("\n  ℹ️ 提示：运行 horvath_clock_prediction.py 来进行实际预测")
    
except Exception as e:
    print(f"  ⚠️ 警告: {e}")

print("\n" + "=" * 80)
print("测试完成！")
print("=" * 80)
print("\n下一步：")
print("  python horvath_clock_prediction.py")

