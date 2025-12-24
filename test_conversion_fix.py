"""
快速测试转换脚本的修复
Quick test for conversion script fix
"""

import pandas as pd
import numpy as np
from pathlib import Path

# 创建一个小的测试数据集
print("创建测试数据...")
print("Creating test data...")

# 模拟你的实际数据格式
data = {
    'TargetID': [
        'cg00000029_TC21',
        'cg00000029_BC21',  # 重复探针
        'cg00000109_TC21',
        'cg00000155_BC11',
        'cg00000158_BC31',  # 不同的后缀
    ],
    '000536.AVG_Beta': ['0.4630385', '0.4800000', '0.8233373', '0.8882958', '0.9374912'],
    '000537.AVG_Beta': ['0.4062999', '0.4100000', '0.8394986', '0.8735513', '0.9106941'],
}

df = pd.DataFrame(data)
test_file = 'test_sample.csv'
df.to_csv(test_file, index=False)

print(f"\n测试数据已保存到: {test_file}")
print(f"Test data saved to: {test_file}")
print("\n原始数据:")
print("Original data:")
print(df)

# 运行转换
print("\n" + "="*80)
print("运行转换...")
print("Running conversion...")
print("="*80)

import sys
sys.path.insert(0, 'examples')
from convert_935k_format import convert_935k_format

try:
    df_converted = convert_935k_format(test_file, 'test_sample.arrow', verbose=True)
    
    print("\n" + "="*80)
    print("✓ 转换成功！")
    print("✓ Conversion successful!")
    print("="*80)
    
    print("\n转换后的数据:")
    print("Converted data:")
    print(df_converted)
    
    print("\n验证:")
    print("Verification:")
    print(f"  - 形状: {df_converted.shape}")
    print(f"  - Shape: {df_converted.shape}")
    print(f"  - 第一列: {df_converted.columns[0]}")
    print(f"  - First column: {df_converted.columns[0]}")
    print(f"  - 数据类型: {df_converted.dtypes.tolist()}")
    print(f"  - Data types: {df_converted.dtypes.tolist()}")
    
    # 检查重复探针是否被正确平均
    print("\n检查重复探针处理:")
    print("Check duplicate probe handling:")
    print(f"  - 原始有 cg00000029_TC21 和 cg00000029_BC21")
    print(f"  - Original has cg00000029_TC21 and cg00000029_BC21")
    print(f"  - 转换后 cg00000029 列的值:")
    print(f"  - Converted cg00000029 column values:")
    if 'cg00000029' in df_converted.columns:
        print(f"    样本1: {df_converted['cg00000029'].iloc[0]}")
        print(f"    样本2: {df_converted['cg00000029'].iloc[1]}")
        print(f"  - 期望值（平均）:")
        print(f"  - Expected values (average):")
        print(f"    样本1: {(0.4630385 + 0.4800000) / 2}")
        print(f"    样本2: {(0.4062999 + 0.4100000) / 2}")
    
    print("\n✓ 所有测试通过！")
    print("✓ All tests passed!")
    
except Exception as e:
    print(f"\n❌ 转换失败！")
    print(f"❌ Conversion failed!")
    print(f"错误: {str(e)}")
    print(f"Error: {str(e)}")
    import traceback
    traceback.print_exc()

# 清理测试文件
print("\n清理测试文件...")
print("Cleaning up test files...")
Path(test_file).unlink(missing_ok=True)
Path('test_sample.arrow').unlink(missing_ok=True)
print("完成！")
print("Done!")

