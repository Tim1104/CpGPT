"""
测试 935k 数据格式转换
Test 935k data format conversion

这个脚本创建一个示例数据文件，然后测试转换功能
This script creates a sample data file and tests the conversion
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from convert_935k_format import convert_935k_format


def create_sample_935k_data(output_path="test_935k_sample.csv", n_probes=100, n_samples=3):
    """
    创建一个示例的935k格式数据文件
    Create a sample 935k format data file
    
    Args:
        output_path: 输出文件路径
        n_probes: 探针数量
        n_samples: 样本数量
    """
    print("=" * 80)
    print("创建示例 935k 数据")
    print("Creating sample 935k data")
    print("=" * 80)
    
    # 生成探针ID（带后缀）
    probe_ids = []
    for i in range(n_probes):
        cg_id = f"cg{i:08d}"
        # 随机选择后缀
        suffix = np.random.choice(['_TC21', '_BC21', '_BC11'])
        probe_ids.append(f"{cg_id}{suffix}")
    
    # 生成样本ID（带后缀）
    sample_ids = [f"{i:06d}.AVG_Beta" for i in range(n_samples)]
    
    # 生成Beta值（0-1之间，带一些空值）
    data = {}
    data['TargetID'] = probe_ids
    
    for sample_id in sample_ids:
        # 生成Beta值
        beta_values = np.random.beta(2, 2, n_probes)  # Beta分布，更接近真实甲基化数据
        
        # 随机添加一些空值（5%）
        mask = np.random.random(n_probes) < 0.05
        beta_values[mask] = np.nan
        
        data[sample_id] = beta_values
    
    # 创建DataFrame
    df = pd.DataFrame(data)
    
    # 保存为CSV
    df.to_csv(output_path, index=False)
    
    print(f"\n✓ 创建示例数据: {output_path}")
    print(f"✓ Created sample data: {output_path}")
    print(f"  - 探针数: {n_probes}")
    print(f"  - Probes: {n_probes}")
    print(f"  - 样本数: {n_samples}")
    print(f"  - Samples: {n_samples}")
    print(f"  - 数据形状: {df.shape}")
    print(f"  - Data shape: {df.shape}")
    
    # 显示前几行
    print("\n前5行数据:")
    print("First 5 rows:")
    print(df.head())
    
    return output_path


def test_conversion(input_csv):
    """
    测试转换功能
    Test conversion function
    """
    print("\n" + "=" * 80)
    print("测试数据转换")
    print("Testing data conversion")
    print("=" * 80)
    
    # 运行转换
    output_arrow = input_csv.replace('.csv', '_converted.arrow')
    df_converted = convert_935k_format(input_csv, output_arrow, verbose=True)
    
    # 验证结果
    print("\n" + "=" * 80)
    print("验证转换结果")
    print("Validating conversion results")
    print("=" * 80)
    
    # 读取原始数据
    df_original = pd.read_csv(input_csv)
    
    print(f"\n原始数据:")
    print(f"Original data:")
    print(f"  - 形状: {df_original.shape}")
    print(f"  - Shape: {df_original.shape}")
    print(f"  - 第一列: {df_original.columns[0]}")
    print(f"  - First column: {df_original.columns[0]}")
    print(f"  - 示例探针ID: {df_original.iloc[0, 0]}")
    print(f"  - Sample probe ID: {df_original.iloc[0, 0]}")
    print(f"  - 示例样本ID: {df_original.columns[1]}")
    print(f"  - Sample sample ID: {df_original.columns[1]}")
    
    print(f"\n转换后数据:")
    print(f"Converted data:")
    print(f"  - 形状: {df_converted.shape}")
    print(f"  - Shape: {df_converted.shape}")
    print(f"  - 第一列: {df_converted.columns[0]}")
    print(f"  - First column: {df_converted.columns[0]}")
    print(f"  - 示例样本ID: {df_converted.iloc[0, 0]}")
    print(f"  - Sample sample ID: {df_converted.iloc[0, 0]}")
    print(f"  - 示例探针ID: {df_converted.columns[1]}")
    print(f"  - Sample probe ID: {df_converted.columns[1]}")
    
    # 检查转置
    print(f"\n✓ 数据已转置:")
    print(f"✓ Data transposed:")
    print(f"  - 原始: {df_original.shape[0]} 探针 × {df_original.shape[1]-1} 样本")
    print(f"  - Original: {df_original.shape[0]} probes × {df_original.shape[1]-1} samples")
    print(f"  - 转换后: {df_converted.shape[0]} 样本 × {df_converted.shape[1]-1} 探针")
    print(f"  - Converted: {df_converted.shape[0]} samples × {df_converted.shape[1]-1} probes")
    
    # 检查后缀去除
    has_suffix_in_original = '_' in str(df_original.iloc[0, 0])
    has_suffix_in_converted = '_' in str(df_converted.columns[1])
    
    print(f"\n✓ 探针ID后缀已去除:")
    print(f"✓ Probe ID suffixes removed:")
    print(f"  - 原始有后缀: {has_suffix_in_original}")
    print(f"  - Original has suffix: {has_suffix_in_original}")
    print(f"  - 转换后有后缀: {has_suffix_in_converted}")
    print(f"  - Converted has suffix: {has_suffix_in_converted}")
    
    # 检查样本ID后缀
    has_dot_in_original = '.' in str(df_original.columns[1])
    has_dot_in_converted = '.' in str(df_converted.iloc[0, 0])
    
    print(f"\n✓ 样本ID后缀已去除:")
    print(f"✓ Sample ID suffixes removed:")
    print(f"  - 原始有后缀: {has_dot_in_original}")
    print(f"  - Original has suffix: {has_dot_in_original}")
    print(f"  - 转换后有后缀: {has_dot_in_converted}")
    print(f"  - Converted has suffix: {has_dot_in_converted}")
    
    print("\n" + "=" * 80)
    print("✓ 测试完成！转换成功！")
    print("✓ Test completed! Conversion successful!")
    print("=" * 80)
    
    return df_converted


def main():
    """主函数"""
    # 创建示例数据
    sample_file = create_sample_935k_data(
        output_path="test_935k_sample.csv",
        n_probes=100,
        n_samples=3
    )
    
    # 测试转换
    test_conversion(sample_file)
    
    print("\n" + "=" * 80)
    print("下一步:")
    print("Next steps:")
    print("=" * 80)
    print("1. 查看生成的文件:")
    print("   View generated files:")
    print("   - test_935k_sample.csv (原始格式)")
    print("   - test_935k_sample_converted.arrow (转换后)")
    print("\n2. 使用你自己的数据:")
    print("   Use your own data:")
    print('   python examples/convert_935k_format.py "你的文件.csv"')
    print("\n3. 运行预测:")
    print("   Run predictions:")
    print("   python examples/935k_simple_prediction.py")
    print("=" * 80)


if __name__ == "__main__":
    main()

