"""
935k 数据格式转换工具
Convert 935k methylation data from manufacturer format to CpGPT format

输入格式 (Input format):
    TargetID,000536.AVG_Beta,000537.AVG_Beta
    cg00000029_TC21,0.4630385,0.4062999
    cg00000109_TC21,0.8233373,0.8394986
    
输出格式 (Output format):
    sample_id,cg00000029,cg00000109
    000536,0.4630385,0.8233373
    000537,0.4062999,0.8394986
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np

def convert_935k_format(input_csv, output_arrow=None, verbose=True):
    """
    转换935k数据格式
    
    Args:
        input_csv: 输入CSV文件路径
        output_arrow: 输出Arrow文件路径（可选，默认与输入同名）
        verbose: 是否显示详细信息
    
    Returns:
        转换后的DataFrame
    """
    
    if verbose:
        print("=" * 80)
        print("935k 数据格式转换工具")
        print("935k Data Format Converter")
        print("=" * 80)
    
    # 1. 读取数据
    if verbose:
        print(f"\n[1/5] 读取数据: {input_csv}")
        print(f"[1/5] Reading data: {input_csv}")
    
    df = pd.read_csv(input_csv)
    
    if verbose:
        print(f"  - 原始数据形状: {df.shape}")
        print(f"  - Original shape: {df.shape}")
        print(f"  - 探针数量: {len(df)}")
        print(f"  - Number of probes: {len(df)}")
        print(f"  - 样本数量: {len(df.columns) - 1}")
        print(f"  - Number of samples: {len(df.columns) - 1}")
    
    # 2. 清理探针ID（去除后缀）
    if verbose:
        print(f"\n[2/5] 清理探针ID（去除 _TC21, _BC21 等后缀）")
        print(f"[2/5] Cleaning probe IDs (removing _TC21, _BC21 suffixes)")
    
    # 获取第一列名称（可能是 TargetID 或其他）
    probe_col = df.columns[0]
    
    # 去除探针ID的后缀（_TC21, _BC21等）
    df[probe_col] = df[probe_col].str.split('_').str[0]
    
    # 检查是否有重复的探针ID
    duplicates = df[probe_col].duplicated()
    if duplicates.any():
        n_duplicates = duplicates.sum()
        if verbose:
            print(f"  ⚠️  发现 {n_duplicates} 个重复探针（去除后缀后）")
            print(f"  ⚠️  Found {n_duplicates} duplicate probes (after removing suffixes)")
            print(f"  - 将对重复探针取平均值")
            print(f"  - Will average duplicate probes")
        
        # 对重复探针取平均值
        df = df.groupby(probe_col, as_index=False).mean()
    
    if verbose:
        print(f"  ✓ 清理后探针数量: {len(df)}")
        print(f"  ✓ Probes after cleaning: {len(df)}")
    
    # 3. 清理样本ID（去除 .AVG_Beta 等后缀）
    if verbose:
        print(f"\n[3/5] 清理样本ID（去除 .AVG_Beta 等后缀）")
        print(f"[3/5] Cleaning sample IDs (removing .AVG_Beta suffixes)")
    
    # 重命名列
    new_columns = [probe_col]  # 保留第一列名称
    for col in df.columns[1:]:
        # 去除 .AVG_Beta 或其他后缀
        clean_name = col.split('.')[0]
        new_columns.append(clean_name)
    
    df.columns = new_columns
    
    if verbose:
        print(f"  ✓ 样本ID示例: {', '.join(df.columns[1:4].tolist())}")
        print(f"  ✓ Sample ID examples: {', '.join(df.columns[1:4].tolist())}")
    
    # 4. 转置数据（行列互换）
    if verbose:
        print(f"\n[4/5] 转置数据（行=样本，列=探针）")
        print(f"[4/5] Transposing data (rows=samples, columns=probes)")
    
    # 设置探针ID为索引
    df = df.set_index(probe_col)
    
    # 转置
    df_transposed = df.T
    
    # 重置索引，使样本ID成为一列
    df_transposed = df_transposed.reset_index()
    df_transposed = df_transposed.rename(columns={'index': 'sample_id'})
    
    if verbose:
        print(f"  ✓ 转置后形状: {df_transposed.shape}")
        print(f"  ✓ Transposed shape: {df_transposed.shape}")
        print(f"  ✓ 样本数（行）: {len(df_transposed)}")
        print(f"  ✓ Samples (rows): {len(df_transposed)}")
        print(f"  ✓ 探针数（列）: {len(df_transposed.columns) - 1}")
        print(f"  ✓ Probes (columns): {len(df_transposed.columns) - 1}")
    
    # 5. 数据质量检查
    if verbose:
        print(f"\n[5/5] 数据质量检查")
        print(f"[5/5] Data quality check")
    
    # 检查Beta值范围
    beta_cols = df_transposed.columns[1:]  # 除了sample_id的所有列
    beta_values = df_transposed[beta_cols].values.flatten()
    beta_values = beta_values[~np.isnan(beta_values)]  # 去除NaN
    
    if len(beta_values) > 0:
        min_val = beta_values.min()
        max_val = beta_values.max()
        mean_val = beta_values.mean()
        na_count = df_transposed[beta_cols].isna().sum().sum()
        
        if verbose:
            print(f"  - Beta值范围: [{min_val:.4f}, {max_val:.4f}]")
            print(f"  - Beta value range: [{min_val:.4f}, {max_val:.4f}]")
            print(f"  - Beta值平均: {mean_val:.4f}")
            print(f"  - Beta value mean: {mean_val:.4f}")
            print(f"  - 缺失值数量: {na_count}")
            print(f"  - Missing values: {na_count}")
            
            if min_val < 0 or max_val > 1:
                print(f"  ⚠️  警告: Beta值超出正常范围 [0, 1]")
                print(f"  ⚠️  Warning: Beta values outside normal range [0, 1]")
    
    # 6. 保存为Arrow格式
    if output_arrow is None:
        output_arrow = str(Path(input_csv).with_suffix('.arrow'))
    
    if verbose:
        print(f"\n保存转换后的数据到: {output_arrow}")
        print(f"Saving converted data to: {output_arrow}")
    
    df_transposed.to_feather(output_arrow)
    
    if verbose:
        print(f"✓ 转换完成！")
        print(f"✓ Conversion completed!")
        print(f"\n下一步: 使用此文件运行预测")
        print(f"Next step: Use this file to run predictions")
        print(f"  python examples/935k_simple_prediction.py")
    
    return df_transposed


def main():
    parser = argparse.ArgumentParser(
        description='Convert 935k methylation data to CpGPT format'
    )
    parser.add_argument(
        'input',
        type=str,
        help='Input CSV file path'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output Arrow file path (default: same as input with .arrow extension)'
    )
    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Quiet mode (no verbose output)'
    )
    
    args = parser.parse_args()
    
    try:
        convert_935k_format(
            args.input,
            args.output,
            verbose=not args.quiet
        )
    except Exception as e:
        print(f"\n错误 Error: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

