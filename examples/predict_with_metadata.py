#!/usr/bin/env python3
"""
带元数据的增强预测脚本

功能：
1. 从 metadata.csv 读取实际年龄和癌症状态
2. 自动计算标准化参数
3. 运行预测
4. 生成对比报告（预测 vs 实际）
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.optimize import minimize
import sys

# 添加项目路径
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================================
# 配置
# ============================================================================

# 元数据文件路径
METADATA_FILE = SCRIPT_DIR / "data" / "sample_metadata.csv"

# 是否自动计算标准化参数
AUTO_CALCULATE_NORMALIZATION = True

# 如果没有元数据，使用默认参数
DEFAULT_NORMALIZATION_PARAMS = {
    'age': {'mean': 50.0, 'std': 15.0},
    'clocks': {
        'altumage': {'mean': 50.0, 'std': 15.0},
        'dunedinpace': {'mean': 1.0, 'std': 0.1},
        'grimage2': {'mean': 50.0, 'std': 15.0},
        'hrsinchphenoage': {'mean': 50.0, 'std': 15.0},
        'pchorvath2013': {'mean': 50.0, 'std': 15.0},
    },
    'proteins': None,
}


def load_metadata():
    """加载元数据"""
    if not METADATA_FILE.exists():
        print(f"⚠️ 元数据文件不存在: {METADATA_FILE}")
        print(f"💡 请创建 sample_metadata.csv 文件，包含以下列：")
        print(f"   - sample_id: 样本ID")
        print(f"   - actual_age: 实际年龄")
        print(f"   - has_cancer: 是否有癌症 (0/1 或 True/False)")
        return None
    
    metadata = pd.read_csv(METADATA_FILE)
    
    # 检查必需的列
    required_cols = ['sample_id']
    missing_cols = [col for col in required_cols if col not in metadata.columns]
    if missing_cols:
        print(f"❌ 元数据文件缺少必需的列: {missing_cols}")
        return None
    
    # 检查可选列
    has_age = 'actual_age' in metadata.columns
    has_cancer = 'has_cancer' in metadata.columns
    
    print(f"✓ 加载元数据: {len(metadata)} 个样本")
    if has_age:
        valid_ages = metadata['actual_age'].notna().sum()
        print(f"  - 有实际年龄的样本: {valid_ages}/{len(metadata)}")
    if has_cancer:
        valid_cancer = metadata['has_cancer'].notna().sum()
        print(f"  - 有癌症状态的样本: {valid_cancer}/{len(metadata)}")
    
    return metadata


def calculate_normalization_params(metadata, predictions_df):
    """从元数据和预测结果计算标准化参数"""
    
    if metadata is None or 'actual_age' not in metadata.columns:
        print(f"⚠️ 无法计算标准化参数：缺少实际年龄数据")
        return None
    
    # 合并数据
    merged = predictions_df.merge(metadata[['sample_id', 'actual_age']], on='sample_id')
    merged = merged[merged['actual_age'].notna()]
    
    if len(merged) < 2:
        print(f"⚠️ 无法计算标准化参数：至少需要 2 个有实际年龄的样本")
        return None
    
    print(f"\n计算标准化参数（使用 {len(merged)} 个样本）...")
    
    # 准备数据
    actual_ages = merged['actual_age'].values
    predicted_values = merged['predicted_age'].values
    
    # 优化：找到最佳的 mean 和 std
    def loss(params):
        mean, std = params
        denormalized = predicted_values * std + mean
        return np.mean((denormalized - actual_ages) ** 2)
    
    # 初始猜测
    initial_guess = [np.mean(actual_ages), np.std(actual_ages)]
    
    # 优化
    result = minimize(loss, x0=initial_guess, method='Nelder-Mead')
    mean, std = result.x
    
    # 计算拟合质量
    denormalized = predicted_values * std + mean
    mse = np.mean((denormalized - actual_ages) ** 2)
    mae = np.mean(np.abs(denormalized - actual_ages))
    r2 = 1 - np.sum((denormalized - actual_ages) ** 2) / np.sum((actual_ages - np.mean(actual_ages)) ** 2)
    
    print(f"  Mean: {mean:.2f}")
    print(f"  Std: {std:.2f}")
    print(f"  MAE: {mae:.2f} 岁")
    print(f"  R²: {r2:.3f}")
    
    return {
        'age': {'mean': mean, 'std': std},
        'clocks': {
            'altumage': {'mean': mean, 'std': std},
            'dunedinpace': {'mean': 1.0, 'std': 0.1},
            'grimage2': {'mean': mean, 'std': std},
            'hrsinchphenoage': {'mean': mean, 'std': std},
            'pchorvath2013': {'mean': mean, 'std': std},
        },
        'proteins': None,
    }


def generate_comparison_report(metadata, predictions_dir):
    """生成对比报告"""
    
    if metadata is None:
        return
    
    print(f"\n生成对比报告...")
    
    # 读取预测结果
    age_pred = pd.read_csv(predictions_dir / "age_predictions.csv")
    cancer_pred = pd.read_csv(predictions_dir / "cancer_predictions.csv")
    
    # 合并数据
    comparison = metadata.copy()
    comparison = comparison.merge(age_pred, on='sample_id', how='left')
    comparison = comparison.merge(
        cancer_pred[['sample_id', 'cancer_probability', 'cancer_prediction']], 
        on='sample_id', 
        how='left'
    )
    
    # 计算误差
    if 'actual_age' in comparison.columns:
        comparison['age_error'] = comparison['predicted_age'] - comparison['actual_age']
        comparison['age_abs_error'] = comparison['age_error'].abs()
    
    # 保存对比报告
    output_file = predictions_dir / "comparison_report.csv"
    comparison.to_csv(output_file, index=False)
    print(f"  ✓ 对比报告已保存: {output_file}")
    
    # 打印统计
    if 'actual_age' in comparison.columns:
        valid_ages = comparison[comparison['actual_age'].notna()]
        if len(valid_ages) > 0:
            print(f"\n年龄预测准确性：")
            print(f"  平均绝对误差: {valid_ages['age_abs_error'].mean():.2f} 岁")
            print(f"  最大误差: {valid_ages['age_abs_error'].max():.2f} 岁")
            print(f"  相关系数: {valid_ages['actual_age'].corr(valid_ages['predicted_age']):.3f}")
    
    if 'has_cancer' in comparison.columns:
        valid_cancer = comparison[comparison['has_cancer'].notna()]
        if len(valid_cancer) > 0:
            # 转换为 0/1
            valid_cancer['has_cancer_binary'] = valid_cancer['has_cancer'].astype(int)
            accuracy = (valid_cancer['has_cancer_binary'] == valid_cancer['cancer_prediction']).mean()
            print(f"\n癌症预测准确性：")
            print(f"  准确率: {accuracy*100:.1f}%")
    
    return comparison


def main():
    """主函数"""
    
    print("=" * 80)
    print("带元数据的增强预测")
    print("=" * 80)
    
    # 加载元数据
    print(f"\n[1/3] 加载元数据...")
    metadata = load_metadata()
    
    # 运行预测（调用原始脚本）
    print(f"\n[2/3] 运行预测...")
    print(f"💡 请先运行 935k_enhanced_prediction.py 生成预测结果")
    print(f"   然后再运行此脚本生成对比报告")
    
    # 检查预测结果是否存在
    predictions_dir = SCRIPT_DIR / "results" / "935k_enhanced_predictions"
    age_pred_file = predictions_dir / "age_predictions.csv"
    
    if not age_pred_file.exists():
        print(f"\n❌ 预测结果不存在: {age_pred_file}")
        print(f"💡 请先运行: python 935k_enhanced_prediction.py")
        return
    
    # 读取预测结果
    age_pred = pd.read_csv(age_pred_file)
    
    # 如果有元数据且启用自动计算，计算标准化参数
    if metadata is not None and AUTO_CALCULATE_NORMALIZATION and 'actual_age' in metadata.columns:
        norm_params = calculate_normalization_params(metadata, age_pred)
        
        if norm_params:
            print(f"\n✓ 标准化参数已计算")
            print(f"💡 将以下代码复制到 935k_enhanced_prediction.py:")
            print(f"\nNORMALIZATION_PARAMS = {norm_params}")
            print(f"\n然后重新运行预测以获得准确结果")
    
    # 生成对比报告
    print(f"\n[3/3] 生成对比报告...")
    comparison = generate_comparison_report(metadata, predictions_dir)
    
    print(f"\n" + "=" * 80)
    print(f"完成！")
    print(f"=" * 80)


if __name__ == "__main__":
    main()

