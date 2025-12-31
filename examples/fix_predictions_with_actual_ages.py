#!/usr/bin/env python3
"""
使用实际年龄修复预测结果

如果你知道样本的实际年龄，这个脚本可以帮你计算正确的标准化参数
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from pathlib import Path

# ============================================================================
# 配置：在这里填入你知道的实际年龄
# ============================================================================

# 方法 1：如果你知道所有样本的实际年龄
KNOWN_AGES = {
    '007012': None,  # 替换为实际年龄，例如: 45
    '000383': None,  # 替换为实际年龄，例如: 55
    '000457': None,
    '000399': None,
    '000698': None,
    '000699': None,
    '000700': None,
}

# 方法 2：如果你知道样本的年龄范围
# 例如：这批样本都是 40-60 岁的健康成年人
EXPECTED_AGE_RANGE = {
    'min': None,  # 例如: 40
    'max': None,  # 例如: 60
    'mean': None,  # 例如: 50（如果知道平均年龄）
}

# 方法 3：如果你有其他检测的结果（例如其他实验室的年龄预测）
OTHER_LAB_RESULTS = {
    '007012': None,  # 其他实验室的年龄预测结果
    '000383': None,
    '000457': None,
    '000399': None,
    '000698': None,
    '000699': None,
    '000700': None,
}

# ============================================================================
# 结果文件路径
# ============================================================================

RESULTS_DIR = Path("results/935k_enhanced_predictions")


def calculate_normalization_params_from_known_ages():
    """从已知年龄计算标准化参数"""
    
    print("=" * 80)
    print("使用实际年龄修复预测结果")
    print("=" * 80)
    
    # 读取预测结果
    age_df = pd.read_csv(RESULTS_DIR / "age_predictions.csv")
    clocks_df = pd.read_csv(RESULTS_DIR / "clocks_predictions.csv")
    
    print(f"\n[1/4] 读取预测结果...")
    print(f"  ✓ 年龄预测: {len(age_df)} 个样本")
    print(f"  ✓ 时钟预测: {len(clocks_df)} 个样本")
    
    # 检查是否有已知年龄
    print(f"\n[2/4] 检查已知年龄...")
    
    known_ages_list = {k: v for k, v in KNOWN_AGES.items() if v is not None}
    other_lab_list = {k: v for k, v in OTHER_LAB_RESULTS.items() if v is not None}
    
    if known_ages_list:
        print(f"  ✓ 找到 {len(known_ages_list)} 个已知年龄")
        reference_ages = known_ages_list
    elif other_lab_list:
        print(f"  ✓ 找到 {len(other_lab_list)} 个其他实验室结果")
        reference_ages = other_lab_list
    elif all(v is not None for v in EXPECTED_AGE_RANGE.values() if v):
        print(f"  ✓ 使用年龄范围估计")
        # 使用预测值的分布来估计
        predicted_ages = age_df.set_index('sample_id')['predicted_age'].to_dict()
        # 线性映射到期望范围
        pred_min = min(predicted_ages.values())
        pred_max = max(predicted_ages.values())
        reference_ages = {
            k: EXPECTED_AGE_RANGE['min'] + (v - pred_min) / (pred_max - pred_min) * 
               (EXPECTED_AGE_RANGE['max'] - EXPECTED_AGE_RANGE['min'])
            for k, v in predicted_ages.items()
        }
    else:
        print(f"  ❌ 错误：未提供任何参考年龄")
        print(f"\n  请在脚本中设置以下之一：")
        print(f"    1. KNOWN_AGES - 已知的实际年龄")
        print(f"    2. OTHER_LAB_RESULTS - 其他实验室的结果")
        print(f"    3. EXPECTED_AGE_RANGE - 期望的年龄范围")
        return
    
    # 计算标准化参数
    print(f"\n[3/4] 计算标准化参数...")
    
    # 准备数据
    sample_ids = list(reference_ages.keys())
    actual_ages = np.array([reference_ages[sid] for sid in sample_ids])
    predicted_ages = np.array([
        age_df[age_df['sample_id'] == sid]['predicted_age'].values[0]
        for sid in sample_ids
    ])
    
    print(f"\n  使用 {len(sample_ids)} 个样本计算参数")
    print(f"  实际年龄范围: {actual_ages.min():.1f} - {actual_ages.max():.1f} 岁")
    print(f"  预测值范围: {predicted_ages.min():.2f} - {predicted_ages.max():.2f}")
    
    # 优化：找到最佳的 mean 和 std
    def loss(params):
        mean, std = params
        denormalized = predicted_ages * std + mean
        return np.mean((denormalized - actual_ages) ** 2)
    
    # 初始猜测
    initial_guess = [np.mean(actual_ages), np.std(actual_ages)]
    
    # 优化
    result = minimize(loss, x0=initial_guess, method='Nelder-Mead')
    mean, std = result.x
    
    # 计算拟合质量
    denormalized = predicted_ages * std + mean
    mse = np.mean((denormalized - actual_ages) ** 2)
    mae = np.mean(np.abs(denormalized - actual_ages))
    r2 = 1 - np.sum((denormalized - actual_ages) ** 2) / np.sum((actual_ages - np.mean(actual_ages)) ** 2)
    
    print(f"\n  计算结果：")
    print(f"    Mean: {mean:.2f}")
    print(f"    Std: {std:.2f}")
    print(f"    MSE: {mse:.2f}")
    print(f"    MAE: {mae:.2f}")
    print(f"    R²: {r2:.3f}")
    
    # 显示对比
    print(f"\n  样本对比：")
    print(f"  {'样本ID':<10} {'实际年龄':<10} {'预测值':<10} {'修正后':<10} {'误差':<10}")
    print(f"  {'-'*50}")
    for sid, actual, pred in zip(sample_ids, actual_ages, predicted_ages):
        corrected = pred * std + mean
        error = corrected - actual
        print(f"  {sid:<10} {actual:<10.1f} {pred:<10.2f} {corrected:<10.1f} {error:+.1f}")
    
    # 生成配置代码
    print(f"\n[4/4] 生成配置代码...")
    print(f"\n" + "=" * 80)
    print(f"复制以下代码到 935k_enhanced_prediction.py 的 NORMALIZATION_PARAMS")
    print(f"=" * 80)
    print(f"""
NORMALIZATION_PARAMS = {{
    'age': {{'mean': {mean:.2f}, 'std': {std:.2f}}},
    'clocks': {{
        'altumage': {{'mean': {mean:.2f}, 'std': {std:.2f}}},
        'dunedinpace': {{'mean': 1.0, 'std': 0.1}},  # DunedinPACE 是速度指标
        'grimage2': {{'mean': {mean:.2f}, 'std': {std:.2f}}},
        'hrsinchphenoage': {{'mean': {mean:.2f}, 'std': {std:.2f}}},
        'pchorvath2013': {{'mean': {mean:.2f}, 'std': {std:.2f}}},
    }},
    'proteins': None,  # 蛋白质保持标准化值
}}
""")
    
    print(f"\n提示：")
    print(f"  1. 复制上面的配置代码")
    print(f"  2. 粘贴到 935k_enhanced_prediction.py 的第 70 行左右")
    print(f"  3. 重新运行预测脚本")
    print(f"  4. 检查新的预测结果")
    print(f"\n" + "=" * 80)


if __name__ == "__main__":
    calculate_normalization_params_from_known_ages()

