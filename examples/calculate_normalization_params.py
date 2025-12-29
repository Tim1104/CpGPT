"""
计算标准化参数的辅助脚本
用于从已知样本反推模型的标准化参数

使用方法：
1. 运行 935k_enhanced_prediction.py 获取预测结果
2. 准备一个包含实际年龄的 CSV 文件
3. 运行此脚本计算标准化参数
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from pathlib import Path

# ============================================================================
# 配置
# ============================================================================

# 预测结果文件路径
PREDICTIONS_FILE = "results/935k_enhanced_predictions/age_predictions.csv"

# 实际年龄数据（两种方式二选一）

# 方式 1：直接在这里输入（样本ID和实际年龄）
KNOWN_AGES = {
    # 'sample_id': actual_age
    # 'Sample1': 45,
    # 'Sample2': 55,
    # 'Sample3': 50,
}

# 方式 2：从 CSV 文件读取（包含 'sample_id' 和 'age' 列）
KNOWN_AGES_FILE = None  # 例如: "data/known_ages.csv"

# ============================================================================
# 主程序
# ============================================================================

def calculate_normalization_params(predicted_values, actual_values):
    """
    从预测值和实际值反推标准化参数
    
    假设：predicted = (actual - mean) / std
    反推：actual = predicted * std + mean
    
    使用最小二乘法找到最佳的 mean 和 std
    """
    predicted = np.array(predicted_values)
    actual = np.array(actual_values)
    
    # 定义损失函数
    def loss(params):
        mean, std = params
        denormalized = predicted * std + mean
        return np.mean((denormalized - actual) ** 2)
    
    # 初始猜测：使用实际值的均值和标准差
    initial_guess = [np.mean(actual), np.std(actual)]
    
    # 优化
    result = minimize(loss, x0=initial_guess, method='Nelder-Mead')
    
    mean, std = result.x
    
    # 计算拟合质量
    denormalized = predicted * std + mean
    mse = np.mean((denormalized - actual) ** 2)
    mae = np.mean(np.abs(denormalized - actual))
    r2 = 1 - np.sum((denormalized - actual) ** 2) / np.sum((actual - np.mean(actual)) ** 2)
    
    return {
        'mean': mean,
        'std': std,
        'mse': mse,
        'mae': mae,
        'r2': r2,
        'denormalized': denormalized
    }


def main():
    print("=" * 80)
    print("标准化参数计算工具")
    print("=" * 80)
    
    # 读取预测结果
    print(f"\n[1/4] 读取预测结果: {PREDICTIONS_FILE}")
    if not Path(PREDICTIONS_FILE).exists():
        print(f"❌ 错误：文件不存在 {PREDICTIONS_FILE}")
        print("请先运行 935k_enhanced_prediction.py 生成预测结果")
        return
    
    predictions_df = pd.read_csv(PREDICTIONS_FILE)
    print(f"  ✓ 读取了 {len(predictions_df)} 个样本的预测结果")
    
    # 读取实际年龄
    print("\n[2/4] 读取实际年龄数据")
    
    if KNOWN_AGES_FILE:
        print(f"  从文件读取: {KNOWN_AGES_FILE}")
        known_ages_df = pd.read_csv(KNOWN_AGES_FILE)
        known_ages = dict(zip(known_ages_df['sample_id'], known_ages_df['age']))
    elif KNOWN_AGES:
        print(f"  从配置读取")
        known_ages = KNOWN_AGES
    else:
        print("❌ 错误：请设置 KNOWN_AGES 或 KNOWN_AGES_FILE")
        print("\n示例配置：")
        print("KNOWN_AGES = {")
        print("    'Sample1': 45,")
        print("    'Sample2': 55,")
        print("    'Sample3': 50,")
        print("}")
        return
    
    print(f"  ✓ 读取了 {len(known_ages)} 个已知年龄的样本")
    
    # 匹配样本
    print("\n[3/4] 匹配预测值和实际值")
    matched_data = []
    for sample_id, actual_age in known_ages.items():
        pred_row = predictions_df[predictions_df['sample_id'] == sample_id]
        if not pred_row.empty:
            predicted_age = pred_row['predicted_age'].values[0]
            matched_data.append({
                'sample_id': sample_id,
                'predicted': predicted_age,
                'actual': actual_age
            })
    
    if len(matched_data) == 0:
        print("❌ 错误：没有找到匹配的样本")
        print("请检查 sample_id 是否正确")
        return
    
    print(f"  ✓ 成功匹配 {len(matched_data)} 个样本")
    
    # 计算标准化参数
    print("\n[4/4] 计算标准化参数")
    matched_df = pd.DataFrame(matched_data)
    
    result = calculate_normalization_params(
        matched_df['predicted'].values,
        matched_df['actual'].values
    )
    
    print("\n" + "=" * 80)
    print("计算结果")
    print("=" * 80)
    print(f"\n标准化参数：")
    print(f"  Mean (均值): {result['mean']:.2f}")
    print(f"  Std (标准差): {result['std']:.2f}")
    
    print(f"\n拟合质量：")
    print(f"  R² (决定系数): {result['r2']:.4f}")
    print(f"  MAE (平均绝对误差): {result['mae']:.2f} 岁")
    print(f"  MSE (均方误差): {result['mse']:.2f}")
    
    print(f"\n样本对比：")
    comparison_df = matched_df.copy()
    comparison_df['denormalized'] = result['denormalized']
    comparison_df['error'] = comparison_df['denormalized'] - comparison_df['actual']
    print(comparison_df.to_string(index=False))
    
    print("\n" + "=" * 80)
    print("配置代码（复制到 935k_enhanced_prediction.py）")
    print("=" * 80)
    print(f"""
NORMALIZATION_PARAMS = {{
    'age': {{'mean': {result['mean']:.2f}, 'std': {result['std']:.2f}}},
    'clocks': {{
        'altumage': {{'mean': {result['mean']:.2f}, 'std': {result['std']:.2f}}},
        'dunedinpace': {{'mean': 1.0, 'std': 0.1}},  # DunedinPACE 是速度指标
        'grimage2': {{'mean': {result['mean']:.2f}, 'std': {result['std']:.2f}}},
        'hrsinchphenoage': {{'mean': {result['mean']:.2f}, 'std': {result['std']:.2f}}},
        'pchorvath2013': {{'mean': {result['mean']:.2f}, 'std': {result['std']:.2f}}},
    }},
    'proteins': None,
}}
""")
    
    print("\n提示：")
    print("  1. 复制上面的配置代码到 935k_enhanced_prediction.py")
    print("  2. 重新运行预测脚本")
    print("  3. 检查新的预测结果是否合理")
    print()


if __name__ == "__main__":
    main()

