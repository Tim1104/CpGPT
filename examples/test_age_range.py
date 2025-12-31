#!/usr/bin/env python3
"""
测试模型的年龄范围限制

检查模型在不同年龄段的预测准确性
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    # 读取数据
    metadata = pd.read_csv('data/sample_metadata.csv')
    predictions = pd.read_csv('results/935k_enhanced_predictions/age_predictions.csv')
    
    # 合并
    merged = metadata.merge(predictions, on='sample_id')
    merged['error'] = merged['predicted_age'] - merged['actual_age']
    merged['abs_error'] = merged['error'].abs()
    
    # 按年龄分组
    print("=" * 60)
    print("年龄范围分析")
    print("=" * 60)
    
    age_groups = [
        ("年轻人 (<30岁)", merged[merged['actual_age'] < 30]),
        ("中年人 (30-60岁)", merged[(merged['actual_age'] >= 30) & (merged['actual_age'] < 60)]),
        ("老年人 (>=60岁)", merged[merged['actual_age'] >= 60]),
    ]
    
    for group_name, group_data in age_groups:
        if len(group_data) > 0:
            print(f"\n{group_name}:")
            print(f"  样本数: {len(group_data)}")
            print(f"  实际年龄范围: {group_data['actual_age'].min():.0f} - {group_data['actual_age'].max():.0f}岁")
            print(f"  预测年龄范围: {group_data['predicted_age'].min():.1f} - {group_data['predicted_age'].max():.1f}岁")
            print(f"  平均误差: {group_data['abs_error'].mean():.1f}岁")
            print(f"  最大误差: {group_data['abs_error'].max():.1f}岁")
    
    # 绘图
    plt.figure(figsize=(12, 5))
    
    # 子图1：实际 vs 预测
    plt.subplot(1, 2, 1)
    plt.scatter(merged['actual_age'], merged['predicted_age'], s=100, alpha=0.6)
    plt.plot([20, 90], [20, 90], 'r--', label='完美预测线')
    plt.xlabel('实际年龄')
    plt.ylabel('预测年龄')
    plt.title('实际年龄 vs 预测年龄')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 添加样本标签
    for _, row in merged.iterrows():
        plt.annotate(f"{row['sample_id']}", 
                    (row['actual_age'], row['predicted_age']),
                    fontsize=8, alpha=0.7)
    
    # 子图2：误差分布
    plt.subplot(1, 2, 2)
    plt.bar(range(len(merged)), merged['error'], alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('样本')
    plt.ylabel('预测误差（岁）')
    plt.title('预测误差分布')
    plt.xticks(range(len(merged)), merged['sample_id'], rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('results/935k_enhanced_predictions/age_range_analysis.png', dpi=150)
    print(f"\n✓ 图表已保存: results/935k_enhanced_predictions/age_range_analysis.png")
    
    # 结论
    print("\n" + "=" * 60)
    print("结论")
    print("=" * 60)
    
    # 检查是否有年龄范围限制
    young_error = merged[merged['actual_age'] < 30]['abs_error'].mean() if len(merged[merged['actual_age'] < 30]) > 0 else 0
    middle_error = merged[(merged['actual_age'] >= 30) & (merged['actual_age'] < 60)]['abs_error'].mean() if len(merged[(merged['actual_age'] >= 30) & (merged['actual_age'] < 60)]) > 0 else 0
    old_error = merged[merged['actual_age'] >= 60]['abs_error'].mean() if len(merged[merged['actual_age'] >= 60]) > 0 else 0
    
    print(f"\n各年龄段平均误差:")
    print(f"  年轻人 (<30岁): {young_error:.1f}岁")
    print(f"  中年人 (30-60岁): {middle_error:.1f}岁")
    print(f"  老年人 (>=60岁): {old_error:.1f}岁")
    
    if young_error > middle_error * 2 or old_error > middle_error * 2:
        print(f"\n⚠️ 发现：模型在极端年龄段（<30岁或>=60岁）的误差明显更大")
        print(f"这说明模型可能在30-60岁的数据上训练，对极端年龄预测不准确")
        print(f"\n建议：")
        print(f"  1. 如果你的样本主要是30-60岁，模型应该能正常工作")
        print(f"  2. 如果需要预测极端年龄，考虑使用其他模型")
        print(f"  3. 或者在你的数据上重新训练模型")
    else:
        print(f"\n✓ 模型在各年龄段的表现相对均衡")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()

