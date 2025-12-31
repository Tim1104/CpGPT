#!/usr/bin/env python3
"""
诊断模型预测问题

检查：
1. 模型是否正确加载
2. 预测值的分布
3. 是否所有样本的预测都相似
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# 添加项目路径
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

def main():
    print("=" * 80)
    print("模型诊断工具")
    print("=" * 80)
    
    # 读取元数据和预测结果
    metadata_file = SCRIPT_DIR / "data" / "sample_metadata.csv"
    predictions_dir = SCRIPT_DIR / "results" / "935k_enhanced_predictions"
    
    if not metadata_file.exists():
        print(f"❌ 元数据文件不存在: {metadata_file}")
        return
    
    metadata = pd.read_csv(metadata_file)
    age_pred = pd.read_csv(predictions_dir / "age_predictions.csv")
    cancer_pred = pd.read_csv(predictions_dir / "cancer_predictions.csv")
    
    print(f"\n[1/4] 数据概览...")
    print(f"  样本数量: {len(metadata)}")
    print(f"  实际年龄范围: {metadata['actual_age'].min():.0f} - {metadata['actual_age'].max():.0f}岁")
    print(f"  实际年龄均值: {metadata['actual_age'].mean():.1f}岁")
    print(f"  实际年龄标准差: {metadata['actual_age'].std():.1f}岁")
    
    print(f"\n[2/4] 年龄预测分析...")
    print(f"  预测值范围: {age_pred['predicted_age'].min():.2f} - {age_pred['predicted_age'].max():.2f}岁")
    print(f"  预测值均值: {age_pred['predicted_age'].mean():.2f}岁")
    print(f"  预测值标准差: {age_pred['predicted_age'].std():.2f}岁")
    
    # 检查预测值是否都很相似
    pred_std = age_pred['predicted_age'].std()
    actual_std = metadata['actual_age'].std()
    
    if pred_std < actual_std / 5:
        print(f"\n  ⚠️ 警告：预测值标准差({pred_std:.2f})远小于实际年龄标准差({actual_std:.1f})")
        print(f"  这说明模型把所有人都预测成了相似的年龄！")
        print(f"  可能原因：")
        print(f"    1. 模型没有正确训练")
        print(f"    2. 模型权重没有正确加载")
        print(f"    3. 输入数据有问题")
    
    print(f"\n[3/4] 癌症预测分析...")
    print(f"  预测为癌症的样本: {cancer_pred['cancer_prediction'].sum()}/{len(cancer_pred)}")
    print(f"  癌症概率范围: {cancer_pred['cancer_probability'].min():.3f} - {cancer_pred['cancer_probability'].max():.3f}")
    
    # 检查是否所有人都被预测为癌症
    if cancer_pred['cancer_prediction'].sum() == len(cancer_pred):
        print(f"\n  ⚠️ 警告：所有样本都被预测为癌症！")
        print(f"  这是不正常的，可能原因：")
        print(f"    1. 模型阈值设置错误")
        print(f"    2. 模型输出有偏差")
        print(f"    3. 数据预处理有问题")
    
    print(f"\n[4/4] 详细对比...")
    merged = metadata.merge(age_pred, on='sample_id')
    merged = merged.merge(cancer_pred[['sample_id', 'cancer_probability', 'cancer_prediction']], on='sample_id')
    
    print(f"\n  {'样本ID':<10} {'实际年龄':<10} {'预测年龄':<10} {'误差':<10} {'实际癌症':<10} {'预测癌症':<10}")
    print(f"  {'-'*70}")
    for _, row in merged.iterrows():
        age_error = row['predicted_age'] - row['actual_age']
        actual_cancer = '是' if row['has_cancer'] == 1 else '否'
        pred_cancer = '是' if row['cancer_prediction'] == 1 else '否'
        print(f"  {row['sample_id']:<10} {row['actual_age']:<10.0f} {row['predicted_age']:<10.1f} {age_error:+.1f}岁      {actual_cancer:<10} {pred_cancer:<10}")
    
    # 计算统计
    age_mae = abs(merged['predicted_age'] - merged['actual_age']).mean()
    age_max_error = abs(merged['predicted_age'] - merged['actual_age']).max()
    cancer_accuracy = (merged['has_cancer'] == merged['cancer_prediction']).mean()
    
    print(f"\n" + "=" * 80)
    print(f"诊断总结")
    print(f"=" * 80)
    print(f"\n年龄预测：")
    print(f"  平均绝对误差: {age_mae:.1f}岁")
    print(f"  最大误差: {age_max_error:.1f}岁")
    print(f"  预测值标准差: {age_pred['predicted_age'].std():.2f}岁")
    print(f"  实际值标准差: {metadata['actual_age'].std():.1f}岁")
    print(f"  标准差比率: {age_pred['predicted_age'].std() / metadata['actual_age'].std():.2%}")
    
    print(f"\n癌症预测：")
    print(f"  准确率: {cancer_accuracy*100:.1f}%")
    print(f"  预测为癌症的比例: {cancer_pred['cancer_prediction'].mean()*100:.1f}%")
    print(f"  实际癌症比例: {metadata['has_cancer'].mean()*100:.1f}%")
    
    print(f"\n问题诊断：")
    issues = []
    
    if pred_std < actual_std / 5:
        issues.append("❌ 年龄预测缺乏区分度（所有预测都很相似）")
    
    if age_mae > 10:
        issues.append("❌ 年龄预测误差过大（平均误差 > 10岁）")
    
    if cancer_pred['cancer_prediction'].sum() == len(cancer_pred):
        issues.append("❌ 所有样本都被预测为癌症")
    
    if cancer_accuracy < 0.5:
        issues.append("❌ 癌症预测准确率低于50%（比随机猜测还差）")
    
    if issues:
        for issue in issues:
            print(f"  {issue}")
        print(f"\n建议：")
        print(f"  1. 检查模型文件是否正确")
        print(f"  2. 检查数据预处理是否正确")
        print(f"  3. 尝试重新下载模型权重")
        print(f"  4. 检查输入数据格式")
    else:
        print(f"  ✅ 未发现明显问题")
    
    print(f"\n" + "=" * 80)


if __name__ == "__main__":
    main()

