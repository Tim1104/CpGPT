#!/usr/bin/env python3
"""
预测结果诊断工具

分析预测结果，找出可能的问题
"""

import pandas as pd
import numpy as np
from pathlib import Path

# 配置
RESULTS_DIR = Path("results/935k_enhanced_predictions")

def diagnose_predictions():
    """诊断预测结果"""
    
    print("=" * 80)
    print("预测结果诊断工具")
    print("=" * 80)
    
    # 读取所有结果
    print("\n[1/5] 读取预测结果...")
    
    age_df = pd.read_csv(RESULTS_DIR / "age_predictions.csv")
    clocks_df = pd.read_csv(RESULTS_DIR / "clocks_predictions.csv")
    proteins_df = pd.read_csv(RESULTS_DIR / "proteins_predictions.csv")
    
    print(f"  ✓ 年龄预测: {len(age_df)} 个样本")
    print(f"  ✓ 时钟预测: {len(clocks_df)} 个样本")
    print(f"  ✓ 蛋白质预测: {len(proteins_df)} 个样本")
    
    # 分析年龄预测
    print("\n[2/5] 分析年龄预测...")
    print(f"\n  年龄预测统计：")
    print(f"    均值: {age_df['predicted_age'].mean():.2f} 岁")
    print(f"    标准差: {age_df['predicted_age'].std():.2f} 岁")
    print(f"    范围: {age_df['predicted_age'].min():.2f} - {age_df['predicted_age'].max():.2f} 岁")
    
    print(f"\n  ⚠️ 诊断：")
    if age_df['predicted_age'].mean() < 30 or age_df['predicted_age'].mean() > 80:
        print(f"    ❌ 年龄均值异常 ({age_df['predicted_age'].mean():.2f} 岁)")
        print(f"    💡 可能原因：缺少反标准化参数")
        print(f"    💡 解决方案：设置 NORMALIZATION_PARAMS['age']")
    else:
        print(f"    ✓ 年龄均值正常")
    
    if age_df['predicted_age'].std() < 5 or age_df['predicted_age'].std() > 30:
        print(f"    ⚠️ 年龄标准差异常 ({age_df['predicted_age'].std():.2f} 岁)")
    else:
        print(f"    ✓ 年龄标准差正常")
    
    # 分析时钟预测
    print("\n[3/5] 分析表观遗传时钟...")
    
    clock_cols = ['altumage', 'grimage2', 'hrsinchphenoage', 'pchorvath2013']
    
    for clock in clock_cols:
        if clock in clocks_df.columns:
            mean_val = clocks_df[clock].mean()
            print(f"\n  {clock}:")
            print(f"    均值: {mean_val:.2f}")
            print(f"    范围: {clocks_df[clock].min():.2f} - {clocks_df[clock].max():.2f}")
            
            # 检查是否需要反标准化
            if abs(mean_val) < 5:
                print(f"    ❌ 可能是标准化值（均值接近0）")
                print(f"    💡 需要反标准化")
            elif 20 < mean_val < 90:
                print(f"    ✓ 看起来像实际年龄")
            else:
                print(f"    ⚠️ 值异常")
    
    # DunedinPACE 特殊处理
    if 'dunedinpace' in clocks_df.columns:
        pace_mean = clocks_df['dunedinpace'].mean()
        print(f"\n  dunedinpace (衰老速度):")
        print(f"    均值: {pace_mean:.2f}")
        print(f"    范围: {clocks_df['dunedinpace'].min():.2f} - {clocks_df['dunedinpace'].max():.2f}")
        
        if 0.8 < pace_mean < 1.2:
            print(f"    ✓ 正常范围（1.0 = 正常衰老速度）")
        elif abs(pace_mean) < 0.5:
            print(f"    ❌ 可能是标准化值")
            print(f"    💡 需要反标准化")
        else:
            print(f"    ⚠️ 值异常")
    
    # 分析蛋白质预测
    print("\n[4/5] 分析蛋白质预测...")
    
    protein_cols = [col for col in proteins_df.columns if col != 'sample_id']
    protein_data = proteins_df[protein_cols]
    
    print(f"\n  蛋白质预测统计（{len(protein_cols)} 种蛋白质）：")
    print(f"    全局均值: {protein_data.mean().mean():.3f}")
    print(f"    全局标准差: {protein_data.std().mean():.3f}")
    print(f"    最小值: {protein_data.min().min():.3f}")
    print(f"    最大值: {protein_data.max().max():.3f}")
    
    print(f"\n  ✓ 诊断：")
    overall_mean = protein_data.mean().mean()
    if abs(overall_mean) < 0.5:
        print(f"    ✅ 蛋白质均值接近 0 ({overall_mean:.3f}) - 这是正常的标准化值")
    else:
        print(f"    ⚠️ 蛋白质均值偏离 0 ({overall_mean:.3f})")
    
    # 检查关键炎症标志物
    inflammation_markers = ['CRP', 'IL6', 'TNF_alpha', 'GDF15']
    print(f"\n  关键炎症标志物：")
    for marker in inflammation_markers:
        if marker in proteins_df.columns:
            mean_val = proteins_df[marker].mean()
            print(f"    {marker}: {mean_val:.3f}", end="")
            if mean_val < -0.5:
                print(f" ✅ (低于平均，健康)")
            elif mean_val > 0.5:
                print(f" ⚠️ (高于平均，需要关注)")
            else:
                print(f" ✓ (正常)")
    
    # 对比年龄和时钟
    print("\n[5/5] 对比年龄预测和表观遗传时钟...")
    
    merged = age_df.merge(clocks_df, on='sample_id')
    
    for clock in clock_cols:
        if clock in merged.columns:
            corr = merged['predicted_age'].corr(merged[clock])
            print(f"\n  predicted_age vs {clock}:")
            print(f"    相关性: {corr:.3f}", end="")
            if corr > 0.8:
                print(f" ✅ (高度相关)")
            elif corr > 0.5:
                print(f" ✓ (中度相关)")
            else:
                print(f" ⚠️ (相关性低)")
    
    # 总结
    print("\n" + "=" * 80)
    print("诊断总结")
    print("=" * 80)
    
    print("\n📊 当前状态：")
    print(f"  • 年龄预测均值: {age_df['predicted_age'].mean():.2f} 岁")
    print(f"  • 蛋白质均值: {protein_data.mean().mean():.3f} (标准化值)")
    
    print("\n⚠️ 发现的问题：")
    
    issues = []
    
    # 检查年龄
    if age_df['predicted_age'].mean() < 30 or age_df['predicted_age'].mean() > 80:
        issues.append("年龄预测可能缺少反标准化")
    
    # 检查时钟
    for clock in clock_cols:
        if clock in clocks_df.columns:
            if abs(clocks_df[clock].mean()) < 5:
                issues.append(f"{clock} 可能缺少反标准化")
    
    if issues:
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
    else:
        print(f"  ✅ 未发现明显问题")
    
    print("\n💡 建议：")
    if issues:
        print(f"  1. 检查 NORMALIZATION_PARAMS 配置")
        print(f"  2. 使用 calculate_normalization_params.py 计算参数")
        print(f"  3. 参考 PREDICTION_FIX_SUMMARY.md")
    else:
        print(f"  ✅ 预测结果看起来正常")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    diagnose_predictions()

