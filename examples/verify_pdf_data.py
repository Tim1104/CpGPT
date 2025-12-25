#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证PDF数据完整性
检查预测结果中是否包含所有必要的列
"""

import pandas as pd
from pathlib import Path

def verify_prediction_data():
    """验证预测结果数据"""
    
    print("="*60)
    print("验证PDF数据完整性")
    print("="*60)
    
    # 查找预测结果文件
    result_files = list(Path("results/935k_enhanced_predictions").glob("*.csv"))
    
    if not result_files:
        print("❌ 未找到预测结果文件")
        return False
    
    # 读取第一个结果文件
    result_file = result_files[0]
    print(f"\n检查文件: {result_file}")
    
    df = pd.read_csv(result_file)
    print(f"✓ 数据行数: {len(df)}")
    print(f"✓ 数据列数: {len(df.columns)}")
    
    # 检查5种时钟
    print("\n" + "="*60)
    print("检查5种表观遗传时钟")
    print("="*60)
    
    clock_columns = {
        'altumage': 'AltumAge',
        'dunedinpace': 'DunedinPACE',
        'grimage2': 'GrimAge2',
        'hrsinchphenoage': 'PhenoAge',
        'pchorvath2013': 'Horvath2013',
    }
    
    clock_found = 0
    for col, name in clock_columns.items():
        if col in df.columns:
            print(f"✓ {name} ({col}): 存在")
            # 显示前3个值
            values = df[col].head(3).tolist()
            print(f"  示例值: {values}")
            clock_found += 1
        else:
            print(f"❌ {name} ({col}): 不存在")
    
    print(f"\n时钟数据完整性: {clock_found}/5")
    
    # 检查器官健康评分
    print("\n" + "="*60)
    print("检查器官健康评分")
    print("="*60)
    
    organ_systems = {
        'heart': '心脏',
        'kidney': '肾脏',
        'liver': '肝脏',
        'immune': '免疫系统',
        'metabolic': '代谢系统',
        'vascular': '血管系统',
    }
    
    organ_found = 0
    for organ_key, organ_name in organ_systems.items():
        score_col = f'{organ_key}_score'
        level_col = f'{organ_key}_level'
        
        if score_col in df.columns:
            print(f"✓ {organ_name} ({score_col}): 存在")
            # 显示前3个值
            values = df[score_col].head(3).tolist()
            print(f"  示例值: {values}")
            organ_found += 1
        else:
            print(f"❌ {organ_name} ({score_col}): 不存在")
    
    print(f"\n器官健康数据完整性: {organ_found}/6")
    
    # 检查癌症预测
    print("\n" + "="*60)
    print("检查癌症预测")
    print("="*60)
    
    if 'cancer_risk' in df.columns:
        print(f"✓ 癌症风险 (cancer_risk): 存在")
        values = df['cancer_risk'].head(3).tolist()
        print(f"  示例值: {values}")
    else:
        print(f"❌ 癌症风险 (cancer_risk): 不存在")
    
    # 检查年龄预测
    print("\n" + "="*60)
    print("检查年龄预测")
    print("="*60)
    
    if 'predicted_age' in df.columns:
        print(f"✓ 预测年龄 (predicted_age): 存在")
        values = df['predicted_age'].head(3).tolist()
        print(f"  示例值: {values}")
    else:
        print(f"❌ 预测年龄 (predicted_age): 不存在")
    
    # 检查蛋白质预测
    print("\n" + "="*60)
    print("检查蛋白质预测")
    print("="*60)
    
    protein_cols = [col for col in df.columns if col.startswith('protein_')]
    print(f"✓ 找到 {len(protein_cols)} 个蛋白质预测列")
    if protein_cols:
        print(f"  示例: {protein_cols[:5]}")
    
    # 总结
    print("\n" + "="*60)
    print("总结")
    print("="*60)
    
    all_good = (clock_found == 5 and organ_found == 6 and 
                'cancer_risk' in df.columns and 'predicted_age' in df.columns)
    
    if all_good:
        print("✅ 所有数据完整，PDF应该能正常显示所有内容")
    else:
        print("⚠️  部分数据缺失，PDF可能不完整")
        print("\n建议:")
        if clock_found < 5:
            print("  - 检查时钟预测是否正确运行")
        if organ_found < 6:
            print("  - 检查器官健康评分计算是否正确")
    
    return all_good

if __name__ == "__main__":
    success = verify_prediction_data()
    exit(0 if success else 1)

