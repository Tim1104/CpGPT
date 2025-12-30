#!/usr/bin/env python3
"""
蛋白质预测结果检查工具

用于验证蛋白质预测结果是否合理，并提供详细的统计分析。
"""

import pandas as pd
import numpy as np
from pathlib import Path

# 配置
PREDICTIONS_FILE = "results/935k_enhanced_predictions/proteins_predictions.csv"

# 关键蛋白质分组
PROTEIN_GROUPS = {
    '炎症标志物': ['CRP', 'IL6', 'TNF_alpha', 'GDF15'],
    '心血管标志物': ['ADM', 'ICAM1', 'VCAM1', 'PAI1', 'E_selectin', 'P_selectin'],
    '肾功能标志物': ['Cystatin_C', 'B2M'],
    '代谢标志物': ['Leptin', 'GDF15'],
    '凝血标志物': ['Fibrinogen', 'vWF', 'D_dimer', 'PAI1'],
}


def check_protein_predictions():
    """检查蛋白质预测结果"""
    
    print("=" * 80)
    print("蛋白质预测结果检查工具")
    print("=" * 80)
    
    # 检查文件是否存在
    if not Path(PREDICTIONS_FILE).exists():
        print(f"\n❌ 错误：文件不存在 {PREDICTIONS_FILE}")
        print("请先运行 935k_enhanced_prediction.py 生成预测结果")
        return
    
    # 读取数据
    print(f"\n[1/5] 读取预测结果: {PREDICTIONS_FILE}")
    df = pd.read_csv(PREDICTIONS_FILE)
    print(f"  ✓ 读取了 {len(df)} 个样本")
    
    # 获取蛋白质列
    protein_cols = [col for col in df.columns if col != 'sample_id']
    print(f"  ✓ 预测了 {len(protein_cols)} 种蛋白质")
    
    # 基本统计
    print(f"\n[2/5] 基本统计分析")
    proteins_data = df[protein_cols]
    
    print(f"\n  所有蛋白质的统计：")
    print(f"    均值: {proteins_data.mean().mean():.3f}")
    print(f"    标准差: {proteins_data.std().mean():.3f}")
    print(f"    最小值: {proteins_data.min().min():.3f}")
    print(f"    最大值: {proteins_data.max().max():.3f}")
    
    # 检查是否在合理范围内
    print(f"\n  ✓ 合理性检查：")
    
    # 检查均值是否接近 0
    overall_mean = proteins_data.mean().mean()
    if abs(overall_mean) < 0.5:
        print(f"    ✅ 均值接近 0 ({overall_mean:.3f}) - 符合标准化预期")
    else:
        print(f"    ⚠️ 均值偏离 0 ({overall_mean:.3f}) - 可能需要检查")
    
    # 检查标准差是否接近 1
    overall_std = proteins_data.std().mean()
    if 0.5 < overall_std < 1.5:
        print(f"    ✅ 标准差合理 ({overall_std:.3f}) - 符合标准化预期")
    else:
        print(f"    ⚠️ 标准差异常 ({overall_std:.3f}) - 可能需要检查")
    
    # 检查极端值
    extreme_low = (proteins_data < -5).sum().sum()
    extreme_high = (proteins_data > 5).sum().sum()
    total_values = len(df) * len(protein_cols)
    
    if extreme_low + extreme_high < total_values * 0.01:
        print(f"    ✅ 极端值比例正常 ({(extreme_low + extreme_high) / total_values * 100:.2f}%)")
    else:
        print(f"    ⚠️ 极端值过多 ({(extreme_low + extreme_high) / total_values * 100:.2f}%)")
        print(f"       < -5: {extreme_low} 个值")
        print(f"       > +5: {extreme_high} 个值")
    
    # 分组分析
    print(f"\n[3/5] 关键蛋白质分组分析")
    
    for group_name, protein_list in PROTEIN_GROUPS.items():
        available_proteins = [p for p in protein_list if p in df.columns]
        if not available_proteins:
            continue
        
        group_data = df[available_proteins]
        group_mean = group_data.mean().mean()
        
        print(f"\n  {group_name}:")
        print(f"    可用蛋白质: {len(available_proteins)}/{len(protein_list)}")
        print(f"    平均值: {group_mean:.3f}")
        
        # 解释
        if group_mean < -0.5:
            print(f"    💚 整体低于平均水平 - 健康状态良好")
        elif group_mean > 0.5:
            print(f"    ⚠️ 整体高于平均水平 - 需要关注")
        else:
            print(f"    ✓ 整体接近平均水平")
    
    # 样本级别分析
    print(f"\n[4/5] 样本级别分析")
    
    for idx, row in df.iterrows():
        sample_id = row['sample_id']
        protein_values = row[protein_cols]
        
        # 统计
        high_count = (protein_values > 2).sum()
        low_count = (protein_values < -2).sum()
        extreme_high = (protein_values > 3).sum()
        extreme_low = (protein_values < -3).sum()
        
        print(f"\n  样本: {sample_id}")
        print(f"    平均值: {protein_values.mean():.3f}")
        print(f"    高于 +2σ: {high_count} 个蛋白质 ({high_count/len(protein_cols)*100:.1f}%)")
        print(f"    低于 -2σ: {low_count} 个蛋白质 ({low_count/len(protein_cols)*100:.1f}%)")
        
        if extreme_high > 0 or extreme_low > 0:
            print(f"    ⚠️ 极端值: {extreme_high} 个 > +3σ, {extreme_low} 个 < -3σ")
        
        # 健康评估
        if protein_values.mean() < -0.3:
            print(f"    💚 整体健康状态良好")
        elif protein_values.mean() > 0.3:
            print(f"    ⚠️ 整体风险偏高，建议关注")
        else:
            print(f"    ✓ 整体健康状态正常")
    
    # 建议
    print(f"\n[5/5] 建议")
    print(f"\n  ✅ 蛋白质预测结果看起来合理")
    print(f"\n  📖 如何解读标准化值：")
    print(f"     • 负值（< 0）：低于人群平均水平（通常更健康）")
    print(f"     • 0：人群平均水平")
    print(f"     • 正值（> 0）：高于人群平均水平（可能有风险）")
    print(f"\n  📖 详细解读指南：")
    print(f"     请参考 PROTEIN_PREDICTION_GUIDE.md")
    print(f"\n  💡 下一步：")
    print(f"     1. 查看 PDF 报告中的器官健康评分")
    print(f"     2. 关注异常升高的蛋白质")
    print(f"     3. 如有需要，咨询医疗专业人士")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    check_protein_predictions()

