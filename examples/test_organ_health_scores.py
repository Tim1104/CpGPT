"""
测试器官健康评分功能
Test organ health scores functionality
"""

import pandas as pd
import numpy as np

# 模拟蛋白质预测数据
def create_mock_protein_data(n_samples=5):
    """创建模拟的蛋白质数据用于测试"""
    
    # 蛋白质名称
    proteins = [
        'ADM', 'B2M', 'Cystatin_C', 'GDF15', 'Leptin', 'PAI1', 'TIMP1',
        'CRP', 'IL6', 'TNF_alpha', 'MMP1', 'MMP9', 'VEGF', 'ICAM1',
        'VCAM1', 'E_selectin', 'P_selectin', 'Fibrinogen', 'vWF', 'D_dimer',
    ]
    
    # 创建随机数据（标准化值，均值0，标准差1）
    data = {'sample_id': [f'Sample_{i+1}' for i in range(n_samples)]}
    
    for protein in proteins:
        # 生成随机标准化值（-3到3之间）
        data[protein] = np.random.randn(n_samples)
    
    return pd.DataFrame(data)


def test_organ_health_scores():
    """测试器官健康评分计算"""
    
    print("=" * 80)
    print("测试器官健康评分功能")
    print("=" * 80)
    
    # 导入函数
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    
    # 从 935k_enhanced_prediction.py 导入函数
    from importlib import import_module
    spec = import_module('935k_enhanced_prediction')
    
    get_organ_specific_proteins = spec.get_organ_specific_proteins
    calculate_organ_health_scores = spec.calculate_organ_health_scores
    
    # 创建模拟数据
    print("\n[1/4] 创建模拟蛋白质数据...")
    proteins_df = create_mock_protein_data(n_samples=5)
    print(f"  ✓ 创建了 {len(proteins_df)} 个样本的蛋白质数据")
    print(f"  ✓ 包含 {len(proteins_df.columns)-1} 种蛋白质")
    
    # 获取器官特异性蛋白质映射
    print("\n[2/4] 获取器官特异性蛋白质映射...")
    organ_proteins = get_organ_specific_proteins()
    print(f"  ✓ 定义了 {len(organ_proteins)} 个器官系统:")
    for organ_key, organ_info in organ_proteins.items():
        print(f"    - {organ_info['name']}: {len(organ_info['proteins'])} 种蛋白质")
    
    # 计算器官健康评分
    print("\n[3/4] 计算器官健康评分...")
    organ_scores = calculate_organ_health_scores(proteins_df)
    print(f"  ✓ 计算完成，生成 {len(organ_scores.columns)} 列数据")
    
    # 显示结果
    print("\n[4/4] 显示结果...")
    print("\n器官健康评分结果:")
    print("-" * 80)
    
    # 显示每个样本的综合评分
    for idx, row in organ_scores.iterrows():
        sample_id = row['sample_id']
        overall_score = row.get('overall_health_score', np.nan)
        overall_level = row.get('overall_health_level', '未知')
        
        print(f"\n样本: {sample_id}")
        print(f"  综合健康评分: {overall_score:.1f} ({overall_level})")
        
        # 显示各器官评分
        for organ_key in organ_proteins.keys():
            score_col = f'{organ_key}_score'
            level_col = f'{organ_key}_level'
            
            if score_col in organ_scores.columns:
                score = row[score_col]
                level = row[level_col]
                organ_name = organ_proteins[organ_key]['name']
                
                if not pd.isna(score):
                    print(f"    {organ_name}: {score:.1f} ({level})")
    
    # 保存结果
    print("\n" + "=" * 80)
    print("保存测试结果...")
    output_file = Path(__file__).parent / "test_organ_health_scores_output.csv"
    organ_scores.to_csv(output_file, index=False)
    print(f"  ✓ 结果已保存到: {output_file}")
    
    # 统计信息
    print("\n统计信息:")
    print("-" * 80)
    
    for organ_key, organ_info in organ_proteins.items():
        score_col = f'{organ_key}_score'
        if score_col in organ_scores.columns:
            avg_score = organ_scores[score_col].mean()
            organ_name = organ_info['name']
            print(f"  {organ_name}: 平均评分 = {avg_score:.1f}")
    
    overall_avg = organ_scores['overall_health_score'].mean()
    print(f"\n  综合健康: 平均评分 = {overall_avg:.1f}")
    
    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)
    
    return organ_scores


if __name__ == "__main__":
    try:
        results = test_organ_health_scores()
        print("\n✅ 所有测试通过！")
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

