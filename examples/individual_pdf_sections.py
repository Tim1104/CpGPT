# 这是个人PDF报告的各个章节代码片段
# 用于插入到 generate_individual_pdf_report 函数中

# ============================================================================
# 第2章：多组织器官年龄预测
# ============================================================================

story.append(PageBreak())
story.append(Paragraph("2. Multi-Tissue Organ Age Prediction / 多组织器官年龄预测", heading_style))

# 2.1 器官年龄预测结果表格
organ_age_data = [['Organ / 器官', 'Predicted Age / 预测年龄', 'Status / 状态']]

organ_columns = {
    'brain_age': 'Brain / 脑',
    'liver_age': 'Liver / 肝脏',
    'heart_age': 'Heart / 心脏',
    'lung_age': 'Lung / 肺',
    'kidney_age': 'Kidney / 肾脏',
    'muscle_age': 'Muscle / 肌肉',
    'adipose_age': 'Adipose / 脂肪',
    'blood_age': 'Blood / 血液',
    'immune_age': 'Immune / 免疫',
    'skin_age': 'Skin / 皮肤',
    'bone_age': 'Bone / 骨骼',
}

has_organ_data = False
for col, name in organ_columns.items():
    if col in sample_data.index and pd.notna(sample_data[col]):
        has_organ_data = True
        age_val = sample_data[col]
        # 判断状态
        if 'predicted_age' in sample_data.index and pd.notna(sample_data['predicted_age']):
            diff = age_val - sample_data['predicted_age']
            if diff > 5:
                status = "Accelerated / 加速"
            elif diff < -5:
                status = "Decelerated / 减缓"
            else:
                status = "Normal / 正常"
        else:
            status = "N/A"
        organ_age_data.append([name, f"{age_val:.1f} years", status])

if has_organ_data:
    organ_table = Table(organ_age_data, colWidths=[2.5*inch, 2*inch, 2*inch])
    organ_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#E74C3C')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(organ_table)
    story.append(Spacer(1, 0.2*inch))
    
    # 2.2 器官年龄雷达图
    try:
        organ_ages = []
        organ_labels = []
        for col, name in organ_columns.items():
            if col in sample_data.index and pd.notna(sample_data[col]):
                organ_ages.append(sample_data[col])
                organ_labels.append(name.split('/')[0].strip())
        
        if len(organ_ages) >= 3:
            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
            
            # 计算角度
            angles = np.linspace(0, 2 * np.pi, len(organ_ages), endpoint=False).tolist()
            organ_ages_plot = organ_ages + [organ_ages[0]]
            angles += angles[:1]
            
            # 绘制雷达图
            ax.plot(angles, organ_ages_plot, 'o-', linewidth=2, color='#E74C3C', label='Organ Age')
            ax.fill(angles, organ_ages_plot, alpha=0.25, color='#E74C3C')
            
            # 添加参考线（实际年龄）
            if 'predicted_age' in sample_data.index and pd.notna(sample_data['predicted_age']):
                ref_age = [sample_data['predicted_age']] * len(angles)
                ax.plot(angles, ref_age, '--', linewidth=2, color='blue', label='Predicted Age')
            
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(organ_labels, fontsize=10)
            ax.set_ylim(0, max(organ_ages) * 1.2)
            ax.set_title('Multi-Tissue Organ Age / 多组织器官年龄', fontsize=14, fontweight='bold', pad=20)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
            ax.grid(True)
            
            organ_radar_path = f"{output_dir}/organ_age_radar_{sample_id}.png"
            plt.tight_layout()
            plt.savefig(organ_radar_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            story.append(Image(organ_radar_path, width=5*inch, height=5*inch))
            story.append(Spacer(1, 0.2*inch))
    except Exception as e:
        print(f"    ⚠ 器官年龄雷达图生成失败: {e}")
        plt.close()
else:
    story.append(Paragraph("No organ age data available / 无器官年龄数据", styles['Normal']))
    story.append(Spacer(1, 0.2*inch))

# ============================================================================
# 第3章：癌症预测
# ============================================================================

story.append(PageBreak())
story.append(Paragraph("3. Cancer Prediction / 癌症预测", heading_style))

# 3.1 癌症预测结果
cancer_result_data = [['Metric / 指标', 'Value / 值']]

if 'cancer_prediction' in sample_data.index and pd.notna(sample_data['cancer_prediction']):
    cancer_status = "Positive / 阳性" if sample_data['cancer_prediction'] == 1 else "Negative / 阴性"
    cancer_result_data.append(['Prediction / 预测结果', cancer_status])

if 'cancer_probability' in sample_data.index and pd.notna(sample_data['cancer_probability']):
    cancer_result_data.append(['Probability / 概率', f"{sample_data['cancer_probability']:.2%}"])
    
    # 风险等级
    prob = sample_data['cancer_probability']
    if prob < 0.3:
        risk_level = "Low / 低风险"
    elif prob < 0.7:
        risk_level = "Medium / 中风险"
    else:
        risk_level = "High / 高风险"
    cancer_result_data.append(['Risk Level / 风险等级', risk_level])

cancer_table = Table(cancer_result_data, colWidths=[3*inch, 3*inch])
cancer_table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#9B59B6')),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, 0), 11),
    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
    ('BACKGROUND', (0, 1), (-1, -1), colors.lavender),
    ('GRID', (0, 0), (-1, -1), 1, colors.black)
]))
story.append(cancer_table)
story.append(Spacer(1, 0.3*inch))

# 3.2 癌症概率可视化
if 'cancer_probability' in sample_data.index and pd.notna(sample_data['cancer_probability']):
    try:
        fig, ax = plt.subplots(figsize=(8, 4))

        prob = sample_data['cancer_probability']
        categories = ['Negative\n阴性', 'Positive\n阳性']
        values = [1 - prob, prob]
        colors_bar = ['#2ECC71', '#E74C3C']

        bars = ax.barh(categories, values, color=colors_bar, edgecolor='black', linewidth=2)
        ax.set_xlim(0, 1)
        ax.set_xlabel('Probability / 概率', fontsize=12)
        ax.set_title('Cancer Prediction Probability / 癌症预测概率', fontsize=14, fontweight='bold')
        ax.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Threshold / 阈值')

        # 添加数值标签
        for bar, val in zip(bars, values):
            ax.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                   f'{val:.1%}', va='center', fontsize=12, fontweight='bold')

        ax.legend()
        ax.grid(True, alpha=0.3, axis='x')

        cancer_prob_path = f"{output_dir}/cancer_probability_{sample_id}.png"
        plt.tight_layout()
        plt.savefig(cancer_prob_path, dpi=150, bbox_inches='tight')
        plt.close()

        story.append(Image(cancer_prob_path, width=5*inch, height=2.5*inch))
        story.append(Spacer(1, 0.2*inch))
    except Exception as e:
        print(f"    ⚠ 癌症概率图生成失败: {e}")
        plt.close()

# ============================================================================
# 第4章：五种表观遗传时钟
# ============================================================================

story.append(PageBreak())
story.append(Paragraph("4. Epigenetic Clocks / 五种表观遗传时钟", heading_style))

# 4.1 时钟结果表格
clock_data = [['Clock / 时钟', 'Age / 年龄', 'Acceleration / 加速']]

clock_columns = {
    'horvath': 'Horvath Clock / Horvath时钟',
    'hannum': 'Hannum Clock / Hannum时钟',
    'phenoage': 'PhenoAge / 表型年龄',
    'grimage': 'GrimAge / Grim年龄',
    'grimage2': 'GrimAge2 / Grim年龄2',
}

has_clock_data = False
for col, name in clock_columns.items():
    if col in sample_data.index and pd.notna(sample_data[col]):
        has_clock_data = True
        clock_age = sample_data[col]

        # 计算加速
        if 'predicted_age' in sample_data.index and pd.notna(sample_data['predicted_age']):
            acceleration = clock_age - sample_data['predicted_age']
            accel_str = f"{acceleration:+.1f} years"
        else:
            accel_str = "N/A"

        clock_data.append([name, f"{clock_age:.1f} years", accel_str])

if has_clock_data:
    clock_table = Table(clock_data, colWidths=[3*inch, 1.5*inch, 2*inch])
    clock_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498DB')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(clock_table)
    story.append(Spacer(1, 0.2*inch))

    # 4.2 时钟对比图
    try:
        clock_ages = []
        clock_labels = []
        for col, name in clock_columns.items():
            if col in sample_data.index and pd.notna(sample_data[col]):
                clock_ages.append(sample_data[col])
                clock_labels.append(name.split('/')[0].strip())

        if len(clock_ages) > 0:
            fig, ax = plt.subplots(figsize=(8, 5))

            x_pos = np.arange(len(clock_labels))
            bars = ax.bar(x_pos, clock_ages, color='#3498DB', edgecolor='black', linewidth=1.5)

            # 添加参考线（实际年龄）
            if 'predicted_age' in sample_data.index and pd.notna(sample_data['predicted_age']):
                ax.axhline(y=sample_data['predicted_age'], color='red', linestyle='--',
                          linewidth=2, label=f"Predicted Age / 预测年龄: {sample_data['predicted_age']:.1f}")

            ax.set_xticks(x_pos)
            ax.set_xticklabels(clock_labels, rotation=45, ha='right')
            ax.set_ylabel('Age (years) / 年龄（岁）', fontsize=12)
            ax.set_title('Epigenetic Clocks Comparison / 表观遗传时钟对比', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')

            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}', ha='center', va='bottom', fontsize=10)

            clock_comp_path = f"{output_dir}/clock_comparison_{sample_id}.png"
            plt.tight_layout()
            plt.savefig(clock_comp_path, dpi=150, bbox_inches='tight')
            plt.close()

            story.append(Image(clock_comp_path, width=5.5*inch, height=3.5*inch))
            story.append(Spacer(1, 0.2*inch))
    except Exception as e:
        print(f"    ⚠ 时钟对比图生成失败: {e}")
        plt.close()
else:
    story.append(Paragraph("No epigenetic clock data available / 无表观遗传时钟数据", styles['Normal']))
    story.append(Spacer(1, 0.2*inch))

# ============================================================================
# 第5章：血浆蛋白质预测
# ============================================================================

story.append(PageBreak())
story.append(Paragraph("5. Plasma Protein Prediction / 血浆蛋白质预测", heading_style))

# 统计蛋白质数据
protein_columns = [col for col in sample_data.index if col.startswith('protein_')]
has_protein_data = len(protein_columns) > 0

if has_protein_data:
    # 5.1 蛋白质统计
    protein_count = len(protein_columns)
    story.append(Paragraph(f"Total Proteins Predicted / 预测蛋白质总数: {protein_count}", styles['Normal']))
    story.append(Spacer(1, 0.1*inch))

    # 5.2 Top 10 蛋白质表格
    protein_values = {}
    for col in protein_columns:
        if pd.notna(sample_data[col]):
            protein_name = col.replace('protein_', '')
            protein_values[protein_name] = sample_data[col]

    if len(protein_values) > 0:
        # 按值排序，取前10
        sorted_proteins = sorted(protein_values.items(), key=lambda x: abs(x[1]), reverse=True)[:10]

        protein_table_data = [['Protein / 蛋白质', 'Predicted Value / 预测值']]
        for protein, value in sorted_proteins:
            protein_table_data.append([protein, f"{value:.3f}"])

        protein_table = Table(protein_table_data, colWidths=[3*inch, 2*inch])
        protein_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#16A085')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightgreen),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(Paragraph("Top 10 Proteins by Absolute Value / 绝对值前10的蛋白质", styles['Heading3']))
        story.append(Spacer(1, 0.1*inch))
        story.append(protein_table)
        story.append(Spacer(1, 0.2*inch))

        # 5.3 蛋白质分布图
        try:
            fig, ax = plt.subplots(figsize=(8, 5))

            proteins = [p[0] for p in sorted_proteins]
            values = [p[1] for p in sorted_proteins]
            colors_bar = ['#E74C3C' if v < 0 else '#2ECC71' for v in values]

            y_pos = np.arange(len(proteins))
            bars = ax.barh(y_pos, values, color=colors_bar, edgecolor='black', linewidth=1)

            ax.set_yticks(y_pos)
            ax.set_yticklabels(proteins, fontsize=9)
            ax.set_xlabel('Predicted Value / 预测值', fontsize=12)
            ax.set_title('Top 10 Plasma Proteins / 前10血浆蛋白质', fontsize=14, fontweight='bold')
            ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
            ax.grid(True, alpha=0.3, axis='x')

            protein_dist_path = f"{output_dir}/protein_distribution_{sample_id}.png"
            plt.tight_layout()
            plt.savefig(protein_dist_path, dpi=150, bbox_inches='tight')
            plt.close()

            story.append(Image(protein_dist_path, width=5.5*inch, height=3.5*inch))
            story.append(Spacer(1, 0.2*inch))
        except Exception as e:
            print(f"    ⚠ 蛋白质分布图生成失败: {e}")
            plt.close()
else:
    story.append(Paragraph("No protein data available / 无蛋白质数据", styles['Normal']))
    story.append(Spacer(1, 0.2*inch))

# ============================================================================
# 第6章：器官健康评分
# ============================================================================

story.append(PageBreak())
story.append(Paragraph("6. Organ Health Scores / 器官健康评分", heading_style))

# 6.1 器官健康评分表格
organ_health_data = [['Organ System / 器官系统', 'Score / 评分', 'Level / 等级']]

organ_health_columns = {
    'heart_score': 'Heart / 心脏',
    'kidney_score': '肾脏 / Kidney',
    'liver_score': 'Liver / 肝脏',
    'immune_score': 'Immune System / 免疫系统',
    'metabolic_score': 'Metabolic System / 代谢系统',
    'vascular_score': 'Vascular System / 血管系统',
}

has_organ_health = False
for col, name in organ_health_columns.items():
    if col in sample_data.index and pd.notna(sample_data[col]):
        has_organ_health = True
        score = sample_data[col]
        level_col = col.replace('_score', '_level')
        level = sample_data[level_col] if level_col in sample_data.index else "N/A"
        organ_health_data.append([name, f"{score:.1f}", str(level)])

if has_organ_health:
    organ_health_table = Table(organ_health_data, colWidths=[3*inch, 1.5*inch, 2*inch])
    organ_health_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#E67E22')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.bisque),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(organ_health_table)
    story.append(Spacer(1, 0.2*inch))

    # 6.2 器官健康雷达图
    try:
        organ_scores = []
        organ_labels = []
        for col, name in organ_health_columns.items():
            if col in sample_data.index and pd.notna(sample_data[col]):
                organ_scores.append(sample_data[col])
                organ_labels.append(name.split('/')[0].strip())

        if len(organ_scores) >= 3:
            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))

            angles = np.linspace(0, 2 * np.pi, len(organ_scores), endpoint=False).tolist()
            organ_scores_plot = organ_scores + [organ_scores[0]]
            angles += angles[:1]

            ax.plot(angles, organ_scores_plot, 'o-', linewidth=2, color='#E67E22', label='Organ Health Score')
            ax.fill(angles, organ_scores_plot, alpha=0.25, color='#E67E22')

            # 添加参考线
            for ref_val, label, color in [(90, 'Excellent/优秀', 'green'),
                                          (75, 'Good/良好', 'blue'),
                                          (60, 'Fair/一般', 'orange'),
                                          (40, 'Poor/较差', 'red')]:
                ref_line = [ref_val] * len(angles)
                ax.plot(angles, ref_line, '--', linewidth=1, color=color, alpha=0.5, label=label)

            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(organ_labels, fontsize=10)
            ax.set_ylim(0, 100)
            ax.set_title('Organ Health Radar / 器官健康雷达图', fontsize=14, fontweight='bold', pad=20)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)
            ax.grid(True)

            organ_health_radar_path = f"{output_dir}/organ_health_radar_{sample_id}.png"
            plt.tight_layout()
            plt.savefig(organ_health_radar_path, dpi=150, bbox_inches='tight')
            plt.close()

            story.append(Image(organ_health_radar_path, width=5.5*inch, height=5.5*inch))
            story.append(Spacer(1, 0.2*inch))
    except Exception as e:
        print(f"    ⚠ 器官健康雷达图生成失败: {e}")
        plt.close()
else:
    story.append(Paragraph("No organ health data available / 无器官健康数据", styles['Normal']))
    story.append(Spacer(1, 0.2*inch))

