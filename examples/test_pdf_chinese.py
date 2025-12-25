#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试PDF中文字体是否正常工作
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from pathlib import Path

def test_chinese_font():
    """测试中文字体在PDF中的显示"""
    
    print("="*60)
    print("测试PDF中文字体")
    print("="*60)
    
    # 1. 查找并注册中文字体
    font_paths = [
        '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        '/System/Library/Fonts/STHeiti Light.ttc',
        'C:\\Windows\\Fonts\\simhei.ttf',
    ]
    
    chinese_font = 'Helvetica'
    for font_path in font_paths:
        if Path(font_path).exists():
            try:
                pdfmetrics.registerFont(TTFont('ChineseFont', font_path))
                chinese_font = 'ChineseFont'
                print(f"✓ 成功注册中文字体: {font_path}")
                break
            except Exception as e:
                print(f"✗ 注册失败: {font_path} - {e}")
    
    if chinese_font == 'Helvetica':
        print("⚠ 未找到中文字体，将使用默认字体")
        return False
    
    # 2. 创建测试PDF
    pdf_path = "test_chinese_font.pdf"
    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    story = []
    styles = getSampleStyleSheet()
    
    # 创建使用中文字体的样式
    title_style = ParagraphStyle(
        'ChineseTitle',
        parent=styles['Heading1'],
        fontName=chinese_font,
        fontSize=24,
        alignment=1
    )
    
    body_style = ParagraphStyle(
        'ChineseBody',
        parent=styles['BodyText'],
        fontName=chinese_font,
        fontSize=12
    )
    
    # 添加标题
    story.append(Paragraph("中文字体测试 / Chinese Font Test", title_style))
    story.append(Spacer(1, 0.3*inch))
    
    # 添加正文
    story.append(Paragraph("这是一段中文测试文本。", body_style))
    story.append(Paragraph("包含数字：0123456789", body_style))
    story.append(Paragraph("包含英文：ABCDEFG abcdefg", body_style))
    story.append(Spacer(1, 0.2*inch))
    
    # 添加表格
    table_data = [
        ['项目 / Item', '值 / Value'],
        ['年龄 / Age', '45岁 / 45 years'],
        ['性别 / Gender', '男 / Male'],
        ['健康状态 / Health', '良好 / Good'],
    ]
    
    table = Table(table_data, colWidths=[3*inch, 3*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498DB')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), chinese_font),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(table)
    
    # 生成PDF
    try:
        doc.build(story)
        print(f"\n✓ 测试PDF已生成: {pdf_path}")
        print("\n请打开PDF文件检查：")
        print("  1. 标题中的中文是否正常显示")
        print("  2. 正文中的中文是否正常显示")
        print("  3. 表格中的中文是否正常显示")
        print("  4. 数字是否正常显示")
        print("\n如果看到黑色方块，说明字体配置有问题。")
        return True
    except Exception as e:
        print(f"\n✗ PDF生成失败: {e}")
        return False

if __name__ == "__main__":
    success = test_chinese_font()
    exit(0 if success else 1)

