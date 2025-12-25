#!/bin/bash
# 快速测试PDF生成

echo "=========================================="
echo "快速测试PDF生成"
echo "=========================================="

cd /home/yc/CpGPT/examples

echo ""
echo "[1/3] 验证数据完整性..."
python3 verify_pdf_data.py

if [ $? -ne 0 ]; then
    echo "⚠️  数据验证失败，但继续生成PDF..."
fi

echo ""
echo "[2/3] 生成PDF报告..."
python3 935k_enhanced_prediction.py

echo ""
echo "[3/3] 检查生成的PDF文件..."
if [ -d "results/935k_enhanced_predictions" ]; then
    echo "✓ 结果目录存在"
    
    pdf_count=$(find results/935k_enhanced_predictions -name "*.pdf" | wc -l)
    echo "✓ 找到 $pdf_count 个PDF文件"
    
    if [ $pdf_count -gt 0 ]; then
        echo ""
        echo "生成的PDF文件："
        ls -lh results/935k_enhanced_predictions/*.pdf | head -5
        
        echo ""
        echo "=========================================="
        echo "✅ PDF生成成功！"
        echo "=========================================="
        echo ""
        echo "请检查以下内容："
        echo "1. 打开 results/935k_enhanced_predictions/report_000536.pdf"
        echo "2. 检查第2章是否显示6个器官健康评分"
        echo "3. 检查第4章是否显示5种时钟数据"
        echo "4. 检查中文是否正常显示（无黑色方块）"
        echo ""
    else
        echo "❌ 未找到PDF文件"
    fi
else
    echo "❌ 结果目录不存在"
fi

